#![feature(iter_repeat_n)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

use clap::Parser;
use std::fs::read_dir;
use std::fs::read_to_string;

use std::collections::HashMap;

use uczenie_maszynowe_fuw::data_streams::*;

use dfdx::data::OneHotEncode;
use dfdx::prelude::*;

#[derive(Sequential, Default, Debug, Clone)]
struct Cbow<const DICTIONARY_SIZE: usize, const EMBEDDING_SIZE: usize> {
    embedding: EmbeddingNet<DICTIONARY_SIZE, EMBEDDING_SIZE>,
    sigmoid: Sigmoid,
    output_layer: LinearConstConfig<EMBEDDING_SIZE, DICTIONARY_SIZE>,
}

#[derive(Sequential, Default, Debug, Clone)]
pub struct EmbeddingNet<const DICTIONARY_SIZE: usize, const EMBEDDING_SIZE: usize> {
    pub linear1: LinearConstConfig<DICTIONARY_SIZE, EMBEDDING_SIZE>,
}

fn cbow_training<
    const DICTIONARY_SIZE: usize,
    const EMBEDDING_SIZE: usize,
    const RADIUS: usize,
    E,
    D,
    M,
>(
    dataset: &[u32],
    dev: D,
    mut module: M,
    inverse_dictionary: &HashMap<u32, String>,
    adam_config: AdamConfig,
    mut model_callback: impl FnMut(&M, E, &[&str]),
) -> Result<M, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
    [E; 2 * RADIUS]:,
    M: Module<
            Tensor<Rank1<DICTIONARY_SIZE>, E, D, OwnedTape<E, D>>,
            Output = Tensor<Rank1<DICTIONARY_SIZE>, E, D, OwnedTape<E, D>>,
        > + UpdateParams<E, D>,
{
    let mut optimizer = Adam::new(&module, adam_config);

    println!("training..");

    for window in dataset.windows(2 * RADIUS + 1) {
        let window_words = window
            .iter()
            .map(|&x| inverse_dictionary[&x].clone())
            .collect::<Vec<_>>();

        let current_token = window[RADIUS] as usize;
        let neighbours: Vec<_> = window[0..RADIUS]
            .iter()
            .chain(window[RADIUS + 1..].iter())
            .filter_map(|&x| match x {
                0 => None,
                _ => Some(x as usize),
            })
            .collect::<Vec<_>>();

        if neighbours.is_empty() {
            continue;
        }

        let input: Tensor<Rank1<DICTIONARY_SIZE>, E, D, OwnedTape<_, _>> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, neighbours)
            .mean::<_, Axis<0>>()
            .retaped();

        let target: Tensor<Rank1<DICTIONARY_SIZE>, E, D> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, [current_token])
            .reshape();

        let output = module.forward(input);

        let loss = cross_entropy_with_logits_loss(output, target);

        let loss_scalar = loss.as_vec()[0];
        let grads = loss.backward();

        optimizer.update(&mut module, &grads)?;

        model_callback(
            &module,
            loss_scalar,
            window_words
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    Ok(module)
}

fn cosine_similarity<const N: usize, E, D>(
    lhs: Tensor<Rank1<N>, E, D>,
    rhs: Tensor<Rank1<N>, E, D>,
) -> Result<Tensor<Rank0, E, D>, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
{
    let lhs_normalized = lhs.normalize(1e-5);
    let rhs_normalized = rhs.normalize(1e-5);

    let dot_product = lhs_normalized
        .clone()
        .try_mul(rhs_normalized.clone())?
        .sum::<Rank0, _>();

    let lhs_norm = lhs_normalized.square().sum::<Rank0, _>().sqrt();
    let rhs_norm = rhs_normalized.square().sum::<Rank0, _>().sqrt();

    let similarity = dot_product / (lhs_norm * rhs_norm);

    Ok(similarity)
}

fn one_hot_encode_token<const DICTIONARY_SIZE: usize, const EMBEDDING_SIZE: usize, E, D>(
    dev: D,
    token: u32,
) -> Result<Tensor<Rank1<DICTIONARY_SIZE>, E, D>, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
{
    let lhs_one_hot = dev
        .one_hot_encode(Const::<DICTIONARY_SIZE>, [token as usize])
        .reshape();

    Ok(lhs_one_hot)
}

fn find_cos_similarities_tokens_dataset<
    const DICTIONARY_SIZE: usize,
    const EMBEDDING_SIZE: usize,
    E,
    D,
    M,
>(
    dev: D,
    target: u32,
    dictionary_tokens: &[u32],
    embedding_net: &M,
) -> Result<Vec<(u32, E)>, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
    M: Module<Tensor<Rank1<DICTIONARY_SIZE>, E, D>, Output = Tensor<Rank1<EMBEDDING_SIZE>, E, D>>,
{
    let target_embedding = embedding_net.forward(
        dev.one_hot_encode(Const::<DICTIONARY_SIZE>, [target as usize])
            .reshape(),
    );

    let mut similarities = Vec::new();

    for &i in dictionary_tokens {
        let input: Tensor<Rank1<DICTIONARY_SIZE>, E, D> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, [i as usize])
            .reshape();

        let output = embedding_net.forward(input.clone());

        let similarity = cosine_similarity(target_embedding.clone(), output.clone())?.as_vec()[0];

        similarities.push((i, similarity));
    }

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Ok(similarities)
}

fn plot_pca<
    const DICTIONARY_SIZE: usize,
    const EMBEDDINGS_SIZE: usize,
    const SAMPLE_SIZE: usize,
    E,
    D,
    M,
>(
    dev: D,
    module: &M,
    dataset_labels: &[&str; SAMPLE_SIZE],
    dictionary: &HashMap<String, u32>,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: Dtype,
    E: num::traits::AsPrimitive<f64>,
    D: Device<E>,
    M: Module<
        Tensor<Rank2<SAMPLE_SIZE, DICTIONARY_SIZE>, E, D>,
        Output = Tensor<Rank2<SAMPLE_SIZE, EMBEDDINGS_SIZE>, E, D>,
    >,
{
    use linfa::dataset::DatasetBase;
    use linfa::traits::{Fit, Predict};
    use linfa_reduction::Pca;

    println!("dictionary: {:?}", dictionary);
    let mapped_dataset: [usize; SAMPLE_SIZE] = dataset_labels.map(|x| dictionary[x] as usize);

    let one_hot_encoded = dev.one_hot_encode(Const::<DICTIONARY_SIZE>, mapped_dataset);

    let embeddings = module.forward(one_hot_encoded);

    let embeddings_ndarray = ndarray::Array2::from_shape_vec(
        (SAMPLE_SIZE, EMBEDDINGS_SIZE),
        embeddings
            .as_vec()
            .into_iter()
            .map(|x| x.as_())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let dataset: DatasetBase<_, _> = embeddings_ndarray.into();

    let embedding_pca = Pca::params(2).fit(&dataset).unwrap();

    let reduced_embeddings: ndarray::Array2<_> = embedding_pca.predict(dataset).targets;

    use plotters::prelude::*;

    let plot =
        SVGBackend::new("plots/word2vec_pca_embeddings.svg", (1024, 1024)).into_drawing_area();

    plot.fill(&WHITE).unwrap();

    let reduced_embeddings = reduced_embeddings.map(|&x| x as f32);
    uczenie_maszynowe_fuw::plots::plot_cartesian2d_points(
        dataset_labels,
        reduced_embeddings,
        "PCA Embeddings",
        &plot,
    )?;

    plot.present()?;

    println!("pca embeddings plotted successfully");

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,

    #[clap(short, long, default_value = "models/embeddings_model.pth")]
    embedding_model_path: String,

    #[clap(short, long, default_value = "false")]
    train: bool,

    #[clap(short, long, default_value = "false")]
    pca: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        dataset_path,
        embedding_model_path,
        train,
        pca,
    }: Args = Args::parse();

    println!("reading files from {}..", dataset_path);

    let files = read_dir(dataset_path)?;

    let mut dataset: Vec<String> = Vec::new();

    for file in files {
        let content = read_to_string(file?.path())?;
        let mut filtered_dataset =
            preprocess_raw_text(&filter_wksf_dataset(&content)).collect::<Vec<String>>();

        dataset.append(&mut filtered_dataset);
    }

    let dictionary = build_dictionary(dataset.iter().map(|x| x.as_str()), 5);

    let dictionary = dictionary
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect::<HashMap<_, _>>();

    //    println!("dataset: {:?}", dataset);
    println!("dictionary length: {:?}", dictionary.len());

    const DICTIONARY_SIZE: usize = 2442;
    assert_eq!(dictionary.len(), DICTIONARY_SIZE);

    println!("tokenizing dataset..");

    let inverse_dictionary = inverse_dictionary(&dictionary);

    let tokenized_datasets_flattened: Vec<u32> =
        tokenize_preprocessed_text(dataset.iter().map(|x| x.as_str()), &dictionary);

    println!("tokenized dataset successfully",);

    let dev = AutoDevice::default();

    let mut adam_config = AdamConfig {
        lr: 1e-3,
        eps: 1e-5,
        ..Default::default()
    };

    let mut module = dev.build_module::<f32>(Cbow::default());

    match module.load_safetensors(&embedding_model_path) {
        Ok(_) => println!("model loaded successfully"),
        Err(_) => println!("model not loaded, proceeding randomly initialized.."),
    }

    let mut iteration = 0;

    const EMBEDDINGS_SIZE: usize = 128;
    const EPOCHS: usize = 1000;

    let mut callback = move |module: &DeviceCbow<DICTIONARY_SIZE, EMBEDDINGS_SIZE, _, _>,
                             loss: f32,
                             window: &[&str]| {
        if iteration % 1000 == 0 {
            println!("iteration: {}, loss: {}", iteration, loss);
            println!("window: {:?}", window);

            module.save_safetensors(&embedding_model_path).unwrap();
        }

        iteration += 1;
    };

    if pca {
        let dataset_sample: [&str; _] = [
            "man", "woman", "mother", "king", "saddle", "horse", "curse", "lady", "father", "dog",
            "kingdom", "crown", "marry", "god", "loving", "kill",
        ];

        let dataset_sample: [&str; _] = dataset_sample;

        plot_pca::<DICTIONARY_SIZE, EMBEDDINGS_SIZE, _, _, _, _>(
            dev.clone(),
            &module.embedding,
            &dataset_sample,
            &dictionary,
        )?;
    }

    match train {
        true => {
            for epoch in 0..EPOCHS {
                println!("epoch: {}", epoch);
                module = cbow_training::<DICTIONARY_SIZE, EMBEDDINGS_SIZE, 1, f32, _, _>(
                    &tokenized_datasets_flattened,
                    dev.clone(),
                    module,
                    &inverse_dictionary,
                    adam_config,
                    &mut callback,
                )?;

                adam_config.lr *= 0.99;
                println!("epoch: {}, lr: {}", epoch, adam_config.lr);
            }
        }
        false => {
            let get_similarities = |word: &str| {
                let similarities = find_cos_similarities_tokens_dataset::<
                    DICTIONARY_SIZE,
                    EMBEDDINGS_SIZE,
                    f32,
                    _,
                    _,
                >(
                    dev.clone(),
                    dictionary[word],
                    dictionary.values().cloned().collect::<Vec<_>>().as_slice(),
                    &module.embedding,
                )?
                .to_vec()
                .into_iter()
                .take(100)
                .map(|(i, x)| (inverse_dictionary[&i].clone(), x))
                .collect::<Vec<_>>();

                Ok::<_, Box<dyn std::error::Error>>(similarities)
            };

            let test_similarity_pair = |lhs: &str, rhs: &str| {
                let lhs = one_hot_encode_token::<DICTIONARY_SIZE, EMBEDDINGS_SIZE, f32, _>(
                    dev.clone(),
                    dictionary[lhs],
                )?;

                let rhs = one_hot_encode_token::<DICTIONARY_SIZE, EMBEDDINGS_SIZE, f32, _>(
                    dev.clone(),
                    dictionary[rhs],
                )?;

                let similarity = cosine_similarity(
                    module.embedding.forward(lhs),
                    module.embedding.forward(rhs),
                )?
                .as_vec()[0];

                Ok::<_, Box<dyn std::error::Error>>(similarity)
            };

            let similarities_king = get_similarities("king")?;

            println!("similarities to king: {:?}", similarities_king);

            let similarity_king_crown = test_similarity_pair("king", "dog")?;

            println!(
                "similarity between king and crown: {}",
                similarity_king_crown
            );
        }
    }

    Ok(())
}
