#![feature(iter_repeat_n)]
#![feature(generic_const_exprs)]

use clap::Parser;
use std::fs::read_dir;
use std::fs::read_to_string;

use std::collections::HashMap;

use uczenie_maszynowe_fuw::data_streams::*;

use dfdx::data::OneHotEncode;
use dfdx::prelude::*;

#[derive(Sequential, Default, Debug, Clone)]
struct SkipGram<const DICTIONARY_SIZE: usize, const EMBEDDING_SIZE: usize> {
    embedding: EmbeddingNet<DICTIONARY_SIZE, EMBEDDING_SIZE>,
    output_layer: LinearConstConfig<EMBEDDING_SIZE, DICTIONARY_SIZE>,
}

#[derive(Sequential, Default, Debug, Clone)]
pub struct EmbeddingNet<const DICTIONARY_SIZE: usize, const EMBEDDING_SIZE: usize> {
    pub linear1: LinearConstConfig<DICTIONARY_SIZE, EMBEDDING_SIZE>,
}

fn skipgram_training<
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
    mut model_callback: impl FnMut(&M),
) -> Result<M, Box<dyn std::error::Error>>
where
    E: Dtype + rand::distributions::uniform::SampleUniform + num::Float + num::FromPrimitive,
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
        let neighbours: [usize; 2 * RADIUS] = window[0..RADIUS]
            .iter()
            .chain(window[RADIUS + 1..].iter())
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let input: Tensor<Rank1<DICTIONARY_SIZE>, E, D, OwnedTape<_, _>> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, neighbours)
            .sum::<_, Axis<0>>()
            .retaped();

        let target: Tensor<Rank1<DICTIONARY_SIZE>, E, D> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, [current_token])
            .reshape();

        let output = module.forward(input);

        let loss = cross_entropy_with_logits_loss(output, target);

        let grads = loss.backward();

        optimizer.update(&mut module, &grads)?;

        model_callback(&module);
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

    for i in 0..DICTIONARY_SIZE as u32 {
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,

    #[clap(short, long, default_value = "models/embeddings_model.pth")]
    embedding_model_path: String,

    #[clap(short, long, default_value = "false")]
    train: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        dataset_path,
        embedding_model_path,
        train,
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

    let dictionary = build_dictionary(dataset.iter().map(|x| x.as_str()))
        .into_iter()
        .collect::<HashMap<_, _>>();

    //    println!("dataset: {:?}", dataset);
    println!("dictionary length: {:?}", dictionary.len());

    const DICTIONARY_SIZE: usize = 10394;
    assert_eq!(dictionary.len(), DICTIONARY_SIZE);

    println!("tokenizing dataset..");

    let inverse_dictionary = inverse_dictionary(&dictionary);

    let tokenized_datasets_flattened: Vec<u32> =
        tokenize_preprocessed_text(dataset.iter().map(|x| x.as_str()), &dictionary);

    println!("tokenized dataset successfully",);

    let dev = AutoDevice::default();

    let adam_config = AdamConfig {
        lr: 1e-3,
        eps: 1e-5,
        ..Default::default()
    };

    let mut module = dev.build_module::<f32>(SkipGram::default());

    match module.load_safetensors(&embedding_model_path) {
        Ok(_) => println!("model loaded successfully"),
        Err(_) => println!("model not loaded, proceeding randomly initialized.."),
    }

    let mut iteration = 0;

    const EMEDDINGS_SIZE: usize = 128;
    const EPOCHS: usize = 1000;

    let mut callback = move |module: &DeviceSkipGram<DICTIONARY_SIZE, EMEDDINGS_SIZE, _, _>| {
        if iteration % 1000 == 0 {
            println!("iteration: {}", iteration);
            println!("saving model..");

            module.save_safetensors(&embedding_model_path).unwrap();
        }

        iteration += 1;
    };

    match train {
        true => {
            for epoch in 0..EPOCHS {
                println!("epoch: {}", epoch);
                module = skipgram_training::<DICTIONARY_SIZE, EMEDDINGS_SIZE, 3, f32, _, _>(
                    &tokenized_datasets_flattened,
                    dev.clone(),
                    module,
                    &inverse_dictionary,
                    adam_config,
                    &mut callback,
                )?;
            }
        }
        false => {
            let get_similarities = |word| {
                let similarities =
                    find_cos_similarities_tokens_dataset::<
                        DICTIONARY_SIZE,
                        EMEDDINGS_SIZE,
                        f32,
                        _,
                        _,
                    >(dev.clone(), dictionary[word], &module.embedding)?
                    .to_vec()
                    .into_iter()
                    .take(10)
                    .map(|(i, x)| (inverse_dictionary[&i].clone(), x))
                    .collect::<Vec<_>>();

                Ok::<_, Box<dyn std::error::Error>>(similarities)
            };

            let test_similarity_pair = |lhs: &str, rhs: &str| {
                let lhs = one_hot_encode_token::<DICTIONARY_SIZE, EMEDDINGS_SIZE, f32, _>(
                    dev.clone(),
                    dictionary[lhs],
                )?;

                let rhs = one_hot_encode_token::<DICTIONARY_SIZE, EMEDDINGS_SIZE, f32, _>(
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

            let similarities_man = get_similarities("man")?;
            println!("embedding similarities: {:?}", similarities_man);

            println!(
                "similarity mother, father: {:?}",
                test_similarity_pair("mother", "father")?
            );

            println!(
                "similarity father, son: {:?}",
                test_similarity_pair("father", "son")?
            );

            println!(
                "similarity lady, sword: {:?}",
                test_similarity_pair("lady", "sword")?
            );
        }
    }

    Ok(())
}
