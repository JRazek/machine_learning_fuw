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
    embedding: LinearConstConfig<DICTIONARY_SIZE, EMBEDDING_SIZE>,
    sigmoid: Sigmoid,
    output_layer: LinearConstConfig<EMBEDDING_SIZE, DICTIONARY_SIZE>,
}

#[derive(Sequential, Default, Debug, Clone)]
struct EmbeddingNet<const VOCAB_SIZE: usize, const EMBEDDING_SIZE: usize> {
    embedding: LinearConstConfig<VOCAB_SIZE, EMBEDDING_SIZE>,
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
            Tensor<Rank2<{ 2 * RADIUS }, DICTIONARY_SIZE>, E, D, OwnedTape<E, D>>,
            Output = Tensor<Rank2<{ 2 * RADIUS }, DICTIONARY_SIZE>, E, D, OwnedTape<E, D>>,
        > + UpdateParams<E, D>,
{
    let stop_iter = std::iter::repeat_n(0, RADIUS); //stop words padding

    let dataset: Vec<u32> = stop_iter
        .clone()
        .chain(dataset.iter().cloned())
        .chain(stop_iter)
        .collect();

    let mut optimizer = Adam::new(&module, adam_config);

    println!("training..");

    for window in dataset.windows(2 * RADIUS + 1) {
        let window_words = window
            .iter()
            .map(|&x| inverse_dictionary[&x].clone())
            .collect::<Vec<_>>();
        println!("window: {:?}", window_words);

        let current_token = window[RADIUS] as usize;
        let neighbours: [usize; 2 * RADIUS] = window[0..RADIUS]
            .iter()
            .chain(window[RADIUS + 1..].iter())
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let current_word = inverse_dictionary[&(current_token as u32)].clone();
        let neighbours_words = neighbours
            .iter()
            .map(|&x| inverse_dictionary[&(x as u32)].clone())
            .collect::<Vec<_>>();

        println!("neighbours of: {:?}, {:?}", current_word, neighbours_words);

        let input: Tensor<Rank2<{ 2 * RADIUS }, DICTIONARY_SIZE>, E, D, OwnedTape<_, _>> = dev
            .one_hot_encode(Const::<DICTIONARY_SIZE>, [current_token; 2 * RADIUS])
            .retaped();

        let target: Tensor<Rank2<{ 2 * RADIUS }, DICTIONARY_SIZE>, E, D> =
            dev.one_hot_encode(Const::<DICTIONARY_SIZE>, neighbours);

        let output = module.forward(input);

        let loss = cross_entropy_with_logits_loss(output, target);

        println!("loss: {:?}", loss.as_vec()[0]);

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
    let dot_product = lhs.clone().try_mul(rhs.clone())?.sum::<Rank0, _>();

    let lhs_norm = lhs.square().sum::<Rank0, _>().sqrt();
    let rhs_norm = rhs.square().sum::<Rank0, _>().sqrt();

    let similarity = dot_product / (lhs_norm * rhs_norm);

    Ok(similarity)
}

fn find_cos_similarities_with_tokens<
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

    const DICTIONARY_SIZE: usize = 11457;
    assert_eq!(dictionary.len(), DICTIONARY_SIZE);

    println!("tokenizing dataset..");

    let inverse_dictionary = inverse_dictionary(&dictionary);

    let tokenized_datasets_flattened: Vec<u32> =
        tokenize_preprocessed_text(dataset.iter().map(|x| x.as_str()), &dictionary);

    println!("tokenized dataset successfully",);

    let dev = AutoDevice::default();

    let adam_config = AdamConfig {
        lr: 4e-3,
        ..Default::default()
    };

    let mut module = dev.build_module::<f32>(SkipGram::default());

    match module.load_safetensors(&embedding_model_path) {
        Ok(_) => println!("model loaded successfully"),
        Err(_) => println!("model not loaded, proceeding randomly initialized.."),
    }

    let mut iteration = 0;

    const EMEDDINGS_SIZE: usize = 256;
    const EPOCHS: usize = 10;

    let mut callback = move |module: &DeviceSkipGram<DICTIONARY_SIZE, EMEDDINGS_SIZE, _, _>| {
        iteration += 1;
        println!("iteration: {}", iteration);

        if iteration % 1000 == 0 {
            println!("saving model..");

            module.save_safetensors(&embedding_model_path).unwrap();
        }
    };

    if train {
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

    let get_similarities = |word| {
        let similarities =
            find_cos_similarities_with_tokens::<DICTIONARY_SIZE, EMEDDINGS_SIZE, f32, _, _>(
                dev.clone(),
                dictionary[word],
                &module.embedding,
            )?
            .to_vec()
            .into_iter()
            .take(5)
            .map(|(i, x)| (inverse_dictionary[&i].clone(), x))
            .collect::<Vec<_>>();

        Ok::<_, Box<dyn std::error::Error>>(similarities)
    };

    let similarities_mother = get_similarities("mother")?;
    println!("embedding similarities: {:?}", similarities_mother);
    let similarities_father = get_similarities("father")?;
    println!("embedding similarities: {:?}", similarities_father);
    let similarities_majesty = get_similarities("majesty")?;
    println!("embedding similarities: {:?}", similarities_majesty);
    let similarities_lady = get_similarities("lady")?;
    println!("embedding similarities: {:?}", similarities_lady);
    let similarities_man = get_similarities("man")?;
    println!("embedding similarities: {:?}", similarities_man);

    let similarities_daughter = get_similarities("master")?;
    println!("embedding similarities: {:?}", similarities_daughter);

    Ok(())
}
