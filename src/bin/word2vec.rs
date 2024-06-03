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
struct EmbeddingNet<const VOCAB_SIZE: usize, const EMBEDDING_SIZE: usize> {
    embedding: LinearConstConfig<VOCAB_SIZE, EMBEDDING_SIZE>,
}

fn embeddings_training<
    const DICTIONARY_SIZE: usize,
    const EMBEDDING_SIZE: usize,
    const RADIUS: usize,
    E,
    D,
    M,
>(
    dataset: Vec<u32>,
    dev: D,
    embedding_net: M,
    adam_config: AdamConfig,
) -> Result<(), Box<dyn std::error::Error>>
where
    E: Dtype + rand::distributions::uniform::SampleUniform + num::Float + num::FromPrimitive,
    D: Device<E>,
    [E; 2 * RADIUS]:,
    M: Module<
            Tensor<Rank2<{ 2 * RADIUS }, DICTIONARY_SIZE>, E, D, OwnedTape<E, D>>,
            Output = Tensor<Rank2<{ 2 * RADIUS }, EMBEDDING_SIZE>, E, D, OwnedTape<E, D>>,
        > + UpdateParams<E, D>,
{
    let output_layer = LinearConstConfig::<EMBEDDING_SIZE, DICTIONARY_SIZE>::default();
    let output_layer_module = output_layer.build_on_device(&dev);

    let stop_iter = std::iter::repeat_n(0, RADIUS); //stop words padding

    let dataset: Vec<u32> = stop_iter
        .clone()
        .chain(dataset.into_iter())
        .chain(stop_iter)
        .collect();

    let mut module = (embedding_net, output_layer_module);

    let mut optimizer = Adam::new(&module, adam_config);

    println!("training..");

    for window in dataset.repeat(1).windows(2 * RADIUS + 1) {
        let current_token = window[RADIUS] as usize;
        let neighbours: [usize; 2 * RADIUS] = window[0..RADIUS]
            .iter()
            .chain(window[RADIUS + 1..].iter())
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        println!("neighbours: {:?}", neighbours);

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
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { dataset_path }: Args = Args::parse();

    println!("reading files from {}..", dataset_path);

    let files = read_dir(dataset_path)?;

    let mut dataset: Vec<String> = Vec::new();

    for file in files {
        let content = read_to_string(file?.path())?;
        let mut filtered_dataset =
            preprocess_raw_text(&filter_wksf_dataset(&content)).collect::<Vec<String>>();

        dataset.append(&mut filtered_dataset);
    }

    let dictionary = build_dictionary(dataset.iter().map(|x| x.as_str()));

    //    println!("dataset: {:?}", dataset);
    println!("dictionary length: {:?}", dictionary.len());

    const DICTIONARY_SIZE: usize = 85379;
    assert_eq!(dictionary.len(), DICTIONARY_SIZE);

    //    let inverse_dictionary = inverse_dictionary(&dictionary);

    println!("tokenizing dataset..");

    let tokenized_datasets_flattened: Vec<u32> =
        tokenize_preprocessed_text(dataset.iter().map(|x| x.as_str()), &dictionary);

    println!("tokenized dataset successfully",);

    let dev = AutoDevice::default();

    let adam_config = AdamConfig::default();

    let module = dev.build_module::<f32>(EmbeddingNet::default());
    embeddings_training::<DICTIONARY_SIZE, 1000, 2, f32, _, _>(
        tokenized_datasets_flattened,
        dev,
        module,
        adam_config,
    )?;

    //    let tokenizer = tokenizers::Tokenizer::new(dictionary);

    Ok(())
}
