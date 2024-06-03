use clap::Parser;
use std::fs::read_dir;
use std::fs::read_to_string;

use uczenie_maszynowe_fuw::data_streams::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,

    #[clap(short, long, default_value = "datasets/dictionary.txt")]
    dictionary_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        dictionary_path,
        dataset_path,
    }: Args = Args::parse();

    println!("reading files from {}..", dataset_path);

    let files = read_dir(dataset_path)?;

    let mut datasets = Vec::new();

    for file in files {
        let content = read_to_string(file?.path())?;
        let filtered_dataset = filter_wksf_dataset(&content);
        datasets.push(filtered_dataset.into_owned());
    }

    let dictionary = prepare_dictionary(&dictionary_path, datasets.iter().map(|x| x.as_str()))?;

    println!("dictionary prepared, size: {}", dictionary.len());
    println!("{:?}", dictionary);

    //    let tokenizer = tokenizers::Tokenizer::new(dictionary);

    Ok(())
}
