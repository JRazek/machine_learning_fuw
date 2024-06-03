use std::fs::read_to_string;

use std::fs::read_dir;

use clap::Parser;

use std::borrow::Cow;

use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,

    #[clap(short, long, default_value = "datasets/dictionary.txt")]
    dictionary_path: String,
}

fn filter_wksf_dataset<'a>(dataset: &'a str) -> Cow<'a, str> {
    let citations_reg =
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)])|[^\w\s])")
            .unwrap();

    let dataset = citations_reg.replace_all(&dataset, "");

    dataset
}

fn build_dictionary<'a>(datasets: impl Iterator<Item = &'a str>) -> HashMap<u32, String> {
    let dictionary = std::iter::once("<stop>".to_string())
        .chain(
            datasets
                .flat_map(|x| x.split_whitespace())
                .map(|x| x.to_lowercase()),
        )
        .enumerate()
        .map(|(i, x)| (i as u32, x))
        .collect();

    dictionary
}

fn prepare_dictionary<'a>(
    dictionary_path: &str,
    datasets: impl Iterator<Item = &'a str>,
) -> Result<HashMap<u32, String>, std::io::Error> {
    let dictionary: HashMap<u32, String> = match read_to_string(dictionary_path) {
        Ok(dictionary) => {
            println!("cached dictionary exists.");

            dictionary
                .lines()
                .map(|x| x.to_string())
                .enumerate()
                .map(|(i, x)| (i as u32, x))
                .collect()
        }
        Err(_) => {
            println!("dictionary not found. Building dictionary..");

            let dictionary = build_dictionary(datasets);

            std::fs::write(
                dictionary_path,
                dictionary
                    .values()
                    .map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )?;

            dictionary
        }
    };

    Ok(dictionary)
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
