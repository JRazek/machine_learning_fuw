use std::fs::read_to_string;

use std::fs::read_dir;

use clap::Parser;

use std::borrow::Cow;

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
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)]))").unwrap();

    let dataset = citations_reg.replace_all(&dataset, "");

    dataset
}

fn build_dictionary<'a>(datasets: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    let mut dictionary = datasets.collect::<Vec<&str>>();

    dictionary.sort();
    dictionary.dedup();

    dictionary
}

fn prepare_dictionary<'a>(
    dictionary_path: &str,
    datasets: impl Iterator<Item = &'a str>,
) -> Result<Vec<(usize, String)>, std::io::Error> {
    let dictionary: Vec<(usize, String)> = match read_to_string(dictionary_path) {
        Ok(dictionary) => {
            println!("cached dictionary exists.");

            dictionary
                .lines()
                .map(|x| x.to_string())
                .enumerate()
                .collect()
        }
        Err(_) => {
            println!("dictionary not found. Building dictionary..");

            let dictionary = build_dictionary(datasets);

            std::fs::write(dictionary_path, dictionary.join("\n"))?;

            dictionary
                .iter()
                .map(|x| x.to_string())
                .enumerate()
                .collect()
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

    Ok(())
}
