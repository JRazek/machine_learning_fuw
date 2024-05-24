use std::fs::read_to_string;

use std::fs::read_dir;

use clap::Parser;

use std::borrow::Cow;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "datasets/korpus/")]
    dataset_path: String,
}

fn filter_wksf_dataset<'a>(dataset: &'a str) -> Cow<'a, str> {
    let citations_reg =
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)]))").unwrap();

    let dataset = citations_reg.replace_all(&dataset, "");

    dataset
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { dataset_path }: Args = Args::parse();

    println!("reading files from {}..", dataset_path);

    let files = read_dir(dataset_path)?;

    for file in files {
        let content = read_to_string(file?.path())?;
        let filtered_dataset = filter_wksf_dataset(&content);
        println!("{}", filtered_dataset);
    }

    Ok(())
}
