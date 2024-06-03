use std::fs::read_to_string;

use std::borrow::Cow;

use std::collections::HashMap;

pub fn filter_wksf_dataset<'a>(dataset: &'a str) -> Cow<'a, str> {
    let citations_reg =
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)])|[^\w\s])")
            .unwrap();

    let dataset = citations_reg.replace_all(&dataset, "");

    dataset
}

pub fn build_dictionary<'a>(datasets: impl Iterator<Item = &'a str>) -> HashMap<u32, String> {
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

pub fn prepare_dictionary<'a>(
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
