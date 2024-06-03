use std::fs::read_to_string;

use std::borrow::Cow;

use std::collections::HashMap;

pub fn filter_wksf_dataset<'a>(dataset: &'a str) -> Cow<'a, str> {
    let citations_reg =
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)])|[^\w\s])")
            .unwrap();

    let dataset = citations_reg.replace_all(&dataset, " ");

    dataset
}

pub fn preprocess_raw_text(dataset: &str) -> impl Iterator<Item = String> + '_ {
    let ds = dataset.split_whitespace().map(|x| x.to_lowercase());

    ds
}

pub fn build_dictionary<'a>(
    preprocessed_datasets: impl Iterator<Item = &'a str>,
) -> HashMap<String, u32> {
    let mut dictionary = preprocessed_datasets
        .map(|x| x.to_string())
        .collect::<Vec<String>>();

    dictionary.sort();
    dictionary.dedup();

    let dictionary = std::iter::once("<stop>".to_string())
        .chain(dictionary)
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    dictionary
}

pub fn tokenize_preprocessed_text<'a>(
    dataset: impl Iterator<Item = &'a str>,
    dictionary: &HashMap<String, u32>,
) -> Vec<u32> {
    dataset
        .map(|x| dictionary.get(x).unwrap_or(&0).clone())
        .collect()
}

//pub fn inverse_dictionary(dictionary: &HashMap<u32, String>) -> HashMap<String, u32> {
//    dictionary.iter().map(|(x, y)| (y.clone(), *x)).collect()
//}
