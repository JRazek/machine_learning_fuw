use std::borrow::Cow;

use std::collections::HashMap;
use std::collections::HashSet;

pub fn filter_wksf_dataset<'a>(dataset: &'a str) -> Cow<'a, str> {
    let citations_reg =
        regex::Regex::new(r"((\d+~[^~]+.*)|(\[(\d+|\/|\+|\#|\?|\&|\~|\,\,|\=|\$)])|[^\w\s])")
            .unwrap();

    let dataset = citations_reg.replace_all(&dataset, " ");

    dataset
}

pub fn preprocess_raw_text(dataset: &str) -> impl Iterator<Item = String> + '_ {
    let stop_words = get_stop_words();

    let ds = dataset
        .split_whitespace()
        .map(|x| x.to_lowercase())
        .filter(move |x| !stop_words.contains(x));

    ds
}

pub fn build_dictionary<'a>(preprocessed_datasets: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut dictionary = preprocessed_datasets
        .map(|x| x.to_string())
        .collect::<Vec<String>>();

    dictionary.sort();
    dictionary.dedup();

    dictionary
}

pub fn tokenize_preprocessed_text<'a>(
    dataset: impl Iterator<Item = &'a str>,
    dictionary: &HashMap<String, u32>,
) -> Vec<u32> {
    dataset
        .filter_map(|x| dictionary.get(x).map(|&x| x))
        .collect()
}

pub fn detokenize_preprocessed_text<'a>(
    dataset: impl Iterator<Item = u32>,
    inverse_dictionary: &HashMap<u32, String>,
) -> Vec<String> {
    dataset
        .filter_map(|x| inverse_dictionary.get(&x).map(|x| x.clone()))
        .collect()
}

pub fn inverse_dictionary(dictionary: &HashMap<String, u32>) -> HashMap<u32, String> {
    dictionary.iter().map(|(k, v)| (*v, k.clone())).collect()
}

pub fn get_stop_words() -> HashSet<String> {
    const STOP_WORDS: &str = include_str!("stop_words.txt");

    STOP_WORDS
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}
