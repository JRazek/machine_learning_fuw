use serde::de::{self, Deserialize, Deserializer, Unexpected};

#[derive(Debug, serde::Deserialize)]
struct Record {
    math: f32,
    biology: f32,

    #[serde(deserialize_with = "bool_from_int")]
    result: bool,
}

fn bool_from_int<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    match u8::deserialize(deserializer)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(de::Error::invalid_value(
            Unexpected::Unsigned(other as u64),
            &"zero or one",
        )),
    }
}

pub fn log_reg() {
    let mut csv = csv::Reader::from_path("data/reg_log_data.txt").unwrap();

    for res in csv.deserialize() {
        let record: Record = res.unwrap();

        println!("{:?}", record);
    }
}
