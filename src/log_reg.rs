use serde::de::{self, Deserialize, Deserializer, Unexpected};

use plotters::prelude::*;

#[derive(Debug, serde::Deserialize)]
struct Record {
    math: f32,
    biology: f32,

    #[serde(deserialize_with = "bool_from_int")]
    pass: bool,
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

fn visualize<'a>(records: impl Iterator<Item = &'a Record>) {
    let backend = BitMapBackend::new("plots/log_reg_data_plot.bmp", (800, 600)).into_drawing_area();

    let math_records = records.map(|r| r.math);

    if let [math_area, biology_area, pass_area] = backend.split_evenly((0, 2)).as_slice() {
        let mut math_builder = ChartBuilder::on(&math_area)
            .caption("math results", &BLACK)
            .build_cartesian_2d(0u32..10u32, 0f32..100f32)
            .unwrap();

        let histogram = Histogram::vertical(&math_builder)
            .style(BLUE.filled())
            .margin(0)
            .data(math_records.map(|x| (1u32, 32.0)));

        math_builder.draw_series(histogram).unwrap();
    }

    backend.present().unwrap();
}

pub fn log_reg() {
    let mut records: Vec<Record> = csv::Reader::from_path("data/reg_log_data.txt")
        .unwrap()
        .deserialize()
        .map(|res| res.unwrap())
        .collect();

    visualize(records.iter());
}
