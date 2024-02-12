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

use plotters::coord::Shift;

fn show_result<DB>(records: impl Iterator<Item = f32>, area: &DrawingArea<DB, Shift>, caption: &str)
where
    DB: DrawingBackend,
{
    let mut ctx = ChartBuilder::on(area)
        .caption(caption, &BLACK)
        .set_all_label_area_size(30)
        .build_cartesian_2d((20u32..110u32).step(10).use_floor(), (0u32..25u32).step(2))
        .unwrap();

    ctx.configure_mesh().x_label_offset(0).draw().unwrap();

    let histogram = Histogram::vertical(&ctx)
        .style(BLUE.stroke_width(1).filled())
        .data(records.map(|x| (x as u32, 1)));

    ctx.draw_series(histogram).unwrap();
}

fn show_passed<DB>(passed: u32, failed: u32, area: &DrawingArea<DB, Shift>, caption: &str)
where
    DB: DrawingBackend,
{
    let mut ctx = ChartBuilder::on(area)
        .caption(caption, &BLACK)
        .set_all_label_area_size(30)
        .build_cartesian_2d((0u32..1u32).into_segmented(), (0u32..80u32).step(2))
        .unwrap();

    ctx.configure_mesh().x_label_offset(0).draw().unwrap();

    let histogram = Histogram::vertical(&ctx)
        .style(BLUE.stroke_width(1).filled())
        .data(vec![(0, failed), (1, passed)]);

    ctx.draw_series(histogram).unwrap();
}

fn show_by_records<'a, DB, MI, BI>(
    math_records: MI,
    biology_records: BI,
    passed: u32,
    failed: u32,
    area: &DrawingArea<DB, Shift>,
) where
    DB: DrawingBackend,
    MI: Iterator<Item = f32> + Clone,
    BI: Iterator<Item = f32> + Clone,
{
    if let [math_area, biology_area, passed_area, ..] = area.split_evenly((1, 3)).as_slice() {
        show_result(math_records, math_area, "math");
        show_result(biology_records, biology_area, "biology");
        show_passed(passed, failed, passed_area, "passed");
    }
}

fn visualize<'a>(records: impl Iterator<Item = &'a Record> + Clone) {
    let backend = SVGBackend::new("plots/log_reg_data_plot.svg", (1200, 800)).into_drawing_area();

    let math_records = records.clone().map(|r| r.math);
    let biology_records = records.clone().map(|r| r.biology);
    let passed = records.clone().filter(|r| r.pass).count() as u32;

    let (lower_area, upper_area) = backend.split_vertically(400);

    show_by_records(
        math_records,
        biology_records,
        passed,
        records.count() as u32 - passed,
        &lower_area,
    );

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
