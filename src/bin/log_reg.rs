#![feature(array_chunks)]
#![feature(iter_array_chunks)]

use serde::de::{self, Deserialize, Deserializer, Unexpected};

use plotters::prelude::*;

#[derive(Copy, Clone, Debug, serde::Deserialize)]
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

    if let [lower_area, center_area, upper_area, ..] = backend.split_evenly((3, 1)).as_slice() {
        let math_records = records.clone().map(|r| r.math);
        let biology_records = records.clone().map(|r| r.biology);
        let passed = records.clone().filter(|r| r.pass).count() as u32;

        show_by_records(
            math_records,
            biology_records,
            passed,
            records.clone().count() as u32 - passed,
            &lower_area,
        );

        let passed_records_filtered = records.clone().filter(|r| r.pass);
    } else {
        panic!("Expected 3 areas");
    }

    backend.present().unwrap();
}

use dfdx::nn::LinearConstConfig;
use dfdx::nn::Sigmoid;
use dfdx::tensor::AutoDevice;

use dfdx::prelude::*;

#[derive(Clone, Sequential, Default)]
struct LogisticRegression {
    linear: LinearConstConfig<2, 3>,
    activation: Sigmoid,
    linear2: LinearConstConfig<3, 1>,
    activation2: Sigmoid,
}

pub fn log_reg() {
    let records: Vec<Record> = csv::Reader::from_path("data/reg_log_data.txt")
        .unwrap()
        .deserialize()
        .map(|res| res.unwrap())
        .collect();

    visualize(records.iter());

    let dev = AutoDevice::default();

    let mut reg = dev.build_module::<f32>(LogisticRegression::default());

    let mut sgd = Sgd::new(
        &reg,
        SgdConfig {
            lr: 1e-1,
            momentum: None,
            weight_decay: None,
        },
    );

    let mut grads = reg.alloc_grads();

    //will skip last records.len() - (records.len() % 10) elements

    const BATCH_SIZE: usize = 1;

    for batch in records
        .repeat(1000)
        .into_iter()
        .array_chunks::<BATCH_SIZE>()
    {
        let batch_input: Vec<_> = batch
            .iter()
            .map(|&Record { math, biology, .. }| [0.01 * math, 0.01 * biology])
            .flatten()
            .collect();

        let batch_expected: [f32; BATCH_SIZE] = batch.map(|Record { pass, .. }| pass as u32 as f32);

        let input_tensor: Tensor<Rank2<BATCH_SIZE, 2>, f32, _, _> =
            dev.tensor_from_vec(batch_input, Rank2::<BATCH_SIZE, 2>::default());

        let prediction: Tensor<Rank1<BATCH_SIZE>, f32, Cpu, _> =
            reg.forward_mut(input_tensor.trace(grads)).reshape();

        let expected_tensor: Tensor<Rank1<BATCH_SIZE>, f32, _, _> = dev.tensor(batch_expected);

        println!(
            "prediction: {:?}, expected: {:?}",
            prediction.array(),
            expected_tensor.array()
        );

        let loss = mse_loss(prediction, expected_tensor);
        println!("current loss: {:?}", loss.array());

        grads = loss.backward();
        //        println!("grads: {:?}", grads);

        sgd.update(&mut reg, &grads).unwrap();

        reg.zero_grads(&mut grads);
    }
}

fn main() {
    log_reg()
}
