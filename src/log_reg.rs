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

use dfdx::shapes::Const;
use dfdx::shapes::ConstShape;
use dfdx::shapes::HasShape;
use dfdx::shapes::Rank1;
use dfdx::shapes::Rank2;
use dfdx::tensor::AutoDevice;
use dfdx::tensor::Storage;
use dfdx::tensor::Tensor;
use dfdx::tensor::TensorFrom;
use dfdx::tensor_ops::Device;
use dfdx::tensor_ops::ReshapeTo;
use dfdx::tensor_ops::TryMatMul;

struct LogisticRegression<const N: usize, St: Storage<f32>> {
    theta: Tensor<(Const<N>,), f32, St>,
}

use std::fmt::Debug;

impl<const N: usize, St: Device<f32>> LogisticRegression<N, St>
where
    Tensor<Rank1<N>, f32, St>: TryMatMul<Tensor<Rank1<N>, f32, St>>,
    <Tensor<Rank1<N>, f32, St> as TryMatMul<Tensor<Rank1<N>, f32, St>>>::Output: Debug,
    <Tensor<(dfdx::prelude::Const<N>,), f32, St> as HasShape>::Shape: ConstShape,
{
    pub fn inference(self, mut x: Tensor<Rank1<N>, f32, St>) -> f32 {
        let dev = AutoDevice::default();

        let x: Tensor<Rank2<1, N>, f32, _> = x.reshape();

        let arg: f32 = -x.matmul(self.theta).as_vec()[0];

        let res = 1. / (1. + arg.exp());

        res
    }
}

pub fn log_reg() {
    let mut records: Vec<Record> = csv::Reader::from_path("data/reg_log_data.txt")
        .unwrap()
        .deserialize()
        .map(|res| res.unwrap())
        .collect();

    visualize(records.iter());

    let dev = AutoDevice::default();

    let log_reg = LogisticRegression {
        theta: dev.tensor([1., 2.]),
    };

    let x = dev.tensor([1., 0.5]);

    log_reg.inference(x);
}
