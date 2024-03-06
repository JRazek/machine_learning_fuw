use dfdx::prelude::*;
use plotters::drawing::IntoDrawingArea;
use plotters::series::Histogram;

use plotters::backend::SVGBackend;

use std::time::Instant;

use std::{
    fs,
    ops::{Add, Mul},
    str::FromStr,
};

const COUNTERS_BIKES_PATH: &str = "data/counters_bikes01.csv";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let counters_01: Vec<String> = fs::read_to_string(COUNTERS_BIKES_PATH)?
        .lines()
        .map(|s| String::from_str(s).expect("infaillable type failed??"))
        .collect();

    println!("line 0: {}", counters_01[0]);

    println!("Current device: {:?}", dev);

    let m1: Tensor<Rank2<3, 3>, f32, _, _> = dev.tensor([[1., 2., 1.], [5., 5., 4.], [0., 1., 2.]]);
    let m2: Tensor<Rank2<3, 3>, f32, _, _> = dev.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]);

    let added = m1.clone().add(m2.clone());
    println!("added: {:?}", added.array());

    let dot = m1.clone().mul(added.clone());
    println!("dot: {:?}", dot.array());

    let matmul = m1.clone().matmul(m2.clone());
    println!("matmul: {:?}", matmul.array());

    let start_variance = Instant::now();

    let m: Tensor<Rank2<100, 100>, f32, _> = dev.sample_uniform();
    let m_sum = m.clone().sum::<Rank0, _>();
    let m_sum_sq = m.clone().square().sum::<Rank0, _>();

    let variance = m_sum_sq / m.len() as f32 - (m_sum / m.len() as f32).square();
    println!("variance: {:?}", (variance * 12.).array());

    let bench_variance = start_variance.elapsed();

    println!("elapsed: {:?}", bench_variance);

    let normal_dist: Tensor<Rank1<100000>, f32, _> = dev.sample_normal();

    let normal_binned = normal_dist
        .array()
        .into_iter()
        .fold([0; 100], |mut acc, x| {
            let idx = (x * 10. + 50.) as usize;
            acc[idx] += 1;
            acc
        });

    let drawing_area =
        SVGBackend::new("./plots/plotting_histogram.svg", (800, 600)).into_drawing_area();

    let (left, right) = drawing_area.split_horizontally(400);

    use plotters::prelude::*;

    let mut histogram_context = ChartBuilder::on(&left);

    let mut histogram = histogram_context
        .x_label_area_size(20)
        .y_label_area_size(20)
        .caption("Histogram Test", ("sans-serif", 50.0))
        .build_cartesian_2d((0u32..100u32).into_segmented(), 0f32..1f32)?;

    histogram
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    println!("normal_binned: {:?}", normal_binned);

    histogram.draw_series(
        Histogram::vertical(&histogram)
            .style(RED.filled())
            .data(
                normal_binned
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| (i as u32, y as f32 / normal_dist.len() as f32))
                    .inspect(|(i, y)| {
                        println!("i: {}, y: {}", i, y);
                    }),
            )
            .margin(0),
    )?;

    drawing_area.present()?;

    Ok(())
}
