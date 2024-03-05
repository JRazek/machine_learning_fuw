use dfdx::prelude::*;

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

    //calculate gaussian with CLT
    let uniform: Tensor<Rank1<100>, f32, _> = dev.sample_uniform();

    Ok(())
}
