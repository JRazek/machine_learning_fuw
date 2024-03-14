use std::ops::{Add, Sub};

use dfdx::prelude::*;
use nalgebra::ComplexField;
use plotters::{drawing::IntoDrawingArea, series::LineSeries};

use plotters::backend::SVGBackend;
use plotters::chart::ChartBuilder;
use plotters::element::Circle;
use plotters::series::PointSeries;

use plotters::style::*;

use rand_distr::{num_traits::Pow, Distribution, Uniform};

const N_POINTS: usize = 100;

const EPSILON: f32 = 0.5;
const A: f32 = 10.;
const F1: [f32; 2] = [-5., 0.];
const F2: [f32; 2] = [5., 0.];

fn loss<D>(
    tensor: Tensor<Rank2<N_POINTS, 2>, f32, D, OwnedTape<f32, D>>,
) -> Tensor<Rank0, f32, D, OwnedTape<f32, D>>
where
    D: Device<f32>,
{
    let dev = tensor.dev();

    let f1 = dev.tensor(F1).broadcast();
    let f2 = dev.tensor(F2).broadcast();

    let tensor_cloned: Tensor<_, f32, D, OwnedTape<_, _>> = tensor.retaped();

    let shift1 = tensor.sub(f1).square().sum();
    let shift2 = tensor_cloned.sub(f2).square().sum();

    let sum = shift1.add(shift2);

    sum
}

fn generate_set<D>(dev: &mut D) -> Tensor<Rank2<N_POINTS, 2>, f32, D, NoneTape>
where
    D: Device<f32>,
{
    dev.sample(Uniform::new(-12f32, 12f32))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = AutoDevice::default();

    let mut tensor = generate_set(&mut dev);

    const ITERATIONS: usize = 200;

    let mut sgd = Sgd::new(&tensor, SgdConfig::default());

    for _ in 0..ITERATIONS {
        let grads = Gradients::leaky();
        let loss = loss(tensor.trace(grads));

        println!("positions: {:?}", tensor.array());
        println!("loss: {}", loss.array());

        let grads = loss.backward();

        sgd.update(&mut tensor, &grads)?;
    }

    Ok(())
}
