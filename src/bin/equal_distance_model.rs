use std::ops::{Add, Sub};

use dfdx::prelude::*;
use plotters::drawing::IntoDrawingArea;

use plotters::backend::SVGBackend;
use plotters::chart::ChartBuilder;
use plotters::element::Circle;
use plotters::series::PointSeries;

use plotters::style::*;

use rand_distr::Uniform;

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
    let dev = tensor.dev().clone();

    let f1 = dev.tensor(F1).broadcast();
    let f2 = dev.tensor(F2).broadcast();

    let tensor_cloned: Tensor<_, f32, D, OwnedTape<_, _>> = tensor.retaped();

    let shift1 = tensor
        .sub(f1)
        .square()
        .sum::<Rank1<N_POINTS>, _>()
        .sqrt()
        .sub(EPSILON * A)
        .square()
        .sum();

    let shift2 = tensor_cloned
        .sub(f2)
        .square()
        .sum::<Rank1<N_POINTS>, _>()
        .sqrt()
        .sub(EPSILON * A)
        .square()
        .sum();

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

    const ITERATIONS: usize = 10000;

    let mut sgd = Sgd::new(&tensor, SgdConfig::default());

    let svg_backend =
        SVGBackend::new("plots/path_descent_plot.svg", (800, 600)).into_drawing_area();

    let mut chart_builder = ChartBuilder::on(&svg_backend);

    let mut chart_context = chart_builder
        .caption("path plot", ("Arial", 20))
        .build_cartesian_2d(-12f32..12f32, -12f32..12f32)?;

    chart_context
        .configure_mesh()
        .x_labels(6)
        .y_labels(6)
        .label_style(("Arial", 15).into_font())
        .draw()?;

    chart_context.draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
        tensor.array().map(|x| (x[0], x[1])),
        2f32,
        BLUE.filled(),
    ))?;

    let mut reference_position1 = Vec::new();
    let mut lossed = Vec::new();

    for i in 0..ITERATIONS {
        let grads = Gradients::leaky();
        let loss = loss(tensor.trace(grads));

        if i % (ITERATIONS / 20) == 0 {
            println!("iteration: {}, loss: {}", i, loss.array());
        }

        reference_position1.push(tensor.clone().select(dev.tensor(30)).array());

        lossed.push(loss.array());

        let grads = loss.backward();

        sgd.update(&mut tensor, &grads)?;
    }

    chart_context.draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
        reference_position1.iter().map(|[x, y]| (*x, *y)),
        1f32,
        RED.filled(),
    ))?;

    chart_context.draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
        tensor.array().map(|x| (x[0], x[1])),
        2f32,
        GREEN.filled(),
    ))?;

    chart_context.draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
        [F1, F2].map(|x| (x[0], x[1])),
        4f32,
        BLACK.filled(),
    ))?;

    //obecnie nie mam pomysłu, jak mogę narysować Circle z promieniem zadanym przez coord system, a nie
    //liczbę pixeli, więc na "rozwiązanie analityczne" mogę tylko powiedzieć, że jest na przecięciu
    //dwóch okręgów o promieniu EPSILON * A i środkach w F1 i F2

    svg_backend.present()?;

    Ok(())
}
