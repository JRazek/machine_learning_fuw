use dfdx::prelude::*;
use plotters::drawing::IntoDrawingArea;

use plotters::backend::SVGBackend;
use plotters::chart::ChartBuilder;
use plotters::element::Circle;
use plotters::series::PointSeries;

use plotters::style::*;

use rand_distr::Uniform;

fn quadratic_eq_gen<const N: usize, D>(dev: &mut D) -> Tensor<Rank2<N, 3>, f32, D>
where
    D: Device<f32>,
{
    let tensor: Tensor<Rank2<N, 3>, f32, D> = dev.sample(Uniform::new(-1.0, 1.0));

    tensor
}

fn quadratic_eq_solve<D, const N: usize>(
    tensor: Tensor<Rank2<N, 3>, f32, D>,
) -> Tensor<Rank2<N, 2>, f32, D>
where
    D: Device<f32>,
{
    let a: Tensor<Rank1<N>, f32, D> = tensor
        .clone()
        .slice((..N, 0..1))
        .realize::<Rank2<N, 1>>()
        .reshape();
    let b: Tensor<Rank1<N>, f32, D> = tensor
        .clone()
        .slice((..N, 1..2))
        .realize::<Rank2<N, 1>>()
        .reshape();
    let c: Tensor<Rank1<N>, f32, D> = tensor
        .clone()
        .slice((..N, 2..3))
        .realize::<Rank2<N, 1>>()
        .reshape();

    let discriminant_sqrt = (b.clone().powi(2) - a.clone() * c.clone() * 4.)
        .sqrt()
        .nans_to(0.);

    let x1 = (-b.clone() + discriminant_sqrt.clone()) / (a.clone() * 2.);
    let x2 = (-b - discriminant_sqrt) / (a * 2.);

    [x1, x2].stack().permute()
}

fn train<M, D, const N: usize>(
    mut model: M,
    dataset: Tensor<Rank2<N, 3>, f32, D>,
    targets: Tensor<Rank2<N, 2>, f32, D>,
    n_iter: usize,
) -> Result<M, Box<dyn std::error::Error>>
where
    D: Device<f32>,
    M: Module<
            Tensor<(usize, Const<3>), f32, D, OwnedTape<f32, D>>,
            Output = Tensor<(usize, Const<2>), f32, D, OwnedTape<f32, D>>,
        > + UpdateParams<f32, D>,
{
    let mut sgd = Sgd::<_, f32, D>::new(
        &model,
        SgdConfig {
            lr: 1e-3,
            momentum: None,
            weight_decay: Some(WeightDecay::L2(1e-9)),
        },
    );

    const BATCH_SIZE: usize = 16;

    let batches = (0..N / BATCH_SIZE).map(|i| {
        let start = i * BATCH_SIZE;
        let end = (i + 1) * BATCH_SIZE;

        let inputs = dataset
            .clone()
            .slice((start..end, ..))
            .retaped::<OwnedTape<f32, D>>();
        let targets = targets.clone().slice((start..end, ..));

        (inputs, targets)
    });

    for i in 0..n_iter {
        for (inputs, targets) in batches.clone() {
            let outputs = model.forward(inputs);
            let loss = mse_loss(outputs, targets.clone()).mean();
            let loss_num = loss.as_vec();

            let grads = loss.backward();

            sgd.update(&mut model, &grads)?;

            if i % 100 == 0 {
                println!("Iter: {} Loss: {:?}", i, loss_num);
            }
        }
    }

    Ok(model)
}

#[derive(Clone, Sequential, Default)]
struct Model {
    linear1: LinearConstConfig<3, 20>,
    activation1: Softmax,
    linear2: LinearConstConfig<20, 20>,
    activation2: LeakyReLU,
    linear3: LinearConstConfig<20, 2>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = AutoDevice::default();

    let tensor: Tensor<Rank2<1000, 3>, _, _> = quadratic_eq_gen(&mut dev);

    let model = dev.build_module::<f32>(Model::default());

    let model = train(model, tensor.clone(), quadratic_eq_solve(tensor), 100_00)?;

    model.save_safetensors("models/quadratic_eq_model")?;

    let svg_backend =
        SVGBackend::new("plots/quadratic_eq_solutions.svg", (1024, 768)).into_drawing_area();

    let mut chart_builder = ChartBuilder::on(&svg_backend);

    let mut chart_context = chart_builder
        .caption("Quadratic equation solutions", ("Arial", 20))
        .build_cartesian_3d(-1.0f32..1.0, -1.0f32..1.0, -1.0f32..1.0)?;

    chart_context
        .configure_axes()
        .x_labels(6)
        .y_labels(6)
        .z_labels(6)
        .draw()?;

    //    let gen = |cord: [f32; 3], size, style| {
    //        let style = if let Some(_) = quadratic_eq_solve(&cord) {
    //            YELLOW.filled()
    //        } else {
    //            style
    //        };
    //
    //        let cord = (cord[0], cord[1], cord[2]);
    //
    //        Circle::new(cord, size, style)
    //    };
    //
    //    let point_series =
    //        PointSeries::of_element(tensor.array().into_iter(), 4., BLACK.filled(), &gen);
    //
    //    chart_context.draw_series(point_series)?;
    //
    //    svg_backend.present()?;

    Ok(())
}
