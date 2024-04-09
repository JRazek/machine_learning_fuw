use dfdx::prelude::*;
use rand::prelude::*;
use rand_distr::Uniform;

fn poly(x: f64) -> f64 {
    2. * x.powi(2) * 3. * x - 2.
}

fn generate_dataset<const N: usize, D>(dev: D) -> Tensor<Rank2<N, 2>, f64, D, NoneTape>
where
    D: Device<f64>,
{
    let rng = StdRng::seed_from_u64(14);
    let uniform = Uniform::new(-1., 1.);

    let dataset = uniform
        .sample_iter(rng)
        .take(N)
        .map(|x| {
            let y = poly(x);
            [x, y]
        })
        .flatten()
        .collect::<Vec<_>>();

    let tensor: Tensor<Rank2<N, 2>, _, _, _> =
        dev.tensor_from_vec(dataset, (Const::<N>, Const::<2>));

    tensor
}

use plotters::prelude::*;

fn draw_dataset<DB>(
    dataset: impl Iterator<Item = (f64, f64, f64)> + Clone,
    mut chart_builder: ChartBuilder<DB>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend + 'static,
{
    let mut chart_context = chart_builder
        .caption("dataset", ("Arial", 20))
        .build_cartesian_3d(-10f64..10f64, -10f64..10f64, -10f64..10f64)?;

    chart_context
        .configure_axes()
        .x_labels(6)
        .y_labels(6)
        .z_labels(6)
        .draw()?;

    chart_context.draw_series(dataset.map(|(a, b, c)| Circle::new((a, b, c), 1, RED.filled())))?;

    Ok(())
}

fn draw_loss_evolution<DB>(
    loss: impl Iterator<Item = (u32, f64)> + ExactSizeIterator + Clone,
    mut left: ChartBuilder<DB>,
    mut right: ChartBuilder<DB>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend + 'static,
{
    let max: f64 = loss
        .clone()
        .map(|(_, l)| l)
        .filter(|&l| l < 0.1)
        .fold(0. / 0., f64::max);

    let mut linear_scale = left
        .caption("loss evolution", ("Arial", 20))
        .set_all_label_area_size(50)
        .build_cartesian_2d(0u32..loss.len() as u32, 0f64..max)?;

    linear_scale
        .configure_mesh()
        .x_labels(10)
        .x_desc("iteration")
        .y_labels(10)
        .y_desc("loss")
        .draw()?;

    linear_scale.draw_series(LineSeries::new(loss.clone(), &RED))?;

    let mut log_scale = right
        .caption("loss evolution", ("Arial", 20))
        .set_all_label_area_size(50)
        .build_cartesian_2d(0u32..loss.len() as u32, (0f64..max).log_scale())?;

    log_scale
        .configure_mesh()
        .x_labels(10)
        .x_desc("iteration")
        .y_labels(10)
        .y_desc("loss")
        .y_label_formatter(&|x| format!("{:.1e}", x))
        .draw()?;

    log_scale.draw_series(LineSeries::new(loss, &BLUE))?;

    Ok(())
}

fn draw_model_results<DB>(
    data: impl Iterator<Item = (f64, (f64, f64))> + Clone,
    mut right: ChartBuilder<DB>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend + 'static,
{
    let mut chart_context = right
        .caption("error = abs(prediction - target)", ("Arial", 20))
        .set_all_label_area_size(50)
        .build_cartesian_2d(-10i32..10i32, 0f64..1.2e-5f64)?;

    chart_context
        .configure_mesh()
        .x_labels(20)
        .x_desc("x")
        .x_label_formatter(&|&x| format!("{:.2}", x as f64 / 10.))
        .y_labels(10)
        .y_desc("error")
        .y_label_formatter(&|&x| format!("{:.1e}", x))
        .draw()?;

    chart_context.draw_series(
        Histogram::vertical(&chart_context).data(
            data.clone()
                .map(|(x, (y, y_t))| ((x * 10.) as i32, (y - y_t).abs())),
        ),
    )?;

    Ok(())
}

fn modeled_function<const N: usize, D>(
    tensor: Tensor<Rank2<N, 2>, f64, D>,
) -> Tensor<Rank1<N>, f64, D>
where
    D: Device<f64>,
{
    let dev = tensor.device().clone();
    let params: Tensor<Rank2<N, 3>, f64, D> = tensor.device().tensor([2., 3., -2.]).broadcast();

    let input: Tensor<Rank2<N, 3>, _, _> =
        (tensor, dev.ones::<Rank2<N, 1>>()).concat_tensor_along(Axis::<1>);

    let res: Tensor<Rank1<N>, _, _, _> = (input * params).sum();

    res
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let mut model = dev.build_module::<f64>(LinearConstConfig::<2, 1>::default());

    const N_ITER: usize = 3500;
    const N_POINTS: usize = 20;

    let mut sgd = Sgd::new(
        &model,
        SgdConfig {
            momentum: None,
            lr: 5e-2,
            weight_decay: Some(WeightDecay::L2(1e-30)),
        },
    );

    let dataset = generate_dataset::<N_POINTS, _>(dev.clone());
    let target = modeled_function(dataset.clone());

    let mut loss = Vec::new();

    for i in 0..N_ITER {
        let input = dataset.clone().traced(model.alloc_grads());

        let output: Tensor<Rank1<N_POINTS>, f64, _, _> = model.forward(input).reshape();

        let mse = mse_loss(output, target.clone());

        let loss_val_norm = mse.array();
        loss.push((i as u32, loss_val_norm));

        if i % 100 == 0 {
            println!("loss: {:+e}", loss_val_norm);
        }

        let grads = mse.backward();

        sgd.update(&mut model, &grads)?;
    }

    let svg_backend = SVGBackend::new("plots/poly_fit.svg", (1400, 768)).into_drawing_area();

    let (up, down) = svg_backend.split_vertically(300);

    let (up_left, up_right) = up.split_horizontally(700);

    let (up_left, up_right) = (ChartBuilder::on(&up_left), ChartBuilder::on(&up_right));

    draw_dataset(
        dataset
            .array()
            .into_iter()
            .map(|[a, b]| (a as f64, b as f64, poly(a as f64) as f64)),
        up_left,
    )?;

    let (down_left, down_right) = down.split_horizontally(700);

    let down_left = ChartBuilder::on(&down_left);
    let down_right = ChartBuilder::on(&down_right);

    draw_loss_evolution(loss.into_iter(), down_left, down_right)?;

    let final_output = dataset.as_vec().into_iter().zip(
        model
            .forward(dataset)
            .reshape::<Rank1<N_POINTS>>()
            .as_vec()
            .into_iter()
            .zip(target.as_vec().into_iter())
            .map(|(a, b)| (a, b)),
    );

    draw_model_results(final_output, up_right)?;

    svg_backend.present()?;

    let weights = model.weight.reshape::<Rank1<2>>().array();
    let biases = model.bias.array();

    let (a, b, c) = (weights[0], weights[1], biases[0]);

    println!("a: {}, b: {}, c: {}", a, b, c);
    println!("expected: a: 2, b: 3, c: -2");

    println!("interpretation: prądzi");

    Ok(())
}
