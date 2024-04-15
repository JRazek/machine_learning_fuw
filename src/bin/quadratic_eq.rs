use dfdx::prelude::*;
use rand_distr::Uniform;

type DTYPE = f32;

fn quadratic_eq_gen<const N: usize, D>(
    dev: &mut D,
    seed: u64,
) -> (Tensor<Rank2<N, 3>, DTYPE, D>, Tensor<Rank2<N, 2>, DTYPE, D>)
where
    D: Device<DTYPE>,
{
    use rand::prelude::*;

    let mut coff_vec = Vec::new();
    let mut sol_vec = Vec::new();

    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1., 1.);

    for _ in 0..N {
        let a = dist.sample(&mut rng);
        let mut x1 = dist.sample(&mut rng);
        let mut x2 = dist.sample(&mut rng);

        if x1 < x2 {
            std::mem::swap(&mut x1, &mut x2);
        }

        let b = -a * (x1 + x2);
        let c = a * x1 * x2;

        let coefficients = [a, b, c];

        coff_vec.push(coefficients);
        sol_vec.push([x1, x2]);
    }

    let coff_tensor: Tensor<Rank2<N, 3>, _, D> = dev.tensor_from_vec(
        coff_vec.into_iter().flatten().collect(),
        Rank2::<N, 3>::default(),
    );

    let sol_tensor: Tensor<Rank2<N, 2>, _, D> = dev.tensor_from_vec(
        sol_vec.into_iter().flatten().collect(),
        Rank2::<N, 2>::default(),
    );

    (coff_tensor, sol_tensor)
}

fn quadratic_eq_solve([a, b, c]: [DTYPE; 3]) -> Option<[DTYPE; 2]> {
    let discriminant = b.clone().powi(2) - 4. * a.clone() * c.clone();

    if discriminant < 0. {
        return None;
    }

    let discriminant_sqrt = discriminant.sqrt();

    let x1 = (-b.clone() + discriminant_sqrt.clone()) / (a.clone() * 2.);
    let x2 = (-b - discriminant_sqrt) / (a * 2.);

    Some([x1, x2])
}

fn train<M, D, O, const N: usize>(
    model: &mut M,
    dataset: Tensor<Rank2<N, 3>, DTYPE, D>,
    targets: Tensor<Rank2<N, 2>, DTYPE, D>,
    batch_size: usize,
    optimizer: &mut O,
) -> Result<Tensor<Rank0, DTYPE, D>, Box<dyn std::error::Error>>
where
    D: Device<DTYPE>,
    M: Module<
            Tensor<(usize, Const<3>), DTYPE, D, OwnedTape<DTYPE, D>>,
            Output = Tensor<(usize, Const<2>), DTYPE, D, OwnedTape<DTYPE, D>>,
        > + UpdateParams<DTYPE, D>,
    O: Optimizer<M, DTYPE, D>,
{
    if N < batch_size {
        Err("Batch size must be less than the dataset size")?;
    }

    let batches = (0..(N.div_euclid(batch_size))).map(|i| {
        let start = i * batch_size;
        let end = (i + 1) * batch_size;

        let inputs = dataset.clone().slice((start..end, ..));
        let targets = targets.clone().slice((start..end, ..));

        (inputs, targets)
    });

    let mut losses = Vec::new();

    for (inputs, targets) in batches.clone() {
        let outputs = model.forward(inputs.retaped::<OwnedTape<_, _>>());
        let loss = mse_loss(outputs, targets.clone());

        losses.push(loss.retaped::<NoneTape>());

        let grads = loss.backward();

        optimizer.update(model, &grads)?;
    }

    let mean_loss = losses.stack().try_mean()?;

    Ok(mean_loss)
}

#[derive(Clone, Sequential, Default)]
struct Model {
    linear1: LinearConstConfig<3, 50>,
    activation1: Tanh,
    linear2: LinearConstConfig<50, 100>,
    activation2: FastGeLU,
    linear3: LinearConstConfig<100, 100>,
    activation3: FastGeLU,
    linear4: LinearConstConfig<100, 100>,
    activation4: FastGeLU,
    linear5: LinearConstConfig<100, 20>,
    activation5: Tanh,
    linear6: LinearConstConfig<20, 20>,
    activation6: FastGeLU,
    linear7: LinearConstConfig<20, 2>,
}

fn eval_errors<M, D, const N: usize>(
    model: &M,
    dataset: Tensor<Rank2<N, 3>, DTYPE, D>,
    targets: Tensor<Rank2<N, 2>, DTYPE, D>,
) -> Result<Tensor<Rank2<N, 2>, DTYPE, D>, Box<dyn std::error::Error>>
where
    D: Device<DTYPE>,
    M: Module<
        Tensor<(usize, Const<3>), DTYPE, D, NoneTape>,
        Output = Tensor<(usize, Const<2>), DTYPE, D, NoneTape>,
    >,
{
    let dataset = dataset.realize();

    let output = model.forward(dataset).realize();

    let rel_err: Tensor<Rank2<N, 2>, DTYPE, D> = output
        .try_sub(targets.clone())?
        .try_div(targets)?
        .nans_to(0.)
        .realize();

    Ok(rel_err)
}

//https://github.com/plotters-rs/plotters/blob/68790025c362ca0dc8ea47f48239cc9d8e09d6f6/plotters/src/coord/ranged1d/combinators/linspace.rs#L161
fn plot_results(errors: Vec<f32>, losses: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let svg_backend =
        SVGBackend::new("plots/quadratic_eq_solutions.svg", (1600, 600)).into_drawing_area();
    svg_backend.fill(&WHITE)?;

    let (left, right) = svg_backend.split_horizontally(800);

    let mut left = ChartBuilder::on(&left);

    let mut chart_context_left = left
        .caption("Quadratic equation solutions", ("Arial", 20))
        .set_all_label_area_size(50)
        .margin(50)
        .build_cartesian_2d((-5f32..5f32).step(0.1).use_round(), 0..500)?;

    chart_context_left
        .configure_mesh()
        .x_labels(30)
        .x_label_formatter(&|x| format!("{:.0}", x))
        .x_desc("Bucket")
        .y_labels(20)
        .y_desc("Relative Error")
        .draw()?;

    let histogram = Histogram::vertical(&chart_context_left)
        .style(BLUE.mix(0.5).filled())
        .margin(0)
        .data(errors.into_iter().map(|e| (e as f32, 1)));

    chart_context_left.draw_series(histogram)?;

    let mut right = ChartBuilder::on(&right);

    let max_loss = *losses
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart_context_right = right
        .caption("Loss", ("Arial", 20))
        .set_all_label_area_size(70)
        .margin(50)
        .build_cartesian_2d(0..losses.len(), (0f32..max_loss).log_scale())?;

    chart_context_right
        .configure_mesh()
        .x_labels(30)
        .x_desc("Iteration")
        .y_labels(20)
        .y_desc("Loss")
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;

    let losses = LineSeries::new(
        losses.into_iter().enumerate().map(|(i, l)| (i, l)),
        BLUE.filled(),
    );

    chart_context_right.draw_series(losses)?;

    svg_backend.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = AutoDevice::default();

    println!("Device: {:?}", dev);

    let mut model = dev.build_module::<DTYPE>(Model::default());

    match model.load_safetensors("models/quadratic_eq_model") {
        Ok(_) => {
            println!("Model loaded successfully!");
        }
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("Proceeding with randomly initialized model...");
        }
    }

    let mut rms_prop = RMSprop::new(
        &model,
        RMSpropConfig {
            lr: 1e-4,
            alpha: 0.9,
            eps: 1e-7,
            momentum: None,
            centered: false,
            weight_decay: Some(WeightDecay::L2(1e-9)),
        },
    );

    let mut min_loss_without_updates = 0;
    let min_mean_loss = 1e6;

    let patience = 100;

    let mut losses = Vec::new();

    let (eval_coeff, eval_solutions): (Tensor<Rank2<1000, 3>, _, _>, _) =
        quadratic_eq_gen(&mut dev, 2);

    for epoch in 0..1000000 {
        let (coefficients, solutions): (Tensor<Rank2<10000, 3>, _, _>, _) =
            quadratic_eq_gen(&mut dev, epoch);

        let mean_loss = train(
            &mut model,
            coefficients.clone(),
            solutions.clone(),
            1000,
            &mut rms_prop,
        )?
        .as_vec()[0];

        losses.push(mean_loss);

        if min_loss_without_updates > patience {
            rms_prop.cfg.lr *= 0.1;
            println!("Learning rate decayed to {}", rms_prop.cfg.lr);

            min_loss_without_updates = 0;
        }

        if mean_loss < min_mean_loss {
            min_loss_without_updates = 0;
        } else {
            min_loss_without_updates += 1;
        }

        println!("Epoch: {} Mean Loss: {:?}", epoch, mean_loss);

        let eval = eval_errors(&model, eval_coeff.clone(), eval_solutions.clone())?.as_vec();

        const DELTA: DTYPE = 1e-2;
        let metric = eval.iter().filter(|&&e| e.abs() < DELTA).count() as f64 / eval.len() as f64;

        println!("Metric on test dataset {:.2}%", metric * 100.);

        if epoch % 1000 == 0 {
            model.save_safetensors("models/quadratic_eq_model")?;
            println!("Model saved on epoch {}", epoch);

            plot_results(eval, losses.clone())?;
        }
    }

    Ok(())
}
