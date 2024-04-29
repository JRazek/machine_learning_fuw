#![feature(iter_array_chunks)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist_loader::*;

type DTYPE = f32;

fn load_mnist<const N: usize, D>(
    dev: &mut D,
    seed: u64,
) -> (Tensor<Rank2<N, 3>, DTYPE, D>, Tensor<Rank2<N, 2>, DTYPE, D>)
where
    D: Device<DTYPE>,
{
    todo!()
}

fn eval<M, E, D, const N: usize, const N_IN: usize, const N_OUT: usize>(
    model: &M,
    data: Tensor<Rank2<N, N_IN>, E, D>,
    labels: Tensor<Rank2<N, N_OUT>, E, D>,
) -> Result<Tensor<Rank0, E, D, OwnedTape<E, D>>, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
    M: Module<
        Tensor<(usize, Const<N_IN>), E, D, OwnedTape<E, D>>,
        Output = Tensor<(usize, Const<N_OUT>), E, D, OwnedTape<E, D>>,
    >,
{
    let dataset = data.reshape_like(&(N, Const::<N_IN>::default()));

    let outputs = model
        .forward(dataset.retaped::<OwnedTape<_, _>>())
        .try_realize::<Rank2<N, N_OUT>>()
        .unwrap();

    let loss = cross_entropy_with_logits_loss(outputs, labels);

    Ok(loss)
}

#[derive(Clone, Sequential, Default)]
struct Model<const N_IN: usize, const N_OUT: usize> {
    linear1: LinearConstConfig<N_IN, 50>,
    activation1: Tanh,
    linear4: LinearConstConfig<50, 100>,
    activation4: FastGeLU,
    linear5: LinearConstConfig<100, 20>,
    activation5: Tanh,
    linear6: LinearConstConfig<20, 20>,
    activation6: FastGeLU,
    linear7: LinearConstConfig<20, N_OUT>,
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

fn load_chunked_mnist_images<'a, const N_IN: usize, E, D>(
    dev: &'a D,
    mnist: &'a [MnistImage<E>],
    chunk_size: usize,
) -> impl Iterator<Item = (Tensor<(usize, Const<N_IN>), E, D>, Vec<u8>)> + Clone + 'a
where
    E: Dtype,
    D: Device<E> + 'a,
{
    let mnist = mnist.chunks(chunk_size).into_iter().map(|chunk| {
        let buffer: Vec<E> = chunk
            .into_iter()
            .map(|MnistImage { image, .. }| image.iter().copied())
            .flatten()
            .collect();

        let len = chunk.len();

        let tensor = dev.tensor_from_vec(buffer, (len, Const::<N_IN>::default()));

        let labels: Vec<u8> = chunk
            .into_iter()
            .map(|&MnistImage { classification, .. }| classification)
            .collect();

        (tensor, labels)
    });

    mnist
}

fn plot_evals() {}

fn make_one_hots<const N: usize, const N_OUT: usize, E, D>(
    dev: &D,
    labels: &[u8],
) -> Result<Tensor<Rank2<N, N_OUT>, E, D>, Box<dyn std::error::Error>>
where
    E: Dtype + FromPrimitive,
    D: Device<E>,
{
    if labels.len() != N {
        return Err("Invalid number of labels".into());
    }
    let one_hots = labels
        .into_iter()
        .flat_map(|&lbl| {
            let mut vec = vec![E::default(); N_OUT];
            vec[lbl as usize] = E::from_f32(1.0).unwrap();

            vec.into_iter()
        })
        .collect::<Vec<_>>();

    let one_hots_tensor = dev
        .tensor_from_vec(one_hots, (labels.len(), Const::<N_OUT>::default()))
        .try_realize::<Rank2<N, N_OUT>>()
        .unwrap();

    Ok(one_hots_tensor)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = AutoDevice::default();
    const BATCH_SIZE: usize = 100;
    let model_path = Path::new("models/character_recognition");

    println!("Device: {:?}", dev);

    const N_IN: usize = 28 * 28;

    println!("Loading mnist...");

    const N_OUT: usize = 36;
    let mut mnist: Vec<_> = load_data::<f32>("/home/user/Pictures/qm_homework/emnist")?
        .into_iter()
        .filter(|img| img.classification < N_OUT as u8)
        .collect();

    println!("Loaded {} images", mnist.len());

    let mut model = dev.build_module::<DTYPE>(Model::<N_IN, N_OUT>::default());

    match model.load_safetensors(model_path) {
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

    let mut rng = StdRng::seed_from_u64(0);

    for epoch in 0..100 {
        let mut loss_epoch = None;

        mnist.shuffle(&mut rng);

        let batch_iter = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist, BATCH_SIZE);

        for (batch, labels) in batch_iter.clone() {
            let tensor = batch.try_realize::<Rank2<BATCH_SIZE, N_IN>>();
            let one_hots = make_one_hots::<BATCH_SIZE, N_OUT, _, _>(&dev, &labels);
            if let (Ok(tensor), Ok(one_hots)) = (tensor, one_hots) {
                assert_eq!(labels.len(), BATCH_SIZE);

                let output = model.try_forward(tensor.retaped::<OwnedTape<_, _>>())?;

                let loss = cross_entropy_with_logits_loss(output, one_hots);
                loss_epoch = Some(loss.array());

                let grads = loss.backward();

                rms_prop.update(&mut model, &grads)?;
            }
        }

        let loss_epoch = loss_epoch.unwrap();
        println!("Epoch: {}, Loss: {}", epoch, loss_epoch);

        model.save_safetensors(model_path)?;
    }

    println!("Saving model...");

    Ok(())
}
