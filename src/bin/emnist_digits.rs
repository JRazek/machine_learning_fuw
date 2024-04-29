#![feature(iter_array_chunks)]
#![feature(ascii_char)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist_loader::*;
use uczenie_maszynowe_fuw::plots::*;

use plotters::prelude::*;

#[derive(Clone, Sequential, Default)]
struct Model<const N_IN: usize, const N_OUT: usize> {
    linear1: LinearConstConfig<N_IN, 128>,
    activation1: Tanh,

    linear2: LinearConstConfig<128, 128>,
    activation2: FastGeLU,

    linear3: LinearConstConfig<128, 128>,
    activation3: FastGeLU,

    linear4: LinearConstConfig<128, 128>,
    activation4: Tanh,

    linear5: LinearConstConfig<128, 128>,
    activation5: FastGeLU,

    linear6: LinearConstConfig<128, 128>,
    activation6: Tanh,

    linear7: LinearConstConfig<128, 128>,
    activation7: FastGeLU,

    linear8: LinearConstConfig<128, 128>,
    activation8: Tanh,

    linear9: LinearConstConfig<128, N_OUT>,
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

fn convert_max_outputs_to_category<const N_OUT: usize, E, D>(
    output: Tensor<(usize, Const<N_OUT>), E, D>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where
    E: Dtype + FromPrimitive,
    D: Device<E>,
{
    let max_categories: Vec<_> = output
        .as_vec()
        .chunks_exact(N_OUT)
        .map(|chunk| {
            chunk
                .into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u8
        })
        .collect();

    Ok(max_categories)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const BATCH_SIZE: usize = 3000;
    const N_OUT: usize = 36;
    const N_IN: usize = 28 * 28;
    const LABELS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    let mut rng = StdRng::seed_from_u64(0);

    //TODO LOAD
    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/emnist/".to_string());

    let dev = AutoDevice::default();
    let model_path = Path::new("models/character_recognition");

    println!("Device: {:?}", dev);

    println!("Loading mnist train...");
    let mut mnist_train: Vec<_> = load_data::<f32, _, _>(
        format!("{}/emnist-balanced-train-images-idx3-ubyte.gz", mnist_path),
        format!("{}/emnist-balanced-train-labels-idx1-ubyte.gz", mnist_path),
    )?;
    mnist_train.shuffle(&mut rng);
    mnist_train = mnist_train
        .into_iter()
        .filter(|img| img.classification < N_OUT as u8)
        .take(6000)
        .collect();
    println!("Loaded {} training images", mnist_train.len());

    println!("Loading mnist test...");
    let mut mnist_test: Vec<_> = load_data::<f32, _, _>(
        format!("{}/emnist-balanced-test-images-idx3-ubyte.gz", mnist_path),
        format!("{}/emnist-balanced-test-labels-idx1-ubyte.gz", mnist_path),
    )?;
    mnist_test.shuffle(&mut rng);
    mnist_test = mnist_test
        .into_iter()
        .filter(|img| img.classification < N_OUT as u8)
        .take(1000)
        .collect();

    println!("Loaded {} test images", mnist_train.len());

    let mut model = dev.build_module::<f32>(Model::<N_IN, N_OUT>::default());
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
            lr: 1e-3,
            alpha: 0.9,
            eps: 1e-7,
            momentum: Some(0.9),
            centered: false,
            weight_decay: Some(WeightDecay::L2(1e-9)),
        },
    );

    let (eval_data, eval_labels) = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_test, 1000)
        .nth(0)
        .unwrap();

    let mut grad_magnitudes = Vec::new();
    let mut losses = Vec::new();

    for epoch in 0..10000 {
        mnist_train.shuffle(&mut rng);
        let batch_iter = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_train, BATCH_SIZE);

        for (batch, labels) in batch_iter.clone() {
            let tensor = batch.try_realize::<Rank2<BATCH_SIZE, N_IN>>();
            let one_hots = make_one_hots::<BATCH_SIZE, N_OUT, _, _>(&dev, &labels);
            if let (Ok(tensor), Ok(one_hots)) = (tensor, one_hots) {
                assert_eq!(labels.len(), BATCH_SIZE);

                let output = model.try_forward(tensor.retaped::<OwnedTape<_, _>>())?;

                let loss = cross_entropy_with_logits_loss(output, one_hots);

                losses.push(loss.as_vec()[0]);

                let grads = loss.backward();
                let tensor_grad_magnitude = grads
                    .get(&tensor)
                    .select(dev.tensor(0))
                    .square()
                    .sum()
                    .sqrt();
                grad_magnitudes.push(tensor_grad_magnitude.as_vec()[0]);

                rms_prop.update(&mut model, &grads)?;
            }
        }

        model.save_safetensors(model_path)?;

        let svg_backend =
            SVGBackend::new("plots/emnist_digits.svg", (1800, 600)).into_drawing_area();

        let predicted_eval =
            convert_max_outputs_to_category(model.try_forward(eval_data.clone())?)?;

        let accuracy =
            predicted_eval
                .iter()
                .zip(eval_labels.iter())
                .fold(
                    0,
                    |acc, (&predicted, &expected)| {
                        if predicted == expected {
                            acc + 1
                        } else {
                            acc
                        }
                    },
                ) as f32
                / eval_labels.len() as f32;

        println!(
            "Epoch: {}, loss_train: {}, accuracy: {:.2}%",
            epoch,
            losses.last().unwrap(),
            accuracy * 100f32
        );

        match svg_backend.split_evenly((1, 3)).as_slice() {
            [error_matrix_area, losses_area, gradients_area, ..] => {
                plot_error_matrix(
                    &eval_labels,
                    &predicted_eval,
                    36,
                    &|idx| LABELS.as_ascii().unwrap()[idx].to_string(),
                    &error_matrix_area,
                )?;

                plot_log_scale_data(&losses, "loss train", &losses_area)?;
                plot_log_scale_data(&grad_magnitudes, "gradient norm", &gradients_area)?;
            }
            _ => panic!(),
        }

        svg_backend.present()?;
    }

    println!("Saving model...");

    Ok(())
}
