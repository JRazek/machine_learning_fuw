#![feature(iter_array_chunks)]
#![feature(ascii_char)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist::*;
use uczenie_maszynowe_fuw::emnist_loader::*;
use uczenie_maszynowe_fuw::plots::*;

use plotters::prelude::*;

const LABELS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

#[derive(Clone, Sequential, Default)]
struct Model<const N_IN: usize, const N_OUT: usize> {
    linear1: LinearConstConfig<N_IN, 128>,
    activation1: Tanh,

    linear2: LinearConstConfig<128, 128>,
    activation2: Tanh,

    linear3: LinearConstConfig<128, 128>,
    activation3: FastGeLU,

    linear7: LinearConstConfig<128, 128>,
    activation7: Tanh,

    linear8: LinearConstConfig<128, 128>,
    activation8: Tanh,

    linear9: LinearConstConfig<128, N_OUT>,
}

fn load_npz_test<const N: usize, const N_IN: usize, E, D>(
    path: &str,
    dev: &D,
) -> Result<Tensor<(usize, Const<N_IN>), E, D>, Box<dyn std::error::Error>>
where
    E: Dtype + FromPrimitive + npyz::Deserialize,
    D: Device<E> + TensorFromVec<E>,
{
    let raw = std::fs::read(path)?;

    let npz = npyz::NpyFile::new(&raw[..])?;

    let n_digits = npz.shape()[0];

    let data = npz.into_vec::<E>()?;

    let tensor = dev.tensor_from_vec(data, (n_digits as usize, Const::<N_IN>::default()));

    Ok(tensor)
}

fn decode_characters_npz<const N_IN: usize, const N_OUT: usize, E, D, M>(
    model: &mut M,
    tensor: Tensor<(usize, Const<N_IN>), E, D>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where
    E: Dtype,
    D: Device<E>,
    M: Module<Tensor<(usize, Const<N_IN>), E, D>, Output = Tensor<(usize, Const<N_OUT>), E, D>>
        + UpdateParams<E, D>,
{
    let out = model.try_forward(tensor)?;
    let categories = convert_max_outputs_to_category(out)?;

    Ok(categories)
}

fn training_pipeline<const N_IN: usize, const N_OUT: usize, D, E, M>(
    emnist_path: &str,
    dev: D,
    model: M,
    model_save_path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>>
where
    D: Device<E>,
    E: Dtype + FromPrimitive + num::Float + npyz::Deserialize + std::fmt::Display + Into<f32>,
    M: Module<
        Tensor<(usize, Const<N_IN>), E, D, OwnedTape<E, D>>,
        Output = Tensor<(usize, Const<N_OUT>), E, D, OwnedTape<E, D>>,
    >,
    M: Module<
        Tensor<(usize, Const<N_IN>), E, D, NoneTape>,
        Output = Tensor<(usize, Const<N_OUT>), E, D, NoneTape>,
    >,
    M: UpdateParams<E, D>,
    M: SaveSafeTensors,
{
    let mut rng = StdRng::seed_from_u64(0);

    println!("Device: {:?}", dev);

    println!("Loading mnist train...");
    let mut mnist_train: Vec<_> = load_data::<E, _, _>(
        format!("{}/emnist-balanced-train-images-idx3-ubyte.gz", emnist_path),
        format!("{}/emnist-balanced-train-labels-idx1-ubyte.gz", emnist_path),
    )?;
    mnist_train.shuffle(&mut rng);
    mnist_train = mnist_train
        .into_iter()
        .filter(|img| img.classification < N_OUT as u8)
        .take(6000)
        .collect();
    println!("Loaded {} training images", mnist_train.len());

    println!("Loading mnist test...");
    let mut mnist_test: Vec<_> = load_data::<E, _, _>(
        format!("{}/emnist-balanced-test-images-idx3-ubyte.gz", emnist_path),
        format!("{}/emnist-balanced-test-labels-idx1-ubyte.gz", emnist_path),
    )?;
    mnist_test.shuffle(&mut rng);
    mnist_test = mnist_test
        .into_iter()
        .filter(|img| img.classification < N_OUT as u8)
        .take(1000)
        .collect();

    println!("Loaded {} test images", mnist_train.len());

    let train_setup = TrainSetup {
        mnist_train,
        mnist_test,
        rng,
    };

    let mut losses_all = Vec::new();
    let mut grad_magnitudes_all = Vec::new();

    train::<N_IN, N_OUT, 3000, _, _, _, _>(
        dev,
        model,
        train_setup,
        |model,
         EpochData {
             epoch,
             losses,
             grad_magnitudes,
             predicted_eval,
             eval_labels,
             accuracy,
         }| {
            losses_all.extend_from_slice(&losses);
            grad_magnitudes_all.extend_from_slice(&grad_magnitudes);

            model.save_safetensors(model_save_path.as_ref())?;

            println!(
                "Epoch: {}, loss_train: {}, accuracy: {:.2}%",
                epoch,
                losses.last().unwrap(),
                accuracy * 100f32
            );

            let svg_backend =
                SVGBackend::new("plots/emnist_digits.svg", (1800, 600)).into_drawing_area();

            match svg_backend.split_evenly((1, 3)).as_slice() {
                [error_matrix_area, losses_area, gradients_area, ..] => {
                    plot_error_matrix(
                        &eval_labels,
                        &predicted_eval,
                        36,
                        &|idx| LABELS.as_ascii().unwrap()[idx].to_string(),
                        &error_matrix_area,
                    )?;

                    let losses: Vec<f32> = losses_all.iter().map(|&x| x.into()).collect();
                    plot_log_scale_data(&losses, "loss train", &losses_area)?;

                    let grad_magnitudes: Vec<f32> =
                        grad_magnitudes_all.iter().map(|&x| x.into()).collect();
                    plot_log_scale_data(&grad_magnitudes, "gradient norm", &gradients_area)?;
                }
                _ => panic!(),
            }

            svg_backend.present()?;

            Ok(())
        },
    )?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_IN: usize = 28 * 28;
    const N_OUT: usize = LABELS.len();

    let mnist_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/emnist/".to_string());

    let model_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/emnist_characters_model".to_string());

    let dev = AutoDevice::default();

    let mut model = dev.build_module::<f32>(Model::<N_IN, N_OUT>::default());

    match model.load_safetensors(&model_path) {
        Ok(_) => {
            println!("Model loaded successfully!");
        }
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("Proceeding with randomly initialized model...");
        }
    }

    training_pipeline(&mnist_path, dev, model, &model_path)?;

    Ok(())
}
