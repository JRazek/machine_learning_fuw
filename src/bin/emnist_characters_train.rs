#![feature(iter_array_chunks)]
#![feature(ascii_char)]
#![feature(generic_const_exprs)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist::*;
use uczenie_maszynowe_fuw::emnist_loader::*;
use uczenie_maszynowe_fuw::plots::*;

use plotters::prelude::*;

const LABELS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";

#[derive(Clone, Sequential, Default)]
struct FullyConnected<const N_IN: usize, const N_OUT: usize> {
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

#[derive(Clone, Default, Sequential)]
struct SubmoduleConvolution {
    conv1: Conv2DConstConfig<1, 10, 3>,
    g1: FastGeLU,

    conv2: Conv2DConstConfig<10, 10, 3>,
    g2: FastGeLU,

    conv3: Conv2DConstConfig<10, 10, 5>,
    g3: FastGeLU,

    conv4: Conv2DConstConfig<10, 10, 5>,
    g4: FastGeLU,

    flatten: Flatten2D,

    l1: LinearConstConfig<2560, 1280>,
    t1: Tanh,

    l2: LinearConstConfig<1280, 640>,
    t2: Tanh,

    l3: LinearConstConfig<640, 320>,
    t3: Tanh,

    l4: LinearConstConfig<320, 160>,
    t4: Tanh,

    l5: LinearConstConfig<160, 37>,
}

#[derive(
    Clone,
    ResetParams,
    UpdateParams,
    ZeroGrads,
    SaveSafeTensors,
    LoadSafeTensors,
    CustomModule,
    Default,
)]
#[built(ConvolutionModel)]
struct ConvolutionModelConfig {
    #[module]
    submodule: SubmoduleConvolution,
}

impl<E, D, TapeT> Module<Tensor<(usize, Const<784>), E, D, TapeT>> for ConvolutionModel<E, D>
where
    E: Dtype,
    D: Device<E>,
    TapeT: Tape<E, D>,

    //TODO Trzeba by pokombinować jak się tego pozbyć..
    Conv2D<Const<1>, Const<10>, Const<3>, Const<1>, Const<0>, Const<1>, Const<1>, E, D>: Module<
        Tensor<(usize, Const<1>, Const<28>, Const<28>), E, D, TapeT>,
        Output = Tensor<(usize, Const<10>, Const<26>, Const<26>), E, D, TapeT>,
    >,
    Conv2D<Const<10>, Const<10>, Const<3>, Const<1>, Const<0>, Const<1>, Const<1>, E, D>: Module<
        Tensor<(usize, Const<10>, Const<26>, Const<26>), E, D, TapeT>,
        Output = Tensor<(usize, Const<10>, Const<24>, Const<24>), E, D, TapeT>,
    >,
    Conv2D<Const<10>, Const<10>, Const<5>, Const<1>, Const<0>, Const<1>, Const<1>, E, D>: Module<
        Tensor<(usize, Const<10>, Const<24>, Const<24>), E, D, TapeT>,
        Output = Tensor<(usize, Const<10>, Const<20>, Const<20>), E, D, TapeT>,
    >,
    Conv2D<Const<10>, Const<10>, Const<5>, Const<1>, Const<0>, Const<1>, Const<1>, E, D>: Module<
        Tensor<(usize, Const<10>, Const<20>, Const<20>), E, D, TapeT>,
        Output = Tensor<(usize, Const<10>, Const<16>, Const<16>), E, D, TapeT>,
    >,
{
    type Output = Tensor<(usize, Const<37>), E, D, TapeT>;

    fn try_forward(
        &self,
        x: Tensor<(usize, Const<784>), E, D, TapeT>,
    ) -> Result<Self::Output, Error> {
        let n = x.shape().0;
        let reshaped = x.try_reshape_like(&(n, Const::<1>, Const::<28>, Const::<28>))?;

        let out = self.submodule.try_forward(reshaped)?;

        Ok(out)
    }
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
    plot_path: &str,
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

    let mut train_losses_all = Vec::new();
    let mut grad_magnitudes_all_train = Vec::new();

    let mut test_losses_all = Vec::new();

    train::<N_IN, N_OUT, 3000, _, _, _, _>(
        dev,
        model,
        train_setup,
        |model,
         EpochData {
             epoch,
             train_losses,
             train_predicted,
             train_labels,
             train_accuracies,

             grad_magnitudes,
             test_predicted,
             test_labels,
             test_accuracies: test_accuracy,
             test_losses,
         }| {
            train_losses_all.extend_from_slice(&train_losses);
            grad_magnitudes_all_train.extend_from_slice(&grad_magnitudes);

            test_losses_all.extend_from_slice(&test_losses);

            model.save_safetensors(model_save_path.as_ref())?;

            println!(
                "Epoch: {}, loss_train: {}, accuracy_train: {:.2}%, accuracy_test: {:.2}%",
                epoch,
                train_losses.last().unwrap(),
                train_accuracies[0] * 100f32,
                test_accuracy[0] * 100f32
            );

            let svg_backend = SVGBackend::new(plot_path, (1800, 1000)).into_drawing_area();

            let (left, gradients_area_train) = svg_backend.split_horizontally(1100);

            match left.split_evenly((2, 2)).as_slice() {
                [error_matrix_area_train, error_matrix_area_test, losses_area_train, losses_area_test] =>
                {
                    plot_error_matrix(
                        &train_labels,
                        &train_predicted,
                        N_OUT,
                        &|idx| LABELS.as_ascii().unwrap()[idx].to_string(),
                        "train",
                        &error_matrix_area_train,
                    )?;

                    let losses_train: Vec<f32> =
                        train_losses_all.iter().map(|&x| x.into()).collect();
                    plot_log_scale_data(&losses_train, "loss train", &losses_area_train)?;

                    let grad_magnitudes_train: Vec<f32> = grad_magnitudes_all_train
                        .iter()
                        .map(|&x| x.into())
                        .collect();
                    plot_log_scale_data(
                        &grad_magnitudes_train,
                        "gradient norm (train)",
                        &gradients_area_train,
                    )?;

                    plot_error_matrix(
                        &test_labels,
                        &test_predicted,
                        N_OUT,
                        &|idx| LABELS.as_ascii().unwrap()[idx].to_string(),
                        "test",
                        &error_matrix_area_test,
                    )?;

                    let test_losses: Vec<f32> = test_losses_all.iter().map(|&x| x.into()).collect();
                    plot_log_scale_data(&test_losses, "loss test", &losses_area_test)?;
                }
                _ => panic!(),
            }

            svg_backend.present()?;

            Ok(())
        },
    )?;

    Ok(())
}

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    pub nn_type: String,

    #[arg(long)]
    pub mode: String,

    #[arg(long)]
    pub model_path: Option<String>,

    #[arg(long)]
    pub emnist_path: Option<String>,

    #[arg(long)]
    pub ngz_path: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.nn_type.as_str() {
        "fc" => run_fully_connected(args)?,
        "convolution" => run_convolution(args)?,
        _ => println!("Unknown nn_type: {}", args.nn_type),
    }

    Ok(())
}

fn run_fully_connected(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    const N_IN: usize = 28 * 28;
    const N_OUT: usize = LABELS.len();

    let dev = AutoDevice::default();

    let mut model = dev.build_module::<f32>(FullyConnected::<N_IN, N_OUT>::default());

    let model_path = args
        .model_path
        .unwrap_or_else(|| "models/emnist_fully_connected".to_string());

    match model.load_safetensors(&model_path) {
        Ok(_) => {
            println!("Model loaded successfully!");
        }
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("Proceeding with randomly initialized model...");
        }
    }

    let mode = args.mode;
    match mode.as_str() {
        "train" => {
            let mnist_path = args
                .emnist_path
                .unwrap_or_else(|| "data/emnist".to_string());

            training_pipeline(
                &mnist_path,
                dev,
                model,
                "plots/emnist_balanced_fully_connected.svg",
                &model_path,
            )?;
        }
        "decode" => {
            let ngz_path = args.ngz_path.unwrap_or_else(|| "data/emnist/".to_string());

            println!("Decoding: {}", ngz_path);
            let tensor = load_npz_test::<N_OUT, N_IN, f64, _>(&ngz_path, &dev)?.to_dtype::<f32>();

            let decoded = decode_characters_npz(&mut model, tensor)?
                .into_iter()
                .map(|idx| LABELS.as_ascii().unwrap()[idx as usize])
                .collect::<Vec<_>>();

            println!("Decoded: {:?}", decoded);
        }
        _ => println!("Unknown mode: {}", mode),
    }

    Ok(())
}

fn run_convolution(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    const N_IN: usize = 28 * 28;
    const N_OUT: usize = LABELS.len();

    let dev = AutoDevice::default();

    let mut model = dev.build_module::<f32>(ConvolutionModelConfig::default());

    let model_path = args
        .model_path
        .unwrap_or_else(|| "models/emnist_convolution".to_string());

    match model.load_safetensors(&model_path) {
        Ok(_) => {
            println!("Model loaded successfully!");
        }
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("Proceeding with randomly initialized model...");
        }
    }

    let mode = args.mode;
    match mode.as_str() {
        "train" => {
            let mnist_path = args
                .emnist_path
                .unwrap_or_else(|| "data/emnist".to_string());

            training_pipeline::<N_IN, N_OUT, _, _, _>(
                &mnist_path,
                dev,
                model,
                "plots/emnist_balanced_convolution.svg",
                &model_path,
            )?;
        }
        "decode" => {
            let ngz_path = args.ngz_path.unwrap_or_else(|| "data/emnist/".to_string());

            println!("Decoding: {}", ngz_path);
            let tensor = load_npz_test::<N_OUT, N_IN, f64, _>(&ngz_path, &dev)?.to_dtype::<f32>();

            let decoded = decode_characters_npz(&mut model, tensor)?
                .into_iter()
                .map(|idx| LABELS.as_ascii().unwrap()[idx as usize])
                .collect::<Vec<_>>();

            println!("Decoded: {:?}", decoded);
        }
        _ => println!("Unknown mode: {}", mode),
    }

    Ok(())
}
