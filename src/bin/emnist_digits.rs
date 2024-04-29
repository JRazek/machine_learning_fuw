#![feature(iter_array_chunks)]
#![feature(ascii_char)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist_loader::*;
use uczenie_maszynowe_fuw::plots::plot_error_matrix;

use plotters::prelude::*;

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
    softmax: Softmax,
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
    const BATCH_SIZE: usize = 100;
    const N_OUT: usize = 36;
    const N_IN: usize = 28 * 28;
    const LABELS: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    //TODO LOAD
    let mnist_path = "/home/user/Pictures/qm_homework/emnist/";

    let dev = AutoDevice::default();
    let model_path = Path::new("models/character_recognition");

    println!("Device: {:?}", dev);

    println!("Loading mnist train...");
    let mut mnist_train: Vec<_> = load_data::<f32, _, _>(
        format!("{}/emnist-letters-train-images-idx3-ubyte.gz", mnist_path),
        format!("{}/emnist-letters-train-labels-idx1-ubyte.gz", mnist_path),
    )?
    .into_iter()
    .filter(|img| img.classification < N_OUT as u8)
    .take(20000)
    .collect();
    println!("Loaded {} training images", mnist_train.len());

    println!("Loading mnist test...");
    let mnist_test: Vec<_> = load_data::<f32, _, _>(
        format!("{}/emnist-letters-test-images-idx3-ubyte.gz", mnist_path),
        format!("{}/emnist-letters-test-labels-idx1-ubyte.gz", mnist_path),
    )?
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
            lr: 1e-4,
            alpha: 0.9,
            eps: 1e-7,
            momentum: None,
            centered: false,
            weight_decay: Some(WeightDecay::L2(1e-9)),
        },
    );

    let mut rng = StdRng::seed_from_u64(0);

    let (eval_data, eval_labels) = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_test, 1000)
        .nth(0)
        .unwrap();

    for epoch in 0..100 {
        let mut loss_epoch = None;

        mnist_train.shuffle(&mut rng);

        let batch_iter = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_train, BATCH_SIZE);

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

        let svg_backend =
            SVGBackend::new("plots/emnist_digits.svg", (800, 600)).into_drawing_area();

        let predicted_eval =
            convert_max_outputs_to_category(model.try_forward(eval_data.clone())?)?;

        plot_error_matrix(
            &eval_labels,
            &predicted_eval,
            36,
            &|idx| LABELS.as_ascii().unwrap()[idx].to_string(),
            &svg_backend,
        )?;
    }

    println!("Saving model...");

    Ok(())
}
