#![feature(iter_array_chunks)]

use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use uczenie_maszynowe_fuw::emnist_loader::*;
use uczenie_maszynowe_fuw::plots::plot_error_matrix;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();
    const BATCH_SIZE: usize = 100;
    let model_path = Path::new("models/character_recognition");

    println!("Device: {:?}", dev);

    const N_IN: usize = 28 * 28;

    println!("Loading mnist...");

    const N_OUT: usize = 36;
    let mut mnist_train: Vec<_> = load_data::<f32, _, _>(
        "/home/user/Pictures/qm_homework/emnist/emnist-letters-train-images-idx3-ubyte.gz",
        "/home/user/Pictures/qm_homework/emnist/emnist-letters-train-labels-idx1-ubyte.gz",
    )?
    .into_iter()
    .filter(|img| img.classification < N_OUT as u8)
    .collect();

    println!("Loaded {} images", mnist_train.len());

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

    let eval: Vec<_> = load_chunked_mnist_images::<1000, _, _>(
        &dev,
        mnist_train.split_off(mnist_train.len() - 1000).as_slice(),
        1,
    )
    .collect();

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
    }

    println!("Saving model...");

    Ok(())
}
