use dfdx::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

use std::path::Path;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use crate::emnist_loader::*;
use crate::plots::*;

use plotters::prelude::*;

pub fn load_chunked_mnist_images<'a, const N_IN: usize, E, D>(
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

pub fn make_one_hots<const N: usize, const N_OUT: usize, E, D>(
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

pub fn convert_max_outputs_to_category<const N_OUT: usize, E, D>(
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

pub fn load_npz_test<const N: usize, const N_IN: usize, E, D>(
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

pub fn decode_characters_npz<const N_IN: usize, const N_OUT: usize, E, D, M>(
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

pub struct TrainSetup<E> {
    pub mnist_train: Vec<MnistImage<E>>,
    pub mnist_test: Vec<MnistImage<E>>,
    pub rng: StdRng,
}

pub struct EpochData<E> {
    pub losses: Vec<E>,
    pub grad_magnitudes: Vec<E>,
    pub predicted_eval: Vec<u8>,
    pub eval_labels: Vec<u8>,
    pub accuracy: f32,
}

fn train<D, E, M, F, const N_IN: usize, const N_OUT: usize, const BATCH_SIZE: usize>(
    dev: D,
    mut model: M,
    train_setup: TrainSetup<E>,
    mut epoch_callback: Option<F>,
) -> Result<(), Box<dyn std::error::Error>>
where
    D: Device<E>,
    E: Dtype + FromPrimitive + npyz::Deserialize + std::fmt::Display,
    M: Module<
        Tensor<(usize, Const<N_IN>), E, D, OwnedTape<E, D>>,
        Output = Tensor<(usize, Const<N_OUT>), E, D, OwnedTape<E, D>>,
    >,
    M: Module<
        Tensor<(usize, Const<N_IN>), E, D, NoneTape>,
        Output = Tensor<(usize, Const<N_OUT>), E, D, NoneTape>,
    >,
    M: UpdateParams<E, D>,
    F: FnMut(&M, EpochData<E>) -> Result<(), Box<dyn std::error::Error>>,
{

    let TrainSetup {
        mut mnist_train,
        mnist_test,
        mut rng,
    } = train_setup;

    println!("Loaded {} test images", mnist_train.len());

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
    let eval_data = eval_data.realize();

    let mut grad_magnitudes = Vec::new();

    for epoch in 0..10000 {
        mnist_train.shuffle(&mut rng);
        let batch_iter = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_train, BATCH_SIZE);

        let mut losses = Vec::new();
        for (batch, labels) in batch_iter.clone() {
            assert_eq!(labels.len(), BATCH_SIZE);

            let one_hots = make_one_hots::<BATCH_SIZE, N_OUT, _, _>(&dev, &labels)?.realize();

            let output = model.try_forward(batch.retaped::<OwnedTape<_, _>>())?;

            let loss = cross_entropy_with_logits_loss(output, one_hots);

            losses.push(loss.as_vec()[0]);

            let grads = loss.backward();
            let tensor_grad_magnitude = grads
                .get(&batch)
                .select(dev.tensor(0))
                .square()
                .sum()
                .sqrt();
            grad_magnitudes.push(tensor_grad_magnitude.as_vec()[0]);

            rms_prop.update(&mut model, &grads)?;
        }

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

        if let Some(ref mut callback) = epoch_callback {
            let epoch_data = EpochData {
                losses,
                grad_magnitudes: grad_magnitudes.clone(),
                predicted_eval,
                eval_labels: eval_labels.clone(),
                accuracy,
            };

            callback(&model, epoch_data)?;
        }
    }

    Ok(())
}
