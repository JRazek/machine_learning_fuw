use dfdx::prelude::*;

use rand::rngs::StdRng;

use rand::seq::SliceRandom;

use num::FromPrimitive;

use crate::emnist_loader::*;

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

        //not all chunks are of chunk_size - last one may not be.
        let current_chunk_len = chunk.len();

        let tensor = dev.tensor_from_vec(buffer, (current_chunk_len, Const::<N_IN>::default()));

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

#[derive(Debug)]
pub struct TrainSetup<E> {
    pub mnist_train: Vec<MnistImage<E>>,
    pub mnist_test: Vec<MnistImage<E>>,
    pub rng: StdRng,
}

#[derive(Debug, Clone)]
pub struct EpochData<E> {
    pub epoch: usize,
    pub grad_magnitudes: Vec<E>,

    pub train_losses: Vec<E>,
    pub train_predicted: Vec<u8>,
    pub train_labels: Vec<u8>,
    pub train_accuracies: Vec<f32>,

    pub test_losses: Vec<E>,
    pub test_predicted: Vec<u8>,
    pub test_labels: Vec<u8>,
    pub test_accuracies: Vec<f32>,
}

pub fn train<const N_IN: usize, const N_OUT: usize, const BATCH_SIZE: usize, D, E, M, F>(
    dev: D,
    mut model: M,
    train_setup: TrainSetup<E>,
    mut epoch_callback: F,
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

    const EVAL_CHUNK_SIZE: usize = 1000;
    let (eval_data, eval_labels) =
        load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_test, EVAL_CHUNK_SIZE)
            .nth(0)
            .unwrap();

    let mut grad_magnitudes = Vec::new();

    let accuracy_fn = |predicted: &[u8], expected: &[u8]| {
        predicted
            .iter()
            .zip(expected.iter())
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
            / expected.len() as f32
    };

    for epoch in 0..10000 {
        mnist_train.shuffle(&mut rng);
        let batch_iter = load_chunked_mnist_images::<N_IN, _, _>(&dev, &mnist_train, BATCH_SIZE);

        let mut train_losses = Vec::new();
        let mut train_accuracies = Vec::new();

        let mut train_predicted = Vec::new();
        let mut train_labels = Vec::new();

        for (i, (batch, labels)) in batch_iter.clone().enumerate() {
            assert_eq!(labels.len(), BATCH_SIZE);

            let one_hots = make_one_hots::<BATCH_SIZE, N_OUT, _, _>(&dev, &labels)?.realize();

            let output = model.try_forward(batch.retaped::<OwnedTape<_, _>>())?;

            let predicted = convert_max_outputs_to_category(output.retaped())?;
            let accuracy = accuracy_fn(&predicted, &labels);
            train_accuracies.push(accuracy);

            let train_loss = cross_entropy_with_logits_loss(output, one_hots);

            train_losses.push(train_loss.as_vec()[0]);

            let grads = train_loss.backward();
            let tensor_grad_magnitude = grads
                .get(&batch)
                .select(dev.tensor(0))
                .square()
                .sum()
                .sqrt();
            grad_magnitudes.push(tensor_grad_magnitude.as_vec()[0]);

            rms_prop.update(&mut model, &grads)?;

            if i == 0 {
                train_predicted = predicted.clone();
                train_labels = labels.clone();
            }
        }

        let test_output = model.try_forward(eval_data.clone())?;

        let predicted_eval = convert_max_outputs_to_category(test_output.clone())?;
        let test_loss = cross_entropy_with_logits_loss(
            test_output,
            make_one_hots::<EVAL_CHUNK_SIZE, N_OUT, _, _>(&dev, &eval_labels)?.realize(),
        )
        .as_vec()[0];

        let test_accuracies = accuracy_fn(&predicted_eval, &eval_labels);

        let epoch_data = EpochData {
            epoch,
            grad_magnitudes: grad_magnitudes.clone(),

            train_losses,
            train_predicted,
            train_labels,
            train_accuracies,

            test_predicted: predicted_eval,
            test_labels: eval_labels.clone(),
            test_losses: vec![test_loss],
            test_accuracies: vec![test_accuracies],
        };

        epoch_callback(&model, epoch_data)?;
    }

    Ok(())
}
