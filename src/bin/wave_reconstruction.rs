use candle_core::{DType, Device, Shape, Tensor};

use rand::rngs::StdRng;
use rand_distr::{DistIter, Distribution, Uniform};

use candle_nn::linear;
use candle_nn::loss::cross_entropy;
use candle_nn::rnn::{lstm, LSTMConfig, LSTMState, LSTM};

use candle_nn::{Module, VarBuilder, VarMap};

use candle_nn::rnn::RNN;

use std::ops::Range;

fn wave_fn(omega: f32, x: f32) -> f32 {
    (omega * x).sin()
}

struct WaveBatch {
    x_tensor: ndarray::Array2<f32>,
    y_tensor: ndarray::Array2<f32>,
}

fn get_wave_data(
    omega: f32,
    samples: usize,
    batch_size: usize,
    step: f32,
    rng: &mut StdRng,
) -> Result<WaveBatch, ndarray::ShapeError> {
    let uniform = Uniform::new(0f32, 2f32 * std::f32::consts::PI);

    let points_x_vec: Vec<f32> = uniform
        .sample_iter(rng)
        .flat_map(|x_0| {
            (0..samples).map(move |i| {
                let x = x_0 + i as f32 * step;
                x
            })
        })
        .collect();

    let shape = (batch_size, samples);

    let x_tensor = ndarray::Array2::from_shape_vec(shape.clone(), points_x_vec.clone())?;

    let y_tensor = ndarray::Array2::from_shape_vec(
        shape,
        points_x_vec
            .into_iter()
            .map(|x| wave_fn(omega, x))
            .collect::<Vec<_>>(),
    )?;

    Ok(WaveBatch { x_tensor, y_tensor })
}

fn train(
    mut lstm: LSTM,
    omega: f32,
    samples_range: Range<usize>,
    batch_size: usize,
    step: f32,
    n_iter: u32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<LSTM, Box<dyn std::error::Error>> {
    let samples_uniform = Uniform::from(samples_range);

    for _ in 0..n_iter {
        let samples = samples_uniform.sample(rng);
        let WaveBatch { x_tensor, y_tensor } =
            get_wave_data(omega, samples, batch_size, step, rng)?;

        for sample in 0..samples {
            //        x_tensor.get_on_dim(
        }
    }

    Ok(lstm)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;

    let lstm = lstm(4, 4, LSTMConfig::default(), varbuilder.clone())?;

    let linear = linear(4, 4, varbuilder)?;

    Ok(())
}
