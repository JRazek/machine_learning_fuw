use candle_core::{DType, Device, Tensor};

use rand::rngs::StdRng;
use rand_distr::{DistIter, Distribution, Uniform};

use candle_nn::linear;
use candle_nn::loss::cross_entropy;
use candle_nn::rnn::{lstm, LSTMConfig, LSTMState, LSTM};

use candle_nn::{Module, VarBuilder, VarMap};

use candle_nn::rnn::RNN;

fn wave_fn(omega: f32, x: f32) -> f32 {
    (omega * x).sin()
}

fn get_wave_data(
    omega: f32,
    samples: usize,
    batch_size: usize,
    step: f32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    let uniform = Uniform::new(0f32, 2f32 * std::f32::consts::PI);

    let points_iter_batches = uniform
        .sample_iter(rng)
        .flat_map(|x_0| (0..samples).map(|i| wave_fn(omega, i as f32 * step)));

    Tensor::from_iter(points_iter_batches, device)
}

fn train(
    lstm: LSTM,
    batch_size: usize,
    rng: &mut StdRng,
) -> Result<LSTM, Box<dyn std::error::Error>> {
    todo!()
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
