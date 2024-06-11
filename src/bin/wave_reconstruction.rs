use candle_core::{safetensors::load, DType, Device, Shape, Tensor};

use rand::rngs::StdRng;
use rand_distr::{DistIter, Distribution, Uniform};

use candle_nn::loss::{cross_entropy, mse};
use candle_nn::rnn::{lstm, LSTMConfig, LSTMState, LSTM};
use candle_nn::{linear, Linear};

use candle_nn::{Module, VarBuilder, VarMap};

use candle_nn::rnn::RNN;

use std::ops::Range;

use rand::SeedableRng;

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
        .take(batch_size)
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

struct WaveNet {
    pub lstm: LSTM,
    //    fc: linear::Linear,
}

impl Module for WaveNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let states = self.lstm.seq(&xs)?;

        let last_state = states
            .last()
            .ok_or(std::io::Error::new(std::io::ErrorKind::Other, "no states"))?;

        //        let y = self.fc.forward(&last_state.h())?;

        Ok(last_state.h().clone())
    }
}

fn train(
    mut wave_net: WaveNet,
    omega: f32,
    samples_range: Range<usize>,
    batch_size: usize,
    step: f32,
    n_iter: u32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<WaveNet, Box<dyn std::error::Error>> {
    let samples_uniform = Uniform::from(samples_range);

    for i in 0..n_iter {
        println!("iter: {}", i);
        let samples = samples_uniform.sample(rng);
        let WaveBatch { x_tensor, y_tensor } =
            get_wave_data(omega, samples, batch_size, step, rng)?;

        assert_eq!(x_tensor.shape(), y_tensor.shape());
        assert_eq!(x_tensor.shape(), &[batch_size, samples]);

        let x_dev_tensor = Tensor::from_iter(x_tensor.into_iter(), device)?
            .reshape(Shape::from_dims(&[batch_size, samples, 1]))?;

        let y_last_dev_tensor = Tensor::from_iter(
            y_tensor
                .axis_iter(ndarray::Axis(1))
                .last()
                .unwrap()
                .into_iter()
                .cloned(),
            device,
        )?
        .reshape(&Shape::from_dims(&[batch_size, 1]))?;

        assert_eq!(
            y_last_dev_tensor.shape(),
            &Shape::from_dims(&[batch_size, 1])
        );

        let pred = wave_net.forward(&x_dev_tensor)?;

        assert_eq!(pred.shape(), &Shape::from_dims(&[batch_size, 1]));

        let loss = mse(&pred, &y_last_dev_tensor)?;

        println!("loss: {:?}", loss);
    }

    Ok(wave_net)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    println!("device: {:?}", device);

    let varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;

    let lstm = lstm(1, 1, LSTMConfig::default(), varbuilder.clone())?;
    let wave_net = WaveNet { lstm };

    let mut rng = StdRng::seed_from_u64(0);

    let wave_net = train(wave_net, 1., 1..10, 17, 0.1, 10000, &mut rng, &device)?;

    Ok(())
}
