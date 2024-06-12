use candle_core::{safetensors::load, DType, Device, Shape, Tensor};

use rand::rngs::StdRng;
use rand_distr::{DistIter, Distribution, Uniform};

use candle_nn::loss::{cross_entropy, mse};
use candle_nn::rnn::{lstm, LSTMConfig, LSTMState, LSTM};
use candle_nn::{linear, Linear};

use candle_nn::Optimizer;
use candle_nn::{AdamW, ParamsAdamW};

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

mod wave_net {
    use super::*;

    pub(super) struct WaveNet {
        pub lstm: LSTM,
        pub fc1: linear::Linear,
        pub fc2: linear::Linear,

        device: Device,
        dtype: DType,
    }

    impl WaveNet {
        pub(super) fn new(
            vb: candle_nn::VarBuilder,
            device: Device,
            dtype: DType,
        ) -> Result<WaveNet, candle_core::Error> {
            let lstm = lstm(1, 32, LSTMConfig::default(), vb.pp("lstm1"))?;
            let fc1 = linear(32, 32, vb.pp("fc1"))?;
            let fc2 = linear(32, 1, vb.pp("fc2"))?;

            Ok(WaveNet {
                lstm,
                fc1,
                fc2,
                device,
                dtype,
            })
        }
    }

    impl RNN for WaveNet {
        type State = Tensor;

        fn zero_state(&self, batch_dim: usize) -> candle_core::Result<Self::State> {
            let tensor = Tensor::zeros(&[batch_dim, 32], self.dtype, &self.device)?;

            Ok(tensor)
        }

        fn step(&self, input: &Tensor, state: &Self::State) -> candle_core::Result<Self::State> {
            let y = candle_nn::ops::leaky_relu(state, 0.01)?;
            let y = self.fc1.forward(&y)?;
            let y = candle_nn::ops::leaky_relu(&y, 0.01)?;
            let y = self.fc2.forward(&y)?;

            Ok(y)
        }

        fn states_to_tensor(&self, states: &[Self::State]) -> candle_core::Result<Tensor> {
            todo!()
            //        self.lstm.states_to_tensor(states)
        }
    }
    //
    //impl Module for WaveNet {
    //    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
    //        let states = self.lstm.seq(&xs)?;
    //
    //        let last_state = states
    //            .last()
    //            .ok_or(std::io::Error::new(std::io::ErrorKind::Other, "no states"))?;
    //
    //        let y = last_state.h().clone();
    //        let y = candle_nn::ops::leaky_relu(&y, 0.01)?;
    //        let y = self.fc1.forward(&y)?;
    //        let y = candle_nn::ops::leaky_relu(&y, 0.01)?;
    //        let y = self.fc2.forward(&y)?;
    //
    //        Ok(y)
    //    }
    //}
}

use wave_net::*;

fn train(
    mut wave_net: WaveNet,
    omega: f32,
    samples_range: Range<usize>,
    batch_size: usize,
    step: f32,
    n_iter: u32,
    mut optimizer: impl Optimizer,
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

        let pred = wave_net.states_to_tensor(&wave_net.seq(&x_dev_tensor)?)?;

        assert_eq!(pred.shape(), &Shape::from_dims(&[batch_size, 1]));

        let loss = mse(&pred, &y_last_dev_tensor)?;

        optimizer.backward_step(&loss)?;

        println!("loss: {:?}", loss);
    }

    Ok(wave_net)
}

fn plot_predictions(
    mut wave_net: &WaveNet,
    omega: f32,
    rng: &mut StdRng,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let wave = get_wave_data(omega, 100, 1, 0.01, rng)?;

    todo!()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    println!("device: {:?}", device);

    let varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let wave_net = WaveNet::new(varbuilder, device.clone(), DType::F32)?;

    let mut rng = StdRng::seed_from_u64(0);

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )?;

    let wave_net = train(
        wave_net,
        1.,
        10..100,
        1000,
        0.01,
        10000,
        opt,
        &mut rng,
        &device,
    )?;

    Ok(())
}
