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
    x_tensor: ndarray::Array3<f32>,
    y_tensor: ndarray::Array3<f32>,
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

    let shape = (batch_size, samples, 1);

    let x_tensor = ndarray::Array3::from_shape_vec(shape.clone(), points_x_vec.clone())?;

    let y_tensor = ndarray::Array3::from_shape_vec(
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

    pub fn input_candle_tensor_from_ndarray(
        x: ndarray::ArrayView3<'_, f32>,
        device: &Device,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        match *x.shape() {
            [batch_size, samples, 1] => {
                let candle_input = Tensor::from_iter(x.iter().cloned(), device)?
                    .reshape(Shape::from_dims(&[batch_size, samples, 1]))?;

                Ok(candle_input)
            }
            _ => Err("Invalid shape".into()),
        }
    }

    pub fn output_candle_tensor_to_ndarray(
        y: Tensor,
    ) -> Result<ndarray::Array3<f32>, Box<dyn std::error::Error>> {
        match y.dims3()? {
            (batch_size, samples, 1) => {
                let y = ndarray::Array3::from_shape_vec(
                    (batch_size, samples, 1),
                    y.flatten_all()?.to_vec1()?,
                )?;

                Ok(y)
            }
            _ => Err("Invalid shape".into()),
        }
    }

    pub(super) struct WaveNet {
        pub fc1: linear::Linear,
        pub lstm1: LSTM,
        pub lstm2: LSTM,

        device: Device,
        dtype: DType,
    }

    impl WaveNet {
        pub(super) fn new(
            vb: candle_nn::VarBuilder,
            device: Device,
            dtype: DType,
        ) -> Result<WaveNet, candle_core::Error> {
            let lstm1 = lstm(1, 32, LSTMConfig::default(), vb.pp("lstm1"))?;
            let lstm2 = lstm(32, 32, LSTMConfig::default(), vb.pp("lstm2"))?;

            let fc1 = linear(32, 1, vb.pp("fc1"))?;

            Ok(WaveNet {
                lstm1,
                lstm2,
                fc1,
                device,
                dtype,
            })
        }

        //expects input of shape (batch_size, samples, 1).
        //It will apply seq for each sample, to the rnn part with initial 0 state and then apply the linear layer to all states.
        //output will be of size (batch_size, samples, 1). It will be the prediction for each sample.
        pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
            let lstm_states = self.seq(x)?;

            //each of these is of size (batch_size, 32)
            let lstm2_states = lstm_states
                .into_iter()
                .map(|state| state.lstm2_state.h().clone())
                .collect::<Vec<_>>();

            //concat along samples dimension
            let y = Tensor::stack(&lstm2_states, 1)?;

            let y = self.fc1.forward(&y)?;

            Ok(y)
        }

        pub(super) fn forward_states(
            &self,
            states: &[WaveNetState],
        ) -> Result<Tensor, candle_core::Error> {
            //(batch_size, states.len(), 32)
            let input = self.states_to_tensor(states)?;

            let y = self.fc1.forward(&input)?;

            Ok(y)
        }
    }

    #[derive(Clone, Debug)]
    pub(super) struct WaveNetState {
        lstm1_state: LSTMState,
        lstm2_state: LSTMState,
    }

    impl RNN for WaveNet {
        type State = WaveNetState;

        fn zero_state(&self, batch_dim: usize) -> candle_core::Result<Self::State> {
            let lstm1_state = self.lstm1.zero_state(batch_dim)?;
            let lstm2_state = self.lstm2.zero_state(batch_dim)?;

            Ok(WaveNetState {
                lstm1_state,
                lstm2_state,
            })
        }

        //takes (batch_size, features=1) tensor.
        fn step(
            &self,
            input: &Tensor,
            WaveNetState {
                lstm1_state,
                lstm2_state,
            }: &Self::State,
        ) -> candle_core::Result<Self::State> {
            assert_eq!(input.rank(), 2);
            let lstm1_state_out = self.lstm1.step(input, lstm1_state)?;
            let lstm2_state_out = self.lstm2.step(lstm1_state_out.h(), lstm2_state)?;

            let wave_net_state = WaveNetState {
                lstm1_state: lstm1_state_out,
                lstm2_state: lstm2_state_out,
            };

            Ok(wave_net_state)
        }

        fn states_to_tensor(&self, states: &[Self::State]) -> candle_core::Result<Tensor> {
            let states_vec: Vec<_> = states
                .iter()
                .map(|state| state.lstm2_state.h().clone())
                .collect();

            let y = Tensor::stack(&states_vec, 1)?;

            Ok(y)
        }
    }
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
        assert_eq!(x_tensor.shape(), &[batch_size, samples, 1]);

        let x_dev_tensor = input_candle_tensor_from_ndarray(x_tensor.view(), device)?;

        let y_dev_tensor = Tensor::from_vec(
            y_tensor.into_raw_vec(),
            Shape::from_dims(&[batch_size, samples, 1]),
            device,
        )?;

        //(batch_size, samples, 1) all predictions.
        let pred = wave_net.forward(&x_dev_tensor)?;

        let loss = mse(&pred, &y_dev_tensor)?;

        optimizer.backward_step(&loss)?;

        println!("loss: {:?}", loss);
    }

    Ok(wave_net)
}

fn plot_predictions(
    mut wave_net: &WaveNet,
    omega: f32,
    model_input_samples: usize,
    step: f32,
    plot_range: Range<f32>,
    rng: &mut StdRng,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    use uczenie_maszynowe_fuw::plots::*;

    let svg = SVGBackend::new("plots/wave_reconstruction.svg", (1024, 768)).into_drawing_area();

    let total_phase = plot_range.end - plot_range.start;
    let n_samples = (total_phase / omega / step).ceil() as usize;

    let wave_data = get_wave_data(omega, n_samples, 1, step, rng)?;
    let input_x = wave_data.x_tensor;
    let y_target_output = wave_data.y_tensor;

    if model_input_samples > input_x.len_of(ndarray::Axis(1)) {
        panic!("model_input_samples > input_x.len_of(ndarray::Axis(1))");
    }

    let model_initial_input_x = input_x.slice(ndarray::s![.., ..model_input_samples, ..]);

    //starts with the last state from initial predictions.
    //states of shape (batch_size, 32)
    let mut wave_net_states = wave_net
        .seq(&input_candle_tensor_from_ndarray(
            model_initial_input_x.view(),
            device,
        )?)?
        .into_iter()
        .rev()
        .take(1)
        .collect::<Vec<_>>();

    for i in model_input_samples + 1..n_samples {
        let wave_net_state = wave_net_states.last().unwrap();

        let x = input_x.select(ndarray::Axis(1), &[i]);
        let x_dev = input_candle_tensor_from_ndarray(x.view(), device)?.get_on_dim(2, 0)?;

        //        println!("x_dev: {:?}", x_dev.shape());
        //        println!("wave_net_state: {:?}", wave_net_state);
        let new_state = wave_net.step(&x_dev, wave_net_state)?;

        wave_net_states.push(new_state);
    }

    let y_predictions_dev_tensor = wave_net.forward_states(&wave_net_states)?;

    let y_predictions = output_candle_tensor_to_ndarray(y_predictions_dev_tensor)?;

    let y_output_predicted = ndarray::concatenate(
        ndarray::Axis(1),
        &[
            y_target_output
                .select(
                    ndarray::Axis(1),
                    (0..model_input_samples).collect::<Vec<_>>().as_slice(),
                )
                .view(),
            y_predictions.view(),
        ],
    )?;

    //    println!("y_output_predicted: {:?}", y_output_predicted);

    let (up, down) = svg.split_vertically(389);

    let MinMax { min, max } =
        uczenie_maszynowe_fuw::plots::find_max_min(input_x.iter().cloned()).unwrap();

    let plot_range = min..max;

    let y_range = -1f32..1f32;
    let first_extrapolated_x = input_x[[0, model_input_samples, 0]];
    let vertical_line_extrapolated = || {
        PathElement::new(
            vec![
                (first_extrapolated_x, y_range.clone().start),
                (first_extrapolated_x, y_range.end),
            ],
            RED.filled(),
        )
    };

    let mut chart = plot_chart(
        input_x.iter().cloned().zip(y_target_output.iter().cloned()),
        "ground truth",
        plot_range.clone(),
        y_range.clone(),
        &up,
    )?;

    //draw vertical line at the center
    chart.draw_series(std::iter::once(vertical_line_extrapolated()))?;

    let mut chart = plot_chart(
        input_x
            .iter()
            .cloned()
            .zip(y_output_predicted.iter().cloned()),
        "extrapolated (after red line)",
        plot_range,
        y_range.clone(),
        &down,
    )?;

    chart.draw_series(std::iter::once(vertical_line_extrapolated()))?;

    svg.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;
    println!("device: {:?}", device);

    let mut varmap = VarMap::new();

    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let wave_net = WaveNet::new(varbuilder, device.clone(), DType::F32)?;

    match varmap.load("models/wave_reconstruction_model") {
        Ok(_) => {
            println!("model loaded");
        }
        Err(_) => {
            println!("model not loaded");
        }
    }

    let mut rng = StdRng::seed_from_u64(20);

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )?;

    const OMEGA: f32 = 1.;

    let wave_net = train(
        wave_net,
        OMEGA,
        1..100,
        100,
        0.001,
        100,
        opt,
        &mut rng,
        &device,
    )?;

    plot_predictions(
        &wave_net,
        OMEGA,
        100,
        0.01 * std::f32::consts::PI,
        0f32..2f32 * std::f32::consts::PI,
        &mut rng,
        &device,
    )?;

    varmap.save("models/wave_reconstruction_model")?;
    println!("model saved");

    Ok(())
}
