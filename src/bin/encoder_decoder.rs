#![feature(iter_array_chunks)]
#![feature(ascii_char)]
#![feature(generic_const_exprs)]

use dfdx::prelude::*;

use rand::prelude::*;

use plotters::prelude::*;

#[derive(Default, Clone, Debug, Sequential)]
pub struct Encoder {
    conv1: Conv2DConstConfig<1, 16, 3, 1, 1>, //256x256 -> 256x256
    conv_tanh1: Tanh,

    conv2: Conv2DConstConfig<16, 32, 3, 1, 1>, //256x256 -> 256x256
    conv_tanh2: Tanh,

    conv3: Conv2DConstConfig<32, 32, 5, 1, 2>, //256x256 -> 256x256
    conv_tanh3: Tanh,

    conv4: Conv2DConstConfig<32, 64, 5, 1, 2>, //256x256 -> 256x256
    conv_tanh4: Tanh,

    conv5: Conv2DConstConfig<64, 128, 3, 4, 2>, //256x256 -> 64x64
    conv_tanh5: FastGeLU,

    conv6: Conv2DConstConfig<128, 128, 3, 2, 2>, //64x64 -> 32x32
    conv_tanh6: FastGeLU,

    conv7: Conv2DConstConfig<128, 128, 3, 2, 1>, //32x32 -> 16x16
    conv_tanh7: FastGeLU,

    conv8: Conv2DConstConfig<128, 128, 3, 2, 1>, //16x16 -> 8x8
}

#[derive(Default, Clone, Debug, Sequential)]
pub struct Decoder {
    conv8: ConvTrans2DConstConfig<128, 128, 3, 2, 1>, //8

    conv_tanh8: FastGeLU,
    trans_conv_7: ConvTrans2DConstConfig<128, 128, 4, 2, 1>, //16x16 -> 32x32

    trans_conv_tanh6: FastGeLU,
    trans_conv_6: ConvTrans2DConstConfig<128, 128, 3, 2, 2>, //32x32 -> 64x64

    trans_conv_tanh5: FastGeLU,
    trans_conv_5: ConvTrans2DConstConfig<128, 64, 4, 4, 2>, //64x64 -> 256x256

    trans_conv_tanh4: Tanh,
    trans_conv_4: ConvTrans2DConstConfig<64, 32, 5, 1, 2>, //256x256 -> 256x256

    trans_conv_tanh3: Tanh,
    trans_conv_3: ConvTrans2DConstConfig<32, 32, 5, 1, 2>, //256x256 -> 256x256

    trans_conv_2: Tanh,
    trans_conv_tanh2: ConvTrans2DConstConfig<32, 16, 3, 1, 1>, //256x256 -> 256x256

    trans_conv_1: ConvTrans2DConstConfig<16, 1, 3, 1, 1>,
}

#[derive(Default, Clone, Debug, Sequential)]
struct EncoderDecoder {
    encoder: Encoder,

    tanh: Tanh,

    decoder: Decoder,
}

fn generate_disk<const N: usize, const M: usize>(
    (x, y): (u32, u32),
    n_radius: u32,
) -> ndarray::Array2<f32> {
    let mut disk = ndarray::Array2::<f32>::zeros((N, M));

    let (x, y) = (x as i32, y as i32);
    let n_radius = n_radius as i32;

    for i in 0..N {
        for j in 0..M {
            let distance_sq = (i as i32 - y).pow(2) + (j as i32 - x).pow(2);

            if distance_sq <= n_radius.pow(2) {
                disk[[i, j]] = 1.0;
            }
        }
    }

    disk
}

struct Circle {
    center: (u32, u32),
    radius: u32,
}

fn disk_generator<const N: usize, const M: usize>(
    seed: u64,
) -> impl Iterator<Item = (ndarray::Array2<f32>, Circle)> {
    let mut rng = StdRng::seed_from_u64(seed);

    let max_r = std::cmp::min(N, M) as u32 / 2;

    use rand_distr::Uniform;

    assert!(max_r > 3);

    let r_dist = Uniform::new(3, max_r);

    (0..).map(move |_| {
        let r = r_dist.sample(&mut rng);

        let x = rng.gen_range(0..((M as u32 - 2 * r) / 2)) + M as u32 / 2;
        let y = rng.gen_range(0..((N as u32 - 2 * r) / 2)) + N as u32 / 2;

        assert!(x + r < M as u32);
        assert!(y + r < N as u32);

        assert!(x >= r);
        assert!(y >= r);

        (
            generate_disk::<N, M>((x, y), r),
            Circle {
                center: (x, y),
                radius: r,
            },
        )
    })
}

use plotters::coord::Shift;

pub fn plot_circle<DB, const N: usize, const M: usize>(
    circle: ndarray::ArrayView2<f32>,
    caption: &str,
    drawing_area: &DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    drawing_area.fill(&WHITE)?;
    let mut chart_builder = ChartBuilder::on(&drawing_area);

    let mut cartesian = chart_builder
        .caption(caption, ("Arial", 20))
        .set_all_label_area_size(70)
        .margin(50)
        .build_cartesian_2d(0..M as u32, 0..N as u32)?;

    cartesian
        .configure_mesh()
        .x_labels(10)
        .x_desc("x")
        .y_labels(10)
        .y_desc("y")
        .draw()?;

    cartesian.draw_series(circle.axis_iter(ndarray::Axis(0)).enumerate().flat_map(
        |(i, row)| {
            row.iter()
                .enumerate()
                .map(move |(j, &v)| {
                    Rectangle::new(
                        [(j as u32, i as u32), (j as u32 + 1, i as u32 + 1)],
                        BLACK.mix(v.into()).filled(),
                    )
                })
                .collect::<Vec<_>>()
        },
    ))?;

    Ok(())
}

const N: usize = 256;
const M: usize = 256;

fn train_encoder_decoder() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let mut encoder_decoder = dev.build_module::<f32>(EncoderDecoder::default());

    match encoder_decoder.load_safetensors("models/encoder_decoder.pt") {
        Ok(_) => println!("Model loaded successfully"),
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("proceeding with random initialization");
        }
    }

    const BATCH: usize = 2;
    let mut generator = disk_generator::<N, M>(0);

    let mut sgd = Sgd::new(
        &encoder_decoder,
        SgdConfig {
            momentum: None,
            lr: 1e-3,
            weight_decay: Some(WeightDecay::L2(1e-2)),
        },
    );

    let mut losses = Vec::new();
    const N_EPOCHS: usize = 1000000;
    for epoch in 0..N_EPOCHS {
        let disks = ndarray::Array3::from_shape_vec(
            (BATCH, N, M),
            (&mut generator)
                .map(|(img, _)| img)
                .take(BATCH)
                .flatten()
                .collect(),
        )?;

        let disks_tensor: Tensor<Rank4<BATCH, 1, N, M>, _, _> =
            dev.tensor_from_vec(disks.clone().into_raw_vec(), Rank4::default());

        let output: Tensor<Rank4<BATCH, 1, N, M>, _, _, OwnedTape<_, _>> =
            encoder_decoder.forward(disks_tensor.retaped());

        let output_none_tape = output.retaped::<NoneTape>();

        let loss = cross_entropy_with_logits_loss(output, disks_tensor);

        let loss_val = loss.as_vec()[0];

        println!("epoch: {}, loss: {:+e}", epoch, loss_val);

        let grads = loss.backward();

        sgd.update(&mut encoder_decoder, &grads)?;

        dbg!(loss_val);

        losses.push(loss_val);

        if (epoch + 1) % 100 == 0 {
            encoder_decoder.save_safetensors("models/encoder_decoder.pt")?;

            let output_array =
                ndarray::Array4::from_shape_vec((BATCH, 1, N, M), output_none_tape.as_vec())
                    .unwrap()
                    .index_axis_move(ndarray::Axis(0), 0)
                    .index_axis_move(ndarray::Axis(0), 0)
                    .into_shape((N, M))?; //acts as an assert

            let svg = SVGBackend::new("plots/disk_plot.svg", (1800, 600)).into_drawing_area();
            let (reference, right) = svg.split_horizontally(600);
            let (generated, loss_plot) = right.split_horizontally(600);

            plot_circle::<_, N, M>(disks.index_axis(ndarray::Axis(0), 0), "disk", &reference)?;
            plot_circle::<_, N, M>(output_array.view(), "output", &generated)?;
            uczenie_maszynowe_fuw::plots::plot_log_scale_data(&losses, "loss", &loss_plot)?;
            svg.present()?;
        }
    }

    Ok(())
}

#[derive(Default, Clone, Debug, Sequential)]
struct CircleGenerator<const BATCH: usize> {
    linear1: LinearConstConfig<3, 256>,
    tanh1: Tanh,

    linear3: LinearConstConfig<256, 256>,
    tanh3: Tanh,

    linear4: LinearConstConfig<256, 512>,
    tanh4: Tanh,

    linear5: LinearConstConfig<512, 1024>,
    tanh5: Tanh,

    linear6: LinearConstConfig<1024, 65536>,

    reshape: Reshape<Rank4<BATCH, 1, 256, 256>>,
}

fn train_fully_connected_generator() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let mut network = dev.build_module::<f32>(CircleGenerator::default());

    match network.load_safetensors("models/circle_generator_fully_connected.pt") {
        Ok(_) => println!("Model loaded successfully"),
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("proceeding with random initialization");
        }
    }

    const BATCH: usize = 20;
    let mut generator = disk_generator::<N, M>(0);

    let mut sgd = Sgd::new(
        &network,
        SgdConfig {
            momentum: None,
            lr: 1e-3,
            weight_decay: Some(WeightDecay::L2(1e-2)),
        },
    );

    let mut losses = Vec::new();
    const N_EPOCHS: usize = 1000000;
    for epoch in 0..N_EPOCHS {
        let batch = (&mut generator).take(BATCH).collect::<Vec<_>>();

        let input: Vec<_> = batch
            .iter()
            .flat_map(
                |(
                    _,
                    Circle {
                        center: (x, y),
                        radius,
                    },
                )| [*x as f32, *y as f32, *radius as f32],
            )
            .collect();

        let input_dev: Tensor<Rank2<BATCH, 3>, _, _> = dev.tensor_from_vec(input, Rank2::default());

        let target_host = ndarray::Array4::from_shape_vec(
            (BATCH, 1, N, M),
            batch
                .into_iter()
                .flat_map(|(disk, _)| disk)
                .collect::<Vec<_>>(),
        )?;

        let target_dev: Tensor<Rank4<BATCH, 1, N, M>, _, _> =
            dev.tensor_from_vec(target_host.clone().into_raw_vec(), Rank4::default());

        let output: Tensor<Rank4<BATCH, 1, N, M>, _, _, OwnedTape<_, _>> =
            network.forward(input_dev.retaped());

        let output_none_tape = output.retaped::<NoneTape>();

        let loss = cross_entropy_with_logits_loss(output, target_dev);

        let loss_val = loss.as_vec()[0];

        dbg!(loss_val);

        let grads = loss.backward();

        sgd.update(&mut network, &grads)?;

        losses.push(loss_val);

        if (epoch + 1) % 100 == 0 {
            if (epoch + 1) % 1000 == 0 {
                network.save_safetensors("models/circle_generator_fully_connected.pt")?;
            }

            let input_array = target_host
                .index_axis_move(ndarray::Axis(0), 0)
                .index_axis_move(ndarray::Axis(0), 0)
                .into_shape((N, M))?; //acts as an assert

            let output_array =
                ndarray::Array4::from_shape_vec((BATCH, 1, N, M), output_none_tape.as_vec())
                    .unwrap()
                    .index_axis_move(ndarray::Axis(0), 0)
                    .index_axis_move(ndarray::Axis(0), 0)
                    .into_shape((N, M))?; //acts as an assert

            let svg =
                SVGBackend::new("plots/disk_plot_generated.svg", (1800, 600)).into_drawing_area();
            let (reference, right) = svg.split_horizontally(600);
            let (generated, loss_plot) = right.split_horizontally(600);

            plot_circle::<_, N, M>(input_array.view(), "disk", &reference)?;
            plot_circle::<_, N, M>(output_array.view(), "output", &generated)?;
            uczenie_maszynowe_fuw::plots::plot_log_scale_data(&losses, "loss", &loss_plot)?;
            svg.present()?;
        }
    }

    Ok(())
}

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Params {
    #[arg(long, default_value = "train", required = true)]
    mode: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = Params::parse();

    match params.mode.as_str() {
        "fully_connected" => train_fully_connected_generator()?,
        "encoder_decoder" => train_encoder_decoder()?,
        _ => panic!("Unknown mode"),
    };

    Ok(())
}
