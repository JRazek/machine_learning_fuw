#![feature(iter_array_chunks)]
#![feature(ascii_char)]
#![feature(generic_const_exprs)]

use dfdx::prelude::*;

use rand::prelude::*;

use plotters::prelude::*;

#[derive(Default, Clone, Copy, Debug, Sequential)]
pub struct Encoder {
    conv1: Conv2DConstConfig<1, 16, 3>, //256x256 -> 254x254
    conv_tanh1: Tanh,

    conv2: Conv2DConstConfig<16, 32, 3>, //254x254 -> 252x252
    conv_tanh2: Tanh,

    conv3: Conv2DConstConfig<32, 64, 5>, //252x252 -> 248x248
    conv_tanh3: Tanh,

    conv4: Conv2DConstConfig<64, 64, 5, 3>, //248x248 -> 82x82
    conv_tanh4: Tanh,

    conv5: Conv2DConstConfig<64, 64, 5, 3>, //82x82 -> 26x26
    conv_tanh5: Tanh,

    conv6: Conv2DConstConfig<64, 64, 5, 3>, //26x26 -> 8x8
    conv_tanh6: Tanh,

    conv7: Conv2DConstConfig<64, 1, 1>, //26x26 -> 8x8
}

#[derive(Default, Clone, Copy, Debug, Sequential)]
pub struct Decoder {
    trans_conv7: ConvTrans2DConstConfig<1, 64, 1>, //8x8 -> 26x26

    conv_tanh6: Tanh,
    trans_conv6: ConvTrans2DConstConfig<64, 64, 5, 3>, //26x26 -> 82x82

    conv_taanh5: Tanh,
    trans_conv5: ConvTrans2DConstConfig<64, 64, 5, 3>, //82x82 -> 248x248

    conv_tanh4: Tanh,
    trans_conv4: ConvTrans2DConstConfig<64, 64, 5, 3>,

    conv_tanh3: Tanh,
    trans_conv3: ConvTrans2DConstConfig<64, 32, 5>,

    conv_tanh2: Tanh,
    trans_conv2: ConvTrans2DConstConfig<32, 16, 3>,

    trans_conv_1: ConvTrans2DConstConfig<16, 1, 3>,
}

#[derive(Default, Clone, Copy, Debug, Sequential)]
struct EncoderDecoder {
    encoder: Encoder,
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

            if distance_sq < n_radius.pow(2) {
                disk[[i, j]] = 1.0;
            }
        }
    }

    disk
}

fn disk_generator<const N: usize, const M: usize>(
    seed: u64,
) -> impl Iterator<Item = ndarray::Array2<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let max_r = std::cmp::min(N, M) as u32 / 2;

    use rand_distr::Uniform;

    assert!(max_r > 2);

    let r_dist = Uniform::new(2, max_r);

    (0..).map(move |_| {
        let r = r_dist.sample(&mut rng);

        let x = rng.gen_range(0..((M as u32 - 2 * r) / 2)) + M as u32 / 2;
        let y = rng.gen_range(0..((N as u32 - 2 * r) / 2)) + N as u32 / 2;

        assert!(x + r < M as u32);
        assert!(y + r < N as u32);

        assert!(x >= r);
        assert!(y >= r);

        generate_disk::<N, M>((x, y), r)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 256;
    const M: usize = 256;

    let dev = AutoDevice::default();

    let mut encoder_decoder = dev.build_module::<f32>(EncoderDecoder::default());

    match encoder_decoder.load_safetensors("models/encoder_decoder.pt") {
        Ok(_) => println!("Model loaded successfully"),
        Err(e) => {
            println!("Error loading model: {:?}", e);
            println!("proceeding with random initialization");
        }
    }

    const BATCH: usize = 1;
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
            (&mut generator).take(BATCH).flatten().collect(),
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