use std::array::from_fn;

use dfdx::prelude::*;
use rand::rngs::ThreadRng;
use rand_distr::Distribution;
use rand_distr::Uniform;

fn poly(x: f32) -> f32 {
    2. * x.powi(2) * 3. * x - 2.
}

fn generate_dataset<const N: usize, D>(dev: D) -> Tensor<Rank2<N, 2>, f32, D, NoneTape>
where
    D: Device<f32>,
{
    let rng = ThreadRng::default();
    let uniform = Uniform::new(-1., 1.);

    let dataset = uniform
        .sample_iter(rng)
        .take(N)
        .map(|x| {
            let y = poly(x);
            [x, y]
        })
        .flatten()
        .collect::<Vec<_>>();

    let tensor: Tensor<Rank2<N, 2>, _, _, _> =
        dev.tensor_from_vec(dataset, (Const::<N>, Const::<2>));

    tensor
}

fn draw_dataset<const N: usize, D>(tensor: Tensor<Rank2<N, 2>, f32, D>)
where
    D: Device<f32>,
{
}

struct NeuronModel {
//    linear: LinearConstConfig
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let dataset = generate_dataset::<1000, _>(dev);

    println!("dataset: {:?}", dataset.array());

    Ok(())
}
