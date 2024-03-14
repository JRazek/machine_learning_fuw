use dfdx::prelude::*;
use rand_distr::Distribution;

fn generate_dataset<const N: usize, D>(dev: D) -> Tensor<Rank2<N, 2>, f32, D, NoneTape>
where
    D: Device<f32>,
{
    let rng = rand::thread_rng();
    let dist = rand_distr::Uniform::new(0., 1.).sample_iter(rng);
    let data: Vec<f32> = dist
        .take(N)
        .map(|x| [x, if x < 0.5 { 0. } else { 1. }])
        .flatten()
        .collect();

    dev.tensor_from_vec(data, Rank2::<N, 2>::default())
}

fn draw_dataset<const N: usize, D>(tensor: Tensor<Rank2<N, 2>, f32, D>)
where
    D: Device<f32>,
{
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let dataset = generate_dataset::<1000, _>(dev);

    println!("dataset: {:?}", dataset.array());

    Ok(())
}
