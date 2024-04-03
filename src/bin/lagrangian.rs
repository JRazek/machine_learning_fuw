use dfdx::prelude::*;

#[derive(Default, Clone, Sequential)]
struct Model {
    input: LinearConstConfig<2, 4>,
    relu1: ReLU,
    hidden: LinearConstConfig<4, 4>,
    tanh: Tanh,
    output: LinearConstConfig<4, 1>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = AutoDevice::default();

    let model = dev.build_module::<f32>(Model::default());

    let input: Tensor<Rank1<2>, f32, _, _> = dev.sample_normal();

    let output: Tensor<Rank0, f32, _, _> = model
        .forward(input.clone().retaped::<OwnedTape<_, _>>())
        .reshape();

    let grad = output.backward();

    let input_grad = grad.get(&input);

    println!("Input grads: {:?}", input_grad);

    Ok(())
}
