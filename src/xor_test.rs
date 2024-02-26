use dfdx::nn::LinearConstConfig;
use dfdx::nn::Sigmoid;
use dfdx::tensor::AutoDevice;

use dfdx::prelude::*;

#[derive(Clone, Sequential, Default)]
struct XORModule {
    l1: LinearConstConfig<2, 3>,
    a1: Sigmoid,
    l2: LinearConstConfig<3, 1>,
    a2: Sigmoid,
}

pub fn log_reg() {
    let dev = AutoDevice::default();

    let mut reg = dev.build_module::<f32>(XORModule::default());

    let mut sgd = Sgd::new(
        &reg,
        SgdConfig {
            lr: 1e-1,
            momentum: None,
            weight_decay: None,
        },
    );

    let mut grads = reg.alloc_grads();

    let inputs = [
        (true, true, false),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ];

    for (i1, i2, o) in inputs.repeat(100) {
        let input: Tensor<Rank1<2>, f32, _> = dev.tensor([i1 as u32 as f32, i2 as u32 as f32]);

        let pred = reg.forward(input.traced(grads));

        let target = dev.tensor([o as u32 as f32]);

        println!(
            "predicted: {:?}, target: {:?}",
            pred.array(),
            target.array(),
        );

        let loss = mse_loss(pred, target);

        println!("loss: {}", loss.array());

        grads = loss.backward();

        sgd.update(&mut reg, &grads).unwrap();
    }
}
