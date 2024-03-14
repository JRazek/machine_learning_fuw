#![feature(generic_const_exprs)]

use dfdx::prelude::*;

#[derive(Clone, Sequential, Default)]
struct Model {
    c1: Conv2DConstConfig<1, 3, 5>,
    r1: ReLU,
}

fn main() {
    let dev = AutoDevice::default();

    let module = dev.build_module::<f32>(Model::default());

    let x = 2;
}
