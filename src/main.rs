#![feature(array_chunks)]
#![feature(iter_array_chunks)]

mod log_reg;
mod reg_lin;

mod xor_test;

use plotters::prelude::*;

use nalgebra::Vector4;
use reg_lin::{normal_equations, stochastic_gradient_descent};

fn reg_lin() {
    let drawing_area = SVGBackend::new("./lin_reg.svg", (800, 600)).into_drawing_area();

    let (left, right) = drawing_area.split_horizontally(400);

    reg_lin::plot(ChartBuilder::on(&left), ChartBuilder::on(&right));

    drawing_area.present().unwrap();

    {
        let currents = Vector4::new(0., 1., 2., 3.);
        let voltage = Vector4::new(0.1, 2.1, 4.3, 6.4);

        let r = normal_equations(currents, voltage).unwrap();
        println!("fitted R: {r}");
    };

    stochastic_gradient_descent();
}

fn main() {
    log_reg::log_reg();
}
