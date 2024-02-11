mod reg_lin;

use dfdx::prelude::DeviceBuildExt;
use plotters::prelude::*;

use nalgebra::Vector4;
use reg_lin::{normal_equations, stochastic_gradient_descent};

use dfdx::nn::builders::Linear;

use dfdx::tensor::TensorFrom;

fn main() {
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

    {
        let dev = dfdx::tensor::AutoDevice::default();

        let voltage_model = dev.tensor([1., 2., 3., 4.]);

        let r = dev.tensor(1.);

        let model = Linear::<1, 1>;

//        let lin = dev.build_module(model);

        stochastic_gradient_descent(r, voltage_model);
    }
}
