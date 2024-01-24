mod reg_lin;

use plotters::prelude::*;

fn main() {
    let drawing_area = SVGBackend::new("./lin_reg.svg", (800, 600)).into_drawing_area();

    reg_lin::run(ChartBuilder::on(&drawing_area));

    drawing_area.present().unwrap();
}
