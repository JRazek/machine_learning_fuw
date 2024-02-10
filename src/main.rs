mod reg_lin;

use plotters::prelude::*;

fn main() {
    let drawing_area = SVGBackend::new("./lin_reg.svg", (800, 600)).into_drawing_area();

    let (left, right) = drawing_area.split_horizontally(400);

    reg_lin::run(ChartBuilder::on(&left), ChartBuilder::on(&right));

    drawing_area.present().unwrap();
}
