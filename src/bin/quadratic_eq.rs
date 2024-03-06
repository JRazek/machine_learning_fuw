use dfdx::prelude::*;
use plotters::drawing::IntoDrawingArea;

use plotters::backend::SVGBackend;
use plotters::chart::ChartBuilder;
use plotters::element::Circle;
use plotters::series::PointSeries;

use plotters::style::*;

use rand_distr::Uniform;

fn quadratic_eq_gen<const N: usize, D>(dev: &mut D) -> Tensor<Rank2<N, 3>, f32, D>
where
    D: Device<f32>,
{
    let tensor: Tensor<Rank2<N, 3>, f32, D> = dev.sample(Uniform::new(-1.0, 1.0));

    tensor
}

fn quadratic_eq_solve([a, b, c]: &[f32; 3]) -> Option<(f32, f32)> {
    let discriminant_sqrt = (b.powi(2) - 4.0 * a * c).sqrt();

    if discriminant_sqrt > 0.0 {
        let x1 = (-b + discriminant_sqrt) / (2.0 * a);
        let x2 = (-b - discriminant_sqrt) / (2.0 * a);

        Some((x1, x2))
    } else {
        None
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dev = AutoDevice::default();

    let tensor: Tensor<Rank2<100, 3>, _, _> = quadratic_eq_gen(&mut dev);

    let svg_backend =
        SVGBackend::new("plots/quadratic_eq_solutions.svg", (1024, 768)).into_drawing_area();

    let mut chart_builder = ChartBuilder::on(&svg_backend);

    let mut chart_context = chart_builder
        .caption("Quadratic equation solutions", ("Arial", 20))
        .build_cartesian_3d(-1.0f32..1.0, -1.0f32..1.0, -1.0f32..1.0)?;

    chart_context
        .configure_axes()
        .x_labels(6)
        .y_labels(6)
        .z_labels(6)
        .draw()?;

    let gen = |cord: [f32; 3], size, style| {
        let style = if let Some(_) = quadratic_eq_solve(&cord) {
            YELLOW.filled()
        } else {
            style
        };

        let cord = (cord[0], cord[1], cord[2]);

        Circle::new(cord, size, style)
    };

    let point_series =
        PointSeries::of_element(tensor.array().into_iter(), 2., BLACK.filled(), &gen);

    chart_context.draw_series(point_series)?;

    svg_backend.present()?;

    Ok(())
}
