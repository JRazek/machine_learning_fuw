use ndarray::Array;
use plotters::prelude::*;

pub fn run<B: DrawingBackend>(mut chart_builder: ChartBuilder<B>) {
    let theta0: f64 = 1.;
    let theta1: f64 = 3.;

    let points_n = 100;

    let x = Array::linspace(0., 100., points_n);
    let y: Array<_, _> = x.iter().map(|&x| theta0 + x * theta1).collect();

    let line_series = LineSeries::new(
        x.iter().cloned().zip(y.clone()).map(|(x, y)| (x, y)),
        &GREEN,
    );

    let mut chart_context = chart_builder
        .caption("function, lmao", &BLACK)
        .set_all_label_area_size(50)
        .margin(10)
        .build_cartesian_2d(0f64..200f64, 0f64..500f64)
        .unwrap();

    chart_context
        .configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .draw()
        .unwrap();

    chart_context
        .draw_series(line_series)
        .unwrap()
        .label("function lol")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    chart_context
        .draw_series(
            x.iter()
                .cloned()
                .zip(y)
                .map(|(x, y)| Pixel::new((x, y), RED)),
        )
        .unwrap()
        .label("function lol2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart_context
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE)
        .draw()
        .unwrap();
}
