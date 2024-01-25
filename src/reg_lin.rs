use ndarray::Array;
use plotters::prelude::*;

use rand::distributions::Standard;
use rand::prelude::*;
use rand::rngs::OsRng;

pub fn run<B>(mut chart_builder: ChartBuilder<B>)
where
    B: DrawingBackend,
{
    let theta0: f64 = 1.;
    let theta1: f64 = 3.;

    let points_n = 100;

    let x = Array::linspace(0., 10., points_n);
    let y_modeled: Array<_, _> = x.iter().map(|&x| theta0 + x * theta1).collect();
    let y = y_modeled
        .iter()
        .map(|&y| y + OsRng.sample::<f64, _>(Standard))
        .collect::<Vec<_>>();

    let line_series = LineSeries::new(
        x.iter()
            .cloned()
            .zip(y_modeled.clone())
            .map(|(x, y)| (x, y)),
        BLUE.stroke_width(2),
    )
    .point_size(1);

    let mut chart_context = chart_builder
        .caption("function, lmao", &BLACK)
        .set_all_label_area_size(50)
        .margin(10)
        .build_cartesian_2d(0f64..10f64, 0f64..32f64)
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
        .label("modeled")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

    chart_context
        .draw_series(
            x.iter()
                .cloned()
                .zip(y)
                .map(|(x, y)| Circle::new((x, y), 2, RED.filled())),
        )
        .unwrap()
        .label("data")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    //    chart_context
    //        .draw_series(
    //            Histogram::vertical(&chart_context)
    //                .style(RED.mix(0.5).filled())
    //                .data(y.iter().map(|&x| (x, 1))),
    //        )
    //        .unwrap();

    chart_context
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE)
        .draw()
        .unwrap();
}
