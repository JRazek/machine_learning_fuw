use plotters::coord::Shift;
use plotters::prelude::*;

pub fn plot_error_matrix<DB>(
    expected: &[u8],
    predictions: &[u8],
    cat_cnt: usize,
    category_formatter: &impl Fn(usize) -> String,
    caption: &str,
    drawing_area: &DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    drawing_area.fill(&WHITE)?;

    let mut left = ChartBuilder::on(&drawing_area);

    let mut chart_context_left = left
        .caption(caption, ("Arial", 20))
        .set_all_label_area_size(50)
        .margin(50)
        .build_cartesian_2d(
            (0..cat_cnt - 1).into_segmented(),
            (0..cat_cnt - 1).into_segmented(),
        )?;

    let label_formatter = |idx: &SegmentValue<usize>| match *idx {
        SegmentValue::Exact(v) => category_formatter(v),
        SegmentValue::CenterOf(v) => category_formatter(v),
        SegmentValue::Last => "N/A".to_string(),
    };
    chart_context_left
        .configure_mesh()
        .light_line_style(&WHITE)
        .x_labels(cat_cnt)
        .x_desc("Predicted")
        .x_label_formatter(&label_formatter)
        .y_labels(cat_cnt)
        .y_desc("Expected")
        .y_label_formatter(&label_formatter)
        .draw()?;

    let mut matrix = vec![vec![0f32; cat_cnt]; cat_cnt];
    expected
        .iter()
        .zip(predictions.iter())
        .for_each(|(&l, &p)| {
            matrix[l as usize][p as usize] += 1.0;
        });

    let max = *matrix
        .iter()
        .flatten()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    matrix.iter_mut().for_each(|row| {
        row.iter_mut().for_each(|v| *v /= max);
    });

    chart_context_left.draw_series(matrix.iter().enumerate().flat_map(|(label_id, row)| {
        row.iter().enumerate().map(move |(predicted_id, &v)| {
            Rectangle::new(
                [
                    (
                        SegmentValue::Exact(label_id),
                        SegmentValue::Exact(predicted_id),
                    ),
                    (
                        SegmentValue::Exact(label_id + 1),
                        SegmentValue::Exact(predicted_id + 1),
                    ),
                ],
                BLACK.mix(v as f64).filled(),
            )
        })
    }))?;

    Ok(())
}

pub fn plot_log_scale_data<DB>(
    data: &[f32],
    label: &str,
    drawing_area: &DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    drawing_area.fill(&WHITE)?;

    let mut drawing_area = ChartBuilder::on(&drawing_area);

    let max_loss = *data
        .iter()
        .max_by(|&a, &b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart_context_right = drawing_area
        .caption(label, ("Arial", 20))
        .set_all_label_area_size(70)
        .margin(50)
        .build_cartesian_2d(0..data.len(), (0f32..max_loss).log_scale())?;

    chart_context_right
        .configure_mesh()
        .x_labels(10)
        .x_desc("Iteration")
        .y_labels(10)
        .y_desc(label)
        .y_label_formatter(&|y| format!("{:.1e}", y))
        .draw()?;

    let losses = LineSeries::new(
        data.into_iter().enumerate().map(|(i, &l)| (i, l)),
        BLUE.filled(),
    );

    chart_context_right.draw_series(losses)?;

    Ok(())
}

use ndarray::Array2;

pub fn plot_cartesian2d_points<DB>(
    labels: &[&str],
    data: Array2<f32>,
    label: &str,
    drawing_area: &DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,

    <DB as DrawingBackend>::ErrorType: 'static,
{
    assert_eq!(labels.len(), data.nrows());

    drawing_area.fill(&WHITE)?;

    let mut drawing_area = ChartBuilder::on(&drawing_area);

    let max_x = data
        .column(0)
        .iter()
        .cloned()
        .map(f32::abs)
        .reduce(f32::max)
        .unwrap_or(1.0);

    let max_y = data
        .column(1)
        .iter()
        .cloned()
        .map(f32::abs)
        .reduce(f32::max)
        .unwrap_or(1.0);

    let mut chart_context = drawing_area
        .caption(label, ("Arial", 20))
        .set_all_label_area_size(70)
        .margin(50)
        .build_cartesian_2d(-max_x..max_x, -max_y..max_y)?;

    chart_context
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .draw()?;

    use plotters::style::full_palette;

    const COLORS: [RGBColor; 5] = [
        full_palette::RED,
        full_palette::GREEN,
        full_palette::BLUE,
        full_palette::CYAN,
        full_palette::BLACK,
    ];

    let get_style_i = |i: usize| COLORS[i % COLORS.len()].filled();

    let points = data
        .axis_iter(ndarray::Axis(0))
        .zip(labels.iter())
        .enumerate()
        .map(|(i, (row, &label))| {
            let element = EmptyElement::at((row[0], row[1]))
                + Circle::new((0, 0), 3f32, get_style_i(i))
                + if labels.len() > 100 {
                    Text::new("".to_owned(), (5, -5), ("Arial", 5))
                } else {
                    Text::new(label.to_owned(), (5, -5), ("Arial", 15))
                };

            element
        });

    chart_context.draw_series(points)?;

    Ok(())
}
