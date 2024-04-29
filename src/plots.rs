use plotters::coord::Shift;
use plotters::prelude::*;

pub fn plot_error_matrix<DB>(
    expected: &[u8],
    predictions: &[u8],
    cat_cnt: usize,
    category_formatter: &impl Fn(usize) -> String,
    drawing_area: &DrawingArea<DB, Shift>,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    use plotters::prelude::*;

    drawing_area.fill(&WHITE)?;

    let mut left = ChartBuilder::on(&drawing_area);

    let mut chart_context_left = left
        .set_all_label_area_size(50)
        .margin(50)
        .build_cartesian_2d(0..cat_cnt - 1, 0..cat_cnt - 1)?;

    let label_formatter = |idx: &usize| category_formatter(*idx);
    chart_context_left
        .configure_mesh()
        .x_labels(cat_cnt)
        .x_label_formatter(&|x| format!("{:.0}", x))
        .x_desc("Predicted")
        .x_label_formatter(&label_formatter)
        .y_labels(cat_cnt)
        .y_desc("Label")
        .y_label_formatter(&label_formatter)
        .draw()?;

    let len = expected.len() as f32;
    let mut matrix = vec![vec![0f32; cat_cnt]; cat_cnt];
    expected
        .iter()
        .zip(predictions.iter())
        .for_each(|(&l, &p)| {
            matrix[l as usize][p as usize] += 1.0;
        });

    matrix.iter_mut().for_each(|row| {
        row.iter_mut().for_each(|x| *x /= len);
    });

    chart_context_left.draw_series(matrix.iter().enumerate().flat_map(|(label_id, row)| {
        row.iter().enumerate().map(move |(predicted_id, &v)| {
            Rectangle::new(
                [
                    (label_id, predicted_id),
                    ((label_id + 1), (predicted_id + 1)),
                ],
                BLACK.mix(v.into()).filled(),
            )
        })
    }))?;

    Ok(())
}
