use plotters::coord::Shift;
use plotters::prelude::*;

pub fn plot_error_matrix<const CAT_CNT: usize, DB>(
    labels: &[u8],
    predictions: &[u8],
    category_mapping: &[char; CAT_CNT],
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
        .build_cartesian_2d(0..CAT_CNT, 0..CAT_CNT)?;

    chart_context_left
        .configure_mesh()
        .x_labels(30)
        .x_label_formatter(&|x| format!("{:.0}", x))
        .x_desc("Predicted")
        .y_labels(20)
        .y_desc("Label")
        .draw()?;

    let len = labels.len() as f32;
    let mut matrix = [[0f32; CAT_CNT]; CAT_CNT];
    labels.iter().zip(predictions.iter()).for_each(|(&l, &p)| {
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
                BLACK.mix(v.into()),
            )
        })
    }))?;

    Ok(())
}
