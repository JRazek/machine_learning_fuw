use std::{collections::HashMap, fs::File};

use dfdx::prelude::*;

use polars::prelude::*;

fn label_column<'a>(series: &'a Series) -> PolarsResult<Series> {
    let map: HashMap<_, _> = series
        .str()?
        .into_iter()
        .enumerate()
        .map(|(i, u)| (u, i as u32))
        .rev()
        .collect();

    let labeled = Series::from_iter(series.str()?.into_iter().map(|u| map[&u]));

    Ok(labeled)
}

fn prepare_dataset(file: File) -> PolarsResult<DataFrame> {
    let mut df = CsvReader::new(file).has_header(true).finish()?;

    let labaled_embarks = label_column(df.column("Embarked")?)?;
    df.replace("Embarked", labaled_embarks)?;
    _ = df.drop_in_place("PassengerId")?;
    _ = df.drop_in_place("Name")?;
    _ = df.drop_in_place("Ticket")?;
    _ = df.drop_in_place("Cabin")?;

    Ok(df)
}

fn normalize_f32_column(series: &Float32Chunked) -> PolarsResult<Vec<f32>> {
    let max = series.max().unwrap_or(1f32);

    let normalized = series
        .into_iter()
        .map(|u| u.map(|u| u as f32))
        .map(|f| match f {
            Some(f) => f / max,
            _ => -1f32,
        })
        .collect::<Vec<_>>();

    Ok(normalized)
}

fn encode_tensors<D>(
    dev: &D,
    df: DataFrame,
) -> PolarsResult<(
    Tensor<(usize, Const<4>), f32, D>,
    Tensor<(usize, Const<1>), f32, D>,
)>
where
    D: Device<f32>,
{
    let survived = df
        .column("Survived")?
        .i64()?
        .into_iter()
        .map(|u| match u {
            Some(0) => 0f32,
            Some(1) => 1f32,
            _ => -1f32,
        })
        .collect::<Vec<_>>();

    let pclass = normalize_f32_column(df.column("Pclass")?.cast(&DataType::Float32)?.f32()?)?;

    let sex = df.column("Sex")?.str()?.into_iter().map(|u| match u {
        Some("male") => 0f32,
        Some("female") => 1f32,
        _ => -1f32,
    });

    let sib_sp = normalize_f32_column(df.column("SibSp")?.cast(&DataType::Float32)?.f32()?)?;

    let parch = normalize_f32_column(df.column("Parch")?.cast(&DataType::Float32)?.f32()?)?;

    use itertools::izip;

    let zip: Vec<f32> = izip!(pclass, sex, sib_sp, parch)
        .map(|(pclass, sex, sib_sp, parch)| [pclass, sex, sib_sp, parch])
        .flatten()
        .collect();

    let len = zip.len() / 4;

    let target: Tensor<(usize, Const<1>), f32, D> =
        dev.tensor_from_vec(survived.clone(), (survived.len(), Const::<1>));

    let dataset: Tensor<(usize, Const<4>), f32, D> = dev.tensor_from_vec(zip, (len, Const::<4>));

    Ok((dataset, target))
}

fn train<M, D>(
    mut module: M,
    dataset: Tensor<(usize, Const<4>), f32, D>,
    targets: Tensor<(usize, Const<1>), f32, D>,
) -> Result<(), Box<dyn std::error::Error>>
where
    M: Module<
        Tensor<(usize, Const<4>), f32, D, OwnedTape<f32, D>>,
        Output = Tensor<(usize, Const<1>), f32, D, OwnedTape<f32, D>>,
    >,
    M: UpdateParams<f32, D>,
    D: Device<f32>,
{
    let mut sgd: Sgd<_, f32, D> = Sgd::new(
        &module,
        SgdConfig {
            lr: 1e-3,
            momentum: None,
            weight_decay: Some(WeightDecay::L2(1e-9)),
        },
    );

    const N_ITER: usize = 10000;

    for i in 0..N_ITER {
        let pred = module.forward(dataset.clone().retaped::<OwnedTape<_, _>>());

        let mse = mse_loss(pred, targets.clone());

        print!("Iter: {} Loss: {:?}\r", i, mse.as_vec());

        let grads = mse.backward();

        sgd.update(&mut module, &grads)?;
    }

    Ok(())
}
fn test_model<M, D>(
    module: M,
    test_dataset: Tensor<(usize, Const<4>), f32, D>,
    test_targets: Tensor<(usize, Const<1>), f32, D>,
) -> Result<f32, Box<dyn std::error::Error>>
where
    M: Module<
        Tensor<(usize, Const<4>), f32, D, OwnedTape<f32, D>>,
        Output = Tensor<(usize, Const<1>), f32, D, OwnedTape<f32, D>>,
    >,
    M: UpdateParams<f32, D>,
    D: Device<f32>,
{
    let pred = module.forward(test_dataset.clone().retaped::<OwnedTape<_, _>>());

    let correct_count = pred
        .as_vec()
        .iter()
        .zip(test_targets.as_vec().iter())
        .filter(|(p, t)| {
            let p = if **p > 0.5 { 1f32 } else { 0f32 };
            let t = **t;
            p == t
        })
        .count();

    let accuracy = correct_count as f32 / test_targets.shape().0 as f32;

    Ok(accuracy)
}

#[derive(Clone, Debug, Sequential)]
struct Model {
    linear1: LinearConstConfig<4, 10>,
    sigmoid: Sigmoid,
    linear2: LinearConstConfig<10, 1>,
}

#[allow(unused_variables, unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_file = std::fs::File::open("data/titanic/train.csv")?;
    let df = prepare_dataset(dataset_file)?;

    let dev = AutoDevice::default();

    let (dataset, targets) = encode_tensors(&dev, df)?;

    let model = (
        LinearConstConfig::<4, 10>::default(),
        Sigmoid,
        LinearConstConfig::<10, 1>::default(),
    );

    let module = dev.build_module::<f32>(model);

    train(module.clone(), dataset, targets)?;

    module.save_safetensors("models/titanic")?;

    let test_dataset_file = std::fs::File::open("data/titanic/test.csv")?;

    todo!("different formats");
    let test_df = prepare_dataset(test_dataset_file)?;

    let (test_dataset, test_targets) = encode_tensors(&dev, test_df)?;

    let accuracy = test_model(module, test_dataset, test_targets)?;

    println!("Accuracy: {}", accuracy);

    Ok(())
}
