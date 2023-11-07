use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};

#[macro_use]
extern crate log;

mod dataset;
mod vocab_builder;
mod word2vec;

use crate::dataset::DatasetOptions;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    run().expect("Failed to run");
    // run2();
}

fn run() -> Result<()> {
    let vocab = vocab_builder::Vocab::from_files().expect("Failed to build vocab");
    let n = vocab.n();
    let dataset = DatasetOptions::new()
        .batch_size(100)
        .epochs(1000)
        .vocab(vocab)
        .build()
        .expect("Failed to build dataset");

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &word2vec::DEVICE);
    const EMBEDDING_SIZE: usize = 100;
    const LEARNING_RATE: f64 = 0.05;
    let nn = word2vec::Word2VecNN::new(n, EMBEDDING_SIZE, vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    for (i, epoch) in dataset.epochs.into_iter().enumerate() {
        println!("epoch {i}");

        let n_batches = epoch.batches.len();
        let mut avg_loss = 0.0;

        for batch in epoch.batches.into_iter() {
            let x = batch.x;
            let y = batch.y;
            // dbg!(x.shape());
            // dbg!(y.shape());
            let logits = nn.forward(&x)?;
            // dbg!(logits.shape());
            // dbg!(logits.to_vec2::<f32>()?);
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            // dbg!(y.to_vec1::<u32>()?);
            let loss = loss::nll(&log_sm, &y)?;
            let loss_scalar = loss.mean_all()?.to_scalar::<f32>()?;
            // println!("loss: {loss_scalar}");
            sgd.backward_step(&loss)?;

            avg_loss += loss_scalar;
        }

        println!("avg_loss: {}", avg_loss / n_batches as f32);
    }

    // for (x, y) in xs.iter().zip(ys.iter()) {}

    // for index in (0..3).map(|_| rand::thread_rng().gen_range(0..n)) {
    //     let x = word2vec::convert_one_hot_to_tensor(vec![vocab.one_hot_encoded[index]], n)?;
    //     let logits = nn.forward(&x)?;
    //     let result = word2vec::tensor_to_number(&logits, n)?;
    //     println!("result: {result}");
    //     println!(
    //         "INDEX={index} INPUT={input} INPUT={input:#0n$b}",
    //         input = vocab.one_hot_encoded[index]
    //     );
    // }

    // let output = word2vec::tensor_to_number(&loss)?;
    // println!("output: {output:b}");

    Ok(())
}

fn run2() {
    word2vec::foo().expect("Failed to foo");
}
