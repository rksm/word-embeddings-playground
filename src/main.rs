use std::path::PathBuf;

use candle_core::{DType, Result, Tensor, D};
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

const EPOCHS: usize = 100;
const EMBEDDING_SIZE: usize = 100;
const LEARNING_RATE: f64 = 0.05;

fn run() -> Result<()> {
    // let vocab = vocab_builder::Vocab::build_from_scratch().expect("Failed to build vocab");
    let vocab = vocab_builder::Vocab::from_files().expect("Failed to build vocab");
    let n = dbg!(vocab.n());

    if false {
        let (context, target) = vocab.context(4);
        dbg!(context);
        dbg!(target);
        for ea in context {
            dbg!(vocab.word_lookup(ea as _));
        }
        dbg!(vocab.word_lookup(target as _));
    }

    println!("----------------");

    let load = false;
    let save = true;
    let nn_file = PathBuf::from("data/word2vec.nn");

    if false {
        let nn = word2vec::Word2VecNN::load(&nn_file, EMBEDDING_SIZE).expect("Failed to load");

        // let x = Tensor::from_vec(vec![0u32; 4], (1, 4), &word2vec::DEVICE)?;

        let context = [
            dbg!(vocab.encode("hörer")).unwrap(),
            dbg!(vocab.encode("investor")).unwrap(),
            dbg!(vocab.encode("tech")).unwrap(),
            dbg!(vocab.encode("earnings")).unwrap(),
        ];
        let x = Tensor::from_vec(context.to_vec(), (1, 4), &word2vec::DEVICE)?;
        let output = nn.forward(&x)?;
        // dbg!(output.shape());
        // dbg!(output.to_vec2::<f32>()?);
        // dbg!(word2vec::tensor_to_number(&output, n)?);
        let n = word2vec::tensor_to_number(&output, n)?;
        dbg!(vocab.word_lookup(n as _));
        // dbg!(output);
    }

    if true {
        let dataset = DatasetOptions::new(&vocab)
            .batch_size(20)
            .batches_per_epoch(500)
            .epochs(EPOCHS)
            .build()
            .expect("Failed to build dataset");

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &word2vec::DEVICE);

        let nn = if load && nn_file.exists() {
            word2vec::Word2VecNN::load(&nn_file, EMBEDDING_SIZE).expect("Failed to load")
        } else {
            word2vec::Word2VecNN::new(n, EMBEDDING_SIZE, vs.clone())?
        };
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

        for (i, epoch) in dataset.epochs.into_iter().enumerate() {
            print!("epoch {i}/{EPOCHS} ");

            let mut avg_loss = 0.0;
            let mut n_batches = 0;

            for (j, batch) in epoch.enumerate() {
                n_batches += 1;
                let x = batch.x;
                let y = batch.y;
                // dbg!(x.shape());
                // dbg!(x.to_vec2::<u32>()?);
                // dbg!(y.shape());
                let logits = nn.forward(&x).expect("Failed to forward");
                // dbg!(logits.shape());
                // dbg!(logits.to_vec2::<f32>()?);
                let log_sm = ops::log_softmax(&logits, D::Minus1).expect("Failed to log_softmax");
                // dbg!(y.to_vec1::<u32>()?);
                let loss = loss::nll(&log_sm, &y).expect("Failed to nll");
                let loss_scalar = loss
                    .mean_all()?
                    .to_scalar::<f32>()
                    .expect("Failed to to_scalar");
                // println!("loss: {loss_scalar}");
                sgd.backward_step(&loss).expect("Failed to backward_step");

                avg_loss += loss_scalar;

                if j % 250 == 0 {
                    println!(
                        "epoch {i} batch {j} loss: {loss_scalar} avg_loss: {}",
                        avg_loss / n_batches as f32
                    );
                }
            }

            println!("avg_loss: {}", avg_loss / n_batches as f32);
            if save {
                nn.save(&nn_file).expect("Failed to save");
                println!("saved to {:?}", nn_file);
            }
        }
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

#[allow(dead_code)]
fn run2() {
    word2vec::foo().expect("Failed to foo");
}
