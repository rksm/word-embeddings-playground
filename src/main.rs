mod dataset;
mod vocab_builder;
mod word2vec;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#[macro_use]
extern crate log;

use candle_core::{Result, Tensor};
use clap::Parser;
use std::path::PathBuf;

use crate::dataset::DatasetOptions;

#[derive(Parser)]
struct Args {
    #[clap(long, action, default_value = "false")]
    adjust_learning_rate: bool,

    #[clap(long, action, default_value = "true")]
    load: bool,

    #[clap(long, action, default_value = "true")]
    save: bool,

    #[clap(long, default_value_t = word2vec::Method::Cbow)]
    method: word2vec::Method,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    run(Args::parse()).expect("Failed to run");
    // run2();
}

fn run(args: Args) -> Result<()> {
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

    let nn_file = PathBuf::from("data/word2vec.nn");

    if false {
        let nn =
            word2vec::Word2VecNN::load(&nn_file, word2vec::EMBEDDING_SIZE).expect("Failed to load");

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
            // .batches_per_epoch(5000)
            .epochs(word2vec::EPOCHS)
            .entire_context_in_one_epoch()
            .build()
            .expect("Failed to build dataset");

        let mut training = if args.load && nn_file.exists() {
            word2vec::Training::load(nn_file).expect("Failed to load")
        } else {
            word2vec::Training::new(n).expect("Failed to create")
        };

        training.adjust_learning_rate(args.adjust_learning_rate);
        training.save(args.save);

        training.run(dataset).expect("Failed to run");
    };

    Ok(())
}

#[allow(dead_code)]
fn run2() {
    word2vec::foo().expect("Failed to foo");
}
