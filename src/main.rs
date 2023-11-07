#[macro_use]
extern crate log;

use candle_core::Result;
use clap::Parser;
use std::path::PathBuf;

use word_embeddings_playground::dataset::DatasetOptions;
use word_embeddings_playground::vocab_builder;
use word_embeddings_playground::word2vec;

#[derive(Parser)]
struct Args {
    #[clap(long, action, default_value = "false")]
    adjust_learning_rate: bool,

    #[clap(short = 'f', long)]
    model_file: Option<PathBuf>,

    #[clap(long, default_value_t = word2vec::Method::Cbow)]
    method: word2vec::Method,

    #[clap(long = "lr", default_value = "0.05")]
    learning_rate: f64,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    run(Args::parse()).expect("Failed to run");
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

    // if false {
    //     let nn = word2vec::Word2VecCbow::load(&nn_file, word2vec::EMBEDDING_SIZE)
    //         .expect("Failed to load");

    //     let context = [
    //         dbg!(vocab.encode("hörer")).unwrap(),
    //         dbg!(vocab.encode("investor")).unwrap(),
    //         dbg!(vocab.encode("tech")).unwrap(),
    //         dbg!(vocab.encode("earnings")).unwrap(),
    //     ];
    //     let x = Tensor::from_vec(context.to_vec(), (1, 4), &word2vec::DEVICE)?;
    // let output = nn.forward(&x)?;
    // dbg!(output.shape());
    // dbg!(output.to_vec2::<f32>()?);
    // dbg!(word2vec::tensor_to_number(&output, n)?);
    // let n = word2vec::tensor_to_number(&output, n)?;
    // dbg!(vocab.word_lookup(n as _));
    // dbg!(output);
    // }

    if true {
        let dataset = DatasetOptions::new(args.method, &vocab)
            .batch_size(20)
            // .batches_per_epoch(5000)
            .epochs(word2vec::EPOCHS)
            .entire_context_in_one_epoch()
            .build()
            .expect("Failed to build dataset");

        let mut training = match &args.model_file {
            Some(file) if file.exists() => {
                word2vec::Training::load(args.method, args.learning_rate, nn_file)
                    .expect("Failed to load")
            }
            _ => word2vec::Training::new(args.method, args.learning_rate, n)
                .expect("Failed to create"),
        };

        training.model_file(args.model_file);
        training.adjust_learning_rate(args.adjust_learning_rate);

        info!("starting training");

        training.run(dataset).expect("Failed to run");
    };

    Ok(())
}
