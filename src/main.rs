use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};

#[macro_use]
extern crate log;

mod vocab_builder;
mod word2vec;

use rand::prelude::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    run();
    // run2();
}

fn run() {
    // vocab_builder::doppelgaenger().expect("Failed to build vocab");
    let vocab_file = "./data/tiny-vocab-doppelgaenger.txt";
    let (n, one_hot_encoded) =
        vocab_builder::vocab_one_hot_encoded(vocab_file).expect("Failed to build vocab");

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    const EMBEDDING_SIZE: usize = 100;
    const LEARNING_RATE: f64 = 0.01;
    const EPOCHS: usize = 5000;
    let nn = word2vec::Word2VecNN::new(n, EMBEDDING_SIZE, vs.clone()).expect("Failed to create NN");
    let mut sgd =
        candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE).expect("Failed to create SGD");

    let mut ys = Vec::new();
    let mut xs = Vec::new();

    for _ in 0..EPOCHS {
        let index = rand::thread_rng().gen_range(0..n);
        let y =
            Tensor::from_vec(vec![index as u32], 1, &Device::Cpu).expect("Failed to create tensor");
        // println!("INDEX={index} INPUT={input} INPUT={input:#0n$b}", input = one_hot_encoded[index]);
        let x = word2vec::convert_one_hot_to_tensor(one_hot_encoded[index], n)
            .expect("Failed to convert one-hot to tensor");
        xs.push(x);
        ys.push(y);
    }

    for (x, y) in xs.iter().zip(ys.iter()) {
        let logits = nn.forward(x).expect("Failed to forward");
        // dbg!(logits.to_vec2::<f32>().expect("Failed to convert to vec2"));
        let log_sm = ops::log_softmax(&logits, D::Minus1).expect("Failed to log softmax");
        let loss = loss::nll(&log_sm, y).expect("Failed to compute loss");
        let loss_scalar = loss.mean_all().unwrap().to_scalar::<f32>().unwrap();
        println!("loss: {loss_scalar}");
        sgd.backward_step(&loss).expect("Failed to backward step");
    }

    for index in (0..3).map(|_| rand::thread_rng().gen_range(0..n)) {
        let x = word2vec::convert_one_hot_to_tensor(one_hot_encoded[index], n)
            .expect("Failed to convert one-hot to tensor");
        let logits = nn.forward(&x).expect("Failed to forward");
        let result =
            word2vec::tensor_to_number(&logits, n).expect("Failed to convert tensor to number");
        println!("result: {result}");
        println!(
            "INDEX={index} INPUT={input} INPUT={input:#0n$b}",
            input = one_hot_encoded[index]
        );
    }

    // let output = word2vec::tensor_to_number(&loss).expect("Fail ed to convert tensor to number");
    // println!("output: {output:b}");
}

fn run2() {
    word2vec::foo().expect("Failed to foo");
}
