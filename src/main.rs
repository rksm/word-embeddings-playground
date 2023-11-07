use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Optimizer, VarBuilder, VarMap};

#[macro_use]
extern crate log;

mod vocab_builder;
mod word2vec;

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
    const EMBEDDING_SIZE: usize = 5;
    const LEARNING_RATE: f64 = 0.05;
    const EPOCHS: usize = 100;
    let nn = word2vec::Word2VecNN::new(n, EMBEDDING_SIZE, vs.clone()).expect("Failed to create NN");
    let mut sgd =
        candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE).expect("Failed to create SGD");

    let index = 2;
    let y = Tensor::from_vec(vec![index as u32], 1, &Device::Cpu).expect("Failed to create tensor");
    let input = one_hot_encoded[index];
    println!();
    // binary + pad to n

    println!("INDEX={index} INPUT={input} INPUT={input:#0n$b}");

    let input2 =
        word2vec::convert_one_hot_to_tensor(input, n).expect("Failed to convert one-hot to tensor");
    // let target = input2.clone();

    let input3 = input2
        .to_dtype(DType::F32)
        .expect("Failed to convert to f32")
        .unsqueeze(0)
        .expect("Failed to unsqueeze");
    // println!(
    //     "input: {:b}",
    //     word2vec::tensor_to_number(&input2).expect("Failed to convert tensor to number")
    // );

    for epoch in 1..EPOCHS + 1 {
        let logits = nn.forward(&input3).expect("Failed to forward");
        dbg!(logits.to_vec2::<f32>().expect("Failed to convert to vec2"));
        let log_sm = ops::log_softmax(&logits, D::Minus1).expect("Failed to log softmax");
        let loss = loss::nll(&log_sm, &y).expect("Failed to compute loss");
        dbg!(loss.to_vec0::<f32>().expect("Failed to convert to vec1"));

        sgd.backward_step(&loss).expect("Failed to backward step");
    }

    // let output = word2vec::tensor_to_number(&loss).expect("Fail ed to convert tensor to number");
    // println!("output: {output:b}");
}

fn run2() {
    word2vec::foo().expect("Failed to foo");
}
