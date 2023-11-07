#[macro_use]
extern crate log;

mod vocab_builder;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    run();
    // run2();
}

fn run() {
    // vocab_builder::doppelgaenger().expect("Failed to build vocab");
    let vocab_file = "./data/vocab-doppelgaenger.txt";
    let (n, one_hot_encoded) =
        vocab_builder::vocab_one_hot_encoded(vocab_file).expect("Failed to build vocab");

    info!("n: {n}");

    for i in one_hot_encoded.iter().take(10) {
        // print as base 2
        println!("{i:b}");
    }

    let embedding_size = 100;
    let nn = word2vec::Word2VecNN::new(n, embedding_size).expect("Failed to create NN");

    let input = word2vec::convert_one_hot_to_tensor(one_hot_encoded[3], n)
        .expect("Failed to convert one-hot to tensor");
    dbg!(&input);
    println!(
        "input: {:b}",
        word2vec::tensor_to_number(&input).expect("Failed to convert tensor to number")
    );

    let output = nn.forward(&input).expect("Failed to forward");
    let output = word2vec::tensor_to_number(&output).expect("Failed to convert tensor to number");
    println!("output: {output:b}");
}

fn run2() {
    word2vec::foo().expect("Failed to foo");
}

mod word2vec {
    use candle_core::{DType, Device, Result, Tensor};
    use lazy_static::lazy_static;

    lazy_static! {
        static ref DEVICE: Device = Device::Cpu;
        static ref ZEROES: Tensor = Tensor::zeros((1,), DType::F64, &DEVICE).unwrap();
    }

    pub fn foo() -> Result<()> {
        let t = Tensor::rand(0.0f64, 1.0f64, (3,), &DEVICE)?;
        dbg!(t.shape());
        dbg!(&t);
        let s = Tensor::new(3.0, &DEVICE)?;
        dbg!(&s);
        let r = t.broadcast_div(&s)?;
        dbg!(&r);
        Ok(())
    }

    pub fn convert_one_hot_to_tensor(
        one_hot: u32,
        n: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let mut numbers = vec![0.0; n];

        (0..n).for_each(|i| {
            if one_hot & (1 << i) != 0 {
                numbers[i] = 1.0;
            }
        });

        Tensor::from_vec(numbers, (n,), &Device::Cpu)
    }

    /// Convert a one-hot encoded tensor to a number
    pub fn tensor_to_number(t: &Tensor) -> Result<u32> {
        let mut number = 0;

        for (i, v) in t.to_vec1::<f64>()?.iter().enumerate() {
            if *v > 0.5 {
                number |= 1 << i;
            }
        }

        Ok(number)
    }

    fn softmax(x: &Tensor) -> Result<Tensor> {
        dbg!(&x);
        dbg!(x.shape());
        let max = x.broadcast_maximum(&ZEROES)?;
        dbg!(&max);
        let exp = x.sub(&max)?;
        dbg!(&exp);
        let exp = exp.exp()?;
        dbg!(exp.shape());
        dbg!(&exp);
        let sum = exp.sum(0)?;
        dbg!(&sum);
        dbg!(sum.to_scalar::<f64>());
        exp.broadcast_div(&sum)
    }

    pub struct Word2VecNN {
        input_layer: Tensor,
        hidden_layer: Tensor,
        output_layer: Tensor,
    }

    impl Word2VecNN {
        pub fn new(vocab_size: usize, embedding_size: usize) -> Result<Self> {
            let input_layer = ZEROES.clone();
            let hidden_layer = Tensor::rand(0.0f64, 1.0f64, (vocab_size, embedding_size), &DEVICE)?;
            let output_layer = Tensor::rand(0.0f64, 1.0f64, (embedding_size, vocab_size), &DEVICE)?;

            Ok(Self {
                input_layer,
                hidden_layer,
                output_layer,
            })
        }

        pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
            info!("input");
            let input = input.reshape((1, input.shape().dims1()?))?;
            info!("hidden");
            let hidden = input.matmul(&self.hidden_layer)?;
            info!("output");
            let output = hidden.matmul(&self.output_layer)?.flatten_to(1)?;
            info!("softmax");
            softmax(&output)
        }
    }
}
