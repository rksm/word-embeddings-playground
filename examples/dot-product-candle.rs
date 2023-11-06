use candle_core::{backend::BackendDevice, Device, Tensor};
use rand::Rng;

fn main() {
    // let a = vec![1.0, 2.0, 3.0];
    // let b = vec![4.0, 5.0, 6.0];
    // run(a, b, vec![3]);

    let dim = 10000000;
    let mut rng = rand::thread_rng();
    let inputs = (0..3)
        .map(|_| {
            let mut v = Vec::new();
            for _ in 0..dim {
                v.push(rng.gen_range(0.0..1.0));
            }
            v
        })
        .collect::<Vec<_>>();

    let start = std::time::Instant::now();
    for a in &inputs {
        for b in &inputs {
            run_manual(a, b);
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);

    // let d = Device::Cpu;
    let d = Device::Cuda(candle_core::CudaDevice::new(0).unwrap());

    let a = inputs
        .into_iter()
        .map(|v| Tensor::from_vec(v, vec![dim], &d).unwrap())
        .collect::<Vec<_>>();
    let b = a.clone();

    let start = std::time::Instant::now();
    for a in &a {
        for b in &b {
            run_candle(a, b);
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
}

fn run_candle(a: &Tensor, b: &Tensor) {
    let c = candle::dot_product(a, b).unwrap();
    println!("{:?}", c);

    // let s = candle::cosine_similarity(a, b).unwrap();
    // println!("{:?}", s);
}

fn run_manual(a: &[f64], b: &[f64]) {
    let c = manual::dot_product(a, b);
    println!("{:?}", c);

    // let s = manual::cosine_similarity_efficiently(a, b);
    // let s = manual::cosine_similarity_efficiently(&a, &b);
    // println!("{:?}", s);
}

mod candle {
    use candle_core::Result;
    use candle_core::Tensor;

    #[inline(always)]
    pub fn dot_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        (a * b)?.sum(0)
    }

    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        dot_product(a, b)? / (dot_product(a, a)? * &dot_product(b, b)?)?.sqrt()?
    }
}

mod manual {

    #[inline(always)]
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    #[allow(dead_code)]
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        dot_product(a, b) / (dot_product(a, a) * dot_product(b, b)).sqrt()
    }

    #[allow(dead_code)]
    pub fn cosine_similarity_efficiently(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut a_squared = 0.0;
        let mut b_squared = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
            a_squared += a[i] * a[i];
            b_squared += b[i] * b[i];
        }
        sum / (a_squared * b_squared).sqrt()
    }
}
