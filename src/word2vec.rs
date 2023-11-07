use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use lazy_static::lazy_static;

#[cfg(feature = "cuda")]
lazy_static! {
    pub static ref DEVICE: Device =
        Device::Cuda(candle_core::backend::BackendDevice::new(0).unwrap());
}

#[cfg(not(feature = "cuda"))]
lazy_static! {
    pub static ref DEVICE: Device = Device::Cpu;
}

lazy_static! {
    static ref ZEROES: Tensor = Tensor::zeros((1,), DType::F32, &DEVICE).unwrap();
}

pub fn foo() -> Result<()> {
    let t = Tensor::rand(0.0f32, 1.0f32, (3,), &DEVICE)?;
    dbg!(t.shape());
    dbg!(&t);
    let s = Tensor::new(3.0, &DEVICE)?;
    dbg!(&s);
    let r = t.broadcast_div(&s)?;
    dbg!(&r);
    Ok(())
}

pub fn convert_one_hot_to_tensor(
    one_hot: Vec<u32>,
    n: usize,
) -> candle_core::Result<candle_core::Tensor> {
    let numbers = one_hot
        .iter()
        .flat_map(|&i| one_hot_to_vec(i, n))
        .collect::<Vec<_>>();
    Tensor::from_vec(numbers, (one_hot.len(), n), &DEVICE)
}

pub fn one_hot_to_vec(one_hot: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(move |i| if one_hot & (1 << i) != 0 { 1.0 } else { 0.0 })
        .collect()
}

/// Convert a one-hot encoded tensor to a number
pub fn tensor_to_number(t: &Tensor, n: usize) -> Result<u32> {
    let t = t.reshape((n,))?;
    let mut number = 0u32;
    let mut max = -1.0;

    for (i, v) in t.to_vec1::<f32>()?.iter().enumerate() {
        if *v > max {
            max = *v;
            number = i as u32;
        }
    }

    Ok(number)
}

pub struct Word2VecNN {
    projection_layer: candle_nn::Linear,
    output_layer: candle_nn::Linear,
}

impl Word2VecNN {
    pub fn new(vocab_size: usize, embedding_size: usize, vs: VarBuilder) -> Result<Self> {
        let projection_layer = candle_nn::linear(vocab_size, embedding_size, vs.pp("projection"))?;
        let output_layer = candle_nn::linear(embedding_size, vocab_size, vs.pp("output"))?;

        Ok(Self {
            projection_layer,
            output_layer,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // dbg!(&self.projection_layer.weight().to_vec2::<f32>()?);
        // dbg!(&self.projection_layer.bias());

        let xs = self.projection_layer.forward(input)?;
        // let xs = xs.relu()?;
        let xs = self.output_layer.forward(&xs)?;
        Ok(xs)
    }
}

pub fn train() -> Result<()> {
    todo!()
}
