use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use lazy_static::lazy_static;

lazy_static! {
    static ref DEVICE: Device = Device::Cpu;
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
    one_hot: u32,
    n: usize,
) -> candle_core::Result<candle_core::Tensor> {
    let mut numbers = vec![0u32; n];

    (0..n).for_each(|i| {
        if one_hot & (1 << i) != 0 {
            numbers[i] = 1;
        }
    });

    // Tensor::from_vec(numbers, n, &Device::Cpu)
    Tensor::from_iter(numbers, &Device::Cpu)
}

/// Convert a one-hot encoded tensor to a number
pub fn tensor_to_number(t: &Tensor) -> Result<u32> {
    let mut number = 0;

    for (i, v) in t.to_vec1::<f32>()?.iter().enumerate() {
        if *v > 0.5 {
            number |= 1 << i;
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

fn softmax(x: &Tensor) -> Result<Tensor> {
    let exp = x.exp()?;
    let sum = exp.sum(0)?;
    exp.broadcast_div(&sum)
}

#[cfg(test)]
mod tests {
    #[test]
    fn softmax_test() {
        let x =
            candle_core::Tensor::from_vec(vec![1f32, 2f32, 3f32], (3,), &candle_core::Device::Cpu)
                .unwrap();
        let y = super::softmax(&x).unwrap();
        let expected = vec![0.090_030_57, 0.244_728_48, 0.665_240_94];
        let actual = y.to_vec1::<f32>().unwrap();
        assert_eq!(expected, actual);
    }
}
