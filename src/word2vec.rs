use std::collections::HashMap;

use candle_core::{
    safetensors::{load, save},
    DType, Device, Module, Result, Tensor,
};
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
    projection_layer: candle_nn::Embedding,
    output_layer: candle_nn::Linear,
}

impl Word2VecNN {
    pub fn new(vocab_size: usize, embedding_size: usize, vs: VarBuilder) -> Result<Self> {
        let projection_layer =
            candle_nn::embedding(vocab_size, embedding_size, vs.pp("projection"))?;
        let output_layer = candle_nn::linear(embedding_size, vocab_size, vs.pp("output"))?;

        Ok(Self {
            projection_layer,
            output_layer,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let xs = self.projection_layer.forward(input)?;
        let xs = xs.mean(1)?;
        let xs = self.output_layer.forward(&xs)?;
        Ok(xs)
    }

    pub fn load(file: impl AsRef<std::path::Path>, embedding_size: usize) -> anyhow::Result<Self> {
        let file = file.as_ref();
        let map = load(file, &DEVICE)?;

        let embeddings = map.get("embeddings").unwrap();
        let projection_layer = candle_nn::Embedding::new(embeddings.clone(), embedding_size);
        let output = map.get("output").unwrap();
        let bias = map.get("bias").unwrap();
        let output_layer = candle_nn::Linear::new(output.clone(), Some(bias.clone()));

        Ok(Self {
            projection_layer,
            output_layer,
        })
    }

    pub fn save(&self, file: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let file = file.as_ref();
        // let embeddings_file = dir.join("embeddings.bin");
        // let output_file = dir.join("output.bin");

        let embeddings = self.projection_layer.embeddings();
        let output = self.output_layer.weight();
        let bias = self.output_layer.bias().unwrap();

        let map = HashMap::from_iter([
            ("embeddings", embeddings.clone()),
            ("output", output.clone()),
            ("bias", bias.clone()),
        ]);

        save(&map, file)?;

        Ok(())
    }
}
