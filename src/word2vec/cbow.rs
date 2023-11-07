use candle_core::{
    safetensors::{load, save},
    DType, Module, Result, Tensor,
};
use candle_nn::VarBuilder;
use lazy_static::lazy_static;
use std::{collections::HashMap, path::PathBuf};

use crate::word2vec::DEVICE;

use super::{files::backup_file, net::Word2VecNet};

lazy_static! {
    static ref ZEROES: Tensor = Tensor::zeros((1,), DType::F32, &DEVICE).unwrap();
}

pub struct Word2VecCbow {
    projection_layer: candle_nn::Embedding,
    output_layer: candle_nn::Linear,
}

impl Word2VecNet for Word2VecCbow {
    fn forward(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        let xs = self.projection_layer.forward(x)?;
        let xs = xs.mean(1)?;
        let xs = self.output_layer.forward(&xs)?;
        Ok(xs)
    }

    fn save(&self, file: PathBuf) -> anyhow::Result<()> {
        backup_file(&file, 10)?;
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

impl Word2VecCbow {
    pub fn new(vocab_size: usize, embedding_size: usize, vs: VarBuilder) -> Result<Self> {
        let projection_layer =
            candle_nn::embedding(vocab_size, embedding_size, vs.pp("projection"))?;
        let output_layer = candle_nn::linear(embedding_size, vocab_size, vs.pp("output"))?;

        Ok(Self {
            projection_layer,
            output_layer,
        })
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
}
