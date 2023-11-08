use candle_core::{
    safetensors::{load, save},
    DType, Module, Result, Tensor,
};
use candle_nn::{ops, VarBuilder};
use lazy_static::lazy_static;
use std::{collections::HashMap, path::PathBuf};

use crate::word2vec::DEVICE;

use super::{files::backup_file, net::Word2VecNet};

lazy_static! {
    static ref ZEROES: Tensor = Tensor::zeros((1,), DType::F32, &DEVICE).unwrap();
}

#[inline(always)]
pub fn dot_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    (a * b)?.sum(1)
}

pub struct Word2VecSkipGram {
    pub in_embed: candle_nn::Embedding,
    pub out_embed: candle_nn::Embedding,
    pub output_layer: candle_nn::Linear,
}

impl Word2VecNet for Word2VecSkipGram {
    fn forward(&self, target: &Tensor, context: &Tensor) -> Result<Tensor> {
        let target_embed = self.in_embed.forward(target)?;
        let context_embed = self.out_embed.forward(context)?;

        // compute log-likelihood using dot-product of the embeddings (similarity)
        let scores = dot_product(&target_embed, &context_embed)?;

        let log_likelyhood = ops::log_softmax(&scores, 1)?;
        // let log_likelyhood = log_likelyhood.sum_keepdim(1)?;
        // let log_likelyhood = log_likelyhood.mean_keepdim(1)?;
        let log_likelyhood = self.output_layer.forward(&log_likelyhood)?;

        Ok(log_likelyhood)
    }

    fn save(&self, file: PathBuf) -> anyhow::Result<()> {
        backup_file(&file, 10)?;
        let in_embed = self.in_embed.embeddings();
        let out_embed = self.out_embed.embeddings();

        let output = self.output_layer.weight();
        let bias = self.output_layer.bias().unwrap();

        let map = HashMap::from_iter([
            ("in_embed", in_embed.clone()),
            ("out_embed", out_embed.clone()),
            ("output", output.clone()),
            ("bias", bias.clone()),
        ]);

        save(&map, file)?;
        Ok(())
    }
}

impl Word2VecSkipGram {
    pub fn new(vocab_size: usize, embedding_size: usize, vs: VarBuilder) -> Result<Self> {
        let in_embed = candle_nn::embedding(vocab_size, embedding_size, vs.pp("in_embed"))?;
        let out_embed = candle_nn::embedding(vocab_size, embedding_size, vs.pp("out_embed"))?;
        let output_layer = candle_nn::linear(embedding_size, vocab_size, vs.pp("output"))?;

        Ok(Self {
            in_embed,
            out_embed,
            output_layer,
        })
    }

    pub fn load(file: impl AsRef<std::path::Path>, embedding_size: usize) -> anyhow::Result<Self> {
        let file = file.as_ref();
        let map = load(file, &DEVICE)?;

        let in_embed = map.get("in_embed").unwrap();
        let in_embed = candle_nn::Embedding::new(in_embed.clone(), embedding_size);

        let out_embed = map.get("out_embed").unwrap();
        let out_embed = candle_nn::Embedding::new(out_embed.clone(), embedding_size);

        let output = map.get("output").unwrap();
        let bias = map.get("bias").unwrap();
        let output_layer = candle_nn::Linear::new(output.clone(), Some(bias.clone()));

        Ok(Self {
            in_embed,
            out_embed,
            output_layer,
        })
    }
}
