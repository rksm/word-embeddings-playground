use candle_core::{Result, Tensor};
use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{vocab_builder::Vocab, word2vec};

#[derive(Default)]
pub struct DatasetOptions {
    pub batch_size: usize,
    pub epochs: usize,
    pub vocab: Vocab,
}

impl DatasetOptions {
    pub fn new() -> Self {
        Self {
            batch_size: 20,
            epochs: 5,
            vocab: Vocab::default(),
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.vocab = vocab;
        self
    }

    pub fn build(self) -> Result<Dataset> {
        let Self {
            batch_size,
            epochs: epoch_count,
            vocab,
        } = self;
        let mut epochs = Vec::new();

        let batches_per_epoch = dbg!(vocab.n()) / dbg!(batch_size);
        dbg!(batches_per_epoch);

        for _ in 0..epoch_count {
            epochs.push(Epoch::random(batches_per_epoch, batch_size, &vocab)?);
        }

        Ok(Dataset {
            n: vocab.n(),
            vocab: vocab.words,
            epochs,
        })
    }
}

pub struct Dataset {
    pub n: usize,
    pub vocab: Vec<String>,
    pub epochs: Vec<Epoch>,
}

pub struct Epoch {
    pub batches: Vec<Batch>,
}

impl Epoch {
    pub fn random(n_batches: usize, batch_size: usize, vocab: &Vocab) -> Result<Self> {
        let mut batches = Vec::new();

        for _ in 0..n_batches {
            batches.push(Batch::random(batch_size, vocab)?);
        }

        Ok(Self { batches })
    }
}

pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch {
    pub fn random(batch_size: usize, vocab: &Vocab) -> Result<Self> {
        let sample = thread_rng().sample_iter(Uniform::new(0u32, vocab.n() as u32));
        let word_indexes = sample.take(batch_size).collect::<Vec<_>>();
        Self::new(&word_indexes)
    }

    pub fn new(word_indexes: &[u32]) -> Result<Self> {
        let batch_size = word_indexes.len();
        let y = Tensor::from_vec(word_indexes.to_vec(), (batch_size,), &word2vec::DEVICE)?;
        let x = Tensor::from_vec(word_indexes.to_vec(), (batch_size,), &word2vec::DEVICE)?;
        Ok(Self { y, x })
    }
}
