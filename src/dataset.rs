use candle_core::{Result, Tensor};
use std::collections::HashSet;

use crate::{vocab_builder::Vocab, word2vec};

#[derive(Default)]
pub struct DatasetOptions {
    pub batch_size: usize,
    pub batches_per_epoch: Option<usize>,
    pub epochs: usize,
    pub vocab: Vocab,
}

impl DatasetOptions {
    pub fn new() -> Self {
        Self {
            batch_size: 20,
            batches_per_epoch: None,
            epochs: 5,
            vocab: Vocab::default(),
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    #[allow(dead_code)]
    pub fn batches_per_epoch(mut self, batches_per_epoch: usize) -> Self {
        self.batches_per_epoch = Some(batches_per_epoch);
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
            batches_per_epoch,
            epochs: epoch_count,
            vocab,
        } = self;
        let mut epochs = Vec::new();

        let batches_per_epoch = batches_per_epoch.unwrap_or_else(|| vocab.context_n() / batch_size);
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
    // pub batches: Vec<Batch>,
    pub batch_size: usize,
    pub remaining_context_indexes: HashSet<u32>,
}

impl Iterator for Epoch {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let b = Batch::random(self.batch_size, &mut self.remaining_context_indexes)
            .expect("Failed to create batch");

        if b.is_empty() {
            None
        } else {
            Some(b)
        }
    }
}

impl Epoch {
    pub fn random(n_batches: usize, batch_size: usize, vocab: &Vocab) -> Result<Self> {
        // let mut remaining = (0u32..vocab.context_n() as u32).collect::<HashSet<_>>();
        let mut remaining = (0u32..vocab.n() as u32).collect::<HashSet<_>>();

        Ok(Self {
            batch_size,
            remaining_context_indexes: remaining,
        })

        // let mut batches = Vec::new();

        // for _ in 0..n_batches {
        //     let b = Batch::random(batch_size, &mut remaining, vocab)?;
        //     if b.is_empty() {
        //         break;
        //     }
        //     batches.push(b);
        // }

        // Ok(Self { batches })
    }
}

pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch {
    pub fn random(batch_size: usize, remaining_context_indexes: &mut HashSet<u32>) -> Result<Self> {
        let word_indexes = remaining_context_indexes
            .iter()
            .take(batch_size)
            .copied()
            .collect::<HashSet<_>>();
        remaining_context_indexes.retain(|i| !word_indexes.contains(i));

        let word_indexes = word_indexes.into_iter().collect::<Vec<_>>();
        Self::new(&word_indexes)
    }

    pub fn new(word_indexes: &[u32]) -> Result<Self> {
        let batch_size = word_indexes.len();
        let y = Tensor::from_vec(word_indexes.to_vec(), (batch_size,), &word2vec::DEVICE)?;
        let x = Tensor::from_vec(word_indexes.to_vec(), (batch_size,), &word2vec::DEVICE)?;
        Ok(Self { y, x })
    }

    pub fn len(&self) -> usize {
        self.x.elem_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
