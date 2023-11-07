use candle_core::{Result, Tensor};
use std::collections::HashSet;

use crate::{vocab_builder::Vocab, word2vec};

pub struct DatasetOptions<'a> {
    pub batch_size: usize,
    pub batches_per_epoch: Option<usize>,
    pub epochs: usize,
    pub vocab: &'a Vocab,
}

impl<'a> DatasetOptions<'a> {
    pub fn new(vocab: &'a Vocab) -> Self {
        Self {
            batch_size: 20,
            batches_per_epoch: None,
            epochs: 5,
            vocab,
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

    /// If called will set `batches_per_epoch` so that the entire context is
    /// trained on in one epoch.
    #[allow(dead_code)]
    pub fn entire_context_in_one_epoch(mut self) {
        self.batches_per_epoch = Some(self.vocab.context_n() / self.batch_size);
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn build(self) -> Result<Dataset<'a>> {
        let Self {
            batch_size,
            batches_per_epoch,
            epochs: epoch_count,
            vocab,
        } = self;
        let mut epochs = Vec::new();

        for _ in 0..epoch_count {
            epochs.push(Epoch::random(batches_per_epoch, batch_size, vocab)?);
        }

        Ok(Dataset {
            n: vocab.n(),
            epochs,
        })
    }
}

pub struct Dataset<'a> {
    pub n: usize,
    pub epochs: Vec<Epoch<'a>>,
}

pub struct Epoch<'a> {
    pub remaining_batch_count: Option<usize>,
    pub batch_size: usize,
    pub remaining_context_indexes: HashSet<u32>,
    pub vocab: &'a Vocab,
}

impl<'a> Iterator for Epoch<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(remaining_batch_count) = &mut self.remaining_batch_count {
            if *remaining_batch_count == 0 {
                return None;
            }
            *remaining_batch_count -= 1;
        }

        let b = Batch::random(
            self.batch_size,
            &mut self.remaining_context_indexes,
            self.vocab,
        )
        .expect("Failed to create batch");

        if b.is_empty() {
            None
        } else {
            Some(b)
        }
    }
}

impl<'a> Epoch<'a> {
    pub fn random(n_batches: Option<usize>, batch_size: usize, vocab: &'a Vocab) -> Result<Self> {
        let remaining = (0u32..vocab.context_n() as u32).collect::<HashSet<_>>();
        Ok(Self {
            remaining_batch_count: n_batches,
            batch_size,
            remaining_context_indexes: remaining,
            vocab,
        })
    }
}

pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch {
    pub fn random(
        batch_size: usize,
        remaining_context_indexes: &mut HashSet<u32>,
        vocab: &Vocab,
    ) -> Result<Self> {
        let word_indexes = remaining_context_indexes
            .iter()
            .take(batch_size)
            .copied()
            .collect::<HashSet<_>>();
        remaining_context_indexes.retain(|i| !word_indexes.contains(i));

        let word_indexes = word_indexes.into_iter().collect::<Vec<_>>();
        Self::new(&word_indexes, vocab)
    }

    pub fn new(word_indexes: &[u32], vocab: &Vocab) -> Result<Self> {
        let mut x = Vec::new();
        let mut y = Vec::new();

        for w in word_indexes {
            let (context, target) = vocab.context(*w as usize);
            x.extend(
                context
                    .into_iter()
                    .map(|i| vocab.context_to_word_index(i))
                    .collect::<Vec<_>>(),
            );
            y.push(vocab.context_to_word_index(target));
        }

        let batch_size = word_indexes.len();

        let x = Tensor::from_vec(x, (batch_size, 4), &word2vec::DEVICE)?;
        let y = Tensor::from_vec(y, (batch_size,), &word2vec::DEVICE)?;

        Ok(Self { y, x })
    }

    pub fn len(&self) -> usize {
        self.x.elem_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
