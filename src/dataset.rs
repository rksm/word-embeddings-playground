use candle_core::{Result, Tensor};
use std::collections::HashSet;

use crate::{vocab_builder::Vocab, word2vec};

pub struct DatasetOptions<'a> {
    pub batch_size: usize,
    pub batches_per_epoch: Option<usize>,
    pub epochs: usize,
    pub vocab: &'a Vocab,
    pub method: word2vec::Method,
}

impl<'a> DatasetOptions<'a> {
    pub fn new(method: word2vec::Method, vocab: &'a Vocab) -> Self {
        Self {
            batch_size: 20,
            batches_per_epoch: None,
            epochs: 5,
            vocab,
            method,
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
    pub fn entire_context_in_one_epoch(mut self) -> Self {
        self.batches_per_epoch = Some(self.vocab.context_n() / self.batch_size);
        self
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
            method,
        } = self;
        let mut epochs = Vec::new();

        for _ in 0..epoch_count {
            epochs.push(Epoch::random(batches_per_epoch, batch_size, method, vocab)?);
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
    pub method: word2vec::Method,
}

impl<'a> Iterator for Epoch<'a> {
    type Item = Box<dyn Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(remaining_batch_count) = &mut self.remaining_batch_count {
            if *remaining_batch_count == 0 {
                return None;
            }
            *remaining_batch_count -= 1;
        }

        let batch: Box<dyn Batch> = match self.method {
            word2vec::Method::Cbow => Box::new(
                CbowBatch::random(
                    self.batch_size,
                    &mut self.remaining_context_indexes,
                    self.vocab,
                )
                .expect("Failed to create batch"),
            ),
            word2vec::Method::SkipGram => Box::new(
                SkipgramBatch::random(
                    self.batch_size,
                    &mut self.remaining_context_indexes,
                    self.vocab,
                )
                .expect("Failed to create batch"),
            ),
        };

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

impl<'a> Epoch<'a> {
    pub fn random(
        n_batches: Option<usize>,
        batch_size: usize,
        method: word2vec::Method,
        vocab: &'a Vocab,
    ) -> Result<Self> {
        let remaining = (0u32..vocab.context_n() as u32).collect::<HashSet<_>>();
        Ok(Self {
            remaining_batch_count: n_batches,
            batch_size,
            remaining_context_indexes: remaining,
            vocab,
            method,
        })
    }
}

pub trait Batch {
    fn x(&self) -> &Tensor;
    fn y(&self) -> &Tensor;

    fn len(&self) -> usize {
        self.x().elem_count()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct CbowBatch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch for CbowBatch {
    fn x(&self) -> &Tensor {
        &self.x
    }

    fn y(&self) -> &Tensor {
        &self.y
    }
}

impl CbowBatch {
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
}

pub struct SkipgramBatch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch for SkipgramBatch {
    fn x(&self) -> &Tensor {
        &self.x
    }

    fn y(&self) -> &Tensor {
        &self.y
    }
}

impl SkipgramBatch {
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
            y.extend(
                context
                    .into_iter()
                    .map(|i| vocab.context_to_word_index(i))
                    .collect::<Vec<_>>(),
            );
            x.extend(vec![vocab.context_to_word_index(target); 4]);
        }
        let batch_size = word_indexes.len() * 4;
        let x = Tensor::from_vec(x, (batch_size, 1), &word2vec::DEVICE)?;
        let y = Tensor::from_vec(y, (batch_size, 1), &word2vec::DEVICE)?;
        Ok(Self { y, x })
    }
}
