use candle_core::{DType, Result};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use std::path::{Path, PathBuf};

use crate::{
    dataset::Dataset,
    word2vec::{net::Word2VecNet, DEVICE, EPOCHS},
};

use super::{skip_gram::Word2VecSkipGram, Method, Word2VecCbow, EMBEDDING_SIZE};

pub struct Training {
    pub nn: Box<dyn Word2VecNet>,
    pub optim: candle_nn::SGD,
    adjust_learning_rate: bool,
    epoch: usize,
    batch: usize,
    last_loss: Option<f32>,
    avg_loss: f32,
    model_file: Option<PathBuf>,
}

impl Training {
    pub fn load(method: Method, lr: f64, nn_file: impl AsRef<Path>) -> Result<Self> {
        let nn: Box<dyn Word2VecNet> = match method {
            Method::Cbow => {
                Box::new(Word2VecCbow::load(nn_file, EMBEDDING_SIZE).expect("Failed to load"))
            }
            Method::SkipGram => {
                Box::new(Word2VecSkipGram::load(nn_file, EMBEDDING_SIZE).expect("Failed to load"))
            }
        };
        let varmap = VarMap::new();
        let optim = candle_nn::SGD::new(varmap.all_vars(), lr)?;

        Ok(Self {
            nn,
            optim,
            adjust_learning_rate: false,
            model_file: None,
            epoch: 0,
            batch: 0,
            avg_loss: 0.0,
            last_loss: None,
        })
    }

    pub fn new(method: Method, lr: f64, vocab_size: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
        let nn: Box<dyn Word2VecNet> = match method {
            Method::Cbow => Box::new(Word2VecCbow::new(vocab_size, EMBEDDING_SIZE, vs.clone())?),
            Method::SkipGram => Box::new(Word2VecSkipGram::new(
                vocab_size,
                EMBEDDING_SIZE,
                vs.clone(),
            )?),
        };
        let optim = candle_nn::SGD::new(varmap.all_vars(), lr)?;
        // let mut optim = candle_nn::AdamW::new(
        //     varmap.all_vars(),
        //     candle_nn::ParamsAdamW {
        //         lr: LEARNING_RATE,
        //         ..Default::default()
        //     },
        // )?;

        Ok(Self {
            nn,
            optim,
            adjust_learning_rate: false,
            model_file: None,
            epoch: 0,
            batch: 0,
            avg_loss: 0.0,
            last_loss: None,
        })
    }

    pub fn adjust_learning_rate(&mut self, adjust_learning_rate: bool) {
        self.adjust_learning_rate = adjust_learning_rate;
    }

    pub fn model_file(&mut self, f: Option<impl Into<PathBuf>>) {
        self.model_file = f.map(|f| f.into());
    }

    pub fn run(&mut self, dataset: Dataset) -> Result<()> {
        for (i, epoch) in dataset.epochs.into_iter().enumerate() {
            print!("epoch {i}/{EPOCHS} ");

            let mut avg_loss = 0.0;
            let mut n_batches = 0;

            for batch in epoch {
                let x = batch.x();
                let y = batch.y();
                n_batches += 1;

                let logits = self.nn.forward(x, y).expect("Failed to forward");
                let y = y.squeeze(1)?;
                let loss = loss::cross_entropy(&logits, &y).expect("Failed to compute loss");
                // let loss = loss::nll(&logits, &y).expect("Failed to compute loss");

                let loss_scalar = loss
                    .mean_all()?
                    .to_scalar::<f32>()
                    .expect("Failed to to_scalar");

                self.optim
                    .backward_step(&loss)
                    .expect("Failed to backward_step");

                avg_loss += loss_scalar;
                self.avg_loss = avg_loss / n_batches as f32;

                self.batch_step(loss_scalar);
            }

            self.epoch_step();
        }

        Ok(())
    }

    fn epoch_step(&mut self) {
        self.avg_loss = 0.0;
        self.epoch += 1;
        self.batch = 0;
        if let Some(f) = &self.model_file {
            self.nn.save(f.to_path_buf()).expect("Failed to save");
        }
    }

    fn batch_step(&mut self, loss: f32) {
        self.batch += 1;

        let Self {
            nn,
            optim,
            adjust_learning_rate,
            epoch,
            batch,
            avg_loss,
            last_loss,
            model_file,
        } = self;

        if *adjust_learning_rate {
            if let Some(last_loss) = last_loss {
                let delta = *last_loss - loss;
                match f32::abs(delta) {
                    delta if delta > 0.75 => {
                        let lr = optim.learning_rate();
                        let new_lr = lr * 0.75;
                        warn!("adjusting learning rate from {} to {}", lr, new_lr);
                        optim.set_learning_rate(new_lr);
                    }
                    delta if delta < 0.01 => {
                        let lr = optim.learning_rate();
                        let new_lr = lr * 1.1;
                        warn!("adjusting learning rate from {} to {}", lr, new_lr);
                        optim.set_learning_rate(new_lr);
                    }
                    _ => (),
                }
            }
        }

        *last_loss = Some(loss);

        if *batch % 250 == 0 {
            println!(
                "epoch={epoch} batch={batch} loss={loss} avg_loss={} lr={}",
                avg_loss,
                optim.learning_rate(),
            );

            if let Some(f) = model_file {
                nn.save(f.to_path_buf()).expect("Failed to save");
            }
        }
    }
}
