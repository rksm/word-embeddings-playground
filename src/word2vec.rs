use std::{collections::HashMap, fmt::Display, path::Path};

use candle_core::{
    safetensors::{load, save},
    DType, Device, Module, Result, Tensor,
};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use lazy_static::lazy_static;

use crate::{
    dataset::{Batch, Dataset},
    vocab_builder::Vocab,
};

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

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum Method {
    Cbow,
    SkipGram,
}

impl Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::Cbow => write!(f, "cbow"),
            Method::SkipGram => write!(f, "skipgram"),
        }
    }
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

        // backups
        if let Some(dir) = file.parent() {
            let fname = file.file_name().unwrap().to_string_lossy();
            let mut prev_files = std::fs::read_dir(dir)?
                .filter_map(|f| f.ok())
                .filter_map(|f| {
                    let n = f.file_name();
                    let n = n.to_string_lossy();
                    if !n.starts_with(&*fname) {
                        return None;
                    }
                    let n = n.trim_start_matches(&*fname);
                    if n.is_empty() {
                        return None;
                    }
                    let n = n.trim_start_matches('.');
                    if n.is_empty() {
                        return None;
                    }
                    let n = n.parse::<u8>().ok()?;
                    Some((n, f.path()))
                })
                .collect::<Vec<_>>();
            prev_files.sort_by_key(|(n, _)| *n);
            prev_files.reverse();
            for (n, f) in prev_files {
                if n >= 9 {
                    std::fs::remove_file(f)?;
                    continue;
                }
                let new_name = format!("{}.{}", fname, n + 1);
                std::fs::rename(f, dir.join(new_name))?;
            }
        }

        if file.exists() {
            std::fs::rename(file, file.with_extension("nn.1"))?;
        }

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

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

pub const EPOCHS: usize = 100;
pub const EMBEDDING_SIZE: usize = 100;
const LEARNING_RATE: f64 = 0.05;

pub struct Training {
    pub nn: Word2VecNN,
    pub optim: candle_nn::SGD,
    adjust_learning_rate: bool,
    epoch: usize,
    batch: usize,
    last_loss: Option<f32>,
    avg_loss: f32,
    save: bool,
}

impl Training {
    pub fn load(nn_file: impl AsRef<Path>) -> Result<Self> {
        let nn = Word2VecNN::load(nn_file, EMBEDDING_SIZE).expect("Failed to load");
        let varmap = VarMap::new();
        let optim = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

        Ok(Self {
            nn,
            optim,
            adjust_learning_rate: false,
            save: false,
            epoch: 0,
            batch: 0,
            avg_loss: 0.0,
            last_loss: None,
        })
    }

    pub fn new(vocab_size: usize) -> Result<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
        let nn = Word2VecNN::new(vocab_size, EMBEDDING_SIZE, vs.clone())?;
        let optim = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
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
            save: false,
            epoch: 0,
            batch: 0,
            avg_loss: 0.0,
            last_loss: None,
        })
    }

    pub fn adjust_learning_rate(&mut self, adjust_learning_rate: bool) {
        self.adjust_learning_rate = adjust_learning_rate;
    }

    pub fn save(&mut self, save: bool) {
        self.save = save;
    }

    pub fn run(&mut self, dataset: Dataset) -> Result<()> {
        for (i, epoch) in dataset.epochs.into_iter().enumerate() {
            print!("epoch {i}/{EPOCHS} ");

            let mut avg_loss = 0.0;
            let mut n_batches = 0;

            for Batch { x, y } in epoch {
                n_batches += 1;
                let logits = self.nn.forward(&x).expect("Failed to forward");
                let loss = loss::cross_entropy(&logits, &y).expect("Failed to compute loss");
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
        if self.save {
            self.nn.save("nn.bin").expect("Failed to save");
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
            save,
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

            if *save {
                nn.save("nn.bin").expect("Failed to save");
            }
        }
    }
}
