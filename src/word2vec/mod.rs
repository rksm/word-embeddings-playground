mod cbow;
mod files;
mod net;
mod skip_gram;
mod training;

pub use cbow::Word2VecCbow;
pub use training::Training;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

use candle_core::{Device, Tensor};

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

pub const EPOCHS: usize = 1;
pub const EMBEDDING_SIZE: usize = 100;

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum Method {
    Cbow,
    SkipGram,
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::Cbow => write!(f, "cbow"),
            Method::SkipGram => write!(f, "skipgram"),
        }
    }
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

/// Convert a one-hot encoded tensor to a number
#[allow(dead_code)]
pub fn tensor_to_number(t: &Tensor, n: usize) -> candle_core::Result<u32> {
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
