use candle_core::Tensor;
use std::path::PathBuf;

pub trait Word2VecNet {
    fn forward(&self, x: &Tensor, y: &Tensor) -> candle_core::Result<Tensor>;
    fn save(&self, file: PathBuf) -> anyhow::Result<()>;
}
