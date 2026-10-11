use proxima_autograd::error::AutogradError;
use proxima_model_interop::InteropError;
use proxima_tensor::TensorError;

#[derive(Debug, thiserror::Error)]
pub enum QatError {
    #[error("config load failed: {0}")]
    Config(String),
    #[error("tensor program failed: {0}")]
    Tensor(#[from] TensorError),
    #[error("symbol binding failed: {0}")]
    Symbols(#[from] InteropError),
    #[error("symbol {symbol} has no bound value among {bound} symbols")]
    UnboundSymbol { symbol: u16, bound: usize },
    #[error("extent {value} does not fit a static extent")]
    ExtentOverflow { value: u64 },
    #[error("logits node has shape {found:?}, expected [{rows}, {vocab}]")]
    LogitsShape {
        found: Vec<u64>,
        rows: u64,
        vocab: u32,
    },
    #[error("autograd failed outside any adjoint rule: {0}")]
    Autograd(#[from] AutogradError),
    #[error("bad argument: {0}")]
    Argument(String),
}
