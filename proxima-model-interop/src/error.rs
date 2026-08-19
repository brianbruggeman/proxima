//! Typed failures for the interop transform, on top of whatever the
//! underlying reader/writer surfaces.

use alloc::string::String;

use proxima_gguf::GgmlType;
use proxima_tensor::DType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InteropError {
    /// `tensor`'s `GgmlType` has no safetensors dtype counterpart — either
    /// it's block-quantized (packs multiple elements per scale/bias, which
    /// a flat typed array can't express without dequantizing) or otherwise
    /// has no fixed-width scalar equivalent.
    #[error(
        "tensor {tensor:?} has ggml type {ggml_type:?}, which has no safetensors dtype counterpart"
    )]
    UnrepresentableGgmlType { tensor: String, ggml_type: GgmlType },

    /// `tensor`'s `DType` has no `GgmlType` counterpart (`Bool` or an
    /// unsigned-integer / 128-bit width ggml never defined).
    #[error("tensor {tensor:?} has dtype {dtype:?}, which has no ggml type counterpart")]
    UnrepresentableDType { tensor: String, dtype: DType },

    /// `tensor`'s shape has more dimensions than GGUF's tensor directory
    /// can hold (`proxima_gguf::tensor::MAX_DIMS`, 4).
    #[error("tensor {tensor:?} has {found} dimensions, gguf supports at most {max}")]
    TooManyDimensions {
        tensor: String,
        found: usize,
        max: usize,
    },

    /// [`crate::bind::gguf_tensor_as_f32`] was asked for a name absent
    /// from the parsed tensor directory.
    #[error("no tensor named {name:?} in the gguf tensor directory")]
    UnknownTensor { name: String },

    #[error(transparent)]
    Gguf(#[from] proxima_gguf::GgufError),

    #[error(transparent)]
    Safetensors(#[from] proxima_safetensors::SafetensorsError),

    /// A block-quantized tensor's bytes didn't fit its codec's own shape
    /// contract (not a whole block multiple, or an output-size mismatch)
    /// -- propagated from [`proxima_gguf::quant`] rather than re-derived.
    #[error(transparent)]
    Quant(#[from] proxima_gguf::quant::QuantError),
}
