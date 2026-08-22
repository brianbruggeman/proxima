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

    /// [`crate::loader::prefault`]'s shared background pool failed to build,
    /// or a spawned page-touch chunk never reported back (a worker panic;
    /// `ProximaBackgroundPool` catches and discards worker panics rather
    /// than propagating them).
    #[error("prefault: {0}")]
    PrefaultPoolUnavailable(String),

    /// [`crate::bind::gguf_tensor_as_packed_block`] found `tensor` stored as
    /// `F32` but its absolute file offset is not a multiple of
    /// `align_of::<f32>()` -- reinterpreting the raw bytes as `&[f32]`
    /// without copying would be unsound, so the caller must fall back to
    /// [`crate::bind::gguf_tensor_as_f32`]'s owned, byte-at-a-time decode
    /// instead.
    #[error("tensor {tensor:?} is f32 but its file offset is not 4-byte aligned, cannot borrow as &[f32]")]
    MisalignedFloat32Tensor { tensor: String },

    /// [`crate::bind::architecture_from_metadata`] needed `key` (either
    /// `general.architecture` itself, or one of that architecture's own
    /// `{architecture}.*` dimension keys) and the parsed gguf metadata had
    /// no such key, or the key was present with the wrong `MetadataValue`
    /// variant.
    #[error("gguf metadata is missing required key {key:?}")]
    MissingMetadataKey { key: String },

    /// [`crate::bind::architecture_from_metadata`]'s vocab derivation: the
    /// `token_embd.weight` tensor's element count did not divide evenly by
    /// `embedding_length`.
    #[error("token_embd.weight has {elements} elements, which does not divide evenly by embedding_length {embedding}")]
    VocabShapeMismatch { elements: u64, embedding: u32 },

    /// [`crate::generate`]'s cached forward program failed to build or
    /// evaluate -- propagated from `proxima_tensor` rather than re-derived.
    #[error(transparent)]
    Tensor(#[from] proxima_tensor::TensorError),

    /// [`crate::bind::dequantize_packed_for_metal`] dequantized a packed
    /// `Q5_K`/`Q6_K` block whose name matches none of
    /// [`crate::bind::matmul_weight_dims`]'s known matmul-weight suffixes --
    /// every name that reaches this codec path is one `bind_matmul_weight`
    /// itself bound, so this is a program-construction invariant violation,
    /// not a caller mistake.
    #[cfg(feature = "metal")]
    #[error("packed weight {name:?} has no known matmul out_dim/in_dim to transpose against")]
    UnknownMatmulWeightName { name: String },

    /// [`crate::generate`]'s prompt encode/decode step failed --
    /// propagated from `proxima_tokenizer` rather than re-derived.
    #[cfg(feature = "std")]
    #[error(transparent)]
    Tokenizer(#[from] proxima_tokenizer::TokenizerError),

    /// [`crate::generate::LoadedModel`]'s evaluator ran but `node` (one of
    /// the logits root or a per-layer cache root) is absent from its
    /// output -- an interpreter/program-construction invariant violation
    /// rather than a caller mistake, surfaced instead of panicking.
    #[cfg(feature = "std")]
    #[error("evaluator output is missing node {node:?}")]
    MissingEvaluatedNode { node: proxima_tensor::op::NodeId },

    /// [`crate::generate::LoadedModel`]'s greedy pick step ran against an
    /// empty logits slice.
    #[cfg(feature = "std")]
    #[error("greedy_pick: logits slice is empty")]
    EmptyLogits,

    /// [`crate::generate::LoadedModel`]'s decode loop asked
    /// [`omega::backend`] to plan or execute a forward step and the backend
    /// itself refused -- an unrecognized/uncompiled backend name, or a
    /// codec the chosen backend's driver has no kernel for.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Backend(#[from] omega::backend::BackendError),

    /// [`crate::generate::BackendRuntime::evaluate`]/`evaluate_op_timed`'s
    /// plan cache: `shape` was just inserted (on miss) or already present (on
    /// hit) immediately above, so a subsequent lookup missing it means the
    /// map lost an entry with no external mutation possible under `&mut
    /// self` -- an interpreter/program-construction invariant violation,
    /// surfaced instead of panicking, mirroring [`Self::MissingEvaluatedNode`].
    #[cfg(feature = "metal")]
    #[error("plan cache for shape {shape:?} is missing the entry inserted moments earlier")]
    PlanCacheEntryVanished { shape: (usize, usize) },
}
