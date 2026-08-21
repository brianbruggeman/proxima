//! GGUF <-> safetensors interop: a sans-IO transform over the one thing
//! both model-weight formats agree on — a named tensor is `(name, dtype,
//! shape, bytes)`. Everything beyond that (GGUF's typed KV metadata vs.
//! safetensors' flat string map, GGUF's block-quantized types vs.
//! safetensors' flat typed arrays) is where the two formats diverge; see
//! [`transform::gguf_to_safetensors`] and [`transform::safetensors_to_gguf`]
//! for exactly what each direction preserves and what it doesn't.
//!
//! [`generate::LoadedModel`] is this crate's other reachable capability:
//! bind a checkpoint's weights once, then run greedy-decode text
//! generation against them through `proxima_primitives::pipe::Pipe` --
//! see that module's own doc for the load-once/generate-repeatedly shape.
//!
//! ONNX is out of scope here: it carries a computation graph, not just
//! named tensors, so an ONNX leg of this transform is a different, larger
//! job (serializing graph structure) than this crate does.
//!
//! Lives as its own crate rather than a feature-gated module on either
//! `proxima-gguf` or `proxima-safetensors` because the dependency is
//! inherently bidirectional — either format crate depending on the other
//! would be an arbitrary direction to pick, and neither reader/writer
//! needs to know the other format exists to do its own job.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod bind;
mod dtype;
mod error;
#[cfg(feature = "std")]
mod generate;
#[cfg(feature = "std")]
mod loader;
mod serving;
mod transform;

pub use bind::{ModelArchitecture, architecture_from_metadata, gguf_tensor_as_f32};
#[cfg(feature = "std")]
pub use bind::gguf_tensor_as_packed_block;
pub use dtype::{dtype_to_ggml, ggml_to_dtype};
pub use error::InteropError;
#[cfg(feature = "std")]
pub use generate::LoadedModel;
#[cfg(feature = "std")]
pub use loader::{PREFAULT_OVERSUBSCRIBE, PREFAULT_STRIDE_BYTES, prefault};
pub use serving::{
    DEFAULT_MODEL_PATH, GPU_LAYERS_ALL, REASONING_BUDGET_UNBOUNDED, ServingConfig, apply_serving_config,
};
pub use transform::{gguf_to_safetensors, safetensors_to_gguf};
