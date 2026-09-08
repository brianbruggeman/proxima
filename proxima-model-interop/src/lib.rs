//! GGUF <-> safetensors interop: a sans-IO transform over the one thing
//! both model-weight formats agree on — a named tensor is `(name, dtype,
//! shape, bytes)`. Everything beyond that (GGUF's typed KV metadata vs.
//! safetensors' flat string map, GGUF's block-quantized types vs.
//! safetensors' flat typed arrays) is where the two formats diverge; see
//! [`transform::gguf_to_safetensors`] and [`transform::safetensors_to_gguf`]
//! for exactly what each direction preserves and what it doesn't.
//!
//! `generate::LoadedModel` (`std`-gated) is this crate's other reachable capability:
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

#[cfg(feature = "std")]
mod architecture;
mod bind;
pub mod capability;
#[cfg(feature = "std")]
mod dense;
mod dtype;
mod error;
#[cfg(feature = "std")]
mod generate;
#[cfg(feature = "std")]
mod hf_bind;
mod hf_config;
#[cfg(feature = "std")]
mod lfm2;
#[cfg(feature = "std")]
mod loader;
// no `feature = "std"` gate: the module is pure alloc/core arithmetic
// (its own doc), but every consumer -- `generate.rs`'s
// `checkpoint_weight_bytes` field and its metal-gated `apply_memory_fit_gate`
// -- is metal-only, so a plain `--features std` build with no metal has no
// call site at all and every pub item reads as dead code under
// `warnings = "deny"`. `cfg(test)` keeps the module (and its own unit
// tests, which exercise every item directly) compiled under every feature
// set nextest runs.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
mod memory_fit;
#[cfg(feature = "std")]
mod qwen35;
#[cfg(feature = "std")]
mod quality;
mod serving;
#[cfg(all(test, feature = "std"))]
mod test_support;
mod transform;

#[cfg(feature = "std")]
pub use architecture::{Architecture, ArchitectureRegistry, BoundProgram, StepState};
#[cfg(feature = "std")]
pub use bind::gguf_tensor_as_packed_block;
pub use bind::{ModelArchitecture, architecture_from_metadata, gguf_tensor_as_f32};
#[cfg(feature = "std")]
pub use bind::{
    BoundWeights, PackedOwnedKind, bind_dense, bind_dense_as, bind_matmul_weight,
    bind_matmul_weight_as, find_tensor, metadata_f32_optional, metadata_str, metadata_str_opt,
    metadata_u32, metadata_u32_optional_or, vocab_from_token_embedding,
};
#[cfg(feature = "std")]
pub use dense::DenseArch;
#[cfg(feature = "std")]
pub use hf_bind::{names as hf_names, node_names as hf_node_names, permute_rope_rows};
pub use dtype::{dtype_to_ggml, ggml_to_dtype};
pub use error::InteropError;
#[cfg(feature = "std")]
pub use generate::{Control, LoadedModel, Phase, PrefixState, TokenEvent};
pub use hf_config::{HfConfig, architecture_from_hf_config, parse_hf_config};
#[cfg(feature = "std")]
pub use lfm2::{
    Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values, run_lfm2_prefill,
};
#[cfg(feature = "std")]
pub use loader::{PREFAULT_OVERSUBSCRIBE, PREFAULT_STRIDE_BYTES, prefault};
#[cfg(feature = "std")]
pub use qwen35::{
    Qwen35Architecture, Qwen35Arch, Qwen35LayerKind, Qwen35SsmShape, bind_qwen35_checkpoint,
};
#[cfg(feature = "std")]
pub use quality::{Prompt, PromptQuality, QualityReport, parse_prompts_jsonl, quality_report};
#[cfg(all(feature = "std", feature = "instrument"))]
pub use quality::print_quality_report;
pub use serving::{
    DEFAULT_MODEL_PATH, GPU_LAYERS_ALL, NamePattern, REASONING_BUDGET_UNBOUNDED, ServingConfig,
    WeightPrecisionRule, apply_serving_config,
};
pub use transform::{gguf_to_safetensors, safetensors_to_gguf};
