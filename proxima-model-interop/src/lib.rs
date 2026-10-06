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
mod lowering;
mod bind;
#[cfg(feature = "std")]
mod bind_leaves;
#[cfg(feature = "std")]
pub mod block_file;
pub mod capability;
#[cfg(feature = "std")]
mod dense;
mod dtype;
mod error;
#[cfg(feature = "std")]
pub mod expert_sidecar;
#[cfg(feature = "std")]
pub mod expert_slab;
#[cfg(feature = "std")]
pub mod sliding_pattern;
#[cfg(feature = "std")]
mod generate;
#[cfg(feature = "std")]
mod hf_bind;
mod hf_config;
#[cfg(feature = "std")]
mod short_conv;
#[cfg(feature = "std")]
mod loader;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod mapping_residency;
#[cfg(feature = "std")]
pub mod recurrent_routed_interval;
#[cfg(feature = "std")]
pub mod residency;
// The arithmetic is pure, but its production consumer is the std+metal
// serving path (including CUDA/Vulkan builds, which inherit the metal feature
// bundle). Keep `cfg(test)` so its unit tests remain available in a bare
// alloc-tier test build without making a plain `--features std` library carry
// an unreachable module under `warnings = "deny"`.
#[cfg(any(test, all(feature = "std", feature = "metal", target_os = "macos")))]
mod memory_fit;
#[cfg(feature = "std")]
mod quality;
#[cfg(feature = "std")]
pub mod profiles;
#[cfg(feature = "std")]
mod recurrent_interval;
pub mod rope_scaling;
mod serving;
mod serving_grammar;
#[cfg(feature = "std")]
mod prompt_cache_settings;
#[cfg(feature = "std")]
mod speculative_settings;
#[cfg(feature = "std")]
mod serving_settings;
#[cfg(all(feature = "std", feature = "proxima-storage"))]
mod source;
pub mod task;
#[cfg(all(test, feature = "std"))]
mod test_support;
mod transform;

#[cfg(feature = "std")]
pub use lowering::{
    BoundProgram, FfnRouting, KvCacheShape, KvLayout, StepInput, StepState, bind_checkpoint,
    bind_checkpoint_with_kv_layout, bind_speculative_verify, bind_symbols, header_descriptor, kv_layers,
    rope_freq_factors, sliding_rope_inputs, step_state, symbols, trained_context_length,
};
#[cfg(feature = "std")]
pub use bind::gguf_tensor_as_packed_block;
#[cfg(feature = "std")]
pub use bind_leaves::bind_program_leaves;
#[cfg(feature = "std")]
pub use bind::{
    BoundWeights, Codec, bind_dense, bind_dense_as, bind_matmul_weight, bind_matmul_weight_as,
    bind_matmul_weight_transposed_f32, bind_moe_expert_weights, bind_native_f32, find_tensor,
    metadata_f32_optional, metadata_str, metadata_str_opt, metadata_u32, metadata_u32_optional_or,
    vocab_from_token_embedding,
};
pub use bind::{ModelHparams, SlidingRope, architecture_from_metadata, gguf_tensor_as_f32};
#[cfg(feature = "std")]
pub use dense::descriptor_from_gguf as dense_descriptor_from_gguf;
pub use dtype::{dtype_to_ggml, ggml_to_dtype};
pub use error::InteropError;
#[cfg(feature = "std")]
pub use expert_sidecar::{
    ExpertSidecar, ExpertSidecarDescriptor, ExpertStackSpec, MappedExpertSidecar,
    write_expert_sidecar,
};
#[cfg(feature = "std")]
pub use expert_slab::{
    ExpertSlab, ExpertSlabMemory, StepGuard, encode_expert_copy, recode_expert_into,
};
#[cfg(feature = "std")]
pub use generate::{
    CachePath, CacheReport, ColdTier, DecodeMetrics, EvictionRule, LoadedModel, MissReason, Phase,
    PrefixState,
    PrewarmReport, PrewarmSkip,
    SpeculativeDecodeStats, SpeculativeTypeStats, TokenEvent,
};
#[cfg(feature = "std")]
pub use hf_bind::{names as hf_names, node_names as hf_node_names, permute_rope_rows};
pub use hf_config::{HfConfig, architecture_from_hf_config, parse_hf_config};
#[cfg(feature = "std")]
pub use short_conv::{
    Lfm2Hparams, lfm2_architecture_from_metadata, lfm2_descriptor, lfm2_forward_values,
};
#[cfg(feature = "std")]
pub use loader::{PREFAULT_OVERSUBSCRIBE, PREFAULT_STRIDE_BYTES, prefault};
#[cfg(all(feature = "std", feature = "instrument"))]
pub use quality::print_quality_report;
#[cfg(feature = "std")]
pub use quality::{Prompt, PromptQuality, QualityReport, parse_prompts_jsonl, quality_report};
#[cfg(feature = "std")]
pub use recurrent_interval::{
    Qwen35Hparams, IntervalLayerKind, SsmShape, bind_qwen35_checkpoint,
    descriptor_from_architecture as qwen35_descriptor_from_architecture, qwen35_architecture_from_metadata,
};
#[cfg(feature = "std")]
pub use residency::{
    ExpertAddress, ExpertPage, ExpertResidency, PrefetchCandidate, PrefetchCandidates,
    ResidencyAction, ResidencyActions, ResidencyConfig, ResidencyError, RoutedExpert,
    ServeDecision, ServePrecision,
};
pub use rope_scaling::RopeScaling;
pub use serving_grammar::{AssembleStep, ReadSpec};
#[cfg(feature = "std")]
pub use serving::GdnPrefillBackend;
pub use serving::{
    ContextLength, DEFAULT_MODEL_PATH, GPU_LAYERS_ALL, NamePattern, NgramMapParams, NgramModParams,
    AttentionConfig, PrefillConfig, PromptCacheConfig, REASONING_BUDGET_UNBOUNDED, ServingConfig, SpeculativeConfig, SpeculativeType,
    SpeculativeTypeSet, WeightPrecisionRule, apply_serving_config, resolve_context_length,
};
#[cfg(feature = "std")]
pub use proxima_core::ServingState;
#[cfg(feature = "std")]
pub use prompt_cache_settings::PromptCacheSettings;
#[cfg(feature = "std")]
pub use speculative_settings::{SpeculativeSettings, SpeculativeTypeName, SpeculativeTypeNameSet};
#[cfg(feature = "std")]
pub use serving_settings::{
    AdmissionScheduleSettings, CacheType, ExpertResidencyScheduleSettings, PhaseScheduleSettings,
    ServingSettings, WeightPrecisionRuleSettings,
};
#[cfg(all(feature = "std", feature = "proxima-storage"))]
pub use source::{CheckpointMapping, CheckpointSourceError};
pub use task::{ModelTask, TaskProfile, classify_task};
pub use transform::{gguf_to_safetensors, safetensors_to_gguf};
