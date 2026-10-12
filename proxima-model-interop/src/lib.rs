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
#[cfg(test)]
extern crate std;

mod bind;
#[cfg(feature = "std")]
mod bind_leaves;
#[cfg(feature = "std")]
pub mod block_file;
pub mod capability;
#[cfg(feature = "std")]
pub mod cold_artifact;
#[cfg(feature = "std")]
mod dense;
#[cfg(feature = "std")]
mod draft_model_compatibility;
mod dtype;
mod error;
#[cfg(feature = "std")]
pub mod expert_sidecar;
#[cfg(feature = "std")]
pub mod expert_slab;
#[cfg(feature = "std")]
mod generate;
#[cfg(feature = "std")]
mod hf_bind;
mod hf_config;
#[cfg(feature = "std")]
mod loader;
#[cfg(feature = "std")]
mod lowering;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod mapping_residency;
pub mod pinned_source;
#[cfg(feature = "std")]
pub mod recurrent_routed_interval;
#[cfg(feature = "std")]
pub mod residency;
#[cfg(feature = "std")]
mod short_conv;
#[cfg(feature = "std")]
pub mod sliding_pattern;
// The arithmetic is pure, but its production consumer is the std+metal
// serving path (including CUDA/Vulkan builds, which inherit the metal feature
// bundle). Keep `cfg(test)` so its unit tests remain available in a bare
// alloc-tier test build without making a plain `--features std` library carry
// an unreachable module under `warnings = "deny"`.
#[cfg(feature = "std")]
mod codec_policy_settings;
#[cfg(feature = "std")]
pub mod convert;
#[cfg(any(test, all(feature = "std", feature = "metal", target_os = "macos")))]
mod memory_fit;
#[cfg(feature = "std")]
pub mod profiles;
#[cfg(feature = "std")]
mod prompt_cache_settings;
mod recurrent_config;
#[cfg(feature = "std")]
mod quality;
#[cfg(feature = "std")]
mod recurrent_interval;
#[cfg(feature = "std")]
mod recurrent_settings;
pub mod rope_scaling;
mod serving;
mod serving_grammar;
#[cfg(feature = "std")]
mod serving_settings;
#[cfg(all(feature = "std", feature = "proxima-storage"))]
mod source;
#[cfg(feature = "std")]
mod speculative_settings;
pub mod task;
#[cfg(all(test, feature = "std"))]
mod test_support;
mod transform;

#[cfg(feature = "std")]
pub use bind::gguf_tensor_as_packed_block;
#[cfg(feature = "std")]
pub use bind::{
    BoundWeights, Codec, bind_dense, bind_dense_as, bind_matmul_weight, bind_matmul_weight_as,
    bind_matmul_weight_transposed_f32, bind_moe_expert_weights, bind_native_f32, find_tensor,
    metadata_f32_optional, metadata_str, metadata_str_opt, metadata_u32, metadata_u32_optional_or,
    vocab_from_token_embedding,
};
pub use bind::{ModelHparams, SlidingRope, architecture_from_metadata, gguf_tensor_as_f32};
#[cfg(feature = "std")]
pub use bind_leaves::bind_program_leaves;
#[cfg(feature = "std")]
pub use codec_policy_settings::{CodecPolicySettings, CodecPolicyWeights};
#[cfg(feature = "std")]
pub use convert::{convert_codec, policy_selects};
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
    PrefixState, PrewarmReport, PrewarmSkip, SpeculativeDecodeStats, SpeculativeTypeStats,
    TokenEvent,
};
#[cfg(feature = "std")]
pub use hf_bind::{
    HfShard, SafetensorsTensorView, bind_safetensors_program_leaves,
    bind_safetensors_program_views, bind_safetensors_shards, names as hf_names,
    node_names as hf_node_names, permute_rope_rows, refuse_mlx_quantized,
    refuse_mlx_quantized_header,
};
pub use hf_config::{
    HfConfig, architecture_from_hf_config, lfm_layer_kinds_from_manifest, parse_hf_config,
};
#[cfg(feature = "std")]
pub use loader::{PREFAULT_OVERSUBSCRIBE, PREFAULT_STRIDE_BYTES, prefault};
#[cfg(feature = "std")]
pub use lowering::{
    BoundProgram, FfnRouting, KvCacheShape, KvLayout, StepInput, StepState, bind_checkpoint,
    bind_checkpoint_with_kv_layout, bind_hf_checkpoint, bind_speculative_verify, bind_symbols,
    header_descriptor, kv_layers, recurrent_speculative_verify, rope_freq_factors,
    sliding_rope_inputs, step_state, symbols,
    trained_context_length,
};
#[cfg(feature = "std")]
pub use prompt_cache_settings::PromptCacheSettings;
pub use recurrent_config::RecurrentConfig;
#[cfg(feature = "std")]
pub use recurrent_settings::RecurrentSettings;
#[cfg(feature = "std")]
pub use proxima_core::ServingState;
#[cfg(all(feature = "std", feature = "instrument"))]
pub use quality::print_quality_report;
#[cfg(feature = "std")]
pub use quality::{Prompt, PromptQuality, QualityReport, parse_prompts_jsonl, quality_report};
#[cfg(feature = "std")]
pub use recurrent_interval::{
    IntervalLayerKind, RecurrentIntervalHparams, SsmShape, bind_recurrent_interval_checkpoint,
    descriptor_from_architecture as recurrent_interval_descriptor_from_architecture,
    recurrent_interval_architecture_from_metadata,
};
#[cfg(feature = "std")]
pub use residency::{
    ExpertAddress, ExpertPage, ExpertResidency, PrefetchCandidate, PrefetchCandidates,
    ResidencyAction, ResidencyActions, ResidencyConfig, ResidencyError, RoutedExpert,
    ServeDecision, ServePrecision,
};
pub use rope_scaling::RopeScaling;
#[cfg(feature = "std")]
pub use serving::GdnPrefillBackend;
pub use serving::{
    AttentionConfig, ContextLength, DEFAULT_BATCH_SIZE, DEFAULT_MODEL_PATH, DEFAULT_UBATCH_SIZE,
    GPU_LAYERS_ALL, NamePattern, NgramMapParams, NgramModParams, PrefillConfig, PromptCacheConfig,
    REASONING_BUDGET_UNBOUNDED, ServingConfig, SpeculativeConfig, SpeculativeType,
    SpeculativeTypeSet, WeightPrecisionRule, apply_serving_config, resolve_context_length,
};
pub use serving_grammar::{AssembleStep, ReadSpec};
#[cfg(all(
    feature = "std",
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
pub use serving_settings::AttentionTileHeightSetting;
#[cfg(feature = "std")]
pub use serving_settings::{
    AdmissionScheduleSettings, CacheType, ExpertResidencyScheduleSettings, PhaseScheduleSettings,
    ServingSettings, WeightPrecisionRuleSettings,
};
#[cfg(feature = "std")]
pub use short_conv::{
    ShortConvHparams, short_conv_architecture_from_metadata, short_conv_descriptor,
    short_conv_forward_values,
};
#[cfg(all(feature = "std", feature = "proxima-storage"))]
pub use source::{CheckpointMapping, CheckpointSourceError};
#[cfg(feature = "std")]
pub use speculative_settings::{SpeculativeSettings, SpeculativeTypeName, SpeculativeTypeNameSet};
pub use task::{ModelTask, TaskProfile, classify_task};
pub use transform::{
    AffineExport, AffinePacked, AffineParams, MLX_INDEX_FILE_NAME, MLX_MAX_SHARD_BYTES,
    MLX_SINGLE_SHARD_FILE_NAME, MlxExport, MlxShard, export_affine, gguf_to_safetensors,
    pack_affine, safetensors_to_gguf,
};
#[cfg(feature = "std")]
pub use transform::{
    DEFAULT_PLAIN_DTYPE, PlainTensor, export_plain, export_plain_named, export_plain_tensor,
};
#[cfg(feature = "std")]
pub use transform::{Layout, MlxNames, SANDWICH_NORM_LAYOUT, STANDARD_LAYOUT, dense_names};
#[cfg(feature = "std")]
pub use transform::{
    QuantizedConventions, copy_config, hf_config_bytes, hf_config_bytes_with_overrides,
    sliding_pattern_hparams,
};
#[cfg(feature = "std")]
pub use transform::{TokenizerFiles, copy_tokenizer, export_tokenizer};
#[cfg(feature = "std")]
pub use transform::{export_affine_named, export_mlx_directory, mlx_directory_files};
