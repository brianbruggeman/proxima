//! What a [`super::prompt_cache::CacheEntry`] was built under. Token ids say
//! which prefix an entry covers; this says whether the rows behind those ids
//! are the rows the next request would have computed itself.
//!
//! [`CacheKey::of`] destructures [`ServingConfig`] with no `..`, so a field
//! added to the config does not compile until someone decides whether it
//! changes the cached rows (and goes in the key) or not (and is named `_`
//! with the reason). Compare with [`LoadedModel::prefill_prefix`] +
//! [`LoadedModel::generate_from_prefix`], where the caller owns the one state
//! and so owns the guarantee that the config did not move between the calls.
//!
//! Fixed for the life of a [`LoadedModel`], so not stored per entry: the
//! weights and program (`load_inner` builds them once; no method takes
//! `&mut self` to replace them), the KV layout (`kv_layers`, set at load) and
//! the checkpoint's own rope scaling (only read through
//! [`LoadedModel::effective_rope_scaling`], which is in the key). The one
//! model-level input that can move is the expert sidecar
//! ([`LoadedModel::attach_expert_sidecar`] takes `&mut self`), which clears the
//! cache instead.

#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::MathMode;
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
use omega::AttentionVariant;
use proxima_tensor::NumericPolicy;

use crate::rope_scaling::RopeScaling;
use crate::serving::{AttentionConfig, GdnPrefillBackend, ServingConfig};
use crate::serving_grammar::ReadSpec;

/// The inputs that change the bytes of a cached KV/state, or how a resumed
/// forward reads them. A plain comparable struct: two requests share an entry
/// only when every field is equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CacheKey {
    /// The scaling the call runs ([`LoadedModel::effective_rope_scaling`]):
    /// it sets the cos/sin every key row is rotated by before it is stored.
    pub(super) rope_scaling: RopeScaling,
    /// Whether the Metal engine ran the forward; the CPU and Metal kernels
    /// round differently, so their rows are not interchangeable.
    pub(super) gpu_route: bool,
    /// Which bind-time rewrites fire and how kernels reduce.
    pub(super) numeric_policy: NumericPolicy,
    /// Metal kernel math mode, the narrower projection of `numeric_policy`
    /// that is applied to the compiled plans.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) math_mode: MathMode,
    /// Cached-attention selection changes the rows computed for this prefix.
    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    pub(super) attention_variant: Option<AttentionVariant>,
    /// CPU quantized dots: exact dequantize-then-fold or int8 fast path.
    pub(super) exact_activations: bool,
    /// Fused cached-attention kernel instead of the unfused graph.
    pub(super) cached_attention_fusion: bool,
    /// Fused gated-delta-net kernel; its recurrent state is carried in the
    /// entry and an extend resumes from it.
    pub(super) gated_delta_net_fusion: bool,
    /// Fused MoE top-k kernel.
    pub(super) moe_topk_fusion: bool,
    /// Which backend runs the gated-delta-net prefill (recurrent state bytes).
    pub(super) gdn_prefill_backend: GdnPrefillBackend,
    /// Router / residency / gather as separate phases instead of one graph.
    pub(super) moe_pre_gather: bool,
    /// All experts at the low codec.
    pub(super) moe_monolithic_all_low: bool,
    /// Original mmap-backed expert stacks in one graph.
    pub(super) moe_monolithic_high_mmap: bool,
    /// Adjacent layers executed in one sidecar window.
    pub(super) moe_layer_window: usize,
    /// Size of the high-precision expert pool, which decides which experts
    /// run at the low codec.
    pub(super) moe_residency_budget_bytes: u64,
    /// Rows each sliding ring keeps past its window: a ring built with less
    /// slack than a request's speculative verify writes would evict rows the
    /// verify step needs.
    pub(super) ring_slack_rows: usize,
    /// Rows every sliding ring writes off its true slot
    /// ([`LoadedModel::with_ring_write_offset_for_parity_control`]); the
    /// builder consumes the model, so entries stored before it ran can exist.
    pub(super) ring_write_offset: usize,
    /// Which cached rows a decode step reads; rows computed under a skipped
    /// read are not the rows a dense read computes.
    pub(super) read: ReadSpec,
}

impl CacheKey {
    /// `uses_gpu` is the engine the request's `BackendRuntime` selected,
    /// `rope_scaling` the resolved [`LoadedModel::effective_rope_scaling`],
    /// `ring_slack_rows` [`super::kv_ring::ring_slack_rows`] and
    /// `ring_write_offset` the model's own.
    pub(super) fn of(
        config: &ServingConfig,
        uses_gpu: bool,
        rope_scaling: RopeScaling,
        ring_slack_rows: usize,
        ring_write_offset: usize,
    ) -> Self {
        let ServingConfig {
            // identity of the weights: the entry lives on one LoadedModel
            model_path: _,
            // resolved length only bounds admission and the memory fit; no
            // row depends on it (rope enters through `rope_scaling`)
            context_length: _,
            rope_scaling: _,
            // admission rejects every value but 1
            parallel_sequences: _,
            // admission rejects every value but F32
            kv_cache_key_quant: _,
            kv_cache_value_quant: _,
            // admission rejects true
            flash_attention: _,
            // chunk widths: a resume already starts its prefill at an
            // arbitrary lcp, so the chunk boundaries vary within one config
            batch_size: _,
            ubatch_size: _,
            prefill_one_evaluation: _,
            prefill_chunk_positions: _,
            // admission admits 0 and ALL; the engine actually selected is
            // `uses_gpu`
            gpu_layers: _,
            // memory-fit and budget gates refuse or shrink; no row changes
            gpu_memory_fit: _,
            gpu_memory_limit_bytes: _,
            dense_weights_budget_bytes: _,
            expert_weights_budget_bytes: _,
            activations_budget_bytes: _,
            kv_cache_budget_bytes: _,
            // admission rejects true
            kv_offload: _,
            multimodal_projector: _,
            // admission rejects every value but 0
            reasoning_budget: _,
            // sampling picks which token comes next; the entry stores the
            // ids that were actually forwarded
            temperature: _,
            top_k: _,
            top_p: _,
            min_p: _,
            repeat_last_n: _,
            repeat_penalty: _,
            frequency_penalty: _,
            presence_penalty: _,
            seed: _,
            // pads the masked tail of the key extent; the extent already
            // varies with prompt length within one config
            kv_bucket_tokens: _,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode,
            #[cfg(all(
                feature = "metal",
                feature = "metal-attn-variants",
                target_os = "macos"
            ))]
            attention_variant,
            numeric_policy,
            // serial vs concurrent encoder: scheduling, same arithmetic
            #[cfg(all(feature = "metal", target_os = "macos"))]
                dispatch_type: _,
            exact_activations,
            // bind-time recode is applied once at load; the decode path
            // never reads this field
            weight_precision: _,
            moe_pre_gather,
            // keeps cut tensors in device buffers: placement, same values
            moe_persistent_cuts: _,
            gdn_prefill_backend,
            moe_residency_budget_bytes,
            // route-history advice for prefetching: timing only
            moe_expert_prefetch: _,
            moe_monolithic_all_low,
            moe_layer_window,
            moe_monolithic_high_mmap,
            // only read under the wgpu driver, which this crate never links
            gpu_correctness_fallback: _,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            // resident vs re-dispatched constants: placement, same values
            plan_time_constants: _,
            // refit vs rebuild at a bucket crossing: the same ops either way
            plan_refit: _,
            // how one step's dispatches split across command buffers
            command_buffer_chunks: _,
            max_command_buffers_per_token: _,
            // device memory held for plans between calls: never changes rows
            resident_prefill_plan_bytes: _,
            overlap_transfer_compute: _,
            warm_model_buffers_at_load: _,
            // request ordering and residency scheduling levels
            admission_schedule: _,
            phase_schedule: _,
            expert_residency_schedule: _,
            // draft width reaches the key as `ring_slack_rows`
            speculative: _,
            // the cache's own policy; slack reaches the key as
            // `ring_slack_rows`
            prompt_cache: _,
            // stages choose where rows come from; a stage that changes what a row means adds its identity to the key
            prefill: _,
            attention: AttentionConfig { read },
        } = *config;
        Self {
            rope_scaling,
            gpu_route: uses_gpu,
            numeric_policy,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode,
            #[cfg(all(
                feature = "metal",
                feature = "metal-attn-variants",
                target_os = "macos"
            ))]
            attention_variant,
            exact_activations,
            cached_attention_fusion,
            gated_delta_net_fusion,
            moe_topk_fusion,
            gdn_prefill_backend,
            moe_pre_gather,
            moe_monolithic_all_low,
            moe_monolithic_high_mmap,
            moe_layer_window,
            moe_residency_budget_bytes,
            ring_slack_rows,
            ring_write_offset,
            read,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::serving::PrefillConfig;
    use crate::serving_grammar::AssembleStep;

    #[test]
    fn cache_key_separates_read_specs() {
        let dense = ServingConfig::default();
        let operand = ServingConfig {
            attention: AttentionConfig { read: ReadSpec::Operand },
            ..dense
        };

        let dense_key = CacheKey::of(&dense, false, RopeScaling::None, 0, 0);
        let operand_key = CacheKey::of(&operand, false, RopeScaling::None, 0, 0);
        let default_key = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0);

        assert_ne!(dense_key, operand_key);
        assert_eq!(dense_key, default_key);
    }

    #[test]
    fn cache_key_ignores_the_assemble_list() {
        let steps = [AssembleStep::Prefix, AssembleStep::Shift];
        let config = ServingConfig {
            prefill: PrefillConfig { assemble: &steps },
            ..ServingConfig::default()
        };

        let staged_key = CacheKey::of(&config, false, RopeScaling::None, 0, 0);
        let default_key = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0);

        assert_eq!(staged_key, default_key);
    }
}
