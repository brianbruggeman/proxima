use super::*;

/// Which forward-program engine [`ModelDescriptor::cache_strategy`] selects
/// for a (future) generic `build_forward` -- [`CacheStrategy::Cacheless`] is
/// [`lfm2_forward_program_with_experts`], today's default path (full
/// reprefill every call); [`CacheStrategy::TwoRange`] is
/// [`lfm2_two_range_cached_forward_program_with_experts`], the WORKING
/// gemma4 two-range kv-cache behind the default-off `gemma4-kv-cache`
/// feature. Genuinely new: no existing type names which forward-program
/// engine a schedule targets -- that choice lives today as a
/// `#[cfg(feature = "gemma4-kv-cache")]` compile-time split
/// (`proxima-model-interop::gemma4::bind::Gemma4Arch::bind`), not as data a
/// caller can hold and branch on at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    Cacheless,
    TwoRange,
    /// [`mistral_cached_forward_program_with_experts_and_layer_taps`]'s own
    /// single-continuous-range KV cache (`kv_cache.{layer}.k_even`/`k_odd`/
    /// `v`) -- a genuinely different scoring algebra from both other
    /// variants: the cached block is NEVER masked (a cached position is
    /// definitionally in the past of every new query, per that function's
    /// own doc on its `is_future` usage), where [`Cacheless`]'s block-local
    /// mask has no cache to skip and [`TwoRange`]'s own single-range
    /// counterpart -- [`lfm2_single_range_cached_forward_program_with_experts`],
    /// the plausible reuse candidate this variant's own doc first
    /// considered -- masks the cached block with a `cached_len`-aware
    /// [`causal_mask_merged_windowed`] to exclude stale KV-bucket padding.
    /// Mistral's cache carries no such padding, so no exclusion mask exists
    /// in its program at all; threading that engine's own masking as a
    /// descriptor knob would mean rewriting its cache algebra, not adding a
    /// parameter. [`build_forward`]'s own arm therefore calls
    /// [`mistral_cached_forward_program_with_experts_and_layer_taps`]
    /// directly -- the SAME "dispatch to an existing, unmodified builder"
    /// shape [`TwoRange`]'s own arm already uses.
    SingleRange,
}

/// A whole model's build-time shape as DATA: the global hyperparameters
/// [`lfm2_forward_program_with_experts`] already takes as loose positional
/// arguments (`vocab`, `embedding`, `block_count`, `expert_count`,
/// `expert_used_count`, `leading_dense_block_count`, `embedding_scale`,
/// `logit_softcap`), one [`LayerSchedule`] per block, and which cache engine
/// to build with. `layers` reuses [`LayerSchedule`] verbatim -- it already
/// composes [`LayerKind`], [`LayerAttentionConfig`], and [`LayerFfnConfig`]
/// (`proxima-model-interop::gemma4::bind::gemma4_layer_schedule` builds
/// exactly this shape today, by hand, at bind time), so this struct does not
/// re-mint a per-layer type. Genuinely new: no existing type bundles a
/// model's full hyperparameter set with its per-layer schedule and a
/// cache-engine choice into one value a caller can build ahead of time --
/// today every caller re-derives and re-passes these as separate positional
/// arguments at each forward-program call site.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDescriptor {
    pub vocab: u32,
    pub embedding: u32,
    /// Dense-branch FFN hidden width (`append_lfm2_layer_ffn`'s own
    /// `feed_forward`) -- every layer's dense-FFN weight shapes derive from
    /// this, same as `expert_feed_forward` does for the routed branch.
    pub feed_forward: u32,
    /// Routed-expert FFN hidden width, distinct from `feed_forward` because
    /// gemma4's dense and routed branches run at different widths
    /// (`append_lfm2_layer_ffn`'s own `expert_feed_forward` parameter).
    pub expert_feed_forward: u32,
    /// Query head count, shared by every layer's attention sub-block
    /// (`build_attention_layer_resources`'s own `query_heads` parameter) --
    /// distinct from `LayerAttentionConfig::kv_heads`, which is per-layer.
    pub query_heads: u32,
    pub block_count: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub leading_dense_block_count: u32,
    /// Short-conv kernel width, consulted only by a [`LayerKind::ShortConv`]
    /// entry's `append_lfm2_conv_mixer` call
    /// (`proxima-tensor/src/spec/attention_forward.rs`'s own `l_cache`
    /// parameter doc) -- unused and safe to leave at any value when
    /// `layers` holds no `ShortConv` entry, as every gemma4 layer is
    /// [`LayerKind::Attention`].
    pub l_cache: u32,
    pub embedding_scale: Option<EmbeddingScale>,
    pub logit_softcap: Option<f32>,
    pub layers: Vec<LayerSchedule>,
    pub cache_strategy: CacheStrategy,
    /// Qwen3-style per-head QK-norm, consulted ONLY by
    /// [`CacheStrategy::SingleRange`]'s arm
    /// (`mistral_cached_forward_program_with_experts_and_layer_taps`'s own
    /// `qk_norm` parameter) -- inert, safe at any value, under
    /// [`CacheStrategy::Cacheless`]/[`CacheStrategy::TwoRange`], same
    /// "unused when the arm never reads it" precedent [`Self::l_cache`]'s
    /// own doc already set. A model-global flag, not a per-layer
    /// [`LayerAttentionConfig`] field, because that builder's own signature
    /// takes it as one flat `bool` applied uniformly to every layer.
    pub qk_norm: bool,
    /// `attn_{q,k,v}.bias` presence, consulted ONLY by
    /// [`CacheStrategy::SingleRange`]'s arm -- same inertness and
    /// model-global-not-per-layer reasoning as [`Self::qk_norm`].
    pub qkv_biases: bool,
    /// Single fused `[2, feed_forward, embedding]` gate+up weight leaf vs.
    /// two separate leaves, consulted ONLY by [`CacheStrategy::SingleRange`]'s
    /// arm -- same inertness and model-global-not-per-layer reasoning as
    /// [`Self::qk_norm`].
    pub paired_gate_up_reduce: bool,
    /// Single fused `[query_heads + 2*kv_heads, head_dim, embedding]` QKV
    /// weight leaf vs. three separate leaves, consulted ONLY by
    /// [`CacheStrategy::SingleRange`]'s arm -- same inertness and
    /// model-global-not-per-layer reasoning as [`Self::qk_norm`].
    pub fused_qkv_reduce: bool,
}

/// Real gemma4 26B-A4B header values, proven by the `#[ignore]`d
/// `builtin_registry_routes_real_gemma4_header_with_exact_tensor_directory`
/// (`proxima-model-interop/tests/real_gemma4_registry_probe.rs`), which
/// asserts each of these against the real checkpoint's own GGUF metadata.
const GEMMA4_BLOCK_COUNT: u32 = 30;
const GEMMA4_EXPERT_COUNT: u32 = 128;
const GEMMA4_EXPERT_USED_COUNT: u32 = 8;
const GEMMA4_EMBEDDING: u32 = 2816;
const GEMMA4_LOGIT_SOFTCAP: f32 = 30.0;
const GEMMA4_FEED_FORWARD: u32 = 2112;
const GEMMA4_EXPERT_FEED_FORWARD: u32 = 704;
const GEMMA4_QUERY_HEADS: u32 = 16;
/// `Gemma4Arch::bind`'s own cacheless `lfm2_forward_program_with_experts`
/// call site (`proxima-model-interop/src/gemma4/bind.rs`) passes `0` here
/// too, and its `gemma4-kv-cache` sibling now builds its program from this
/// same constant via `gemma4_descriptor` -- consulted only by a
/// `LayerKind::ShortConv` entry, and gemma4 has none.
const GEMMA4_L_CACHE: u32 = 0;
/// Full-attention head dim (`architecture.key_length`); sliding layers use
/// [`GEMMA4_HEAD_DIM_SWA`] instead.
const GEMMA4_HEAD_DIM_FULL: u32 = 512;
const GEMMA4_HEAD_DIM_SWA: u32 = 256;
const GEMMA4_KV_HEADS_FULL: u32 = 2;
const GEMMA4_KV_HEADS_SWA: u32 = 8;
const GEMMA4_SLIDING_WINDOW: u32 = 1024;
/// The real header marks every 6th layer (0-indexed 5, 11, 17, 23, 29) full
/// attention, every other layer sliding -- `(layer + 1).is_multiple_of(6)`
/// mirrors the real registry probe's own `expected_sliding_window_pattern`.
const GEMMA4_FULL_LAYER_PERIOD: u32 = 6;
/// `Gemma4Arch::bind`'s own two call sites into
/// `lfm2_forward_program_with_experts`/`lfm2_two_range_cached_forward_program_with_experts`
/// (`proxima-model-interop/src/gemma4/bind.rs`) both pass `0` here: every
/// gemma4 layer runs the parallel dense+MoE combination
/// ([`FfnCombination::ParallelDenseMoe`]), so there is no leading run of
/// dense-only blocks to select past.
const GEMMA4_LEADING_DENSE_BLOCK_COUNT: u32 = 0;

/// Builds the real gemma4 26B-A4B [`ModelDescriptor`] -- SWA/full dual-base
/// RoPE, unscaled attention score, value-norm, shared-KV on full layers,
/// parallel dense+MoE FFN, final-logit softcap 30. Every field this
/// function sets mirrors
/// `proxima-model-interop::gemma4::bind::gemma4_layer_schedule` node-for-node
/// (same [`LayerAttentionConfig`]/[`LayerFfnConfig`] values, same
/// sliding-vs-full split), just built here as plain data instead of at GGUF
/// bind time. `vocab` is a parameter rather than a baked-in constant because
/// it is genuinely per-checkpoint data (`hparams::Architecture::vocab` reads
/// it from the `token_embd.weight` tensor's own row count, not from a fixed
/// architecture metadata field) -- every other value here is architecture,
/// not tokenizer, and is proven fixed by the real registry probe cited on
/// each constant above.
#[must_use]
pub fn gemma4_descriptor(vocab: u32) -> ModelDescriptor {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        dense_feed_forward: None,
        exclusive_dense_post_norm: false,
        activation: Activation::GeluTanh,
        ple: false,
    };

    let layers: Vec<LayerSchedule> = (0..GEMMA4_BLOCK_COUNT)
        .map(|layer| {
            let is_full = (layer + 1).is_multiple_of(GEMMA4_FULL_LAYER_PERIOD);
            let attention = if is_full {
                LayerAttentionConfig {
                    head_dim: GEMMA4_HEAD_DIM_FULL,
                    kv_heads: GEMMA4_KV_HEADS_FULL,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    key_source_kind: KeySourceKind::ProjectedK,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos",
                        sin_name: "rope_sin",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: GEMMA4_HEAD_DIM_FULL / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            } else {
                LayerAttentionConfig {
                    head_dim: GEMMA4_HEAD_DIM_SWA,
                    kv_heads: GEMMA4_KV_HEADS_SWA,
                    mask_window: Some(GEMMA4_SLIDING_WINDOW),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    key_source_kind: KeySourceKind::ProjectedK,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa",
                        sin_name: "rope_sin_swa",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: GEMMA4_HEAD_DIM_SWA / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn,
            }
        })
        .collect();

    ModelDescriptor {
        vocab,
        embedding: GEMMA4_EMBEDDING,
        feed_forward: GEMMA4_FEED_FORWARD,
        expert_feed_forward: GEMMA4_EXPERT_FEED_FORWARD,
        query_heads: GEMMA4_QUERY_HEADS,
        block_count: GEMMA4_BLOCK_COUNT,
        expert_count: GEMMA4_EXPERT_COUNT,
        expert_used_count: GEMMA4_EXPERT_USED_COUNT,
        leading_dense_block_count: GEMMA4_LEADING_DENSE_BLOCK_COUNT,
        l_cache: GEMMA4_L_CACHE,
        embedding_scale: Some(EmbeddingScale::Sqrt),
        logit_softcap: Some(GEMMA4_LOGIT_SOFTCAP),
        layers,
        cache_strategy: CacheStrategy::Cacheless,
        // inert: neither `Cacheless` nor `TwoRange` ever reads these four
        // fields (`ModelDescriptor::qk_norm`'s own doc) -- gemma4 has no
        // concept of any of them.
        qk_norm: false,
        qkv_biases: false,
        paired_gate_up_reduce: false,
        fused_qkv_reduce: false,
    }
}

/// Real openchat-3.5-1210 / Mistral-7B-v0.1 header shape, proven by
/// `single_range_cached_attention_fuses_one_step_per_layer_on_the_real_openchat_shape`
/// (`proxima-tensor/src/bind/tests.rs`), which binds this exact shape
/// against the real `openchat-3.5-1210.Q4_K_S.gguf` checkpoint.
const MISTRAL_EMBEDDING: u32 = 4096;
const MISTRAL_FEED_FORWARD: u32 = 14336;
const MISTRAL_QUERY_HEADS: u32 = 32;
const MISTRAL_KV_HEADS: u32 = 8;
const MISTRAL_HEAD_DIM: u32 = 128;
const MISTRAL_BLOCK_COUNT: u32 = 32;
/// `DenseArch::bind`'s own call site
/// (`proxima-model-interop/src/dense.rs`) reads `expert_count`/
/// `expert_used_count` straight off the checkpoint's own metadata
/// (`architecture.expert_count`/`expert_used_count`) -- openchat-3.5-1210
/// has no `ffn_gate_inp.weight` tensor, so both read `0` there, selecting
/// `mistral_cached_forward_program_with_experts_and_layer_taps`'s dense
/// branch.
const MISTRAL_EXPERT_COUNT: u32 = 0;
const MISTRAL_EXPERT_USED_COUNT: u32 = 0;

/// Builds the real openchat-3.5-1210 / Mistral-7B-v0.1 dense [`ModelDescriptor`]
/// -- interleaved RoPE off one shared table, `1/sqrt(head_dim)` attention
/// score scale, plain projected-V (no value-norm, no shared-KV), exclusive
/// SwiGLU FFN, no embedding scale, no logit softcap. Every field mirrors
/// `mistral_cached_forward_program_with_experts_and_layer_taps`'s own
/// builder (`proxima-tensor/src/spec/attention_forward.rs`) node-for-node --
/// see that function's call site in `DenseArch::bind`
/// (`proxima-model-interop/src/dense.rs`) for the real argument set this
/// descriptor's constants were read off. `vocab` stays a parameter, not a
/// baked-in constant, for the same reason [`gemma4_descriptor`]'s own doc
/// gives: it is per-checkpoint tokenizer data, not architecture.
///
/// [`ModelDescriptor::cache_strategy`] is [`CacheStrategy::SingleRange`] --
/// see that variant's own doc for why its [`build_forward`] arm dispatches
/// straight to `mistral_cached_forward_program_with_experts_and_layer_taps`
/// rather than through the schedule-driven
/// [`lfm2_single_range_cached_forward_program_with_experts`] (that engine's
/// own unconditional per-head QK-norm and `cached_len`-aware merged mask
/// are not openchat-3.5-1210's own program, node for node -- see
/// `build_forward_matches_direct_builder_call_at_real_mistral_dims`'s own
/// doc, `proxima-tensor/src/spec/tests.rs`, for the byte-identical proof
/// this field's value makes true).
///
/// `expert_feed_forward` reuses [`MISTRAL_FEED_FORWARD`] rather than a
/// separate constant: unlike gemma4's split dense/routed widths,
/// `mistral_cached_forward_program_with_experts_and_layer_taps`'s own MoE
/// branch (`append_mistral_cached_moe_layer`,
/// `proxima-tensor/src/spec/single_range_moe_cached.rs`) sizes
/// `ffn_gate_exps.weight`/`ffn_up_exps.weight`/`ffn_down_exps.weight` off
/// the SAME `feed_forward` parameter the dense branch uses -- one width,
/// not two. `leading_dense_block_count` is set to the full
/// [`MISTRAL_BLOCK_COUNT`] ("every layer is dense") rather than `0`
/// because openchat-3.5-1210 itself is dense (`expert_count == 0`); this
/// builder picks dense-vs-MoE for the WHOLE checkpoint, not per layer, so
/// this field is likewise inert for mistral until a future slice wires it.
/// `routed_gating`/`routed_expert_bias` on [`LayerFfnConfig`] mirror
/// `append_mistral_cached_moe_layer`'s own hardcoded router instead
/// (`ExpertGatingFunc::Softmax`, no bias, `single_range_moe_cached.rs`
/// lines 1428-1430) -- the SAME builder this dense checkpoint's program
/// comes from runs that router whenever a Mixtral-family checkpoint's
/// `expert_count > 0`, so this reflects real (if here unexercised, since
/// openchat-3.5-1210 itself never takes that branch) behaviour rather than
/// [`LayerFfnConfig::exclusive`]'s own unrelated LFM2 default
/// (`ExpertGatingFunc::Sigmoid`, bias `true`).
/// [`mistral_descriptor`]'s own shape, but every field [`DenseArch::bind`]
/// already reads off a real checkpoint's own metadata
/// (`ModelArchitecture`'s `embedding`/`feed_forward`/`query_heads`/
/// `kv_heads`/`head_dim`/`block_count`/`expert_count`/`expert_used_count`)
/// stays a parameter here rather than a `MISTRAL_*` constant --
/// `DenseArch` is the un-registered-by-name fallback for `llama`, `mistral`,
/// `qwen3`, `mixtral`, and any other architecture this crate has no
/// dedicated hybrid binder for (`crate::dense`'s own module doc), so a
/// SINGLE proven checkpoint's dims (openchat-3.5-1210's, `mistral_descriptor`
/// below) are only ever right for that one checkpoint -- a Mixtral header's
/// `expert_count > 0`, or any header with a different `head_dim`/
/// `block_count`, would silently mis-shape the whole program if `DenseArch`
/// built its descriptor from those constants instead of this function.
/// [`mistral_descriptor`] itself now delegates here with its own
/// `MISTRAL_*` constants, so the two never drift against each other.
///
/// [`DenseArch::bind`]: ../../../proxima_model_interop/dense/struct.DenseArch.html
#[expect(clippy::too_many_arguments, reason = "mirrors the builder's own flat positional signature this descriptor replaces -- see build_forward's SingleRange arm, which reads every one of these fields straight back off the descriptor it builds")]
#[must_use]
pub fn mistral_descriptor_from_shape(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
) -> ModelDescriptor {
    let ffn = LayerFfnConfig {
        post_attention_norm: false,
        combination: FfnCombination::Exclusive,
        output_scale: false,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        dense_feed_forward: None,
        exclusive_dense_post_norm: false,
        activation: Activation::Silu,
        ple: false,
    };

    let attention = LayerAttentionConfig {
        head_dim,
        kv_heads,
        mask_window: None,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos",
            sin_name: "rope_sin",
        },
        rope_pairing: RopePairing::Interleaved,
        score_scale: AttentionScoreScale::InverseSqrtQueryPreAttnScalar(head_dim),
        value_norm: false,
    };

    let layers: Vec<LayerSchedule> = (0..block_count)
        .map(|_| LayerSchedule {
            kind: LayerKind::Attention,
            attention,
            ffn,
        })
        .collect();

    ModelDescriptor {
        vocab,
        embedding,
        feed_forward,
        expert_feed_forward: feed_forward,
        query_heads,
        block_count,
        expert_count,
        expert_used_count,
        leading_dense_block_count: block_count,
        // no `LayerKind::ShortConv` layer ever appears in a SingleRange
        // schedule, and that arm's own builder never reads `l_cache` --
        // same inertness as `Self::l_cache`'s own doc.
        l_cache: 0,
        embedding_scale: None,
        logit_softcap: None,
        layers,
        cache_strategy: CacheStrategy::SingleRange,
        qk_norm,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
    }
}

#[must_use]
pub fn mistral_descriptor(vocab: u32) -> ModelDescriptor {
    mistral_descriptor_from_shape(
        vocab,
        MISTRAL_EMBEDDING,
        MISTRAL_FEED_FORWARD,
        MISTRAL_QUERY_HEADS,
        MISTRAL_KV_HEADS,
        MISTRAL_HEAD_DIM,
        MISTRAL_BLOCK_COUNT,
        MISTRAL_EXPERT_COUNT,
        MISTRAL_EXPERT_USED_COUNT,
        // openchat-3.5-1210's own real header: no per-head QK-norm weights,
        // no bias tensors, and `DenseArch::bind`'s own call site
        // (`proxima-model-interop/src/dense.rs`) always passes `false` for
        // both reduce-fusion diagnostics -- see slice 1's own real-shape
        // citation on `MISTRAL_HEAD_DIM` above.
        false,
        false,
        false,
        false,
    )
}

/// [`build_forward`]'s own return shape: the lowered program, its `logits`
/// root, one [`CachedLayerRoots`] per layer (empty under
/// [`CacheStrategy::Cacheless`]), one [`MoeSite`] per MoE layer, one
/// residual [`NodeId`] per layer (empty under
/// [`CacheStrategy::Cacheless`]/[`CacheStrategy::TwoRange`], neither of
/// which tracks it -- [`CacheStrategy::SingleRange`]'s own arm is the only
/// one that populates it, straight from
/// `mistral_cached_forward_program_with_experts_and_layer_taps`'s own
/// fourth return element), and the `hidden` root (`ForwardRoots::hidden` --
/// the last-norm activation `logits` projects from, the node
/// `proxima-model-interop`'s `LoadedModel::embed` pooling path needs and
/// `DenseArch::bind`'s mistral arm carried before it routed through this
/// function). `None` under [`CacheStrategy::Cacheless`]/[`CacheStrategy::TwoRange`]:
/// neither `lfm2_forward_program_with_experts` nor
/// `lfm2_two_range_cached_forward_program_with_experts` expose a hidden
/// node at all, so there is no value here to forward, not merely one this
/// function declines to read -- same "degenerate default for an engine
/// that does not produce this value" precedent `cache_roots` and the
/// residual `Vec<NodeId>` already set.
pub type BuildForwardProgram = (
    Vec<Op>,
    NodeId,
    Vec<CachedLayerRoots>,
    MoeSites,
    Vec<NodeId>,
    Option<NodeId>,
);

/// Generalizes [`lfm2_two_range_cached_forward_program_with_experts`] (the
/// working gemma4 two-range engine, already schedule-driven rather than
/// gemma4-hardcoded internally) and [`lfm2_forward_program_with_experts`]
/// (the cacheless engine) behind one [`ModelDescriptor`]-shaped entry point:
/// plain sync data->op-graph construction, no async/`Future`/`Box<dyn>`
/// anywhere, dispatching purely on [`ModelDescriptor::cache_strategy`] and
/// rewriting none of either engine's math. [`CacheStrategy::TwoRange`] calls
/// the two-range builder directly, unchanged. [`CacheStrategy::Cacheless`]
/// is the degenerate case the two-range engine's own cache leaves
/// (`kv_cache.{layer}.k_even`/`k_odd`/`v`) are elided for: it delegates to
/// the plain builder and wraps that builder's three-tuple return
/// (`program, logits, moe_sites`, no cache roots) into this function's own
/// return shape with `cache_roots` always empty, so callers never match on
/// which engine actually built the program. `last_row_only` stays a call
/// parameter rather than a `ModelDescriptor` field because it shapes a
/// single call's graph (whole-sequence logits vs. one gathered row), not the
/// model itself -- the same role it already plays as each builder's own
/// trailing positional argument.
///
/// [`CacheStrategy::SingleRange`] dispatches to
/// `mistral_cached_forward_program_with_experts_and_layer_taps` directly,
/// unchanged -- see that variant's own doc for why (a genuinely different
/// cache-scoring algebra, not a knob the other two engines can express).
/// That builder's per-layer residual outputs are this function's own fifth
/// return element, empty for [`CacheStrategy::Cacheless`]/
/// [`CacheStrategy::TwoRange`] (neither engine tracks them) -- the same
/// "degenerate default for an engine that does not produce this value"
/// precedent `cache_roots` itself already sets on the [`Cacheless`][CacheStrategy::Cacheless]
/// arm above.
///
/// [`BuildForwardProgram`]'s own doc names each element -- the same
/// named-tuple-alias precedent
/// [`MistralMoeForwardProgramWithLayerTaps`] already sets for a
/// same-shaped return.
pub fn build_forward(
    descriptor: &ModelDescriptor,
    last_row_only: bool,
) -> Result<BuildForwardProgram, TensorError> {
    match descriptor.cache_strategy {
        CacheStrategy::TwoRange => {
            let (program, logits, cache_roots, moe_sites) =
                lfm2_two_range_cached_forward_program_with_experts(
                    descriptor.vocab,
                    descriptor.embedding,
                    descriptor.feed_forward,
                    descriptor.expert_feed_forward,
                    descriptor.query_heads,
                    descriptor.block_count,
                    descriptor.expert_count,
                    descriptor.expert_used_count,
                    descriptor.leading_dense_block_count,
                    &descriptor.layers,
                    descriptor.embedding_scale,
                    descriptor.logit_softcap,
                    last_row_only,
                )?;
            Ok((program, logits, cache_roots, moe_sites, Vec::new(), None))
        }
        CacheStrategy::Cacheless => {
            let (program, logits, moe_sites) = lfm2_forward_program_with_experts(
                descriptor.vocab,
                descriptor.embedding,
                descriptor.feed_forward,
                descriptor.expert_feed_forward,
                descriptor.query_heads,
                descriptor.block_count,
                descriptor.expert_count,
                descriptor.expert_used_count,
                descriptor.leading_dense_block_count,
                descriptor.l_cache,
                &descriptor.layers,
                descriptor.embedding_scale,
                descriptor.logit_softcap,
                last_row_only,
                None,
            )?;
            Ok((program, logits, Vec::new(), moe_sites, Vec::new(), None))
        }
        CacheStrategy::SingleRange => {
            if descriptor.layers.len() != descriptor.block_count as usize {
                return Err(TensorError::LayerScheduleCountMismatch {
                    expected: descriptor.block_count,
                    found: descriptor.layers.len(),
                });
            }
            let Some(first) = descriptor.layers.first() else {
                return Err(TensorError::LayerScheduleCountMismatch {
                    expected: descriptor.block_count,
                    found: 0,
                });
            };
            // This builder takes one flat `head_dim`/`kv_heads` for the
            // whole model, not a per-layer value (`DenseArch::bind`'s own
            // `architecture.uniform_kv_heads()?` call makes the same
            // requirement at bind time) -- silently reading `layers[0]`
            // over a schedule that actually varies per layer would build a
            // structurally wrong program with no error at all, the exact
            // failure mode `TensorError::UnsupportedInBuilder`'s own doc
            // says to raise instead of work around.
            if descriptor
                .layers
                .iter()
                .any(|layer| layer.attention != first.attention)
            {
                return Err(TensorError::UnsupportedInBuilder {
                    builder: "build_forward(CacheStrategy::SingleRange)",
                    feature: "non-uniform per-layer attention config",
                });
            }
            let attention = first.attention;
            let (program, roots, cache_roots, layer_residuals, moe_sites) =
                mistral_cached_forward_program_with_experts_and_layer_taps(
                    descriptor.vocab,
                    descriptor.embedding,
                    descriptor.feed_forward,
                    descriptor.query_heads,
                    attention.kv_heads,
                    attention.head_dim,
                    descriptor.block_count,
                    descriptor.expert_count,
                    descriptor.expert_used_count,
                    descriptor.qk_norm,
                    descriptor.qkv_biases,
                    descriptor.paired_gate_up_reduce,
                    descriptor.fused_qkv_reduce,
                    last_row_only,
                )?;
            Ok((
                program,
                roots.logits,
                cache_roots,
                moe_sites,
                layer_residuals,
                Some(roots.hidden),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read from `token_embd.weight`'s own row count at bind time on a real
    /// checkpoint (`hparams::Architecture::vocab`) -- any value proves this
    /// shape test, since nothing here reads `vocab` back out.
    const TEST_VOCAB: u32 = 32;

    #[test]
    fn gemma4_descriptor_matches_real_checkpoint_shape() {
        let descriptor = gemma4_descriptor(TEST_VOCAB);

        assert_eq!(descriptor.block_count, 30);
        assert_eq!(descriptor.layers.len(), 30);
        assert_eq!(descriptor.expert_count, 128);
        assert_eq!(descriptor.expert_used_count, 8);
        assert_eq!(descriptor.embedding, 2816);
        assert_eq!(descriptor.leading_dense_block_count, 0);
        assert_eq!(descriptor.logit_softcap, Some(30.0));
        assert_eq!(descriptor.embedding_scale, Some(EmbeddingScale::Sqrt));
        assert_eq!(descriptor.cache_strategy, CacheStrategy::Cacheless);

        let full_layers: Vec<u32> = descriptor
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.attention.mask_window.is_none())
            .map(|(index, _)| index as u32)
            .collect();
        assert_eq!(full_layers, alloc::vec![5, 11, 17, 23, 29]);

        for (index, layer) in descriptor.layers.iter().enumerate() {
            assert_eq!(layer.kind, LayerKind::Attention);
            match layer.ffn.combination {
                FfnCombination::ParallelDenseMoe(_) => {}
                FfnCombination::Exclusive => {
                    panic!("layer {index} must run parallel dense+MoE FFN")
                }
            }
        }

        let sliding_kv_heads: Vec<u32> = descriptor
            .layers
            .iter()
            .filter(|layer| layer.attention.mask_window.is_some())
            .map(|layer| layer.attention.kv_heads)
            .collect();
        assert!(
            sliding_kv_heads.iter().all(|&kv_heads| kv_heads == 8),
            "every sliding layer uses 8 kv-heads: {sliding_kv_heads:?}"
        );

        let full_kv_heads: Vec<u32> = descriptor
            .layers
            .iter()
            .filter(|layer| layer.attention.mask_window.is_none())
            .map(|layer| layer.attention.kv_heads)
            .collect();
        assert!(
            full_kv_heads.iter().all(|&kv_heads| kv_heads == 2),
            "every full layer uses 2 kv-heads: {full_kv_heads:?}"
        );
    }

    #[test]
    fn mistral_descriptor_matches_real_openchat_checkpoint_shape() {
        let descriptor = mistral_descriptor(TEST_VOCAB);

        assert_eq!(descriptor.block_count, 32);
        assert_eq!(descriptor.layers.len(), 32);
        assert_eq!(descriptor.embedding, 4096);
        assert_eq!(descriptor.feed_forward, 14336);
        assert_eq!(descriptor.expert_feed_forward, 14336);
        assert_eq!(descriptor.query_heads, 32);
        assert_eq!(descriptor.expert_count, 0);
        assert_eq!(descriptor.expert_used_count, 0);
        assert_eq!(descriptor.leading_dense_block_count, 32);
        assert_eq!(descriptor.l_cache, 0);
        assert_eq!(descriptor.embedding_scale, None);
        assert_eq!(descriptor.logit_softcap, None);
        assert_eq!(descriptor.cache_strategy, CacheStrategy::SingleRange);
        assert!(!descriptor.qk_norm);
        assert!(!descriptor.qkv_biases);
        assert!(!descriptor.paired_gate_up_reduce);
        assert!(!descriptor.fused_qkv_reduce);

        for layer in &descriptor.layers {
            assert_eq!(layer.kind, LayerKind::Attention);
            assert_eq!(layer.attention.head_dim, 128);
            assert_eq!(layer.attention.kv_heads, 8);
            assert_eq!(layer.attention.mask_window, None);
            assert_eq!(layer.attention.value_source_kind, ValueSourceKind::ProjectedV);
            assert_eq!(layer.attention.rope_table.cos_name, "rope_cos");
            assert_eq!(layer.attention.rope_table.sin_name, "rope_sin");
            assert_eq!(layer.attention.rope_pairing, RopePairing::Interleaved);
            assert_eq!(
                layer.attention.score_scale,
                AttentionScoreScale::InverseSqrtQueryPreAttnScalar(128)
            );
            assert!(!layer.attention.value_norm);
            assert_eq!(layer.ffn.combination, FfnCombination::Exclusive);
            assert_eq!(layer.ffn.activation, Activation::Silu);
            assert!(!layer.ffn.post_attention_norm);
            assert!(!layer.ffn.output_scale);
            assert_eq!(layer.ffn.routed_gating, ExpertGatingFunc::Softmax);
            assert!(!layer.ffn.routed_expert_bias);
        }
    }
}
