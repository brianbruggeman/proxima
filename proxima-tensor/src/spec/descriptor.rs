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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum CacheStrategy {
    #[default]
    Cacheless,
    TwoRange,
    /// [`mistral_cached_forward_program_with_experts_and_layer_taps`]'s own
    /// single-continuous-range KV cache (`kv_cache.{layer}.k_even`/`k_odd`/
    /// `v`) -- a genuinely different scoring algebra from both other
    /// variants: the cached block is NEVER masked (a cached position is
    /// definitionally in the past of every new query, per that function's
    /// own doc on its `is_future` usage), where [`CacheStrategy::Cacheless`]'s block-local
    /// mask has no cache to skip and [`CacheStrategy::TwoRange`]'s own single-range
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
    /// shape [`CacheStrategy::TwoRange`]'s own arm already uses.
    SingleRange,
}

/// A whole model's build-time shape as DATA: the global hyperparameters
/// [`lfm2_forward_program_with_experts`] already takes as loose positional
/// arguments (`vocab`, `embedding`, `block_count`, `expert_count`,
/// `expert_used_count`, `leading_dense_block_count`, `embedding_scale`,
/// `logit_softcap`), one [`LayerSchedule`] per block, and which cache engine
/// to build with. `layers` reuses [`LayerSchedule`] verbatim -- it already
/// composes [`LayerKind`], [`LayerAttentionConfig`], and [`LayerFfnConfig`]
/// (`proxima-model-interop::gemma4_descriptor_from_gguf` builds
/// exactly this shape today, by hand, at bind time), so this struct does not
/// re-mint a per-layer type. Genuinely new: no existing type bundles a
/// model's full hyperparameter set with its per-layer schedule and a
/// cache-engine choice into one value a caller can build ahead of time --
/// today every caller re-derives and re-passes these as separate positional
/// arguments at each forward-program call site.
///
/// At the std boundary (`config` feature) this is a conflaguration config:
/// `Settings` reads the scalar fields from `PROXIMA_MODEL_*` env vars over a
/// seeded value (`conflaguration::builder().value(base).env().file(path)`),
/// and the `bon` builder constructs the same value fluently. The structured
/// fields (`layers`, `embedding_scale`, `cache_strategy`) have no env
/// spelling; a TOML layer sets them.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "config", derive(bon::Builder, conflaguration::Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_MODEL"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[serde(deny_unknown_fields)]
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
    #[cfg_attr(feature = "config", setting(skip))]
    pub embedding_scale: Option<EmbeddingScale>,
    pub logit_softcap: Option<f32>,
    /// Divisor on the final logits, `logits / logit_scale`: the checkpoint's
    /// `<family>.logit_scale`, 6.0 for granite; `None` leaves the logits untouched.
    pub logit_scale: Option<f32>,
    /// Multiplier on each sublayer output before it joins the residual stream,
    /// `x + scale * sublayer`: the checkpoint's `<family>.residual_scale`, 0.22 for
    /// granite; `None` is the plain add.
    pub residual_scale: Option<f32>,
    /// One entry per block. In a config file an entry may carry `repeat = N`
    /// ("this layer, N times"), or hold a `pattern` of entries that `repeat`s
    /// (gemma4's four sliding layers then a full one, three times); it expands
    /// to one [`LayerSchedule`] per block on load and serializes expanded.
    #[cfg_attr(feature = "config", setting(skip))]
    #[serde(deserialize_with = "layer_runs::deserialize")]
    pub layers: Vec<LayerSchedule>,
    #[cfg_attr(feature = "config", setting(skip))]
    pub cache_strategy: CacheStrategy,
    /// gemma4 E2B/E4B's per-layer-embedding preamble width
    /// (`lfm2_forward_program_with_experts`'s own `ple_dim` parameter doc,
    /// `Some(256)` for E2B) -- consulted by [`CacheStrategy::Cacheless`] and
    /// [`CacheStrategy::TwoRange`] alike (both route to a PLE-aware
    /// builder); `None` for every checkpoint with no PLE tensors, and inert
    /// under [`CacheStrategy::SingleRange`], which has no PLE concept at
    /// all -- same "unused when the arm never reads it" precedent
    /// [`Self::qk_norm`]'s own doc already sets.
    pub ple_dim: Option<u32>,
    /// [`CacheStrategy::TwoRange`] and [`CacheStrategy::SingleRange`]: lay every
    /// windowed layer's KV cache out as a ring ([`SLIDING_KV_SYMBOL`],
    /// [`SLIDING_CACHED_LEN_INPUT`], and the builders' own `sliding_kv_ring`
    /// argument); a layer with no window keeps the full cache. Inert under
    /// [`CacheStrategy::Cacheless`], the same "unused when the arm never reads
    /// it" precedent as [`Self::qk_norm`].
    pub sliding_kv_ring: bool,
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
    /// LM-head repeat count for the head-cost measurement harness, consulted
    /// by [`CacheStrategy::Cacheless`] and [`CacheStrategy::TwoRange`]: `1`
    /// builds only the production head, `2` and `3` append that many minus one
    /// byte-identical duplicate head chains and return their roots as
    /// `duplicate_head_roots`. Values outside `1..=3` clamp. Inert under
    /// [`CacheStrategy::SingleRange`], same "unused when the arm never reads it"
    /// precedent as [`Self::qk_norm`].
    pub head_repeats: u32,
    /// `true` gathers the LM head to the last new row, which is all decode and prefill sample;
    /// `false` keeps every new row's logits, which a speculative verify step and pooled
    /// embeddings read.
    pub last_row_only: bool,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for ModelDescriptor {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        let mut require = |holds: bool, field: &str, message: &str| {
            if !holds {
                errors.push(conflaguration::ValidationMessage::new(field, message));
            }
        };
        require(self.vocab > 0, "vocab", "must be positive");
        require(self.embedding > 0, "embedding", "must be positive");
        require(self.query_heads > 0, "query_heads", "must be positive");
        require(self.block_count > 0, "block_count", "must be positive");
        require(
            self.layers.len() == self.block_count as usize,
            "layers",
            "must hold one entry per block",
        );
        require(
            self.expert_used_count <= self.expert_count,
            "expert_used_count",
            "must not exceed expert_count",
        );
        require(
            self.leading_dense_block_count <= self.block_count,
            "leading_dense_block_count",
            "must not exceed block_count",
        );
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

/// The values a family's GGUF header and HF `config.json` do not carry, as
/// DATA: the one record both [`gemma4_descriptor_from_gguf`] and
/// [`mistral_descriptor_from_shape`] read, so neither holds a per-family
/// literal. `proxima-model-interop` parses one TOML file per family into this
/// (`serde`, via the [`Deserialize`] derives on [`LayerFfnConfig`],
/// [`ParallelDenseMoeConfig`], [`EmbeddingScale`], [`Activation`] and
/// [`ExpertGatingFunc`]), keyed by the checkpoint's own family string.
///
/// Parsing lives in the consumer because this crate's alloc tier carries no
/// TOML parser; a descriptor builder stays a pure data-in, data-out function
/// that takes the already-parsed profile.
///
/// Everything per-checkpoint (head dims, widths, per-layer windows, PLE
/// width) stays in the GGUF/HF reader: [`LayerFfnConfig::dense_feed_forward`]
/// and [`LayerFfnConfig::ple`] are left at their defaults here and overridden
/// per layer by the builder.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyProfile {
    /// Multiplier on the embedding lookup (`Some(Sqrt)` for a family that
    /// scales it by `sqrt(embedding)`, `None` otherwise).
    #[serde(default)]
    pub embedding_scale: Option<EmbeddingScale>,
    /// Every layer's FFN knobs for a dense (`expert_count == 0`) checkpoint.
    pub ffn: LayerFfnConfig,
    /// When present and the checkpoint has experts, each layer runs its dense
    /// and routed FFNs in parallel with these sub-norm knobs instead of
    /// [`Self::ffn`]'s own [`LayerFfnConfig::combination`].
    #[serde(default)]
    pub parallel_dense_moe: Option<ParallelDenseMoeConfig>,
    /// `true` scales attention scores by `1/sqrt(head_dim)`
    /// ([`AttentionScoreScale::InverseSqrtQueryPreAttnScalar`]); `false` leaves
    /// them unscaled ([`AttentionScoreScale::Unscaled`]).
    pub score_scale_inverse_sqrt_head_dim: bool,
    /// Per-kv-head value RMSNorm without a learned scale
    /// ([`LayerAttentionConfig::value_norm`]) on layers that own their `V`.
    pub value_norm: bool,
    /// How this family pairs RoPE channels, per llama.cpp's
    /// `llama_model_rope_type` (`src/llama-model.cpp`): NEOX/MROPE/IMROPE
    /// families are [`RopeLayout::SplitHalf`], NORM families are
    /// [`RopeLayout::Adjacent`]. Never inferred from tensor presence.
    pub rope_layout: RopeLayout,
}

/// Which channels RoPE rotates together: `(i, i + rotary_dim / 2)` or
/// `(2i, 2i + 1)`. The count of rotating dims is not here; it is the
/// checkpoint's own `<arch>.rope.dimension_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeLayout {
    SplitHalf,
    Adjacent,
}

impl FamilyProfile {
    /// One layer's [`LayerFfnConfig`]: [`Self::ffn`], with the combination
    /// taken from [`Self::parallel_dense_moe`] when the checkpoint has experts.
    #[must_use]
    pub fn layer_ffn(&self, expert_count: u32) -> LayerFfnConfig {
        let combination = match self.parallel_dense_moe {
            Some(parallel) if expert_count > 0 => FfnCombination::ParallelDenseMoe(parallel),
            _ => self.ffn.combination,
        };
        LayerFfnConfig {
            combination,
            ..self.ffn
        }
    }

    /// The attention score scale for a layer of width `head_dim`.
    #[must_use]
    pub const fn score_scale(&self, head_dim: u32) -> AttentionScoreScale {
        if self.score_scale_inverse_sqrt_head_dim {
            AttentionScoreScale::InverseSqrtQueryPreAttnScalar(head_dim)
        } else {
            AttentionScoreScale::Unscaled
        }
    }

    /// RoPE pairing for a head whose rotating width is `rotary_dim`
    /// (`<arch>.rope.dimension_count`, or the head width when the header has
    /// no such key).
    #[must_use]
    pub const fn rope_pairing(&self, rotary_dim: u32) -> RopePairing {
        match self.rope_layout {
            RopeLayout::SplitHalf => RopePairing::SplitHalf { pairs: rotary_dim / 2 },
            RopeLayout::Adjacent => RopePairing::Interleaved,
        }
    }
}

/// Builds the single-range dense [`ModelDescriptor`] [`DenseArch::bind`]
/// (`proxima-model-interop/src/dense.rs`) hands to [`build_forward`]:
/// interleaved or caller-chosen RoPE off one shared table, `1/sqrt(head_dim)`
/// attention score scale, plain projected-V (no value-norm, no shared-KV),
/// exclusive SwiGLU FFN, no embedding scale, no logit softcap.
/// [`CacheStrategy::SingleRange`] makes [`build_forward`] dispatch straight to
/// `mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing`
/// (see that variant's own doc). Every dimension is a parameter because
/// `DenseArch` serves llama, mistral, qwen2, qwen3 and mixtral headers alike;
/// `expert_feed_forward` is `feed_forward` because that builder sizes the
/// routed branch off the same width as the dense one, and
/// `leading_dense_block_count` is `block_count` because the dense-vs-MoE
/// choice is whole-checkpoint (`expert_count`), not per layer.
/// `profile` carries every value the checkpoint header does not: the FFN
/// activation and norm knobs, the score-scale rule, value-norm, the embedding
/// scale, and whether RoPE is split-half regardless of QK-norm tensors (a
/// split-half family with no QK-norm tensors cannot be told from `qk_norm`
/// alone).
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
    profile: &FamilyProfile,
) -> ModelDescriptor {
    let ffn = profile.layer_ffn(expert_count);

    let attention = LayerAttentionConfig {
        head_dim,
        kv_heads,
        mask_window: None,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: profile.rope_pairing(head_dim),
        score_scale: profile.score_scale(head_dim),
        value_norm: profile.value_norm,
    };

    let layers: Vec<LayerSchedule> = (0..block_count)
        .map(|_| LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention.clone(),
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
        embedding_scale: profile.embedding_scale,
        logit_softcap: None,
        logit_scale: None,
        residual_scale: None,
        layers,
        cache_strategy: CacheStrategy::SingleRange,
        // `CacheStrategy::SingleRange` has no PLE concept at all -- same
        // inertness as `Self::qk_norm`'s own doc.
        ple_dim: None,
        sliding_kv_ring: false,
        qk_norm,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        head_repeats: 1,
        last_row_only: true,
    }
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
/// residual `Vec<NodeId>` already set. The seventh element,
/// `duplicate_head_roots`, is `PROXIMA_HEAD_REPEATS`'s own scratch output
/// (`lfm2_forward_program_with_experts`'s own doc) forwarded through under
/// [`CacheStrategy::Cacheless`] only -- empty everywhere else, same
/// degenerate-default precedent.
pub type BuildForwardProgram = (
    Vec<Op>,
    NodeId,
    Vec<CachedLayerRoots>,
    MoeSites,
    Vec<NodeId>,
    Option<NodeId>,
    Vec<NodeId>,
);

fn refuse_when(
    set: bool,
    builder: &'static str,
    feature: &'static str,
) -> Result<(), TensorError> {
    if set {
        return Err(TensorError::UnsupportedInBuilder { builder, feature });
    }
    Ok(())
}

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
/// which engine actually built the program. `last_row_only` is a
/// [`ModelDescriptor`] field, so the verify shape is data: one config lowers
/// the decode program or the verify program by flipping it.
///
/// [`CacheStrategy::SingleRange`] dispatches to
/// `mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing`
/// directly, passing this descriptor's own `attention.rope_pairing` rather
/// than re-inferring it from `qk_norm` -- see that variant's own doc for why
/// (a genuinely different cache-scoring algebra, not a knob the other two
/// engines can express).
/// That builder's per-layer residual outputs are this function's own fifth
/// return element, empty for [`CacheStrategy::Cacheless`]/
/// [`CacheStrategy::TwoRange`] (neither engine tracks them) -- the same
/// "degenerate default for an engine that does not produce this value"
/// precedent `cache_roots` itself already sets on the [`Cacheless`][CacheStrategy::Cacheless]
/// arm above.
///
/// [`BuildForwardProgram`]'s own doc names each element -- the same
/// named-tuple-alias precedent
/// `MistralMoeForwardProgramWithLayerTaps` already sets for a
/// same-shaped return.
pub fn build_forward(
    descriptor: &ModelDescriptor,
) -> Result<BuildForwardProgram, TensorError> {
    match descriptor.cache_strategy {
        CacheStrategy::TwoRange => {
            refuse_when(
                descriptor.logit_scale.is_some(),
                "build_forward(CacheStrategy::TwoRange)",
                "a logit scale",
            )?;
            refuse_when(
                descriptor.residual_scale.is_some(),
                "build_forward(CacheStrategy::TwoRange)",
                "a residual scale",
            )?;
            let (program, logits, cache_roots, moe_sites, duplicate_head_roots) =
                lfm2_two_range_cached_forward_program_with_experts_and_head_repeats(
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
                    descriptor.last_row_only,
                    descriptor.ple_dim,
                    descriptor.sliding_kv_ring,
                    descriptor.head_repeats,
                )?;
            Ok((
                program,
                logits,
                cache_roots,
                moe_sites,
                Vec::new(),
                None,
                duplicate_head_roots,
            ))
        }
        CacheStrategy::Cacheless => {
            refuse_when(
                descriptor.logit_scale.is_some(),
                "build_forward(CacheStrategy::Cacheless)",
                "a logit scale",
            )?;
            refuse_when(
                descriptor.residual_scale.is_some(),
                "build_forward(CacheStrategy::Cacheless)",
                "a residual scale",
            )?;
            let (program, logits, moe_sites, duplicate_head_roots) = lfm2_forward_program_with_experts_and_head_repeats(
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
                descriptor.last_row_only,
                descriptor.ple_dim,
                descriptor.head_repeats,
            )?;
            Ok((
                program,
                logits,
                Vec::new(),
                moe_sites,
                Vec::new(),
                None,
                duplicate_head_roots,
            ))
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
            if descriptor.layers.iter().any(|layer| {
                LayerAttentionConfig {
                    mask_window: first.attention.mask_window,
                    ..layer.attention.clone()
                } != first.attention
            }) {
                return Err(TensorError::UnsupportedInBuilder {
                    builder: "build_forward(CacheStrategy::SingleRange)",
                    feature: "non-uniform per-layer attention config",
                });
            }
            let attention = &first.attention;
            let layer_windows: Vec<Option<u32>> =
                descriptor.layers.iter().map(|layer| layer.attention.mask_window).collect();
            let pairing_the_moe_layer_derives = if descriptor.qk_norm {
                RopePairing::SplitHalf { pairs: attention.head_dim / 2 }
            } else {
                RopePairing::Interleaved
            };
            refuse_when(
                descriptor.expert_count > 0
                    && attention.rope_pairing != pairing_the_moe_layer_derives,
                "build_forward(CacheStrategy::SingleRange)",
                "a rope pairing the moe layer cannot express",
            )?;
            // `attention.rope_pairing` is this descriptor's own data, not
            // re-inferred from `qk_norm` here -- Qwen2 needs split-half RoPE
            // with `qk_norm` still `false` (no QK-norm tensors at all), a
            // combination the qk_norm-inferring wrapper cannot express (its
            // own doc on that limitation, `mistral_descriptor_from_shape`'s
            // `rope_pairing` parameter above).
            let (program, roots, cache_roots, layer_residuals, moe_sites) =
                mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
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
                    descriptor.last_row_only,
                    attention.rope_pairing,
                    descriptor.embedding_scale,
                    descriptor.logit_scale,
                    attention.score_scale,
                    descriptor.residual_scale,
                    &layer_windows,
                    descriptor.sliding_kv_ring,
                )?;
            Ok((
                program,
                roots.logits,
                cache_roots,
                moe_sites,
                layer_residuals,
                Some(roots.hidden),
                Vec::new(),
            ))
        }
    }
}
