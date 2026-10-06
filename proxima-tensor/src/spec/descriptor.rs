use super::*;

/// Whether a forward program keeps a KV cache, as data on
/// [`ModelDescriptor::cache_strategy`]. [`CacheStrategy::Cacheless`] is
/// [`lfm2_forward_program_with_experts`] (full reprefill every call);
/// [`CacheStrategy::Cached`] is the two-block cached engine, whose cached-block
/// mask [`ModelDescriptor::cache_mask`] selects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum CacheStrategy {
    #[default]
    Cacheless,
    Cached,
}

/// How a cached program excludes the cache bucket's zero padding from the
/// cached block's softmax, as data on [`ModelDescriptor::cache_mask`]. Both arms
/// score through [`append_cached_block_scores`](super::two_block_attention) and
/// [`append_local_block_and_combine`](super::two_block_attention); they differ
/// only in the mask node the cached block carries.
///
/// A program with the same layers but the other mask is a different op graph
/// (a `Select` node per layer, a `cached_len`-aware compare per distinct
/// window), which is why the choice is a field and not a rewrite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum CacheMask {
    /// The cached block is never masked in the graph: the executor's
    /// `cached_len` operand bounds it, so a past position is always visible.
    /// Only a windowed layer adds a mask, because a window is a distance the
    /// bound cannot express. This is the dense and MoE families' lowering
    /// ([`mistral_cached_forward_program_with_experts_and_layer_taps`]).
    #[default]
    Bounded,
    /// Every cached block is masked in the graph
    /// ([`causal_mask_cached_windowed`]): padding at or past `cached_len` is
    /// excluded by a node, and a window composes onto the same mask. The
    /// lowering for a schedule with shared KV or per-layer widths
    /// ([`lfm2_two_range_cached_forward_program_with_experts`]).
    Padded,
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
/// fields (`layers`, `embedding_scale`, `cache_strategy`, `cache_mask`) have no env
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
    /// Which mask a cached program's cached block carries; read only when
    /// [`Self::cache_strategy`] is [`CacheStrategy::Cached`].
    #[cfg_attr(feature = "config", setting(skip))]
    #[serde(default)]
    pub cache_mask: CacheMask,
    /// gemma4 E2B/E4B's per-layer-embedding preamble width
    /// (`lfm2_forward_program_with_experts`'s own `ple_dim` parameter doc,
    /// `Some(256)` for E2B) -- consulted by [`CacheStrategy::Cacheless`] and
    /// [`CacheMask::Padded`] alike (both route to a PLE-aware
    /// builder); `None` for every checkpoint with no PLE tensors, and inert
    /// under [`CacheMask::Bounded`], which has no PLE concept at
    /// all -- same "unused when the arm never reads it" precedent
    /// [`Self::qk_norm`]'s own doc already sets.
    pub ple_dim: Option<u32>,
    /// [`CacheMask::Padded`] and [`CacheMask::Bounded`]: lay every
    /// windowed layer's KV cache out as a ring ([`SLIDING_KV_SYMBOL`],
    /// [`SLIDING_CACHED_LEN_INPUT`], and the builders' own `sliding_kv_ring`
    /// argument); a layer with no window keeps the full cache. Inert under
    /// [`CacheStrategy::Cacheless`], the same "unused when the arm never reads
    /// it" precedent as [`Self::qk_norm`].
    pub sliding_kv_ring: bool,
    /// Qwen3-style per-head QK-norm, consulted ONLY by
    /// [`CacheMask::Bounded`]'s arm
    /// (`mistral_cached_forward_program_with_experts_and_layer_taps`'s own
    /// `qk_norm` parameter) -- inert, safe at any value, under
    /// [`CacheStrategy::Cacheless`]/[`CacheMask::Padded`], same
    /// "unused when the arm never reads it" precedent [`Self::l_cache`]'s
    /// own doc already set. A model-global flag, not a per-layer
    /// [`LayerAttentionConfig`] field, because that builder's own signature
    /// takes it as one flat `bool` applied uniformly to every layer.
    pub qk_norm: bool,
    /// `attn_{q,k,v}.bias` presence, consulted ONLY by
    /// [`CacheMask::Bounded`]'s arm -- same inertness and
    /// model-global-not-per-layer reasoning as [`Self::qk_norm`].
    pub qkv_biases: bool,
    /// Single fused `[2, feed_forward, embedding]` gate+up weight leaf vs.
    /// two separate leaves, consulted ONLY by [`CacheMask::Bounded`]'s
    /// arm -- same inertness and model-global-not-per-layer reasoning as
    /// [`Self::qk_norm`].
    pub paired_gate_up_reduce: bool,
    /// Single fused `[query_heads + 2*kv_heads, head_dim, embedding]` QKV
    /// weight leaf vs. three separate leaves, consulted ONLY by
    /// [`CacheMask::Bounded`]'s arm -- same inertness and
    /// model-global-not-per-layer reasoning as [`Self::qk_norm`].
    pub fused_qkv_reduce: bool,
    /// LM-head repeat count for the head-cost measurement harness, consulted
    /// by [`CacheStrategy::Cacheless`] and [`CacheMask::Padded`]: `1`
    /// builds only the production head, `2` and `3` append that many minus one
    /// byte-identical duplicate head chains and return their roots as
    /// `duplicate_head_roots`. Values outside `1..=3` clamp. Inert under
    /// [`CacheMask::Bounded`], same "unused when the arm never reads it"
    /// precedent as [`Self::qk_norm`].
    pub head_repeats: u32,
    /// `true` gathers the LM head to the last new row, which is all decode and prefill sample;
    /// `false` keeps every new row's logits, which a speculative verify step and pooled
    /// embeddings read.
    pub last_row_only: bool,
    /// `true` lowers a second program with every new row's logits
    /// ([`Self::verify`]) next to the decode program, so a speculative step can
    /// check a drafted run in one forward. Off unless the family's measured
    /// verify cost pays for the drafts it checks; the family profile carries the
    /// default and a config layer overrides it.
    #[serde(default)]
    pub speculative_verify: bool,
    /// Gated-DeltaNet short convolution kernel width (`<family>.ssm.conv_kernel`),
    /// consulted only by a [`LayerKind::Gdn`] entry; `0` when `layers` holds none.
    #[serde(default)]
    pub ssm_conv_kernel: u32,
    /// Per-head key width of the recurrence (`<family>.ssm.state_size`).
    #[serde(default)]
    pub ssm_state_size: u32,
    /// Key head count of the recurrence (`<family>.ssm.group_count`).
    #[serde(default)]
    pub ssm_group_count: u32,
    /// Value head count of the recurrence (`<family>.ssm.time_step_rank`); a
    /// multiple of [`Self::ssm_group_count`].
    #[serde(default)]
    pub ssm_time_step_rank: u32,
    /// Total value width of the recurrence (`<family>.ssm.inner_size`).
    #[serde(default)]
    pub ssm_inner_size: u32,
    /// Epsilon of the recurrence's per-head norm, baked into the program as a
    /// constant (`<family>.attention.layer_norm_rms_epsilon`, `1e-6` for qwen3.5).
    #[serde(default)]
    pub ssm_epsilon: f32,
    /// Whether the checkpoint stores the recurrence's value heads reordered
    /// (`<family>.ssm.v_head_reordered`).
    #[serde(default)]
    pub v_head_reordered: bool,
    /// Width of the shared expert a [`FfnCombination::RoutedWithSharedExpert`]
    /// layer runs next to its routed experts
    /// (`<family>.expert_shared_feed_forward_length`); `0` when no layer has one.
    #[serde(default)]
    pub expert_shared_feed_forward: u32,
    /// `Some(width)` lowers the program for exactly `width` new positions
    /// instead of a symbolic count. A recurrent layer unrolls its scan in Rust,
    /// so a batched prefill needs the length at lowering time; `None` is the
    /// per-step program every decode call resolves dynamically.
    #[serde(default)]
    pub prefill_width: Option<u32>,
    /// Lowers attention layers as the gated variant a recurrent-hybrid stack
    /// pairs with its recurrent layers: `attn_q` projects `[Q | gate]` per head,
    /// the gate multiplies the attention output through a sigmoid, per-head
    /// QK-norm applies, and only the leading rotary width of each head rotates.
    /// A schedule holding a [`LayerKind::Gdn`] layer needs it; an all-attention
    /// stack of the same family sets it too, which is why it is a field and not
    /// inferred from the layer kinds.
    #[serde(default)]
    pub gated_attention: bool,
}

impl ModelDescriptor {
    /// The descriptor of the verify program: this one with every new row's
    /// logits kept. `None` unless [`Self::speculative_verify`] is set, the
    /// program samples a last row (a pooled-embedding program has no draft to
    /// check), and every layer rewinds by truncating its cache
    /// ([`LayerKind::Attention`]; a state-space or short-conv layer carries
    /// state a rejected draft would have to undo).
    #[must_use]
    pub fn verify(&self) -> Option<Self> {
        let rewinds = self.layers.iter().all(|layer| layer.kind == LayerKind::Attention);
        (self.speculative_verify && self.last_row_only && rewinds).then(|| Self {
            last_row_only: false,
            ..self.clone()
        })
    }
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
            !self.speculative_verify || self.verify().is_some() || !self.last_row_only,
            "speculative_verify",
            "needs layers that rewind by truncating their cache",
        );
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
    /// Lower the verify program by default ([`ModelDescriptor::speculative_verify`]):
    /// set only for a family whose measured verify step costs less than the
    /// tokens its drafts save.
    #[serde(default)]
    pub speculative_verify: bool,
    /// Which header reader fills this family's [`ModelDescriptor`]: the layout
    /// the GGUF header uses to spell the layer schedule. Config selects a
    /// compiled reader; a family whose header already reads as one of these
    /// layouts is a profile file with no Rust.
    #[serde(default)]
    pub schedule_source: ScheduleSource,
    /// The decode-time cache shape the lowered program needs from the runtime
    /// loop, read once per load. A value other than [`KvCacheShape::Uniform`]
    /// keeps the family off the placed single-range cached program.
    #[serde(default)]
    pub kv_cache_shape: KvCacheShape,
    /// Whether the feed-forward routes through experts with a pre-gather
    /// execution protocol ([`FfnRouting::Routed`]) or evaluates the same FFN
    /// every layer.
    #[serde(default)]
    pub ffn_routing: FfnRouting,
    /// Measured default for the number of command-buffer chunks one decode
    /// step is submitted in. `1` unless this family's decode was measured to
    /// gain from chunked submission; an explicit caller value always wins.
    #[serde(default = "default_command_buffer_chunks")]
    pub command_buffer_chunks: u32,
}

const fn default_command_buffer_chunks() -> u32 {
    1
}

/// How a family's GGUF header spells its layer schedule, as data on
/// [`FamilyProfile::schedule_source`]. Each variant names one compiled header
/// reader; none names a family.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleSource {
    /// Scalar `{family}.*` keys describe every layer alike; a layer whose
    /// `head_count_kv` entry is zero is a short-convolution layer.
    #[default]
    Uniform,
    /// A per-layer sliding-window pattern, trailing shared-KV layers and
    /// per-layer widths.
    SlidingPattern,
    /// A scalar `full_attention_interval` places an attention layer every
    /// interval; the layers between are gated delta net, over a dense FFN.
    RecurrentInterval,
    /// [`Self::RecurrentInterval`] over a routed FFN with a shared expert.
    RecurrentRoutedInterval,
}

/// The decode-time shape of a lowered program's KV cache, as data on
/// [`FamilyProfile::kv_cache_shape`]. Distinct from [`CacheStrategy`], which
/// picks the lowering engine: this describes what the already-built program's
/// cache needs from the decode loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheShape {
    /// One uniform per-layer attention cache, which the single-range cached
    /// program covers.
    #[default]
    Uniform,
    /// Layers the single-range program has no concept of: recurrent state
    /// beside attention, or shared-KV and per-layer widths.
    Custom,
    /// Routed FFN and recurrent state whose decode-step cache leaves stay
    /// device-resident and segment-isolated together.
    Monolithic,
}

/// Whether a family's feed-forward is evaluated unconditionally every layer or
/// routed through experts under the pre-gather protocol, as data on
/// [`FamilyProfile::ffn_routing`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnRouting {
    #[default]
    Dense,
    Routed,
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
/// [`CacheMask::Bounded`] makes [`build_forward`] dispatch straight to
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
#[expect(clippy::too_many_arguments, reason = "mirrors the builder's own flat positional signature this descriptor replaces -- see build_forward's Bounded arm, which reads every one of these fields straight back off the descriptor it builds")]
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
        // no `LayerKind::ShortConv` layer ever appears in a `CacheMask::Bounded`
        // schedule, and that arm's own builder never reads `l_cache` --
        // same inertness as `Self::l_cache`'s own doc.
        l_cache: 0,
        embedding_scale: profile.embedding_scale,
        logit_softcap: None,
        logit_scale: None,
        residual_scale: None,
        layers,
        cache_strategy: CacheStrategy::Cached,
        cache_mask: CacheMask::Bounded,
        // `CacheMask::Bounded` has no PLE concept at all -- same
        // inertness as `Self::qk_norm`'s own doc.
        ple_dim: None,
        sliding_kv_ring: false,
        qk_norm,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        head_repeats: 1,
        last_row_only: true,
        speculative_verify: profile.speculative_verify,
        ssm_conv_kernel: 0,
        ssm_state_size: 0,
        ssm_group_count: 0,
        ssm_time_step_rank: 0,
        ssm_inner_size: 0,
        ssm_epsilon: 0.0,
        v_head_reordered: false,
        expert_shared_feed_forward: 0,
        prefill_width: None,
        gated_attention: false,
    }
}

/// [`build_forward`]'s return: the lowered program with every root a caller
/// reads back out of it. A root an engine does not produce is the empty value
/// of its type (`None`, an empty `Vec`), never a placeholder node.
#[derive(Debug, Clone)]
pub struct ForwardProgram {
    pub program: Vec<Op>,
    /// The root a decode step samples from.
    pub logits: NodeId,
    /// One entry per layer that carries or reads a cache, in layer order:
    /// [`LayerCacheRoots::Attention`] for a layer that owns a KV cache,
    /// [`LayerCacheRoots::SharedFromLayer`] for a gemma4 shared-KV layer that
    /// reads another layer's. Empty under [`CacheStrategy::Cacheless`], which
    /// keeps no cache.
    pub layer_roots: Vec<LayerCacheRoots>,
    /// One [`MoeSite`] per routed layer; empty for a dense program.
    pub moe_sites: MoeSites,
    /// One residual root per layer, populated only by [`CacheMask::Bounded`]'s
    /// engine (`mistral_cached_forward_program_with_experts_and_layer_taps`).
    pub layer_residuals: Vec<NodeId>,
    /// The last-norm activation `logits` projects from, which a pooled
    /// embedding reads. `None` for the engines that expose no hidden node.
    pub hidden: Option<NodeId>,
    /// `PROXIMA_HEAD_REPEATS`'s scratch output
    /// ([`ModelDescriptor::head_repeats`]): the duplicate head chains' roots.
    pub duplicate_head_roots: Vec<NodeId>,
    /// One diagnostic boundary per layer, populated only by the recurrent
    /// hybrid engine's routed arm; empty for every other engine.
    pub layer_diagnostics: Vec<MoeLayerDiagnostics>,
}

/// One [`LayerCacheRoots`] per layer for a cached program: the engine returns
/// one [`CachedLayerRoots`] per layer whose [`KeySourceKind`] is
/// [`KeySourceKind::ProjectedK`] (a shared-KV layer owns no cache leaves),
/// and this zips them back against the schedule so the result has one entry
/// per layer index. A count that disagrees means an engine pushed a different
/// number of roots than the schedule declares cache-owning layers.
pub(super) fn layer_roots_from_cache(
    layers: &[LayerSchedule],
    cache_roots: Vec<CachedLayerRoots>,
) -> Result<Vec<LayerCacheRoots>, TensorError> {
    let expected = layers
        .iter()
        .filter(|layer| layer.attention.key_source_kind == KeySourceKind::ProjectedK)
        .count();
    let produced = cache_roots.len();
    let mismatch = || TensorError::CacheRootsCountMismatch { produced, expected };
    let mut cache_roots = cache_roots.into_iter();
    let layer_roots = layers
        .iter()
        .map(|layer| match layer.attention.key_source_kind {
            KeySourceKind::ProjectedK => cache_roots
                .next()
                .map(LayerCacheRoots::Attention)
                .ok_or_else(mismatch),
            KeySourceKind::SharedFromLayer(source) => Ok(LayerCacheRoots::SharedFromLayer(source)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    match cache_roots.next() {
        Some(_) => Err(mismatch()),
        None => Ok(layer_roots),
    }
}

pub(super) fn refuse_when(
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
/// rewriting none of either engine's math. [`CacheMask::Padded`] calls
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
/// [`CacheMask::Bounded`] dispatches to
/// `mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing`
/// directly, passing this descriptor's own `attention.rope_pairing` rather
/// than re-inferring it from `qk_norm` -- see that variant's own doc for why
/// (a genuinely different cache-scoring algebra, not a knob the other two
/// engines can express).
/// That builder's per-layer residual outputs are this function's own fifth
/// return element, empty for [`CacheStrategy::Cacheless`]/
/// [`CacheMask::Padded`] (neither engine tracks them) -- the same
/// "degenerate default for an engine that does not produce this value"
/// precedent `cache_roots` itself already sets on the [`Cacheless`][CacheStrategy::Cacheless]
/// arm above.
///
/// A descriptor that sets [`ModelDescriptor::gated_attention`] or holds a
/// [`LayerKind::Gdn`] layer is the recurrent-hybrid engine's: the layer kinds
/// come from `layers[i].kind` and the recurrence shape from the descriptor's
/// `ssm_*` fields, and it keeps its own state cache beside the KV cache, so [`ModelDescriptor::cache_strategy`] and
/// [`ModelDescriptor::cache_mask`] do not select it.
///
/// [`ForwardProgram`]'s own doc names each root and which engine fills it.
pub fn build_forward(descriptor: &ModelDescriptor) -> Result<ForwardProgram, TensorError> {
    if descriptor.gated_attention || descriptor.layers.iter().any(|layer| layer.kind == LayerKind::Gdn) {
        return if descriptor.layers.iter().any(|layer| layer.ffn.combination == FfnCombination::RoutedWithSharedExpert) {
            hybrid_routed_forward(descriptor)
        } else {
            hybrid_dense_forward(descriptor)
        };
    }
    match (descriptor.cache_strategy, descriptor.cache_mask) {
        (CacheStrategy::Cached, CacheMask::Padded) => {
            refuse_when(
                descriptor.logit_scale.is_some(),
                "build_forward(CacheMask::Padded)",
                "a logit scale",
            )?;
            refuse_when(
                descriptor.residual_scale.is_some(),
                "build_forward(CacheMask::Padded)",
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
            Ok(ForwardProgram {
                program,
                logits,
                layer_roots: layer_roots_from_cache(&descriptor.layers, cache_roots)?,
                moe_sites,
                layer_residuals: Vec::new(),
                hidden: None,
                duplicate_head_roots,
                layer_diagnostics: Vec::new(),
            })
        }
        (CacheStrategy::Cacheless, _) => {
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
            Ok(ForwardProgram {
                program,
                logits,
                layer_roots: Vec::new(),
                moe_sites,
                layer_residuals: Vec::new(),
                hidden: None,
                duplicate_head_roots,
                layer_diagnostics: Vec::new(),
            })
        }
        (CacheStrategy::Cached, CacheMask::Bounded) => {
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
                    builder: "build_forward(CacheMask::Bounded)",
                    feature: "non-uniform per-layer attention config",
                });
            }
            refuse_when(
                descriptor.layers.iter().any(|layer| layer.kind != LayerKind::Attention),
                "build_forward(CacheMask::Bounded)",
                "a layer that is not attention",
            )?;
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
                "build_forward(CacheMask::Bounded)",
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
            Ok(ForwardProgram {
                program,
                logits: roots.logits,
                layer_roots: layer_roots_from_cache(&descriptor.layers, cache_roots)?,
                moe_sites,
                layer_residuals,
                hidden: Some(roots.hidden),
                duplicate_head_roots: Vec::new(),
                layer_diagnostics: Vec::new(),
            })
        }
    }
}
