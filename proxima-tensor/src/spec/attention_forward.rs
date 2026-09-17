use super::*;

/// Where [`append_attention_mixer`] reads its per-head `V` tensor from.
/// [`Self::Projected`] is every existing caller's behaviour today -- `V` is
/// its own weight-projected tensor, computed here node-for-node the way this
/// function always has. [`Self::SharedWithKey`] is the shape Gemma 3n/4's
/// shared-KV layers need (no `attn_v.weight` tensor at all; the key
/// projection's own raw output stands in as `V`) -- unused by any caller in
/// this crate yet, added here so the caller that needs it does not have to
/// fork this function to get it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSource {
    Projected(NodeId),
    SharedWithKey,
}

/// [`ValueSource`] without the resolved weight [`NodeId`] -- a schedule
/// entry names WHICH shape a layer's `V` takes, but the actual `wv` leaf (if
/// any) is only known once [`lfm2_forward_program_with_experts`]'s own loop
/// reaches that layer and can build (or skip) its `attn_v.weight` leaf, so
/// [`LayerAttentionConfig`] carries this kind rather than a [`ValueSource`]
/// itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSourceKind {
    ProjectedV,
    SharedWithKey,
}

/// Names one RoPE table a layer reads its `cos`/`sin` from, by the exact
/// [`Op::Input`] leaf names [`lfm2_forward_program_with_experts`] declares
/// for it -- e.g. `("rope_cos", "rope_sin")`. Every layer naming the SAME
/// pair shares the SAME declared leaf (declared once, at first use), so a
/// heterogeneous schedule with two distinct windows (Gemma 4's sliding vs
/// full layers) declares two leaf pairs total, not one per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeTableSel {
    pub cos_name: &'static str,
    pub sin_name: &'static str,
}

/// A multiplier applied to the embedding lookup's own output before the
/// first layer ever reads it -- Gemma's `hidden_states = hidden_states *
/// sqrt(embedding)` step, absent from every architecture
/// [`lfm2_forward_program_with_experts`] served before this knob existed.
/// `None` (every caller in this crate today) reproduces the prior
/// unscaled embedding node-for-node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingScale {
    /// Multiply by `sqrt(embedding)`.
    Sqrt,
}

/// The nonlinearity [`append_activation`] composes for an FFN's
/// gate/up product -- FFN activation was a hardcoded `sigmoid(gate) * gate`
/// (SiLU) chain buried in [`append_dense_swiglu_ffn`] and
/// [`append_moe_round_output`] until Gemma 4's GeGLU (`gelu_pytorch_tanh`)
/// needed a different nonlinearity on the same graph shape. `Silu` (every
/// caller in this crate today, [`LayerFfnConfig::exclusive`]'s default)
/// reproduces the prior hardcoded chain node-for-node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `silu(x) = x * sigmoid(x)`.
    Silu,
    /// `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x +
    /// 0.044715 * x^3)))` -- Gemma's GeGLU activation.
    GeluTanh,
}

/// Composes [`Activation`]'s chosen nonlinearity from elementwise/reduce
/// primitives -- the single call site [`append_dense_swiglu_ffn`] and
/// [`append_moe_round_output`] both route their gate activation through, so
/// a layer's [`LayerFfnConfig::activation`] governs both the dense and
/// routed FFN branches identically. `ones` is the caller's own
/// `scalar_constant(program, 1.0)` node, reused rather than rebuilt so
/// [`Activation::Silu`]'s call sites stay byte-identical to the prior
/// inline chain.
pub(super) fn append_activation(
    program: &mut Vec<Op>,
    x: NodeId,
    ones: NodeId,
    activation: Activation,
) -> Result<NodeId, TensorError> {
    match activation {
        Activation::Silu => {
            let neg_x = elementwise(program, DType::Float32, ScalarOp::Negate, &[(x, "sg->sg")])?;
            let exp_neg_x = elementwise(
                program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(neg_x, "sg->sg")],
            )?;
            let one_plus_exp = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(exp_neg_x, "sg->sg"), (ones, "->sg")],
            )?;
            let sigmoid_x = elementwise(
                program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(one_plus_exp, "sg->sg")],
            )?;
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x, "sg->sg"), (sigmoid_x, "sg->sg")],
            )
        }
        Activation::GeluTanh => {
            let half = scalar_constant(program, 0.5);
            let cubic_coeff = scalar_constant(program, 0.044_715);
            let sqrt_two_over_pi = scalar_constant(program, 0.797_884_6);

            let x_squared = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x, "sg->sg"), (x, "sg->sg")],
            )?;
            let x_cubed = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x_squared, "sg->sg"), (x, "sg->sg")],
            )?;
            let cubic_term = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x_cubed, "sg->sg"), (cubic_coeff, "->sg")],
            )?;
            let inner = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(x, "sg->sg"), (cubic_term, "sg->sg")],
            )?;
            let scaled_inner = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(inner, "sg->sg"), (sqrt_two_over_pi, "->sg")],
            )?;
            let tanh_term = elementwise(
                program,
                DType::Float32,
                ScalarOp::Tanh,
                &[(scaled_inner, "sg->sg")],
            )?;
            let one_plus_tanh = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(tanh_term, "sg->sg"), (ones, "->sg")],
            )?;
            let half_x = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x, "sg->sg"), (half, "->sg")],
            )?;
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(half_x, "sg->sg"), (one_plus_tanh, "sg->sg")],
            )
        }
    }
}

/// How a layer's post-attention output and its feed-forward output combine
/// -- [`FfnCombination::Exclusive`] is [`lfm2_forward_program_with_experts`]'s
/// prior behaviour (a layer runs the dense-triple FFN XOR
/// [`append_moe_ffn`], selected by `leading_dense_block_count`).
/// [`FfnCombination::ParallelDenseMoe`] is Gemma 4's shape: BOTH FFNs run
/// over the same normalized input and their outputs are summed, each
/// (optionally) normalized on its own before the sum, and the sum
/// (optionally) normalized again -- see [`LayerFfnConfig`]'s own fields for
/// which sub-norms apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnCombination {
    Exclusive,
    ParallelDenseMoe,
}

/// One layer's post-attention/feed-forward knobs -- generalizes
/// [`lfm2_forward_program_with_experts`]'s previously-uniform "ffn_norm,
/// then dense-triple XOR routed FFN, then residual add" sequence the same
/// way [`LayerAttentionConfig`] generalized the attention sub-block, so a
/// heterogeneous schedule (Gemma 4's parallel dense+MoE, its
/// `post_attention_norm`, its per-layer `layer_output_scale`) can vary
/// these per layer while every uniform caller in this crate today
/// reproduces the prior program byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerFfnConfig {
    /// `true` builds and applies `blk.{layer}.post_attention_norm.weight`
    /// to the attention sub-block's output before its residual add
    /// (threaded into [`append_attention_mixer`]). `false` (every caller
    /// today) reproduces the prior unnormalized residual add.
    pub post_attention_norm: bool,
    /// [`FfnCombination::Exclusive`] for every caller in this crate today.
    pub combination: FfnCombination,
    /// Sub-norm placement [`FfnCombination::ParallelDenseMoe`] applies --
    /// ignored under [`FfnCombination::Exclusive`]. Gemma names these
    /// `blk.{layer}.post_ffw_norm_1.weight` (dense branch),
    /// `blk.{layer}.post_ffw_norm_2.weight` (routed branch), and
    /// `blk.{layer}.post_ffw_norm.weight` (post-sum).
    pub dense_post_norm: bool,
    pub routed_post_norm: bool,
    pub combined_post_norm: bool,
    /// `true` builds `blk.{layer}.layer_output_scale.weight` (a rank-0
    /// leaf) and multiplies it into this layer's output right after the
    /// FFN residual add. `false` (every caller today) reproduces the prior
    /// unscaled residual.
    pub output_scale: bool,
    /// Gating function [`append_routed_expert_ffn`] applies to the routed
    /// branch's router logits. `Sigmoid` (every caller in this crate today,
    /// [`FfnCombination::Exclusive`]'s own prior hardcoded choice) reproduces
    /// LFM2's own MoE softmax-free routing; Gemma 4 uses `Softmax`.
    pub routed_gating: ExpertGatingFunc,
    /// `true` (every caller today) binds `blk.{layer}.exp_probs_b.bias` and
    /// adds it into the router logits before argmax selection, exactly
    /// [`append_routed_expert_ffn`]'s prior hardcoded leaf. Gemma 4 has no
    /// such bias on its routed branch and sets `false`.
    pub routed_expert_bias: bool,
    /// `true` normalizes the routed branch's own input through
    /// `blk.{layer}.pre_ffw_norm_2.weight` instead of reusing the dense
    /// branch's `ffn_norm`-normed input -- [`FfnCombination::ParallelDenseMoe`]
    /// only; ignored under [`FfnCombination::Exclusive`]. `false` (every
    /// caller today) reproduces the prior shared-input behaviour. Gemma 4
    /// sets `true`.
    pub routed_pre_norm: bool,
    /// `true` binds `blk.{layer}.ffn_gate_inp.scale` (`[embedding]`) as the
    /// gamma of a `with_scale=False` RMSNorm over the router's own input,
    /// then multiplies the normed result by the constant
    /// `embedding**-0.5`, before the router projection -- never into the
    /// experts' own input (`Gemma4TextRouter.forward`). `false` (every
    /// caller today) reproduces the prior unscaled router input. Gemma 4
    /// sets `true`.
    pub router_scale: bool,
    /// `true` binds `blk.{layer}.ffn_down_exps.scale` (`[expert_count]`)
    /// and folds it, gathered by each round's selected expert, into that
    /// round's combination weight AFTER softmax-over-selected
    /// renormalization ([`append_moe_ffn`]'s own doc on `MoeFfnSpec::expert_scale`).
    /// `false` (every caller today) reproduces the prior unscaled
    /// combination. Gemma 4 sets `true`.
    pub expert_output_scale: bool,
    /// Nonlinearity [`append_activation`] applies to both the dense and
    /// routed branches' gate/up product. `Silu` (every caller in this
    /// crate today) reproduces the prior hardcoded SiLU chain
    /// node-for-node; Gemma 4 sets `GeluTanh` for its GeGLU FFN.
    pub activation: Activation,
}

impl LayerFfnConfig {
    /// [`lfm2_forward_program_with_experts`]'s prior fixed behaviour: no
    /// post-attention norm, exclusive dense/routed FFN selection, no
    /// sub-norms, no output scale.
    #[must_use]
    pub const fn exclusive() -> Self {
        LayerFfnConfig {
            post_attention_norm: false,
            combination: FfnCombination::Exclusive,
            dense_post_norm: false,
            routed_post_norm: false,
            combined_post_norm: false,
            output_scale: false,
            routed_gating: ExpertGatingFunc::Sigmoid,
            routed_expert_bias: true,
            routed_pre_norm: false,
            router_scale: false,
            expert_output_scale: false,
            activation: Activation::Silu,
        }
    }
}

/// How a [`LayerAttentionConfig`] layer scales its raw attention scores
/// before the causal mask -- see [`LayerAttentionConfig::score_scale`]'s own
/// doc for which architecture uses which variant and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionScoreScale {
    /// `1/sqrt(query_pre_attn_scalar)`, the Gemma 2/3 convention. LFM2 and
    /// every other pre-existing caller passes its own `head_dim` here,
    /// reproducing the old always-`1/sqrt(head_dim)` behaviour.
    InverseSqrtQueryPreAttnScalar(u32),
    /// No score scaling (`self.scaling = 1.0`): Gemma 4's `Gemma4TextAttention`,
    /// sliding and full layers alike.
    Unscaled,
}

/// One [`LayerKind::Attention`] block's own attention shape --
/// [`lfm2_forward_program_with_experts`]'s per-layer generalization of the
/// single crate-wide `head_dim`/`kv_heads` it used to compute once outside
/// its layer loop. Every field here is exactly what varies across Gemma 4's
/// sliding (`head_dim=256, kv_heads=8`) vs full (`head_dim=512, kv_heads=2`)
/// layers today, plus [`ValueSource`]'s and [`RopePairing`]'s own per-call
/// knobs [`append_attention_mixer`] already generalized in the slice before
/// this one -- `query_heads` stays a top-level uniform parameter because
/// every caller in this crate still needs it uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAttentionConfig {
    pub head_dim: u32,
    pub kv_heads: u32,
    /// `None`/`Some(0)` is [`causal_mask_windowed`]'s own unwindowed
    /// (whole-prefix causal) case.
    pub mask_window: Option<u32>,
    pub value_source_kind: ValueSourceKind,
    pub rope_table: RopeTableSel,
    pub rope_pairing: RopePairing,
    /// The multiplier applied to raw attention scores before the causal
    /// mask. [`AttentionScoreScale::InverseSqrtQueryPreAttnScalar`] is the
    /// Gemma 2/3 spec's `query_pre_attn_scalar` convention (queries scaled
    /// by `1/sqrt(query_pre_attn_scalar)`, a FIXED value that REPLACES
    /// `1/sqrt(head_dim)`); every caller of this field before
    /// [`AttentionScoreScale`] existed passed `head_dim` itself here,
    /// reproducing `1/sqrt(head_dim)` byte-for-byte. Gemma 4 has no
    /// `query_pre_attn_scalar` at all -- HF `Gemma4TextAttention` hard-codes
    /// `self.scaling = 1.0` for both sliding and full layers, relying on
    /// QK-norm instead -- so gemma4's layers use
    /// [`AttentionScoreScale::Unscaled`].
    pub score_scale: AttentionScoreScale,
    /// Gemma 4's `v_norm` (`Gemma4TextAttention.forward`,
    /// `modeling_gemma4.py:1256-1265`): a per-kv-head RMSNorm applied to `V`
    /// AFTER value projection/sharing and BEFORE the attention product, with
    /// NO learned scale (`Gemma4RMSNorm(head_dim, eps, with_scale=False)`) --
    /// there is no `attn_v_norm.weight` tensor on disk for it to read. `V`
    /// stays un-roped either way; this only changes whether it is
    /// normalized. Every caller before this field existed (LFM2 and every
    /// other architecture this crate serves) sets this `false`, reproducing
    /// the prior raw-`V` behaviour byte-for-byte; Gemma 4 sets it `true`.
    pub value_norm: bool,
}

/// [`lfm2_forward_program_with_experts`]'s own per-attention-layer bundle:
/// the shared nodes ONE [`LayerAttentionConfig`] resolves to, already
/// deduplicated against every other layer's own config. Kept separate from
/// [`LayerAttentionConfig`] itself since these are [`NodeId`]s already
/// placed in the program, never a caller-facing description.
struct AttentionLayerResources {
    group: u32,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
}

/// Returns `cache`'s existing value for `key` if one is already there,
/// otherwise runs `build`, appends `(key, value)`, and returns the fresh
/// value -- the linear-scan dedup [`lfm2_forward_program_with_experts`]'s
/// pre-pass uses for every shared per-attention-config resource except the
/// RoPE table (one extra field) and the causal mask (fallible), which
/// inline the same scan-then-insert shape directly.
fn find_or_insert<Key, Value>(
    cache: &mut Vec<(Key, Value)>,
    key: Key,
    build: impl FnOnce() -> Value,
) -> Value
where
    Key: PartialEq,
    Value: Copy,
{
    for (candidate, value) in cache.iter() {
        if *candidate == key {
            return *value;
        }
    }
    let value = build();
    cache.push((key, value));
    value
}

/// The decode loop only ever samples the LAST row's logits, prefill or
/// not (`proxima-model-interop::generate`'s own `logits[(new_count - 1)
/// * vocab_size..]` slice, every call site) -- greedy sampling needs one
/// row, never the whole prefill. Slicing here, before the vocab-sized
/// `output.weight` matmul, is what turns a 915-row Q6K reduce into a
/// 1-row one on an 850+-token prefill (`docs/discipline.md` ROW 418's
/// own `output.weight` measurement: 14.7s of 43.8s GPU time, 33% of
/// total, on the FULL 915-row projection). [`embedding_lookup`] is reused
/// verbatim, not a new primitive: it is already exactly `table[ids[s],
/// d]`, the same [`IndexMap::Computed`] gather this needs, just with a
/// 1-entry `lm_head_row` index instead of a `new_count`-entry `ids`.
/// `lm_head_row` is host-supplied (`new_count - 1`, same convention as
/// `cached_len`/`ids` above) rather than derived in-graph from
/// `Extent::Symbolic(0)`: `cpu.rs`'s own
/// `evaluate_typed_names_a_computed_gather_index_node_as_not_yet_supported`
/// test is this crate's own proof that an in-program-computed gather
/// index (an `Op::Iota`/`Op::Reduce` chain, not a caller-supplied
/// `Op::Input` block) is a named `NotLowerable` gap on the typed
/// evaluator, not a silently-guessed execution path -- a host-supplied
/// leaf is the one gather-index shape this crate's gather machinery
/// already proves correct end to end ([`embedding_lookup`]'s own `ids`).
/// `last_row_only: false` skips this leaf entirely (not merely bypasses
/// it) so the program a `false` caller gets is byte-for-byte the one
/// this function has always built -- no new node, no new required
/// binding, every existing per-position-logits caller unaffected.
fn gather_last_row(program: &mut Vec<Op>, normed_final: NodeId, last_row_only: bool) -> NodeId {
    if last_row_only {
        let lm_head_row = input_leaf(
            program,
            DType::Int32,
            alloc::vec![Extent::Static(1)],
            "lm_head_row",
        );
        embedding_lookup(program, normed_final, lm_head_row)
    } else {
        normed_final
    }
}

/// [`append_mistral_layer`]'s attention sub-block in isolation (RoPE + GQA +
/// causal mask + residual, no FFN) -- the piece [`lfm2_forward_program_with_experts`]
/// needs on its own, since an attention block there sits beside
/// [`append_lfm2_conv_mixer`] rather than always beside the same FFN choice
/// [`append_mistral_layer`] bundles it with. Node-for-node the same attention
/// graph [`append_mistral_layer`] runs before its own FFN call, extracted
/// rather than shared by refactoring that function, so the dense Mistral/Llama
/// path's own generated program bytes never change shape because this
/// function exists next to it.
///
/// `value_source` ([`ValueSource`]), `rope_pairing` ([`RopePairing`]), the
/// `is_future`/`neg_infinity` pair (built by [`causal_mask`] or
/// [`causal_mask_windowed`] at the call site), and `post_attention_norm_weight`
/// are this mixer's per-call knobs -- the heterogeneous-layer building
/// block a caller composes per layer, rather than this function guessing a
/// model's shape from its own parameters. Every caller in this crate today
/// passes `ValueSource::Projected(wv)`, `RopePairing::Interleaved`,
/// `causal_mask_windowed(program, None)`'s output, and `None` for
/// `post_attention_norm_weight`, which reproduces this function's prior
/// fixed behaviour node-for-node. `Some(gamma)` normalizes the attention
/// sub-block's output ([`rmsnorm`] with `gamma`) BEFORE the residual add
/// below -- Gemma 4's `post_attention_norm.weight`, absent from every
/// architecture this function served before this knob existed. `value_norm`
/// applies [`rmsnorm_per_head_no_scale`] to `V` right after the
/// `value_source` match below, mirroring `k`'s own [`rmsnorm_per_head`] call
/// but with no learned scale and no RoPE -- Gemma 4's `v_norm`
/// (`Gemma4TextAttention.forward`, `modeling_gemma4.py:1256-1265`). Every
/// caller before this knob existed passes `false`, reproducing the prior
/// raw-`V` program byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn append_attention_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    value_source: ValueSource,
    wo: NodeId,
    rope_pairing: RopePairing,
    post_attention_norm_weight: Option<NodeId>,
    value_norm: bool,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;
    // [`Lfm2MoeAttention.q_layernorm`]'s own placement
    // (`modeling_lfm2_moe.py:331`): normalizes right after the head
    // reshape, BEFORE `apply_rotary_pos_emb` -- never after.
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;
    let k = rmsnorm_per_head(program, k_raw, k_norm_weight, inv_head_dim, eps, "u")?;

    let v_raw = match value_source {
        ValueSource::Projected(wv) => {
            let v_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wv, "iud->sudi")],
            )?;
            reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                v_product,
                "sudi->sudi",
                "sud->sudi",
            )?
        }
        ValueSource::SharedWithKey => k_raw,
    };
    // Gemma 4's `v_norm` (`Gemma4TextAttention.forward`,
    // `modeling_gemma4.py:1256-1265`): weightless per-kv-head RMSNorm, no
    // RoPE. Every non-Gemma-4 caller passes `value_norm: false` and gets the
    // prior raw-`V` node back unchanged.
    let v = if value_norm {
        rmsnorm_per_head_no_scale(program, v_raw, inv_head_dim, eps, "u")?
    } else {
        v_raw
    };

    let (rotated_q_even, rotated_q_odd) =
        fused_rope_pair(program, q, 'h', cos, sin, rope_pairing)?;

    let (rotated_k_even, rotated_k_odd) =
        fused_rope_pair(program, k, 'u', cos, sin, rope_pairing)?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (rotated_k_even, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->stugi"),
            (rotated_k_odd, "tui->stugi"),
        ],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;

    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let attn_out = match post_attention_norm_weight {
        Some(gamma) => rmsnorm(program, attn_out, gamma, inv_dim, eps)?,
        None => attn_out,
    };

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )
}

/// The dense-triple SwiGLU FFN branch [`lfm2_forward_program_with_experts`]
/// ran inline before [`FfnCombination::ParallelDenseMoe`] needed the same
/// six-op sequence available a second time (once for its own dense branch,
/// once for [`FfnCombination::Exclusive`]'s leading dense blocks) --
/// extracted node-for-node, so [`FfnCombination::Exclusive`]'s own call
/// site reproduces the prior inline program unchanged.
pub(crate) fn append_dense_swiglu_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    normed: NodeId,
    embedding: u32,
    feed_forward: u32,
    ones: NodeId,
    activation: Activation,
) -> Result<NodeId, TensorError> {
    let w_gate = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
        &alloc::format!("blk.{layer}.ffn_gate.weight"),
    );
    let w_up = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
        &alloc::format!("blk.{layer}.ffn_up.weight"),
    );
    let w_down = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
        &alloc::format!("blk.{layer}.ffn_down.weight"),
    );
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (w_gate, "dg->sdg")],
    )?;
    let gate = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sdg->sdg",
        "sg->sdg",
    )?;
    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (w_up, "dg->sdg")],
    )?;
    let up = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let activated_gate = append_activation(program, gate, ones, activation)?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(activated_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )
}

/// The routed MoE FFN branch [`lfm2_forward_program_with_experts`] ran
/// inline before [`FfnCombination::ParallelDenseMoe`] needed the same
/// weight leaves and [`append_moe_ffn`] call available beside
/// [`append_dense_swiglu_ffn`] -- extracted node-for-node, so
/// [`FfnCombination::Exclusive`]'s own call site (passing
/// [`LayerFfnConfig::exclusive`]'s `routed_gating: Sigmoid` and
/// `routed_expert_bias: true`) reproduces the prior inline program
/// unchanged. `gating`/`use_expert_bias` are [`LayerFfnConfig`]'s own
/// `routed_gating`/`routed_expert_bias` fields threaded straight through --
/// Gemma 4's own [`FfnCombination::ParallelDenseMoe`] caller sets
/// `Softmax`/`false`.
/// `router_input` feeds the router projection; `normed` feeds the expert
/// gate/up/down projections. Every caller before Gemma 4 passes the same
/// node for both (byte-identical to the prior single-`normed` signature).
/// Gemma 4 passes the raw post-attention residual as `router_input` and its
/// own `pre_ffw_norm_2`-normed value as `normed` -- `router_input` is never
/// the routed branch's own `pre_ffw_norm_2`-normed input; `router_scale`
/// (below) then applies `Gemma4TextRouter.forward`'s own norm/scale/root
/// transform to `router_input` before the router projection.
/// `router_scale`/`expert_output_scale` bind and fold
/// gemma4's `ffn_gate_inp.scale`/`ffn_down_exps.scale` (`false` for every
/// other caller today, so those two `Input` leaves are never declared and
/// the emitted program stays byte-for-byte the same).
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_routed_expert_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    router_input: NodeId,
    normed: NodeId,
    embedding: u32,
    expert_feed_forward: u32,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    use_expert_bias: bool,
    router_scale: bool,
    expert_output_scale: bool,
    activation: Activation,
    inv_dim: NodeId,
    eps: NodeId,
    moe_sites: &mut Vec<MoeSite>,
) -> Result<NodeId, TensorError> {
    let gate_inp = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
        &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
    );
    // Gemma4TextRouter.forward: norm(x, with_scale=False) * scale * hidden_size**-0.5,
    // then the router projection -- `ffn_gate_inp.scale` is `norm`'s own gamma
    // (bound raw, no `1 +` offset), not a post-hoc multiplier on raw resid.
    let scaled_router_input = if router_scale {
        let router_scale_weight = input_leaf(
            program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_gate_inp.scale"),
        );
        // `hidden_size**-0.5` folds into the RMSNorm gamma multiply itself
        // (`normed * (scale * root) == normed * scale * root`) rather than
        // a separate op consuming the norm's own output -- an equivalent
        // op-count-wise placement of the same constant multiply.
        let inv_sqrt_embedding = scalar_constant(program, 1.0 / (embedding as f32).sqrt());
        let rooted_router_scale_weight = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(router_scale_weight, "d->d"), (inv_sqrt_embedding, "->d")],
        )?;
        rmsnorm(
            program,
            router_input,
            rooted_router_scale_weight,
            inv_dim,
            eps,
        )?
    } else {
        router_input
    };
    let expert_scale = if expert_output_scale {
        Some(input_leaf(
            program,
            DType::Float32,
            alloc::vec![Extent::Static(expert_count)],
            &alloc::format!("blk.{layer}.ffn_down_exps.scale"),
        ))
    } else {
        None
    };
    let expert_w_gate = input_leaf(
        program,
        DType::Float32,
        alloc::vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_feed_forward)
        ],
        &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
    );
    let expert_w_up = input_leaf(
        program,
        DType::Float32,
        alloc::vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_feed_forward)
        ],
        &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
    );
    let expert_w_down = input_leaf(
        program,
        DType::Float32,
        alloc::vec![
            Extent::Static(expert_count),
            Extent::Static(expert_feed_forward),
            Extent::Static(embedding)
        ],
        &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
    );
    let expert_bias = if use_expert_bias {
        Some(input_leaf(
            program,
            DType::Float32,
            alloc::vec![Extent::Static(expert_count)],
            &alloc::format!("blk.{layer}.exp_probs_b.bias"),
        ))
    } else {
        None
    };
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scaled_router_input, "sd->sde"), (gate_inp, "de->sde")],
    )?;
    let router_logits = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sde->sde",
        "se->sde",
    )?;
    let moe_spec = MoeFfnSpec {
        router: MoeRouter::Logits(router_logits),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        gating,
        expert_bias,
        expert_scale,
        activation,
        strategy: MoeProjectionStrategy::PerRoute,
    };
    let (ffn_out, site) = append_moe_ffn(program, layer, normed, &moe_spec)?;
    moe_sites.push(site);
    Ok(ffn_out)
}

/// LFM2.5-8B-A1B's hybrid forward pass: `block_count` blocks, each either
/// `append_attention_mixer` or `append_lfm2_conv_mixer` per its own
/// `layer_kinds[layer]` (derived by [`LayerKind::from_tensor_names`] from the
/// real checkpoint's tensor directory, since `layer_types` is not a metadata
/// key this architecture writes), then a shared RMSNorm and
/// `append_moe_ffn`/dense-triple FFN exactly like
/// [`mistral_forward_program`]'s own MoE branch --
/// `leading_dense_block_count` (LFM2.5-8B-A1B: `2`) is threaded per layer
/// rather than a single crate-wide dense/MoE switch, since this checkpoint's
/// first two blocks are dense and the rest are routed.
///
/// Prefill-only: takes the whole prompt as one `[seq, embedding]` pass, the
/// same scope [`mistral_forward_program`] has. A KV-cached and
/// conv-state-cached incremental counterpart (mirroring
/// [`mistral_cached_forward_program_with_experts`]) is a further step this
/// function's own doc does not claim -- `causal_conv1d`'s masked-gather
/// composition only needs the whole sequence to be present at once, which a
/// one-token-at-a-time decode call does not have.
///
/// `ffn_configs` ([`LayerFfnConfig`]) generalizes the per-layer
/// post-attention/FFN sequence the same way `attention_configs`
/// generalized the attention sub-block -- every element
/// [`LayerFfnConfig::exclusive`] (every caller in this crate today)
/// reproduces this function's prior fixed FFN-selection and unnormalized
/// residual behaviour node-for-node. `embedding_scale`
/// ([`EmbeddingScale`]) and `logit_softcap` are `None` for every caller
/// today, reproducing the prior unscaled embedding and untransformed
/// final logits.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    expert_feed_forward: u32,
    query_heads: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    leading_dense_block_count: u32,
    l_cache: u32,
    layer_kinds: &[LayerKind],
    attention_configs: &[LayerAttentionConfig],
    ffn_configs: &[LayerFfnConfig],
    embedding_scale: Option<EmbeddingScale>,
    logit_softcap: Option<f32>,
    last_row_only: bool,
) -> Result<(Vec<Op>, NodeId, MoeSites), TensorError> {
    if layer_kinds.len() != block_count as usize {
        return Err(TensorError::LayerKindCountMismatch {
            expected: block_count,
            found: layer_kinds.len(),
        });
    }
    if attention_configs.len() != block_count as usize {
        return Err(TensorError::AttentionConfigCountMismatch {
            expected: block_count,
            found: attention_configs.len(),
        });
    }
    if ffn_configs.len() != block_count as usize {
        return Err(TensorError::FfnConfigCountMismatch {
            expected: block_count,
            found: ffn_configs.len(),
        });
    }

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);
    if let Some(scale) = embedding_scale {
        let multiplier = match scale {
            EmbeddingScale::Sqrt => (embedding as f32).sqrt(),
        };
        let multiplier = scalar_constant(&mut program, multiplier);
        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(x, "sd->sd"), (multiplier, "->sd")],
        )?;
    }

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);

    // Every shared per-attention-config resource (RoPE table, causal mask,
    // GQA broadcast constant, per-head scaling scalars) is declared ONCE per
    // DISTINCT value, at the first layer that needs it, walked in schedule
    // order -- so a uniform schedule (every caller in this crate today)
    // declares each resource exactly once, in exactly the same relative
    // order [`lfm2_forward_program_with_experts`] always declared them in,
    // before this loop existed: `inv_sqrt_head_dim`, `inv_head_dim`,
    // `rope_cos`/`rope_sin`, `group_ones`, then the causal mask. A
    // heterogeneous schedule instead grows each cache to one entry per
    // distinct value actually referenced. `attention_resources[index]` is
    // that same attention layer's own resolved bundle, in schedule order --
    // the main loop below reads it by a running counter rather than
    // searching these caches a second time.
    let mut score_scale_cache: Vec<(AttentionScoreScale, NodeId)> = Vec::new();
    let mut inv_head_dim_cache: Vec<(u32, NodeId)> = Vec::new();
    let mut rope_table_cache: Vec<(RopeTableSel, u32, NodeId, NodeId)> = Vec::new();
    let mut group_ones_cache: Vec<((u32, u32), NodeId)> = Vec::new();
    let mut mask_cache: Vec<(Option<u32>, NodeId, NodeId)> = Vec::new();
    let mut attention_resources: Vec<AttentionLayerResources> = Vec::new();

    for (layer, kind) in layer_kinds.iter().enumerate() {
        if *kind != LayerKind::Attention {
            continue;
        }
        let config = &attention_configs[layer];
        let group = query_heads / config.kv_heads;

        let inv_sqrt_head_dim = find_or_insert(&mut score_scale_cache, config.score_scale, || {
            let multiplier = match config.score_scale {
                AttentionScoreScale::InverseSqrtQueryPreAttnScalar(scalar) => {
                    1.0 / (scalar as f32).sqrt()
                }
                AttentionScoreScale::Unscaled => 1.0,
            };
            scalar_constant(&mut program, multiplier)
        });
        let inv_head_dim = find_or_insert(&mut inv_head_dim_cache, config.head_dim, || {
            scalar_constant(&mut program, 1.0 / config.head_dim as f32)
        });
        let (cos, sin) = match rope_table_cache
            .iter()
            .find(|(table, ..)| *table == config.rope_table)
        {
            Some((_, _, cos, sin)) => (*cos, *sin),
            None => {
                let pairs = config.head_dim / 2;
                let cos = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
                    config.rope_table.cos_name,
                );
                let sin = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
                    config.rope_table.sin_name,
                );
                rope_table_cache.push((config.rope_table, pairs, cos, sin));
                (cos, sin)
            }
        };
        let group_ones = find_or_insert(&mut group_ones_cache, (config.kv_heads, group), || {
            op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(config.kv_heads), Extent::Static(group)],
                    value: 1.0,
                },
            )
        });
        // `None` delegates straight to `causal_mask` (that function's own
        // doc), so a uniform never-windowed schedule reproduces the prior
        // program byte-for-byte.
        let mask_result: Result<(NodeId, NodeId), TensorError> = match mask_cache
            .iter()
            .find(|(window, ..)| *window == config.mask_window)
        {
            Some((_, is_future, neg_infinity)) => Ok((*is_future, *neg_infinity)),
            None => {
                let built = causal_mask_windowed(&mut program, config.mask_window)?;
                mask_cache.push((config.mask_window, built.0, built.1));
                Ok(built)
            }
        };
        let (is_future, neg_infinity) = mask_result?;

        attention_resources.push(AttentionLayerResources {
            group,
            inv_sqrt_head_dim,
            inv_head_dim,
            cos,
            sin,
            group_ones,
            is_future,
            neg_infinity,
        });
    }
    let mut attention_layer_index: usize = 0;
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for (layer, kind) in layer_kinds.iter().enumerate() {
        let layer = layer as u32;
        let ffn_config = &ffn_configs[layer as usize];
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let post_attention_norm_weight = if ffn_config.post_attention_norm {
            Some(input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.post_attention_norm.weight"),
            ))
        } else {
            None
        };

        let post_mixer = match kind {
            LayerKind::Attention => {
                let config = &attention_configs[layer as usize];
                let head_dim = config.head_dim;
                let kv_heads = config.kv_heads;

                let resources = &attention_resources[attention_layer_index];
                attention_layer_index += 1;
                let group = resources.group;
                let inv_sqrt_head_dim = resources.inv_sqrt_head_dim;
                let inv_head_dim = resources.inv_head_dim;
                let cos = resources.cos;
                let sin = resources.sin;
                let group_ones = resources.group_ones;
                let is_future = resources.is_future;
                let neg_infinity = resources.neg_infinity;

                let wq = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(query_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_q.weight"),
                );
                let wk = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_k.weight"),
                );
                let value_source = match config.value_source_kind {
                    ValueSourceKind::ProjectedV => {
                        let wv = input_leaf(
                            &mut program,
                            DType::Float32,
                            alloc::vec![
                                Extent::Static(embedding),
                                Extent::Static(kv_heads),
                                Extent::Static(head_dim)
                            ],
                            &alloc::format!("blk.{layer}.attn_v.weight"),
                        );
                        ValueSource::Projected(wv)
                    }
                    ValueSourceKind::SharedWithKey => ValueSource::SharedWithKey,
                };
                let wo = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(kv_heads),
                        Extent::Static(group),
                        Extent::Static(head_dim),
                        Extent::Static(embedding),
                    ],
                    &alloc::format!("blk.{layer}.attn_output.weight"),
                );
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                append_attention_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    inv_sqrt_head_dim,
                    inv_head_dim,
                    cos,
                    sin,
                    group_ones,
                    is_future,
                    neg_infinity,
                    group,
                    attn_norm_weight,
                    q_norm_weight,
                    k_norm_weight,
                    wq,
                    wk,
                    value_source,
                    wo,
                    config.rope_pairing,
                    post_attention_norm_weight,
                    config.value_norm,
                )?
            }
            LayerKind::ShortConv => {
                // `b_proj`/`c_proj`/`x_proj` are the real checkpoint's single
                // fused `blk.{layer}.shortconv.in_proj.weight`
                // (`[embedding, 3*embedding]`) split three ways -- see
                // `append_lfm2_conv_mixer`'s own doc for why this graph
                // cannot instead slice one fused `Input` by offset. Binding
                // these three names from that one on-disk tensor is a
                // binder-side split this session does not implement; the
                // names here are this program's contract for whoever does.
                let b_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.b"),
                );
                let c_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.c"),
                );
                let x_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.x"),
                );
                // `[embedding, l_cache]`, NOT `[l_cache, embedding]` --
                // `causal_conv1d`'s own doc on its `dl->sld` map explains why:
                // the real on-disk tensor has `l_cache` as its fastest axis,
                // and `row_major_strides` (`bind.rs`) makes the LAST declared
                // shape axis the fastest one.
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(l_cache)],
                    &alloc::format!("blk.{layer}.shortconv.conv.weight"),
                );
                let out_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.out_proj.weight"),
                );
                append_lfm2_conv_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    attn_norm_weight,
                    b_proj,
                    c_proj,
                    x_proj,
                    conv_weight,
                    out_proj,
                    l_cache,
                )?
            }
        };

        let normed2 = rmsnorm(&mut program, post_mixer, ffn_norm_weight, inv_dim, eps)?;

        let ffn_out = match ffn_config.combination {
            FfnCombination::Exclusive => {
                if layer < leading_dense_block_count {
                    append_dense_swiglu_ffn(
                        &mut program,
                        layer,
                        normed2,
                        embedding,
                        feed_forward,
                        ones,
                        ffn_config.activation,
                    )?
                } else {
                    append_routed_expert_ffn(
                        &mut program,
                        layer,
                        normed2,
                        normed2,
                        embedding,
                        expert_feed_forward,
                        expert_count,
                        expert_used_count,
                        ones,
                        ffn_config.routed_gating,
                        ffn_config.routed_expert_bias,
                        false,
                        false,
                        ffn_config.activation,
                        inv_dim,
                        eps,
                        &mut moe_sites,
                    )?
                }
            }
            FfnCombination::ParallelDenseMoe => {
                let dense_out = append_dense_swiglu_ffn(
                    &mut program,
                    layer,
                    normed2,
                    embedding,
                    feed_forward,
                    ones,
                    ffn_config.activation,
                )?;
                let dense_out = if ffn_config.dense_post_norm {
                    let gamma = input_leaf(
                        &mut program,
                        DType::Float32,
                        alloc::vec![Extent::Static(embedding)],
                        &alloc::format!("blk.{layer}.post_ffw_norm_1.weight"),
                    );
                    rmsnorm(&mut program, dense_out, gamma, inv_dim, eps)?
                } else {
                    dense_out
                };

                let routed_input = if ffn_config.routed_pre_norm {
                    let gamma = input_leaf(
                        &mut program,
                        DType::Float32,
                        alloc::vec![Extent::Static(embedding)],
                        &alloc::format!("blk.{layer}.pre_ffw_norm_2.weight"),
                    );
                    rmsnorm(&mut program, post_mixer, gamma, inv_dim, eps)?
                } else {
                    normed2
                };
                // `routed_pre_norm` also marks the authoritative-graph case
                // (gemma4) where the router's own input is the RAW
                // post-attention residual BEFORE `append_routed_expert_ffn`
                // applies its own `router_scale` norm/scale/root -- never
                // the routed branch's `pre_ffw_norm_2`-normed input; experts
                // still consume `routed_input`, only the router's own
                // projection input differs.
                let router_input = if ffn_config.routed_pre_norm {
                    post_mixer
                } else {
                    routed_input
                };
                let routed_out = append_routed_expert_ffn(
                    &mut program,
                    layer,
                    router_input,
                    routed_input,
                    embedding,
                    expert_feed_forward,
                    expert_count,
                    expert_used_count,
                    ones,
                    ffn_config.routed_gating,
                    ffn_config.routed_expert_bias,
                    ffn_config.router_scale,
                    ffn_config.expert_output_scale,
                    ffn_config.activation,
                    inv_dim,
                    eps,
                    &mut moe_sites,
                )?;
                let routed_out = if ffn_config.routed_post_norm {
                    let gamma = input_leaf(
                        &mut program,
                        DType::Float32,
                        alloc::vec![Extent::Static(embedding)],
                        &alloc::format!("blk.{layer}.post_ffw_norm_2.weight"),
                    );
                    rmsnorm(&mut program, routed_out, gamma, inv_dim, eps)?
                } else {
                    routed_out
                };

                let combined = elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(dense_out, "sd->sd"), (routed_out, "sd->sd")],
                )?;
                if ffn_config.combined_post_norm {
                    let gamma = input_leaf(
                        &mut program,
                        DType::Float32,
                        alloc::vec![Extent::Static(embedding)],
                        &alloc::format!("blk.{layer}.post_ffw_norm.weight"),
                    );
                    rmsnorm(&mut program, combined, gamma, inv_dim, eps)?
                } else {
                    combined
                }
            }
        };

        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            &[(ffn_out, "sd->sd"), (post_mixer, "sd->sd")],
        )?;

        if ffn_config.output_scale {
            let output_scale = input_leaf(
                &mut program,
                DType::Float32,
                Vec::new(),
                &alloc::format!("blk.{layer}.layer_output_scale.weight"),
            );
            x = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x, "sd->sd"), (output_scale, "->sd")],
            )?;
        }
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let normed_last = gather_last_row(&mut program, normed_final, last_row_only);

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    let logits = match logit_softcap {
        Some(cap) => {
            let cap_node = scalar_constant(&mut program, cap);
            let scaled = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Divide,
                &[(logits, "sv->sv"), (cap_node, "->sv")],
            )?;
            let tanh = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Tanh,
                &[(scaled, "sv->sv")],
            )?;
            elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(tanh, "sv->sv"), (cap_node, "->sv")],
            )?
        }
        None => logits,
    };

    Ok((program, logits, MoeSites(moe_sites)))
}

/// [`mistral_forward_program`]'s key/value-cached counterpart: the same
/// architecture, but `ids`/`rope_cos`/`rope_sin` carry only the `new`
/// positions this call introduces (symbol 0), attention also draws on a
/// per-layer already-rotated key/value cache sized by symbol 1
/// (`kv_cache.{layer}.k_even`/`k_odd`/`v`, bound [`Op::Input`]s each layer's
/// own online-softmax attention combines with its freshly computed
/// key/value), and the returned roots are `(logits,
/// per_layer_cache_roots)` instead of one implicit last-node root, since a
/// caller now needs the per-layer [`CachedLayerRoots`] to grow its cache for
/// the next call. A caller passes `symbols = [new_positions, cached_len]`
/// to [`crate::shape::infer`]/[`crate::cpu::evaluate_quantized_named`], and
/// on the very first call binds every `kv_cache.*` name to a zero-length
/// buffer (`cached_len == 0`) -- the cached-block reduces both fold over an
/// empty range, which [`ReduceInit::Zero`]/[`ReduceInit::NegativeInfinity`]
/// already define as identity/`-inf`, so the first call degenerates to
/// plain causal self-attention over the whole prompt with no special case.
///
/// Dense-only: always binds `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple. Delegates to
/// [`mistral_cached_forward_program_with_experts`] with `expert_count = 0`,
/// `expert_used_count = 0` -- that function's own doc explains why those two
/// values select the identical dense program this function has always
/// built. Kept as its own entry point (rather than folding the two extra
/// parameters in here) because this signature already has real callers
/// outside this crate that a dense-only checkpoint never needs to pass an
/// expert config to.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        false,
        false,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s Qwen3 dense-attention counterpart:
/// the identical interleaved-RoPE cached layer, plus per-head QK-norm
/// (Qwen3's own `q_norm`/`k_norm`, `modeling_qwen3.py`'s `Qwen3Attention`)
/// applied to `q`/`k_new` before RoPE -- see
/// `append_mistral_cached_layer`'s `qk_norm` parameter doc for the exact
/// two ops this adds over the plain Mistral layer. Qwen3 has no
/// mixture-of-experts variant this crate has bound yet, so this takes no
/// `expert_count`/`expert_used_count`, the same dense-only shape
/// [`mistral_cached_forward_program`] itself uses.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        true,
        false,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s mixture-of-experts-capable
/// counterpart, carrying the same `expert_count`/`expert_used_count`
/// parameters [`mistral_forward_program`] already takes. `expert_count == 0`
/// binds every layer through `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple, node-for-node the same program
/// [`mistral_cached_forward_program`] has always built, so a dense
/// checkpoint's generated program is unaffected by this function's
/// existence. `expert_count > 0` routes each layer through
/// `append_mistral_cached_moe_layer` instead, gathering one of
/// `expert_count` experts' weight slabs per token per `append_moe_ffn`'s
/// doc -- the same routed FFN [`mistral_forward_program`]'s own MoE branch
/// already runs, reused rather than reconstructed.
///
/// `paired_gate_up_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.ffn_gate_up.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_paired`). No effect on
/// the `expert_count > 0` branch (MoE's own gate/up weights are a separate
/// per-expert stack this flag does not touch).
///
/// `fused_qkv_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.attn_qkv.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_triple`). Requires
/// `qk_norm == false` (`append_mistral_cached_layer`'s own doc); no effect
/// on the `expert_count > 0` branch (attention projections are untouched by
/// which FFN branch runs).
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts(
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
) -> Result<(Vec<Op>, ForwardRoots, Vec<CachedLayerRoots>, MoeSites), TensorError> {
    let (program, roots, cache_roots, _layer_residuals, moe_sites) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            vocab,
            embedding,
            feed_forward,
            query_heads,
            kv_heads,
            head_dim,
            block_count,
            expert_count,
            expert_used_count,
            qk_norm,
            qkv_biases,
            paired_gate_up_reduce,
            fused_qkv_reduce,
            false,
        )?;
    Ok((program, roots, cache_roots, moe_sites))
}

/// [`mistral_cached_forward_program_with_experts`]'s full implementation,
/// additionally returning one [`NodeId`] per layer -- the residual
/// (`x_next`, the post-MoE-add activation) each block hands the next layer,
/// in layer order, `block_count` entries. A caller bisecting a CPU-vs-Metal
/// divergence requests these as extra program outputs (48 x [seq, embedding]
/// floats for a 48-layer checkpoint, trivially small) to find the first
/// layer whose output disagrees, without materializing every intermediate
/// node in the graph as an output (the CPU evaluator keeps every requested
/// output's full lifetime alive, so requesting ALL nodes is the >130 GB
/// failure mode this narrower request set avoids).
///
/// `last_row_only` gates the vocab-projection matmul's own row count:
/// `true` slices the final-norm activation to its last row before
/// `output.weight` ever multiplies it (a host-supplied `lm_head_row`
/// `Op::Input`, gathered through the same [`IndexMap::Computed`] shape
/// [`embedding_lookup`] already proves correct -- see that leaf's own doc
/// at the call site below for why it is host-supplied rather than
/// in-graph-derived), so the matmul computes one row instead of
/// `new_count`. `false` (every existing caller today) reproduces the prior
/// per-row-logits program unchanged -- a caller genuinely needing every
/// new row's own logits (multi-token verification, prefill scoring,
/// logprobs) opts into that by passing `false`, not by this crate guessing
/// which one a caller wants.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts_and_layer_taps(
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
    last_row_only: bool,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    let rope_pairing = if qk_norm {
        RopePairing::SplitHalf {
            pairs: head_dim / 2,
        }
    } else {
        RopePairing::Interleaved
    };
    mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        qk_norm,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        last_row_only,
        rope_pairing,
    )
}

/// Qwen2's dense/MoE graph variant. Qwen2 uses split-half (NEOX) RoPE even
/// though it has no QK-norm weights, so its pairing must be selected from the
/// architecture name rather than inferred from the presence of norm tensors.
#[allow(clippy::too_many_arguments)]
pub fn qwen2_cached_forward_program_with_experts_and_layer_taps(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    last_row_only: bool,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        false,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        last_row_only,
        RopePairing::SplitHalf {
            pairs: head_dim / 2,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
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
    last_row_only: bool,
    rope_pairing: RopePairing,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    // only materialized when a layer actually consumes it (`qk_norm`), so a
    // dense checkpoint with no QK-norm keeps the identical node count this
    // function has always emitted.
    let inv_head_dim = qk_norm.then(|| scalar_constant(&mut program, 1.0 / head_dim as f32));
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    // Only ever consulted by `append_mistral_cached_layer`'s
    // `fused_qkv_reduce` branch (that parameter's own doc) -- built ONLY
    // when the flag is set, so `false` reproduces today's program
    // node-for-node (`cached_attention_rewrite_replaces_the_bound_attention_subgraph`'s
    // own literal bound-op-count fixture is the guard: it caught the
    // unconditional-`Op::Constant` version of this as a real +2 node
    // regression before this comment existed).
    let (head_shape_ones, kv_head_shape_ones) = if fused_qkv_reduce {
        let head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        let kv_head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        (head_shape_ones, kv_head_shape_ones)
    } else {
        // Never read (`append_mistral_cached_layer`'s `fused_qkv_reduce`
        // branch is the only reader, and it never runs when the flag is
        // `false`) -- `ones` (already built above) is reused as the
        // placeholder rather than adding an `Option` the callee would need
        // to `expect()` out of (this crate's own no-`expect`-in-production
        // rule), or building a real constant no `false` caller ever needs.
        (ones, ones)
    };
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Rank-0 `Op::Input`, same precedent `eps`/`rope_cos`/`rope_sin` set
    // (`causal_mask_merged`'s own doc), named "cached_len" so
    // `proxima_tensor::bind::cached_attention_candidates` can find it by
    // name -- this crate's own precedent for what a name is for
    // (`Op::Input`'s own doc: "identity, not decoration"). It feeds no
    // arithmetic in this program: the host supplies the REAL `cached_len`
    // every call, independent of `kv_cache.{layer}.*`'s own
    // `Extent::Symbolic(1)` extent (which a caller may round up to a
    // bucket boundary, `ServingConfig::kv_bucket_tokens`, without
    // rebuilding this program), and the fused `BoundOpKind::CachedAttention`
    // reads it as a NINTH, runtime operand at execution time instead --
    // the bucket's own zero-padding is excluded by that bound, never by a
    // mask node in this graph (`BoundOpKind::CachedAttention`'s own doc on
    // the `cached_key_rows != 0` discriminator).
    let _cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);
    let mut layer_residuals: Vec<NodeId> = Vec::with_capacity(block_count as usize);
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let (wq, wk, wv) = if fused_qkv_reduce {
            let rows = (query_heads + 2 * kv_heads) * head_dim;
            let w_qkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(rows), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.attn_qkv.weight"),
            );
            (w_qkv, w_qkv, w_qkv)
        } else {
            let wq = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(query_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_q.weight"),
            );
            let wk = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_k.weight"),
            );
            let wv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_v.weight"),
            );
            (wq, wk, wv)
        };
        let q_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_q.bias"),
            )
        });
        let k_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_k.bias"),
            )
        });
        let v_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_v.bias"),
            )
        });
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        let k_even_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_even"),
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_odd"),
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("kv_cache.{layer}.v"),
        );

        let (x_next, layer_roots) = if expert_count == 0 {
            let (w_gate, w_up) = if paired_gate_up_reduce {
                let w_gate_up = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(2),
                        Extent::Static(feed_forward),
                        Extent::Static(embedding)
                    ],
                    &alloc::format!("blk.{layer}.ffn_gate_up.weight"),
                );
                (w_gate_up, w_gate_up)
            } else {
                let w_gate = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                    &alloc::format!("blk.{layer}.ffn_gate.weight"),
                );
                let w_up = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                    &alloc::format!("blk.{layer}.ffn_up.weight"),
                );
                (w_gate, w_up)
            };
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                (q_norm_weight, k_norm_weight, inv_head_dim)
            });

            append_mistral_cached_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                head_shape_ones,
                kv_head_shape_ones,
                is_future,
                group,
                head_dim,
                query_heads,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                k_even_cache,
                k_odd_cache,
                v_cache,
                qk_norm_weights,
                q_bias,
                k_bias,
                v_bias,
                paired_gate_up_reduce,
                fused_qkv_reduce,
                rope_pairing,
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );
            let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                (q_norm_weight, k_norm_weight, inv_head_dim)
            });

            let (next_x, next_roots, site) = append_mistral_cached_moe_layer(
                &mut program,
                layer,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future,
                group,
                head_dim,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                k_even_cache,
                k_odd_cache,
                v_cache,
                qk_norm_weights,
            )?;
            moe_sites.push(site);
            (next_x, next_roots)
        };
        x = x_next;
        cache_roots.push(layer_roots);
        layer_residuals.push(x_next);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let normed_last = gather_last_row(&mut program, normed_final, last_row_only);

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok((
        program,
        ForwardRoots {
            logits,
            hidden: normed_last,
        },
        cache_roots,
        layer_residuals,
        MoeSites(moe_sites),
    ))
}

/// Per-layer roots [`qwen35_forward_program`]'s own caller threads back in
/// as next-call cache [`Op::Input`]s -- [`Qwen35DenseAttentionRoots`]'s own
/// 4-wide KV-cache shape for a dense-attention layer
/// (`append_qwen35_dense_attention_layer`'s own doc walks through why it
/// is 4-wide, not [`CachedLayerRoots`]'s 3), or `append_qwen35_ssm_mixer`'s
/// own `(qkv_mixed, state_out)` return for an SSM layer. A discriminated
/// enum, not a bool flag riding alongside a fixed-shape tuple: the layer
/// kinds carry genuinely different cache shapes, the same reason
/// [`LayerKind`] exists as its own type rather than a boolean.
///
/// `Attention(CachedLayerRoots)` is [`mistral_cached_forward_program_with_experts`]'s
/// own 3-wide shape, still constructed by that program's caller
/// (`crate::generate::LoadedModel::load`) for every non-qwen35 checkpoint --
/// kept as its own variant rather than folded into `DenseAttention` so that
/// caller's cache-threading loop, and its `LayerCache`, are unaffected by
/// this checkpoint's own partial-rotary gap.
#[derive(Debug, Clone, Copy)]
pub enum Qwen35LayerRoots {
    Attention(CachedLayerRoots),
    DenseAttention(Qwen35DenseAttentionRoots),
    Ssm {
        qkv_mixed: NodeId,
        state_out: NodeId,
    },
}

/// Qwen3.5's whole-model incremental forward program: `full_attention_interval`
/// dense-attention layers (`append_mistral_cached_layer`, the same KV-cache
/// pattern [`mistral_cached_forward_program_with_experts`] already runs)
/// interleaved with gated-DeltaNet layers (`append_qwen35_ssm_mixer`),
/// following llama.cpp's own `hparams.is_recr_impl[i] = (i < n_layer) &&
/// ((i + 1) % full_attention_interval != 0)` (`qwen35.cpp:19-20`) -- layer
/// `full_attention_interval - 1`, `2 * full_attention_interval - 1`, ... are
/// dense attention, every other layer is SSM. Qwen3.5 never routes FFN
/// through experts (`qwen35.cpp:471`, `GGML_ASSERT(model.layers[il].ffn_gate_inp
/// == nullptr)`), so every layer's FFN is the plain dense triple
/// [`mistral_cached_forward_program_with_experts`]'s own `expert_count == 0`
/// branch already builds -- reused here rather than reconstructed.
///
/// `ssm_d_state`/`ssm_dt_rank`/`ssm_n_group`/`ssm_d_inner`/`ssm_d_conv` name
/// the same five hyperparameters `qwen35.cpp:335-343`'s own
/// `build_layer_attn_linear` reads off `hparams`, unpacked into
/// `append_qwen35_ssm_mixer`'s own `key_dim = ssm_d_state * ssm_n_group`,
/// `value_dim = ssm_d_inner`, `kv_heads = ssm_n_group`, `group = ssm_dt_rank
/// / ssm_n_group`, `l_cache = ssm_d_conv` (`head_v_dim = ssm_d_inner /
/// ssm_dt_rank` falls out inside the mixer itself, matching the oracle's own
/// `head_v_dim = d_inner / num_v_heads`). `rms_eps` is
/// `hparams.f_norm_rms_eps` baked as a graph-build-time constant, the same
/// choice this module already makes for `inv_dim`/`inv_sqrt_head_dim`
/// (Rust-side config values, not runtime-bound `Input`s) rather than a fresh
/// runtime-bound tensor shaped to `append_qwen35_ssm_mixer`'s own
/// `head_eps` (`[kv_heads, group]`) -- there is exactly one epsilon value
/// per checkpoint, known at program-build time.
///
/// Dense attention's own layers (`append_qwen35_dense_attention_layer`,
/// not `append_mistral_cached_layer`) run split-half RoPE over the
/// checkpoint's PARTIAL rotary width plus a concatenated-by-sum pass-through
/// remainder, and a per-head sigmoid gate on the attention output --
/// `append_qwen35_dense_attention_layer`'s own doc walks through why the
/// declared 3-section MRoPE (`rope.dimension_sections`) collapses to plain
/// single-section RoPE for this checkpoint's text-only forward program.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_head_dim: u32,
    block_count: u32,
    full_attention_interval: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    ssm_d_inner: u32,
    ssm_d_conv: u32,
    rms_eps: f32,
) -> Result<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), TensorError> {
    qwen35_forward_program_with_last_row(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        attn_head_dim,
        block_count,
        full_attention_interval,
        ssm_d_state,
        ssm_dt_rank,
        ssm_n_group,
        ssm_d_inner,
        ssm_d_conv,
        rms_eps,
        false,
    )
}

/// Builds the Qwen35 program while optionally reducing the final vocabulary
/// projection to a host-selected row before the packed weight is read.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_forward_program_with_last_row(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_head_dim: u32,
    block_count: u32,
    full_attention_interval: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    ssm_d_inner: u32,
    ssm_d_conv: u32,
    rms_eps: f32,
    last_row_only: bool,
) -> Result<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), TensorError> {
    if full_attention_interval == 0 {
        return Err(TensorError::InvalidFullAttentionInterval {
            full_attention_interval,
        });
    }

    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;
    let ssm_group = ssm_dt_rank / ssm_n_group;
    let ssm_key_dim = ssm_d_state * ssm_n_group;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let one = ones;
    // `head_dim` here is `rope.dimension_count` -- this checkpoint's
    // PARTIAL-rotary width (`rotary_dim`), never the real per-head width.
    // Dense attention's own score scale is `attn_head_dim`-based
    // (`self.scaling = self.head_dim**-0.5` where `self.head_dim` is the
    // real width, `modeling_qwen3_next.py:262,264`), not
    // `rotary_dim`-based.
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (attn_head_dim as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (ssm_d_state as f32).sqrt());
    let head_v_dim = ssm_d_inner / ssm_dt_rank;
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let head_eps = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(ssm_n_group), Extent::Static(ssm_group)],
            value: rms_eps,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Same rank-0 leaf [`mistral_cached_forward_program_with_experts`] adds
    // right after its own `causal_mask` call, and for the same reason: named
    // "cached_len" so `bind::cached_attention_candidates`'s `find_named_input`
    // picks it up by NAME on the `Attention` arm's fused `CachedAttention`
    // op. The `DenseAttention` arm has no equivalent fusion, so this same
    // node is ALSO threaded directly into every
    // [`append_qwen35_dense_attention_layer`] call below to mask its own
    // padded cached range (that function's own doc).
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut layer_roots: Vec<Qwen35LayerRoots> = Vec::with_capacity(block_count as usize);

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        // Named `post_attention_norm.weight` on disk, not `ffn_norm.weight`
        // -- this checkpoint's own GGUF writer names this tensor
        // differently from every other architecture this crate binds
        // (`proxima_model_interop::qwen35`'s own module doc, confirmed via
        // `strings` on the real file: no `blk.N.ffn_norm.weight` key
        // exists anywhere), on both layer kinds.
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.post_attention_norm.weight"),
        );

        // `hparams.is_recr_impl[i] = (i + 1) % full_attention_interval != 0`
        // (`qwen35.cpp:19-20`) is TRUE for SSM layers -- dense attention is
        // its negation, `(i + 1) % full_attention_interval == 0`.
        let is_dense_attention = (layer + 1) % full_attention_interval == 0;

        let (x_next, roots) = if is_dense_attention {
            // real per-head width read off metadata (`attention.key_length`,
            // `attn_head_dim` param) rather than `embedding / query_heads`
            // -- the latter is not even an integer on the 27B checkpoint
            // (`5120 / 24 = 213.33`), confirmed wrong against the real file
            // by [`crate::qwen35::qwen35_architecture_from_metadata`]'s own
            // caller-side doc.
            let wq_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(query_heads * attn_head_dim * 2)
                ],
                &alloc::format!("blk.{layer}.attn_q.weight"),
            );
            // A pure reshape (multiply by a broadcast-ones constant), the
            // same lossless-reshape donor trick `wk`/`wv` already use below
            // -- NEVER `per_head_channel_slice` on this packed leaf. That
            // per-head WEIGHT-level slice inserted a select-then-reduce
            // between `wq_flat` and the real contraction, which
            // `is_quantized_matmul_operand`/`run_reduce_quantized` (`cpu.rs`)
            // then misidentifies as the whole quantized matmul shape and
            // derives `rows`/`k` from the wrong axis pair -- `q`/`gate` now
            // split on the ACTIVATION side instead, inside
            // [`append_qwen35_dense_attention_only_with_taps`], via
            // [`per_head_channel_range`].
            let qg_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![
                        Extent::Static(query_heads),
                        Extent::Static(attn_head_dim * 2)
                    ],
                    value: 1.0,
                },
            );
            let wq_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wq_flat,
                        alloc::format!("i,{}*h+c->ihc", attn_head_dim * 2).as_str(),
                    ),
                    (qg_head_ones, "hc->ihc"),
                ],
            )?;
            let wk_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads * attn_head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_k.weight"),
            );
            // `k` carries no gate and no partial-rotary truncation at the
            // weight level (the split into rotated/pass halves happens on
            // the ACTIVATION inside [`append_qwen35_dense_attention_layer`]
            // now that `q_norm`/`k_norm` need the full width first) -- the
            // same lossless-reshape donor trick `v`/`o` already use below.
            let k_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let wk = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wk_flat,
                        alloc::format!("i,{attn_head_dim}*u+d->iud").as_str(),
                    ),
                    (k_head_ones, "ud->iud"),
                ],
            )?;
            let v_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let o_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![
                        Extent::Static(kv_heads),
                        Extent::Static(group),
                        Extent::Static(attn_head_dim)
                    ],
                    value: 1.0,
                },
            );
            let wv_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads * attn_head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_v.weight"),
            );
            let wv = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wv_flat,
                        alloc::format!("i,{attn_head_dim}*u+d->iud").as_str(),
                    ),
                    (v_head_ones, "ud->iud"),
                ],
            )?;
            let wo_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(query_heads * attn_head_dim),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.attn_output.weight"),
            );
            let wo = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wo_flat,
                        alloc::format!("{}*u+{attn_head_dim}*g+d,e->ugde", attn_head_dim * group)
                            .as_str(),
                    ),
                    (o_head_ones, "ugd->ugde"),
                ],
            )?;
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let q_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_q_norm.weight"),
            );
            let k_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_k_norm.weight"),
            );
            let pass_dim = attn_head_dim - head_dim;
            let k_first_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_first"),
            );
            let k_second_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_second"),
            );
            let k_pass_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pass_dim)
                ],
                &alloc::format!("kv_cache.{layer}.k_pass"),
            );
            let v_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(attn_head_dim)
                ],
                &alloc::format!("kv_cache.{layer}.v"),
            );

            let (x_next, dense_attention_roots) = append_qwen35_dense_attention_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_attn_head_dim,
                inv_attn_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future,
                cached_len,
                group,
                head_dim,
                attn_head_dim,
                attn_norm_weight,
                ffn_norm_weight,
                q_norm_weight,
                k_norm_weight,
                wq_gate,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                k_first_cache,
                k_second_cache,
                k_pass_cache,
                v_cache,
            )?;
            (
                x_next,
                Qwen35LayerRoots::DenseAttention(dense_attention_roots),
            )
        } else {
            let qkv_dim = 2 * ssm_key_dim + ssm_d_inner;
            let wqkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
                &alloc::format!("blk.{layer}.ssm_in.weight"),
            );
            let wqkv_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_d_inner)],
                &alloc::format!("blk.{layer}.ssm_gate.weight"),
            );
            let conv_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(qkv_dim), Extent::Static(ssm_d_conv)],
                &alloc::format!("blk.{layer}.ssm_conv1d.weight"),
            );
            let conv_history_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_conv - 1), Extent::Static(qkv_dim)],
                &alloc::format!("ssm_cache.{layer}.conv_history"),
            );
            let ssm_beta = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_beta.weight"),
            );
            let ssm_alpha = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_alpha.weight"),
            );
            let ssm_dt_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_dt.bias"),
            );
            let ssm_a = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_a"),
            );
            let ssm_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_v_dim)],
                &alloc::format!("blk.{layer}.ssm_norm.weight"),
            );
            let ssm_out = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_inner), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ssm_out.weight"),
            );
            let state_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(ssm_d_state),
                    Extent::Static(head_v_dim),
                    Extent::Static(ssm_n_group),
                    Extent::Static(ssm_group)
                ],
                &alloc::format!("ssm_cache.{layer}.state"),
            );

            let (mixer_out, qkv_mixed, state_out) = append_qwen35_ssm_mixer(
                &mut program,
                x,
                inv_dim,
                eps,
                head_eps,
                one,
                inv_sqrt_key_dim,
                inv_head_v_dim,
                Some(attn_norm_weight),
                wqkv,
                wqkv_gate,
                conv_weight,
                conv_history_in,
                ssm_beta,
                ssm_alpha,
                ssm_dt_bias,
                ssm_a,
                ssm_norm_weight,
                ssm_out,
                state_in,
                ssm_key_dim,
                ssm_d_inner,
                ssm_n_group,
                ssm_group,
                ssm_d_conv,
                GdnOutputGate::Silu,
            )?;

            // Unlike `append_mistral_cached_layer` (bundles FFN internally),
            // `append_qwen35_ssm_mixer` is mixer-plus-residual only -- the
            // same scope `append_lfm2_conv_mixer` has -- so the SSM branch
            // runs its own dense FFN pass here, matching
            // `mistral_cached_forward_program_with_experts`'s own
            // `expert_count == 0` FFN math exactly (Qwen3.5 never routes FFN
            // through experts, `qwen35.cpp:471`).
            let normed2 = rmsnorm(&mut program, mixer_out, ffn_norm_weight, inv_dim, eps)?;
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let gate_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
            )?;
            let gate = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let up_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
            )?;
            let up = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let silu_gate = silu(&mut program, gate, one, "sg->sg")?;
            let ffn_hidden = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(silu_gate, "sg->sg"), (up, "sg->sg")],
            )?;
            let down_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
            )?;
            let ffn_out = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                down_product,
                "sgd->sgd",
                "sd->sgd",
            )?;
            let x_after_ffn = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(ffn_out, "sd->sd"), (mixer_out, "sd->sd")],
            )?;

            (
                x_after_ffn,
                Qwen35LayerRoots::Ssm {
                    qkv_mixed,
                    state_out,
                },
            )
        };

        x = x_next;
        layer_roots.push(roots);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let normed_last = gather_last_row(&mut program, normed_final, last_row_only);
    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok((program, logits, layer_roots))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod activation_tests {
    use super::*;

    const VALUES: [f32; 4] = [-2.0, -0.5, 0.7, 3.0];

    fn evaluate_activation(activation: Activation) -> Vec<f32> {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(VALUES.len() as u32)],
            "x",
        );
        let ones = scalar_constant(&mut program, 1.0);
        let root =
            append_activation(&mut program, x, ones, activation).expect("append_activation lowers");

        let symbols: [u64; 0] = [];
        let blocks: [&[f32]; 1] = [&VALUES];
        let evaluated = crate::cpu::evaluate(&program, &symbols, &blocks, &[root])
            .expect("append_activation evaluates");
        evaluated.root().to_vec()
    }

    /// [`append_activation`]'s [`Activation::Silu`] arm must match the prior
    /// inline `sigmoid(x) * x` chain [`append_dense_swiglu_ffn`] and
    /// [`append_moe_round_output`] hardcoded before this helper existed --
    /// the whole point of extracting it is that this arm is byte-identical
    /// to that chain, never a rewrite.
    #[test]
    fn silu_matches_sigmoid_times_x() {
        let activated = evaluate_activation(Activation::Silu);
        for (value, activated_value) in VALUES.iter().zip(activated.iter()) {
            let expected = value / (1.0 + (-value).exp());
            assert!(
                (activated_value - expected).abs() < 1e-5,
                "silu({value}) = {activated_value}, expected {expected}"
            );
        }
    }

    /// [`append_activation`]'s [`Activation::GeluTanh`] arm against the
    /// closed-form `gelu_pytorch_tanh` formula (Gemma's GeGLU nonlinearity),
    /// computed independently here rather than by mirroring the composed
    /// op sequence, so a wiring or op-order bug in the composition is
    /// actually caught.
    #[test]
    fn gelu_tanh_matches_closed_form_reference() {
        let activated = evaluate_activation(Activation::GeluTanh);
        for (value, activated_value) in VALUES.iter().zip(activated.iter()) {
            let cubic = value + 0.044_715 * value.powi(3);
            let expected = 0.5 * value * (1.0 + (0.797_884_6 * cubic).tanh());
            assert!(
                (activated_value - expected).abs() < 1e-5,
                "gelu_tanh({value}) = {activated_value}, expected {expected}"
            );
        }
    }
}
