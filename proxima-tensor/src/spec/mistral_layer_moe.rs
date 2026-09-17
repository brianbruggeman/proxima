use super::*;

/// One transformer layer, node-for-node the same graph
/// `specs/mistral_layer.toml` spells — attention (RoPE + GQA + causal mask)
/// then the SwiGLU feed-forward, each wrapped in its own residual. `x` in,
/// the layer's own residual-summed output out; every other argument is
/// either a per-layer weight (`wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/`w_down`)
/// or one of the position-only constants [`causal_mask`]/`cos`/`sin` share
/// across every layer.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let (rotated_q_even, rotated_q_odd) =
        fused_rope_pair(program, q, 'h', cos, sin, RopePairing::Interleaved)?;

    let (rotated_k_even, rotated_k_odd) =
        fused_rope_pair(program, k, 'u', cos, sin, RopePairing::Interleaved)?;

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

    // attention's usual `1/sqrt(d_k)`: without it QK^T over a real head_dim
    // (128) saturates softmax toward one-hot instead of blending.
    // `inv_sqrt_head_dim` is a rank-0 `Op::Constant`, so it broadcasts via
    // an empty operand side, the same way `neg_infinity` does.
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

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
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
        &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
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

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, "sg->sg")],
    )?;
    let exp_neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_gate, "sg->sg")],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )
}

/// Rust-code counterpart of `specs/moe_block.toml`'s `ffn_product` node:
/// gathers `stack[route[s], :, :]` (`stack` is a `[expert_count, d_in,
/// d_out]` weight slab) and multiplies it elementwise against `x`'s `[s,
/// d_in]`, broadcast over the `d_out` axis, ready for a later [`reduce`]
/// over `d_in` to finish the matmul.
#[must_use]
pub fn gathered_expert_product(
    program: &mut Vec<Op>,
    stack: NodeId,
    route: NodeId,
    x: NodeId,
) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let x_map = IndexMap::Affine(map::projection(3, &[0, 1]));
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(stack, gathered_map), (x, x_map)],
            name: None,
        },
    )
}

/// Builds the grouped counterpart of [`gathered_expert_product`]. The route
/// tensor is `[sequence, selected]`, the activation is `[sequence, d_in]`,
/// and the result is `[sequence, selected, d_in, d_out]`; a caller reduces
/// the contraction axis and combines the selected outputs with its routing
/// weights. Keeping the selected axis explicit lets one computed gather serve
/// every top-k round without introducing a new operation kind.
#[must_use]
pub fn grouped_gathered_expert_product(
    program: &mut Vec<Op>,
    stack: NodeId,
    route: NodeId,
    x: NodeId,
) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(4, &[0, 1]),
        base: IndexPattern {
            iter_rank: 4,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(3)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let x_map = IndexMap::Affine(map::projection(4, &[0, 2]));
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(stack, gathered_map), (x, x_map)],
            name: None,
        },
    )
}

/// Packs independently selected expert ids (`[sequence]` each) into the
/// `[sequence, selected]` index tensor consumed by
/// [`grouped_gathered_expert_product`].
///
/// This is graph-construction work, not a runtime host allocation: the
/// selected axis is an [`Op::Iota`] and each column is selected with the
/// existing elementwise algebra. Expert ids remain exact in `f32` for every
/// representable model-sized expert table and are converted only by the
/// computed-gather boundary.
pub fn stack_selected_routes(
    program: &mut Vec<Op>,
    routes: &[NodeId],
) -> Result<NodeId, TensorError> {
    let selected_count =
        u32::try_from(routes.len()).map_err(|_| TensorError::InvalidExpertConfig {
            expert_count: u32::MAX,
            expert_used_count: u32::MAX,
        })?;
    if selected_count == 0 {
        return Err(TensorError::InvalidExpertConfig {
            expert_count: 0,
            expert_used_count: 0,
        });
    }

    let selected_axis = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(selected_count),
        },
    );
    let mut stacked = None;
    for (round, route) in routes.iter().copied().enumerate() {
        let round_value = op::append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: round as f32,
            },
        );
        let round_mask = elementwise(
            program,
            DType::Float32,
            ScalarOp::Equal,
            &[(selected_axis, "k->k"), (round_value, "->k")],
        )?;
        let selected_route = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(route, "s->sk"), (round_mask, "k->sk")],
        )?;
        stacked = Some(match stacked {
            Some(previous) => elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(previous, "sk->sk"), (selected_route, "sk->sk")],
            )?,
            None => selected_route,
        });
    }

    stacked.ok_or(TensorError::InvalidExpertConfig {
        expert_count: 0,
        expert_used_count: 0,
    })
}

/// Selects one `[sequence, feature]` plane from a grouped
/// `[sequence, selected, feature]` projection without materializing a copy.
#[must_use]
pub fn select_grouped_round(
    program: &mut Vec<Op>,
    grouped: NodeId,
    round: u32,
    dtype: DType,
) -> NodeId {
    let map = IndexMap::Affine(IndexPattern {
        iter_rank: 2,
        axes: alloc::vec![
            AxisIndex {
                terms: core::iter::once(AxisTerm::projection(0)).collect(),
                offset: 0,
                len: None,
            },
            AxisIndex {
                terms: Default::default(),
                offset: round as i32,
                len: None,
            },
            AxisIndex {
                terms: core::iter::once(AxisTerm::projection(1)).collect(),
                offset: 0,
                len: None,
            },
        ],
    });
    op::append(
        program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Identity,
            operands: alloc::vec![(grouped, map)],
            name: None,
        },
    )
}

/// `scale[route[s]]`: gathers one scalar per selected expert, the
/// rank-one counterpart of [`gathered_expert_product`]'s weight-matrix
/// gather. Used to fold a per-expert output scale (gemma4's
/// `blk.{layer}.ffn_down_exps.scale`, `[expert_count]`) into that expert's
/// routing weight before combination.
#[must_use]
fn gather_expert_scale(program: &mut Vec<Op>, scale: NodeId, route: NodeId) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(1, &[0]),
        base: IndexPattern {
            iter_rank: 1,
            axes: alloc::vec![AxisIndex::default()],
        },
        gathered_dim: 0,
    };
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(scale, gathered_map)],
            name: None,
        },
    )
}

/// Which function turns a MoE gate's raw logits into per-expert routing
/// scores -- llama.cpp's own `llama_expert_gating_func_type`
/// (`llama-hparams.h:11-14`), read from a checkpoint's own
/// `{architecture}.expert_gating_func` metadata key when present.
/// `Softmax` is llama.cpp's own fallback when that key is absent
/// (`llama-model.cpp:1237-1240`, "existing models that have no
/// `expert_gating_func` model parameter set") -- Mixtral carries no such
/// key, so `append_mistral_moe_layer`/`append_mistral_cached_moe_layer`
/// always pass `Softmax` unconditionally rather than reading a key that
/// does not exist on that checkpoint. `Sigmoid` is `_TYPE_SIGMOID` (`2`),
/// LFM2's own value (`transformers/models/lfm2_moe/modeling_lfm2_moe.py:209`'s
/// `router_logits.sigmoid()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGatingFunc {
    Softmax,
    Sigmoid,
}

/// One [`append_moe_ffn`] call's routing decision, returned alongside its
/// output node so the decode loop -- not this kernel-building function --
/// decides whether to evaluate and observe it. `selected` holds one
/// [`NodeId`] per `expert_used_count` round (each round's own `route`
/// reduce, `Int32`, one value per token position); `weights` holds each
/// round's own `weight` node in the same order, followed by the final
/// `weight_total` node used to renormalize them -- so `weights.len() ==
/// selected.len() + 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeSite {
    pub layer: u32,
    pub selected: Vec<NodeId>,
    pub weights: Vec<NodeId>,
}

/// Every [`MoeSite`] a forward-program builder's [`append_moe_ffn`] calls
/// produced, in layer order -- empty for a dense (non-MoE) program. The
/// decode loop (`crate::instrument::ExpertObserver`'s consumer, gated behind
/// this crate's `instrument` feature) reads this to know which extra nodes
/// to request as evaluation outputs, rather than the kernel emitting a
/// routing event per gathered position the way
/// `crate::instrument::notify_expert_routed` used to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MoeSites(pub Vec<MoeSite>);

/// The routed feed-forward `specs/moe_block.toml`/`specs/moe_topk2_probe.toml`
/// describe: a gate projects `x` to one logit per expert, `expert_used_count`
/// rounds of top-1 argmax-with-exclusion each route one token to one more
/// expert (`moe_topk2_probe.toml`'s own header proves a *fixed* k stays
/// inside the affine-only + `Iota`/`Computed`-gather algebra, no new
/// `Op`/`ScalarOp` variant, unrolled at spec-build time the same way this
/// whole function is), and each round's gathered expert runs the same
/// SwiGLU [`append_mistral_layer`]'s dense path uses, weighted by its own
/// share among only the selected experts.
///
/// `gating` picks how raw `logits` become the per-expert `scores` used both
/// to select AND (absent a bias) to weight experts:
/// [`ExpertGatingFunc::Softmax`] leaves `scores` aliased to `logits` --
/// `weight_r = exp(max_logit_r - max_logit_0)` then a final
/// divide-by-`weight_total` is *exactly* softmax restricted to the selected
/// top-k and renormalized (the standard Mixtral combination formula), so
/// this never diverges by so much as one node from the code this function
/// has always built. [`ExpertGatingFunc::Sigmoid`] materializes
/// `sigmoid(logits)` up front via the same `Negate`+`Exponential`+`Add(1)`+
/// `Reciprocal` construction the dense SwiGLU path already builds
/// (`spec.rs`'s own `silu_gate` node a few lines below this one) --
/// `ScalarOp` gained no `Sigmoid` variant for this, since the four ops
/// already existed for a different consumer.
///
/// `expert_bias` (`blk.{layer}.exp_probs_b.bias` on a real LFM2 checkpoint,
/// `[expert_count]`) is llama.cpp's own `ffn_exp_probs_b` /
/// `route_tokens_to_experts`'s own `self.expert_bias`
/// (`modeling_lfm2_moe.py:210-213`, `llama-graph.cpp`'s own
/// `build_moe_ffn`'s `selection_probs = ggml_add(probs, exp_probs_b)`
/// comment: "leave probs unbiased as it's later used to get expert
/// weights"): added to `scores` ONLY for the argmax that picks
/// `expert_used_count` experts, never for the weight a selected expert's
/// output is scaled by -- getting that backwards would still select
/// *plausible* experts (the bias is small relative to genuine routing
/// signal) while silently reweighting every token's combination, exactly
/// the "plausible output, wrong routing" failure mode metadata-absent
/// checkpoints (Mixtral, `expert_bias: None`) cannot exhibit since they
/// never reach this branch.
///
/// The routed feed-forward block [`lfm2_forward_program_with_experts`]
/// and [`mistral_cached_forward_program_with_experts`] both call per
/// layer; see [`qwen35_forward_program`] for this crate's own worked
/// example of a full per-layer builder chain (a dense, non-MoE FFN there).
#[allow(clippy::too_many_arguments)]
pub fn append_moe_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    activation: Activation,
) -> Result<(NodeId, MoeSite), TensorError> {
    append_moe_ffn_with_expert_scale(
        program,
        layer,
        x,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        gating,
        expert_bias,
        None,
        activation,
    )
}

/// [`append_moe_ffn`] with an extra OPTIONAL per-expert output scale
/// (gemma4's `blk.{layer}.ffn_down_exps.scale`, `[expert_count]`), gathered
/// by each round's own selected route and folded into that round's
/// combination weight AFTER softmax-over-selected renormalization, matching
/// the authoritative `topk_weights = topk_weights * expert_scales` fold.
/// `None` reproduces [`append_moe_ffn`] byte-for-byte -- every caller that
/// has no such scale on disk (Mixtral, LFM2, qwen3.6 MoE) passes `None` and
/// this function never diverges from the pre-scale program.
#[allow(clippy::too_many_arguments)]
pub fn append_moe_ffn_with_expert_scale(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    expert_scale: Option<NodeId>,
    activation: Activation,
) -> Result<(NodeId, MoeSite), TensorError> {
    append_moe_ffn_with_projection_strategy(
        program,
        layer,
        x,
        gate_inp,
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
        MoeProjectionStrategy::PerRoute,
    )
}

/// Builds a routed feed-forward block from an already-computed router-logit
/// node. Callers that expose the router as a graph boundary use this entry
/// point so the routing observation and the gathered expert products consume
/// the same node rather than rebuilding an independent projection.
#[allow(clippy::too_many_arguments)]
pub fn append_moe_ffn_from_logits(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    logits: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    expert_scale: Option<NodeId>,
    activation: Activation,
) -> Result<(NodeId, MoeSite), TensorError> {
    append_moe_ffn_with_projection_strategy_from_logits(
        program,
        layer,
        x,
        logits,
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
        MoeProjectionStrategy::PerRoute,
    )
}

/// [`append_moe_ffn`] with gate and up projections grouped over the selected
/// axis while each composed hidden activation still enters its own gathered
/// down projection.
#[allow(clippy::too_many_arguments)]
pub fn append_moe_ffn_grouped_gate_up(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    activation: Activation,
) -> Result<(NodeId, MoeSite), TensorError> {
    append_moe_ffn_with_projection_strategy(
        program,
        layer,
        x,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        gating,
        expert_bias,
        None,
        activation,
        MoeProjectionStrategy::GroupedGateUp,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MoeProjectionStrategy {
    PerRoute,
    GroupedGateUp,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_moe_round_output(
    program: &mut Vec<Op>,
    gate: NodeId,
    up: NodeId,
    expert_w_down: NodeId,
    route: NodeId,
    weight: NodeId,
    ones: NodeId,
    activation: Activation,
) -> Result<NodeId, TensorError> {
    let activated_gate = append_activation(program, gate, ones, activation)?;
    let hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(activated_gate, "sg->sg"), (up, "sg->sg")],
    )?;
    let down_product = gathered_expert_product(program, expert_w_down, route, hidden);
    let round_output = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sio->sio",
        "so->sio",
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(round_output, "sd->sd"), (weight, "s->sd")],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_moe_ffn_with_projection_strategy(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    expert_scale: Option<NodeId>,
    activation: Activation,
    projection_strategy: MoeProjectionStrategy,
) -> Result<(NodeId, MoeSite), TensorError> {
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sde"), (gate_inp, "de->sde")],
    )?;
    let logits = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sde->sde",
        "se->sde",
    )?;
    append_moe_ffn_with_projection_strategy_from_logits(
        program,
        layer,
        x,
        logits,
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
        projection_strategy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_moe_ffn_with_projection_strategy_from_logits(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    logits: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
    expert_scale: Option<NodeId>,
    activation: Activation,
    projection_strategy: MoeProjectionStrategy,
) -> Result<(NodeId, MoeSite), TensorError> {
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(TensorError::InvalidExpertConfig {
            expert_count,
            expert_used_count,
        });
    }

    let scores = match gating {
        ExpertGatingFunc::Softmax => logits,
        ExpertGatingFunc::Sigmoid => {
            let neg_logits = elementwise(
                program,
                DType::Float32,
                ScalarOp::Negate,
                &[(logits, "se->se")],
            )?;
            let exp_neg_logits = elementwise(
                program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(neg_logits, "se->se")],
            )?;
            let one_plus_exp = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(exp_neg_logits, "se->se"), (ones, "->se")],
            )?;
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(one_plus_exp, "se->se")],
            )?
        }
    };
    let mut selection_scores = match expert_bias {
        Some(bias) => elementwise(
            program,
            DType::Float32,
            ScalarOp::Add,
            &[(scores, "se->se"), (bias, "e->se")],
        )?,
        None => scores,
    };

    // Index expressions are carried in the compute stream as exact f32
    // values.  The gather boundary converts them to integer offsets; keeping
    // the route arithmetic in f32 is what lets the existing CPU and Metal
    // elementwise kernels execute it without a second mixed-dtype pipeline.
    let expert_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(expert_count),
        },
    );
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);

    let mut max_selection_0: Option<NodeId> = None;
    let mut selected_routes: Vec<NodeId> = Vec::with_capacity(expert_used_count as usize);
    let mut round_weights: Vec<NodeId> = Vec::with_capacity(expert_used_count as usize);
    let mut combine_weights: Vec<NodeId> = Vec::with_capacity(expert_used_count as usize);
    let mut weighted_sum = None;
    let mut weight_total = None;

    for round in 0..expert_used_count {
        let max_selection = reduce(
            program,
            DType::Float32,
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            selection_scores,
            "se->se",
            "s->se",
        )?;
        let mask = elementwise(
            program,
            DType::Float32,
            ScalarOp::Equal,
            &[(selection_scores, "se->se"), (max_selection, "s->se")],
        )?;
        let candidate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(mask, "se->se"), (expert_index, "e->se")],
        )?;
        let route = reduce(
            program,
            DType::Int32,
            ScalarOp::Maximum,
            ReduceInit::Zero,
            candidate,
            "se->se",
            "s->se",
        )?;

        let weight = match gating {
            ExpertGatingFunc::Softmax => {
                let first_max = *max_selection_0.get_or_insert(max_selection);
                let shifted = elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Subtract,
                    &[(max_selection, "s->s"), (first_max, "s->s")],
                )?;
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Exponential,
                    &[(shifted, "s->s")],
                )?
            }
            ExpertGatingFunc::Sigmoid => {
                // unbiased `scores` at the masked (selected) position, never
                // `max_selection` itself -- that would be the biased score.
                let masked_scores = elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[(mask, "se->se"), (scores, "se->se")],
                )?;
                reduce(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    ReduceInit::Zero,
                    masked_scores,
                    "se->se",
                    "s->se",
                )?
            }
        };
        selected_routes.push(route);
        round_weights.push(weight);
        // `weight_total`/`round_weights` stay on the UNSCALED softmax-over-
        // selected weight (the renormalization denominator); the per-expert
        // scale multiplies only the combination weight each round's own
        // output is scaled by, matching the authoritative
        // `topk_weights = topk_weights * expert_scales` fold applied AFTER
        // renormalization, not before it.
        let combine_weight = match expert_scale {
            Some(scale) => {
                let gathered_scale = gather_expert_scale(program, scale, route);
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[(weight, "s->s"), (gathered_scale, "s->s")],
                )?
            }
            None => weight,
        };
        combine_weights.push(combine_weight);

        if projection_strategy == MoeProjectionStrategy::PerRoute {
            let gate_product = gathered_expert_product(program, expert_w_gate, route, x);
            let gate = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sio->sio",
                "so->sio",
            )?;
            let up_product = gathered_expert_product(program, expert_w_up, route, x);
            let up = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sio->sio",
                "so->sio",
            )?;
            let weighted_round = append_moe_round_output(
                program,
                gate,
                up,
                expert_w_down,
                route,
                combine_weight,
                ones,
                activation,
            )?;
            weighted_sum = Some(match weighted_sum {
                Some(accumulated) => elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(accumulated, "sd->sd"), (weighted_round, "sd->sd")],
                )?,
                None => weighted_round,
            });
            weight_total = Some(match weight_total {
                Some(accumulated) => elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(accumulated, "s->s"), (weight, "s->s")],
                )?,
                None => weight,
            });
        }

        if round + 1 < expert_used_count {
            selection_scores = elementwise(
                program,
                DType::Float32,
                ScalarOp::Select,
                &[
                    (mask, "se->se"),
                    (neg_infinity, "->se"),
                    (selection_scores, "se->se"),
                ],
            )?;
        }
    }

    if projection_strategy == MoeProjectionStrategy::GroupedGateUp {
        let routes = stack_selected_routes(program, &selected_routes)?;
        let gate_product = grouped_gathered_expert_product(program, expert_w_gate, routes, x);
        let grouped_gate = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
            "skio->skio",
            "sko->skio",
        )?;
        let up_product = grouped_gathered_expert_product(program, expert_w_up, routes, x);
        let grouped_up = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_product,
            "skio->skio",
            "sko->skio",
        )?;

        for (round, ((route, weight), combine_weight)) in selected_routes
            .iter()
            .copied()
            .zip(round_weights.iter().copied())
            .zip(combine_weights.iter().copied())
            .enumerate()
        {
            let round = u32::try_from(round).map_err(|_| TensorError::InvalidExpertConfig {
                expert_count,
                expert_used_count,
            })?;
            let gate = select_grouped_round(program, grouped_gate, round, DType::Float32);
            let up = select_grouped_round(program, grouped_up, round, DType::Float32);
            let weighted_round = append_moe_round_output(
                program,
                gate,
                up,
                expert_w_down,
                route,
                combine_weight,
                ones,
                activation,
            )?;
            weighted_sum = Some(match weighted_sum {
                Some(accumulated) => elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(accumulated, "sd->sd"), (weighted_round, "sd->sd")],
                )?,
                None => weighted_round,
            });
            weight_total = Some(match weight_total {
                Some(accumulated) => elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(accumulated, "s->s"), (weight, "s->s")],
                )?,
                None => weight,
            });
        }
    }

    let weighted_sum = weighted_sum.ok_or(TensorError::InvalidExpertConfig {
        expert_count,
        expert_used_count,
    })?;
    let weight_total = weight_total.ok_or(TensorError::InvalidExpertConfig {
        expert_count,
        expert_used_count,
    })?;
    let inv_weight_total = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_total, "s->s")],
    )?;
    let output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weighted_sum, "sd->sd"), (inv_weight_total, "s->sd")],
    )?;
    round_weights.push(weight_total);
    let site = MoeSite {
        layer,
        selected: selected_routes,
        weights: round_weights,
    };
    Ok((output, site))
}

/// [`append_mistral_layer`]'s mixture-of-experts counterpart: identical
/// attention block (RoPE + GQA + causal mask, node-for-node the same code),
/// [`append_moe_ffn`] in place of the dense SwiGLU triple. Kept as a
/// separate function rather than a branch inside [`append_mistral_layer`]
/// so the dense path's own node sequence — and therefore its generated
/// program bytes — never changes shape by so much as one node merely
/// because this function exists next to it.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_moe_layer(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<(NodeId, MoeSite), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let (rotated_q_even, rotated_q_odd) =
        fused_rope_pair(program, q, 'h', cos, sin, RopePairing::Interleaved)?;

    let (rotated_k_even, rotated_k_odd) =
        fused_rope_pair(program, k, 'u', cos, sin, RopePairing::Interleaved)?;

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

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let (ffn_out, site) = append_moe_ffn(
        program,
        layer,
        normed2,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        ExpertGatingFunc::Softmax,
        None,
        Activation::Silu,
    )?;

    let output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;
    Ok((output, site))
}
