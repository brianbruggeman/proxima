//! `qwen35moe`'s forward program: composes proxima's `proxima-tensor` `spec`
//! builders the same way [`crate::qwen4exp::program`] does -- token
//! embedding, [`append_qwen35_dense_attention_only`] on dense-attention
//! layers / [`append_qwen35_ssm_mixer_with_taps`] on GDN layers (Qwen3.5's own hybrid
//! split, `proxima_tensor::spec::qwen35_forward_program`'s worked example),
//! `post_attention_norm` -> [`append_moe_ffn`] (routed) plus
//! [`super::shared_expert::append_sigmoid_gated_shared_expert`] (the gated
//! shared expert every layer also carries), residual add, and a final
//! `output_norm` + `lm_head`.
//!
//! # What this covers, and the known deviation from the reference
//!
//! [`append_qwen35_dense_attention_only`] (proxima main `950bb363`) is
//! [`proxima_tensor::spec::append_qwen35_dense_attention_layer`]'s own
//! attention block, split out because `qwen35moe`'s attention layers carry
//! no dense `blk.N.ffn_{gate,up,down}.weight` at all -- every layer's FFN is
//! routed, `qwen35_forward_program`'s own fused builder has no seam for
//! that. `rotary_dim` is [`Architecture::rope_dims`]
//! (`{architecture}.rope.dimension_count`), never `attn_head_dim` -- the
//! real checkpoint declares a PARTIAL rotary width (`64` of a `256`-wide
//! `attn_head_dim`), the same rotary-vs-real-head-width split proxima's own
//! dense `qwen35` bind's `head_dim`/`attn_head_dim` pair already makes
//! (`proxima-model-interop/src/qwen35.rs`).
//!
//! The real checkpoint also carries `{architecture}.rope.dimension_sections`/
//! `rope.mrope_section` (`[11, 11, 10]`, summing to `rope_dims / 2` pairs)
//! and `rope.mrope_interleaved` -- Qwen2-VL/Qwen3-VL-style multi-axis
//! (text/height/width) RoPE. This program still applies one uniform
//! split-half rotation across the full `rope_dims` width, the same single-
//! axis rotation the dense `qwen35` arm applies (that arch carries no
//! `mrope_*` keys at all): a real per-section multi-axis rotation is a
//! follow-up, the same way [`crate::qwen4exp::program`]'s own doc names its
//! deviations.
//!
//! # Teaching pointer
//!
//! Composes [`proxima_tensor::spec::embedding_lookup`], [`causal_mask`],
//! [`append_qwen35_dense_attention_only`] / [`append_qwen35_ssm_mixer_with_taps`] per
//! layer, [`rmsnorm`] for `post_attention_norm`, [`append_moe_ffn`] for the
//! routed FFN, [`super::shared_expert::append_sigmoid_gated_shared_expert`]
//! for the gated shared expert, and [`elementwise`]/[`reduce`] for the
//! residual adds and `lm_head`. No new abstraction belongs here -- every
//! step is `Vec<Op>` plus a `spec` builder or a direct `elementwise`/
//! `reduce` call, the same discipline
//! [`crate::qwen4exp::program::qwen4exp_forward_program`] follows.

use proxima_tensor::spec::{
    ExpertGatingFunc, ForwardRoots, GdnOutputGate, MoeSite, MoeSites, Qwen35DenseAttentionTaps,
    Qwen35GdnSequenceTail, Qwen35LayerRoots, SsmMixerTaps, append_moe_ffn,
    append_qwen35_dense_attention_only_with_taps, append_qwen35_gdn_sequence_tail,
    append_qwen35_ssm_mixer_with_taps_and_layout, causal_mask, elementwise, embedding_lookup,
    input_leaf, reduce, rmsnorm, scalar_constant, symbolic_leaf,
};
use proxima_tensor::{DType, Extent, NodeId, Op, ReduceInit, ScalarOp};

use super::hparams::{Architecture, LayerKind};
use super::shared_expert::append_sigmoid_gated_shared_expert;
use crate::error::InteropError;

fn append_qwen35moe_router(
    program: &mut Vec<Op>,
    mixer_out: NodeId,
    post_attention_norm_weight: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    gate_inp: NodeId,
) -> Result<(NodeId, NodeId), proxima_tensor::TensorError> {
    let normed = rmsnorm(
        program,
        mixer_out,
        post_attention_norm_weight,
        inv_dim,
        eps,
    )?;
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sde"), (gate_inp, "de->sde")],
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
    Ok((normed, router_logits))
}

/// Runs `x`'s post-attention-norm hidden state through the routed
/// [`append_moe_ffn`] plus the gated shared expert, and adds the result back
/// onto `mixer_out` (the pre-FFN residual stream) -- the one FFN sub-block
/// shape every `qwen35moe` layer shares, dense-attention or GDN alike.
#[allow(clippy::too_many_arguments)]
fn append_qwen35moe_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    mixer_out: NodeId,
    post_attention_norm_weight: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    one: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    gate_inp_shexp: NodeId,
    gate_shexp: NodeId,
    up_shexp: NodeId,
    down_shexp: NodeId,
) -> Result<(NodeId, MoeSite, NodeId, NodeId, NodeId, NodeId), proxima_tensor::TensorError> {
    // `append_moe_ffn`'s own leading two ops, shared with the batched GDN
    // prefill route so both paths select experts from the same algebra.
    let (normed, router_logits) = append_qwen35moe_router(
        program,
        mixer_out,
        post_attention_norm_weight,
        inv_dim,
        eps,
        gate_inp,
    )?;

    let (routed_out, moe_site) = append_moe_ffn(
        program,
        layer,
        normed,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        one,
        ExpertGatingFunc::Softmax,
        None,
    )?;
    let shared_out = append_sigmoid_gated_shared_expert(
        program,
        normed,
        gate_inp_shexp,
        gate_shexp,
        up_shexp,
        down_shexp,
        one,
    )?;

    let ffn_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(routed_out, "sd->sd"), (shared_out, "sd->sd")],
    )?;
    let residual = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (mixer_out, "sd->sd")],
    )?;
    Ok((
        residual,
        moe_site,
        normed,
        router_logits,
        routed_out,
        shared_out,
    ))
}

/// One layer's own diagnostic [`NodeId`]s -- production-neutral: every field
/// is a root the forward program already computes for its normal decode
/// work, just named and returned so a test can request them from
/// [`proxima_model_interop::LoadedModel::forward_node_values`] without
/// hand-counting `Op::Input`s the way earlier bisection tests in this crate
/// did. `ssm_taps` is `Some` only for [`LayerKind::Gdn`] layers,
/// `dense_attention_taps` only for [`LayerKind::Attention`] layers
/// -- exactly one of the two is `Some` for any given layer.
///
/// `mixer_output` is the mixer's own PRE-residual result (`taps.ssm_out_result`
/// for a GDN layer, `taps.o_proj_out` for a dense-attention layer --
/// [`proxima_tensor::spec::Qwen35DenseAttentionTaps::o_proj_out`], the
/// `o_proj` reduce before its own residual add); `post_mixer_residual` is
/// that result added back onto this layer's `block_input` (what the mixer
/// builders themselves call `mixer_out`/`x_next`/`residual1`).
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MoeLayerDiagnostics {
    pub block_input: NodeId,
    pub ssm_taps: Option<SsmMixerTaps>,
    pub dense_attention_taps: Option<Qwen35DenseAttentionTaps>,
    pub mixer_output: NodeId,
    pub post_mixer_residual: NodeId,
    pub post_attention_norm_output: NodeId,
    pub router_logits: NodeId,
    pub routed_output: NodeId,
    pub shared_output: NodeId,
    pub block_output: NodeId,
    pub gdn_prefill: Option<Qwen35MoeGdnPrefillTaps>,
}

/// Sequence-preserving roots used by the caller-driven GDN prefill scan.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MoeGdnPrefillTaps {
    pub delta_out_input: NodeId,
    pub post_mixer_residual: NodeId,
    pub post_attention_norm_output: NodeId,
    pub router_logits: NodeId,
}

/// Builds `qwen35moe`'s whole-model forward program -- see the module doc
/// for the worked example and the one known deviation from the real
/// checkpoint (full-width rotary, no partial-rotary "pass" columns).
///
/// # Errors
///
/// [`Error::TensorProgram`] if any composed builder fails to lower.
/// [`qwen35moe_forward_program`]'s own return shape: the built `Vec<Op>`,
/// its logits/hidden roots, each layer's production cache-root tag, every
/// [`MoeSite`] `append_moe_ffn` registered, and each layer's own
/// [`Qwen35MoeLayerDiagnostics`] side table.
pub type Qwen35MoeForwardProgram = (
    Vec<Op>,
    ForwardRoots,
    Vec<Qwen35LayerRoots>,
    MoeSites,
    Vec<Qwen35MoeLayerDiagnostics>,
);

#[allow(clippy::too_many_lines)]
pub fn qwen35moe_forward_program(
    architecture: &Architecture,
) -> Result<Qwen35MoeForwardProgram, InteropError> {
    let embedding = architecture.embedding;
    let attn_head_dim = architecture.attn_head_dim;
    let rope_dims = architecture.rope_dims;
    let pairs = rope_dims / 2;
    let pass_dim = attn_head_dim - rope_dims;

    let mut program = Vec::new();

    let ids = input_leaf(&mut program, DType::Int32, vec![Extent::Symbolic(0)], "ids");
    let table = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(architecture.vocab),
            Extent::Static(embedding),
        ],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let one = ones;

    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (attn_head_dim as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );

    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count.max(1);
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank.max(1);
    let inv_sqrt_key_dim = scalar_constant(
        &mut program,
        1.0 / (architecture.ssm_state_size as f32).sqrt(),
    );
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim.max(1) as f32);
    let head_eps = proxima_tensor::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(architecture.ssm_group_count),
                Extent::Static(ssm_group),
            ],
            value: architecture.rms_epsilon,
        },
    );

    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut layer_roots: Vec<Qwen35LayerRoots> = Vec::with_capacity(architecture.layer_kinds.len());
    let mut moe_sites: Vec<MoeSite> = Vec::with_capacity(architecture.layer_kinds.len());
    let mut layer_diagnostics: Vec<Qwen35MoeLayerDiagnostics> =
        Vec::with_capacity(architecture.layer_kinds.len());

    for (layer, kind) in architecture.layer_kinds.iter().enumerate() {
        let layer = layer as u32;
        let block_input = x;

        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Static(embedding)],
            &format!("blk.{layer}.attn_norm.weight"),
        );
        let post_attention_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Static(embedding)],
            &format!("blk.{layer}.post_attention_norm.weight"),
        );
        let gate_inp = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(embedding),
                Extent::Static(architecture.expert_count),
            ],
            &format!("blk.{layer}.ffn_gate_inp.weight"),
        );

        let (
            mixer_out,
            ssm_taps,
            dense_attention_taps,
            mixer_output_pre_residual,
            gdn_prefill,
        ) = match kind {
            LayerKind::Attention => {
                let kv_heads = architecture.kv_heads_by_layer[layer as usize];
                let group = architecture.query_heads / kv_heads.max(1);
                let group_ones = proxima_tensor::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: vec![Extent::Static(kv_heads), Extent::Static(group)],
                        value: 1.0,
                    },
                );

                // `attn_q.weight` carries `[Q | gate]` per head, doubled
                // width -- the same fused Q-gate `proxima_tensor::spec`'s own
                // real `qwen35` dense checkpoint reads off `attn_q.weight`
                // (`spec.rs:8908-8944`). Reshaped via the same lossless
                // broadcast-multiply-by-ones trick `wk`/`wv` use below, NEVER
                // a per-head WEIGHT-level slice -- splitting a packed
                // quantized weight per head before the real contraction runs
                // breaks `cpu::is_quantized_matmul_operand`'s recognizer,
                // which then derives the packed row length from the wrong
                // axis (`per_head_channel_slice`'s own former call site here).
                // `append_qwen35_dense_attention_only_with_taps` does the
                // real `x_normed @ wq_gate` contraction and narrows to
                // `q`/`gate` per head on the ACTIVATION via
                // `per_head_channel_range`.
                let wq_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(architecture.query_heads * attn_head_dim * 2),
                    ],
                    &format!("blk.{layer}.attn_q.weight"),
                );
                let qg_head_ones = proxima_tensor::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: vec![
                            Extent::Static(architecture.query_heads),
                            Extent::Static(attn_head_dim * 2),
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
                            format!("i,{}*h+c->ihc", attn_head_dim * 2).as_str(),
                        ),
                        (qg_head_ones, "hc->ihc"),
                    ],
                )?;

                // `wk`/`wv`/`wo` reshape their own flat matmul-bound leaf via
                // the same lossless broadcast-multiply-by-ones trick
                // proxima's own dense `qwen35_forward_program` uses
                // (`proxima-tensor/src/spec.rs:8777-8853`) rather than
                // declaring the multi-axis shape directly on the `Op::Input`
                // leaf: the leaf's on-disk bytes are `bind_matmul_weight`'s
                // dequantized-and-transposed `[in_dim, out_dim]` buffer
                // (`bind.rs`'s `dense_attention_out_in_dims`), and a flat 2-D
                // leaf declaration is what actually matches that buffer's
                // real byte order -- baking `kv_heads`/`attn_head_dim` into
                // the leaf's own axis list here left the interpreter reading
                // those bytes under the wrong axis order.
                let wk_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads * attn_head_dim),
                    ],
                    &format!("blk.{layer}.attn_k.weight"),
                );
                let k_head_ones = proxima_tensor::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                        value: 1.0,
                    },
                );
                let wk = elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[
                        (wk_flat, format!("i,{attn_head_dim}*u+d->iud").as_str()),
                        (k_head_ones, "ud->iud"),
                    ],
                )?;

                let wv_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads * attn_head_dim),
                    ],
                    &format!("blk.{layer}.attn_v.weight"),
                );
                let v_head_ones = proxima_tensor::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                        value: 1.0,
                    },
                );
                let wv = elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[
                        (wv_flat, format!("i,{attn_head_dim}*u+d->iud").as_str()),
                        (v_head_ones, "ud->iud"),
                    ],
                )?;

                let wo_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(architecture.query_heads * attn_head_dim),
                        Extent::Static(embedding),
                    ],
                    &format!("blk.{layer}.attn_output.weight"),
                );
                let o_head_ones = proxima_tensor::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: vec![
                            Extent::Static(kv_heads),
                            Extent::Static(group),
                            Extent::Static(attn_head_dim),
                        ],
                        value: 1.0,
                    },
                );
                let wo = elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[
                        (
                            wo_flat,
                            format!("{}*u+{attn_head_dim}*g+d,e->ugde", attn_head_dim * group)
                                .as_str(),
                        ),
                        (o_head_ones, "ugd->ugde"),
                    ],
                )?;

                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(attn_head_dim)],
                    &format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(attn_head_dim)],
                    &format!("blk.{layer}.attn_k_norm.weight"),
                );

                let k_first_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs),
                    ],
                    &format!("kv_cache.{layer}.k_first"),
                );
                let k_second_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs),
                    ],
                    &format!("kv_cache.{layer}.k_second"),
                );
                let k_pass_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pass_dim),
                    ],
                    &format!("kv_cache.{layer}.k_pass"),
                );
                let v_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(attn_head_dim),
                    ],
                    &format!("kv_cache.{layer}.v"),
                );

                // `_with_taps` -- byte-identical program to
                // `append_qwen35_dense_attention_only`
                // (`dense_attention_only_and_with_taps_produce_the_same_program`
                // proves it in proxima-tensor), so this is a diagnostic-only
                // change, never a production behaviour change.
                let (residual1, dense_taps) = append_qwen35_dense_attention_only_with_taps(
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
                    rope_dims,
                    attn_head_dim,
                    attn_norm_weight,
                    q_norm_weight,
                    k_norm_weight,
                    wq_gate,
                    wk,
                    wv,
                    wo,
                    k_first_cache,
                    k_second_cache,
                    k_pass_cache,
                    v_cache,
                )?;
                layer_roots.push(Qwen35LayerRoots::DenseAttention((
                    dense_taps.rotated_k_new_first,
                    dense_taps.rotated_k_new_second,
                    dense_taps.k_pass,
                    dense_taps.v_new,
                )));
                let o_proj_out = dense_taps.o_proj_out;
                (residual1, None, Some(dense_taps), o_proj_out, None)
            }
            LayerKind::Gdn => {
                let qkv_dim = 2 * ssm_key_dim + architecture.ssm_inner_size;
                let wqkv = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
                    &format!("blk.{layer}.attn_qkv.weight"),
                );
                let wqkv_gate = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(architecture.ssm_inner_size),
                    ],
                    &format!("blk.{layer}.attn_gate.weight"),
                );
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(qkv_dim),
                        Extent::Static(architecture.ssm_conv_kernel),
                    ],
                    &format!("blk.{layer}.ssm_conv1d.weight"),
                );
                let conv_history_in = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(architecture.ssm_conv_kernel.saturating_sub(1)),
                        Extent::Static(qkv_dim),
                    ],
                    &format!("ssm_cache.{layer}.conv_history"),
                );
                let ssm_beta = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(architecture.ssm_time_step_rank),
                    ],
                    &format!("blk.{layer}.ssm_beta.weight"),
                );
                let ssm_alpha = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(embedding),
                        Extent::Static(architecture.ssm_time_step_rank),
                    ],
                    &format!("blk.{layer}.ssm_alpha.weight"),
                );
                let ssm_dt_bias = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(architecture.ssm_time_step_rank)],
                    &format!("blk.{layer}.ssm_dt.bias"),
                );
                let ssm_a = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(architecture.ssm_time_step_rank)],
                    &format!("blk.{layer}.ssm_a"),
                );
                let ssm_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![Extent::Static(head_v_dim)],
                    &format!("blk.{layer}.ssm_norm.weight"),
                );
                let ssm_out = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(architecture.ssm_inner_size),
                        Extent::Static(embedding),
                    ],
                    &format!("blk.{layer}.ssm_out.weight"),
                );
                let state_in = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Static(architecture.ssm_state_size),
                        Extent::Static(head_v_dim),
                        Extent::Static(architecture.ssm_group_count),
                        Extent::Static(ssm_group),
                    ],
                    &format!("ssm_cache.{layer}.state"),
                );

                // `_with_taps` -- byte-identical program to
                // `append_qwen35_ssm_mixer` (that wrapper's own doc: "this
                // only reshapes the return value the shared builder already
                // computed"), so this is a diagnostic-only change, never a
                // production behaviour change.
                let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
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
                    architecture.ssm_inner_size,
                    architecture.ssm_group_count,
                    ssm_group,
                    architecture.ssm_conv_kernel,
                    GdnOutputGate::Silu,
                    architecture.v_head_reordered,
                )?;
                layer_roots.push(Qwen35LayerRoots::Ssm {
                    qkv_mixed: taps.qkv_mixed,
                    state_out: taps.state_out,
                });
                let delta_out_input = input_leaf(
                    &mut program,
                    DType::Float32,
                    vec![
                        Extent::Symbolic(0),
                        Extent::Static(head_v_dim),
                        Extent::Static(architecture.ssm_group_count),
                        Extent::Static(ssm_group),
                    ],
                    &format!("gdn_prefill.{layer}.delta_out"),
                );
                let prefill_mixer_out = append_qwen35_gdn_sequence_tail(
                    &mut program,
                    Qwen35GdnSequenceTail {
                        x,
                        delta_out: delta_out_input,
                        z: taps.z_sequence,
                        head_eps,
                        inv_head_v_dim,
                        norm_weight: ssm_norm_weight,
                        out_weight: ssm_out,
                        head_v_dim,
                        kv_heads: architecture.ssm_group_count,
                        group: ssm_group,
                    },
                )?;
                let (prefill_normed, prefill_router_logits) = append_qwen35moe_router(
                    &mut program,
                    prefill_mixer_out,
                    post_attention_norm_weight,
                    inv_dim,
                    eps,
                    gate_inp,
                )?;
                let ssm_out_result = taps.ssm_out_result;
                (
                    mixer_out,
                    Some(taps),
                    None,
                    ssm_out_result,
                    Some(Qwen35MoeGdnPrefillTaps {
                        delta_out_input,
                        post_mixer_residual: prefill_mixer_out,
                        post_attention_norm_output: prefill_normed,
                        router_logits: prefill_router_logits,
                    }),
                )
            }
        };

        let expert_w_gate = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.expert_count),
                Extent::Static(embedding),
                Extent::Static(architecture.expert_feed_forward),
            ],
            &format!("blk.{layer}.ffn_gate_exps.weight"),
        );
        let expert_w_up = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.expert_count),
                Extent::Static(embedding),
                Extent::Static(architecture.expert_feed_forward),
            ],
            &format!("blk.{layer}.ffn_up_exps.weight"),
        );
        let expert_w_down = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.expert_count),
                Extent::Static(architecture.expert_feed_forward),
                Extent::Static(embedding),
            ],
            &format!("blk.{layer}.ffn_down_exps.weight"),
        );
        let gate_inp_shexp = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Static(embedding)],
            &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        );
        let gate_shexp = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(embedding),
                Extent::Static(architecture.expert_shared_feed_forward),
            ],
            &format!("blk.{layer}.ffn_gate_shexp.weight"),
        );
        let up_shexp = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(embedding),
                Extent::Static(architecture.expert_shared_feed_forward),
            ],
            &format!("blk.{layer}.ffn_up_shexp.weight"),
        );
        let down_shexp = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(architecture.expert_shared_feed_forward),
                Extent::Static(embedding),
            ],
            &format!("blk.{layer}.ffn_down_shexp.weight"),
        );

        let (
            residual,
            moe_site,
            post_attention_norm_output,
            router_logits,
            routed_output,
            shared_output,
        ) = append_qwen35moe_ffn(
            &mut program,
            layer,
            mixer_out,
            post_attention_norm_weight,
            inv_dim,
            eps,
            one,
            gate_inp,
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            architecture.expert_count,
            architecture.expert_used_count,
            gate_inp_shexp,
            gate_shexp,
            up_shexp,
            down_shexp,
        )?;
        layer_diagnostics.push(Qwen35MoeLayerDiagnostics {
            block_input,
            ssm_taps,
            dense_attention_taps,
            mixer_output: mixer_output_pre_residual,
            post_mixer_residual: mixer_out,
            post_attention_norm_output,
            router_logits,
            routed_output,
            shared_output,
            block_output: residual,
            gdn_prefill,
        });
        x = residual;
        moe_sites.push(moe_site);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    // The decode loop only ever samples the LAST new position's logits
    // (`proxima_model_interop::generate`'s own `logits[(new_count - 1) *
    // vocab_size..]` slice) -- gather that one row of `normed_final` BEFORE
    // the vocab-sized `output.weight` matmul, mirroring
    // `proxima_tensor::spec::mistral_cached_forward_program_with_experts_and_layer_taps`'s
    // own `last_row_only` leaf (`lm_head_row`, host-supplied unconditionally
    // by `generate.rs` every step): a static `[1]` `Op::Input`, not a new
    // symbol slot, since the gather always produces exactly one row
    // regardless of how many new positions this step computed.
    // `ForwardRoots::hidden` stays `normed_final` (every row) -- see
    // `crate::qwen4exp::program`'s identical comment on its own
    // `final_mixed`/`final_mixed_last` split.
    let lm_head_row = input_leaf(
        &mut program,
        DType::Int32,
        vec![Extent::Static(1)],
        "lm_head_row",
    );
    let normed_last = embedding_lookup(&mut program, normed_final, lm_head_row);

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(embedding),
            Extent::Static(architecture.vocab),
        ],
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
            hidden: normed_final,
        },
        layer_roots,
        MoeSites(moe_sites),
        layer_diagnostics,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_moe_program_builds_one_gdn_and_one_attention_layer() {
        let architecture = Architecture {
            vocab: 16,
            embedding: 8,
            query_heads: 2,
            kv_heads_by_layer: vec![0, 1],
            attn_head_dim: 4,
            rope_dims: 2,
            rope_dimension_sections: vec![1],
            rope_mrope_interleaved: false,
            block_count: 2,
            full_attention_interval: 2,
            rope_freq_base: 10_000.0,
            rms_epsilon: 1e-6,
            ssm_conv_kernel: 2,
            ssm_state_size: 2,
            ssm_group_count: 1,
            ssm_time_step_rank: 2,
            ssm_inner_size: 4,
            v_head_reordered: false,
            expert_count: 2,
            expert_used_count: 1,
            expert_feed_forward: 4,
            expert_shared_feed_forward: 4,
            layer_kinds: vec![LayerKind::Gdn, LayerKind::Attention],
        };

        let (program, roots, layer_roots, moe_sites, diagnostics) =
            qwen35moe_forward_program(&architecture).expect("hybrid MoE program lowers");

        assert!(!program.is_empty(), "the forward graph has operations");
        assert_eq!(layer_roots.len(), 2, "one cache root per layer");
        assert_eq!(moe_sites.0.len(), 2, "one router site per layer");
        assert_eq!(diagnostics.len(), 2, "one diagnostic record per layer");
        let gdn_taps = diagnostics[0]
            .ssm_taps
            .expect("the first synthetic layer is the gdn layer");
        assert!(
            gdn_taps.query_sequence.0 < gdn_taps.state_out.0
                && gdn_taps.key_sequence.0 < gdn_taps.state_out.0
                && gdn_taps.value_sequence.0 < gdn_taps.state_out.0
                && gdn_taps.gate_sequence.0 < gdn_taps.state_out.0
                && gdn_taps.beta_sequence.0 < gdn_taps.state_out.0,
            "batched recurrence inputs must precede the state transition"
        );
        assert!(
            gdn_taps.z_head.0 < gdn_taps.delta_out.0
                && gdn_taps.state_in.0 < gdn_taps.state_out.0,
            "the production scan cut must expose its carried inputs before recurrence"
        );
        assert_ne!(
            roots.logits, roots.hidden,
            "logits follow the output projection"
        );
    }
}
