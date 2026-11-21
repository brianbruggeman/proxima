//! Model-agnostic lowerings for diffusion-style backbones: a norm with no
//! learned affine, adaptive modulation, bidirectional attention, multi-axis
//! RoPE tables and a layer-tap stack.
//!
//! Every builder composes the existing algebra ([`elementwise_like`],
//! [`reduce_like`], [`scalar_constant_like`], [`super::fused_rope_pair`]) and nothing
//! else, so each is `Elementwise`/`Reduce`/`Iota`/`Constant` and autograd can
//! differentiate it. Element types are read off the nodes passed in, never
//! named here. Learn the pattern from [`super::rmsnorm`] (the affine, root-mean-square
//! sibling of [`layer_norm_no_affine`]) and from the attention core shared with
//! [`super::append_attention_mixer`], whose causal mask is the switch
//! [`non_causal_attention`] leaves off.

use alloc::vec::Vec;

use crate::error::TensorError;
use crate::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_numeric::{DType, FormatCodec};

use super::attention_forward::append_masked_softmax_mix;
use super::{elementwise_like, produced_dtype_or_binary32, reduce_like, scalar_constant_like};

fn row_mean(program: &mut Vec<Op>, x: NodeId, inv_width: NodeId) -> Result<NodeId, TensorError> {
    let sum = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        x,
        "sd->sd",
        "s->sd",
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(sum, "s->s"), (inv_width, "->s")],
    )
}

/// `(x - mean(x)) / sqrt(var(x) + eps)` over the last axis of `x` (`[s, d]`,
/// `d == width`), with no learned scale or shift: the DiT pre-modulation norm,
/// and the `elementwise_affine=false` case of LayerNorm. Compose
/// [`modulate`] after it for adaLN; [`super::rmsnorm`] is the centred-less affine
/// sibling.
///
/// # Errors
///
/// [`TensorError`] when `x` is not a node of `program`.
pub fn layer_norm_no_affine(
    program: &mut Vec<Op>,
    x: NodeId,
    width: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let inv_width = scalar_constant_like(program, 1.0 / width as f32, x)?;
    let mean = row_mean(program, x, inv_width)?;
    let centered = elementwise_like(
        program,
        x,
        ScalarOp::Subtract,
        &[(x, "sd->sd"), (mean, "s->sd")],
    )?;
    let squared = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "sd->sd"), (centered, "sd->sd")],
    )?;
    let variance = row_mean(program, squared, inv_width)?;
    let eps_node = scalar_constant_like(program, eps, x)?;
    let spread = elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(variance, "s->s"), (eps_node, "->s")],
    )?;
    let deviation = elementwise_like(program, x, ScalarOp::SquareRoot, &[(spread, "s->s")])?;
    let inverse = elementwise_like(program, x, ScalarOp::Reciprocal, &[(deviation, "s->s")])?;
    elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "sd->sd"), (inverse, "s->sd")],
    )
}

/// adaLN modulation: `normed * (1 + scale) + shift`, with `shift` and `scale`
/// one `[d]` vector each (per sample, broadcast over the sequence axis).
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn modulate(
    program: &mut Vec<Op>,
    normed: NodeId,
    shift: NodeId,
    scale: NodeId,
) -> Result<NodeId, TensorError> {
    let one = scalar_constant_like(program, 1.0, normed)?;
    let gain = elementwise_like(
        program,
        normed,
        ScalarOp::Add,
        &[(scale, "d->d"), (one, "->d")],
    )?;
    let scaled = elementwise_like(
        program,
        normed,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (gain, "d->sd")],
    )?;
    elementwise_like(
        program,
        normed,
        ScalarOp::Add,
        &[(scaled, "sd->sd"), (shift, "d->sd")],
    )
}

/// Softmax attention with every key visible to every query: `q` is
/// `[s, u, i]`, `k` is `[t, u, i]`, `v` is `[t, u, d]`, the result is
/// `[s, u, 1, d]` (row-major identical to `[s, u, d]`). `s` and `t` are
/// independent extents, so this also serves cross and joint attention. The
/// scale/softmax/mix is `append_masked_softmax_mix`, the very core
/// [`super::append_attention_mixer`] runs with its causal mask; here the mask is
/// `None`. `inv_sqrt_head_dim` is a rank-0 node (e.g. [`scalar_constant_like`]).
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn non_causal_attention(
    program: &mut Vec<Op>,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    inv_sqrt_head_dim: NodeId,
) -> Result<NodeId, TensorError> {
    let group_dtype = produced_dtype_or_binary32(program, q);
    let single_group = op::append(
        program,
        Op::Constant {
            dtype: group_dtype,
            shape: alloc::vec![Extent::Static(1)],
            value: 1.0,
        },
    );
    let grouped = elementwise_like(
        program,
        q,
        ScalarOp::Multiply,
        &[(q, "sui->sugi"), (single_group, "g->sugi")],
    )?;
    let product = elementwise_like(
        program,
        q,
        ScalarOp::Multiply,
        &[(grouped, "sugi->stugi"), (k, "tui->stugi")],
    )?;
    let scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    append_masked_softmax_mix(program, scores, inv_sqrt_head_dim, None, None, 1, v)
}

fn encode_table(values: &[f32], dtype: DType) -> Result<Vec<u8>, TensorError> {
    let unsupported = TensorError::NotLowerable {
        node: NodeId(0),
        reason: "rope table dtype has no scalar codec",
    };
    let format = dtype.format().ok_or(unsupported.clone())?;
    let codec = FormatCodec::for_format(&format).ok_or(unsupported.clone())?;
    let mut bytes = alloc::vec![0u8; codec.packed_len(values.len()).ok_or(unsupported)?];
    codec.encode(values, &mut bytes)?;
    Ok(bytes)
}

/// Host-side `(cos, sin)` tables for RoPE whose head dimension is split into
/// axis sections (`sections[a]` channels rotate by position stream `a` at base
/// `thetas[a]`), encoded as `dtype` bytes through [`FormatCodec`], each
/// `[rows, sum(sections) / 2]`. `positions` is `[rows, sections.len()]`
/// row-major; `dtype` is the table leaf's own element type. Feed the tables to
/// [`super::fused_rope_pair`] with `RopePairing::Interleaved`: that builder already
/// rotates adjacent channel pairs, and each axis owns a contiguous pair range,
/// so the IR is the existing one and only the table is new. Single-axis RoPE
/// is `sections.len() == 1`.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when a section is odd, `thetas` and
/// `sections` differ in length, `positions` is not a whole number of rows, or
/// `dtype` cannot be encoded.
pub fn multi_axis_rope_tables(
    positions: &[f32],
    sections: &[u32],
    thetas: &[f32],
    dtype: DType,
) -> Result<(Vec<u8>, Vec<u8>), TensorError> {
    let axes = sections.len();
    let well_formed = axes > 0
        && thetas.len() == axes
        && positions.len().is_multiple_of(axes)
        && sections.iter().all(|section| section % 2 == 0);
    if !well_formed {
        return Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "multi-axis rope needs even sections, a theta per axis and whole rows",
        });
    }
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for row in positions.chunks(axes) {
        for (axis, &section) in sections.iter().enumerate() {
            for pair in 0..section / 2 {
                let exponent = -((2 * pair) as f32) / section as f32;
                let angle = row[axis] * libm::powf(thetas[axis], exponent);
                cos.push(libm::cosf(angle));
                sin.push(libm::sinf(angle));
            }
        }
    }
    Ok((encode_table(&cos, dtype)?, encode_table(&sin, dtype)?))
}

/// Stacks the outputs of the chosen layers into one `[s, taps, d]` node:
/// `layer_outputs[taps[k]]` (each `[s, d]`) lands at index `k` of the middle
/// axis. A consumer that wants the concatenated feature axis reads it as
/// `[s, taps * d]` (row-major layout makes that a relabel, not a copy).
/// Each tap is routed by one `Select` on an `Iota == k` mask, so the stack is
/// plain `Elementwise` and gradients reach every tapped layer.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `taps` is empty or names a layer past
/// `layer_outputs`.
pub fn tap_layers(
    program: &mut Vec<Op>,
    layer_outputs: &[NodeId],
    taps: &[u32],
) -> Result<NodeId, TensorError> {
    let chosen = taps
        .iter()
        .map(|&layer| layer_outputs.get(layer as usize).copied())
        .collect::<Option<Vec<NodeId>>>()
        .filter(|chosen| !chosen.is_empty())
        .ok_or(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "tap_layers needs at least one tap and every tap inside the layer list",
        })?;
    let slot_dtype = produced_dtype_or_binary32(program, chosen[0]);
    let slots = op::append(
        program,
        Op::Iota {
            dtype: slot_dtype,
            extent: Extent::Static(chosen.len() as u32),
        },
    );
    let mut stacked = scalar_constant_like(program, 0.0, chosen[0])?;
    for (slot, &layer) in chosen.iter().enumerate() {
        let index = scalar_constant_like(program, slot as f32, slots)?;
        let is_slot = elementwise_like(
            program,
            slots,
            ScalarOp::Equal,
            &[(slots, "k->k"), (index, "->k")],
        )?;
        let acc_map = if slot == 0 { "->skd" } else { "skd->skd" };
        stacked = elementwise_like(
            program,
            layer,
            ScalarOp::Select,
            &[(is_slot, "k->skd"), (layer, "sd->skd"), (stacked, acc_map)],
        )?;
    }
    Ok(stacked)
}
