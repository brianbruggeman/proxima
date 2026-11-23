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

use alloc::format;
use alloc::vec::Vec;

use crate::error::TensorError;
use crate::map::{self, AxisIndex, AxisTerm, IndexPattern};
use crate::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_numeric::{DType, FormatCodec};

use super::attention_forward::append_masked_softmax_mix;
use super::{
    elementwise_like, input_leaf, produced_dtype_or_binary32, reduce_like, scalar_constant_like,
};

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

/// Affine LayerNorm: `layer_norm_no_affine(x) * weight + bias` over the last axis of
/// `x` (`[s, d]`), with `weight` and `bias` one `[d]` vector each broadcast over `s`.
/// Composes [`layer_norm_no_affine`] with an `Elementwise` multiply and add (the
/// `"d->sd"` operand map [`modulate`] uses); no new `Reduce`, so autograd differentiates
/// it with the rules it already has. Reach for [`layer_norm_no_affine`] when the norm
/// carries no learned parameters.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn layer_norm(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    bias: NodeId,
    width: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let normed = layer_norm_no_affine(program, x, width, eps)?;
    let scaled = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (weight, "d->sd")],
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(scaled, "sd->sd"), (bias, "d->sd")],
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

/// `y[t, o] = sum_i x[t, i] * weight[o, i] + bias[o]`: a checkpoint's
/// row-major `[out, in]` weight read as-is, with `t` the activation's rows.
/// One `Multiply` absorbed into one `Add`-reduce, so a packed weight keeps the
/// backend's row-blocked/tiled matmul path (`weight` is bound as a packed
/// codec by the caller) and a float weight takes the cooperative reduce.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn linear(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(x, "ti->tio"), (weight, "oi->tio")],
    )?;
    let projected = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tio->tio",
        "to->tio",
    )?;
    match bias {
        Some(bias) => elementwise_like(
            program,
            projected,
            ScalarOp::Add,
            &[(projected, "to->to"), (bias, "o->to")],
        ),
        None => Ok(projected),
    }
}

/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, the tanh
/// approximation of GELU. `full` is `x`'s own identity map.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn gelu_tanh(program: &mut Vec<Op>, x: NodeId, full: &str) -> Result<NodeId, TensorError> {
    let squared = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(x, full), (x, full)],
    )?;
    let cubic = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(squared, full), (x, full)],
    )?;
    let coefficient = scalar_constant_like(program, 0.044_715, x)?;
    let scaled_cubic = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(cubic, full), (coefficient, broadcast(full).as_str())],
    )?;
    let inner = elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(x, full), (scaled_cubic, full)],
    )?;
    let root = scalar_constant_like(program, 0.797_884_6, x)?;
    let argument = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(inner, full), (root, broadcast(full).as_str())],
    )?;
    let turned = elementwise_like(program, x, ScalarOp::Tanh, &[(argument, full)])?;
    let one = scalar_constant_like(program, 1.0, x)?;
    let gate = elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(turned, full), (one, broadcast(full).as_str())],
    )?;
    let half = scalar_constant_like(program, 0.5, x)?;
    let scaled = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(x, full), (half, broadcast(full).as_str())],
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(scaled, full), (gate, full)],
    )
}

/// The broadcast counterpart of an identity map: the operand has one axis and
/// the iteration space all of `full`'s. `"sd->sd"` gives `"->sd"`.
fn broadcast(full: &str) -> alloc::string::String {
    match full.split_once("->") {
        Some((_, space)) => format!("->{space}"),
        None => alloc::string::String::new(),
    }
}

/// Channel-group normalisation over `[rows, g, cg]` with `g * cg == channels`: per
/// group, the mean and variance of `rows * cg` values (the spatial axes are
/// already flattened into `rows`), then
/// `(x - mean) / sqrt(var + eps) * weight[g, cg] + bias[g, cg]`. `count` is
/// `rows * cg`; `weight`/`bias` are one `[g, cg]` vector each.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn group_norm(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    bias: NodeId,
    count: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let sum = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        x,
        "rgc->rgc",
        "g->rgc",
    )?;
    let inverse_count = scalar_constant_like(program, 1.0 / count as f32, x)?;
    let mean = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(sum, "g->g"), (inverse_count, "->g")],
    )?;
    let centered = elementwise_like(
        program,
        x,
        ScalarOp::Subtract,
        &[(x, "rgc->rgc"), (mean, "g->rgc")],
    )?;
    let squared = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "rgc->rgc"), (centered, "rgc->rgc")],
    )?;
    let variance_sum = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "rgc->rgc",
        "g->rgc",
    )?;
    let variance = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(variance_sum, "g->g"), (inverse_count, "->g")],
    )?;
    let eps_node = scalar_constant_like(program, eps, x)?;
    let spread = elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(variance, "g->g"), (eps_node, "->g")],
    )?;
    let deviation = elementwise_like(program, x, ScalarOp::SquareRoot, &[(spread, "g->g")])?;
    let inverse = elementwise_like(program, x, ScalarOp::Reciprocal, &[(deviation, "g->g")])?;
    let normalised = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "rgc->rgc"), (inverse, "g->rgc")],
    )?;
    let scaled = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(normalised, "rgc->rgc"), (weight, "gc->rgc")],
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(scaled, "rgc->rgc"), (bias, "gc->rgc")],
    )
}

/// Gathers `source`'s axis `gathered_dim` through `indices` (a rank-1 integer
/// leaf), keeping every other axis addressed straight through. The gathered
/// axis's extent comes from `indices`' own declared length, which is what
/// makes an arbitrary (non-affine) row permutation -- a pad, an upsample, a
/// concat half -- expressible in the affine algebra at all.
#[must_use]
pub fn gather_axis(
    program: &mut Vec<Op>,
    source: NodeId,
    indices: NodeId,
    rank: u16,
    gathered_dim: u16,
    dtype: DType,
) -> NodeId {
    let axes = (0..rank)
        .map(|axis| {
            if axis == gathered_dim {
                return AxisIndex::default();
            }
            AxisIndex {
                terms: core::iter::once(AxisTerm::projection(axis)).collect(),
                offset: 0,
                len: None,
            }
        })
        .collect();
    let base = IndexPattern {
        iter_rank: rank,
        axes,
    };
    let index_map = map::projection(rank, &[gathered_dim]);
    super::gather_computed(program, source, indices, index_map, base, gathered_dim, dtype)
}

/// [`gather_axis`] on a node's leading axis, keeping `trailing_rank` further
/// axes addressed in place.
#[must_use]
pub fn gather_leading_axis(
    program: &mut Vec<Op>,
    source: NodeId,
    indices: NodeId,
    trailing_rank: u16,
    dtype: DType,
) -> NodeId {
    gather_axis(program, source, indices, trailing_rank + 1, 0, dtype)
}

/// Nearest-neighbour 2x upsample of `[rows, g, cg]` to `[4 * rows, g, cg]`,
/// through one leading-axis gather: `index[4 * h * width + 2 * w + y]` names source
/// row `h * width + w` ([`upsample_x2_index`]), so the flat token axis stays a
/// single axis a later [`conv3x3_same`] can offset.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn nearest_upsample_x2(
    program: &mut Vec<Op>,
    x: NodeId,
    index: NodeId,
) -> Result<NodeId, TensorError> {
    let dtype = produced_dtype_or_binary32(program, x);
    Ok(gather_leading_axis(program, x, index, 2, dtype))
}

/// [`conv3x3_same`] with the nine taps contracted in ONE reduce, reading one
/// `[go, cgo, 3, 3, gi, cgi]` weight and writing ONE output tensor.
///
/// The nine-tap form is nine reduces plus eight adds, which the planner fuses
/// into a single kernel whose epilogue carries the eight sibling products --
/// measured at ~3.4 GB/s while re-reading eight ~1 GB operands per output
/// element. This form gives the planner nothing to fuse: one reduce, no
/// epilogue.
///
/// The whole stencil becomes the leading axis of one gathered operand
/// ([`conv3x3_columns_index`]: nine blocks, block `3 * y + x` the padded grid
/// shifted to that stencil position, `conv3x3_block_rows` rows each). The reduce
/// then keeps `(r, g, c)` and contracts `(t, i, j)` -- the stencil block and the
/// input channel pair -- addressing that operand as
/// `block_rows * t + r @ block_rows`. That two-term address is the split
/// ("reshape") form the index grammar recognises; a three-term
/// `block * x + r + width * y` is expressible and evaluates correctly on the
/// CPU, but its block axis is unsized by the emitter and the Metal dispatch
/// never completes, so the tap row is folded into the block index instead of
/// carried as a coefficient. Each block carries `2 * (width + 2)` trailing slack
/// rows, so its shifted read stays inside itself.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_same_three(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    pad_index: NodeId,
    pad_mask: NodeId,
    crop_index: NodeId,
    taps_index: NodeId,
) -> Result<NodeId, TensorError> {
    let dtype = produced_dtype_or_binary32(program, x);
    let gathered = gather_leading_axis(program, x, pad_index, 2, dtype);
    let zero = scalar_constant_like(program, 0.0, x)?;
    let padded = elementwise_like(
        program,
        gathered,
        ScalarOp::Select,
        &[
            (pad_mask, "r->rij"),
            (gathered, "rij->rij"),
            (zero, "->rij"),
        ],
    )?;
    // The stencil staged tap-inner: `[9 * rows, in_g, in_c]` with row `r`'s
    // nine taps adjacent, so the contraction `(t, i, j)` is contiguous.
    let staged = gather_leading_axis(program, padded, taps_index, 2, dtype);
    // `9 * r + t @ 9`: the outer coefficient equals `t`'s declared length, the
    // split rule's own reshape form, so `r` and `t` separate again here.
    let image_map = alloc::format!("9*r+t@9,i,j->rgctij");
    let product = elementwise_like(
        program,
        staged,
        ScalarOp::Multiply,
        &[
            (staged, image_map.as_str()),
            (weight, "g,c,t,i,j->rgctij"),
        ],
    )?;
    let sum = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "rgctij->rgctij",
        "rgc->rgctij",
    )?;
    Ok(gather_leading_axis(program, sum, crop_index, 2, dtype))
}

/// One 3x3 stride-1 "same" convolution of `[rows, g, cg]` through nine
/// `linear`-shaped reduces sharing one `[go, cgo, gi, cgi]` weight, plus the
/// gather that crops the padding ring back to `height * width` rows.
///
/// The activation is first gathered onto the `(height + 2) * (width + 2)`-row
/// grid whose interior holds the image ([`conv3x3_pad_index`]); each tap is
/// then one reduce whose image operand is a flat `tap_row * (width + 2) +
/// tap_column` ahead of the output row. The token axis stays a single axis, so
/// the nine taps keep the contiguous token/feature split the tiled matmul path
/// admits. The `ones` operand only pins the padded row count and is removed
/// when the reduce's body is composed.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_same(
    program: &mut Vec<Op>,
    x: NodeId,
    weights: &[NodeId; 9],
    pad_index: NodeId,
    pad_mask: NodeId,
    crop_index: NodeId,
    height: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let dtype = produced_dtype_or_binary32(program, x);
    let gathered = gather_leading_axis(program, x, pad_index, 2, dtype);
    let zero = scalar_constant_like(program, 0.0, x)?;
    let padded = elementwise_like(
        program,
        gathered,
        ScalarOp::Select,
        &[
            (pad_mask, "r->rij"),
            (gathered, "rij->rij"),
            (zero, "->rij"),
        ],
    )?;
    let padded_width = width + 2;
    let out_rows = (height + 2) * padded_width;
    let one = scalar_constant_like(program, 1.0, x)?;
    let mut accumulator: Option<NodeId> = None;
    for tap_row in 0..3usize {
        for tap_column in 0..3usize {
            let offset = tap_row as u32 * padded_width + tap_column as u32;
            let image_map = format!("r+{offset}@{out_rows},i,j->rij");
            let shifted = elementwise_like(
                program,
                padded,
                ScalarOp::Multiply,
                &[(padded, image_map.as_str()), (one, "->rij")],
            )?;
            let product = elementwise_like(
                program,
                shifted,
                ScalarOp::Multiply,
                &[
                    (shifted, "r,i,j->rgcij"),
                    (weights[tap_row * 3 + tap_column], "g,c,i,j->rgcij"),
                ],
            )?;
            let tap = reduce_like(
                program,
                ScalarOp::Add,
                ReduceInit::Zero,
                product,
                "rgcij->rgcij",
                "rgc->rgcij",
            )?;
            accumulator = Some(match accumulator {
                None => tap,
                Some(partial) => elementwise_like(
                    program,
                    tap,
                    ScalarOp::Add,
                    &[(partial, "rgc->rgc"), (tap, "rgc->rgc")],
                )?,
            });
        }
    }
    let sum = accumulator.ok_or(TensorError::NotLowerable {
        node: x,
        reason: "conv3x3_same needs at least one tap",
    })?;
    Ok(gather_leading_axis(program, sum, crop_index, 2, dtype))
}

/// `y[o] = sum_i x[i] * weight[o, i] + bias[o]`: [`linear`]'s rank-1 form, for
/// a checkpoint's modulation/timestep linears whose activation is one vector.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn matvec(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(x, "i->io"), (weight, "oi->io")],
    )?;
    let projected = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "io->io",
        "o->io",
    )?;
    match bias {
        Some(bias) => elementwise_like(
            program,
            projected,
            ScalarOp::Add,
            &[(projected, "o->o"), (bias, "o->o")],
        ),
        None => Ok(projected),
    }
}

/// [`linear`] over an input-column window of `weight`: `x`'s feature axis is
/// `input_width` wide and reads the weight's columns `[input_offset,
/// input_offset + input_width)`. A fused projection whose consumers take
/// different columns (a residual concatenation) splits here rather than
/// re-binding a non-contiguous byte range.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn linear_window(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    input_offset: u32,
    input_width: u32,
) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[
            (x, "ti->tio"),
            (
                weight,
                format!("o,{input_offset}+i@{input_width}->tio").as_str(),
            ),
        ],
    )?;
    reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tio->tio",
        "to->tio",
    )
}

/// `[start, start + width)` of a rank-1 `[c]` node, as a rank-1 `[width]` one.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn slice_range(
    program: &mut Vec<Op>,
    x: NodeId,
    start: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    elementwise_like(
        program,
        x,
        ScalarOp::Identity,
        &[(x, format!("c+{start}@{width}->c").as_str())],
    )
}

/// A rank-1 constant leaf of extent 1, the shape a per-row scalar operand
/// (`eps`) is declared at by the norm builders.
#[must_use]
pub fn unit_constant(program: &mut Vec<Op>, value: f32, dtype: DType) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype,
            shape: alloc::vec![Extent::Static(1)],
            value,
        },
    )
}

/// Columns `[start, start + width)` of a `[t, c]` node as a `[t, width]` one.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn slice_columns(
    program: &mut Vec<Op>,
    x: NodeId,
    start: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    elementwise_like(
        program,
        x,
        ScalarOp::Identity,
        &[(x, format!("t,c+{start}@{width}->tc").as_str())],
    )
}

/// Rows `[start, start + rows)` of a `[t, c]` node.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn slice_rows(
    program: &mut Vec<Op>,
    x: NodeId,
    start: u32,
    rows: u32,
) -> Result<NodeId, TensorError> {
    elementwise_like(
        program,
        x,
        ScalarOp::Identity,
        &[(x, format!("t+{start}@{rows},c->tc").as_str())],
    )
}

/// [`conv3x3_same`]'s gather index: the `(height + 2) * (width + 2)`-row grid
/// plus the tail the last tap offset reads past it, entry `r` naming image row
/// `(row - 1) * width + column - 1` for the interior and the image's last row
/// everywhere else (the ring's outputs are cropped, so their values are not
/// part of the result and only the address has to stay in bounds).
#[must_use]
pub fn conv3x3_pad_rows(height: u32, width: u32) -> u32 {
    let padded_width = width + 2;
    (height + 2) * padded_width + 2 * padded_width + 2
}

#[must_use]
pub fn conv3x3_pad_index(height: u32, width: u32) -> Vec<i32> {
    let padded_width = width + 2;
    (0..conv3x3_pad_rows(height, width))
        .map(|row| {
            let along = row / padded_width;
            let across = row % padded_width;
            if along == 0 || across == 0 || along > height || across > width {
                return 0;
            }
            ((along - 1) * width + across - 1) as i32
        })
        .collect()
}

/// [`conv3x3_same`]'s zero mask: 1 on the padded grid's interior rows (the
/// image) and 0 on the ring, which is what makes the gather a zero pad.
#[must_use]
pub fn conv3x3_pad_mask(height: u32, width: u32) -> Vec<i32> {
    let padded_width = width + 2;
    (0..conv3x3_pad_rows(height, width))
        .map(|row| {
            let along = row / padded_width;
            let across = row % padded_width;
            let interior = along >= 1 && along <= height && across >= 1 && across <= width;
            i32::from(interior)
        })
        .collect()
}

/// [`conv3x3_same`]'s crop index: the `height * width` output rows that hold
/// the image's own window of the padded grid.
#[must_use]
pub fn conv3x3_crop_index(height: u32, width: u32) -> Vec<i32> {
    let padded_width = width + 2;
    (0..height * width)
        .map(|row| ((row / width) * padded_width + row % width) as i32)
        .collect()
}

/// [`conv3x3_same_three`]'s tap index: the padded grid staged with the tap axis
/// **inside** the row axis, `(row, tap)` flattened row-major, so each output
/// row's nine stencil positions are adjacent in memory and the reduce's
/// contraction `(t, i, j)` is one contiguous run.
///
/// Entry `row * 9 + t` names source row `row + y * (width + 2) + x` for
/// `t = 3 * y + x`. The gather itself produces a rank-1-axis view
/// (`[9 * rows, in_g, in_c]`); the reduce un-flattens it with
/// `9 * r + t @ 9`, which the split rule
/// (`AxisIndex::split_outer_axis`: the outer term's coefficient equals the
/// inner axis's declared `len`) recognises, so `r` comes back as the row axis
/// and `t` as the nine-wide tap axis without a second op.
///
/// The *order* is the whole point: staging the taps row-inner rather than
/// block-major is what makes the contraction contiguous in the activation, and
/// a strided contraction is what keeps a reduce off the K-tiled GEMM and on the
/// per-element kernel (measured at ~3.4 GB/s in this decoder's profile).
#[must_use]
pub fn conv3x3_taps_index(height: u32, width: u32) -> Vec<i32> {
    let padded_width = width + 2;
    let rows = conv3x3_block_rows(height, width);
    (0..9 * rows)
        .map(|index| {
            let row = index / 9;
            let tap = index % 9;
            (row + (tap / 3) * padded_width + tap % 3) as i32
        })
        .collect()
}

/// Rows one [`conv3x3_taps_index`] output row's tap run spans: the padded
/// output rows, one per stencil position (the shift lives in the gather).
#[must_use]
pub fn conv3x3_block_rows(height: u32, width: u32) -> u32 {
    (height + 2) * (width + 2)
}

/// [`nearest_upsample_x2`]'s gather index for a `height` x `width` image:
/// `4 * height * width` rows, row `2 * h * (2 * width) + 2 * w + dy` naming
/// source row `h * width + w`.
#[must_use]
pub fn upsample_x2_index(height: u32, width: u32) -> Vec<i32> {
    let mut index = Vec::with_capacity(4 * (height * width) as usize);
    for row in 0..2 * height {
        for column in 0..2 * width {
            index.push(((row / 2) * width + column / 2) as i32);
        }
    }
    index
}

/// Two `[t, c]` nodes joined along the token axis: `head`'s `head_rows` rows
/// followed by `tail`'s `tail_rows`. `head_index`/`tail_index` are host-built
/// leaves of the joined length whose entries clamp into each half (the half a
/// `Select` on `head_mask` discards never reaches the result), so each half is
/// one leading-axis gather and the join is one `Select`.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn concat_rows(
    program: &mut Vec<Op>,
    head: NodeId,
    tail: NodeId,
    head_index: NodeId,
    tail_index: NodeId,
    head_mask: NodeId,
    trailing: &str,
) -> Result<NodeId, TensorError> {
    let dtype = produced_dtype_or_binary32(program, head);
    let trailing_rank = trailing.len() as u16;
    let gathered_head = gather_leading_axis(program, head, head_index, trailing_rank, dtype);
    let gathered_tail = gather_leading_axis(program, tail, tail_index, trailing_rank, dtype);
    elementwise_like(
        program,
        head,
        ScalarOp::Select,
        &[
            (head_mask, format!("t->t{trailing}").as_str()),
            (gathered_head, format!("t{trailing}->t{trailing}").as_str()),
            (gathered_tail, format!("t{trailing}->t{trailing}").as_str()),
        ],
    )
}

/// [`non_causal_attention`] for a query/key pair already split into the two
/// halves [`super::fused_rope_pair`] returns: the score is the sum of the two
/// halves' dots (`q_even . k_even + q_odd . k_odd`), which is the same dot
/// product the interleaved split computes, without re-interleaving the halves
/// into one node. `q_even`/`q_odd` are `[s, u, p]`, `k_even`/`k_odd`
/// `[t, u, p]`, `v` is `[t, u, d]`.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn non_causal_attention_halves(
    program: &mut Vec<Op>,
    q_even: NodeId,
    q_odd: NodeId,
    k_even: NodeId,
    k_odd: NodeId,
    v: NodeId,
    inv_sqrt_head_dim: NodeId,
) -> Result<NodeId, TensorError> {
    let grouped_dtype = produced_dtype_or_binary32(program, q_even);
    let single_group = op::append(
        program,
        Op::Constant {
            dtype: grouped_dtype,
            shape: alloc::vec![Extent::Static(1)],
            value: 1.0,
        },
    );
    let even_grouped = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[
            (q_even, "sup->sugp"),
            (single_group, "g->sugp"),
        ],
    )?;
    let odd_grouped = elementwise_like(
        program,
        q_odd,
        ScalarOp::Multiply,
        &[
            (q_odd, "sup->sugp"),
            (single_group, "g->sugp"),
        ],
    )?;
    let even_product = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[(even_grouped, "sugp->stugp"), (k_even, "tup->stugp")],
    )?;
    let odd_product = elementwise_like(
        program,
        q_odd,
        ScalarOp::Multiply,
        &[(odd_grouped, "sugp->stugp"), (k_odd, "tup->stugp")],
    )?;
    let even_scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        even_product,
        "stugp->stugp",
        "stug->stugp",
    )?;
    let odd_scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        odd_product,
        "stugp->stugp",
        "stug->stugp",
    )?;
    let scores = elementwise_like(
        program,
        q_even,
        ScalarOp::Add,
        &[(even_scores, "stug->stug"), (odd_scores, "stug->stug")],
    )?;
    append_masked_softmax_mix(program, scores, inv_sqrt_head_dim, None, None, 1, v)
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

/// One head-split projection of the encoder memory: `memory` is `[t, m]`,
/// `weight` is `[m, u, o]`, the result is `[t, u, o]`; an optional `bias`
/// (`[u, o]`) is added per position. A multiply and a `Reduce` over `m`, so
/// autograd differentiates it with the rules it already has.
fn project_memory(
    program: &mut Vec<Op>,
    memory: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        memory,
        ScalarOp::Multiply,
        &[(memory, "tm->tmuo"), (weight, "muo->tmuo")],
    )?;
    let projected = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tmuo->tmuo",
        "tuo->tmuo",
    )?;
    match bias {
        Some(bias) => elementwise_like(
            program,
            memory,
            ScalarOp::Add,
            &[(projected, "tuo->tuo"), (bias, "uo->tuo")],
        ),
        None => Ok(projected),
    }
}

/// The once-per-utterance cross-attention K and V: both projected from the
/// encoder `memory` (`[t, m]`) with `k_weight` (`[m, u, i]`) and `v_weight`
/// (`[m, u, d]`), giving `k` as `[t, u, i]` and `v` as `[t, u, d]`. A bias is an
/// operand or it is absent (`None`); a model whose K projection has no bias
/// passes `None`, nothing branches on a name.
///
/// Feed the pair to [`non_causal_attention`] (`s != t`). Inference binds the
/// cached `k`/`v` tensors as `Input`s in place of this builder's output;
/// training calls this builder in the same graph, so the adjoint flows through
/// the projection to the memory and the weights. The attention ops are the same
/// either way: there is no inference-only op.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn cross_kv(
    program: &mut Vec<Op>,
    memory: NodeId,
    k_weight: NodeId,
    k_bias: Option<NodeId>,
    v_weight: NodeId,
    v_bias: Option<NodeId>,
) -> Result<(NodeId, NodeId), TensorError> {
    let k = project_memory(program, memory, k_weight, k_bias)?;
    let v = project_memory(program, memory, v_weight, v_bias)?;
    Ok((k, v))
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

/// Window applied to each frame before the DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum WindowKind {
    /// Hann, periodic form: `0.5 * (1 - cos(2*pi*i / n_fft))`.
    #[default]
    HannPeriodic,
    /// `0.5 * (1 - cos(2*pi*i / (n_fft - 1)))`.
    HannSymmetric,
    /// All ones: the bare DFT of each frame.
    Rectangular,
}

/// Every constant of [`stft_log_mel`]; the defaults are one family's reference values (sourced in
/// the spec's reference table, not here), and none is selected by a name.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct StftLogMelConfig {
    /// Window applied to each frame before the DFT.
    pub window: WindowKind,
    /// DFT length and frame length in samples, default 400.
    pub n_fft: u32,
    /// Stride between frames in samples, default 160.
    pub hop: u32,
    /// Lower bound on a mel energy before the log, default 1e-10.
    pub floor: f32,
    /// Decades below the whole-tensor peak at which log values are raised, default 8.
    pub dynamic_range: f32,
    /// Multiplier of the final affine rescale, default 0.25 (`(x + 4) / 4` with the offset).
    pub scale: f32,
    /// Addend of the final affine rescale, default 1.0.
    pub offset: f32,
}

impl Default for StftLogMelConfig {
    fn default() -> Self {
        Self {
            window: WindowKind::HannPeriodic,
            n_fft: 400,
            hop: 160,
            floor: 1e-10,
            dynamic_range: 8.0,
            scale: 0.25,
            offset: 1.0,
        }
    }
}

/// Name of the `[n_fft]` window leaf [`stft_log_mel`] declares.
pub const STFT_WINDOW_INPUT: &str = "stft_window";
/// Name of the `[n_fft, bins]` cosine-table leaf [`stft_log_mel`] declares.
pub const STFT_COS_INPUT: &str = "stft_cos";
/// Name of the `[n_fft, bins]` sine-table leaf [`stft_log_mel`] declares.
pub const STFT_SIN_INPUT: &str = "stft_sin";

/// Host-built leaf bytes for [`stft_log_mel`], each encoded as the leaf's dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StftTables {
    /// `[n_fft]`, bound to [`STFT_WINDOW_INPUT`].
    pub window: Vec<u8>,
    /// `[n_fft, bins]`, bound to [`STFT_COS_INPUT`].
    pub cos: Vec<u8>,
    /// `[n_fft, bins]`, bound to [`STFT_SIN_INPUT`].
    pub sin: Vec<u8>,
}

impl StftLogMelConfig {
    /// Frequency bins kept from the DFT: `n_fft / 2 + 1`.
    #[must_use]
    pub const fn bins(&self) -> u32 {
        self.n_fft / 2 + 1
    }

    fn frame_count(&self, samples: u32) -> Result<u32, TensorError> {
        if self.n_fft < 2 || self.hop == 0 || samples < self.n_fft {
            return Err(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "stft needs n_fft >= 2, a nonzero hop and at least one whole frame",
            });
        }
        Ok((samples - self.n_fft) / self.hop + 1)
    }

    fn window_values(&self) -> Vec<f32> {
        let length = f64::from(self.n_fft);
        (0..self.n_fft)
            .map(|index| {
                let phase = f64::from(index) * core::f64::consts::TAU;
                match self.window {
                    WindowKind::HannPeriodic => 0.5 * (1.0 - libm::cos(phase / length)),
                    WindowKind::HannSymmetric => 0.5 * (1.0 - libm::cos(phase / (length - 1.0))),
                    WindowKind::Rectangular => 1.0,
                }
            })
            .map(|value| value as f32)
            .collect()
    }
}

/// Host tables for [`stft_log_mel`]: the window and the `[n_fft, bins]` cosine and sine matrices
/// (`cos(2*pi*t*k / n_fft)`, `sin(...)`), encoded as `dtype` through [`FormatCodec`] like
/// [`multi_axis_rope_tables`]. The sine table is positive; the power spectrum squares it, so the
/// sign convention of the DFT is immaterial.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `n_fft < 2` or `dtype` cannot be encoded.
pub fn stft_tables(config: &StftLogMelConfig, dtype: DType) -> Result<StftTables, TensorError> {
    if config.n_fft < 2 {
        return Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "stft needs n_fft >= 2",
        });
    }
    let length = f64::from(config.n_fft);
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for tap in 0..config.n_fft {
        for bin in 0..config.bins() {
            let angle = core::f64::consts::TAU * f64::from(tap) * f64::from(bin) / length;
            cos.push(libm::cos(angle) as f32);
            sin.push(libm::sin(angle) as f32);
        }
    }
    Ok(StftTables {
        window: encode_table(&config.window_values(), dtype)?,
        cos: encode_table(&cos, dtype)?,
        sin: encode_table(&sin, dtype)?,
    })
}

/// Log-mel spectrogram of a waveform `signal` (`[samples]`) as `[mels, frames]`, with
/// `frames = (samples - n_fft) / hop + 1`: window each frame, DFT as a matmul against host
/// cos/sin tables, power, projection through `filterbank` (`[mels, bins]`, a bound operand),
/// `Maximum` floor, `log10`, a whole-tensor `Reduce Maximum` then `Maximum` for the dynamic-range
/// clamp, and the affine rescale. It declares the window, cosine and sine leaves
/// ([`STFT_WINDOW_INPUT`], [`STFT_COS_INPUT`], [`STFT_SIN_INPUT`]) in the signal's dtype; bind them
/// from [`stft_tables`]. Framing is an `Elementwise` read through the two-term window map
/// `hop*frame + tap`, which autograd differentiates by mask composition; every other step is
/// `Elementwise` or an Add/Maximum `Reduce`, so the whole front end is trainable to the waveform
/// and to the filterbank. Padding the signal is the caller's job. Compose [`layer_norm`] or a
/// convolution after it; [`non_causal_attention`] shows the same leaf-plus-table pattern.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `n_fft < 2`, `hop == 0` or `samples < n_fft`.
pub fn stft_log_mel(
    program: &mut Vec<Op>,
    config: &StftLogMelConfig,
    signal: NodeId,
    samples: u32,
    filterbank: NodeId,
) -> Result<NodeId, TensorError> {
    let frames = config.frame_count(samples)?;
    let dtype = produced_dtype_or_binary32(program, signal);
    let table = |extents: &[u32]| {
        extents
            .iter()
            .map(|&e| Extent::Static(e))
            .collect::<Vec<_>>()
    };
    let window = input_leaf(program, dtype, table(&[config.n_fft]), STFT_WINDOW_INPUT);
    let shape = table(&[config.n_fft, config.bins()]);
    let cos = input_leaf(program, dtype, shape.clone(), STFT_COS_INPUT);
    let sin = input_leaf(program, dtype, shape, STFT_SIN_INPUT);
    let stamp = op::append(
        program,
        Op::Constant {
            dtype,
            shape: table(&[frames, config.n_fft]),
            value: 1.0,
        },
    );
    let window_map = alloc::format!("{}*f+t->ft", config.hop);
    let framed = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(signal, window_map.as_str()), (stamp, "ft->ft")],
    )?;
    let windowed = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(framed, "ft->ft"), (window, "t->ft")],
    )?;
    let real = dft_half(program, windowed, cos)?;
    let imaginary = dft_half(program, windowed, sin)?;
    let real_sq = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(real, "fk->fk"), (real, "fk->fk")],
    )?;
    let imaginary_sq = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(imaginary, "fk->fk"), (imaginary, "fk->fk")],
    )?;
    let power = elementwise_like(
        program,
        signal,
        ScalarOp::Add,
        &[(real_sq, "fk->fk"), (imaginary_sq, "fk->fk")],
    )?;
    let weighted = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(power, "fk->mfk"), (filterbank, "mk->mfk")],
    )?;
    let mel = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        weighted,
        "mfk->mfk",
        "mf->mfk",
    )?;
    clamp_and_rescale(program, config, mel)
}

fn dft_half(program: &mut Vec<Op>, windowed: NodeId, table: NodeId) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        windowed,
        ScalarOp::Multiply,
        &[(windowed, "ft->ftk"), (table, "tk->ftk")],
    )?;
    reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "ftk->ftk",
        "fk->ftk",
    )
}

fn clamp_and_rescale(
    program: &mut Vec<Op>,
    config: &StftLogMelConfig,
    mel: NodeId,
) -> Result<NodeId, TensorError> {
    let floor = scalar_constant_like(program, config.floor, mel)?;
    let floored = elementwise_like(
        program,
        mel,
        ScalarOp::Maximum,
        &[(mel, "mf->mf"), (floor, "->mf")],
    )?;
    let natural = elementwise_like(program, mel, ScalarOp::Logarithm, &[(floored, "mf->mf")])?;
    let per_decade = scalar_constant_like(program, core::f32::consts::LOG10_E, mel)?;
    let decades = elementwise_like(
        program,
        mel,
        ScalarOp::Multiply,
        &[(natural, "mf->mf"), (per_decade, "->mf")],
    )?;
    let peak = reduce_like(
        program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        decades,
        "mf->mf",
        "->mf",
    )?;
    let range = scalar_constant_like(program, config.dynamic_range, mel)?;
    let lowest = elementwise_like(
        program,
        mel,
        ScalarOp::Subtract,
        &[(peak, "->"), (range, "->")],
    )?;
    let clamped = elementwise_like(
        program,
        mel,
        ScalarOp::Maximum,
        &[(decades, "mf->mf"), (lowest, "->mf")],
    )?;
    let scale = scalar_constant_like(program, config.scale, mel)?;
    let scaled = elementwise_like(
        program,
        mel,
        ScalarOp::Multiply,
        &[(clamped, "mf->mf"), (scale, "->mf")],
    )?;
    let offset = scalar_constant_like(program, config.offset, mel)?;
    elementwise_like(
        program,
        mel,
        ScalarOp::Add,
        &[(scaled, "mf->mf"), (offset, "->mf")],
    )
}

#[cfg(test)]
mod layer_norm_tests {
    use super::*;
    use crate::spec::input_leaf;

    fn reference(x: &[f32], weight: &[f32], bias: &[f32], width: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::new();
        for row in x.chunks(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inverse = 1.0 / (variance + eps).sqrt();
            for (column, value) in row.iter().enumerate() {
                out.push((value - mean) * inverse * weight[column] + bias[column]);
            }
        }
        out
    }

    #[test]
    fn affine_layer_norm_matches_naive_reference_on_a_non_model_shape() {
        let (rows, width) = (3usize, 5usize);
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 7 % 11) as f32) - 4.5)
            .collect();
        let weight = [0.5_f32, -1.0, 2.0, 1.5, 0.25];
        let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.5];
        let mut program = Vec::new();
        let shape = |dims: &[u32]| dims.iter().map(|&d| Extent::Static(d)).collect::<Vec<_>>();
        let x_node = input_leaf(
            &mut program,
            DType::FLOAT32,
            shape(&[rows as u32, width as u32]),
            "x",
        );
        let w_node = input_leaf(
            &mut program,
            DType::FLOAT32,
            shape(&[width as u32]),
            "weight",
        );
        let b_node = input_leaf(&mut program, DType::FLOAT32, shape(&[width as u32]), "bias");
        let out = layer_norm(&mut program, x_node, w_node, b_node, width as u32, 1e-5)
            .expect("affine layer norm builds");
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[("x", &x), ("weight", &weight), ("bias", &bias)],
            &[out],
        )
        .expect("affine layer norm evaluates");
        let got = evaluated.get(out).expect("output requested").0.to_vec();
        let want = reference(&x, &weight, &bias, width, 1e-5);
        assert_eq!(got.len(), rows * width);
        for (index, (actual, expected)) in got.iter().zip(&want).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "element {index}: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(test)]
mod stft_log_mel_tests {
    use super::*;
    use crate::spec::input_leaf;

    const SAMPLES: usize = 64;
    const MELS: usize = 4;

    fn config() -> StftLogMelConfig {
        StftLogMelConfig {
            n_fft: 16,
            hop: 4,
            ..StftLogMelConfig::default()
        }
    }

    fn decode(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn waveform() -> Vec<f32> {
        (0..SAMPLES)
            .map(|index| {
                let time = index as f32;
                3.0 * (0.9 * time).sin() + 1.5 * (2.1 * time).cos() + 0.1 * ((index * 5 % 7) as f32)
            })
            .collect()
    }

    fn filterbank(bins: usize, silent_row: Option<usize>) -> Vec<f32> {
        let mut rows = Vec::new();
        for mel in 0..MELS {
            for bin in 0..bins {
                let center = (mel as f32 + 0.5) * bins as f32 / MELS as f32;
                let weight = (1.0 - (bin as f32 - center).abs() / 3.0).max(0.0);
                rows.push(if silent_row == Some(mel) { 0.0 } else { weight });
            }
        }
        rows
    }

    fn reference(signal: &[f32], bank: &[f32], config: &StftLogMelConfig) -> Vec<f64> {
        let (n_fft, hop, bins) = (
            config.n_fft as usize,
            config.hop as usize,
            config.bins() as usize,
        );
        let window = config.window_values();
        let frames = (signal.len() - n_fft) / hop + 1;
        let mut decades = vec![0f64; MELS * frames];
        for frame in 0..frames {
            let power: Vec<f64> = (0..bins)
                .map(|bin| {
                    let (mut real, mut imaginary) = (0f64, 0f64);
                    for tap in 0..n_fft {
                        let sample = f64::from(signal[frame * hop + tap]) * f64::from(window[tap]);
                        let angle = core::f64::consts::TAU * (tap * bin) as f64 / n_fft as f64;
                        real += sample * angle.cos();
                        imaginary -= sample * angle.sin();
                    }
                    real * real + imaginary * imaginary
                })
                .collect();
            for mel in 0..MELS {
                let energy: f64 = (0..bins)
                    .map(|bin| power[bin] * f64::from(bank[mel * bins + bin]))
                    .sum();
                decades[mel * frames + frame] = energy.max(f64::from(config.floor)).log10();
            }
        }
        let peak = decades.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        decades
            .iter()
            .map(|value| {
                value.max(peak - f64::from(config.dynamic_range)) * f64::from(config.scale)
                    + f64::from(config.offset)
            })
            .collect()
    }

    fn run(config: &StftLogMelConfig, bank: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let bins = config.bins();
        let mut program = Vec::new();
        let signal = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(SAMPLES as u32)],
            "signal",
        );
        let fb = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(MELS as u32), Extent::Static(bins)],
            "filterbank",
        );
        let out = stft_log_mel(&mut program, config, signal, SAMPLES as u32, fb)
            .expect("stft log-mel builds");
        let tables = stft_tables(config, DType::FLOAT32).expect("tables encode");
        let (window, cos, sin) = (
            decode(&tables.window),
            decode(&tables.cos),
            decode(&tables.sin),
        );
        let wave = waveform();
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("signal", &wave),
                ("filterbank", bank),
                (STFT_WINDOW_INPUT, &window),
                (STFT_COS_INPUT, &cos),
                (STFT_SIN_INPUT, &sin),
            ],
            &[out],
        )
        .expect("stft log-mel evaluates");
        (
            evaluated.get(out).expect("output requested").0.to_vec(),
            wave,
        )
    }

    #[test]
    fn stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal() {
        let config = config();
        let bank = filterbank(config.bins() as usize, None);
        let (got, wave) = run(&config, &bank);
        let want = reference(&wave, &bank, &config);
        assert_eq!(got.len(), MELS * 13);
        for (index, (actual, expected)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-4,
                "element {index}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn stft_log_mel_raises_a_bin_far_below_the_peak_to_max_minus_range() {
        let config = config();
        let bank = filterbank(config.bins() as usize, Some(3));
        let (got, wave) = run(&config, &bank);
        let frames = 13;
        let peak_decades = (0..MELS - 1)
            .flat_map(|mel| (0..frames).map(move |frame| (mel, frame)))
            .map(|(mel, frame)| (f64::from(got[mel * frames + frame]) - 1.0) / 0.25)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            peak_decades - 8.0 > -10.0,
            "silent bin must sit below the clamp"
        );
        let want = (peak_decades - 8.0) * 0.25 + 1.0;
        let silent = &got[3 * frames..4 * frames];
        assert!(
            silent
                .iter()
                .all(|value| (f64::from(*value) - want).abs() < 1e-4)
        );
        let reference = reference(&wave, &bank, &config);
        assert!((f64::from(silent[0]) - reference[3 * frames]).abs() < 1e-4);
    }

    #[test]
    fn stft_log_mel_rejects_a_signal_shorter_than_one_frame() {
        let mut program = Vec::new();
        let signal = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(8)],
            "signal",
        );
        let fb = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(1), Extent::Static(9)],
            "fb",
        );
        assert!(stft_log_mel(&mut program, &config(), signal, 8, fb).is_err());
    }
}

/// [`non_causal_attention_halves`] with the feature axes grouped the way the
/// caller declares them and no broadcast group axis. `heads` is the letter run
/// of the per-query feature axes: `"u"` for multi-head attention
/// (`q_even`/`q_odd` are `[s, u, p]`, `k_even`/`k_odd` `[t, u, p]`, `v`
/// `[t, u, d]`), `"gc"` for one attention head over a two-axis channel group
/// (`[s, g, c]`), `""` for a single axis.
///
/// Every step is a two-operand `Multiply` or a one-value `Maximum`/`Add`
/// reduce, which is the shape a backend's dense batched matmul recognises;
/// [`non_causal_attention_halves`] with the feature axes spelled as the caller
/// declares them and no broadcast group axis. Three letter runs describe the
/// operands: `kept` are the per-query axes the score keeps (`"u"` for
/// multi-head attention, `""` for one head over the whole channel), `reduced`
/// are the axes the dot product sums over (`"p"` for the rotated pairs,
/// `"gc"` for a two-axis channel group) and `value` is the per-key value axis
/// (`"d"`, or `""` when the value axes are the kept/channel axes themselves).
///
/// Every step is a two-operand `Multiply` or a one-value `Maximum`/`Add`
/// reduce, which is the shape a backend's dense batched matmul recognises;
/// [`non_causal_attention_halves`]'s group axis needs a broadcast multiply that
/// its body check does not accept.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
#[allow(clippy::too_many_arguments)]
pub fn non_causal_attention_heads(
    program: &mut Vec<Op>,
    q_even: NodeId,
    q_odd: NodeId,
    k_even: NodeId,
    k_odd: NodeId,
    v: NodeId,
    inv_sqrt_head_dim: NodeId,
    kept: &str,
    reduced: &str,
    value: &str,
) -> Result<NodeId, TensorError> {
    let combined = alloc::format!("{kept}{reduced}");
    let comma = |letters: &str| -> alloc::string::String {
        if letters.is_empty() {
            return alloc::string::String::new();
        }
        alloc::format!(
            ",{}",
            letters
                .chars()
                .map(|letter| letter.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    // The score tensor is keyed `[{kept}, s, t]`: the kept (head) axes first and
    // the key axis `t` innermost, which is the layout a backend's batched matmul
    // writes and reads contiguously (one `[s, t]` plane per head, its key axis
    // the fast one). `1/sqrt(head_dim)` is folded into the queries below, so the
    // product that produces the score is one `Multiply` of two operands with no
    // trailing scale node over the score body. With no kept axes the two layouts
    // coincide, which is why the VAE's single-head call is unaffected.
    let score_axes = alloc::format!("{kept}st");
    // the product space: the kept axes, the query axis, the key axis, the
    // reduced axes -- `t` last, so the key axis is the contiguous one.
    let iteration = alloc::format!("{kept}st{reduced}");
    let scores_iteration = alloc::format!("{iteration}->{iteration}");
    let query = alloc::format!("s{}->{iteration}", comma(&combined));
    let key = alloc::format!("t{}->{iteration}", comma(&combined));
    let scores_out = alloc::format!("{score_axes}->{iteration}");
    let same = alloc::format!("{score_axes}->{score_axes}");
    let row = alloc::format!("{kept}s->{score_axes}");
    let broadcast = alloc::format!("{score_axes}->{score_axes}{value}");
    let value_map = alloc::format!("t{}->{score_axes}{value}", comma(&alloc::format!("{kept}{value}")));
    let attended_in = alloc::format!("{score_axes}{value}->{score_axes}{value}");
    let attended_out = alloc::format!("{kept}s{value}->{score_axes}{value}");
    // `[{kept}, s, value]` is the order this reduce produces; the caller reads
    // `[s, {kept}, value]`, so a head-first score needs one permutation back.
    let attended_order = alloc::format!("{kept}s{value}->s{kept}{value}");
    let query_shape = alloc::format!("s{combined}->s{combined}");
    let query_scalar = alloc::format!("->s{combined}");
    let q_even = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[(q_even, query_shape.as_str()), (inv_sqrt_head_dim, query_scalar.as_str())],
    )?;
    let q_odd = elementwise_like(
        program,
        q_odd,
        ScalarOp::Multiply,
        &[(q_odd, query_shape.as_str()), (inv_sqrt_head_dim, query_scalar.as_str())],
    )?;
    let even_product = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[(q_even, query.as_str()), (k_even, key.as_str())],
    )?;
    let odd_product = elementwise_like(
        program,
        q_odd,
        ScalarOp::Multiply,
        &[(q_odd, query.as_str()), (k_odd, key.as_str())],
    )?;
    let even_scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        even_product,
        scores_iteration.as_str(),
        scores_out.as_str(),
    )?;
    let odd_scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        odd_product,
        scores_iteration.as_str(),
        scores_out.as_str(),
    )?;
    let scores = elementwise_like(
        program,
        q_even,
        ScalarOp::Add,
        &[(even_scores, same.as_str()), (odd_scores, same.as_str())],
    )?;
    let peak = reduce_like(
        program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores,
        same.as_str(),
        row.as_str(),
    )?;
    let shifted = elementwise_like(
        program,
        q_even,
        ScalarOp::Subtract,
        &[(scores, same.as_str()), (peak, row.as_str())],
    )?;
    let weights = elementwise_like(
        program,
        q_even,
        ScalarOp::Exponential,
        &[(shifted, same.as_str())],
    )?;
    let total = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        same.as_str(),
        row.as_str(),
    )?;
    let inverse = elementwise_like(
        program,
        q_even,
        ScalarOp::Reciprocal,
        &[(total, alloc::format!("s{kept}->s{kept}").as_str())],
    )?;
    let probabilities = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[(weights, same.as_str()), (inverse, row.as_str())],
    )?;
    let attended_product = elementwise_like(
        program,
        q_even,
        ScalarOp::Multiply,
        &[
            (probabilities, broadcast.as_str()),
            (v, value_map.as_str()),
        ],
    )?;
    let attended = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        attended_in.as_str(),
        attended_out.as_str(),
    )?;
    if kept.is_empty() {
        return Ok(attended);
    }
    elementwise_like(
        program,
        attended,
        ScalarOp::Identity,
        &[(attended, attended_order.as_str())],
    )
}
