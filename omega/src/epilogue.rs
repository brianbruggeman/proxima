//! The [`ComposedBody`] step-declaration walk every renderer
//! ([`crate::msl`], [`crate::wgsl`], [`crate::cuda`]) does identically: read
//! each step's args (an earlier step or an operand slot), render the step's
//! [`ScalarOp`] through the caller's own expression table, and declare the
//! result under the caller's own `step{n}` naming and syntax.
//!
//! Pulled out once ROW 294 found the same loop duplicated three times over —
//! `crate::msl::push_body_steps`/`push_epilogue_body_steps`,
//! `crate::wgsl::push_body_steps`, `crate::cuda::push_body_steps` all walked
//! `body.steps` and matched [`StepArg::Operand`]/[`StepArg::Step`]
//! identically. What differs per language is never the walk: it is the
//! `ScalarOp` -> expression text table (MSL's ternary vs WGSL's `select` vs
//! CUDA's `fmaxf`/`fminf`) and the declaration syntax (`{type} step{n} = ...;`
//! in MSL/CUDA C vs `let step{n}: {type} = ...;` in WGSL). Both stay
//! per-language, passed in as closures, so this module has no opinion on
//! either.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{ComposedBody, Layout, Lookup, NodeId, ScalarOp, StepArg};
#[cfg(any(feature = "cuda", feature = "metal-wide-cooperative-reduce"))]
use proxima_tensor::{BoundOp, BoundOpKind};

#[cfg(any(feature = "cuda", feature = "metal-wide-cooperative-reduce"))]
use crate::lowering::Codec;

/// Declares one step per [`ComposedBody`] entry, in order, returning the
/// declared name of the body's own result (its last step). `operand_prefix`
/// names the array a [`StepArg::Operand`] indexes into (`"scratch"` for an
/// element body, `"epi_scratch"` for a fused reduce epilogue); `step_prefix`
/// names this walk's own declared values (`"step"`/`"epi_step"`) so an
/// epilogue's steps never collide with the fold's own `step{n}` slots even
/// when a real operand index matches.
///
/// `scalar_op_expr` renders one [`ScalarOp`] over already-named argument
/// strings into the target language's expression syntax.
/// `declare_step` emits the actual `{index}`-th declaration statement given
/// that expression — the one piece of true per-language syntax (the type
/// annotation position, `let` vs bare declaration, the trailing indent).
pub(crate) fn declare_steps(
    source: &mut String,
    body: &ComposedBody,
    operand_prefix: &str,
    step_prefix: &str,
    mut scalar_op_expr: impl FnMut(ScalarOp, &[&str]) -> String,
    mut declare_step: impl FnMut(&mut String, usize, &str),
) -> String {
    for (index, step) in body.steps.iter().enumerate() {
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("{operand_prefix}[{operand_index}]"),
                StepArg::Step(step_index) => format!("{step_prefix}{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        declare_step(source, index, &expr);
    }
    format!("{step_prefix}{}", body.steps.len().saturating_sub(1))
}

/// [`declare_steps`]'s split form for a body rendered inside a per-lane write
/// loop (ROW 370): some steps depend only on operands that do not vary across
/// loop iterations (a folded reduce scalar, a broadcast constant) while
/// others depend on an operand read at the current iteration's coordinate
/// (`x[s,d]`, `gamma[d]`). Re-declaring the whole chain every iteration wastes
/// the invariant half's arithmetic 16x over for a [1,4096]-shaped fold at
/// cooperative width 256. This walk classifies each step by
/// `is_operand_invariant` (transitively through [`StepArg::Step`]) and routes
/// its declaration to `declare_invariant_step` (emitted once, before the
/// loop) or `declare_element_step` (emitted once per iteration, inside the
/// loop) accordingly. The caller still owns loop structure and operand reads;
/// this only decides where each step's declaration text lands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn declare_steps_partitioned(
    invariant_source: &mut String,
    element_source: &mut String,
    body: &ComposedBody,
    operand_prefix: &str,
    step_prefix: &str,
    is_operand_invariant: impl Fn(usize) -> bool,
    mut scalar_op_expr: impl FnMut(ScalarOp, &[&str]) -> String,
    mut declare_invariant_step: impl FnMut(&mut String, usize, &str),
    mut declare_element_step: impl FnMut(&mut String, usize, &str),
) -> String {
    let mut invariant = alloc::vec![false; body.steps.len()];
    for (index, step) in body.steps.iter().enumerate() {
        let step_is_invariant = step.args.iter().all(|arg| match arg {
            StepArg::Operand(operand_index) => is_operand_invariant(usize::from(*operand_index)),
            StepArg::Step(step_index) => invariant[usize::from(*step_index)],
        });
        invariant[index] = step_is_invariant;
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("{operand_prefix}[{operand_index}]"),
                StepArg::Step(step_index) => format!("{step_prefix}{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        if step_is_invariant {
            declare_invariant_step(invariant_source, index, &expr);
        } else {
            declare_element_step(element_source, index, &expr);
        }
    }
    format!("{step_prefix}{}", body.steps.len().saturating_sub(1))
}

/// Largest element count a 32-bit index may address. The cooperative loops stride by the
/// block width, so an extent at the full u32 range would wrap `r += blockDim.x`; 2^30 leaves
/// room for any block width a backend takes.
pub(crate) const INDEX32_REACH_LIMIT: u64 = 1 << 30;

pub(crate) fn index32_reach_fits(reach: u64) -> bool {
    reach <= INDEX32_REACH_LIMIT
}

/// Lanes a cooperating block gives one reduction: a power of two sized so each lane folds
/// about `elements_per_lane` elements serially, held between the backend's `simd_width` and
/// its `cap`.
#[cfg(any(feature = "cuda", feature = "metal-wide-cooperative-reduce"))]
pub(crate) fn reduce_block_width(
    total: u64,
    elements_per_lane: u64,
    simd_width: u64,
    cap: u64,
) -> u64 {
    (total / elements_per_lane)
        .next_power_of_two()
        .clamp(simd_width, cap)
}

/// Whether a reduce with a broadcast epilogue (the decode-shaped norm) takes a whole block
/// for its single row instead of one simd group: a single output row, no packed operands.
#[cfg(any(feature = "cuda", feature = "metal-wide-cooperative-reduce"))]
pub(crate) fn broadcast_reduce_takes_block(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
) -> bool {
    let BoundOpKind::Reduce {
        output_axes,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return false;
    };
    let rows: u64 = output_axes
        .iter()
        .map(|dim| resolved.extents[usize::from(*dim)])
        .product();
    !epilogue_broadcast_axes.is_empty() && quantized.iter().all(Option::is_none) && rows == 1
}

/// True when the epilogue operand's layout never varies along any of `reduce_dims` and it is
/// not gathered, so its value is the same on every pass of the per-lane write loop and can be
/// loaded once before it. A gathered operand's address depends on an index buffer this does
/// not follow.
pub(crate) fn epilogue_operand_is_loop_invariant(
    operand: &(NodeId, Layout, Option<Lookup>),
    reduce_dims: &[u16],
) -> bool {
    let (_, layout, lookup) = operand;
    lookup.is_none() && reduce_dims.iter().all(|&dim| layout.stride(dim) == 0)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn operand(strides: &[i64], lookup: Option<Lookup>) -> (NodeId, Layout, Option<Lookup>) {
        let layout = Layout {
            base: 0,
            strides: strides.to_vec().into(),
        };
        (NodeId(0), layout, lookup)
    }

    #[test]
    fn a_hidden_width_row_takes_the_next_power_of_two_of_a_quarter() {
        assert_eq!(reduce_block_width(1536, 4, 32, 1024), 512);
        assert_eq!(reduce_block_width(4096, 4, 32, 1024), 1024);
    }

    #[test]
    fn a_short_row_is_held_to_one_simd_group() {
        assert_eq!(reduce_block_width(7, 4, 32, 1024), 32);
        assert_eq!(reduce_block_width(0, 4, 32, 1024), 32);
    }

    #[test]
    fn a_long_row_is_held_to_the_cap() {
        assert_eq!(reduce_block_width(1 << 20, 4, 32, 256), 256);
    }

    #[test]
    fn a_stride_zero_operand_is_loop_invariant() {
        assert!(epilogue_operand_is_loop_invariant(&operand(&[0, 0], None), &[1]));
    }

    #[test]
    fn an_operand_striding_along_the_reduce_axis_is_not_invariant() {
        assert!(!epilogue_operand_is_loop_invariant(&operand(&[0, 1], None), &[1]));
    }

    #[test]
    fn a_stride_zero_operand_that_is_gathered_is_not_invariant() {
        let layout = Layout {
            base: 0,
            strides: vec![0_i64, 0].into(),
        };
        let lookup = Lookup {
            indices: NodeId(1),
            index_layout: layout,
            element_stride: 1,
            extent: 4,
        };
        assert!(!epilogue_operand_is_loop_invariant(&operand(&[0, 0], Some(lookup)), &[1]));
    }

    #[test]
    fn index_reach_stops_short_of_the_stride_wrap() {
        assert!(index32_reach_fits(INDEX32_REACH_LIMIT));
        assert!(!index32_reach_fits(INDEX32_REACH_LIMIT + 1));
        assert!(!index32_reach_fits(u64::from(u32::MAX)));
    }
}
