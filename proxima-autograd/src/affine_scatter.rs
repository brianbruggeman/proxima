//! The adjoint of a read through an affine operand map with a multi-term or
//! non-unit-coefficient axis: the scatter-add of the upstream gradient along
//! the same map.
//!
//! An operand `x` read as `x[m(i)]`, `m(i) = A i + c`, contributes
//! `grad_x[j] = sum over every iteration point i with m(i) = j of g[i]`. A
//! pure projection (one unit-coefficient term per axis) writes each `g[i]`
//! straight to `i`'s own coordinates, which `crate::adjoint`'s existing
//! `Reduce(Add)` with the operand map as its `out_map` already expresses; that
//! path is untouched. `proxima-tensor/src/shape.rs:446` refuses any other
//! `out_map`, so a general map takes the mask-composition idiom
//! `crate::conv` (`window_mask`) and `proxima-onnx`'s
//! `scatter_mask_axis` already use: every irregular operand axis `a` of extent
//! `D` gets a fresh iteration axis `p` over `0..D`, a `0.0/1.0` mask
//! `Equal(Iota(D)[p], m_a(i))` selects the points that land on coordinate `p`,
//! the gradient is multiplied by each mask, and one `Reduce(Add)` writes the
//! result through a pure projection whose irregular axes are the new `p`
//! axes. The same expression is exact for injective maps (a strided
//! interleave, a head-group merge) and for colliding ones (a window).
//!
//! Cost: the widened iteration space is `prod(iteration extents) * prod(D)`
//! over the irregular axes. A dense mask is what the algebra expresses
//! without a scatter lowering; the `crate::conv` module accepts the same
//! trade.

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap, IndexPattern};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

use crate::expr;

fn is_regular(axis: &AxisIndex) -> bool {
    matches!(axis.terms.as_slice(), [term] if term.coeff == 1)
}

fn static_extent(extent: u64) -> Option<Extent> {
    u32::try_from(extent).ok().map(Extent::Static)
}

fn axes_below(count: usize) -> Option<Vec<u16>> {
    (0..count).map(|axis| u16::try_from(axis).ok()).collect()
}

/// The terms of `axis` with repeated iteration axes merged and zero
/// coefficients dropped, in order of first appearance.
fn merged_terms(axis: &AxisIndex) -> Vec<(u16, i64)> {
    let mut merged: Vec<(u16, i64)> = Vec::new();
    for term in &axis.terms {
        match merged.iter_mut().find(|(seen, _)| *seen == term.axis) {
            Some((_, coeff)) => *coeff += i64::from(term.coeff),
            None => merged.push((term.axis, i64::from(term.coeff))),
        }
    }
    merged.retain(|(_, coeff)| *coeff != 0);
    merged
}

/// `true` when every value `axis` takes over the iteration box lies in
/// `0..operand_extent`.
fn axis_in_range(axis: &AxisIndex, iteration_extents: &[u64], operand_extent: u64) -> Option<bool> {
    let (mut low, mut high) = (i64::from(axis.offset), i64::from(axis.offset));
    for (iteration_axis, coeff) in merged_terms(axis) {
        let last = i64::try_from(*iteration_extents.get(usize::from(iteration_axis))?).ok()? - 1;
        let reach = coeff.checked_mul(last)?;
        low += reach.min(0);
        high += reach.max(0);
    }
    Some(low >= 0 && high < i64::try_from(operand_extent).ok()?)
}

fn iota(program: &mut Vec<Op>, extent: u64) -> Option<NodeId> {
    Some(op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: static_extent(extent)?,
        },
    ))
}

/// `value[j0, .., jt-1] = sum(coeff * j) + offset` over the axes of `terms`,
/// built rank by rank so every node's iteration space is covered by its own
/// operands. `None` for a constant axis (`terms` empty): the value is then the
/// rank-0 `offset` constant.
fn axis_value(
    program: &mut Vec<Op>,
    terms: &[(u16, i64)],
    offset: i32,
    iteration_extents: &[u64],
) -> Option<NodeId> {
    let mut value: Option<NodeId> = None;
    for (position, (iteration_axis, coeff)) in terms.iter().enumerate() {
        let positions = iota(
            program,
            *iteration_extents.get(usize::from(*iteration_axis))?,
        )?;
        let term = match coeff {
            1 => positions,
            _ => {
                let scale =
                    expr::constant(program, DType::Float32, i32::try_from(*coeff).ok()? as f32);
                expr::binary(
                    program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    (positions, expr::identity(1)),
                    (scale, expr::broadcast(1)),
                )
            }
        };
        let rank = u16::try_from(position + 1).ok()?;
        value = Some(match value {
            None => term,
            Some(prior) => expr::binary(
                program,
                DType::Float32,
                ScalarOp::Add,
                (
                    prior,
                    IndexMap::Affine(map::projection(rank, &axes_below(position)?)),
                ),
                (term, IndexMap::Affine(map::projection(rank, &[rank - 1]))),
            ),
        });
    }
    if offset == 0 && value.is_some() {
        return value;
    }
    let shift = expr::constant(program, DType::Float32, offset as f32);
    Some(match value {
        None => shift,
        Some(prior) => {
            let rank = u16::try_from(terms.len()).ok()?;
            expr::binary(
                program,
                DType::Float32,
                ScalarOp::Add,
                (prior, expr::identity(rank)),
                (shift, expr::broadcast(rank)),
            )
        }
    })
}

/// `mask[j0, .., jt-1, p] = (p == axis_value(j))` over the iteration axes the
/// operand axis reads and one fresh position axis `p` of `operand_extent`.
fn axis_mask(
    program: &mut Vec<Op>,
    axis: &AxisIndex,
    iteration_extents: &[u64],
    operand_extent: u64,
) -> Option<(NodeId, Vec<u16>)> {
    let terms = merged_terms(axis);
    let value = axis_value(program, &terms, axis.offset, iteration_extents)?;
    let position = iota(program, operand_extent)?;
    let rank = u16::try_from(terms.len() + 1).ok()?;
    let mask = expr::binary(
        program,
        DType::Float32,
        ScalarOp::Equal,
        (
            position,
            IndexMap::Affine(map::projection(rank, &[rank - 1])),
        ),
        (
            value,
            IndexMap::Affine(map::projection(rank, &axes_below(terms.len())?)),
        ),
    );
    let read: Vec<u16> = terms
        .iter()
        .map(|(iteration_axis, _)| *iteration_axis)
        .collect();
    Some((mask, read))
}

/// The operand-shaped gradient of an operand read through `pattern`, given
/// `contribution` at the consumer's own iteration space (`iteration_extents`).
///
/// `None` when the map cannot be lowered: an operand rank that disagrees with
/// `pattern`, a term naming an iteration axis outside `iteration_extents`, an
/// extent beyond `u32`, or an axis whose values leave `0..operand_extent`
/// (the forward read would not have lowered either).
pub(crate) fn scatter_add_along_map(
    program: &mut Vec<Op>,
    dtype: DType,
    contribution: NodeId,
    pattern: &IndexPattern,
    iteration_extents: &[u64],
    operand_extents: &[u64],
) -> Option<NodeId> {
    let rank = usize::from(pattern.iter_rank);
    if pattern.axes.len() != operand_extents.len() || iteration_extents.len() != rank {
        return None;
    }
    let mut irregular: Vec<usize> = Vec::new();
    for (index, axis) in pattern.axes.iter().enumerate() {
        if is_regular(axis) {
            continue;
        }
        if !axis_in_range(axis, iteration_extents, operand_extents[index])? {
            return None;
        }
        irregular.push(index);
    }

    let mut product = contribution;
    for (ordinal, &operand_axis) in irregular.iter().enumerate() {
        let (mask, mut read) = axis_mask(
            program,
            &pattern.axes[operand_axis],
            iteration_extents,
            operand_extents[operand_axis],
        )?;
        let widened = u16::try_from(rank + ordinal + 1).ok()?;
        read.push(widened - 1);
        product = expr::binary(
            program,
            dtype,
            ScalarOp::Multiply,
            (
                product,
                IndexMap::Affine(map::projection(widened, &axes_below(rank + ordinal)?)),
            ),
            (mask, IndexMap::Affine(map::projection(widened, &read))),
        );
    }

    let widened = u16::try_from(rank + irregular.len()).ok()?;
    let out_axes = pattern
        .axes
        .iter()
        .enumerate()
        .map(
            |(index, axis)| match irregular.iter().position(|&seen| seen == index) {
                None => Some(axis.clone()),
                Some(ordinal) => Some(AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(
                        u16::try_from(rank + ordinal).ok()?,
                    ))
                    .collect(),
                    offset: 0,
                    len: None,
                }),
            },
        )
        .collect::<Option<Vec<AxisIndex>>>()?;
    Some(expr::reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        expr::identity(widened),
        IndexMap::Affine(IndexPattern {
            iter_rank: widened,
            axes: out_axes,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn rope_odd_pattern() -> IndexPattern {
        map::affine(
            2,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::scaled(1, 2)], 1),
            ],
        )
    }

    fn contribution(program: &mut Vec<Op>) -> NodeId {
        op::append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: vec![Extent::Static(3), Extent::Static(4)],
                value: 1.0,
            },
        )
    }

    #[test]
    fn rope_odd_map_over_a_wide_enough_operand_lowers() {
        let mut program = Vec::new();
        let node = contribution(&mut program);

        let lowered = scatter_add_along_map(
            &mut program,
            DType::Float32,
            node,
            &rope_odd_pattern(),
            &[3, 4],
            &[3, 8],
        );

        assert!(lowered.is_some());
    }

    #[test]
    fn an_axis_that_reaches_past_the_operand_extent_is_refused() {
        let mut program = Vec::new();
        let node = contribution(&mut program);

        let lowered = scatter_add_along_map(
            &mut program,
            DType::Float32,
            node,
            &rope_odd_pattern(),
            &[3, 4],
            &[3, 7],
        );

        assert!(lowered.is_none(), "2*3+1 = 7 is outside 0..7");
    }

    #[test]
    fn an_operand_rank_that_disagrees_with_the_pattern_is_refused() {
        let mut program = Vec::new();
        let node = contribution(&mut program);

        let lowered = scatter_add_along_map(
            &mut program,
            DType::Float32,
            node,
            &rope_odd_pattern(),
            &[3, 4],
            &[24],
        );

        assert!(lowered.is_none());
    }

    #[test]
    fn a_term_naming_an_axis_outside_the_iteration_space_is_refused() {
        let mut program = Vec::new();
        let node = contribution(&mut program);
        let stray = map::affine(2, &[(&[AxisTerm::scaled(5, 2)], 0)]);

        let lowered =
            scatter_add_along_map(&mut program, DType::Float32, node, &stray, &[3, 4], &[8]);

        assert!(lowered.is_none());
    }
}
