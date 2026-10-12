//! A plain reduction to the MIL `reduce_*` operators over the axes the output drops.

use alloc::vec;

use proxima_tensor::{NodeId, Reduce};

use super::message::{
    Immediate, Name, begin_operation, input_immediate, input_name, operation_output,
};
use super::node::kind_of;
use super::views::{Current, Site, Stack, affine, not_emitted, pattern_axes};
use super::{Context, MilError};
use crate::census::reduce_operator;
use crate::wire::Encoder;

pub(super) fn emit(
    context: &Context<'_>,
    node: NodeId,
    reduce: &Reduce,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    let operator =
        reduce_operator(reduce.body, reduce.init).map_err(|reason| MilError::NoMilForm {
            offenders: vec![crate::census::NoMilForm { node, reason }],
        })?;
    let kind = kind_of(node, reduce.dtype)?;
    let site = Site {
        host: node,
        position: 0,
        kind,
    };
    let in_pattern = affine(node, &reduce.in_map)?;
    let read = Current::read(context, &site, reduce.operand, in_pattern)?;
    if read.has_offsets() {
        return Err(not_emitted(node, "reduction over an offset window"));
    }
    let sorted = read.sorted_axes(node)?;
    let aligned = read
        .permute_to(encoder, &site, sorted.as_slice(), site.name("t", 1))?
        .expand_to(encoder, &site, usize::from(in_pattern.iter_rank), true)?;
    let kept = pattern_axes(node, affine(node, &reduce.out_map)?)?;
    let dropped = dropped_axes(node, in_pattern.iter_rank, kept.as_slice())?;
    let ascending = ascending_kept(node, kept.as_slice())?;
    let in_order = ascending.as_slice() == kept.as_slice();
    let result = if in_order {
        Name::node(node.0)
    } else {
        site.name("r", 3)
    };
    let dims = kept_dims(node, &aligned, ascending.as_slice())?;
    let operation = begin_operation(encoder, operator);
    input_name(encoder, "x", &aligned.name);
    input_immediate(encoder, "axes", Immediate::Ints(dropped.as_slice()));
    input_immediate(encoder, "keep_dims", Immediate::Flag(false));
    operation_output(encoder, &result, kind, dims.as_slice());
    encoder.close(operation);
    if !in_order {
        let reduced = Current::from_parts(result, dims, ascending);
        reduced.permute_to(encoder, &site, kept.as_slice(), Name::node(node.0))?;
    }
    Ok(())
}

fn dropped_axes(node: NodeId, rank: u16, kept: &[u16]) -> Result<Stack<i64>, MilError> {
    let mut dropped = Stack::new();
    (0..rank)
        .filter(|axis| !kept.contains(axis))
        .try_for_each(|axis| dropped.push(node, i64::from(axis)))?;
    Ok(dropped)
}

fn ascending_kept(node: NodeId, kept: &[u16]) -> Result<Stack<u16>, MilError> {
    let mut ascending = Stack::from_slice(node, kept)?;
    ascending.sort();
    Ok(ascending)
}

fn kept_dims(node: NodeId, aligned: &Current, ascending: &[u16]) -> Result<Stack<u64>, MilError> {
    let mut dims = Stack::new();
    ascending
        .iter()
        .try_for_each(|axis| dims.push(node, aligned.dims.as_slice()[usize::from(*axis)]))?;
    Ok(dims)
}
