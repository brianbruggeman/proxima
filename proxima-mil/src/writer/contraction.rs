//! A contraction (a sum over a product of two operands) as one `matmul`,
//! classified by the census's own test so the writer emits what the census mapped.

use proxima_tensor::{AxisIndex, IndexMap, IndexPattern, NodeId, Op, Reduce, ReduceInit, ScalarOp};

use super::message::{
    Immediate, Name, begin_operation, input_immediate, input_name, operation_output,
};
use super::node::kind_of;
use super::views::{Current, Site, Stack, affine, not_emitted, pattern_axes};
use super::{Context, MilError};
use crate::census::is_contraction;
use crate::wire::Encoder;

/// A reduce node proven to be a contraction of its two-operand product.
#[derive(Debug, Clone, Copy)]
pub(super) struct Contraction<'a> {
    pub(super) product: NodeId,
    pub(super) reduce: &'a Reduce,
    pub(super) left: (NodeId, &'a IndexPattern),
    pub(super) right: (NodeId, &'a IndexPattern),
}

pub(super) fn detect<'a>(context: &Context<'a>, node: NodeId) -> Option<Contraction<'a>> {
    let Op::Reduce(reduce) = context.program.get(node.0 as usize)? else {
        return None;
    };
    let Op::Elementwise {
        body: ScalarOp::Multiply,
        operands,
        ..
    } = context.program.get(reduce.operand.0 as usize)?
    else {
        return None;
    };
    let [(left_node, left_map), (right_node, right_map)] = operands.as_slice() else {
        return None;
    };
    let (IndexMap::Affine(left), IndexMap::Affine(right)) = (left_map, right_map) else {
        return None;
    };
    let sums = reduce.body == ScalarOp::Add && reduce.init == ReduceInit::Zero;
    (sums && is_contraction(reduce, left_map, right_map)).then_some(Contraction {
        product: reduce.operand,
        reduce,
        left: (*left_node, left),
        right: (*right_node, right),
    })
}

/// The axes of a matmul: shared kept axes, row axes, one column axis, one depth axis.
struct Plan {
    left: Stack<u16>,
    right: Stack<u16>,
    result: Stack<u16>,
}

impl Plan {
    fn of(node: NodeId, left: &[u16], right: &[u16], kept: &[u16]) -> Result<Self, MilError> {
        let shared = |axis: &u16| left.contains(axis) && right.contains(axis);
        let batch = filtered(node, kept, shared)?;
        let rows = filtered(node, left, |axis| {
            kept.contains(axis) && !right.contains(axis)
        })?;
        let columns = filtered(node, right, |axis| {
            kept.contains(axis) && !left.contains(axis)
        })?;
        let depth = filtered(node, left, |axis| !kept.contains(axis))?;
        let ([column], [inner]) = (columns.as_slice(), depth.as_slice()) else {
            return Err(not_emitted(node, "contraction that is not one matmul"));
        };
        let batched_rows = !batch.as_slice().is_empty() && rows.as_slice().len() != 1;
        if rows.as_slice().is_empty() || batched_rows {
            return Err(not_emitted(
                node,
                "contraction with batch axes and several row axes",
            ));
        }
        Ok(Self {
            left: ordered(node, [batch.as_slice(), rows.as_slice(), &[*inner]])?,
            right: ordered(node, [batch.as_slice(), &[*inner, *column], &[]])?,
            result: ordered(node, [batch.as_slice(), rows.as_slice(), &[*column]])?,
        })
    }
}

fn filtered(
    node: NodeId,
    axes: &[u16],
    keep: impl Fn(&u16) -> bool,
) -> Result<Stack<u16>, MilError> {
    let mut kept = Stack::new();
    axes.iter()
        .filter(|axis| keep(axis))
        .try_for_each(|axis| kept.push(node, *axis))?;
    Ok(kept)
}

fn ordered(node: NodeId, parts: [&[u16]; 3]) -> Result<Stack<u16>, MilError> {
    let mut axes = Stack::new();
    parts
        .into_iter()
        .flatten()
        .try_for_each(|axis| axes.push(node, *axis))?;
    Ok(axes)
}

pub(super) fn emit(
    context: &Context<'_>,
    node: NodeId,
    contraction: &Contraction<'_>,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    let kept = pattern_axes(node, affine(node, &contraction.reduce.out_map)?)?;
    let kind = kind_of(node, contraction.reduce.dtype)?;
    let extents = context.shapes.of(contraction.product);
    let sites = [0, 1].map(|position| Site {
        host: node,
        position,
        kind,
    });
    let depth = reduced_axes(node, contraction.reduce, kept.as_slice())?;
    let left = grouped(
        context,
        encoder,
        &sites[0],
        contraction.left,
        depth.as_slice(),
        extents,
    )?;
    let right = grouped(
        context,
        encoder,
        &sites[1],
        contraction.right,
        depth.as_slice(),
        extents,
    )?;
    let plan = Plan::of(
        node,
        left.axes.as_slice(),
        right.axes.as_slice(),
        kept.as_slice(),
    )?;
    let left = left.permute_to(
        encoder,
        &sites[0],
        plan.left.as_slice(),
        sites[0].name("t", 1),
    )?;
    let right = right.permute_to(
        encoder,
        &sites[1],
        plan.right.as_slice(),
        sites[1].name("t", 1),
    )?;
    let in_order = plan.result.as_slice() == kept.as_slice();
    let product = if in_order {
        Name::node(node.0)
    } else {
        sites[0].name("m", 3)
    };
    let dims = result_dims(node, &left, &right)?;
    let operation = begin_operation(encoder, "matmul");
    input_name(encoder, "x", &left.name);
    input_name(encoder, "y", &right.name);
    input_immediate(encoder, "transpose_x", Immediate::Flag(false));
    input_immediate(encoder, "transpose_y", Immediate::Flag(false));
    operation_output(encoder, &product, kind, dims.as_slice());
    encoder.close(operation);
    if !in_order {
        let result = Current::from_parts(product, dims, plan.result);
        result.permute_to(encoder, &sites[0], kept.as_slice(), Name::node(node.0))?;
    }
    Ok(())
}

fn reduced_axes(node: NodeId, reduce: &Reduce, kept: &[u16]) -> Result<Stack<u16>, MilError> {
    let mut depth = Stack::new();
    (0..reduce.in_map.affine().iter_rank)
        .filter(|axis| !kept.contains(axis))
        .try_for_each(|axis| depth.push(node, axis))?;
    Ok(depth)
}

/// Reads an operand with the contraction's depth axes made one axis, outermost first:
/// a source axis that already merges them (`h * 32 + d`) stays, separate adjacent
/// axes are reshaped into one, so the depth is a single `matmul` axis either way.
fn grouped(
    context: &Context<'_>,
    encoder: &mut Encoder<'_>,
    site: &Site,
    operand: (NodeId, &IndexPattern),
    depth: &[u16],
    extents: &[u64],
) -> Result<Current, MilError> {
    let host = site.host;
    let dims = Stack::from_slice(host, context.shapes.of(operand.0))?;
    let mut axes = Stack::new();
    for index in &operand.1.axes {
        axes.push(host, dim_axis(host, index, depth, extents)?)?;
    }
    if axes.as_slice().len() != dims.as_slice().len() {
        return Err(not_emitted(
            host,
            "operand pattern rank differs from its shape",
        ));
    }
    Current::from_parts(Name::node(operand.0.0), dims, axes).merge_run(encoder, site, depth)
}

fn dim_axis(
    host: NodeId,
    index: &AxisIndex,
    depth: &[u16],
    extents: &[u64],
) -> Result<u16, MilError> {
    if index.offset != 0 || index.len.is_some() {
        return Err(not_emitted(
            host,
            "contraction operand over an offset window",
        ));
    }
    match index.terms.as_slice() {
        [term] if term.coeff == 1 => Ok(term.axis),
        [_, _, ..] => merged_dim(host, index, depth, extents),
        _ => Err(not_emitted(host, "strided contraction operand axis")),
    }
}

fn merged_dim(
    host: NodeId,
    index: &AxisIndex,
    depth: &[u16],
    extents: &[u64],
) -> Result<u16, MilError> {
    let mut terms = Stack::<(i32, u16)>::new();
    index
        .terms
        .iter()
        .try_for_each(|term| terms.push(host, (term.coeff, term.axis)))?;
    terms.sort_by_key(|(coeff, _)| core::cmp::Reverse(*coeff));
    let in_depth = terms
        .as_slice()
        .iter()
        .map(|(_, axis)| *axis)
        .eq(depth.iter().copied());
    let nested = terms.as_slice().windows(2).all(|pair| {
        extents
            .get(usize::from(pair[1].1))
            .and_then(|extent| i64::try_from(*extent).ok())
            .is_some_and(|extent| extent * i64::from(pair[1].0) == i64::from(pair[0].0))
    });
    match (in_depth, nested) {
        (true, true) => Ok(depth[0]),
        _ => Err(not_emitted(
            host,
            "merged contraction axis that is not the depth in row-major order",
        )),
    }
}

fn result_dims(node: NodeId, left: &Current, right: &Current) -> Result<Stack<u64>, MilError> {
    let left_dims = left.dims.as_slice();
    let mut dims = Stack::from_slice(node, &left_dims[..left_dims.len() - 1])?;
    dims.push(node, *right.dims.as_slice().last().unwrap_or(&1))?;
    Ok(dims)
}
