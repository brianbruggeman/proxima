//! The backend-neutral structure of a derived activation quantize: the `BlockQuantize` node a plan
//! adds beside the program's own ops, the readers it repoints, and the undo, and the pass
//! ([`derive_activation_quantize`]) that decides which readers get one from the backend's grant,
//! the policy, the weight's int8 decomposition and the activation's shape. Renderer-only choices
//! (tiles, alignment, cooperative reduce, epilogue broadcast, gather admission) stay with the
//! backend as refusals at emit.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::int8_dot::INT8_DOT_ELEMENTS;
use proxima_numeric::{DType, NumericFormat, NumericPolicy, NumericRewrite, admit, int8_dot_shape};

use super::{ACTIVATION_GROUP_ELEMENTS, ActivationLimits, BoundOp, BoundOpKind, Executes, Form, Layout, StepArg};
use crate::op::{Keep, NodeId, Op, ReduceInit, ScalarOp};
use crate::shape::Shapes;

/// A quantize node the plan added: its id is at or past the program's length.
#[must_use]
pub fn is_derived(bound: &BoundOp, program_len: usize) -> bool {
    bound.node.0 as usize >= program_len && matches!(bound.kind, BoundOpKind::BlockQuantize { .. })
}

/// Whether the reader's activation operand is one dense `[tokens, depth]` run: token axis kept in
/// the output, every other live axis a contiguous reduced run of `depth` elements.
#[must_use]
pub fn activation_is_flat(bound: &BoundOp, layout: &Layout, tokens: u64, depth: u64) -> bool {
    let BoundOpKind::Reduce { output_axes, .. } = &bound.kind else {
        return false;
    };
    let mut live: Vec<(i64, u16)> = (0..bound.extents.len() as u16)
        .filter(|axis| layout.stride(*axis) != 0)
        .map(|axis| (layout.stride(axis), axis))
        .collect();
    live.sort_by_key(|(stride, _)| Reverse(*stride));
    let Some((&(token_stride, token_axis), depth_axes)) = live.split_first() else {
        return false;
    };
    let dense_run = depth_axes.iter().rev().try_fold(1i64, |expected, (stride, axis)| {
        (*stride == expected).then(|| expected * bound.extents[usize::from(*axis)] as i64)
    });
    layout.base == 0
        && dense_run == Some(depth as i64)
        && token_stride == depth as i64
        && bound.extents[usize::from(token_axis)] == tokens
        && output_axes.contains(&token_axis)
        && depth_axes.iter().all(|(_, axis)| !output_axes.contains(axis))
}

/// `[tokens, depth]` of an activation shape: the leading axis and the product of the rest.
#[must_use]
pub fn collapsed_extent(shape: &[u64]) -> Option<[u64; 2]> {
    let (tokens, trailing) = shape.split_first()?;
    (!trailing.is_empty()).then(|| [*tokens, trailing.iter().product()])
}

/// The `BlockQuantize` node `derived` that holds `activation` in the block layout `held`, over the
/// flat `[tokens, depth]` extent, addressed like `template` with a dense row-major source.
#[must_use]
pub fn quantize_op(derived: NodeId, activation: NodeId, extent: [u64; 2], template: &Layout, held: &BlockLayout) -> BoundOp {
    let [tokens, depth] = extent;
    let mut layout = template.clone();
    layout.base = 0;
    layout.strides.clear();
    layout.strides.extend([depth as i64, 1]);
    let bytes = depth / u64::from(held.elements) * u64::from(held.block_bytes);
    BoundOp {
        node: derived,
        dtype: DType::UINT8,
        extents: vec![tokens, bytes],
        kind: BoundOpKind::BlockQuantize {
            operands: vec![(activation, layout, None)],
            layout: Box::new(*held),
        },
    }
}

/// Points operand `slot` of a reduce at `derived`.
pub fn repoint(bound: &mut BoundOp, slot: usize, derived: NodeId) {
    if let BoundOpKind::Reduce { operands, .. } = &mut bound.kind
        && let Some(operand) = operands.get_mut(slot)
    {
        operand.0 = derived;
    }
}

/// The bound ops without the plan-local quantize nodes, each reader reading its original f32
/// activation again: the form every program-keyed refresh understands.
#[must_use]
pub fn underive_block_quantize(resolved: &[BoundOp], program_len: usize) -> Vec<BoundOp> {
    let sources: BTreeMap<NodeId, NodeId> = resolved
        .iter()
        .filter(|bound| is_derived(bound, program_len))
        .filter_map(|bound| Some((bound.node, bound.operands().first()?.0)))
        .collect();
    resolved
        .iter()
        .filter(|bound| !is_derived(bound, program_len))
        .cloned()
        .map(|mut bound| {
            if let BoundOpKind::Reduce { operands, .. } = &mut bound.kind {
                operands.iter_mut().for_each(|operand| {
                    operand.0 = sources.get(&operand.0).copied().unwrap_or(operand.0);
                });
            }
            bound
        })
        .collect()
}

struct Consumer {
    position: usize,
    slot: usize,
}

struct Geometry {
    dims: Vec<u16>,
    extents: Vec<u64>,
    reduction: u64,
    output_total: u64,
}

fn geometry(bound: &BoundOp, output_axes: &[u16]) -> Geometry {
    let dims: Vec<u16> = (0..bound.extents.len() as u16).filter(|dim| !output_axes.contains(dim)).collect();
    let extents: Vec<u64> = dims.iter().map(|dim| bound.extents[usize::from(*dim)]).collect();
    let reduction = extents.iter().product();
    let output_total = output_axes.iter().map(|axis| bound.extents[usize::from(*axis)]).product();
    Geometry { dims, extents, reduction, output_total }
}

/// The int8-dot bounds the plan may derive against: the policy grants activation quantization and
/// the backend's [`Executes`] grants [`Form::Int8DotReduce`] with limits.
fn granted_limits(policy: NumericPolicy, executes: &Executes) -> Option<ActivationLimits> {
    admit(policy, NumericRewrite::ActivationQuantization).ok()?;
    if executes.forms.contains(Form::Int8DotReduce) { executes.activation } else { None }
}

fn is_plain_product(bound: &BoundOp) -> bool {
    let [step] = bound.element_body().steps.as_slice() else {
        return false;
    };
    step.op == ScalarOp::Multiply
        && (step.args == [StepArg::Operand(0), StepArg::Operand(1)] || step.args == [StepArg::Operand(1), StepArg::Operand(0)])
}

/// The operand slot holding the packed weight and its format: exactly one of two operands is in
/// `weights` (ignoring plan-local nodes at or past `program_len`), and it is not itself an
/// already-quantized activation.
fn packed_weight(bound: &BoundOp, program_len: usize, weights: &BTreeMap<NodeId, NumericFormat>) -> Option<(usize, NumericFormat)> {
    let held = |node: &NodeId| weights.get(node).filter(|_| (node.0 as usize) < program_len).copied();
    let [(first, _, _), (second, _, _)] = bound.operands() else {
        return None;
    };
    match (held(first), held(second)) {
        (Some(format), None) if !format.is_quantized_activation() => Some((0, format)),
        (None, Some(format)) if !format.is_quantized_activation() => Some((1, format)),
        _ => None,
    }
}

/// The reduced axes are one unit-stride run read from the buffer's origin.
fn is_contiguous_from_origin(layout: &Layout, geometry: &Geometry) -> bool {
    let mut expected = 1i64;
    for (dim, extent) in geometry.dims.iter().zip(&geometry.extents).rev() {
        if layout.stride(*dim) != expected {
            return false;
        }
        expected *= i64::try_from(*extent).unwrap_or(i64::MAX);
    }
    layout.base == 0
}

/// The activation varies only along axes the weight's gather index also varies on, in whole
/// quantized blocks, so every row of a slot shares one quantized activation row.
fn follows_gather(bound: &BoundOp, varying: &[u16], weight_slot: usize, layout: &Layout, block_elements: i64) -> bool {
    let Some((_, _, Some(lookup))) = bound.operands().get(weight_slot) else {
        return false;
    };
    varying.iter().all(|axis| {
        let stride = layout.stride(*axis);
        lookup.index_layout.stride(*axis) != 0 && stride > 0 && stride % block_elements == 0
    })
}

/// The layout forms the staged matvec reads a quantized activation in: one row shared by every
/// output row, or one per gather slot, over a contiguous reduced run from the buffer origin.
fn staged_layout_reads(
    bound: &BoundOp,
    output_axes: &[u16],
    layout: &Layout,
    weight_slot: usize,
    held: &BlockLayout,
    geometry: &Geometry,
) -> bool {
    let varying: Vec<u16> = output_axes
        .iter()
        .copied()
        .filter(|axis| layout.stride(*axis) != 0 && bound.extents[usize::from(*axis)] != 1)
        .collect();
    let shared = varying.is_empty();
    let readable = shared || follows_gather(bound, &varying, weight_slot, layout, i64::from(held.elements));
    readable && is_contiguous_from_origin(layout, geometry)
}

/// The activation operand a quantize node would feed this reader, as `(node, slot)`, decided from
/// the grant, the weight's int8 decomposition and the activation's dtype and shape, never from a
/// renderer.
fn gemm_activation(
    bound: &BoundOp,
    program: &[Op],
    shapes: &Shapes,
    weights: &BTreeMap<NodeId, NumericFormat>,
    limits: ActivationLimits,
    storage: Option<NumericFormat>,
) -> Option<(NodeId, usize)> {
    let BoundOpKind::Reduce { keep: Keep::Reduce, reduce_op: ScalarOp::Add, init: ReduceInit::Zero, output_axes, operands, .. } =
        &bound.kind
    else {
        return None;
    };
    if bound.dtype != DType::FLOAT32 || !is_plain_product(bound) {
        return None;
    }
    let (weight_slot, weight) = packed_weight(bound, program.len(), weights)?;
    int8_dot_shape(&weight.block_layout()?)?;
    let slot = 1 - weight_slot;
    let (activation, layout, _) = operands.get(slot)?;
    let [tokens, depth] = collapsed_extent(shapes.of(*activation))?;
    let float_node = matches!(program.get(activation.0 as usize)?.dtype(), DType::FLOAT32);
    let held = activation_codec(depth, storage).block_layout()?;
    let geometry = geometry(bound, output_axes);
    let in_device = tokens > 0
        && !geometry.dims.is_empty()
        && geometry.reduction > 0
        && geometry.output_total <= limits.output_total_max
        && geometry.reduction.is_multiple_of(INT8_DOT_ELEMENTS as u64);
    let dense_gemm = tokens > 1 && activation_is_flat(bound, layout, tokens, depth);
    let staged = geometry.reduction <= limits.max_reduction
        && staged_layout_reads(bound, output_axes, layout, weight_slot, &held, &geometry);
    let eligible = float_node && depth.is_multiple_of(group_elements(storage)) && in_device && (dense_gemm || staged);
    eligible.then_some((*activation, slot))
}

fn consumers_by_activation(
    resolved: &[BoundOp],
    program: &[Op],
    shapes: &Shapes,
    weights: &BTreeMap<NodeId, NumericFormat>,
    limits: ActivationLimits,
    storage: Option<NumericFormat>,
) -> BTreeMap<NodeId, Vec<Consumer>> {
    let mut by_activation: BTreeMap<NodeId, Vec<Consumer>> = BTreeMap::new();
    for (position, bound) in resolved.iter().enumerate() {
        if let Some((activation, slot)) = gemm_activation(bound, program, shapes, weights, limits, storage) {
            by_activation.entry(activation).or_default().push(Consumer { position, slot });
        }
    }
    by_activation
}

/// Elements one scale covers in the activation a plan stores: the configured format's block, or the
/// default's group.
fn group_elements(storage: Option<NumericFormat>) -> u64 {
    storage.map_or(ACTIVATION_GROUP_ELEMENTS, |format| u64::from(format.packed_elements()))
}

/// The format an activation of this depth is stored in: the configured one, else the default pair
/// (the aligned grouped Q8_1 layout where the depth fills whole groups, else the plain one).
fn activation_codec(depth: u64, storage: Option<NumericFormat>) -> NumericFormat {
    if let Some(format) = storage {
        return format;
    }
    let grouped = BlockLayout::Q8_1_GROUPED;
    if depth.is_multiple_of(u64::from(grouped.elements)) { NumericFormat::Q8_1Grouped } else { NumericFormat::Q8_1 }
}

/// Adds one quantize node per activation that has an int8-dot reader, positioned just before the
/// first such reader, repoints only those readers, and returns the nodes' held formats beside the
/// new bound ops. Derived ids are `program.len() + k` in activation-node order, so the same inputs
/// always give the same ids and positions.
///
/// A reader is eligible when, with no renderer consulted:
/// - the policy admits [`NumericRewrite::ActivationQuantization`] and `executes` grants
///   [`Form::Int8DotReduce`] with [`ActivationLimits`];
/// - it is an `Add` reduce of one `Multiply`, f32 out, whose one operand is in `weights` as a
///   non-activation format whose block layout has an [`int8_dot_shape`] and whose other operand is
///   unpacked and an f32 program node;
/// - that activation collapses to `[tokens, depth]` with `depth` a multiple of the group (32, or
///   `storage`'s block elements), the reduction a multiple of 32, output total within
///   `output_total_max`;
/// - either many tokens over a dense flat activation, or the staged form: reduction within
///   `max_reduction` and the activation shared by every row or per gather slot, contiguous from
///   its origin.
///
/// `storage` `None` holds Q8_1Grouped when `depth % 128 == 0`, else Q8_1. Teaching pointer: the
/// node is [`quantize_op`], readers are re-aimed with [`repoint`], and [`underive_block_quantize`]
/// undoes it; the decision composes [`Executes`], [`admit`] and [`int8_dot_shape`].
#[must_use]
pub fn derive_activation_quantize(
    resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    weights: &BTreeMap<NodeId, NumericFormat>,
    policy: NumericPolicy,
    executes: &Executes,
    storage: Option<NumericFormat>,
) -> (Vec<BoundOp>, BTreeMap<NodeId, NumericFormat>) {
    let mut held_formats = BTreeMap::new();
    let Some(limits) = granted_limits(policy, executes) else {
        return (resolved, held_formats);
    };
    let activations = consumers_by_activation(&resolved, program, shapes, weights, limits, storage);
    let ids: BTreeMap<NodeId, NodeId> = activations
        .keys()
        .enumerate()
        .map(|(index, activation)| (*activation, NodeId((program.len() + index) as u32)))
        .collect();
    let mut inserts: BTreeMap<usize, Vec<BoundOp>> = BTreeMap::new();
    let mut repoints: BTreeMap<usize, Vec<(usize, NodeId)>> = BTreeMap::new();
    for (activation, consumers) in &activations {
        let derived = ids[activation];
        let Some([tokens, depth]) = collapsed_extent(shapes.of(*activation)) else {
            continue;
        };
        let held_codec = activation_codec(depth, storage);
        let Some(held) = held_codec.block_layout() else {
            continue;
        };
        let first = consumers.iter().map(|consumer| consumer.position).min().unwrap_or(0);
        let template = &resolved[first].operands()[consumers[0].slot].1;
        inserts.entry(first).or_default().push(quantize_op(derived, *activation, [tokens, depth], template, &held));
        for consumer in consumers {
            repoints.entry(consumer.position).or_default().push((consumer.slot, derived));
        }
        held_formats.insert(derived, held_codec);
    }
    let mut out = Vec::with_capacity(resolved.len() + ids.len());
    for (position, mut bound) in resolved.into_iter().enumerate() {
        out.extend(inserts.remove(&position).into_iter().flatten());
        for (slot, derived) in repoints.remove(&position).into_iter().flatten() {
            repoint(&mut bound, slot, derived);
        }
        out.push(bound);
    }
    (out, held_formats)
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;

    use proxima_numeric::FloatLayout;

    use super::*;
    use crate::bind::{bind_with_fusion, correct_packed_matmul_layouts};
    use crate::map::{IndexMap, projection};
    use crate::op::{Extent, Reduce, append};
    use crate::shape::infer;

    const DEPTH: u32 = 128;
    const FEATURES: u32 = 4;
    const STAGED_REDUCTION_LIMIT: u64 = 32_768;

    struct Fixture {
        program: Vec<Op>,
        weights: BTreeMap<NodeId, NumericFormat>,
        resolved: Vec<BoundOp>,
        shapes: Shapes,
    }

    type Derived = (Vec<BoundOp>, BTreeMap<NodeId, NumericFormat>);

    fn input(program: &mut Vec<Op>, dtype: DType, rows: u32) -> NodeId {
        let shape = vec![Extent::Static(rows), Extent::Static(DEPTH)];
        append(program, Op::Input { dtype, shape, name: None })
    }

    fn gemm_over(program: &mut Vec<Op>, weight: NodeId, activation: NodeId, trailing: usize) -> NodeId {
        let loops = (2 + trailing) as u16;
        let reduced: Vec<u16> = (2..loops).collect();
        let weight_axes: Vec<u16> = [&[1u16][..], &reduced].concat();
        let activation_axes: Vec<u16> = [&[0u16][..], &reduced].concat();
        let all: Vec<u16> = (0..loops).collect();
        let product = append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(loops, &weight_axes))),
                    (activation, IndexMap::Affine(projection(loops, &activation_axes))),
                ],
                name: None,
            },
        );
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(loops, &all)),
                out_map: IndexMap::Affine(projection(loops, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        )
    }

    fn wide_input(program: &mut Vec<Op>, dtype: DType, rows: u32) -> NodeId {
        let shape = [rows, 4, 2, 16].map(Extent::Static).to_vec();
        append(program, Op::Input { dtype, shape, name: None })
    }

    fn fixture(tokens: u32, plain_reader: bool) -> Fixture {
        build_fixture(tokens, plain_reader, false)
    }

    fn build_fixture(tokens: u32, plain_reader: bool, wide: bool) -> Fixture {
        let mut program = Vec::new();
        let trailing = if wide { 3 } else { 1 };
        let make = |program: &mut Vec<Op>, dtype, rows| {
            if wide { wide_input(program, dtype, rows) } else { input(program, dtype, rows) }
        };
        let activation = make(&mut program, DType::FLOAT32, tokens);
        let first = make(&mut program, DType::UINT8, FEATURES);
        let second = make(&mut program, DType::UINT8, FEATURES);
        let plain = make(&mut program, DType::FLOAT32, FEATURES);
        let gate = gemm_over(&mut program, first, activation, trailing);
        let up = gemm_over(&mut program, second, activation, trailing);
        let sum = if plain_reader { gemm_over(&mut program, plain, activation, trailing) } else { gate };
        let outputs = [gate, up, sum];
        let shapes = infer(&program, &[]).expect("program infers");
        let weights = BTreeMap::from([(first, NumericFormat::Pq2_0), (second, NumericFormat::Pq2_0)]);
        let policy = NumericPolicy::llama_relaxed();
        let mut resolved = bind_with_fusion(&program, &shapes, &outputs, policy.reassociation, policy).expect("binds");
        let packed_nodes: BTreeSet<NodeId> = weights.keys().copied().collect();
        correct_packed_matmul_layouts(&mut resolved, &packed_nodes);
        Fixture { program, weights, resolved, shapes }
    }

    fn granted() -> NumericPolicy {
        NumericPolicy::llama_relaxed().with_activation_quantization(true)
    }

    // the value omega::cuda::executes builds for a card_asm target (omega/src/cuda.rs:4488-4492),
    // limits from staged_matvec::MAX_STAGED_REDUCTION (omega/src/cuda/staged_matvec.rs:30)
    fn int8_grant() -> Executes {
        Executes::BASE.with(Form::BlockQuantize).with_int8_dot(ActivationLimits {
            max_reduction: STAGED_REDUCTION_LIMIT,
            output_total_max: u64::from(u32::MAX),
        })
    }

    fn derive(fixture: &Fixture, policy: NumericPolicy, executes: &Executes) -> Derived {
        derive_activation_quantize(fixture.resolved.clone(), &fixture.program, &fixture.shapes, &fixture.weights, policy, executes, None)
    }

    fn derived_ids(resolved: &[BoundOp], program_len: usize) -> Vec<NodeId> {
        resolved.iter().filter(|bound| is_derived(bound, program_len)).map(|bound| bound.node).collect()
    }

    #[test]
    fn bound_ops_are_unchanged_when_activation_quantization_is_off() {
        let fixture = fixture(3, false);
        let (derived, held) = derive(&fixture, NumericPolicy::llama_relaxed(), &int8_grant());
        assert_eq!(derived, fixture.resolved);
        assert!(held.is_empty());
    }

    #[test]
    fn a_backend_without_an_activation_grant_derives_nothing_under_a_granting_policy() {
        let fixture = fixture(3, false);
        for executes in [Executes::BASE, Executes::CPU, Executes::BASE.with(Form::BlockQuantize)] {
            let (derived, held) = derive(&fixture, granted(), &executes);
            assert_eq!(derived, fixture.resolved);
            assert!(held.is_empty());
        }
    }

    #[test]
    fn an_f16_weight_is_declined_because_it_has_no_int8_decomposition() {
        assert!(NumericFormat::Scalar(FloatLayout::F16).block_layout().is_none());
        for tokens in [1, 3] {
            let mut fixture = fixture(tokens, false);
            fixture.weights.values_mut().for_each(|format| *format = NumericFormat::Scalar(FloatLayout::F16));
            let (derived, held) = derive(&fixture, granted(), &int8_grant());
            assert_eq!(derived, fixture.resolved, "{tokens} tokens");
            assert!(held.is_empty(), "{tokens} tokens");
        }
    }

    #[test]
    fn one_activation_feeding_two_gemms_gets_exactly_one_derived_node_before_the_first() {
        let fixture = fixture(3, false);
        let (derived, held) = derive(&fixture, granted(), &int8_grant());
        let program_len = fixture.program.len();
        let ids = derived_ids(&derived, program_len);
        assert_eq!(ids, vec![NodeId(program_len as u32)]);
        assert_eq!(held.get(&ids[0]), Some(&NumericFormat::Q8_1Grouped));
        let position = |node: NodeId| derived.iter().position(|bound| bound.node == node).expect("node present");
        let readers: Vec<usize> = (0..derived.len())
            .filter(|index| {
                derived[*index].operands().iter().any(|(node, _, _)| *node == ids[0]) && !is_derived(&derived[*index], program_len)
            })
            .collect();
        assert_eq!(readers.len(), 2);
        assert_eq!(position(ids[0]) + 1, readers[0], "derived node sits just before its first reader");
        assert_eq!(derive(&fixture, granted(), &int8_grant()).0, derived, "derivation is deterministic");
    }

    fn readers_of(resolved: &[BoundOp], node: NodeId) -> Vec<NodeId> {
        resolved
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::Reduce { .. }))
            .filter(|bound| bound.operands().iter().any(|(source, _, _)| *source == node))
            .map(|bound| bound.node)
            .collect()
    }

    fn assert_mixed_readers_share_one_derived_node(tokens: u32) {
        let fixture = fixture(tokens, true);
        let (derived, held) = derive(&fixture, granted(), &int8_grant());
        let program_len = fixture.program.len();
        let ids = derived_ids(&derived, program_len);
        assert_eq!(ids, vec![NodeId(program_len as u32)], "{tokens} tokens");
        assert_eq!(held.get(&ids[0]), Some(&NumericFormat::Q8_1Grouped));
        let activation = NodeId(0);
        let eligible = readers_of(&derived, ids[0]);
        let original = readers_of(&fixture.resolved, activation);
        assert_eq!(original.len(), 3, "{tokens} tokens");
        assert_eq!(eligible, original[..2].to_vec(), "{tokens} tokens");
        let ineligible = readers_of(&derived, activation).into_iter().filter(|node| !eligible.contains(node));
        assert_eq!(ineligible.collect::<Vec<_>>(), original[2..].to_vec(), "{tokens} tokens");
        assert_eq!(underive_block_quantize(&derived, program_len), fixture.resolved);
    }

    #[test]
    fn a_non_eligible_reader_keeps_the_f32_activation_while_eligible_readers_share_one_derived_node() {
        assert_mixed_readers_share_one_derived_node(3);
    }

    #[test]
    fn mixed_readers_are_handled_for_a_single_token_too() {
        assert_mixed_readers_share_one_derived_node(1);
    }

    #[test]
    fn an_activation_with_only_non_eligible_readers_gets_no_derived_node() {
        for tokens in [1, 3] {
            let mut fixture = fixture(tokens, true);
            fixture.weights.clear();
            let (derived, _) = derive(&fixture, granted(), &int8_grant());
            assert_eq!(derived, fixture.resolved, "{tokens} tokens");
        }
    }

    #[test]
    fn the_derived_node_sits_before_the_first_eligible_reader_not_the_first_reader() {
        let mut fixture = fixture(3, true);
        let program_len = fixture.program.len();
        let plain_reader = fixture.resolved.pop().expect("plain reader is last");
        fixture.resolved.insert(0, plain_reader.clone());
        let (derived, _) = derive(&fixture, granted(), &int8_grant());
        let ids = derived_ids(&derived, program_len);
        let position = derived.iter().position(|bound| bound.node == ids[0]).expect("derived present");
        assert_eq!(derived[0], plain_reader, "ineligible reader stays first and unchanged");
        assert_eq!(position, 1);
        assert!(readers_of(&derived, ids[0]).contains(&derived[position + 1].node));
        assert_eq!(derive(&fixture, granted(), &int8_grant()).0, derived, "derivation is deterministic");
    }

    #[test]
    fn naming_the_default_format_derives_the_same_plan_as_naming_none() {
        for tokens in [1, 3] {
            let fixture = fixture(tokens, false);
            let default = derive(&fixture, granted(), &int8_grant());
            let named = derive_activation_quantize(
                fixture.resolved.clone(),
                &fixture.program,
                &fixture.shapes,
                &fixture.weights,
                granted(),
                &int8_grant(),
                Some(NumericFormat::Q8_1Grouped),
            );
            assert_eq!(named, default, "{tokens} tokens");
        }
    }

    #[test]
    fn a_single_token_plan_gets_a_derived_node_for_the_staged_matvec() {
        let fixture = fixture(1, false);
        let (derived, held) = derive(&fixture, granted(), &int8_grant());
        let ids = derived_ids(&derived, fixture.program.len());
        assert_eq!(ids.len(), 1);
        assert_eq!(held.get(&ids[0]), Some(&NumericFormat::Q8_1Grouped));
    }

    #[test]
    fn a_single_token_reduction_past_the_granted_limit_is_declined() {
        let fixture = fixture(1, false);
        let tight = Executes::BASE.with_int8_dot(ActivationLimits { max_reduction: u64::from(DEPTH) - 1, ..ActivationLimits::UNLIMITED });
        assert_eq!(derive(&fixture, granted(), &tight).0, fixture.resolved);
        let exact = Executes::BASE.with_int8_dot(ActivationLimits { max_reduction: u64::from(DEPTH), ..ActivationLimits::UNLIMITED });
        assert_eq!(derived_ids(&derive(&fixture, granted(), &exact).0, fixture.program.len()).len(), 1);
    }

    #[test]
    fn an_output_larger_than_the_granted_total_is_declined() {
        let fixture = fixture(3, false);
        let tight = Executes::BASE.with_int8_dot(ActivationLimits { output_total_max: 3 * u64::from(FEATURES) - 1, ..ActivationLimits::UNLIMITED });
        assert_eq!(derive(&fixture, granted(), &tight).0, fixture.resolved);
    }

    #[test]
    fn a_contiguous_multi_axis_activation_gets_a_derived_node_over_its_flat_run() {
        for tokens in [1, 3] {
            let fixture = build_fixture(tokens, false, true);
            let (derived, held) = derive(&fixture, granted(), &int8_grant());
            let ids = derived_ids(&derived, fixture.program.len());
            assert_eq!(ids.len(), 1, "{tokens} tokens");
            assert_eq!(held.get(&ids[0]), Some(&NumericFormat::Q8_1Grouped));
            let node = derived.iter().find(|bound| bound.node == ids[0]).expect("derived node present");
            let grouped = BlockLayout::Q8_1_GROUPED;
            let bytes = u64::from(DEPTH) / u64::from(grouped.elements) * u64::from(grouped.block_bytes);
            assert_eq!(node.extents, vec![u64::from(tokens), bytes]);
            assert_eq!(underive_block_quantize(&derived, fixture.program.len()), fixture.resolved);
        }
    }

    #[test]
    fn depth_picks_the_aligned_grouped_layout_only_when_it_fills_whole_groups() {
        assert_eq!(activation_codec(4096, None), NumericFormat::Q8_1Grouped);
        assert_eq!(activation_codec(128, None), NumericFormat::Q8_1Grouped);
        assert_eq!(activation_codec(96, None), NumericFormat::Q8_1);
        assert_eq!(activation_codec(32, None), NumericFormat::Q8_1);
    }

    #[test]
    fn underive_restores_the_original_bound_ops() {
        let fixture = fixture(3, false);
        let (derived, _) = derive(&fixture, granted(), &int8_grant());
        assert_ne!(derived, fixture.resolved);
        assert_eq!(underive_block_quantize(&derived, fixture.program.len()), fixture.resolved);
    }
}
