//! The quantized activation as a storage decision of one plan: where every consumer of an f32
//! activation is an int8 gemm (several tokens) or a staged matvec (one token), the plan adds a single `BoundOpKind::BlockQuantize` node beside
//! the program's own ops and points those gemms at it. The node's id is `program.len() + k`, so
//! it never collides with a program node and never leaves the plan.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use proxima_primitives::block_layout::BlockLayout;
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Keep, Layout, NodeId, NumericPolicy, NumericRewrite, Op, ScalarOp, Shapes, admit,
};

use crate::backend::GpuDriver;
use crate::cuda::{int8_gemm_serves, staged_matvec_serves};
use crate::identity::operand_codecs;
use crate::lowering::{Codec, PackedOperands};

const QUANTIZE_GROUP: u64 = 32;

struct Consumer {
    position: usize,
    slot: usize,
}

fn is_derived(bound: &BoundOp, program_len: usize) -> bool {
    bound.node.0 as usize >= program_len && matches!(bound.kind, BoundOpKind::BlockQuantize { .. })
}

fn reads_of(resolved: &[BoundOp], node: NodeId) -> usize {
    resolved
        .iter()
        .flat_map(BoundOp::all_read_sources)
        .filter(|(source, _, _)| *source == node)
        .count()
}

fn activation_is_flat(bound: &BoundOp, layout: &Layout, tokens: u64, depth: u64) -> bool {
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

fn collapsed_extent(shape: &[u64]) -> Option<[u64; 2]> {
    let (tokens, trailing) = shape.split_first()?;
    (!trailing.is_empty()).then(|| [*tokens, trailing.iter().product()])
}

fn gemm_activation(
    bound: &BoundOp,
    program: &[Op],
    shapes: &Shapes,
    packed: &PackedOperands,
    policy: NumericPolicy,
) -> Option<(NodeId, usize)> {
    let BoundOpKind::Reduce { keep: Keep::Reduce, reduce_op: ScalarOp::Add, operands, .. } = &bound.kind else {
        return None;
    };
    let codecs = operand_codecs(bound, packed);
    let slot = match codecs.as_slice() {
        [Some(codec), None] if !codec.is_quantized_activation() => 1,
        [None, Some(codec)] if !codec.is_quantized_activation() => 0,
        _ => return None,
    };
    let (activation, layout, _) = operands.get(slot)?;
    let [tokens, depth] = collapsed_extent(shapes.of(*activation))?;
    let float_node = matches!(program.get(activation.0 as usize)?.dtype(), DType::FLOAT32);
    let held = activation_codec(depth);
    let served = if tokens == 1 {
        staged_matvec_serves(bound, packed, slot, held, policy)
    } else {
        tokens > 1
            && activation_is_flat(bound, layout, tokens, depth)
            && int8_gemm_serves(bound, packed, slot, held, policy)
    };
    let eligible = float_node && depth.is_multiple_of(QUANTIZE_GROUP) && served;
    eligible.then_some((*activation, slot))
}

fn consumers_by_activation(
    resolved: &[BoundOp],
    program: &[Op],
    shapes: &Shapes,
    packed: &PackedOperands,
    policy: NumericPolicy,
) -> BTreeMap<NodeId, Vec<Consumer>> {
    let mut by_activation: BTreeMap<NodeId, Vec<Consumer>> = BTreeMap::new();
    for (position, bound) in resolved.iter().enumerate() {
        if let Some((activation, slot)) = gemm_activation(bound, program, shapes, packed, policy) {
            by_activation.entry(activation).or_default().push(Consumer { position, slot });
        }
    }
    by_activation.retain(|activation, consumers| reads_of(resolved, *activation) == consumers.len());
    by_activation
}

/// The aligned grouped layout where the depth fills whole groups, else the plain 32-element one.
fn activation_codec(depth: u64) -> Codec {
    let grouped = BlockLayout::Q8_1_GROUPED;
    if depth.is_multiple_of(u64::from(grouped.elements)) { Codec::Q8_1Grouped } else { Codec::Q8_1 }
}

fn quantize_op(derived: NodeId, activation: NodeId, extent: [u64; 2], template: &Layout) -> BoundOp {
    let [tokens, depth] = extent;
    let mut layout = template.clone();
    layout.base = 0;
    layout.strides.clear();
    layout.strides.extend([depth as i64, 1]);
    let layout_q8 = activation_codec(depth).block_layout().unwrap_or(BlockLayout::Q8_1);
    let bytes = depth / u64::from(layout_q8.elements) * u64::from(layout_q8.block_bytes);
    BoundOp {
        node: derived,
        dtype: DType::UINT8,
        extents: vec![tokens, bytes],
        kind: BoundOpKind::BlockQuantize {
            operands: vec![(activation, layout, None)],
            layout: Box::new(layout_q8),
        },
    }
}

fn repoint(bound: &mut BoundOp, slot: usize, derived: NodeId) {
    if let BoundOpKind::Reduce { operands, .. } = &mut bound.kind
        && let Some(operand) = operands.get_mut(slot)
    {
        operand.0 = derived;
    }
}

/// Adds one q8_1 quantize node per activation whose every reader is an int8 gemm this backend
/// serves, positioned just before the first such gemm, and records it in `packed` as a `Q8_1`
/// operand. The same inputs always yield the same ids and positions.
pub(crate) fn derive_block_quantize(
    resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    packed: &mut PackedOperands,
    policy: NumericPolicy,
    driver: GpuDriver,
) -> Vec<BoundOp> {
    packed.retain(|node, _| (node.0 as usize) < program.len());
    if !driver.lowers_block_quantize() || admit(policy, NumericRewrite::ActivationQuantization).is_err() {
        return resolved;
    }
    let activations = consumers_by_activation(&resolved, program, shapes, packed, policy);
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
        let first = consumers.iter().map(|consumer| consumer.position).min().unwrap_or(0);
        let template = &resolved[first].operands()[consumers[0].slot].1;
        inserts.entry(first).or_default().push(quantize_op(derived, *activation, [tokens, depth], template));
        for consumer in consumers {
            repoints.entry(consumer.position).or_default().push((consumer.slot, derived));
        }
        packed.insert(derived, activation_codec(depth));
    }
    let mut out = Vec::with_capacity(resolved.len() + ids.len());
    for (position, mut bound) in resolved.into_iter().enumerate() {
        out.extend(inserts.remove(&position).into_iter().flatten());
        for (slot, derived) in repoints.remove(&position).into_iter().flatten() {
            repoint(&mut bound, slot, derived);
        }
        out.push(bound);
    }
    out
}

/// The bound ops without the plan-local quantize nodes, each gemm reading its original f32
/// activation again: the form every program-keyed refresh understands.
pub(crate) fn underive_block_quantize(resolved: &[BoundOp], program_len: usize) -> Vec<BoundOp> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_tensor::{
        Extent, IndexMap, Reduce, ReduceInit, append, bind_with_fusion, correct_packed_matmul_layouts, infer, projection,
    };

    const DEPTH: u32 = 128;
    const FEATURES: u32 = 4;

    struct Fixture {
        program: Vec<Op>,
        packed: PackedOperands,
        resolved: Vec<BoundOp>,
        shapes: Shapes,
    }

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
        let packed = PackedOperands::from([(first, Codec::Pq2_0), (second, Codec::Pq2_0)]);
        let policy = NumericPolicy::llama_relaxed();
        let mut resolved = bind_with_fusion(&program, &shapes, &outputs, policy.reassociation, policy).expect("binds");
        correct_packed_matmul_layouts(&mut resolved, &packed.keys().copied().collect());
        Fixture { program, packed, resolved, shapes }
    }

    fn granted() -> NumericPolicy {
        NumericPolicy::llama_relaxed().with_activation_quantization(true)
    }

    fn derive(fixture: &Fixture, policy: NumericPolicy, driver: GpuDriver) -> (Vec<BoundOp>, PackedOperands) {
        let mut packed = fixture.packed.clone();
        let derived =
            derive_block_quantize(fixture.resolved.clone(), &fixture.program, &fixture.shapes, &mut packed, policy, driver);
        (derived, packed)
    }

    fn derived_ids(resolved: &[BoundOp], program_len: usize) -> Vec<NodeId> {
        resolved.iter().filter(|bound| is_derived(bound, program_len)).map(|bound| bound.node).collect()
    }

    #[test]
    fn bound_ops_are_unchanged_when_activation_quantization_is_off() {
        let fixture = fixture(3, false);
        let (derived, _) = derive(&fixture, NumericPolicy::llama_relaxed(), GpuDriver::Cuda);
        assert_eq!(derived, fixture.resolved);
    }

    #[test]
    fn metal_and_wgpu_never_gain_a_block_quantize() {
        let fixture = fixture(3, false);
        for driver in [GpuDriver::Metal, GpuDriver::Wgpu] {
            let (derived, _) = derive(&fixture, granted(), driver);
            assert_eq!(derived, fixture.resolved);
        }
    }

    #[test]
    fn one_activation_feeding_two_gemms_gets_exactly_one_derived_node_before_the_first() {
        let fixture = fixture(3, false);
        let (derived, packed) = derive(&fixture, granted(), GpuDriver::Cuda);
        let program_len = fixture.program.len();
        let ids = derived_ids(&derived, program_len);
        assert_eq!(ids, vec![NodeId(program_len as u32)]);
        assert_eq!(packed.get(&ids[0]), Some(&Codec::Q8_1Grouped));
        let position = |node: NodeId| derived.iter().position(|bound| bound.node == node).expect("node present");
        let readers: Vec<usize> = (0..derived.len())
            .filter(|index| derived[*index].operands().iter().any(|(node, _, _)| *node == ids[0]) && !is_derived(&derived[*index], program_len))
            .collect();
        assert_eq!(readers.len(), 2);
        assert_eq!(position(ids[0]) + 1, readers[0], "derived node sits just before its first reader");
        assert_eq!(derive(&fixture, granted(), GpuDriver::Cuda).0, derived, "derivation is deterministic");
    }

    #[test]
    fn a_non_eligible_reader_leaves_the_activation_f32() {
        let fixture = fixture(3, true);
        let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
        assert_eq!(derived, fixture.resolved);
    }

    #[test]
    fn a_single_token_plan_gets_a_derived_node_for_the_staged_matvec() {
        let fixture = fixture(1, false);
        let (derived, packed) = derive(&fixture, granted(), GpuDriver::Cuda);
        let ids = derived_ids(&derived, fixture.program.len());
        assert_eq!(ids.len(), 1);
        assert_eq!(packed.get(&ids[0]), Some(&Codec::Q8_1Grouped));
    }

    #[test]
    fn a_single_token_plan_with_a_non_eligible_reader_gets_no_derived_node() {
        let fixture = fixture(1, true);
        let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
        assert_eq!(derived, fixture.resolved);
    }

    #[test]
    fn a_contiguous_multi_axis_activation_gets_a_derived_node_over_its_flat_run() {
        for tokens in [1, 3] {
            let fixture = build_fixture(tokens, false, true);
            let (derived, packed) = derive(&fixture, granted(), GpuDriver::Cuda);
            let ids = derived_ids(&derived, fixture.program.len());
            assert_eq!(ids.len(), 1, "{tokens} tokens");
            assert_eq!(packed.get(&ids[0]), Some(&Codec::Q8_1Grouped));
            let node = derived.iter().find(|bound| bound.node == ids[0]).expect("derived node present");
            let grouped = BlockLayout::Q8_1_GROUPED;
            let bytes = u64::from(DEPTH) / u64::from(grouped.elements) * u64::from(grouped.block_bytes);
            assert_eq!(node.extents, vec![u64::from(tokens), bytes]);
            assert_eq!(underive_block_quantize(&derived, fixture.program.len()), fixture.resolved);
        }
    }

    #[test]
    fn depth_picks_the_aligned_grouped_layout_only_when_it_fills_whole_groups() {
        assert_eq!(activation_codec(4096), Codec::Q8_1Grouped);
        assert_eq!(activation_codec(128), Codec::Q8_1Grouped);
        assert_eq!(activation_codec(96), Codec::Q8_1);
        assert_eq!(activation_codec(32), Codec::Q8_1);
    }

    #[test]
    fn underive_restores_the_original_bound_ops() {
        let fixture = fixture(3, false);
        let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
        assert_ne!(derived, fixture.resolved);
        assert_eq!(underive_block_quantize(&derived, fixture.program.len()), fixture.resolved);
    }
}
