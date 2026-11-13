//! The quantized activation as a storage decision of one plan: where at least one consumer of an
//! f32 activation is an int8 gemm (several tokens) or a staged matvec (one token), the plan adds a
//! single `BoundOpKind::BlockQuantize` node beside the program's own ops and points those eligible
//! readers at it; every other reader keeps the f32 activation. The node's id is
//! `program.len() + k`, so it never collides with a program node and never leaves the plan.

use std::collections::BTreeMap;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::{DType, NumericFormat, NumericPolicy, NumericRewrite, admit};
use proxima_tensor::{
    ACTIVATION_GROUP_ELEMENTS, BoundOp, BoundOpKind, Keep, Missing, NodeId, Op, ScalarOp, Shapes, activation_is_flat,
    collapsed_extent, quantize_op, repoint,
};

use crate::backend::GpuDriver;
use crate::cuda::{consumer_missing, int8_gemm_serves, producer_missing, staged_matvec_serves};
use crate::identity::operand_codecs;
use crate::lowering::{Codec, PackedOperands};

/// Elements per scale the default activation (the Q8_1 pair) needs the depth to fill.
const DEFAULT_GROUP: u64 = ACTIVATION_GROUP_ELEMENTS;

/// A configured activation storage format the plan cannot honour, and the render pieces it lacks.
/// The plan refuses instead of quietly keeping the activation in binary32.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("activation storage {format:?} is not servable: producer {producer:?}, consumer {consumer:?}")]
pub struct ActivationRefusal {
    pub format: NumericFormat,
    /// The CUDA quantize render's missing piece, with the file and line that found it missing.
    pub producer: Option<Missing>,
    /// The activation readers' missing piece (int8 gemm, staged matvec), likewise.
    pub consumer: Option<Missing>,
}

struct Consumer {
    position: usize,
    slot: usize,
}

fn gemm_activation(
    bound: &BoundOp,
    program: &[Op],
    shapes: &Shapes,
    packed: &PackedOperands,
    policy: NumericPolicy,
    storage: Option<Codec>,
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
    let held = activation_codec(depth, storage);
    let served = if tokens == 1 {
        staged_matvec_serves(bound, packed, slot, held, policy)
    } else {
        tokens > 1
            && ((activation_is_flat(bound, layout, tokens, depth) && int8_gemm_serves(bound, packed, slot, held, policy))
                || staged_matvec_serves(bound, packed, slot, held, policy))
    };
    let eligible = float_node && depth.is_multiple_of(group_elements(storage)) && served;
    eligible.then_some((*activation, slot))
}

fn consumers_by_activation(
    resolved: &[BoundOp],
    program: &[Op],
    shapes: &Shapes,
    packed: &PackedOperands,
    policy: NumericPolicy,
    storage: Option<Codec>,
) -> BTreeMap<NodeId, Vec<Consumer>> {
    let mut by_activation: BTreeMap<NodeId, Vec<Consumer>> = BTreeMap::new();
    for (position, bound) in resolved.iter().enumerate() {
        if let Some((activation, slot)) = gemm_activation(bound, program, shapes, packed, policy, storage) {
            by_activation.entry(activation).or_default().push(Consumer { position, slot });
        }
    }
    by_activation
}

/// Elements one scale covers in the activation a plan stores: the configured format's block, or the
/// default's group.
fn group_elements(storage: Option<Codec>) -> u64 {
    storage.map_or(DEFAULT_GROUP, |format| u64::from(format.packed_elements()))
}

/// The format an activation of this depth is stored in: the configured one, else the default pair
/// (the aligned grouped Q8_1 layout where the depth fills whole groups, else the plain one).
fn activation_codec(depth: u64, storage: Option<Codec>) -> Codec {
    if let Some(format) = storage {
        return format;
    }
    let grouped = BlockLayout::Q8_1_GROUPED;
    if depth.is_multiple_of(u64::from(grouped.elements)) { Codec::Q8_1Grouped } else { Codec::Q8_1 }
}

/// Why a configured format cannot be the plan's activation storage, `None` when the producer and
/// the readers can serve it. A non-block format (a scalar) is not a BlockQuantize concern.
fn storage_refusal(format: NumericFormat) -> Option<ActivationRefusal> {
    let Some(layout) = format.block_layout() else {
        return Some(ActivationRefusal { format, producer: None, consumer: None });
    };
    let producer = producer_missing(&layout).err();
    let consumer = consumer_missing(&layout).err();
    (producer.is_some() || consumer.is_some()).then_some(ActivationRefusal { format, producer, consumer })
}

/// Whether the plan can store activations in the configured block `format` (a descriptor's
/// `activation_storage`); `None` format is the default Q8_1 pair, always servable.
///
/// # Errors
/// [`ActivationRefusal`] naming the producer and reader pieces the format lacks, so a configured
/// format is never silently left in binary32. Boxed: the refusal carries a 700-byte format and is
/// built once per plan, never on a hot path.
pub(crate) fn validate_storage(format: Option<NumericFormat>) -> Result<(), Box<ActivationRefusal>> {
    match format.and_then(storage_refusal) {
        Some(refusal) => Err(Box::new(refusal)),
        None => Ok(()),
    }
}

/// Adds one quantize node per activation that has an int8 gemm reader this backend serves,
/// positioned just before the first such gemm, repoints only those readers, and records the node
/// in `packed`. `storage` `None` is the default q8_1 pair; otherwise the caller has passed it
/// through [`validate_storage`]. The same inputs always yield the same ids and positions.
pub(crate) fn derive_with_storage(
    resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    packed: &mut PackedOperands,
    policy: NumericPolicy,
    driver: GpuDriver,
    storage: Option<Codec>,
) -> Vec<BoundOp> {
    packed.retain(|node, _| (node.0 as usize) < program.len());
    if !driver.lowers_block_quantize() || admit(policy, NumericRewrite::ActivationQuantization).is_err() {
        return resolved;
    }
    let activations = consumers_by_activation(&resolved, program, shapes, packed, policy, storage);
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
        packed.insert(derived, held_codec);
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

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_tensor::{
        Extent, IndexMap, Reduce, ReduceInit, append, bind_with_fusion, correct_packed_matmul_layouts,
        derive_activation_quantize, infer, is_derived, projection, underive_block_quantize,
    };
    use crate::cuda::executes;
    use crate::cuda::target::Target;

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
            derive_with_storage(fixture.resolved.clone(), &fixture.program, &fixture.shapes, &mut packed, policy, driver, None);
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
        let (derived, packed) = derive(&fixture, granted(), GpuDriver::Cuda);
        let program_len = fixture.program.len();
        let ids = derived_ids(&derived, program_len);
        assert_eq!(ids, vec![NodeId(program_len as u32)], "{tokens} tokens");
        assert_eq!(packed.get(&ids[0]), Some(&Codec::Q8_1Grouped));
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
    fn mixed_readers_are_handled_for_the_staged_matvec_too() {
        assert_mixed_readers_share_one_derived_node(1);
    }

    #[test]
    fn an_activation_with_only_non_eligible_readers_gets_no_derived_node() {
        for tokens in [1, 3] {
            let mut fixture = fixture(tokens, true);
            fixture.packed.clear();
            let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
            assert_eq!(derived, fixture.resolved, "{tokens} tokens");
        }
    }

    #[test]
    fn the_derived_node_sits_before_the_first_eligible_reader_not_the_first_reader() {
        let mut fixture = fixture(3, true);
        let program_len = fixture.program.len();
        let plain_reader = fixture.resolved.pop().expect("plain reader is last");
        fixture.resolved.insert(0, plain_reader.clone());
        let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
        let ids = derived_ids(&derived, program_len);
        let position = derived.iter().position(|bound| bound.node == ids[0]).expect("derived present");
        assert_eq!(derived[0], plain_reader, "ineligible reader stays first and unchanged");
        assert_eq!(position, 1);
        assert!(readers_of(&derived, ids[0]).contains(&derived[position + 1].node));
        assert_eq!(derive(&fixture, granted(), GpuDriver::Cuda).0, derived, "derivation is deterministic");
    }

    fn derive_stored(fixture: &Fixture, format: NumericFormat) -> Result<(Vec<BoundOp>, PackedOperands), Box<ActivationRefusal>> {
        validate_storage(Some(format))?;
        let mut packed = fixture.packed.clone();
        let derived = derive_with_storage(
            fixture.resolved.clone(),
            &fixture.program,
            &fixture.shapes,
            &mut packed,
            granted(),
            GpuDriver::Cuda,
            Some(format),
        );
        Ok((derived, packed))
    }

    #[test]
    fn naming_the_default_format_derives_the_same_plan_as_naming_none() {
        for tokens in [1, 3] {
            let fixture = fixture(tokens, false);
            let default = derive(&fixture, granted(), GpuDriver::Cuda);
            let named = derive_stored(&fixture, NumericFormat::Q8_1Grouped).expect("the default format is servable");
            assert_eq!(named, default, "{tokens} tokens");
        }
    }

    #[test]
    fn a_float_coded_block_format_is_refused_naming_producer_and_reader_pieces_with_lines() {
        let fixture = fixture(3, false);
        let refusal = derive_stored(&fixture, NumericFormat::Mxfp4).expect_err("mxfp4 has no cuda render yet");
        let producer = refusal.producer.expect("the quantize render lacks a piece");
        let consumer = refusal.consumer.expect("the int8 readers lack a piece");
        assert!(producer.file.ends_with("cuda/block_quantize.rs") && producer.line > 0, "{producer:?}");
        assert!(consumer.file.ends_with("cuda/block_quantize.rs") && consumer.line > 0, "{consumer:?}");
    }

    #[test]
    fn a_weight_only_int8_layout_without_the_stored_sum_is_refused_not_misrendered() {
        let fixture = fixture(3, false);
        assert!(derive_stored(&fixture, NumericFormat::Q8_0).is_err());
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
        assert_eq!(activation_codec(4096, None), Codec::Q8_1Grouped);
        assert_eq!(activation_codec(128, None), Codec::Q8_1Grouped);
        assert_eq!(activation_codec(96, None), Codec::Q8_1);
        assert_eq!(activation_codec(32, None), Codec::Q8_1);
    }

    fn tensor_derive(
        fixture: &Fixture,
        policy: NumericPolicy,
        storage: Option<NumericFormat>,
    ) -> (Vec<BoundOp>, BTreeMap<NodeId, NumericFormat>) {
        derive_activation_quantize(
            fixture.resolved.clone(),
            &fixture.program,
            &fixture.shapes,
            &fixture.packed,
            policy,
            &executes(&Target::CUDA),
            storage,
        )
    }

    #[test]
    fn the_tensor_pass_binds_the_same_ops_and_held_formats_as_the_renderer_asking_pass() {
        let storages = [None, Some(NumericFormat::Q8_1Grouped), Some(NumericFormat::Q8_1)]
            .into_iter()
            .filter(|storage| validate_storage(*storage).is_ok());
        let mut derived_total = 0;
        for storage in storages {
            for (tokens, plain_reader, wide) in
                [1, 3].into_iter().flat_map(|tokens| [false, true].into_iter().flat_map(move |plain| [false, true].map(move |wide| (tokens, plain, wide))))
            {
                let fixture = build_fixture(tokens, plain_reader, wide);
                for (label, policy) in [("granted", granted()), ("off", NumericPolicy::llama_relaxed())] {
                    let mut packed = fixture.packed.clone();
                    let omega = derive_with_storage(
                        fixture.resolved.clone(),
                        &fixture.program,
                        &fixture.shapes,
                        &mut packed,
                        policy,
                        GpuDriver::Cuda,
                        storage,
                    );
                    let (tensor, held) = tensor_derive(&fixture, policy, storage);
                    let case = format!("{tokens} tokens plain_reader={plain_reader} wide={wide} policy={label} storage={storage:?}");
                    assert_eq!(tensor, omega, "bound ops: {case}");
                    let omega_held: BTreeMap<NodeId, NumericFormat> =
                        packed.into_iter().filter(|(node, _)| node.0 as usize >= fixture.program.len()).collect();
                    assert_eq!(held, omega_held, "held formats: {case}");
                    derived_total += derived_ids(&omega, fixture.program.len()).len();
                }
            }
        }
        assert!(derived_total > 0, "no case derived a node, so equality compared nothing");
    }

    #[test]
    fn underive_restores_the_original_bound_ops() {
        let fixture = fixture(3, false);
        let (derived, _) = derive(&fixture, granted(), GpuDriver::Cuda);
        assert_ne!(derived, fixture.resolved);
        assert_eq!(underive_block_quantize(&derived, fixture.program.len()), fixture.resolved);
    }
}
