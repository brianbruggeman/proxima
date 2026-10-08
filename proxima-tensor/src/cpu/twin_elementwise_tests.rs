//! The twin-output elementwise against the two-dispatch path it replaces, over the program
//! `spec::fused_rope_pair` really builds. The twin has no CPU interpreter of its own, so each test
//! reads it through an independent scalar walk (`evaluate_twin`) over its shared operand list and
//! compares bit for bit with `evaluate`, which runs the two plain ops.

use super::*;
use crate::bind::{BoundOp, BoundOpKind, fuse_twin_elementwise};
use crate::op::Extent;
use crate::spec::{RopePairing, fused_rope_pair, input_leaf};
use crate::test_support::Lcg;

const SEQUENCE: u32 = 3;
const HEADS: u32 = 2;
const PAIRS: u32 = 4;

struct RopeFixture {
    program: Vec<Op>,
    first: NodeId,
    second: NodeId,
    source: NodeId,
    cosine: NodeId,
    sine: NodeId,
    inputs: Vec<Vec<f32>>,
}

fn random_values(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn rope_fixture(head_dim: u32, pairing: RopePairing) -> RopeFixture {
    let mut program = Vec::new();
    let source = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(SEQUENCE),
            Extent::Static(HEADS),
            Extent::Static(head_dim),
        ],
        "x",
    );
    let trig_shape = || vec![Extent::Static(SEQUENCE), Extent::Static(PAIRS)];
    let cosine = input_leaf(&mut program, DType::Float32, trig_shape(), "cos");
    let sine = input_leaf(&mut program, DType::Float32, trig_shape(), "sin");
    let (first, second) = fused_rope_pair(&mut program, source, 'h', cosine, sine, pairing)
        .expect("rope pair builds over a [seq, heads, head_dim] source");
    let inputs = vec![
        random_values(0x70b3_0001, (SEQUENCE * HEADS * head_dim) as usize),
        random_values(0x70b3_0002, (SEQUENCE * PAIRS) as usize),
        random_values(0x70b3_0003, (SEQUENCE * PAIRS) as usize),
    ];
    RopeFixture {
        program,
        first,
        second,
        source,
        cosine,
        sine,
        inputs,
    }
}

fn bind_rope(fixture: &RopeFixture) -> Vec<BoundOp> {
    let shapes = crate::shape::infer(&fixture.program, &[]).expect("rope program infers");
    crate::bind::bind(
        &fixture.program,
        &shapes,
        &[fixture.first, fixture.second],
        NumericPolicy::default(),
    )
    .expect("rope program binds")
}

fn evaluate_twin(
    bound: &BoundOp,
    inputs: &BTreeMap<NodeId, &[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let BoundOpKind::ElementwiseTwin {
        body,
        operands,
        twin_body,
        ..
    } = &bound.kind
    else {
        panic!("evaluate_twin takes an ElementwiseTwin");
    };
    let total: u64 = bound.extents.iter().product();
    let mut operand_values = vec![0.0f32; operands.len()];
    let mut steps = vec![0.0f32; body.steps.len().max(twin_body.steps.len())];
    let (mut primary, mut twin) = (Vec::new(), Vec::new());
    for linear in 0..total {
        let mut remaining = linear;
        let mut coordinate = vec![0u64; bound.extents.len()];
        for (axis, extent) in bound.extents.iter().enumerate().rev() {
            coordinate[axis] = remaining % extent;
            remaining /= extent;
        }
        for (slot, (node, layout, _)) in operands.iter().enumerate() {
            let buffer = inputs[node];
            operand_values[slot] = buffer[layout.offset_of(&coordinate) as usize];
        }
        primary.push(apply_body(body, &operand_values, &mut steps));
        twin.push(apply_body(twin_body, &operand_values, &mut steps));
    }
    (primary, twin)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn reference_outputs(fixture: &RopeFixture) -> (Vec<f32>, Vec<f32>) {
    let blocks: Vec<&[f32]> = fixture.inputs.iter().map(Vec::as_slice).collect();
    let evaluated = evaluate(
        &fixture.program,
        &[],
        &blocks,
        &[fixture.first, fixture.second],
    )
    .expect("the two-dispatch rope path evaluates");
    let first = evaluated.get(fixture.first).expect("first output").0.to_vec();
    let second = evaluated.get(fixture.second).expect("second output").0.to_vec();
    (first, second)
}

fn input_table(fixture: &RopeFixture) -> BTreeMap<NodeId, &[f32]> {
    [fixture.source, fixture.cosine, fixture.sine]
        .into_iter()
        .zip(fixture.inputs.iter().map(Vec::as_slice))
        .collect()
}

fn twin_in_original_order(fixture: &RopeFixture, twin: &BoundOp) -> (Vec<f32>, Vec<f32>) {
    let (primary, secondary) = evaluate_twin(twin, &input_table(fixture));
    match twin.node == fixture.first {
        true => (primary, secondary),
        false => (secondary, primary),
    }
}

#[proxima::test]
#[case::split_half_full_rotary(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::split_half_partial_rotary(16, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn one_twin_dispatch_matches_the_two_dispatch_rope_path_bit_for_bit(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let fixture = rope_fixture(head_dim, pairing);
    let bound = bind_rope(&fixture);
    assert_eq!(bound.len(), 2, "fused_rope_pair binds to two plain elementwise ops");

    let fused = fuse_twin_elementwise(bound, &fixture.program);

    assert_eq!(fused.len(), 1, "the two sibling ops collapse to one dispatch");
    let twin = &fused[0];
    assert_eq!(twin.kind.name(), "elementwise_twin");
    let outputs = [twin.node, twin.twin_node().expect("twin carries a second node")];
    assert!(outputs.contains(&fixture.first) && outputs.contains(&fixture.second));
    let (reference_first, reference_second) = reference_outputs(&fixture);
    let (twin_first, twin_second) = twin_in_original_order(&fixture, twin);
    assert_eq!(bits(&twin_first), bits(&reference_first), "first rotated half");
    assert_eq!(bits(&twin_second), bits(&reference_second), "second rotated half");
}

#[proxima::test]
#[case::split_half(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn twin_halves_are_the_two_plain_ops_the_pass_merged(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let fixture = rope_fixture(head_dim, pairing);
    let bound = bind_rope(&fixture);
    let fused = fuse_twin_elementwise(bound.clone(), &fixture.program);

    let (primary, secondary) = fused[0].twin_halves().expect("an ElementwiseTwin splits");

    let restored: BTreeSet<NodeId> = [primary.node, secondary.node].into_iter().collect();
    let original: BTreeSet<NodeId> = bound.iter().map(|op| op.node).collect();
    assert_eq!(restored, original, "the halves are exactly the merged nodes");
    for half in [&primary, &secondary] {
        assert_eq!(half.kind.name(), "elementwise");
        assert_eq!(half.extents, bound[0].extents);
        assert_eq!(half.operands().len(), 4, "x same, x partner, cos, sin");
    }
}

#[proxima::test]
async fn a_twin_with_its_bodies_swapped_does_not_match_the_reference() {
    let fixture = rope_fixture(8, RopePairing::SplitHalf { pairs: PAIRS });
    let mut fused = fuse_twin_elementwise(bind_rope(&fixture), &fixture.program);
    let BoundOpKind::ElementwiseTwin {
        body, twin_body, ..
    } = &mut fused[0].kind
    else {
        panic!("rope pair fuses to a twin");
    };
    core::mem::swap(body, twin_body);

    let (swapped_first, swapped_second) = twin_in_original_order(&fixture, &fused[0]);

    let (reference_first, reference_second) = reference_outputs(&fixture);
    assert_ne!(bits(&swapped_first), bits(&reference_first));
    assert_ne!(bits(&swapped_second), bits(&reference_second));
}

#[proxima::test]
async fn elementwise_ops_with_different_reads_are_not_merged() {
    let fixture = rope_fixture(8, RopePairing::SplitHalf { pairs: PAIRS });
    let mut bound = bind_rope(&fixture);
    let BoundOpKind::Elementwise { operands, .. } = &mut bound[1].kind else {
        panic!("rope halves bind to elementwise ops");
    };
    operands[1].1.base += 1;

    let fused = fuse_twin_elementwise(bound, &fixture.program);

    assert_eq!(fused.len(), 2, "a differing layout keeps both dispatches");
    assert!(fused.iter().all(|op| op.twin_node().is_none()));
}

#[proxima::test]
async fn a_lone_elementwise_op_passes_through_unchanged() {
    let fixture = rope_fixture(8, RopePairing::Interleaved);
    let mut bound = bind_rope(&fixture);
    bound.truncate(1);
    let expected = bound.clone();

    let fused = fuse_twin_elementwise(bound, &fixture.program);

    assert_eq!(fused, expected);
}
