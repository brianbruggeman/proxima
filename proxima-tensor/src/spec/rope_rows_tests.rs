//! `fused_rope_rows` against `fused_rope_pair`: one `[s, head, 2, P]` node must hold, at
//! `p = 0` and `p = 1`, exactly the two nodes the pair builder returns.

use super::*;
use crate::cpu::evaluate_named;
use crate::test_support::Lcg;

const SEQUENCE: u32 = 3;
const HEADS: u32 = 2;
const PAIRS: u32 = 4;
const HEAD_DIM: u32 = 2 * PAIRS;
const THETA: f32 = 10000.0;
const DECODE_POSITIONS: [f32; 3] = [37.0, 38.0, 39.0];

struct RowsFixture {
    program: Vec<Op>,
    pair: (NodeId, NodeId),
    rows: NodeId,
    inputs: Vec<(&'static str, Vec<f32>)>,
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: &[u32]) -> NodeId {
    let extents = shape.iter().map(|&extent| Extent::Static(extent)).collect();
    input_leaf(program, DType::FLOAT32, extents, name)
}

fn rows_fixture(pairing: RopePairing) -> RowsFixture {
    let (cos_bytes, sin_bytes) =
        multi_axis_rope_tables(&DECODE_POSITIONS, &[HEAD_DIM], &[THETA], DType::FLOAT32)
            .expect("rope tables build from real positions and theta");
    let mut program = Vec::new();
    let source = leaf(&mut program, "x", &[SEQUENCE, HEADS, HEAD_DIM]);
    let cos = leaf(&mut program, "cos", &[SEQUENCE, PAIRS]);
    let sin = leaf(&mut program, "sin", &[SEQUENCE, PAIRS]);
    let half = leaf(&mut program, "half", &[2]);
    let pair =
        fused_rope_pair(&mut program, source, 'h', cos, sin, pairing).expect("rope pair builds");
    let rows = fused_rope_rows(&mut program, source, 'h', cos, sin, pairing, half)
        .expect("rope rows builds with every axis sized");
    let mut lcg = Lcg(0x6210_0001);
    let activations = (0..SEQUENCE * HEADS * HEAD_DIM)
        .map(|_| lcg.next_unit())
        .collect();
    let inputs = vec![
        ("x", activations),
        ("cos", decode(&cos_bytes)),
        ("sin", decode(&sin_bytes)),
        ("half", vec![1.0, 0.0]),
    ];
    RowsFixture {
        program,
        pair,
        rows,
        inputs,
    }
}

fn evaluated(fixture: &RowsFixture) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let named: Vec<(&str, &[f32])> = fixture
        .inputs
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect();
    let outputs = [fixture.pair.0, fixture.pair.1, fixture.rows];
    let result = evaluate_named(&fixture.program, &[], &named, &outputs)
        .expect("pair and rows evaluate together");
    let read = |node| result.get(node).expect("output requested").0.to_vec();
    (
        read(fixture.pair.0),
        read(fixture.pair.1),
        read(fixture.rows),
    )
}

fn half_of_rows(rows: &[f32], half: usize) -> Vec<f32> {
    let pairs = PAIRS as usize;
    rows.chunks(2 * pairs)
        .flat_map(|row| row[half * pairs..(half + 1) * pairs].to_vec())
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[proxima::test]
#[case::split_half(RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(RopePairing::Interleaved)]
async fn rope_rows_infers_one_node_of_shape_s_head_two_pairs(#[case] pairing: RopePairing) {
    let fixture = rows_fixture(pairing);

    let shapes = crate::shape::infer(&fixture.program, &[]).expect("no UnconstrainedDim");

    assert_eq!(
        shapes.of(fixture.rows),
        &[u64::from(SEQUENCE), u64::from(HEADS), 2, u64::from(PAIRS)]
    );
}

#[proxima::test]
#[case::split_half(RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(RopePairing::Interleaved)]
async fn rope_rows_halves_equal_the_pair_outputs_bit_for_bit(#[case] pairing: RopePairing) {
    let fixture = rows_fixture(pairing);

    let (even, odd, rows) = evaluated(&fixture);

    assert_eq!(rows.len(), (SEQUENCE * HEADS * 2 * PAIRS) as usize);
    assert_eq!(
        bits(&half_of_rows(&rows, 0)),
        bits(&even),
        "p = 0 is the even node"
    );
    assert_eq!(
        bits(&half_of_rows(&rows, 1)),
        bits(&odd),
        "p = 1 is the odd node"
    );
    std::println!("{pairing:?} row[0,0] = {:?}", &rows[..2 * PAIRS as usize]);
}
