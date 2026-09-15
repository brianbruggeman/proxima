//! Metal-vs-CPU parity for `qwen35_gdn_sequence_position` (checkpoint-free:
//! `proxima_tensor::spec::qwen35_gdn_sequence_position`, private, reproduced
//! here verbatim from the two public ops it lowers to) -- a `[s,d,u,g]`
//! sequence-preserving tensor sliced at a caller-known-at-build-time
//! position `p` via an `elementwise` `Identity` with a constant `s+p@1`
//! offset, then squeezed back off the unit `s` axis with a `reduce(Add,
//! Zero)`. `qwen35_gdn_recurrence_step` calls this once per tap per prompt
//! position in the M>1 branch of `append_qwen35_ssm_mixer_with_taps_and_layout`.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, ReduceInit, ScalarOp};

const SEQUENCE_LEN: u32 = 13;
const HEAD_DIM: u32 = 128;
const HEADS: u32 = 16;
const GROUPS: u32 = 2;

/// Reproduces `proxima_tensor::spec::qwen35_gdn_sequence_position` verbatim
/// (the fn is private) -- an affine-offset slice of the `s` axis followed by
/// a sum-of-one reduction that squeezes it back off.
fn sequence_position(program: &mut Vec<Op>, node: NodeId, rest_letters: &str, position: u32) -> NodeId {
    let rest_terms = rest_letters
        .chars()
        .map(|letter| letter.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let iteration = format!("s{rest_letters}");
    let sliced = elementwise(
        program,
        DType::Float32,
        ScalarOp::Identity,
        &[(node, format!("s+{position}@1,{rest_terms}->{iteration}").as_str())],
    )
    .expect("affine s-offset slice lowers");
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        sliced,
        format!("{iteration}->{iteration}").as_str(),
        format!("{rest_letters}->{iteration}").as_str(),
    )
    .expect("sum-of-one squeeze lowers")
}

struct SliceFixture {
    program: Vec<Op>,
    slices: Vec<(u32, NodeId)>,
}

fn build_fixture(positions: &[u32]) -> SliceFixture {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(SEQUENCE_LEN),
            Extent::Static(HEAD_DIM),
            Extent::Static(HEADS),
            Extent::Static(GROUPS),
        ],
        "x",
    );
    let slices = positions
        .iter()
        .map(|&position| (position, sequence_position(&mut program, x, "dug", position)))
        .collect();
    SliceFixture { program, slices }
}

/// Row `row` is filled with `row * 1_000 + flat_index_within_row` so a
/// slice's values name their own source row, never all-zero/all-one filler
/// (guiding-principle 9).
fn build_x_data() -> Vec<f32> {
    let row_len = (HEAD_DIM * HEADS * GROUPS) as usize;
    (0..SEQUENCE_LEN as usize)
        .flat_map(|row| (0..row_len).map(move |within_row| (row * 1_000 + within_row) as f32))
        .collect()
}

fn row_slice(x_data: &[f32], row: usize) -> &[f32] {
    let row_len = (HEAD_DIM * HEADS * GROUPS) as usize;
    &x_data[row * row_len..(row + 1) * row_len]
}

fn matching_row(x_data: &[f32], row_len: usize, candidate: &[f32]) -> Option<usize> {
    (0..SEQUENCE_LEN as usize).find(|&row| row_slice(x_data, row)[..row_len.min(4)] == candidate[..row_len.min(4)])
}

#[test]
fn metal_sequence_position_slice_matches_cpu_and_source_row_at_each_position() {
    let positions = [0_u32, 5, 12];
    let fixture = build_fixture(&positions);
    let x_data = build_x_data();
    let row_len = (HEAD_DIM * HEADS * GROUPS) as usize;

    let outputs: Vec<NodeId> = fixture.slices.iter().map(|(_, node)| *node).collect();
    let symbols: Vec<u64> = Vec::new();
    let named: Vec<(&str, proxima_tensor::QuantizedBlock)> =
        vec![("x", proxima_tensor::QuantizedBlock::Float32(&x_data))];

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &fixture.program,
        &symbols,
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu reference evaluates the slice program");

    let plan = omega::plan_named(&fixture.program, &symbols, &named, &outputs, NumericPolicy::default())
        .expect("metal plan builds");
    let metal = omega::execute_plan_named(&plan, &named).expect("metal evaluates the slice program");

    let mut failures = Vec::new();
    for &(position, node) in &fixture.slices {
        let cpu_values = cpu.get(node).map(|(values, _)| values.to_vec());
        let metal_values = metal.get(node).map(|(values, _)| values.to_vec());
        let expected_row = row_slice(&x_data, position as usize);

        match (&cpu_values, &metal_values) {
            (Some(cpu_values), Some(metal_values)) if cpu_values == metal_values && metal_values.as_slice() == expected_row => {}
            _ => {
                let metal_head = metal_values.as_ref().map_or_else(Vec::new, |values| values[..4.min(values.len())].to_vec());
                let cpu_head = cpu_values.as_ref().map_or_else(Vec::new, |values| values[..4.min(values.len())].to_vec());
                let matched_row = metal_values
                    .as_ref()
                    .and_then(|values| matching_row(&x_data, row_len, values));
                failures.push(format!(
                    "position {position}: cpu head={cpu_head:?} metal head={metal_head:?} metal actually matches source row {matched_row:?} (expected row {position})"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "sequence position slice parity failures:\n{}",
        failures.join("\n")
    );
}
