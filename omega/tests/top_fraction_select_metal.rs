#![cfg(all(
    target_os = "macos",
    feature = "metal",
    feature = "instrument",
    feature = "top-fraction-fusion"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{input_leaf, top_fraction_mask};
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

const TWELVE_SCORES: [f32; 12] = [0.5, 3.0, 1.5, 3.0, 0.1, 2.5, 0.9, 4.0, 1.1, 2.0, 0.3, 5.0];

struct Fixture {
    program: Vec<Op>,
    mask: NodeId,
}

fn fixture(with_keep_rows: bool) -> Fixture {
    let mut program = Vec::new();
    let scores = input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "scores");
    let keep_count = input_leaf(&mut program, DType::Float32, Vec::new(), "keep_count");
    let keep_rows = with_keep_rows
        .then(|| input_leaf(&mut program, DType::Float32, vec![Extent::Symbolic(0)], "keep_rows"));
    let mask = top_fraction_mask(&mut program, scores, keep_count, keep_rows).expect("mask lowers");
    Fixture { program, mask }
}

fn tiled_scores(rows: usize) -> Vec<f32> {
    (0..rows).map(|row| TWELVE_SCORES[row % 12]).collect()
}

fn quarter_of_rows(rows: usize) -> usize {
    rows.div_ceil(4)
}

fn hand_derived_set(rows: usize) -> Vec<usize> {
    (0..rows)
        .filter(|row| match row % 12 {
            11 | 7 => true,
            1 | 3 => row / 12 <= 10,
            _ => false,
        })
        .collect()
}

fn selected(values: &[f32]) -> Vec<usize> {
    values.iter().enumerate().filter(|(_, value)| **value == 1.0).map(|(row, _)| row).collect()
}

fn run_metal(rows: usize, keep_count: usize, keep_rows_data: Option<&[f32]>) -> Vec<f32> {
    let fixture = fixture(keep_rows_data.is_some());
    let scores = tiled_scores(rows);
    let keep = [keep_count as f32];
    let mut named = vec![
        ("scores", QuantizedBlock::Float32(&scores)),
        ("keep_count", QuantizedBlock::Float32(&keep)),
    ];
    if let Some(data) = keep_rows_data {
        named.push(("keep_rows", QuantizedBlock::Float32(data)));
    }
    let symbols = [rows as u64];
    let plan = omega::plan_named(&fixture.program, &symbols, &named, &[fixture.mask], NumericPolicy::default())
        .expect("metal plan builds");
    let evaluated = omega::execute_plan_named(&plan, &named).expect("metal evaluates");
    let (values, _shape) = evaluated.get(fixture.mask).expect("mask requested");
    values.to_vec()
}

fn run_cpu_plain(rows: usize, keep_count: usize, keep_rows_data: Option<&[f32]>) -> Vec<f32> {
    let fixture = fixture(keep_rows_data.is_some());
    let scores = tiled_scores(rows);
    let keep = [keep_count as f32];
    let mut inputs: Vec<(&str, &[f32])> = vec![("scores", &scores), ("keep_count", &keep)];
    if let Some(data) = keep_rows_data {
        inputs.push(("keep_rows", data));
    }
    let evaluated = proxima_tensor::cpu::evaluate_named(&fixture.program, &[rows as u64], &inputs, &[fixture.mask])
        .expect("cpu evaluates");
    let (values, _shape) = evaluated.get(fixture.mask).expect("mask requested");
    values.to_vec()
}

fn ranked_reference_set(rows: usize) -> Vec<usize> {
    let scores = tiled_scores(rows);
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]).then(left.cmp(right)));
    let mut chosen: Vec<usize> = order.into_iter().take(quarter_of_rows(rows)).collect();
    chosen.sort_unstable();
    chosen
}

#[test]
fn selection_kernel_matches_hand_derived_set_at_threshold() {
    let rows = 256;
    let metal = run_metal(rows, quarter_of_rows(rows), None);
    assert_eq!(metal.len(), rows);
    assert_eq!(selected(&metal), hand_derived_set(rows));
    assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), None));
    let keys = omega::pipeline_cache_keys();
    assert!(
        keys.iter().any(|key| key.contains("top_fraction_select_r256_k0")),
        "the selection kernel must be what ran, got {keys:?}"
    );
}

#[test]
fn plain_expression_matches_hand_derived_set_below_threshold() {
    let rows = 255;
    let metal = run_metal(rows, quarter_of_rows(rows), None);
    assert_eq!(metal.len(), rows);
    assert_eq!(selected(&metal), hand_derived_set(rows));
    assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), None));
    let keys = omega::pipeline_cache_keys();
    assert!(
        keys.iter().all(|key| !key.contains("top_fraction_select")),
        "below the threshold only the plain expression may run, got {keys:?}"
    );
}

#[test]
fn selection_kernel_unions_keep_rows() {
    let rows = 256;
    let mut keep = vec![0.0_f32; rows];
    keep[0] = 1.0;
    keep[255] = 1.0;
    let metal = run_metal(rows, quarter_of_rows(rows), Some(&keep));
    let mut expected = hand_derived_set(rows);
    expected.extend([0, 255]);
    expected.sort_unstable();
    assert_eq!(selected(&metal), expected);
    assert_eq!(metal, run_cpu_plain(rows, quarter_of_rows(rows), Some(&keep)));
    assert!(
        omega::pipeline_cache_keys().iter().any(|key| key.contains("top_fraction_select_r256_k1")),
        "the selection kernel must be what ran with the keep flags operand"
    );
}

#[test]
fn selection_kernel_breaks_ties_by_index_across_thread_chunks() {
    for rows in [3000_usize, 4097] {
        let metal = run_metal(rows, quarter_of_rows(rows), None);
        assert_eq!(metal.len(), rows);
        assert_eq!(selected(&metal), ranked_reference_set(rows), "rows {rows}");
        let expected_key = format!("top_fraction_select_r{rows}_k0");
        assert!(
            omega::pipeline_cache_keys().iter().any(|key| key.contains(&expected_key)),
            "the selection kernel must be what ran at {rows} rows"
        );
    }
}

#[test]
fn selection_kernel_keeps_only_flagged_rows_when_the_count_is_zero() {
    let rows = 256;
    let mut keep = vec![0.0_f32; rows];
    keep[5] = 1.0;
    let metal = run_metal(rows, 0, Some(&keep));
    assert_eq!(metal.len(), rows);
    assert_eq!(selected(&metal), vec![5]);
    assert_eq!(metal, run_cpu_plain(rows, 0, Some(&keep)));
    assert!(selected(&run_metal(rows, 0, None)).is_empty());
    let keys = omega::pipeline_cache_keys();
    for expected_key in ["top_fraction_select_r256_k0", "top_fraction_select_r256_k1"] {
        assert!(
            keys.iter().any(|key| key.contains(expected_key)),
            "the selection kernel must be what ran, missing {expected_key} in {keys:?}"
        );
    }
}
