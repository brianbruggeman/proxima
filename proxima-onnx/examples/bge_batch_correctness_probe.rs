//! Batch-merge correctness probe (`proxima-wt-batch` task, 2026-09-01):
//! `width_tile_plan`'s new `(0, 0)` two-leading-axis merge (`cpu.rs`) folds
//! `[batch, seq]` into one flat row axis whenever the weight operand is
//! invariant over BOTH -- if the composed row stride is wrong, a row's
//! output would silently read another row's slice rather than crash. The
//! per-row independence a standard encoder graph guarantees (no op crosses
//! the batch axis: MatMul/FFN/LayerNorm/self-attention all stay within one
//! sequence) makes this directly checkable: row `i` of a `batch=8` pass MUST
//! equal a separate `batch=1` pass over row `i`'s own tokens, bit-for-bit at
//! `stride_a`/`stride_b` values indistinguishable from row-major, floating
//! point only through summation-order-preserving 1:1 same-arithmetic reuse.
//! `evaluate_named_with_arena` is used both ways so packing/fusion are
//! identical between the batched and single-row arms.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path as FsPath;

use proxima_tensor::cpu::{build_static_arena_with_constants, evaluate_named_with_arena};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const SEQUENCE_LENGTH: usize = 128;
const BATCH: u64 = 8;
const VOCAB_SIZE: i64 = 30522;

fn synthetic_tokens(length: usize, seed: u64) -> Vec<i64> {
    let interior = length - 2;
    let mut tokens = Vec::with_capacity(length);
    tokens.push(101i64);
    for index in 0..interior {
        let value = (index as u64)
            .wrapping_mul(9301)
            .wrapping_add(seed.wrapping_mul(49297))
            % 26000;
        let id = 2000i64 + value as i64;
        assert!(id < VOCAB_SIZE, "generated id out of vocab range");
        tokens.push(id);
    }
    tokens.push(102i64);
    tokens
}

fn dynamic_inputs_batch(rows: &[Vec<i64>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut token_type_ids = Vec::new();
    for row in rows {
        input_ids.extend(row.iter().map(|&id| id as f32));
        attention_mask.extend(std::iter::repeat_n(1.0f32, row.len()));
        token_type_ids.extend(std::iter::repeat_n(0.0f32, row.len()));
    }
    (input_ids, attention_mask, token_type_ids)
}

fn runtime_named_inputs<'d>(
    graph_inputs: &'d [String],
    input_ids: &'d [f32],
    attention_mask: &'d [f32],
    token_type_ids: &'d [f32],
) -> Vec<(&'d str, &'d [f32])> {
    graph_inputs
        .iter()
        .map(|name| {
            let data: &[f32] = match name.as_str() {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
                other => panic!("unexpected graph input {other:?}"),
            };
            (name.as_str(), data)
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!("skipping: set {MODEL_PATH_ENV}");
        return;
    };
    if !FsPath::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    let rows: Vec<Vec<i64>> = (0..BATCH)
        .map(|row_index| synthetic_tokens(SEQUENCE_LENGTH, row_index * 31 + 128))
        .collect();

    // batched pass, arm B (batch=8)
    let mut pins_batched = BTreeMap::new();
    pins_batched.insert("batch_size", BATCH);
    pins_batched.insert("sequence_length", SEQUENCE_LENGTH as u64);
    let lowered_batched = proxima_onnx::lower::lower_graph_pinned(graph, &pins_batched)
        .expect("lower BGE-small batch=8");
    let output_batched = lowered_batched.graph_outputs.first().expect("output").1;
    let constants_batched: Vec<(&str, &[f32])> = lowered_batched
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let mut arena_batched = build_static_arena_with_constants(
        &lowered_batched.program,
        &[],
        &[output_batched],
        &constants_batched,
    )
    .expect("build batch=8 arena");
    let (input_ids_batched, attention_mask_batched, token_type_ids_batched) =
        dynamic_inputs_batch(&rows);
    let named_batched = runtime_named_inputs(
        &lowered_batched.graph_inputs,
        &input_ids_batched,
        &attention_mask_batched,
        &token_type_ids_batched,
    );
    let evaluated_batched = evaluate_named_with_arena(&mut arena_batched, &named_batched)
        .expect("evaluate batch=8 on the fixed width-tile path");
    let (data_batched, shape_batched) = evaluated_batched
        .get(output_batched)
        .expect("last_hidden_state present");
    assert_eq!(shape_batched, &[BATCH, SEQUENCE_LENGTH as u64, 384u64]);
    assert!(data_batched.iter().all(|value| value.is_finite()));

    // single-row passes, arm A (batch=1 each), same tokens per row
    let mut pins_single = BTreeMap::new();
    pins_single.insert("batch_size", 1u64);
    pins_single.insert("sequence_length", SEQUENCE_LENGTH as u64);
    let lowered_single =
        proxima_onnx::lower::lower_graph_pinned(graph, &pins_single).expect("lower BGE-small batch=1");
    let output_single = lowered_single.graph_outputs.first().expect("output").1;
    let constants_single: Vec<(&str, &[f32])> = lowered_single
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let mut arena_single = build_static_arena_with_constants(
        &lowered_single.program,
        &[],
        &[output_single],
        &constants_single,
    )
    .expect("build batch=1 arena");

    let row_stride = SEQUENCE_LENGTH * 384;
    println!("row | max_abs_diff(batch=8 row, batch=1 same-tokens run)");
    let mut worst = 0.0f32;
    for (row_index, tokens) in rows.iter().enumerate() {
        let (input_ids_single, attention_mask_single, token_type_ids_single) =
            dynamic_inputs_batch(std::slice::from_ref(tokens));
        let named_single = runtime_named_inputs(
            &lowered_single.graph_inputs,
            &input_ids_single,
            &attention_mask_single,
            &token_type_ids_single,
        );
        let evaluated_single = evaluate_named_with_arena(&mut arena_single, &named_single)
            .expect("evaluate batch=1 reference row");
        let (data_single, shape_single) = evaluated_single
            .get(output_single)
            .expect("last_hidden_state present");
        assert_eq!(shape_single, &[1u64, SEQUENCE_LENGTH as u64, 384u64]);

        let batched_row = &data_batched[row_index * row_stride..(row_index + 1) * row_stride];
        let diff = max_abs_diff(batched_row, data_single);
        worst = worst.max(diff);
        println!("{row_index:>3} | {diff:e}");
    }
    println!("\nworst max_abs_diff across {BATCH} rows = {worst:e}");
    assert!(
        worst < 1e-4,
        "batch=8 row diverged from its own batch=1 reference beyond float tolerance: {worst:e}"
    );
    println!("PASS: batch=8 width-tile-merge path matches batch=1 per-row reference within tolerance");
}
