//! Post-fix batch-sweep seal (`perf/batch-seal` task, 2026-09-02):
//! re-runs `bge_traffic_sweep.rs`'s section (C) batch sweep at
//! `b781d0c`'s `composed_reduction_stride` fix, with BOTH Accelerate arms
//! (`bge_traffic_sweep.rs` section (C) only ever measured NEON) and a
//! `width_tile_plan` engagement census per `batch` (`bge_batch_route_census.rs`'s
//! own counters), so one binary produces every "ours" cell this task's
//! table needs. B in {1, 8, 32}, S=128, same `synthetic_tokens` formula as
//! every other harness in this family so token content is comparable
//! across `ours`/`onnxruntime`.
//!
//! At least 8 measured calls per (batch, accelerate) cell (task floor is
//! 5); CoV% reported per cell so a CoV exceeding 5% reads as a range, not a
//! point.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path as FsPath;

use proxima_tensor::cpu;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const SEQUENCE_LENGTH: usize = 128;
const BATCH_SIZES: [u64; 3] = [1, 8, 32];
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 8;
const CENSUS_CALLS: usize = 30;
const VOCAB_SIZE: i64 = 30522;

fn synthetic_tokens(length: usize, seed: u64) -> Vec<i64> {
    assert!(length >= 3, "need room for [CLS] + >=1 interior + [SEP]");
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

fn named_inputs<'a>(
    lowered: &'a proxima_onnx::lower::Lowered,
    input_ids: &'a [f32],
    attention_mask: &'a [f32],
    token_type_ids: &'a [f32],
) -> Vec<(&'a str, &'a [f32])> {
    let mut named: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in &lowered.graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }
    named
}

fn lower_for_batch(
    graph: &proxima_onnx::messages::GraphProto<'_>,
    batch: u64,
) -> (proxima_onnx::lower::Lowered, proxima_tensor::NodeId, Vec<Vec<i64>>) {
    let mut pins = BTreeMap::new();
    pins.insert("batch_size", batch);
    pins.insert("sequence_length", SEQUENCE_LENGTH as u64);
    let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
        .expect("lower BGE-small with pinned batch/sequence axes");
    let output = lowered
        .graph_outputs
        .first()
        .expect("last_hidden_state output")
        .1;
    let rows: Vec<Vec<i64>> = (0..batch)
        .map(|row_index| synthetic_tokens(SEQUENCE_LENGTH, row_index * 31 + 128))
        .collect();
    (lowered, output, rows)
}

fn run_one(lowered: &proxima_onnx::lower::Lowered, output: proxima_tensor::NodeId, rows: &[Vec<i64>]) {
    let (input_ids, attention_mask, token_type_ids) = dynamic_inputs_batch(rows);
    let named = named_inputs(lowered, &input_ids, &attention_mask, &token_type_ids);
    let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[output])
        .expect("evaluate BGE-small on the generic executor");
    std::hint::black_box(&evaluated);
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn coefficient_of_variation_percent(values: &[f64]) -> f64 {
    let average = mean(values);
    if average == 0.0 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / average * 100.0
}

/// One (batch, accelerate) cell: `MEASURED_CALLS` independent full-call
/// timings, each call timed whole (lower is cached across calls via the
/// process-global arena cache the same way `run_pass_arena_shape` is,
/// but this harness re-lowers once per batch up front and only times
/// `run_one`, matching `bge_traffic_sweep.rs` section (C)'s own protocol).
fn time_cell(
    lowered: &proxima_onnx::lower::Lowered,
    output: proxima_tensor::NodeId,
    rows: &[Vec<i64>],
    accelerate: bool,
) -> Vec<f64> {
    cpu::set_accelerate_gemm_enabled(accelerate);
    for _ in 0..WARMUP_CALLS {
        run_one(lowered, output, rows);
    }
    let mut times_ms = Vec::with_capacity(MEASURED_CALLS);
    for _ in 0..MEASURED_CALLS {
        let start = std::time::Instant::now();
        run_one(lowered, output, rows);
        times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    cpu::set_accelerate_gemm_enabled(false);
    times_ms
}

/// `width_tile_plan` engagement census at one `batch`, independent of the
/// accelerate toggle (accelerate only reroutes the downstream
/// `neon_tile_plan`/DotFast GEMM call, never the width-tile gate itself --
/// `bge_batch_route_census.rs`'s own instrumentation, reused verbatim here
/// so this binary is the sole source for both the timing and the
/// engagement halves of this task's table).
fn census_engagement(graph: &proxima_onnx::messages::GraphProto<'_>, batch: u64) {
    let (lowered, output, rows) = lower_for_batch(graph, batch);
    for _ in 0..WARMUP_CALLS {
        run_one(&lowered, output, &rows);
    }

    proxima_tensor::instrument::reset_reduce_gemm_path();
    proxima_tensor::instrument::reset_width_tile_decline();
    let (width_gate_before, _, _) = cpu::width_tile_counters();

    for _ in 0..CENSUS_CALLS {
        run_one(&lowered, output, &rows);
    }

    let (dot_fast_calls, _, width_fast_calls, _, conv_tile_calls, _, generic_calls, _) =
        proxima_tensor::instrument::reduce_gemm_path_totals();
    let total_gemm = dot_fast_calls + width_fast_calls + conv_tile_calls + generic_calls;
    let per_call_gemm = total_gemm / CENSUS_CALLS as u64;

    let (width_gate_after, _, _) = cpu::width_tile_counters();
    let width_gate_delta = width_gate_after - width_gate_before;
    let engaged_per_call = width_gate_delta / CENSUS_CALLS.max(1) as u64;

    println!(
        "batch={batch:>2}: width_tile_plan engagement = {engaged_per_call}/{per_call_gemm} \
         gemm-shaped reduce calls per call ({total_gemm} total over {CENSUS_CALLS} calls)"
    );
    assert!(
        engaged_per_call > 0,
        "width_tile_plan engagement N==0 at batch={batch} -- RED, this is exactly the regression \
         this task exists to catch"
    );

    let declines = proxima_tensor::instrument::width_tile_decline_snapshot();
    if declines.is_empty() {
        println!("  no width_tile_plan declines at batch={batch}");
    } else {
        println!("  {} distinct decline node/reason pairs:", declines.len());
        for (node_id, reason, calls, matmul_m, matmul_k, matmul_n, stride_a, stride_b) in &declines {
            let onnx_name = lowered
                .matmul_names
                .iter()
                .find(|(node, _)| node.0 == *node_id)
                .map_or("<not-a-matmul-output>", |(_, name)| name.as_str());
            println!(
                "    %{node_id:<4} | {onnx_name:<55} | {reason:?} | calls={calls:>3} | \
                 m={matmul_m:>4} k={matmul_k:>4} n={matmul_n:>4} | stride_a={stride_a:>2} stride_b={stride_b:>2}"
            );
        }
    }
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!(
            "skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout"
        );
        return;
    };
    if !FsPath::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    println!("=== bge_batch_seal: engagement census ===");
    for &batch in &BATCH_SIZES {
        census_engagement(graph, batch);
    }

    println!("\n=== bge_batch_seal: timing sweep (ours, S={SEQUENCE_LENGTH}) ===");
    println!(
        "{:>5} | {:>10} | {:>13} | {:>7} | {:>13} | {:>7} | {:>13} | {:>14}",
        "batch", "arm", "total_ms/call", "CoV%", "ms/sentence", "CoV%", "sent/sec", "sent/sec(mean)"
    );
    for &batch in &BATCH_SIZES {
        let (lowered, output, rows) = lower_for_batch(graph, batch);

        for (arm_name, accelerate) in [("neon", false), ("accelerate", true)] {
            let times_ms = time_cell(&lowered, output, &rows, accelerate);
            let call_mean = mean(&times_ms);
            let call_cov = coefficient_of_variation_percent(&times_ms);
            let per_sentence: Vec<f64> = times_ms.iter().map(|value| value / batch as f64).collect();
            let sentence_mean = mean(&per_sentence);
            let sentence_cov = coefficient_of_variation_percent(&per_sentence);
            let throughput_mean = 1000.0 / sentence_mean;
            println!(
                "{batch:>5} | {arm_name:>10} | {call_mean:>13.4} | {call_cov:>6.2}% | \
                 {sentence_mean:>13.4} | {sentence_cov:>6.2}% | {:>13.1} | {throughput_mean:>14.1}",
                batch as f64 / (call_mean / 1000.0),
            );
        }
    }

    println!("\n\n=== bge_batch_seal complete ===");
}
