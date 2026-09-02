//! Batch regression root-cause probe: `bge_route_census.rs`'s own route
//! census, generalized to a `batch` parameter and pinned at `S=128`
//! (`proxima-wt-batch` task, 2026-09-01) -- answers "does `width_tile_plan`
//! engagement drop the moment `batch_size` is pinned above 1, and if so,
//! which of its own `return None` points fires".
//!
//! `pins.insert("batch_size", batch)` the same way `bge_traffic_sweep.rs`'s
//! own section (C) pins it; token content is `bge_traffic_sweep.rs`'s
//! `synthetic_tokens` formula, reproduced here rather than imported (both
//! are example binaries, no shared lib target between them).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path as FsPath;

use proxima_tensor::cpu;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const SEQUENCE_LENGTH: usize = 128;
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 60;
const SPLIT_REPS: usize = 5;
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

fn percent(nanos: u64, total_nanos: u64) -> f64 {
    if total_nanos == 0 {
        0.0
    } else {
        nanos as f64 / total_nanos as f64 * 100.0
    }
}

/// One `batch`'s full census: engagement N/96, decline reasons,
/// `ROWS`-instantiation counts (main-tile vs row-remainder), and the
/// kernel-vs-surround split with in-kernel GMAC/s.
fn census_one_batch(graph: &proxima_onnx::messages::GraphProto<'_>, batch: u64) {
    println!("\n\n=== batch={batch} S={SEQUENCE_LENGTH} ===");
    let (lowered, output, rows) = lower_for_batch(graph, batch);

    for _ in 0..WARMUP_CALLS {
        run_one(&lowered, output, &rows);
    }

    // --- engagement + decline census ---
    cpu::epilogue_profile_reset();
    proxima_tensor::instrument::reset_reduce_gemm_path();
    proxima_tensor::instrument::reset_width_tile_decline();
    let (width_gate_before, width_invocations_before, width_fallback_before) =
        cpu::width_tile_counters();
    let (neon_gate_before, neon_invocations_before, _) = cpu::neon_tile_counters();
    let row_remainder_invocations_before = cpu::width_tile_row_remainder_invocations();
    let row_remainder_elements_before = cpu::width_tile_row_remainder_elements();

    for _ in 0..MEASURED_CALLS {
        run_one(&lowered, output, &rows);
    }

    let (
        dot_fast_calls,
        _dot_fast_ticks,
        width_fast_calls,
        _width_fast_ticks,
        conv_tile_calls,
        _conv_tile_ticks,
        generic_calls,
        _generic_ticks,
    ) = proxima_tensor::instrument::reduce_gemm_path_totals();
    let total_gemm = dot_fast_calls + width_fast_calls + conv_tile_calls + generic_calls;

    let (width_gate_after, width_invocations_after, width_fallback_after) =
        cpu::width_tile_counters();
    let (neon_gate_after, neon_invocations_after, _) = cpu::neon_tile_counters();
    let row_remainder_invocations_after = cpu::width_tile_row_remainder_invocations();
    let row_remainder_elements_after = cpu::width_tile_row_remainder_elements();

    let width_gate_delta = width_gate_after - width_gate_before;
    let width_invocations_delta = width_invocations_after - width_invocations_before;
    let width_fallback_delta = width_fallback_after - width_fallback_before;
    let neon_gate_delta = neon_gate_after - neon_gate_before;
    let neon_invocations_delta = neon_invocations_after - neon_invocations_before;
    let row_remainder_invocations_delta =
        row_remainder_invocations_after - row_remainder_invocations_before;
    let row_remainder_elements_delta = row_remainder_elements_after - row_remainder_elements_before;

    let per_call_gemm = total_gemm / MEASURED_CALLS as u64;
    println!(
        "gemm-shaped reduce calls: {total_gemm} total ({per_call_gemm}/call, expect 96/call)"
    );
    println!(
        "width_tile_plan gate: {width_gate_delta} of {width_fast_calls} WidthFast-classified calls \
         resolved Some ({width_invocations_delta} main-tile kernel invocations, {width_fallback_delta} \
         scalar fallback elements) -- N/96 engagement this run = {}/{per_call_gemm}",
        width_gate_delta / MEASURED_CALLS.max(1) as u64
    );
    println!(
        "neon_tile_plan gate  : {neon_gate_delta} of {dot_fast_calls} DotFast-classified calls \
         resolved Some ({neon_invocations_delta} tile invocations)"
    );
    println!(
        "ROWS instantiation (width tile): main-tile ROWS=4 invocations={width_invocations_delta}, \
         row-remainder invocations={row_remainder_invocations_delta} (sum of ROWS across those \
         calls={row_remainder_elements_delta} elements / tile_cols)"
    );

    let declines = proxima_tensor::instrument::width_tile_decline_snapshot();
    println!(
        "width_tile_plan declines ({} distinct node/reason pairs):",
        declines.len()
    );
    println!("  node | onnx matmul name | reason | calls | m | k | n | stride_a | stride_b");
    for (node_id, reason, calls, matmul_m, matmul_k, matmul_n, stride_a, stride_b) in &declines {
        let onnx_name = lowered
            .matmul_names
            .iter()
            .find(|(node, _)| node.0 == *node_id)
            .map_or("<not-a-matmul-output>", |(_, name)| name.as_str());
        println!(
            "  %{node_id:<4} | {onnx_name:<55} | {reason:?} | {calls:>3} | {matmul_m:>4} | \
             {matmul_k:>4} | {matmul_n:>4} | {stride_a:>2} | {stride_b:>2}"
        );
    }

    // --- kernel-vs-surround split, H2/H3-style, SPLIT_REPS independent reps ---
    let mut kernel_ns_per_call = Vec::with_capacity(SPLIT_REPS);
    let mut surround_ns_per_call = Vec::with_capacity(SPLIT_REPS);
    let mut gmacs_per_rep = Vec::with_capacity(SPLIT_REPS);
    let mut last_fn_calls = 0u64;
    let mut last_width_fast_calls = 0u64;

    for _ in 0..SPLIT_REPS {
        proxima_tensor::instrument::reset_width_tile_split();
        proxima_tensor::instrument::reset_reduce_gemm_path();

        for _ in 0..MEASURED_CALLS {
            run_one(&lowered, output, &rows);
        }

        let (kernel_ticks, kernel_macs, fn_ticks, fn_calls) =
            proxima_tensor::instrument::width_tile_split_totals();
        let (_, _, width_fast_calls, width_fast_ticks, _, _, _, _) =
            proxima_tensor::instrument::reduce_gemm_path_totals();

        let kernel_ns = proxima_tensor::instrument::ticks_to_nanos(kernel_ticks) as f64;
        let fn_ns = proxima_tensor::instrument::ticks_to_nanos(fn_ticks) as f64;
        let reduce_ns = proxima_tensor::instrument::ticks_to_nanos(width_fast_ticks) as f64;
        let surround_ns = (fn_ns - kernel_ns).max(0.0);
        let _outside_ns = (reduce_ns - fn_ns).max(0.0);

        kernel_ns_per_call.push(kernel_ns / MEASURED_CALLS as f64);
        surround_ns_per_call.push(surround_ns / MEASURED_CALLS as f64);
        gmacs_per_rep.push(kernel_macs as f64 / kernel_ns.max(1.0));
        last_fn_calls = fn_calls;
        last_width_fast_calls = width_fast_calls;
    }

    println!(
        "H2/H3 split: {last_fn_calls} run_width_tile_neon calls total ({last_width_fast_calls} \
         width_fast run_reduce calls total, match={})",
        if last_fn_calls == last_width_fast_calls {
            "OK"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "  kernel  : {:>9.1} ns/call mean, CoV={:>5.2}%",
        mean(&kernel_ns_per_call),
        coefficient_of_variation_percent(&kernel_ns_per_call)
    );
    println!(
        "  surround: {:>9.1} ns/call mean, CoV={:>5.2}%",
        mean(&surround_ns_per_call),
        coefficient_of_variation_percent(&surround_ns_per_call)
    );
    println!(
        "  in-kernel GMAC/s: {:>7.3}, CoV={:>5.2}%",
        mean(&gmacs_per_rep),
        coefficient_of_variation_percent(&gmacs_per_rep)
    );

    // --- generic-path share of gemm wall time, direct from route totals ---
    proxima_tensor::instrument::reset_reduce_gemm_path();
    for _ in 0..MEASURED_CALLS {
        run_one(&lowered, output, &rows);
    }
    let (
        dot_fast_calls2,
        dot_fast_ticks2,
        width_fast_calls2,
        width_fast_ticks2,
        conv_tile_calls2,
        conv_tile_ticks2,
        generic_calls2,
        generic_ticks2,
    ) = proxima_tensor::instrument::reduce_gemm_path_totals();
    let dot_fast_nanos = proxima_tensor::instrument::ticks_to_nanos(dot_fast_ticks2);
    let width_fast_nanos = proxima_tensor::instrument::ticks_to_nanos(width_fast_ticks2);
    let conv_tile_nanos = proxima_tensor::instrument::ticks_to_nanos(conv_tile_ticks2);
    let generic_nanos = proxima_tensor::instrument::ticks_to_nanos(generic_ticks2);
    let route_total_nanos = dot_fast_nanos + width_fast_nanos + conv_tile_nanos + generic_nanos;
    println!(
        "route wall time: dot_fast={dot_fast_calls2}calls/{dot_fast_nanos}ns ({:.2}%), \
         width_fast={width_fast_calls2}calls/{width_fast_nanos}ns ({:.2}%), \
         conv_tile={conv_tile_calls2}calls/{conv_tile_nanos}ns ({:.2}%), \
         generic={generic_calls2}calls/{generic_nanos}ns ({:.2}%)",
        percent(dot_fast_nanos, route_total_nanos),
        percent(width_fast_nanos, route_total_nanos),
        percent(conv_tile_nanos, route_total_nanos),
        percent(generic_nanos, route_total_nanos),
    );
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

    for &batch in &[1u64, 8u64] {
        census_one_batch(graph, batch);
    }

    println!("\n\n=== bge_batch_route_census complete ===");
}
