//! BGE-small-en-v1.5 per-node-class profile: the same `epilogue-profile-diag`
//! probe `epilogue_profile.rs` runs against mnist, aimed at `bge_eval.rs`'s
//! own three real sentences instead. Attributes wall time inside
//! `evaluate_named`'s resolved-node loop (`cpu::run_resolved_nodes_in_arena`,
//! the loop `evaluate_named` -- and so `bge_eval.rs` -- actually walks) into
//! (a) every `Keep::Reduce` fold (the 96 `MatMul`s a `cargo run --example
//! bge_eval` graph inspection found -- 48 `attn_q/k/v/o` at (384,384), 24
//! FFN at (384,1536)/(1536,384), 24 `Q@K^T`/`softmax@V` with no initializer
//! operand), (b) a post-reduce elementwise epilogue (bias-add, the fused
//! `LayerNorm` tail ROW 191 measured), (c) everything else (softmax,
//! transpose/reshape/concat glue, embeddings gather). Non-timed diagnostic
//! (never the sealed bench) -- names the next lever behind the 26.68 ms/sentence
//! e2e number `rooflines.md`'s BGE lane cites.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path;

use proxima_tensor::cpu;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
/// ROW 201's own per-sentence variant of this file's earlier
/// `PROFILE_ITERATIONS=20`/combined-across-sentences method (ROW 195/199):
/// per-sentence isolation, 3-call warm-up excluded from the reset window,
/// 60 measured calls per sentence — so composition effects across BGE's
/// three real `M` shapes are visible individually rather than averaged
/// into one combined number.
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 60;

fn sentences() -> [(&'static str, Vec<i64>); 3] {
    [
        (
            "the cat sat on the mat",
            vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102],
        ),
        (
            "a cat is sitting on a mat",
            vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102],
        ),
        (
            "quantum physics explains atomic energy",
            vec![101, 8559, 5584, 7607, 9593, 2943, 102],
        ),
    ]
}

fn run_one(lowered: &proxima_onnx::lower::Lowered, output: proxima_tensor::NodeId, tokens: &[i64]) {
    let sequence_length = tokens.len();
    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];

    let mut named: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in &lowered.graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "token_type_ids" => &token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }
    let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[output])
        .expect("evaluate BGE-small on the generic executor");
    std::hint::black_box(&evaluated);
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!(
            "skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout"
        );
        return;
    };
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    let items = sentences();
    // one lowering per sentence length -- sequence_length is a pinned
    // symbolic axis, same as bge_eval.rs's own run_pass.
    let lowered_per_sentence: Vec<(proxima_onnx::lower::Lowered, proxima_tensor::NodeId)> = items
        .iter()
        .map(|(_, tokens)| {
            let mut pins = std::collections::BTreeMap::new();
            pins.insert("batch_size", 1u64);
            pins.insert("sequence_length", tokens.len() as u64);
            let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
                .expect("lower BGE-small with pinned symbolic axes");
            let output = lowered
                .graph_outputs
                .first()
                .expect("last_hidden_state output")
                .1;
            (lowered, output)
        })
        .collect();

    let percent = |nanos: u64, total_nanos: u64| -> f64 {
        if total_nanos == 0 {
            0.0
        } else {
            nanos as f64 / total_nanos as f64 * 100.0
        }
    };

    println!(
        "bge_epilogue_profile: per-sentence, {WARMUP_CALLS}-call warm-up excluded, {MEASURED_CALLS} measured calls/sentence"
    );

    let mut combined_total_nanos = 0u64;
    let mut combined_calls = 0u64;
    let mut combined_wall_nanos = 0u64;

    for ((lowered, output), (name, tokens)) in lowered_per_sentence.iter().zip(items.iter()) {
        let sequence_length = tokens.len();

        // warm-up: outside the reset window so first-call effects (allocator
        // warm-up, page faults, per-lowering JIT-adjacent setup) never
        // pollute this sentence's own attributed breakdown.
        for _ in 0..WARMUP_CALLS {
            run_one(lowered, *output, tokens);
        }

        cpu::epilogue_profile_reset();
        let start = std::time::Instant::now();
        for _ in 0..MEASURED_CALLS {
            run_one(lowered, *output, tokens);
        }
        let elapsed = start.elapsed();

        let (reduce_nanos, reduce_calls, epilogue_nanos, epilogue_calls, other_nanos, other_calls) =
            cpu::epilogue_profile_totals();
        let (reduce_gemm_nanos, reduce_gemm_calls, reduce_small_nanos, reduce_small_calls) =
            cpu::epilogue_profile_reduce_split_totals();
        let total_nanos = reduce_nanos + epilogue_nanos + other_nanos;
        let total_calls = reduce_calls + epilogue_calls + other_calls;

        println!("--- {name:?} (M={sequence_length}) ---");
        println!(
            "  (a) reduce-fold (MatMul/attention matmuls) : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
            reduce_calls,
            reduce_nanos,
            percent(reduce_nanos, total_nanos),
            reduce_nanos as f64 / reduce_calls.max(1) as f64
        );
        println!(
            "      (a.1) gemm-shaped (96 MatMuls)         : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
            reduce_gemm_calls,
            reduce_gemm_nanos,
            percent(reduce_gemm_nanos, total_nanos),
            reduce_gemm_nanos as f64 / reduce_gemm_calls.max(1) as f64
        );
        println!(
            "      (a.2) small non-gemm (LayerNorm/pool)  : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
            reduce_small_calls,
            reduce_small_nanos,
            percent(reduce_small_nanos, total_nanos),
            reduce_small_nanos as f64 / reduce_small_calls.max(1) as f64
        );
        println!(
            "  (b) post-reduce epilogue (LayerNorm/bias)   : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
            epilogue_calls,
            epilogue_nanos,
            percent(epilogue_nanos, total_nanos),
            epilogue_nanos as f64 / epilogue_calls.max(1) as f64
        );
        println!(
            "  (c) everything else (softmax/glue/gather)   : {:>10} calls, {:>12} ns total, {:6.2}% of step time, {:.1} ns/call",
            other_calls,
            other_nanos,
            percent(other_nanos, total_nanos),
            other_nanos as f64 / other_calls.max(1) as f64
        );
        println!(
            "  total attributed                            : {total_calls:>10} calls, {total_nanos:>12} ns total over {MEASURED_CALLS} calls ({:.4} ms/sentence attributed)",
            total_nanos as f64 / MEASURED_CALLS as f64 / 1e6
        );
        println!(
            "  wall-clock: {elapsed:?} over {MEASURED_CALLS} calls ({:.4} ms/sentence e2e, includes lowering-adjacent glue outside the profiled loop)",
            elapsed.as_secs_f64() * 1000.0 / MEASURED_CALLS as f64
        );

        combined_total_nanos += total_nanos;
        combined_calls += total_calls;
        combined_wall_nanos += elapsed.as_nanos() as u64;
    }

    let sentences_run = MEASURED_CALLS * items.len();
    println!(
        "=== combined across {} sentences, {sentences_run} total calls ===",
        items.len()
    );
    println!(
        "  total attributed: {combined_calls} calls, {combined_total_nanos} ns ({:.4} ms/sentence attributed mean)",
        combined_total_nanos as f64 / sentences_run as f64 / 1e6
    );
    println!(
        "  wall-clock: {combined_wall_nanos} ns ({:.4} ms/sentence e2e mean)",
        combined_wall_nanos as f64 / sentences_run as f64 / 1e6
    );
}
