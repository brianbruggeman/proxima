//! `perf/plan-cache` session: direct measurement of `lower_graph_pinned`
//! and `build_static_arena_with_constants` cost, per call, against the real
//! BGE-small-en-v1.5 graph -- the two functions `docs/discipline.md` ROW 195
//! named as the "~2.5 ms/sentence lowering-adjacent glue outside the
//! profiled loop" (`discipline.md:17469`). This file measures both DIRECTLY
//! (their own wall time, not a derived residual) rather than trusting that
//! attribution, per this session's own task: "measure before you build."
//!
//! `total` is the production path's own real per-sentence cost:
//! `evaluate_named` (`bge_eval.rs::embed`'s own call, unmodified), which
//! today re-runs `shape::infer` + `bind::bind` + `run_rewrite_worklist` on
//! every call (`cpu.rs:2484/2546/2566`) -- lowering is a SEPARATE, additional
//! cost on top of that, since `bge_eval.rs`'s own loop calls
//! `lower_graph_pinned` once per sentence per pass before ever reaching
//! `evaluate_named` (`bge_eval.rs:96`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use proxima_onnx::lower::lower_graph_pinned_cached;
use proxima_tensor::cpu::{build_static_arena_with_constants, evaluate_named};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const ITERATIONS: usize = 20;
const PAIRED_RUNS: usize = 5;
const PAIRED_CALLS_PER_RUN: usize = 10;

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

fn mean_ms(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn mean_cov(samples: &[f64]) -> (f64, f64) {
    let mean = mean_ms(samples);
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    (mean, variance.sqrt() / mean * 100.0)
}

/// One `lower(via cache) + evaluate_named` step, timed end to end -- the
/// same shape a real per-sentence production call takes. `cache` starting
/// empty every call (`uncached` arm) forces `lower_graph_pinned_cached` to
/// miss every time, standing in for today's un-cached behavior (every call
/// pays the full ~26-30ms decode+clone this session measured); `cache`
/// persisting across calls (`cached` arm) hits on every call after the
/// first, standing in for this session's landed plan cache.
fn timed_step(
    cache: &mut BTreeMap<u64, proxima_onnx::lower::Lowered>,
    graph: &proxima_onnx::messages::GraphProto<'_>,
    tokens: &[i64],
) -> f64 {
    let sequence_length = tokens.len();
    let mut pins = BTreeMap::new();
    pins.insert("batch_size", 1u64);
    pins.insert("sequence_length", sequence_length as u64);
    let cache_key = sequence_length as u64;

    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];

    let start = Instant::now();
    let (lowered, _hit) =
        lower_graph_pinned_cached(cache, graph, &pins, cache_key).expect("lower (cached path)");
    let output = lowered
        .graph_outputs
        .first()
        .expect("last_hidden_state output")
        .1;
    let mut named: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for input_name in &lowered.graph_inputs {
        let data: &[f32] = match input_name.as_str() {
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "token_type_ids" => &token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((input_name.as_str(), data));
    }
    let evaluated = evaluate_named(&lowered.program, &[], &named, &[output])
        .expect("evaluate BGE-small (production path)");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(&evaluated);
    elapsed_ms
}

fn paired_arms(graph: &proxima_onnx::messages::GraphProto<'_>) {
    println!();
    println!(
        "=== paired arms: uncached (fresh cache every call) vs cached (persistent cache, warmed) ==="
    );
    for (name, tokens) in sentences() {
        let mut uncached_ms = Vec::with_capacity(PAIRED_RUNS);
        let mut cached_ms = Vec::with_capacity(PAIRED_RUNS);

        // warm the persistent cache once, outside any timed arm.
        let mut persistent_cache: BTreeMap<u64, proxima_onnx::lower::Lowered> = BTreeMap::new();
        let _ = timed_step(&mut persistent_cache, graph, &tokens);

        for run in 0..PAIRED_RUNS {
            let uncached_first = run % 2 == 0;
            let run_uncached = |samples: &mut Vec<f64>| {
                let mut totals = Vec::with_capacity(PAIRED_CALLS_PER_RUN);
                for _ in 0..PAIRED_CALLS_PER_RUN {
                    let mut fresh_cache: BTreeMap<u64, proxima_onnx::lower::Lowered> =
                        BTreeMap::new();
                    totals.push(timed_step(&mut fresh_cache, graph, &tokens));
                }
                samples.push(mean_ms(&totals));
            };
            let run_cached =
                |samples: &mut Vec<f64>,
                 cache: &mut BTreeMap<u64, proxima_onnx::lower::Lowered>| {
                    let mut totals = Vec::with_capacity(PAIRED_CALLS_PER_RUN);
                    for _ in 0..PAIRED_CALLS_PER_RUN {
                        totals.push(timed_step(cache, graph, &tokens));
                    }
                    samples.push(mean_ms(&totals));
                };
            if uncached_first {
                run_uncached(&mut uncached_ms);
                run_cached(&mut cached_ms, &mut persistent_cache);
            } else {
                run_cached(&mut cached_ms, &mut persistent_cache);
                run_uncached(&mut uncached_ms);
            }
        }

        let (uncached_mean, uncached_cov) = mean_cov(&uncached_ms);
        let (cached_mean, cached_cov) = mean_cov(&cached_ms);
        let ratio = cached_mean / uncached_mean;
        println!("--- {name:?} (M={}) ---", tokens.len());
        println!(
            "  uncached: mean={uncached_mean:.4}ms CoV={uncached_cov:.2}% samples={uncached_ms:?}"
        );
        println!("  cached:   mean={cached_mean:.4}ms CoV={cached_cov:.2}% samples={cached_ms:?}");
        println!(
            "  -> cached/uncached ratio: {ratio:.4}x  ({:.2}% delta)  persistent_cache.len()={}",
            (ratio - 1.0) * 100.0,
            persistent_cache.len()
        );
    }
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

    println!(
        "bge_lowering_cache_profile: direct per-call cost of lower_graph_pinned and build_static_arena_with_constants, {ITERATIONS} iterations/sentence, real BGE-small-en-v1.5"
    );

    for (name, tokens) in sentences() {
        let sequence_length = tokens.len();
        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];

        let mut lower_ms = Vec::with_capacity(ITERATIONS);
        let mut arena_build_ms = Vec::with_capacity(ITERATIONS);
        let mut eval_ms = Vec::with_capacity(ITERATIONS);

        for _ in 0..ITERATIONS {
            let mut pins = BTreeMap::new();
            pins.insert("batch_size", 1u64);
            pins.insert("sequence_length", sequence_length as u64);

            let lower_start = Instant::now();
            let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
                .expect("lower BGE-small with pinned symbolic axes");
            lower_ms.push(lower_start.elapsed().as_secs_f64() * 1000.0);

            let output = lowered
                .graph_outputs
                .first()
                .expect("last_hidden_state output")
                .1;
            let weights: Vec<(&str, &[f32])> = lowered
                .initializers
                .iter()
                .map(|(weight_name, data)| (weight_name.as_str(), data.as_slice()))
                .collect();

            let arena_start = Instant::now();
            let arena =
                build_static_arena_with_constants(&lowered.program, &[], &[output], &weights)
                    .expect("build static arena");
            arena_build_ms.push(arena_start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&arena);

            let mut named: Vec<(&str, &[f32])> = weights;
            for input_name in &lowered.graph_inputs {
                let data: &[f32] = match input_name.as_str() {
                    "input_ids" => &input_ids,
                    "attention_mask" => &attention_mask,
                    "token_type_ids" => &token_type_ids,
                    other => panic!("unexpected graph input {other:?}"),
                };
                named.push((input_name.as_str(), data));
            }
            let eval_start = Instant::now();
            let evaluated = evaluate_named(&lowered.program, &[], &named, &[output])
                .expect("evaluate BGE-small on the generic executor (production path)");
            eval_ms.push(eval_start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&evaluated);
        }

        let lower_mean = mean_ms(&lower_ms);
        let arena_mean = mean_ms(&arena_build_ms);
        let eval_mean = mean_ms(&eval_ms);
        let total_mean = lower_mean + eval_mean;
        let lowering_share_of_total = lower_mean / total_mean * 100.0;
        let lowering_plus_arena_share_of_eval = (lower_mean + arena_mean) / eval_mean * 100.0;

        println!("--- {name:?} (M={sequence_length}) ---");
        println!("  lower_graph_pinned:            mean={lower_mean:.4}ms samples={lower_ms:?}");
        println!(
            "  build_static_arena_with_constants: mean={arena_mean:.4}ms samples={arena_build_ms:?}"
        );
        println!("  evaluate_named (production, no arena): mean={eval_mean:.4}ms");
        println!(
            "  lowering share of (lowering+eval) production total: {lowering_share_of_total:.2}%"
        );
        println!(
            "  (lowering+arena-build) as % of evaluate_named alone: {lowering_plus_arena_share_of_eval:.2}%"
        );
    }

    paired_arms(graph);
}
