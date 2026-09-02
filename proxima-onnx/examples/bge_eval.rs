//! BGE-small-en-v1.5 end-to-end acceptance run: parse -> lower (with pinned
//! symbolic batch/sequence axes, cached across calls) -> evaluate on the
//! `StaticArena` fast path -> CLS-pooled, L2-normalized sentence embeddings
//! -> cosine-similarity sanity check.
//!
//! `docs/discipline.md` ROW 217 named this file's own gap: everything
//! needed to reach the 9.734 ms/sentence `StaticArena` + Accelerate number
//! was landed in the tree (ROW 214's arena+fusion unification, ROW 212's
//! Accelerate route, ROW 211's lowering cache, ROW 207's plan-time weight
//! packing) but this SEALED harness still called plain `cpu::evaluate_named`
//! and published 16.588 ms. This file closes that gap: it builds one
//! [`proxima_tensor::cpu::StaticArena`] per distinct pinned sentence length
//! (the corpus has 3: 7/8/9 tokens), naming the graph's own initializers as
//! `constant_inputs` (they are constant by construction -- the same
//! invariant [`proxima_tensor::cpu::build_static_arena_with_constants`]'s
//! own doc asks a caller to assert), reuses that arena across every call
//! against the same pinned shape, and keeps
//! [`proxima_onnx::lower::lower_graph_pinned_cached`] amortizing the
//! ~26-30ms/call re-decode `lower_graph_pinned` alone would otherwise pay
//! every time.
//!
//! Tokenization: `proxima-tokenizer` is a byte-level BPE tokenizer (see its
//! own crate doc); BGE's `tokenizer.json` declares `BertNormalizer` /
//! `BertPreTokenizer` / WordPiece, a different scheme this crate does not
//! implement. Per this task's own acceptance note, the fallback is a
//! hand-built token-id array using the model's real `vocab.txt` (a plain
//! line-numbered id table, `id = line_number - 1`) and BERT's own
//! `[CLS]=101 ... [SEP]=102` convention -- documented here, not asserted as
//! tokenizer support this crate does not have.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use proxima_tensor::NodeId;
use proxima_tensor::cpu::{
    self, StaticArena, arena_packed_node_count, build_static_arena_with_constants, evaluate_named,
    evaluate_named_with_arena,
};

/// This crate never hardcodes a path onto another repo's checkout -- the
/// model lives on whichever host happens to have it cached, named by
/// `BGE_MODEL_PATH` so this example stays runnable (and skips cleanly
/// when absent) without embedding that path in source.
const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";

/// `[CLS]`, three sentences, `[SEP]` -- ids read directly off the real
/// `vocab.txt` shipped next to the model (`grep -n` for each whole word,
/// `id = line - 1`): sentence A/B are paraphrases of the same claim
/// (cat on a mat), sentence C is topically unrelated (quantum physics).
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

/// Lowering-plan cache, keyed by pinned sequence length -- ROW 211's own
/// lever, amortizing the ~26-30ms/call re-decode of every real weight
/// initializer down to once per distinct pinned shape.
type LowerCache = BTreeMap<u64, proxima_onnx::lower::Lowered>;

/// Arena cache, keyed by the SAME pinned sequence length -- this file's own
/// new lever. An entry is built exactly once, on that shape's first call,
/// and reused by every subsequent call against the same shape for the rest
/// of the process.
type ArenaCache = BTreeMap<u64, StaticArena>;

fn dynamic_inputs(tokens: &[i64]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let sequence_length = tokens.len();
    let input_ids = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];
    (input_ids, attention_mask, token_type_ids)
}

/// Every `Op::Input` name this graph declares, initializers included --
/// `evaluate_named_with_arena`'s own `require_all = true` binding loop
/// expects an entry for every input the arena sized at build time, even the
/// ones `constant_inputs` already pre-bound (a caller re-sending the same
/// bytes overwrites them with the identical value, per
/// `build_static_arena_with_constants`'s own doc). What `constant_inputs`
/// buys is the plan-time width-tile weight packing, not a skipped rebind.
fn named_inputs<'data>(
    initializers: &'data [(String, Vec<f32>)],
    graph_inputs: &'data [String],
    input_ids: &'data [f32],
    attention_mask: &'data [f32],
    token_type_ids: &'data [f32],
) -> Vec<(&'data str, &'data [f32])> {
    let mut named: Vec<(&str, &[f32])> = initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in graph_inputs {
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

fn cls_normalize(data: &[f32]) -> Vec<f32> {
    let cls = &data[0..384];
    let norm = cls.iter().map(|value| value * value).sum::<f32>().sqrt();
    cls.iter().map(|&value| value / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn coefficient_of_variation(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 || mean == 0.0 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|&value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean * 100.0
}

fn mean_ms(durations: &[Duration]) -> f64 {
    let total: Duration = durations.iter().sum();
    (total.as_secs_f64() * 1000.0) / durations.len() as f64
}

/// One pass over the whole corpus on the `StaticArena` fast path: for each
/// sentence, look up (or build, on first sight of that pinned shape) the
/// arena via `arena_cache`, look up (or lower, on first sight) the plan via
/// `lower_cache`, then evaluate. Timing starts AFTER both cache lookups --
/// the point of both caches is amortizing exactly that cost off the timed
/// window. `arena_builds`/`lower_hits`/`lower_misses` are the engagement
/// bookkeeping this file's own report asserts nonzero/exact against.
#[allow(clippy::too_many_arguments)]
fn run_pass_arena(
    graph: &proxima_onnx::messages::GraphProto<'_>,
    items: &[(&str, Vec<i64>)],
    lower_cache: &mut LowerCache,
    arena_cache: &mut ArenaCache,
    accelerate: bool,
    arena_builds: &mut usize,
    lower_hits: &mut usize,
    lower_misses: &mut usize,
) -> (Vec<Duration>, Vec<Vec<f32>>) {
    cpu::set_accelerate_gemm_enabled(accelerate);
    let mut durations = Vec::with_capacity(items.len());
    let mut embeddings = Vec::with_capacity(items.len());
    for (_, tokens) in items {
        let mut pins = BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", tokens.len() as u64);
        let cache_key = tokens.len() as u64;
        let (lowered, cache_hit) =
            proxima_onnx::lower::lower_graph_pinned_cached(lower_cache, graph, &pins, cache_key)
                .expect("lower BGE-small with pinned symbolic axes (cached)");
        if cache_hit {
            *lower_hits += 1;
        } else {
            *lower_misses += 1;
        }
        let output: NodeId = lowered
            .graph_outputs
            .first()
            .expect("last_hidden_state output")
            .1;

        let arena = arena_cache.entry(cache_key).or_insert_with(|| {
            *arena_builds += 1;
            let constant_inputs: Vec<(&str, &[f32])> = lowered
                .initializers
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice()))
                .collect();
            build_static_arena_with_constants(&lowered.program, &[], &[output], &constant_inputs)
                .expect("build BGE-small static arena")
        });

        let (input_ids, attention_mask, token_type_ids) = dynamic_inputs(tokens);
        let named = named_inputs(
            &lowered.initializers,
            &lowered.graph_inputs,
            &input_ids,
            &attention_mask,
            &token_type_ids,
        );

        let eval_start = Instant::now();
        let evaluated = evaluate_named_with_arena(arena, &named)
            .expect("evaluate BGE-small on the StaticArena fast path");
        durations.push(eval_start.elapsed());

        let (data, shape) = evaluated.get(output).expect("last_hidden_state present");
        assert_eq!(
            shape,
            &[1u64, tokens.len() as u64, 384u64],
            "unexpected last_hidden_state shape"
        );
        assert!(
            data.iter().all(|value| value.is_finite()),
            "non-finite value in last_hidden_state"
        );
        embeddings.push(cls_normalize(data));
    }
    (durations, embeddings)
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
    let runs: usize = env::var("BGE_EVAL_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    #[cfg(feature = "bge-eval-diag")]
    {
        proxima_tensor::instrument::reset_checkout_arena_key_compare();
        proxima_tensor::instrument::reset_bind_rebind_compare();
    }

    let mut lower_cache = LowerCache::new();
    let mut arena_cache = ArenaCache::new();
    let mut arena_builds = 0usize;
    let mut lower_hits = 0usize;
    let mut lower_misses = 0usize;

    let mut neon_run_means = Vec::new();
    let mut accel_run_means = Vec::new();
    let mut last_neon_embeddings = Vec::new();
    let mut last_accel_embeddings = Vec::new();

    for run in 0..runs {
        let (durations, embeddings) = run_pass_arena(
            graph,
            &items,
            &mut lower_cache,
            &mut arena_cache,
            false,
            &mut arena_builds,
            &mut lower_hits,
            &mut lower_misses,
        );
        let mean = mean_ms(&durations);
        println!("run {run} neon(accelerate=off): per-sentence={durations:?} mean_ms={mean:.4}");
        neon_run_means.push(mean);
        last_neon_embeddings = embeddings;

        let (durations, embeddings) = run_pass_arena(
            graph,
            &items,
            &mut lower_cache,
            &mut arena_cache,
            true,
            &mut arena_builds,
            &mut lower_hits,
            &mut lower_misses,
        );
        let mean = mean_ms(&durations);
        println!("run {run} accelerate(on): per-sentence={durations:?} mean_ms={mean:.4}");
        accel_run_means.push(mean);
        last_accel_embeddings = embeddings;
    }
    cpu::set_accelerate_gemm_enabled(false);

    #[cfg(feature = "bge-eval-diag")]
    {
        let (checkout_calls, checkout_ticks) =
            proxima_tensor::instrument::checkout_arena_key_compare_totals();
        let (bind_calls, bind_ticks) = proxima_tensor::instrument::bind_rebind_compare_totals();
        let checkout_ns_per_call = proxima_tensor::instrument::ticks_to_nanos(checkout_ticks)
            .checked_div(checkout_calls)
            .unwrap_or(0);
        let bind_ns_per_call = proxima_tensor::instrument::ticks_to_nanos(bind_ticks)
            .checked_div(bind_calls)
            .unwrap_or(0);
        println!(
            "\n=== HIT-COST INSTRUMENTATION (SEALED loop, {} evaluate_named_with_arena calls) ===",
            runs * 2 * items.len()
        );
        println!(
            "checkout_arena key-compare: calls={checkout_calls} total_ns={} ns/call={checkout_ns_per_call} (expected 0: SEALED loop holds its own arena, never calls checkout_arena)",
            proxima_tensor::instrument::ticks_to_nanos(checkout_ticks)
        );
        println!(
            "bind_named_inputs rebind byte-compare: calls={bind_calls} total_ns={} ns/call={bind_ns_per_call}",
            proxima_tensor::instrument::ticks_to_nanos(bind_ticks)
        );
        proxima_tensor::instrument::reset_checkout_arena_key_compare();
        proxima_tensor::instrument::reset_bind_rebind_compare();
    }

    // Correctness: bit-identity vs the `evaluate_named` oracle for the NEON
    // arm. The lowering cache is already warm from the timed loop above, so
    // this reads it directly (no `lower_graph_pinned_cached` call, and so no
    // effect on `lower_hits`/`lower_misses` -- those numbers describe only
    // the timed loop's own 30 calls).
    let mut oracle_embeddings = Vec::with_capacity(items.len());
    for (_, tokens) in &items {
        let cache_key = tokens.len() as u64;
        let lowered = lower_cache
            .get(&cache_key)
            .expect("lowering cache warm from the timed loop above");
        let output = lowered
            .graph_outputs
            .first()
            .expect("last_hidden_state output")
            .1;
        let (input_ids, attention_mask, token_type_ids) = dynamic_inputs(tokens);
        let named = named_inputs(
            &lowered.initializers,
            &lowered.graph_inputs,
            &input_ids,
            &attention_mask,
            &token_type_ids,
        );
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output])
            .expect("oracle evaluate_named");
        let (data, _) = evaluated.get(output).expect("oracle output");
        oracle_embeddings.push(cls_normalize(data));
    }

    let bit_identical = oracle_embeddings.len() == last_neon_embeddings.len()
        && oracle_embeddings
            .iter()
            .zip(last_neon_embeddings.iter())
            .all(|(oracle, arena)| {
                oracle.len() == arena.len()
                    && oracle
                        .iter()
                        .zip(arena.iter())
                        .all(|(&left, &right)| left.to_bits() == right.to_bits())
            });
    println!(
        "\nbit_identical(StaticArena NEON vs evaluate_named oracle, last run) = {bit_identical}"
    );
    assert!(
        bit_identical,
        "StaticArena NEON path drifted from the evaluate_named oracle -- correctness bar violated"
    );

    let accel_max_diffs: Vec<f32> = last_accel_embeddings
        .iter()
        .zip(oracle_embeddings.iter())
        .map(|(accel, oracle)| max_abs_diff(accel, oracle))
        .collect();
    println!(
        "accelerate max_abs_diff vs evaluate_named oracle per sentence (last run) = {accel_max_diffs:?}"
    );

    println!("\n=== ENGAGEMENT PROOF ===");
    println!(
        "1. arena builds = {arena_builds} (expected 3, one per distinct pinned sequence length, never per call)"
    );
    assert_eq!(
        arena_builds, 3,
        "engagement N mismatch: StaticArena should build exactly once per distinct pinned shape"
    );

    let lower_total = lower_hits + lower_misses;
    println!(
        "2. lowering cache: {lower_misses} misses / {lower_hits} hits across {lower_total} embed() calls"
    );
    assert_eq!(
        lower_misses, 3,
        "engagement N mismatch: lowering cache misses"
    );
    assert_eq!(
        lower_hits,
        runs * 2 * items.len() - 3,
        "engagement N mismatch: lowering cache hits"
    );

    let packed_nodes: usize = arena_cache.values().map(arena_packed_node_count).sum();
    println!("3. packed width-tile node count (summed over 3 arenas) = {packed_nodes}");
    assert!(
        packed_nodes > 0,
        "engagement N==0 is RED: no width-tile node was packed on the real BGE graph"
    );

    let (accel_hits, accel_declined) = cpu::accelerate_gemm_totals();
    println!(
        "4. accelerate_gemm_totals (process-cumulative): hits={accel_hits} declined={accel_declined}"
    );
    assert!(
        accel_hits > 0,
        "engagement N==0 is RED: the Accelerate GEMM route never fired"
    );
    assert_eq!(
        accel_declined, 0,
        "Accelerate route declined a call it should have accepted -- see ACCELERATE_GEMM_DECLINED's own doc"
    );

    #[cfg(feature = "bge-eval-diag")]
    {
        // `width_tile_plan`'s own gate-pass/invocation counters live behind
        // `proxima-tensor/instrument`, which this crate's own `mnist-diag`
        // feature already documents as adding 30-40% overhead to every
        // `run_reduce`/`run_elementwise` call -- never safe to enable on the
        // timed loop above. This block runs ONE extra, untimed corpus pass
        // (reusing the already-built arenas) purely to read the counters.
        proxima_tensor::instrument::reset_reduce_gemm_path();
        proxima_tensor::instrument::reset_width_tile_decline();
        let (gate_before, invocations_before, _) = cpu::width_tile_counters();
        cpu::set_accelerate_gemm_enabled(false);
        for (_, tokens) in &items {
            let cache_key = tokens.len() as u64;
            let lowered = lower_cache.get(&cache_key).expect("lowering cache warm");
            let (input_ids, attention_mask, token_type_ids) = dynamic_inputs(tokens);
            let named = named_inputs(
                &lowered.initializers,
                &lowered.graph_inputs,
                &input_ids,
                &attention_mask,
                &token_type_ids,
            );
            let arena = arena_cache
                .get_mut(&cache_key)
                .expect("arena already built by the timed loop");
            let _ = evaluate_named_with_arena(arena, &named).expect("diag pass eval");
        }
        let (_, _, width_fast_calls, _, _, _, _, _) =
            proxima_tensor::instrument::reduce_gemm_path_totals();
        let (gate_after, invocations_after, _) = cpu::width_tile_counters();
        let gate_delta = gate_after - gate_before;
        let invocations_delta = invocations_after - invocations_before;
        println!(
            "5. width_tile_plan gate (diag build, one untimed corpus pass, 3 sentences): {gate_delta} of {width_fast_calls} WidthFast-classified calls resolved Some ({invocations_delta} tile invocations)"
        );
        assert!(
            gate_delta > 0,
            "engagement N==0 is RED: width_tile_plan never engaged on the arena path"
        );
    }

    // `docs/discipline.md` "one execution path" collapse: plain `evaluate_named`
    // (no `StaticArena`, no local cache, no `constant_inputs` opt-in) is now
    // `cpu::evaluate_named_via_arena` internally -- this block proves that
    // reaching it directly, the way a caller who never read this file's own
    // arena plumbing would, still gets the SAME arena-backed default (builds
    // 3 arenas over `default_arm_runs * items.len()` calls, then reuses
    // them) with no code on the caller's side beyond calling `evaluate_named`.
    println!(
        "\n=== DEFAULT-PATH MEASUREMENT (plain evaluate_named, no caller opt-in) ==="
    );
    let default_arm_runs: usize = 5;
    for accelerate in [false, true] {
        cpu::set_accelerate_gemm_enabled(accelerate);
        #[cfg(feature = "bge-eval-diag")]
        {
            cpu::arena_cache_reset();
            cpu::rewrite_engine_reset();
            proxima_tensor::instrument::reset_reduce_gemm_path();
            proxima_tensor::instrument::reset_checkout_arena_key_compare();
            proxima_tensor::instrument::reset_bind_rebind_compare();
        }
        let mut run_means = Vec::new();
        for _run in 0..default_arm_runs {
            let mut durations = Vec::with_capacity(items.len());
            for (_, tokens) in &items {
                let mut pins = BTreeMap::new();
                pins.insert("batch_size", 1u64);
                pins.insert("sequence_length", tokens.len() as u64);
                let cache_key = tokens.len() as u64;
                let (lowered, _hit) = proxima_onnx::lower::lower_graph_pinned_cached(
                    &mut lower_cache,
                    graph,
                    &pins,
                    cache_key,
                )
                .expect("lower BGE-small with pinned symbolic axes (cached)");
                let output: NodeId = lowered
                    .graph_outputs
                    .first()
                    .expect("last_hidden_state output")
                    .1;
                let (input_ids, attention_mask, token_type_ids) = dynamic_inputs(tokens);
                let named = named_inputs(
                    &lowered.initializers,
                    &lowered.graph_inputs,
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                );
                let eval_start = Instant::now();
                let _evaluated = evaluate_named(&lowered.program, &[], &named, &[output])
                    .expect("default-path evaluate_named");
                durations.push(eval_start.elapsed());
            }
            run_means.push(mean_ms(&durations));
        }
        let mean = run_means.iter().sum::<f64>() / run_means.len() as f64;
        let cov = coefficient_of_variation(&run_means, mean);
        println!(
            "accelerate={accelerate}: per-run means={run_means:?} mean_ms/sentence={mean:.4} CoV%={cov:.2}"
        );
        #[cfg(feature = "bge-eval-diag")]
        {
            let (builds, hits) = cpu::arena_cache_totals();
            let packed = cpu::arena_cache_packed_node_count();
            let (depth1, depth2) = cpu::rewrite_engine_depth_fires();
            let (_, _, width_fast_calls, _, _, _, generic_calls, _) =
                proxima_tensor::instrument::reduce_gemm_path_totals();
            println!(
                "  ENGAGEMENT: arena_cache builds={builds} hits={hits} packed_node_count={packed} \
                 rewrite_depth1_fires={depth1} rewrite_depth2_fires={depth2} \
                 width_fast_gemm_calls={width_fast_calls} generic_gemm_calls={generic_calls} \
                 over {} evaluate_named calls",
                default_arm_runs * items.len()
            );
            assert_eq!(
                builds, 3,
                "engagement N mismatch: default evaluate_named path should build exactly 3 arenas (one per distinct pinned length), never per call"
            );
            // `checkout_arena` now derives `constant_inputs` from `named`
            // itself on a cache miss (every genuine `Op::Input` name,
            // filtered against `block_node_ids` so a stray `named` entry can
            // never trip `UnboundInputName`) -- `evaluate_named`'s own
            // signature still carries no explicit weights-vs-activations
            // signal, so this is a structural GUESS, never a caller promise:
            // `build_packed_width_panels` only actually packs the subset
            // that is ALSO the 2-D `b` operand of a width-tile-eligible
            // `Reduce`, and `bind_named_inputs_into_arena`'s own rebind
            // check drops a packed panel the instant a later call proves the
            // guess wrong for that node. `packed > 0` here is the proof this
            // now engages with zero caller opt-in.
            assert!(
                packed > 0,
                "engagement N==0 is RED: packed_node_count should be nonzero now that checkout_arena derives constant_inputs from the program"
            );
            assert!(
                depth1 > 0 || depth2 > 0,
                "engagement N==0 is RED: rewrite fusion never fired on the default evaluate_named path"
            );
            // Accelerate's own route (`try_run_accelerate_width_gemm`) returns
            // before `run_reduce` ever reaches the `record_reduce_gemm_path_ticks`
            // call this counter feeds, so `width_fast_calls` is the right N
            // only with Accelerate off; with it on, `accelerate_gemm_totals`
            // is the counter that proves the SAME `width_tile_plan` gate fired.
            if accelerate {
                let (accel_hits, _) = cpu::accelerate_gemm_totals();
                assert!(
                    accel_hits > 0,
                    "engagement N==0 is RED: Accelerate route never fired on the default evaluate_named path"
                );
            } else {
                assert!(
                    width_fast_calls > 0,
                    "engagement N==0 is RED: width_tile_plan (WidthFast) never engaged on the default evaluate_named path"
                );
            }
            let (checkout_calls, checkout_ticks) =
                proxima_tensor::instrument::checkout_arena_key_compare_totals();
            let (bind_calls, bind_ticks) =
                proxima_tensor::instrument::bind_rebind_compare_totals();
            let checkout_ns_per_call = proxima_tensor::instrument::ticks_to_nanos(checkout_ticks)
                .checked_div(checkout_calls)
                .unwrap_or(0);
            let bind_ns_per_call = proxima_tensor::instrument::ticks_to_nanos(bind_ticks)
                .checked_div(bind_calls)
                .unwrap_or(0);
            println!(
                "  HIT-COST (DEFAULT-PATH, accelerate={accelerate}, {} evaluate_named calls): \
                 checkout_arena key-compare calls={checkout_calls} ns/call={checkout_ns_per_call}; \
                 bind rebind byte-compare calls={bind_calls} ns/call={bind_ns_per_call}",
                default_arm_runs * items.len()
            );
        }
    }
    cpu::set_accelerate_gemm_enabled(false);

    println!("\n=== SEALED MEASUREMENT (StaticArena fast path, ms/sentence) ===");
    println!("runs={runs}");
    let neon_mean = neon_run_means.iter().sum::<f64>() / neon_run_means.len() as f64;
    let neon_cov = coefficient_of_variation(&neon_run_means, neon_mean);
    let accel_mean = accel_run_means.iter().sum::<f64>() / accel_run_means.len() as f64;
    let accel_cov = coefficient_of_variation(&accel_run_means, accel_mean);
    println!(
        "neon(accelerate=off) per-run means: {neon_run_means:?} mean={neon_mean:.4} CoV%={neon_cov:.2}"
    );
    println!(
        "accelerate(on) per-run means: {accel_run_means:?} mean={accel_mean:.4} CoV%={accel_cov:.2}"
    );

    let similar = cosine(&last_neon_embeddings[0], &last_neon_embeddings[1]);
    let dissimilar_a = cosine(&last_neon_embeddings[0], &last_neon_embeddings[2]);
    let dissimilar_b = cosine(&last_neon_embeddings[1], &last_neon_embeddings[2]);
    println!("cosine(A,B similar)={similar:.6}");
    println!("cosine(A,C dissimilar)={dissimilar_a:.6}");
    println!("cosine(B,C dissimilar)={dissimilar_b:.6}");
    for (name, embedding) in ["A", "B", "C"].iter().zip(last_neon_embeddings.iter()) {
        println!("embedding[{name}][:8]={:?}", &embedding[0..8]);
    }
    assert!(
        similar > dissimilar_a,
        "similar pair should score higher than dissimilar pair A"
    );
    assert!(
        similar > dissimilar_b,
        "similar pair should score higher than dissimilar pair B"
    );
    assert!(
        (similar - 0.936311).abs() < 1e-5,
        "cosine(A,B) drifted from the sealed oracle"
    );
    assert!(
        (dissimilar_a - 0.378777).abs() < 1e-5,
        "cosine(A,C) drifted from the sealed oracle"
    );
    assert!(
        (dissimilar_b - 0.334176).abs() < 1e-5,
        "cosine(B,C) drifted from the sealed oracle"
    );
    println!("sanity check passed: similar sentence pair scores higher than dissimilar pairs");
}
