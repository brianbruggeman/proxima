//! Route census: for BGE-small-en-v1.5's own 96 `MatMul` folds
//! (`bge_epilogue_profile.rs`'s own gemm-shaped bucket), which `run_reduce`
//! route actually executes -- width tile / dot tile / conv-gemm tile /
//! generic interpreted fallback -- and how much wall time each owns.
//! Answers this session's own H1 ("does any MatMul take the interpreted
//! fallback") directly from `instrument::reduce_gemm_path_totals()`
//! (`proxima-tensor/src/instrument.rs`), a gemm-restricted split of the
//! SAME `Path` classification `run_reduce` (`cpu.rs`) already records for
//! every reduce fold, added specifically because the all-reduce split
//! conflates the 96 `MatMul`s with 74 small single-operand reduces
//! (LayerNorm mean/variance, softmax max/sum) that structurally can also
//! land in `Path::WidthFast`/`Path::DotFast` without ever reaching
//! `width_tile_plan`/`neon_tile_plan` (both require a `Binary(Multiply,
//! ..)` body, which a `Unary` LayerNorm reduce never has).
//!
//! Cross-checked against `cpu::width_tile_counters()`/`cpu::neon_tile_counters()`
//! -- the tile kernels' own gate-pass/invocation counters -- to discriminate
//! "structurally classified WidthFast but the finer per-shape gate inside
//! `width_tile_plan` declined" from "genuinely reached and ran the NEON
//! kernel", which the `Path` label alone cannot separate (H1 vs H2).
//!
//! Also reports H4's size: the fixed `shape::infer` + `bind::bind` +
//! per-node `vec![0.0; n]` cost `evaluate_named`'s own doc
//! (`proxima-tensor/src/cpu.rs:464-479`) names as paid on every single
//! call, by diffing wall time against the same graph run through a
//! pre-bound `StaticArena` (`evaluate_named_with_arena`), which skips
//! exactly that cost and nothing else -- for ranking only, per this
//! task's own instruction not to build a fix for it.
//!
//! Gated entirely behind `epilogue-profile-diag` (enables
//! `proxima-tensor/instrument` + `proxima-tensor/epilogue-profile-probe`);
//! the production hot path is byte-identical with the feature off, since
//! every counter write this file reads is already `#[cfg(feature =
//! "instrument")]`-gated inside `cpu.rs`, not new gating introduced here.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path as FsPath;
use std::time::Instant;

use proxima_tensor::cpu;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 60;

fn sentences() -> [(&'static str, Vec<i64>); 3] {
    [
        ("the cat sat on the mat", vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
        ("a cat is sitting on a mat", vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
        ("quantum physics explains atomic energy", vec![101, 8559, 5584, 7607, 9593, 2943, 102]),
    ]
}

fn named_inputs<'a>(
    lowered: &'a proxima_onnx::lower::Lowered,
    input_ids: &'a [f32],
    attention_mask: &'a [f32],
    token_type_ids: &'a [f32],
) -> Vec<(&'a str, &'a [f32])> {
    let mut named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
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

fn run_one(lowered: &proxima_onnx::lower::Lowered, output: proxima_tensor::NodeId, tokens: &[i64]) {
    let sequence_length = tokens.len();
    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];
    let named = named_inputs(lowered, &input_ids, &attention_mask, &token_type_ids);
    let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate BGE-small on the generic executor");
    std::hint::black_box(&evaluated);
}

fn percent(nanos: u64, total_nanos: u64) -> f64 {
    if total_nanos == 0 {
        0.0
    } else {
        nanos as f64 / total_nanos as f64 * 100.0
    }
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!("skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout");
        return;
    };
    if !FsPath::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    let items = sentences();
    let lowered_per_sentence: Vec<(proxima_onnx::lower::Lowered, proxima_tensor::NodeId)> = items
        .iter()
        .map(|(_, tokens)| {
            let mut pins = std::collections::BTreeMap::new();
            pins.insert("batch_size", 1u64);
            pins.insert("sequence_length", tokens.len() as u64);
            let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins).expect("lower BGE-small with pinned symbolic axes");
            let output = lowered.graph_outputs.first().expect("last_hidden_state output").1;
            (lowered, output)
        })
        .collect();

    println!("bge_route_census: per-sentence, {WARMUP_CALLS}-call warm-up excluded, {MEASURED_CALLS} measured calls/sentence");
    println!("route | node-calls | ns total | % of gemm time | ns/call\n");

    let mut combined_dot_fast = (0u64, 0u64);
    let mut combined_width_fast = (0u64, 0u64);
    let mut combined_conv_tile = (0u64, 0u64);
    let mut combined_generic = (0u64, 0u64);

    for ((lowered, output), (name, tokens)) in lowered_per_sentence.iter().zip(items.iter()) {
        let sequence_length = tokens.len();

        for _ in 0..WARMUP_CALLS {
            run_one(lowered, *output, tokens);
        }

        cpu::epilogue_profile_reset();
        proxima_tensor::instrument::reset_reduce_gemm_path();
        #[cfg(target_arch = "aarch64")]
        let (width_gate_before, width_invocations_before, _) = cpu::width_tile_counters();
        #[cfg(target_arch = "aarch64")]
        let (neon_gate_before, neon_invocations_before, _) = cpu::neon_tile_counters();

        for _ in 0..MEASURED_CALLS {
            run_one(lowered, *output, tokens);
        }

        let (reduce_gemm_nanos, reduce_gemm_calls, _, _) = cpu::epilogue_profile_reduce_split_totals();
        let (dot_fast_calls, dot_fast_ticks, width_fast_calls, width_fast_ticks, conv_tile_calls, conv_tile_ticks, generic_calls, generic_ticks) =
            proxima_tensor::instrument::reduce_gemm_path_totals();
        let dot_fast_nanos = proxima_tensor::instrument::ticks_to_nanos(dot_fast_ticks);
        let width_fast_nanos = proxima_tensor::instrument::ticks_to_nanos(width_fast_ticks);
        let conv_tile_nanos = proxima_tensor::instrument::ticks_to_nanos(conv_tile_ticks);
        let generic_nanos = proxima_tensor::instrument::ticks_to_nanos(generic_ticks);
        let route_total_nanos = dot_fast_nanos + width_fast_nanos + conv_tile_nanos + generic_nanos;
        let route_total_calls = dot_fast_calls + width_fast_calls + conv_tile_calls + generic_calls;

        #[cfg(target_arch = "aarch64")]
        let (width_gate_after, width_invocations_after, _) = cpu::width_tile_counters();
        #[cfg(target_arch = "aarch64")]
        let (neon_gate_after, neon_invocations_after, _) = cpu::neon_tile_counters();

        println!("--- {name:?} (M={sequence_length}) ---");
        println!(
            "  epilogue-profile-probe cross-check: {reduce_gemm_calls} gemm-shaped reduce calls, {reduce_gemm_nanos} ns wall ({} expected == {MEASURED_CALLS} x 96)",
            reduce_gemm_calls / MEASURED_CALLS as u64
        );
        println!(
            "  route census (gemm-restricted, {route_total_calls} of {reduce_gemm_calls} classified, sum-check {})",
            if route_total_calls == reduce_gemm_calls { "OK" } else { "MISMATCH" }
        );
        println!(
            "    dot_fast    : {dot_fast_calls:>8} calls, {dot_fast_nanos:>12} ns, {:6.2}%, {:.1} ns/call",
            percent(dot_fast_nanos, route_total_nanos),
            dot_fast_nanos as f64 / dot_fast_calls.max(1) as f64
        );
        println!(
            "    width_fast  : {width_fast_calls:>8} calls, {width_fast_nanos:>12} ns, {:6.2}%, {:.1} ns/call",
            percent(width_fast_nanos, route_total_nanos),
            width_fast_nanos as f64 / width_fast_calls.max(1) as f64
        );
        println!(
            "    conv_tile   : {conv_tile_calls:>8} calls, {conv_tile_nanos:>12} ns, {:6.2}%, {:.1} ns/call",
            percent(conv_tile_nanos, route_total_nanos),
            conv_tile_nanos as f64 / conv_tile_calls.max(1) as f64
        );
        println!(
            "    generic     : {generic_calls:>8} calls, {generic_nanos:>12} ns, {:6.2}%, {:.1} ns/call  <-- N==0 is the H1 answer",
            percent(generic_nanos, route_total_nanos),
            generic_nanos as f64 / generic_calls.max(1) as f64
        );

        #[cfg(target_arch = "aarch64")]
        {
            let width_gate_delta = width_gate_after - width_gate_before;
            let width_invocations_delta = width_invocations_after - width_invocations_before;
            let neon_gate_delta = neon_gate_after - neon_gate_before;
            let neon_invocations_delta = neon_invocations_after - neon_invocations_before;
            println!(
                "  width_tile_plan gate: {width_gate_delta} of {width_fast_calls} WidthFast-classified calls actually resolved Some ({width_invocations_delta} tile invocations) -- {} the finer per-shape gate never declines a WidthFast node",
                if width_gate_delta == width_fast_calls { "confirms" } else { "REFUTES" }
            );
            println!(
                "  neon_tile_plan gate : {neon_gate_delta} of {dot_fast_calls} DotFast-classified calls actually resolved Some ({neon_invocations_delta} tile invocations) -- {} the finer per-shape gate never declines a DotFast node",
                if neon_gate_delta == dot_fast_calls { "confirms" } else { "REFUTES" }
            );
        }

        combined_dot_fast.0 += dot_fast_calls;
        combined_dot_fast.1 += dot_fast_nanos;
        combined_width_fast.0 += width_fast_calls;
        combined_width_fast.1 += width_fast_nanos;
        combined_conv_tile.0 += conv_tile_calls;
        combined_conv_tile.1 += conv_tile_nanos;
        combined_generic.0 += generic_calls;
        combined_generic.1 += generic_nanos;
    }

    let combined_total_nanos = combined_dot_fast.1 + combined_width_fast.1 + combined_conv_tile.1 + combined_generic.1;
    let combined_total_calls = combined_dot_fast.0 + combined_width_fast.0 + combined_conv_tile.0 + combined_generic.0;
    println!("\n=== combined across {} sentences, {combined_total_calls} gemm-shaped calls ===", items.len());
    println!(
        "  dot_fast    : {:>8} calls ({:5.2}% of 96), {:>12} ns, {:6.2}% of gemm time, {:.1} ns/call",
        combined_dot_fast.0,
        combined_dot_fast.0 as f64 / combined_total_calls.max(1) as f64 * 100.0,
        combined_dot_fast.1,
        percent(combined_dot_fast.1, combined_total_nanos),
        combined_dot_fast.1 as f64 / combined_dot_fast.0.max(1) as f64
    );
    println!(
        "  width_fast  : {:>8} calls ({:5.2}% of 96), {:>12} ns, {:6.2}% of gemm time, {:.1} ns/call",
        combined_width_fast.0,
        combined_width_fast.0 as f64 / combined_total_calls.max(1) as f64 * 100.0,
        combined_width_fast.1,
        percent(combined_width_fast.1, combined_total_nanos),
        combined_width_fast.1 as f64 / combined_width_fast.0.max(1) as f64
    );
    println!(
        "  conv_tile   : {:>8} calls ({:5.2}% of 96), {:>12} ns, {:6.2}% of gemm time, {:.1} ns/call",
        combined_conv_tile.0,
        combined_conv_tile.0 as f64 / combined_total_calls.max(1) as f64 * 100.0,
        combined_conv_tile.1,
        percent(combined_conv_tile.1, combined_total_nanos),
        combined_conv_tile.1 as f64 / combined_conv_tile.0.max(1) as f64
    );
    println!(
        "  generic     : {:>8} calls ({:5.2}% of 96), {:>12} ns, {:6.2}% of gemm time, {:.1} ns/call  <-- N==0 is the H1 answer",
        combined_generic.0,
        combined_generic.0 as f64 / combined_total_calls.max(1) as f64 * 100.0,
        combined_generic.1,
        percent(combined_generic.1, combined_total_nanos),
        combined_generic.1 as f64 / combined_generic.0.max(1) as f64
    );

    // H2 vs H3 split: within `width_fast`'s own hit calls, ns inside
    // `gemm_width_tile_neon` vs ns in the rest of `run_width_tile_neon`
    // (address computation, column tail, row-remainder dispatch, output
    // store) is NOT separately timed by any existing counter -- the
    // `WIDTH_TILE_*` family counts calls/invocations/fallback elements,
    // never ticks. `REDUCE_GEMM_PATH_WIDTH_FAST_TICKS` above is the whole
    // `run_reduce` call including that overhead, so it upper-bounds but
    // does not isolate H2's own share. Named here as the residual, not
    // measured (see report).
    println!("\nH2/H3 note: no existing counter times ns strictly inside gemm_width_tile_neon alone (vs the rest of run_width_tile_neon) -- REDUCE_GEMM_PATH_WIDTH_FAST_TICKS above is the whole run_reduce call for width_fast-routed nodes, an upper bound on the kernel's own share, not an isolation of it.");

    // H4 sizing: evaluate_named (bind + shape::infer + per-node alloc EVERY
    // call, per that function's own doc, cpu.rs:464-479) versus a pre-bound
    // StaticArena's evaluate_named_with_arena (same graph, same weights,
    // arena built ONCE outside the timed loop) -- the wall-time delta is
    // exactly the per-call cost evaluate_named pays that the arena path
    // does not, for RANKING ONLY per this task's own instruction not to
    // build a fix.
    println!("\n=== H4 sizing: evaluate_named (bind+infer+alloc per call) vs pre-bound StaticArena ===");
    for ((lowered, output), (name, tokens)) in lowered_per_sentence.iter().zip(items.iter()) {
        let sequence_length = tokens.len();
        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];
        let named = named_inputs(lowered, &input_ids, &attention_mask, &token_type_ids);

        for _ in 0..WARMUP_CALLS {
            let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[*output]).expect("warm up evaluate_named");
            std::hint::black_box(&evaluated);
        }
        let named_start = Instant::now();
        for _ in 0..MEASURED_CALLS {
            let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[*output]).expect("evaluate_named");
            std::hint::black_box(&evaluated);
        }
        let named_elapsed = named_start.elapsed();

        let mut arena = cpu::build_static_arena(&lowered.program, &[], &[*output]).expect("build_static_arena");
        for _ in 0..WARMUP_CALLS {
            let evaluated = cpu::evaluate_named_with_arena(&mut arena, &named).expect("warm up evaluate_named_with_arena");
            std::hint::black_box(&evaluated);
        }
        let arena_start = Instant::now();
        for _ in 0..MEASURED_CALLS {
            let evaluated = cpu::evaluate_named_with_arena(&mut arena, &named).expect("evaluate_named_with_arena");
            std::hint::black_box(&evaluated);
        }
        let arena_elapsed = arena_start.elapsed();

        let named_ns_per_call = named_elapsed.as_nanos() as f64 / MEASURED_CALLS as f64;
        let arena_ns_per_call = arena_elapsed.as_nanos() as f64 / MEASURED_CALLS as f64;
        let overhead_ns_per_call = named_ns_per_call - arena_ns_per_call;
        println!(
            "  {name:?} (M={sequence_length}): evaluate_named={named_ns_per_call:.0} ns/call, arena={arena_ns_per_call:.0} ns/call, bind+infer+alloc overhead={overhead_ns_per_call:.0} ns/call ({:.2}% of evaluate_named)",
            overhead_ns_per_call / named_ns_per_call * 100.0
        );
    }
}
