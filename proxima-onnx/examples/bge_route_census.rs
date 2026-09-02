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
// composition-split task (2026-09-01): closes ROW 213's own named residual
// (this file's "H2/H3 note", below) -- independent reps for a CoV, not a
// single aggregate measurement, the same discipline `bge_width_tile_accs.rs`
// already uses for its own GMAC/s table.
const SPLIT_REPS: usize = 5;

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

/// Per-call overhead of one `read_ticks()`/`elapsed_ticks()` pair, timed
/// around nothing -- the number the reader subtracts from every kernel-call
/// boundary reading below, since a ticks read around a ~1-2us kernel call is
/// cheap but not free.
fn measure_ticks_pair_overhead_ns() -> f64 {
    const OVERHEAD_SAMPLES: usize = 100_000;
    let started = proxima_tensor::instrument::read_ticks();
    for _ in 0..OVERHEAD_SAMPLES {
        let pair_start = proxima_tensor::instrument::read_ticks();
        let elapsed = proxima_tensor::instrument::elapsed_ticks(pair_start);
        std::hint::black_box(elapsed);
    }
    let total_ticks = proxima_tensor::instrument::elapsed_ticks(started);
    let total_ns = proxima_tensor::instrument::ticks_to_nanos(total_ticks);
    total_ns as f64 / OVERHEAD_SAMPLES as f64
}

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

fn run_one(lowered: &proxima_onnx::lower::Lowered, output: proxima_tensor::NodeId, tokens: &[i64]) {
    let sequence_length = tokens.len();
    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];
    let named = named_inputs(lowered, &input_ids, &attention_mask, &token_type_ids);
    let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[output])
        .expect("evaluate BGE-small on the generic executor");
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

    let items = sentences();
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

    println!(
        "bge_route_census: per-sentence, {WARMUP_CALLS}-call warm-up excluded, {MEASURED_CALLS} measured calls/sentence"
    );
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
        proxima_tensor::instrument::reset_width_tile_decline();
        #[cfg(target_arch = "aarch64")]
        let (width_gate_before, width_invocations_before, _) = cpu::width_tile_counters();
        #[cfg(target_arch = "aarch64")]
        let (neon_gate_before, neon_invocations_before, _) = cpu::neon_tile_counters();

        for _ in 0..MEASURED_CALLS {
            run_one(lowered, *output, tokens);
        }

        let (reduce_gemm_nanos, reduce_gemm_calls, _, _) =
            cpu::epilogue_profile_reduce_split_totals();
        let (
            dot_fast_calls,
            dot_fast_ticks,
            width_fast_calls,
            width_fast_ticks,
            conv_tile_calls,
            conv_tile_ticks,
            generic_calls,
            generic_ticks,
        ) = proxima_tensor::instrument::reduce_gemm_path_totals();
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
            if route_total_calls == reduce_gemm_calls {
                "OK"
            } else {
                "MISMATCH"
            }
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
                if width_gate_delta == width_fast_calls {
                    "confirms"
                } else {
                    "REFUTES"
                }
            );
            println!(
                "  neon_tile_plan gate : {neon_gate_delta} of {dot_fast_calls} DotFast-classified calls actually resolved Some ({neon_invocations_delta} tile invocations) -- {} the finer per-shape gate never declines a DotFast node",
                if neon_gate_delta == dot_fast_calls {
                    "confirms"
                } else {
                    "REFUTES"
                }
            );
        }

        // Step 1 (width-gate-decline task, 2026-09-01): every `width_tile_plan`
        // `None` for a gemm-shaped node, named back to its ONNX `MatMul`
        // output tensor via `lowered.matmul_names` -- the `Path::WidthFast`
        // label alone cannot distinguish "the NEON tile ran" from "the node
        // fell through to the untiled scalar loop", both commit the same
        // label (see `instrument::WidthDeclineReason`'s own doc).
        let declines = proxima_tensor::instrument::width_tile_decline_snapshot();
        println!(
            "\n  width_tile_plan declines ({} distinct node/reason pairs):",
            declines.len()
        );
        println!("  node | onnx matmul name | reason | calls | m | k | n | stride_a | stride_b");
        for (node_id, reason, calls, matmul_m, matmul_k, matmul_n, stride_a, stride_b) in &declines
        {
            let onnx_name = lowered
                .matmul_names
                .iter()
                .find(|(node, _)| node.0 == *node_id)
                .map_or("<not-a-matmul-output>", |(_, name)| name.as_str());
            println!(
                "    %{node_id:<4} | {onnx_name:<55} | {reason:?} | {calls:>3} | {matmul_m:>4} | {matmul_k:>4} | {matmul_n:>4} | {stride_a:>2} | {stride_b:>2}"
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

    let combined_total_nanos =
        combined_dot_fast.1 + combined_width_fast.1 + combined_conv_tile.1 + combined_generic.1;
    let combined_total_calls =
        combined_dot_fast.0 + combined_width_fast.0 + combined_conv_tile.0 + combined_generic.0;
    println!(
        "\n=== combined across {} sentences, {combined_total_calls} gemm-shaped calls ===",
        items.len()
    );
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

    // H2/H3 split (composition-split task, 2026-09-01): closes the residual
    // this section used to name -- `instrument::width_tile_split_totals()`
    // now times ns strictly inside `gemm_width_tile_neon` separately from
    // the rest of `run_width_tile_neon`, and `reduce_gemm_path_totals()`'s
    // own `width_fast_ticks` (already read above, whole `run_reduce`) gives
    // the third bucket by subtraction. `SPLIT_REPS` independent reps, not
    // one aggregate measurement, so every cell below carries a CoV%.
    let overhead_ns_per_pair = measure_ticks_pair_overhead_ns();
    println!(
        "\n=== H2/H3: kernel-vs-surround-vs-outside split ({SPLIT_REPS} reps x {MEASURED_CALLS} calls/rep) ==="
    );
    println!(
        "  per-call ticks-pair instrumentation overhead: {overhead_ns_per_pair:.2} ns/pair (measured over 100000 empty read_ticks/elapsed_ticks pairs)"
    );

    let mut combined_kernel_ns = Vec::new();
    let mut combined_surround_ns = Vec::new();
    let mut combined_outside_ns = Vec::new();
    let mut combined_kernel_macs_total = 0u64;
    let mut combined_kernel_ns_total = 0.0f64;

    for ((lowered, output), (name, tokens)) in lowered_per_sentence.iter().zip(items.iter()) {
        let sequence_length = tokens.len();
        for _ in 0..WARMUP_CALLS {
            run_one(lowered, *output, tokens);
        }

        let mut kernel_ns_per_call = Vec::with_capacity(SPLIT_REPS);
        let mut surround_ns_per_call = Vec::with_capacity(SPLIT_REPS);
        let mut outside_ns_per_call = Vec::with_capacity(SPLIT_REPS);
        let mut gmacs_per_rep = Vec::with_capacity(SPLIT_REPS);
        let mut last_kernel_invocations = 0u64;
        let mut last_fn_calls = 0u64;
        let mut last_width_fast_calls = 0u64;

        for _ in 0..SPLIT_REPS {
            proxima_tensor::instrument::reset_width_tile_split();
            proxima_tensor::instrument::reset_reduce_gemm_path();
            #[cfg(target_arch = "aarch64")]
            let (_, main_invocations_before, _) = cpu::width_tile_counters();
            #[cfg(target_arch = "aarch64")]
            let row_remainder_invocations_before = cpu::width_tile_row_remainder_invocations();

            for _ in 0..MEASURED_CALLS {
                run_one(lowered, *output, tokens);
            }

            let (kernel_ticks, kernel_macs, fn_ticks, fn_calls) =
                proxima_tensor::instrument::width_tile_split_totals();
            let (_, _, width_fast_calls, width_fast_ticks, _, _, _, _) =
                proxima_tensor::instrument::reduce_gemm_path_totals();
            #[cfg(target_arch = "aarch64")]
            let (_, main_invocations_after, _) = cpu::width_tile_counters();
            #[cfg(target_arch = "aarch64")]
            let row_remainder_invocations_after = cpu::width_tile_row_remainder_invocations();

            let kernel_ns = proxima_tensor::instrument::ticks_to_nanos(kernel_ticks) as f64;
            let fn_ns = proxima_tensor::instrument::ticks_to_nanos(fn_ticks) as f64;
            let reduce_ns = proxima_tensor::instrument::ticks_to_nanos(width_fast_ticks) as f64;
            let surround_ns = (fn_ns - kernel_ns).max(0.0);
            let outside_ns = (reduce_ns - fn_ns).max(0.0);

            kernel_ns_per_call.push(kernel_ns / MEASURED_CALLS as f64);
            surround_ns_per_call.push(surround_ns / MEASURED_CALLS as f64);
            outside_ns_per_call.push(outside_ns / MEASURED_CALLS as f64);
            gmacs_per_rep.push(kernel_macs as f64 / kernel_ns.max(1.0));
            combined_kernel_macs_total += kernel_macs;
            combined_kernel_ns_total += kernel_ns;
            last_fn_calls = fn_calls;
            last_width_fast_calls = width_fast_calls;
            #[cfg(target_arch = "aarch64")]
            {
                last_kernel_invocations = (main_invocations_after - main_invocations_before)
                    + (row_remainder_invocations_after - row_remainder_invocations_before);
            }
        }

        println!("--- {name:?} (M={sequence_length}) ---");
        println!(
            "  N: {last_fn_calls} run_width_tile_neon calls/rep ({last_width_fast_calls} width_fast run_reduce calls/rep, match={}), {last_kernel_invocations} gemm_width_tile_neon kernel invocations/rep",
            if last_fn_calls == last_width_fast_calls {
                "OK"
            } else {
                "MISMATCH"
            }
        );
        println!(
            "  kernel   (gemm_width_tile_neon)      : {:>9.1} ns/call, CoV={:>5.2}%",
            mean(&kernel_ns_per_call),
            coefficient_of_variation_percent(&kernel_ns_per_call)
        );
        println!(
            "  surround (rest of run_width_tile_neon): {:>9.1} ns/call, CoV={:>5.2}%",
            mean(&surround_ns_per_call),
            coefficient_of_variation_percent(&surround_ns_per_call)
        );
        println!(
            "  outside  (run_reduce outside the fn)  : {:>9.1} ns/call, CoV={:>5.2}%",
            mean(&outside_ns_per_call),
            coefficient_of_variation_percent(&outside_ns_per_call)
        );
        println!(
            "  in-kernel GMAC/s: {:>7.3}, CoV={:>5.2}%  (isolated gemm_width_tile_neon ceiling: 48.0-48.8 GMAC/s, ROW 210)",
            mean(&gmacs_per_rep),
            coefficient_of_variation_percent(&gmacs_per_rep)
        );

        combined_kernel_ns.extend(kernel_ns_per_call);
        combined_surround_ns.extend(surround_ns_per_call);
        combined_outside_ns.extend(outside_ns_per_call);
    }

    println!(
        "\n=== composition split, combined across {} sentences ===",
        items.len()
    );
    println!(
        "  kernel   : {:>9.1} ns/call mean, CoV={:>5.2}%",
        mean(&combined_kernel_ns),
        coefficient_of_variation_percent(&combined_kernel_ns)
    );
    println!(
        "  surround : {:>9.1} ns/call mean, CoV={:>5.2}%",
        mean(&combined_surround_ns),
        coefficient_of_variation_percent(&combined_surround_ns)
    );
    println!(
        "  outside  : {:>9.1} ns/call mean, CoV={:>5.2}%",
        mean(&combined_outside_ns),
        coefficient_of_variation_percent(&combined_outside_ns)
    );
    println!(
        "  in-kernel GMAC/s (combined, macs-weighted): {:.3}",
        combined_kernel_macs_total as f64 / combined_kernel_ns_total.max(1.0)
    );

    // H4 sizing: evaluate_named (bind + shape::infer + per-node alloc EVERY
    // call, per that function's own doc, cpu.rs:464-479) versus a pre-bound
    // StaticArena's evaluate_named_with_arena (same graph, same weights,
    // arena built ONCE outside the timed loop) -- the wall-time delta is
    // exactly the per-call cost evaluate_named pays that the arena path
    // does not, for RANKING ONLY per this task's own instruction not to
    // build a fix.
    println!(
        "\n=== H4 sizing: evaluate_named (bind+infer+alloc per call) vs pre-bound StaticArena ==="
    );
    for ((lowered, output), (name, tokens)) in lowered_per_sentence.iter().zip(items.iter()) {
        let sequence_length = tokens.len();
        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];
        let named = named_inputs(lowered, &input_ids, &attention_mask, &token_type_ids);

        for _ in 0..WARMUP_CALLS {
            let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[*output])
                .expect("warm up evaluate_named");
            std::hint::black_box(&evaluated);
        }
        let named_start = Instant::now();
        for _ in 0..MEASURED_CALLS {
            let evaluated = cpu::evaluate_named(&lowered.program, &[], &named, &[*output])
                .expect("evaluate_named");
            std::hint::black_box(&evaluated);
        }
        let named_elapsed = named_start.elapsed();

        let mut arena =
            cpu::build_static_arena(&lowered.program, &[], &[*output]).expect("build_static_arena");
        for _ in 0..WARMUP_CALLS {
            let evaluated = cpu::evaluate_named_with_arena(&mut arena, &named)
                .expect("warm up evaluate_named_with_arena");
            std::hint::black_box(&evaluated);
        }
        let arena_start = Instant::now();
        for _ in 0..MEASURED_CALLS {
            let evaluated = cpu::evaluate_named_with_arena(&mut arena, &named)
                .expect("evaluate_named_with_arena");
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
