//! ROW 206 MILLI rung: `bge_epilogue_profile`'s own per-sentence, per-class
//! wall-time attribution (ROW 195/197/198/199/200/201's own ~18.5-19.3ms
//! band), packed vs unpacked, on the real BGE-small-en-v1.5 graph via
//! `BGE_MODEL_PATH` (never written to a tracked file).
//!
//! Both arms use `StaticArena` (unpacked: `build_static_arena`; packed:
//! `build_static_arena_with_constants` with every weight initializer named
//! as a `constant_inputs` entry -- only the rank-2 width-tile-eligible ones
//! actually get packed, matching ROW 205's own per-node gate), so the delta
//! isolates packing from `StaticArena`'s own already-landed bind+alloc
//! amortization (ROW 164/175), same discipline as ROW 205/206-MICRO's
//! `arena_cold_arm`/`packed_cold_arm` split.
//!
//! Arms are INTERLEAVED per run (unpacked-then-packed on even runs,
//! packed-then-unpacked on odd runs) so a monotonic host-load drift across
//! the whole session cannot bias one arm.
//!
//! PRE-REGISTRATION (recorded before this file was ever run, ONE rung ahead
//! of the file's own MICRO cells only -- not a derived e2e claim):
//! `bge_epilogue_profile`'s own class (a.1) is the 96 constant-weight GEMMs
//! -- but only 72 of them have a RANK-2 constant `b` operand eligible for
//! packing (`QKVO` 48 + `FFN` 24 = 72 attn_q/k/v/o + FFN matmuls read a
//! genuine weight initializer as `b`; the remaining 24 are `Q@K^T`/
//! `softmax@V`, whose `b` operand is another NODE's runtime output, not an
//! `Op::Input` name, so `constant_inputs` can never match them -- INELIGIBLE
//! by construction, not by a missed opportunity). If class (a.1)'s own
//! measured %-of-step-time share (ROW 201's own per-sentence number) is
//! S_gemm, and this rung's own MICRO cold-form measured packed/unpacked
//! speedup is X, the SCALED prediction is: milli step time should move by
//! roughly `(72/96) * S_gemm * (1 - 1/X)`, i.e. the eligible-72 fraction of
//! the GEMM share, not the full 96 -- this is a ONE-RUNG PREDICTION from
//! this file's OWN micro cells, printed at runtime once the micro numbers
//! are known, not a derived e2e claim; the milli number below is MEASURED
//! independently of it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path;

use proxima_tensor::cpu::{
    self, StaticArena, build_static_arena, build_static_arena_with_constants,
    evaluate_named_with_arena,
};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 60;
const RUNS: usize = 5;

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

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

struct ArmTotals {
    reduce_gemm_nanos: u64,
    reduce_gemm_calls: u64,
    total_nanos: u64,
}

fn timed_arm(
    arena: &mut StaticArena,
    weights: &NamedInputs<'_>,
    input_ids: &[f32],
    attention_mask: &[f32],
    token_type_ids: &[f32],
    input_names: &[String; 3],
) -> (f64, ArmTotals) {
    let mut named: NamedInputs<'_> = weights.clone();
    named.push((input_names[0].as_str(), input_ids));
    named.push((input_names[1].as_str(), attention_mask));
    named.push((input_names[2].as_str(), token_type_ids));

    for _ in 0..WARMUP_CALLS {
        let evaluated = evaluate_named_with_arena(arena, &named).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    cpu::epilogue_profile_reset();
    let start = std::time::Instant::now();
    for _ in 0..MEASURED_CALLS {
        let evaluated = evaluate_named_with_arena(arena, &named).expect("timed eval");
        std::hint::black_box(&evaluated);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / MEASURED_CALLS as f64;

    let (reduce_nanos, reduce_calls, epilogue_nanos, epilogue_calls, other_nanos, other_calls) =
        cpu::epilogue_profile_totals();
    let (reduce_gemm_nanos, reduce_gemm_calls, _reduce_small_nanos, _reduce_small_calls) =
        cpu::epilogue_profile_reduce_split_totals();
    let _ = (reduce_calls, epilogue_calls, other_calls);
    let total_nanos = reduce_nanos + epilogue_nanos + other_nanos;
    (
        elapsed_ms,
        ArmTotals {
            reduce_gemm_nanos,
            reduce_gemm_calls,
            total_nanos,
        },
    )
}

fn mean_cov(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    (mean, variance.sqrt() / mean * 100.0)
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
        "bge_epilogue_profile_pack: ROW 206 MILLI rung -- packed vs unpacked, {RUNS} interleaved runs, {WARMUP_CALLS}-call warm-up excluded, {MEASURED_CALLS} measured calls/arm/run"
    );
    println!(
        "PRE-REGISTRATION: see file doc comment -- scaled prediction uses ONLY the 72/96 eligible-GEMM fraction, printed per-sentence once the sentence's own class (a.1) share is measured."
    );

    let items = sentences();
    for (name, tokens) in items.iter() {
        let sequence_length = tokens.len();
        let mut pins = std::collections::BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", sequence_length as u64);
        let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
            .expect("lower BGE-small with pinned symbolic axes");
        let output = lowered
            .graph_outputs
            .first()
            .expect("last_hidden_state output")
            .1;

        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];
        let mut input_names: [String; 3] = [String::new(), String::new(), String::new()];
        for graph_input in &lowered.graph_inputs {
            match graph_input.as_str() {
                "input_ids" => input_names[0] = graph_input.clone(),
                "attention_mask" => input_names[1] = graph_input.clone(),
                "token_type_ids" => input_names[2] = graph_input.clone(),
                other => panic!("unexpected graph input {other:?}"),
            }
        }

        let weights: NamedInputs<'_> = lowered
            .initializers
            .iter()
            .map(|(weight_name, data)| (weight_name.as_str(), data.as_slice()))
            .collect();

        let mut unpacked_arena =
            build_static_arena(&lowered.program, &[], &[output]).expect("build unpacked arena");
        let mut packed_arena =
            build_static_arena_with_constants(&lowered.program, &[], &[output], &weights)
                .expect("build packed arena");

        println!("--- {name:?} (M={sequence_length}) ---");
        let mut unpacked_ms = Vec::with_capacity(RUNS);
        let mut packed_ms = Vec::with_capacity(RUNS);
        let mut unpacked_gemm_share = Vec::with_capacity(RUNS);
        let mut packed_gemm_share = Vec::with_capacity(RUNS);

        for run in 0..RUNS {
            let unpacked_first = run % 2 == 0;
            if unpacked_first {
                let (unpacked_elapsed_ms, unpacked_totals) = timed_arm(
                    &mut unpacked_arena,
                    &weights,
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    &input_names,
                );
                unpacked_ms.push(unpacked_elapsed_ms);
                unpacked_gemm_share.push(
                    unpacked_totals.reduce_gemm_nanos as f64
                        / unpacked_totals.total_nanos.max(1) as f64
                        * 100.0,
                );
                let (packed_elapsed_ms, packed_totals) = timed_arm(
                    &mut packed_arena,
                    &weights,
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    &input_names,
                );
                packed_ms.push(packed_elapsed_ms);
                packed_gemm_share.push(
                    packed_totals.reduce_gemm_nanos as f64
                        / packed_totals.total_nanos.max(1) as f64
                        * 100.0,
                );
                println!(
                    "  run {run} (unpacked, packed): unpacked={unpacked_elapsed_ms:.4}ms (gemm-calls={}) packed={packed_elapsed_ms:.4}ms (gemm-calls={})",
                    unpacked_totals.reduce_gemm_calls, packed_totals.reduce_gemm_calls
                );
            } else {
                let (packed_elapsed_ms, packed_totals) = timed_arm(
                    &mut packed_arena,
                    &weights,
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    &input_names,
                );
                packed_ms.push(packed_elapsed_ms);
                packed_gemm_share.push(
                    packed_totals.reduce_gemm_nanos as f64
                        / packed_totals.total_nanos.max(1) as f64
                        * 100.0,
                );
                let (unpacked_elapsed_ms, unpacked_totals) = timed_arm(
                    &mut unpacked_arena,
                    &weights,
                    &input_ids,
                    &attention_mask,
                    &token_type_ids,
                    &input_names,
                );
                unpacked_ms.push(unpacked_elapsed_ms);
                unpacked_gemm_share.push(
                    unpacked_totals.reduce_gemm_nanos as f64
                        / unpacked_totals.total_nanos.max(1) as f64
                        * 100.0,
                );
                println!(
                    "  run {run} (packed, unpacked): packed={packed_elapsed_ms:.4}ms (gemm-calls={}) unpacked={unpacked_elapsed_ms:.4}ms (gemm-calls={})",
                    packed_totals.reduce_gemm_calls, unpacked_totals.reduce_gemm_calls
                );
            }
        }

        let (unpacked_mean, unpacked_cov) = mean_cov(&unpacked_ms);
        let (packed_mean, packed_cov) = mean_cov(&packed_ms);
        let (unpacked_gemm_mean, _) = mean_cov(&unpacked_gemm_share);
        let (packed_gemm_mean, _) = mean_cov(&packed_gemm_share);
        let ratio = packed_mean / unpacked_mean;

        println!(
            "  unpacked: mean={unpacked_mean:.4}ms CoV={unpacked_cov:.2}% gemm-share={unpacked_gemm_mean:.2}% samples={:?}",
            unpacked_ms
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
        );
        println!(
            "  packed:   mean={packed_mean:.4}ms CoV={packed_cov:.2}% gemm-share={packed_gemm_mean:.2}% samples={:?}",
            packed_ms
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
        );
        println!(
            "  -> packed/unpacked step-time ratio: {ratio:.4}x  ({:.2}% delta)",
            (ratio - 1.0) * 100.0
        );
        if unpacked_cov > 5.0 || packed_cov > 5.0 {
            println!(
                "  -> CoV above 5% trust line on at least one arm -- report the RANGE, not the point estimate, for this sentence."
            );
        }

        // `docs/discipline.md` ROW 207 promotion confirmation: the valve is
        // a PLAN-BUILD-TIME gate (`build_packed_width_panels` checks it
        // once, inside `build_static_arena_with_constants`), never a
        // hot-path branch -- so proving it gates anything means building a
        // FRESH arena after flipping it off, not reusing `packed_arena`
        // (already packed at its own build time, permanently). A fresh
        // valve-off arena should collapse back to ~unpacked, since
        // `build_packed_width_panels` returns empty and every width-tile
        // reduce falls back to `run_node_into`'s ordinary dispatch, exactly
        // like `build_static_arena`. The valve is `aarch64`-only (same gate
        // as the packing mechanism itself), so this whole confirmation is a
        // no-op off `aarch64`.
        #[cfg(target_arch = "aarch64")]
        {
            cpu::set_pack_at_plan_time_enabled(false);
            let mut valve_off_arena =
                build_static_arena_with_constants(&lowered.program, &[], &[output], &weights)
                    .expect("build valve-off arena");
            let (valve_off_ms, valve_off_totals) = timed_arm(
                &mut valve_off_arena,
                &weights,
                &input_ids,
                &attention_mask,
                &token_type_ids,
                &input_names,
            );
            cpu::set_pack_at_plan_time_enabled(true);
            let valve_off_ratio = valve_off_ms / unpacked_mean;
            println!(
                "  valve-off (fresh arena built with packing disabled): {valve_off_ms:.4}ms (gemm-calls={}) vs unpacked mean {unpacked_mean:.4}ms -> ratio {valve_off_ratio:.4}x",
                valve_off_totals.reduce_gemm_calls
            );
        }
    }
}
