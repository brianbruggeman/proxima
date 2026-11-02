//! ROW 376's timed cells: the real openchat decode shape (32 query heads, 8 kv
//! heads, head_dim 128, GQA group 4) through
//! `gqa_single_range_cached_forward_program`, timed two ways on one plan --
//! a whole-plan single-command-buffer batched total divided by `LAYERS`, and a
//! per-op-timed run whose records are selected by `OpGpuTiming::kind` and
//! count-asserted against `LAYERS` on every repeat. The fixture is shared with
//! `tests/row_376_cached_attention_batched.rs`'s parity checks.
//!
//! Select a cell by name: `a_context40`, `b_context512`, or `c_context4096`.
//! Run with
//! `cargo run -p proxima-omega --release --features metal,cached-attention-streaming,instrument --example row_376_cached_attention_timed_cells -- <cell>`

#![allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
#[path = "../tests/support/mod.rs"]
mod support;

#[cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
#[path = "../tests/support/row_376_fixture.rs"]
mod fixture;

#[cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
mod cells {
    use std::time::Instant;

    use proxima_tensor::NumericPolicy;

    use crate::fixture::{HEAD_DIM, KV_HEADS, LAYERS, build_fixture};
    use crate::support::as_named_blocks;

    const REPEATS: usize = 7;
    const ATTENTION_KIND: &str = "cached_attention";

    struct Stats {
        mean_us: f64,
        median_us: f64,
        min_us: f64,
        max_us: f64,
        cov_pct: f64,
    }

    fn stats(samples: &[f64]) -> Stats {
        let len = samples.len();
        let mut sorted = samples.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).expect("no NaN sample"));
        let mean = samples.iter().sum::<f64>() / len as f64;
        let median = if len.is_multiple_of(2) {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        };
        let variance = samples
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / len as f64;
        let cov_pct = if mean.abs() > f64::MIN_POSITIVE {
            100.0 * variance.sqrt() / mean
        } else {
            0.0
        };
        Stats {
            mean_us: mean,
            median_us: median,
            min_us: sorted[0],
            max_us: sorted[len - 1],
            cov_pct,
        }
    }

    /// Runs one context-length cell. Two independent timed loops share the
    /// SAME `plan`:
    ///
    /// - the BATCHED loop calls [`omega::metal::execute_plan_named_timed`]
    ///   (one shared command buffer for the whole 32-layer plan) and divides
    ///   its total `gpu_ns` by `LAYERS` -- this is the reported per-dispatch
    ///   number, and the KV bytes/GB/s below are derived from its median.
    /// - the VALIDATION loop calls `execute_plan_named_op_timed` (one command
    ///   buffer per op) on every repeat, filters the returned records to
    ///   `kind == "cached_attention"` and asserts there are exactly `LAYERS`
    ///   of them -- every summed record is identified by its OWN kind field
    ///   from THIS SAME execution, never by a position borrowed from a
    ///   different bind.
    pub fn run_cell(label: &str, context_length: u64) {
        let (program, symbols, roots, named) = build_fixture(context_length);
        let named_blocks = as_named_blocks(&named);

        let plan = omega::plan_named(
            &program,
            &symbols,
            &named_blocks,
            &roots,
            NumericPolicy::llama_relaxed(),
        )
        .expect("metal plans the real-shape fixture");

        omega::metal::execute_plan_named(&plan, &named_blocks).expect("plain warm-up run");

        omega::metal::execute_plan_named_timed(&plan, &named_blocks).expect("batched warm-up run");
        let mut batched_us_per_dispatch = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            let (_evaluated, total_gpu_ns) =
                omega::metal::execute_plan_named_timed(&plan, &named_blocks)
                    .expect("batched timed run");
            let wall_us = started.elapsed().as_secs_f64() * 1e6;
            batched_us_per_dispatch.push(total_gpu_ns as f64 / LAYERS as f64 / 1e3);
            println!(
                "ROW 376 {label} batched raw wall_us={wall_us:.2} total_gpu_ns={total_gpu_ns}"
            );
        }
        let batched_stat = stats(&batched_us_per_dispatch);

        let (_evaluated, warm_op_timings) =
            omega::metal::execute_plan_named_op_timed(&plan, &named_blocks, None)
                .expect("per-op warm-up run");
        drop(warm_op_timings);
        let mut per_op_us_per_dispatch = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let (_evaluated, timings) =
                omega::metal::execute_plan_named_op_timed(&plan, &named_blocks, None)
                    .expect("per-op validation run");
            let attention: Vec<&omega::metal::OpGpuTiming> = timings
                .iter()
                .filter(|timing| timing.kind == ATTENTION_KIND)
                .collect();
            assert_eq!(
                attention.len(),
                LAYERS as usize,
                "{label}: expected exactly {LAYERS} records with kind==\"{ATTENTION_KIND}\", found {}",
                attention.len()
            );
            let attention_gpu_ns: u64 = attention.iter().map(|timing| timing.gpu_ns).sum();
            per_op_us_per_dispatch.push(attention_gpu_ns as f64 / LAYERS as f64 / 1e3);
        }
        let per_op_stat = stats(&per_op_us_per_dispatch);

        let kv_bytes_per_dispatch = context_length * KV_HEADS as u64 * HEAD_DIM as u64 * 2 * 4;
        let effective_gbps = kv_bytes_per_dispatch as f64 / (batched_stat.median_us * 1e-6) / 1e9;
        println!(
            "ROW 376 {label} context_length={context_length} dispatches={LAYERS} \
         batched_us_per_dispatch: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}% \
         per_op_timed_kind_validated_us_per_dispatch: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}% \
         kv_bytes_per_dispatch={kv_bytes_per_dispatch} effective_gbps_from_batched={effective_gbps:.2}",
            batched_stat.mean_us,
            batched_stat.median_us,
            batched_stat.min_us,
            batched_stat.max_us,
            batched_stat.cov_pct,
            per_op_stat.mean_us,
            per_op_stat.median_us,
            per_op_stat.min_us,
            per_op_stat.max_us,
            per_op_stat.cov_pct,
        );
    }
}

#[cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
fn main() -> std::process::ExitCode {
    let Some(cell) = std::env::args().nth(1) else {
        eprintln!(
            "usage: row_376_cached_attention_timed_cells <a_context40|b_context512|c_context4096>"
        );
        return std::process::ExitCode::FAILURE;
    };
    match cell.as_str() {
        "a_context40" => cells::run_cell("a_context40", 40),
        "b_context512" => cells::run_cell("b_context512", 512),
        "c_context4096" => cells::run_cell("c_context4096", 4096),
        unknown => {
            eprintln!("unknown cell `{unknown}`");
            return std::process::ExitCode::FAILURE;
        }
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(not(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
)))]
fn main() {
    println!(
        "row_376_cached_attention_timed_cells requires --features metal,cached-attention-streaming,instrument on macOS; skipped"
    );
}
