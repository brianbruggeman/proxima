//! Measures two numbers the CPU->GPU cut-boundary decision needs and that
//! whole-model decode instrumentation cannot isolate on its own: the FIXED
//! cost of one Metal command-buffer `commit()`+`waitUntilCompleted()` when
//! the encoded work is as small as this algebra can express, and the
//! readback cost of a single crossing-payload-sized output (1024 f32
//! elements = 4096 bytes, the qwen3 0.6B checkpoint's `embedding_length`,
//! confirmed against the running program by `examples/gguf_generate.rs`'s
//! own metadata dump rather than assumed).
//!
//! Same production entry point the decode loop uses per token
//! (`omega::execute_plan`, one command buffer for the whole program, see
//! `omega/src/metal.rs`'s `execute_plan` doc) — never the per-op-timed
//! diagnostic variant, so this floor is the SAME code path whole-model
//! decode pays, just with a program small enough that its own compute cost
//! is negligible next to the submit/wait/readback fixed costs.
//!
//! `membw_probe.rs`'s two-size marginal technique is for isolating a
//! streaming kernel's true GB/s from per-call overhead; this probe wants
//! the OPPOSITE number -- the per-call overhead itself -- so it reports the
//! small-size numbers directly rather than subtracting them out.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", feature = "cpu", feature = "instrument", target_os = "macos"))]
fn cov_percent(samples: &[f64]) -> f64 {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    (variance.sqrt() / mean) * 100.0
}

#[cfg(all(feature = "metal", feature = "cpu", feature = "instrument", target_os = "macos"))]
fn run() {
    use std::hint::black_box;
    use std::time::Instant;

    use proxima_tensor::instrument::ticks_to_nanos;
    use proxima_tensor::{DType, Extent, IndexMap, NodeId, Op, QuantizedBlock, ScalarOp, append, map};

    fn elementwise_add_program(elements: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(elements)],
                name: Some("a".into()),
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(elements)],
                name: Some("b".into()),
            },
        );
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(1, &[0]))),
                    (rhs, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        (program, sum)
    }

    fn measure(elements: u32, runs: usize) -> (Vec<f64>, omega::metal::MetalStageTotals) {
        let a: Vec<f32> = (0..elements).map(|index| (index as f32) * 1e-3).collect();
        let b: Vec<f32> = (0..elements).map(|index| (index as f32) * -1e-3).collect();
        let (program, sum) = elementwise_add_program(elements);
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved = omega::plan(
            &program,
            &[],
            &blocks,
            &[sum],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("probe plans");
        // warm-up: pipeline compile happens on the first call
        // (`pipeline_misses`), never on steady-state decode; discard it and
        // the counters it left behind before the timed loop.
        omega::execute_plan(&resolved, &blocks).expect("probe warms up");
        let _ = omega::metal::metal_stage_totals();

        let mut samples = Vec::with_capacity(runs);
        let mut accumulated = omega::metal::MetalStageTotals::default();
        for _ in 0..runs {
            let started = Instant::now();
            let out = omega::execute_plan(&resolved, &blocks).expect("probe executes");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            black_box(&out);
            let stage = omega::metal::metal_stage_totals();
            accumulated.gpu_exec_calls += stage.gpu_exec_calls;
            accumulated.gpu_exec_ticks += stage.gpu_exec_ticks;
            accumulated.readback_calls += stage.readback_calls;
            accumulated.readback_ticks += stage.readback_ticks;
            accumulated.readback_bytes += stage.readback_bytes;
            accumulated.encode_dispatch_ticks += stage.encode_dispatch_ticks;
            accumulated.op_setup_ticks += stage.op_setup_ticks;
        }
        (samples, accumulated)
    }

    const RUNS: usize = 30;
    let ticks_to_ms = |ticks: u64, calls: u64| -> f64 {
        if calls == 0 {
            return 0.0;
        }
        // same conversion `generate.rs`'s own `token_breakdown_metal` line
        // applies (`ticks_to_nanos` at the print edge, per
        // `proxima_tensor::instrument`'s own doc: never converted per call).
        (ticks_to_nanos(ticks) as f64 / calls as f64) / 1e6
    };

    println!(
        "cut_boundary_submit_floor: production omega::execute_plan (ONE command buffer, \
         commit+waitUntilCompleted once), {RUNS} runs per size, min/mean/CoV over wall-clock \
         Instant, plus the SAME calls' own instrument counters averaged per call"
    );

    for elements in [1_u32, 1024_u32] {
        let (mut samples, stage) = measure(elements, RUNS);
        samples.sort_by(f64::total_cmp);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let bytes_out = elements as usize * 4;
        println!(
            "  elements={elements} bytes_out={bytes_out} min_ms={:.4} mean_ms={:.4} p95_ms={:.4} CoV={:.2}% \
             samples_ms={samples:?}",
            samples[0],
            mean,
            samples[(samples.len() * 95 / 100).min(samples.len() - 1)],
            cov_percent(&samples),
        );
        println!(
            "    instrument counters (mean over {RUNS} calls): gpu_exec_ms={:.4} readback_ms={:.4} \
             readback_bytes_per_call={} encode_dispatch_ms={:.4} op_setup_ms={:.4}",
            ticks_to_ms(stage.gpu_exec_ticks, stage.gpu_exec_calls),
            ticks_to_ms(stage.readback_ticks, stage.gpu_exec_calls),
            stage.readback_bytes / RUNS as u64,
            ticks_to_ms(stage.encode_dispatch_ticks, stage.gpu_exec_calls),
            ticks_to_ms(stage.op_setup_ticks, stage.gpu_exec_calls),
        );
    }
}

#[cfg(not(all(feature = "metal", feature = "cpu", feature = "instrument", target_os = "macos")))]
fn run() {
    println!(
        "cut_boundary_submit_floor requires --features metal,cpu,instrument on macOS; skipped"
    );
}

fn main() {
    run();
}
