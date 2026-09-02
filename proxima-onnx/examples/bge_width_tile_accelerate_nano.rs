//! ROW 209 NANO rung: `cblas_sgemm` (via the new width-tile Accelerate gate
//! arm, `cpu.rs`'s `try_run_accelerate_width_gemm`) vs the NEON width tile
//! (`run_width_tile_neon`), at BGE's own real shapes, in isolation.
//!
//! Shapes: QKVO `(M,384)x(384,384)`, FFN-up `(M,384)x(384,1536)`, FFN-down
//! `(M,1536)x(1536,384)`; `M` in BGE's own real per-sentence lengths
//! `{1, 7, 8, 9}` (`docs/discipline.md` ROW 199/200). The `[M,K]` unbatched
//! shape (no leading batch axis) is used throughout -- ROW 198/200 already
//! established this is required for `width_tile_plan`'s own
//! `leading_output_axes.len() == 1` gate to fire at all; the batched
//! `[1,M,K]` shape never reaches either tile.
//!
//! PRE-REGISTRATION (recorded before this file was ever run):
//!   ROW 189 measured Accelerate 1.97x-5.51x over NEON at mnist conv's
//!   M=8-24 shapes, and 467 GFLOP/s at an M=64 synthetic; ROW 188 measured a
//!   TIE at fc's M=1 GEVM shapes. Same mechanism (one `cblas_sgemm` call
//!   replacing a caller-side tiled loop) now wired to the width-tile route
//!   instead of the dot-tile route, so the SAME shape-dependent prediction
//!   applies: WIN at M=7/8/9 (AMX amortizes its own call overhead over
//!   enough rows), TIE-OR-LOSS at M=1 (a single output row does not amortize
//!   `cblas_sgemm`'s own call/dispatch overhead against NEON's zero-call-
//!   overhead inline tile). A MISS at this rung kills the climb -- no MICRO
//!   or MILLI measurement of a dead mechanism.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::evaluate_named;

const CALLS_PER_REPEAT: usize = 300;
const REPEATS: usize = 5;

fn deterministic_data(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let mixed = (index as u32).wrapping_mul(2654435761).wrapping_add(salt);
            (mixed as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn f32_initializer(name: &'static str, dims: Vec<i64>, data: Vec<f32>) -> TensorProto<'static> {
    TensorProto {
        dims,
        data_type: 1,
        float_data: data,
        name,
        ..TensorProto::default()
    }
}

struct Timed {
    mean_ns: f64,
    cov_pct: f64,
    samples: Vec<f64>,
}

fn run_arm(m: usize, k: usize, n: usize, accelerate: bool) -> Timed {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    proxima_tensor::cpu::set_accelerate_gemm_enabled(accelerate);
    let _ = accelerate;

    let lhs_data = deterministic_data(m * k, 0x1234_5678);
    let rhs_data = deterministic_data(k * n, 0x9abc_def0);
    let lhs = f32_initializer("lhs", vec![m as i64, k as i64], lhs_data);
    let rhs = f32_initializer("rhs", vec![k as i64, n as i64], rhs_data);
    let node = NodeProto {
        input: vec!["lhs", "rhs"],
        output: vec!["y"],
        op_type: "MatMul",
        name: "matmul",
        ..NodeProto::default()
    };
    let graph = GraphProto {
        node: vec![node],
        name: "nano_accelerate_graph",
        initializer: vec![lhs, rhs],
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    let lowered = lower_graph(&graph).expect("lower synthetic MatMul");
    let named: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let output = lowered.graph_outputs[0].1;

    for _ in 0..50 {
        let evaluated =
            evaluate_named(&lowered.program, &[], &named, &[output]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    let mut ns_per_call_per_repeat = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        for _ in 0..CALLS_PER_REPEAT {
            let evaluated =
                evaluate_named(&lowered.program, &[], &named, &[output]).expect("timed eval");
            std::hint::black_box(&evaluated);
        }
        let elapsed = start.elapsed();
        ns_per_call_per_repeat.push(elapsed.as_nanos() as f64 / CALLS_PER_REPEAT as f64);
    }
    let mean = ns_per_call_per_repeat.iter().sum::<f64>() / ns_per_call_per_repeat.len() as f64;
    let variance = ns_per_call_per_repeat
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / ns_per_call_per_repeat.len() as f64;
    let cov = variance.sqrt() / mean * 100.0;
    Timed {
        mean_ns: mean,
        cov_pct: cov,
        samples: ns_per_call_per_repeat,
    }
}

fn report(label: &str, m: usize, k: usize, n: usize, timed: &Timed) {
    let macs = (m * k * n) as f64;
    let gmac_s = macs / (timed.mean_ns / 1e9) / 1e9;
    println!(
        "  {label:<12} ns/call={:>10.1}  CoV={:>5.2}%  GMAC/s={:>8.3}  samples={:?}",
        timed.mean_ns,
        timed.cov_pct,
        gmac_s,
        timed
            .samples
            .iter()
            .map(|value| format!("{value:.0}"))
            .collect::<Vec<_>>()
    );
}

fn shape_block(name: &str, m: usize, k: usize, n: usize) {
    let macs = (m * k * n) as f64 / 1e9;
    println!("\n=== {name}: M={m} K={k} N={n} ({macs:.4} GMAC/call) ===");
    let neon = run_arm(m, k, n, false);
    report("neon", m, k, n, &neon);
    let accelerate = run_arm(m, k, n, true);
    report("accelerate", m, k, n, &accelerate);
    let ratio = accelerate.mean_ns / neon.mean_ns;
    println!(
        "    -> accelerate/neon: {ratio:.3}x (<1 = accelerate faster, >1 = accelerate slower)"
    );
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (hits, declined) = proxima_tensor::cpu::accelerate_gemm_totals();
        println!("    -> accelerate_gemm_totals() cumulative: hits={hits} declined={declined}");
    }
    let prediction = if m == 1 {
        "TIE-OR-LOSS (ratio near or above 1.0)"
    } else {
        "WIN (ratio well below 1.0)"
    };
    let outcome = match m {
        1 => {
            if ratio <= 1.15 {
                "HIT"
            } else {
                "MISS"
            }
        }
        _ => {
            if ratio < 0.95 {
                "HIT"
            } else {
                "MISS"
            }
        }
    };
    println!("    -> pre-registered prediction: {prediction} -> {outcome}");
}

fn main() {
    println!(
        "bge_width_tile_accelerate_nano: ROW 209 NANO rung -- cblas_sgemm vs NEON width tile, BGE real shapes, M in {{1,7,8,9}}"
    );
    println!(
        "PRE-REGISTRATION (see file doc comment): WIN (ratio<0.95) at M=7/8/9, TIE-OR-LOSS (ratio<=1.15) at M=1. A MISS here kills the climb."
    );
    for &m in &[1usize, 7, 8, 9] {
        shape_block("QKVO", m, 384, 384);
        shape_block("FFN-up", m, 384, 1536);
        shape_block("FFN-down", m, 1536, 384);
    }
}
