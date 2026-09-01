//! ROW 209 MICRO rung: the NANO rung's cells (`bge_width_tile_accelerate_nano.rs`)
//! through the real plan machinery -- `StaticArena` (mirrors
//! `bge_matmul_micro_pack.rs`'s own warm-arm shape), paired Accelerate-on vs
//! Accelerate-off, interleaved per repeat so a monotonic host-load drift
//! cannot bias one arm. `accelerate_gemm_totals()` is asserted non-zero on
//! the accelerate arm -- N==0 hits is RED, not green, same discipline every
//! other rung in this session applies.
//!
//! PRE-REGISTRATION (recorded before this file was ever run, ONE rung ahead
//! of NANO's own cells only): NANO measured accelerate/neon ratios of
//! 0.501x-0.904x across all 12 (shape, M) cells, HIT at every cell against
//! its own per-cell prediction. Through the real `StaticArena` plan
//! machinery the per-call ratio should land in the SAME band (same kernel,
//! same shapes, only the surrounding call machinery differs, and that
//! machinery is identical between the two arms) -- predicted 0.50x-0.95x at
//! every cell, with M=1 nearest the top of that band per NANO's own
//! narrower M=1 margin.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::{Lowered, lower_graph};
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::{self, StaticArena, build_static_arena, evaluate_named_with_arena};

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
    TensorProto { dims, data_type: 1, float_data: data, name, ..TensorProto::default() }
}

fn build_instance(m: usize, k: usize, n: usize, salt: u32) -> Lowered {
    let lhs_data = deterministic_data(m * k, salt);
    let rhs_data = deterministic_data(k * n, salt.wrapping_add(0x1111_1111));
    let lhs = f32_initializer("lhs", vec![m as i64, k as i64], lhs_data);
    let rhs = f32_initializer("rhs", vec![k as i64, n as i64], rhs_data);
    let node = NodeProto { input: vec!["lhs", "rhs"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
    let graph = GraphProto {
        node: vec![node],
        name: "micro_accelerate_graph",
        initializer: vec![lhs, rhs],
        output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
        ..GraphProto::default()
    };
    lower_graph(&graph).expect("lower synthetic MatMul")
}

struct Timed {
    mean_ns: f64,
    cov_pct: f64,
    samples: Vec<f64>,
}

fn time_calls<F: FnMut(usize)>(mut call: F) -> Timed {
    let mut ns_per_call_per_repeat = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        for index in 0..CALLS_PER_REPEAT {
            call(index);
        }
        let elapsed = start.elapsed();
        ns_per_call_per_repeat.push(elapsed.as_nanos() as f64 / CALLS_PER_REPEAT as f64);
    }
    let mean = ns_per_call_per_repeat.iter().sum::<f64>() / ns_per_call_per_repeat.len() as f64;
    let variance = ns_per_call_per_repeat.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / ns_per_call_per_repeat.len() as f64;
    let cov = variance.sqrt() / mean * 100.0;
    Timed { mean_ns: mean, cov_pct: cov, samples: ns_per_call_per_repeat }
}

fn report(label: &str, m: usize, k: usize, n: usize, timed: &Timed) {
    let macs = (m * k * n) as f64;
    let gmac_s = macs / (timed.mean_ns / 1e9) / 1e9;
    println!(
        "  {label:<12} ns/call={:>10.1}  CoV={:>5.2}%  GMAC/s={:>8.3}  samples={:?}",
        timed.mean_ns,
        timed.cov_pct,
        gmac_s,
        timed.samples.iter().map(|value| format!("{value:.0}")).collect::<Vec<_>>()
    );
}

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

fn arm(m: usize, k: usize, n: usize, salt: u32, accelerate: bool) -> Timed {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cpu::set_accelerate_gemm_enabled(accelerate);
    let _ = accelerate;

    let lowered = build_instance(m, k, n, salt);
    let output = lowered.graph_outputs[0].1;
    let mut arena: StaticArena = build_static_arena(&lowered.program, &[], &[output]).expect("build arena");
    let named: NamedInputs<'_> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    for _ in 0..50 {
        let evaluated = evaluate_named_with_arena(&mut arena, &named).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }
    time_calls(|_index| {
        let evaluated = evaluate_named_with_arena(&mut arena, &named).expect("timed eval");
        std::hint::black_box(&evaluated);
    })
}

fn shape_block(name: &str, m: usize, k: usize, n: usize) {
    let macs = (m * k * n) as f64 / 1e9;
    println!("\n=== {name}: M={m} K={k} N={n} ({macs:.4} GMAC/call) ===");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let (hits_before, _declined_before) = cpu::accelerate_gemm_totals();

    // interleaved: neon-then-accelerate on even repeats, accelerate-then-neon
    // on odd repeats -- REPEATS is folded into `arm`'s own 5-repeat timing
    // loop already, so this file interleaves at the ARM level (whole-arm
    // ordering), same discipline `bge_epilogue_profile_pack.rs` applies at
    // its own per-run granularity.
    let (neon, accelerate) = if m % 2 == 1 {
        let neon = arm(m, k, n, 0x2000_0000, false);
        let accelerate = arm(m, k, n, 0x3000_0000, true);
        (neon, accelerate)
    } else {
        let accelerate = arm(m, k, n, 0x3000_0000, true);
        let neon = arm(m, k, n, 0x2000_0000, false);
        (neon, accelerate)
    };
    report("neon", m, k, n, &neon);
    report("accelerate", m, k, n, &accelerate);
    let ratio = accelerate.mean_ns / neon.mean_ns;
    println!("    -> accelerate/neon: {ratio:.3}x (<1 = accelerate faster, >1 = accelerate slower)");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (hits_after, declined_after) = cpu::accelerate_gemm_totals();
        let engagement = hits_after - hits_before;
        println!("    -> accelerate_gemm_totals() delta this cell: hits={engagement} declined_cumulative={declined_after}");
        assert!(engagement > 0, "engagement proof: accelerate arm must record at least one hit, got 0");
    }

    let outcome = if (0.50..=0.95).contains(&ratio) { "HIT" } else { "MISS" };
    println!("    -> pre-registered prediction: 0.50x-0.95x -> {outcome}");
}

fn main() {
    println!("bge_width_tile_accelerate_micro: ROW 209 MICRO rung -- real StaticArena plan machinery, paired interleaved arms, BGE real shapes, M in {{1,7,8,9}}");
    println!("PRE-REGISTRATION (see file doc comment): ratio in 0.50x-0.95x at every cell, engagement (accelerate_gemm_totals hits) > 0 required.");
    for &m in &[1usize, 7, 8, 9] {
        shape_block("QKVO", m, 384, 384);
        shape_block("FFN-up", m, 384, 1536);
        shape_block("FFN-down", m, 1536, 384);
    }
}
