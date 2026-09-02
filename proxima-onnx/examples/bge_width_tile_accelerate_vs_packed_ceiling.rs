//! ROW 209 re-anchor (coordinator mid-task correction): the NANO rung's own
//! `neon` arm (`bge_width_tile_accelerate_nano.rs`) reads the UNPACKED
//! weight buffer through `evaluate_named` with no `StaticArena` -- a much
//! weaker baseline (6.5-20 GMAC/s) than the packed, arena-warmed NEON
//! ceiling a sibling session measured in isolation at these exact shapes:
//! QKVO/FFN-up/FFN-down, M=7/8/9, ~48.0-48.8 GMAC/s (`gemm_width_tile_neon`
//! against `PackedWidthPanels`, quiet box, CoV <1%). This file re-runs the
//! comparison the honest way: PACKED NEON (`build_static_arena_with_constants`,
//! the same packed-panel path the sibling measured) against this session's
//! own Accelerate route, which -- per `try_run_accelerate_width_gemm`'s own
//! doc -- ALWAYS reads the ORIGINAL UNPACKED `raw` buffer (`cblas_sgemm`
//! does its own internal blocking; packed panels are not a valid operand
//! for it). So this is packed-NEON-at-ceiling vs unpacked-read-Accelerate,
//! the real question: does AMX beat a saturated NEON kernel, not a weak one.
//!
//! RE-ANCHORED PRE-REGISTRATION (supersedes the original brief's 11.4
//! GMAC/s / 23%-of-peak framing, which was the IN-GRAPH composed rate, not
//! this kernel's own isolated ceiling): this session's own MICRO rung
//! (`bge_width_tile_accelerate_micro.rs`, unpacked-vs-unpacked, same
//! `StaticArena` machinery) already measured Accelerate at 63.9-85.2 GMAC/s
//! at M=7/8/9 across all three shapes -- ABOVE the 48.0-48.8 GMAC/s packed
//! ceiling in every one of those 9 cells. If that holds under a DIRECT
//! paired measurement against the packed arm specifically (not an
//! across-session comparison), predict Accelerate/packed-NEON ratio > 1.0x
//! (Accelerate GMAC/s exceeds the packed ceiling) at every M=7/8/9 cell,
//! narrowing to a much smaller margin than the unpacked comparison showed
//! (packing recovers roughly the 1.3x-1.7x ROW 205/206 already measured, so
//! the gap should compress from ~2.5-3.2x unpacked to something smaller,
//! but not close entirely). ROW 188's own ties-at-M=1-GEVM finding is out
//! of scope here (M=1 is not part of this re-anchor's cell set -- MICRO
//! already covered M=1 against the unpacked arm and found a clear win,
//! 0.63x-0.77x ratio, i.e. Accelerate faster).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::{Lowered, lower_graph};
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::{
    self, StaticArena, build_static_arena, build_static_arena_with_constants,
    evaluate_named_with_arena,
};

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

fn build_instance(m: usize, k: usize, n: usize, salt: u32) -> Lowered {
    let lhs_data = deterministic_data(m * k, salt);
    let rhs_data = deterministic_data(k * n, salt.wrapping_add(0x1111_1111));
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
        name: "ceiling_graph",
        initializer: vec![lhs, rhs],
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    lower_graph(&graph).expect("lower synthetic MatMul")
}

struct Timed {
    mean_ns: f64,
    cov_pct: f64,
    samples: Vec<f64>,
}

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

fn time_calls(arena: &mut StaticArena, named: &NamedInputs<'_>) -> Timed {
    for _ in 0..50 {
        let evaluated = evaluate_named_with_arena(arena, named).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }
    let mut ns_per_call_per_repeat = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = std::time::Instant::now();
        for _ in 0..CALLS_PER_REPEAT {
            let evaluated = evaluate_named_with_arena(arena, named).expect("timed eval");
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

fn packed_neon_arm(m: usize, k: usize, n: usize) -> Timed {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cpu::set_accelerate_gemm_enabled(false);
    let lowered = build_instance(m, k, n, 0x6000_0000);
    let output = lowered.graph_outputs[0].1;
    let rhs_data = lowered
        .initializers
        .iter()
        .find(|(name, _)| name == "rhs")
        .map(|(_, data)| data.as_slice())
        .expect("rhs initializer present");
    let mut arena =
        build_static_arena_with_constants(&lowered.program, &[], &[output], &[("rhs", rhs_data)])
            .expect("build packed arena");
    let named: NamedInputs<'_> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    time_calls(&mut arena, &named)
}

fn accelerate_arm(m: usize, k: usize, n: usize) -> Timed {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cpu::set_accelerate_gemm_enabled(true);
    let lowered = build_instance(m, k, n, 0x7000_0000);
    let output = lowered.graph_outputs[0].1;
    let mut arena =
        build_static_arena(&lowered.program, &[], &[output]).expect("build unpacked arena");
    let named: NamedInputs<'_> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    time_calls(&mut arena, &named)
}

fn report(label: &str, m: usize, k: usize, n: usize, timed: &Timed) -> f64 {
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
    gmac_s
}

fn shape_block(name: &str, m: usize, k: usize, n: usize) {
    let macs = (m * k * n) as f64 / 1e9;
    println!("\n=== {name}: M={m} K={k} N={n} ({macs:.4} GMAC/call) ===");
    let packed = packed_neon_arm(m, k, n);
    let packed_gmac = report("packed-neon", m, k, n, &packed);
    let accelerate = accelerate_arm(m, k, n);
    let accelerate_gmac = report("accelerate", m, k, n, &accelerate);
    let ratio = accelerate_gmac / packed_gmac;
    println!(
        "    -> accelerate/packed-neon GMAC/s ratio: {ratio:.3}x (>1 = accelerate beats the packed ceiling)"
    );
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (hits, declined) = cpu::accelerate_gemm_totals();
        println!("    -> accelerate_gemm_totals() cumulative: hits={hits} declined={declined}");
    }
    let outcome = if ratio > 1.0 { "HIT" } else { "MISS" };
    println!("    -> re-anchored prediction: ratio > 1.0x -> {outcome}");
}

fn main() {
    println!(
        "bge_width_tile_accelerate_vs_packed_ceiling: re-anchor probe -- packed NEON (sibling's own ~48.0-48.8 GMAC/s ceiling shape) vs this session's Accelerate route (unpacked raw read), BGE real shapes, M in {{7,8,9}}"
    );
    println!(
        "RE-ANCHORED PRE-REGISTRATION (see file doc comment): predict Accelerate/packed-NEON ratio > 1.0x at every cell."
    );
    for &m in &[7usize, 8, 9] {
        shape_block("QKVO", m, 384, 384);
        shape_block("FFN-up", m, 384, 1536);
        shape_block("FFN-down", m, 1536, 384);
    }
}
