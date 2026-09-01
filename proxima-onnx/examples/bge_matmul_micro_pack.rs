//! ROW 206 MICRO rung: `bge_matmul_micro`'s own real-shape sweep (QKVO
//! `(M,384)x(384,384)`, FFN-down `(M,1536)x(1536,384)`, `M` in BGE's own
//! real per-sentence lengths `{1, 7, 8, 9}`, ROW 200/201) crossed with ROW
//! 205's own packed-vs-unpacked axis, in BOTH cache regimes: warm (a single
//! `StaticArena` reused every call, same weight buffer, cache-hot after the
//! first touch) and cold (ROW 181's own `ROTATION=64` round-robin, a fresh
//! weight buffer every call, exceeding any plausible on-chip cache tier).
//!
//! ROW 205's own nano rung (`bge_matmul_cache_regime.rs`) measured ONLY the
//! cold form, M in {7,8,9}: packed-cold/arena-cold = 0.754x/0.673x/0.661x
//! (QKVO) and 0.683x/0.582x/0.597x (FFN-down) -- i.e. a 1.33x-1.72x speedup,
//! CoV 0.19%-30.09% across those 18 arms.
//!
//! PRE-REGISTRATION (recorded before this file was ever run):
//!   - cold cells at M=7/8/9 should reproduce ROW 205's own 1.33x-1.72x
//!     packed/unpacked speedup band, same shapes same mechanism (this file
//!     adds nothing new to the cold path beyond M=1).
//!   - M=1 (never measured packed before) is a genuine unknown: the panel
//!     layout still applies since `width_tile_plan` now accepts M=1 (ROW
//!     201's axis-restore fix), but a single output row means the DMA/first-
//!     touch cost `PackedWidthPanels` amortizes over `M` rows amortizes over
//!     only 1 -- predicted speedup SMALLER than the M=7/8/9 band, possibly
//!     near 1.0x, because the fixed packing overhead (build once, still paid
//!     at arena-build time not in the timed loop, so this is about the READ
//!     pattern only) has less row-reuse to pay for itself against.
//!   - warm cells (single arena, same buffer 300x, cache-hot every call
//!     after the first) are predicted to show a SMALLER packed/unpacked
//!     delta than cold, since the unpacked read is already resident in
//!     L1/L2 after the first touch -- packing's sequential-read advantage
//!     over a strided-but-cached read is a much smaller effect than
//!     packing's advantage over a strided COLD read. Predicted range:
//!     0.85x-1.05x (near-tied, possibly a small packed loss from the extra
//!     panel-indexing arithmetic with no cache-miss cost left to hide it
//!     behind).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::{Lowered, lower_graph};
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::{StaticArena, build_static_arena, build_static_arena_with_constants, evaluate_named_with_arena};

const ROTATION: usize = 64;
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
    let lhs = f32_initializer("lhs", vec![1, m as i64, k as i64], lhs_data);
    let rhs = f32_initializer("rhs", vec![k as i64, n as i64], rhs_data);
    let node = NodeProto { input: vec!["lhs", "rhs"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
    let graph = GraphProto {
        node: vec![node],
        name: "micro_pack_graph",
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
        "  {label:<40} ns/call={:>10.1}  CoV={:>5.2}%  GMAC/s={:>8.3}  samples={:?}",
        timed.mean_ns,
        timed.cov_pct,
        gmac_s,
        timed.samples.iter().map(|value| format!("{value:.0}")).collect::<Vec<_>>()
    );
}

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

/// Warm, unpacked: one `StaticArena` (no `constant_inputs`), reused every
/// call -- same `rhs` buffer stays cache-resident from the first call on.
fn warm_unpacked_arm(m: usize, k: usize, n: usize) -> Timed {
    let lowered = build_instance(m, k, n, 0x5000_0000);
    let output = lowered.graph_outputs[0].1;
    let mut arena = build_static_arena(&lowered.program, &[], &[output]).expect("build unpacked arena");
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

/// Warm, packed: same shape as [`warm_unpacked_arm`], `rhs` packed once at
/// `build_static_arena_with_constants` time, same arena reused every call.
fn warm_packed_arm(m: usize, k: usize, n: usize) -> Timed {
    let lowered = build_instance(m, k, n, 0x6000_0000);
    let output = lowered.graph_outputs[0].1;
    let rhs_data = lowered.initializers.iter().find(|(name, _)| name == "rhs").map(|(_, data)| data.as_slice()).expect("rhs initializer present");
    let mut arena = build_static_arena_with_constants(&lowered.program, &[], &[output], &[("rhs", rhs_data)]).expect("build packed arena");
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

/// Cold, unpacked: ROW 205's own `arena_cold_arm` -- `ROTATION` distinct
/// `StaticArena`s (no `constant_inputs`), round-robined so no weight buffer
/// is touched twice within a `ROTATION`-call window.
fn cold_unpacked_arm(m: usize, k: usize, n: usize) -> Timed {
    let instances: Vec<Lowered> = (0..ROTATION).map(|index| build_instance(m, k, n, 0x7000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9)))).collect();
    let mut arenas: Vec<StaticArena> = instances
        .iter()
        .map(|lowered| {
            let output = lowered.graph_outputs[0].1;
            build_static_arena(&lowered.program, &[], &[output]).expect("build unpacked arena")
        })
        .collect();
    let named_per_instance: Vec<NamedInputs<'_>> = instances.iter().map(|lowered| lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect()).collect();

    for index in 0..ROTATION {
        let evaluated = evaluate_named_with_arena(&mut arenas[index], &named_per_instance[index]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    time_calls(|index| {
        let rotation_index = index % ROTATION;
        let evaluated = evaluate_named_with_arena(&mut arenas[rotation_index], &named_per_instance[rotation_index]).expect("timed eval");
        std::hint::black_box(&evaluated);
    })
}

/// Cold, packed: ROW 205's own `packed_cold_arm` -- identical rotation,
/// `rhs` packed once per instance at arena-build time.
fn cold_packed_arm(m: usize, k: usize, n: usize) -> Timed {
    let instances: Vec<Lowered> = (0..ROTATION).map(|index| build_instance(m, k, n, 0x8000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9)))).collect();
    let mut arenas: Vec<StaticArena> = instances
        .iter()
        .map(|lowered| {
            let rhs_data = lowered.initializers.iter().find(|(name, _)| name == "rhs").map(|(_, data)| data.as_slice()).expect("rhs initializer present");
            let output = lowered.graph_outputs[0].1;
            build_static_arena_with_constants(&lowered.program, &[], &[output], &[("rhs", rhs_data)]).expect("build packed arena")
        })
        .collect();
    let named_per_instance: Vec<NamedInputs<'_>> = instances.iter().map(|lowered| lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect()).collect();

    for index in 0..ROTATION {
        let evaluated = evaluate_named_with_arena(&mut arenas[index], &named_per_instance[index]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    time_calls(|index| {
        let rotation_index = index % ROTATION;
        let evaluated = evaluate_named_with_arena(&mut arenas[rotation_index], &named_per_instance[rotation_index]).expect("timed eval");
        std::hint::black_box(&evaluated);
    })
}

fn shape_block(name: &str, m: usize, k: usize, n: usize) {
    let macs = (m * k * n) as f64 / 1e9;
    println!("\n=== {name}: M={m} K={k} N={n} ({macs:.4} GMAC/call) ===");

    let warm_unpacked = warm_unpacked_arm(m, k, n);
    report("warm/unpacked", m, k, n, &warm_unpacked);
    let warm_packed = warm_packed_arm(m, k, n);
    report("warm/packed", m, k, n, &warm_packed);
    let warm_ratio = warm_packed.mean_ns / warm_unpacked.mean_ns;
    println!("    -> warm packed/unpacked: {warm_ratio:.3}x (>1 = packed slower, <1 = packed faster)");

    let cold_unpacked = cold_unpacked_arm(m, k, n);
    report("cold/unpacked", m, k, n, &cold_unpacked);
    let cold_packed = cold_packed_arm(m, k, n);
    report("cold/packed", m, k, n, &cold_packed);
    let cold_ratio = cold_packed.mean_ns / cold_unpacked.mean_ns;
    println!("    -> cold packed/unpacked: {cold_ratio:.3}x (>1 = packed slower, <1 = packed faster)");
}

fn main() {
    println!("bge_matmul_micro_pack: ROW 206 MICRO rung -- packed vs unpacked, warm AND cold cache regime, M in {{1, 7, 8, 9}}");
    println!("PRE-REGISTRATION (see file doc comment): cold M=7/8/9 should reproduce ROW 205's 1.33x-1.72x speedup band; M=1 predicted smaller (less row-reuse to amortize); warm predicted near-tied (0.85x-1.05x), since the unpacked read is already cache-resident.");
    for &m in &[1usize, 7, 8, 9] {
        shape_block("QKVO", m, 384, 384);
        shape_block("FFN-down", m, 1536, 384);
    }
}
