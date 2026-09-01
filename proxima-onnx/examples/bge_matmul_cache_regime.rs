//! BGE ladder iteration 6, H1 (CACHE REGIME) nano cell.
//!
//! ROW 200/201's own `bge_matmul_micro` warm arm reuses the SAME weight
//! buffer 300x per timed repeat -- cache-hot on every call after the first.
//! In-graph, BGE's 96 real weight matrices (~132MB f32 total, ROW 201's own
//! citation) stream through ONCE per sentence -- every weight read is cold.
//!
//! This cell adds a `cold` arm: `ROTATION` independently-lowered graph
//! instances per shape, each carrying its own distinct weight buffer, round-
//! robined across the timed loop so no weight buffer is touched twice within
//! any window smaller than `ROTATION` calls -- the ROW 181 round-robin
//! method. `ROTATION=64` sized so total rotated bytes per shape (FFN-down:
//! 64 x 2.25MB = ~144MB, QKVO: 64 x 0.5625MB = ~36MB) exceeds any plausible
//! on-chip cache tier, not just L1/L2.
//!
//! Both arms report ns/call, GMAC/s, and effective GB/s (triad bytes moved /
//! time), so the cold arm's bandwidth can be checked against the machine's
//! own measured streaming ceiling (`rooflines.md`: 69.95 GB/s same-shape
//! triad / 81.21 GB/s DRAM-bound triad, both ROW 176).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::{Lowered, lower_graph};
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::NodeId;
use proxima_tensor::cpu::{StaticArena, build_static_arena_with_constants, evaluate_named, evaluate_named_with_arena};

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
        name: "cache_regime_graph",
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
    let triad_bytes = ((k * n) + (m * k) + (m * n)) as f64 * 4.0;
    let gb_s = triad_bytes / (timed.mean_ns / 1e9) / 1e9;
    println!(
        "  {label:<48} ns/call={:>10.1}  CoV={:>5.2}%  GMAC/s={:>8.3}  triad-GB/s={:>7.3}  samples={:?}",
        timed.mean_ns,
        timed.cov_pct,
        gmac_s,
        gb_s,
        timed.samples.iter().map(|value| format!("{value:.0}")).collect::<Vec<_>>()
    );
}

fn warm_arm(m: usize, k: usize, n: usize) -> Timed {
    let lowered = build_instance(m, k, n, 0x1000_0000);
    let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    let output = lowered.graph_outputs[0].1;
    for _ in 0..50 {
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }
    time_calls(|_index| {
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("timed eval");
        std::hint::black_box(&evaluated);
    })
}

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

fn cold_arm(m: usize, k: usize, n: usize) -> Timed {
    let instances: Vec<Lowered> = (0..ROTATION).map(|index| build_instance(m, k, n, 0x2000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9)))).collect();
    let named_and_output: Vec<(NamedInputs<'_>, NodeId)> = instances
        .iter()
        .map(|lowered| {
            let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
            (named, lowered.graph_outputs[0].1)
        })
        .collect();

    // untimed single warm-up pass over the WHOLE rotation set -- forces
    // allocation/compile paths without leaving any single buffer resident,
    // since ROTATION x shape-bytes already exceeds cache before the timed
    // loop starts.
    for index in 0..ROTATION {
        let (named, output) = &named_and_output[index];
        let evaluated = evaluate_named(&instances[index].program, &[], named, &[*output]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    time_calls(|index| {
        let rotation_index = index % ROTATION;
        let (named, output) = &named_and_output[rotation_index];
        let evaluated = evaluate_named(&instances[rotation_index].program, &[], named, &[*output]).expect("timed eval");
        std::hint::black_box(&evaluated);
    })
}

/// Law 6∘5 nano cell: the SAME ROW 181 round-robin `cold_arm` uses (`rhs`
/// never touched twice within a `ROTATION`-call window), except each
/// instance's `rhs` (the weight, `[k,n]`) is packed once at
/// `build_static_arena_with_constants` time, off the timed loop, into the
/// width-tile kernel's own panel layout (`docs/rewrite-algebra.md` section
/// 6). Pre-registration (recorded before this arm was ever run): if H1
/// (cache regime / first-touch latency) is the mechanism ROW 203 measured,
/// packed-cold's effective GB/s should sit closer to `warm_arm`'s than to
/// `cold_arm`'s 6.6-10.3 GB/s, since a packed read is sequential instead of
/// one page-fault-costed touch per weight row.
fn packed_cold_arm(m: usize, k: usize, n: usize) -> Timed {
    let instances: Vec<Lowered> = (0..ROTATION)
        .map(|index| build_instance(m, k, n, 0x3000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9))))
        .collect();
    let mut arenas: Vec<StaticArena> = instances
        .iter()
        .map(|lowered| {
            let rhs_data = lowered.initializers.iter().find(|(name, _)| name == "rhs").map(|(_, data)| data.as_slice()).expect("rhs initializer present");
            let output = lowered.graph_outputs[0].1;
            build_static_arena_with_constants(&lowered.program, &[], &[output], &[("rhs", rhs_data)]).expect("build packed arena")
        })
        .collect();
    let named_per_instance: Vec<NamedInputs<'_>> = instances
        .iter()
        .map(|lowered| lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect())
        .collect();

    // untimed single warm-up pass over the WHOLE rotation set, same shape
    // `cold_arm`'s own warm-up has -- forces any first-call-only setup
    // without leaving any single buffer resident.
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

/// Isolates law 6∘5's OWN contribution from `StaticArena`'s already-landed
/// bind+alloc amortization (ROW 164/175): identical rotation and identical
/// `evaluate_named_with_arena` call path as [`packed_cold_arm`], but built
/// via plain `build_static_arena` (no `constant_inputs`), so `rhs` is never
/// packed. Without this arm, `packed_cold_arm` vs `cold_arm` conflates two
/// effects (arena reuse skips per-call `shape::infer`/`bind::bind`, packing
/// fixes the weight's read stride) into one number.
fn arena_cold_arm(m: usize, k: usize, n: usize) -> Timed {
    let instances: Vec<Lowered> = (0..ROTATION)
        .map(|index| build_instance(m, k, n, 0x4000_0000u32.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9))))
        .collect();
    let mut arenas: Vec<StaticArena> = instances
        .iter()
        .map(|lowered| {
            let output = lowered.graph_outputs[0].1;
            proxima_tensor::cpu::build_static_arena(&lowered.program, &[], &[output]).expect("build unpacked arena")
        })
        .collect();
    let named_per_instance: Vec<NamedInputs<'_>> = instances
        .iter()
        .map(|lowered| lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect())
        .collect();

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
    let rotated_mib = (k * n * 4 * ROTATION) as f64 / (1024.0 * 1024.0);
    println!("\n=== {name}: M={m} K={k} N={n} ({macs:.4} GMAC/call, cold rotation={ROTATION} x weight = {rotated_mib:.1} MiB) ===");
    let warm = warm_arm(m, k, n);
    report("warm (existing form, same buffer 300x)", m, k, n, &warm);
    let cold = cold_arm(m, k, n);
    report("cold (ROW 181 round-robin, 64 distinct weights)", m, k, n, &cold);
    let slowdown = cold.mean_ns / warm.mean_ns;
    println!("    -> cold/warm slowdown: {slowdown:.3}x");
    let arena_cold = arena_cold_arm(m, k, n);
    report("arena-cold (StaticArena, rhs unpacked, isolates arena reuse)", m, k, n, &arena_cold);
    let packed_cold = packed_cold_arm(m, k, n);
    report("packed-cold (law 6∘5, rhs packed at plan time)", m, k, n, &packed_cold);
    let packed_vs_cold = packed_cold.mean_ns / cold.mean_ns;
    let packed_vs_arena_cold = packed_cold.mean_ns / arena_cold.mean_ns;
    let packed_vs_warm = packed_cold.mean_ns / warm.mean_ns;
    println!(
        "    -> packed-cold/cold: {packed_vs_cold:.3}x   packed-cold/arena-cold (law 6∘5's OWN delta): {packed_vs_arena_cold:.3}x   packed-cold/warm: {packed_vs_warm:.3}x"
    );
}

/// H1 pre-registration (per the task brief, recorded here so the printed
/// output carries it alongside the measurement): if H1 (cache regime) is the
/// dominant mechanism behind the in-graph 10.7 GMAC/s vs isolated 26-40
/// GMAC/s gap, the cold arm should land NEAR the in-graph rate, and its
/// effective triad GB/s should approach (not wildly exceed) the machine's
/// own measured streaming ceiling (69.95-81.21 GB/s, ROW 176/`rooflines.md`).
fn main() {
    println!("bge_matmul_cache_regime: H1 nano cell -- warm (cache-hot, existing form) vs cold (ROW 181 round-robin, {ROTATION} distinct weight buffers) at BGE's own real M shapes");
    println!("PRE-REGISTRATION: if H1 is the mechanism, cold GMAC/s should land near the in-graph ~10.7 GMAC/s figure (ROW 202's own 96-GEMM attribution), and cold triad-GB/s should approach the 69.95-81.21 GB/s machine streaming ceiling (ROW 176).");
    for &m in &[7usize, 8, 9] {
        shape_block("QKVO", m, 384, 384);
        shape_block("FFN-down", m, 1536, 384);
    }
}
