//! Nano/micro cells for the BGE matmul lever session — isolates the two
//! dominant matmul shapes (QKVO: `(M,384)x(384,384)`, FFN-down:
//! `(M,1536)x(1536,384)`) in a minimal in-process timing loop, comparing
//! (a) the batched shape BGE's real graph actually produces (`[1,M,K]x[K,N]`,
//! `leading_output_axes.len() == 2` because the size-1 batch axis is never
//! flattened into the token axis) against (b) the same shape with the
//! trivial batch axis removed (`[M,K]x[K,N]`, `leading_output_axes.len() ==
//! 1`) — the smallest fixed variant the `bge_matmul` diagnosis names, since
//! `neon_tile_plan`/`width_tile_plan` both gate on exactly one leading axis
//! (`cpu.rs:8043`, `cpu.rs:7546`) — and (c) `PROXIMA_ACCELERATE_GEMM=1` on
//! the batched shape, to confirm it cannot help here (Accelerate's own gate
//! sits behind the SAME `neon_tile_plan` precondition, `cpu.rs:6032`).
//! No e2e bge_eval sweep — single-shape, in-process, seconds-long, per the
//! session's owner-directed scope.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::{GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::evaluate_named;

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

fn matmul_shape_gmac_s(m: usize, k: usize, n: usize) -> f64 {
    (m * k * n) as f64 / 1e9
}

struct Arm {
    label: &'static str,
    lhs_dims: Vec<i64>,
    accelerate: bool,
}

fn run_arm(k: usize, n: usize, arm: &Arm) -> (f64, f64) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    proxima_tensor::cpu::set_accelerate_gemm_enabled(arm.accelerate);
    let _ = arm.accelerate;

    let lhs_len: usize = arm.lhs_dims.iter().map(|&dim| dim as usize).product();
    let lhs_data = deterministic_data(lhs_len, 0x1234_5678);
    let rhs_data = deterministic_data(k * n, 0x9abc_def0);

    let lhs = f32_initializer("lhs", arm.lhs_dims.clone(), lhs_data);
    let rhs = f32_initializer("rhs", vec![k as i64, n as i64], rhs_data);
    let node = NodeProto { input: vec!["lhs", "rhs"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
    let graph = GraphProto {
        node: vec![node],
        name: "micro_matmul_graph",
        initializer: vec![lhs, rhs],
        output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
        ..GraphProto::default()
    };
    let lowered = lower_graph(&graph).expect("lower synthetic MatMul");
    let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    let output = lowered.graph_outputs[0].1;

    for _ in 0..50 {
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    let calls_per_repeat = 300usize;
    let mut ns_per_call_per_repeat = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = std::time::Instant::now();
        for _ in 0..calls_per_repeat {
            let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("timed eval");
            std::hint::black_box(&evaluated);
        }
        let elapsed = start.elapsed();
        ns_per_call_per_repeat.push(elapsed.as_nanos() as f64 / calls_per_repeat as f64);
    }

    let mean = ns_per_call_per_repeat.iter().sum::<f64>() / ns_per_call_per_repeat.len() as f64;
    let variance = ns_per_call_per_repeat.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / ns_per_call_per_repeat.len() as f64;
    let cov = variance.sqrt() / mean * 100.0;
    println!(
        "  {:<28} ns/call(mean of 5x{calls_per_repeat}) = {mean:>10.1}  CoV={cov:>5.2}%  samples={:?}",
        arm.label,
        ns_per_call_per_repeat.iter().map(|value| format!("{value:.0}")).collect::<Vec<_>>()
    );
    (mean, cov)
}

fn shape_block(name: &str, m: usize, k: usize, n: usize) {
    let macs = (m * k * n) as f64;
    let gmac_s_from_ns = |ns_per_call: f64| macs / (ns_per_call / 1e9) / 1e9;
    println!("\n=== {name}: M={m} K={k} N={n} ({:.4} GMAC total) ===", matmul_shape_gmac_s(m, k, n));

    let arms = [
        Arm { label: "batched [1,M,K] (BGE real shape, defect present)", lhs_dims: vec![1, m as i64, k as i64], accelerate: false },
        Arm { label: "unbatched [M,K] (leading_axes=1, gate fires)", lhs_dims: vec![m as i64, k as i64], accelerate: false },
        Arm { label: "batched + PROXIMA_ACCELERATE_GEMM=1", lhs_dims: vec![1, m as i64, k as i64], accelerate: true },
    ];
    for arm in &arms {
        let (mean_ns, cov) = run_arm(k, n, arm);
        println!("    -> {:.3} GMAC/s (CoV {:.2}%)", gmac_s_from_ns(mean_ns), cov);
    }
}

fn main() {
    println!("bge_matmul_micro: pre-registered predictions (mechanism: cpu.rs neon_tile_plan/width_tile_plan both require leading_output_axes.len() == 1)");
    println!("  batched arm: expect near the crate's own scalar-FMA roofline (~6 GMAC/s, ROW 145/150/154/156), since neither AArch64 tile gate fires");
    println!("  unbatched arm: expect a large multiple of the batched arm (tile gate fires; N=384/N=1536 are both multiples of WIDTH_TILE_VECS*4=16)");
    println!("  accelerate arm: expect NO CHANGE vs the batched default — Accelerate's own gate sits behind the same leading_axes==1 precondition (cpu.rs:6032, 6044)");

    shape_block("QKVO", 8, 384, 384);
    shape_block("FFN-down", 8, 1536, 384);
}
