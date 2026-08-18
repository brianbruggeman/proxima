//! Measurement-only harness for the parallel GEMM scaling investigation
//! (`scratchpad` perf task, 2026-08-17). Reuses `profile_hot.rs`'s
//! transposed-RHS GEMM program builder verbatim, then instruments four
//! distinct questions about where 8-worker speedup is lost relative to
//! `evaluate`'s single-thread baseline: wall-clock scaling per worker
//! count, the fixed cost of the parallel path at workers=1, the chunk
//! geometry `BoundOp::split` produces, and the NEON tile kernel's fallback
//! rate at the chunk edges it produces.
//!
//! Not part of the crate's public surface; deleted at the end of the
//! session that needed it.

#![allow(dead_code)]

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, bind, evaluate, evaluate_parallel, infer, map};

/// Same GEMM, RHS stored transposed (`[n, k]`, ggml's own `mul_mat`
/// layout) instead of `[k, n]` — copied verbatim from `profile_hot.rs`.
fn matmul_program_rhs_transposed(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul_rhs_transposed".into()),
        }),
    );
    (program, sum)
}

fn transpose(rhs: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut rhs_t = vec![0.0f32; k * n];
    for ki in 0..k {
        for ni in 0..n {
            rhs_t[ni * k + ki] = rhs[ki * n + ni];
        }
    }
    rhs_t
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn time_evaluate(program: &[Op], lhs: &[f32], rhs_t: &[f32], runs: usize) -> Vec<Duration> {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            match evaluate(program, &[], &[lhs, rhs_t], &[]) {
                Ok(_) => start.elapsed(),
                Err(error) => panic!("evaluate failed: {error:?}"),
            }
        })
        .collect()
}

fn time_evaluate_parallel(program: &[Op], lhs: &[f32], rhs_t: &[f32], workers: usize, runs: usize) -> Vec<Duration> {
    let workers = match NonZeroUsize::new(workers) {
        Some(value) => value,
        None => panic!("workers must be nonzero"),
    };
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            match evaluate_parallel(program, &[], &[lhs, rhs_t], &[], workers) {
                Ok(_) => start.elapsed(),
                Err(error) => panic!("evaluate_parallel failed: {error:?}"),
            }
        })
        .collect()
}

/// Output element count of a `Keep::Reduce` chunk — mirrors
/// `cpu.rs::node_output_len`'s `Keep::Reduce` arm (private to that module,
/// so reimplemented here from the public `BoundOp`/`BoundOpKind` fields).
fn reduce_output_len(chunk: &proxima_tensor::BoundOp) -> usize {
    match &chunk.kind {
        proxima_tensor::BoundOpKind::Reduce { output_axes, .. } => {
            let (leading, last) = output_axes
                .as_slice()
                .split_at(output_axes.len().saturating_sub(1));
            let leading_product: u64 = leading.iter().map(|axis| chunk.extents[*axis as usize]).product();
            let width = last.first().map_or(1, |axis| chunk.extents[*axis as usize]);
            leading_product as usize * width as usize
        }
        proxima_tensor::BoundOpKind::Elementwise { .. } => {
            chunk.extents.iter().product::<u64>() as usize
        }
    }
}

fn report_worker_sweep(label: &str, program: &[Op], lhs: &[f32], rhs_t: &[f32]) {
    println!("\n=== {label}: serial baseline (evaluate) ===");
    let serial = time_evaluate(program, lhs, rhs_t, 3);
    for (index, sample) in serial.iter().enumerate() {
        println!("  evaluate run {index}: {:.3} ms", sample.as_secs_f64() * 1000.0);
    }
    let serial_median = median(serial);
    println!("  evaluate median: {:.3} ms", serial_median.as_secs_f64() * 1000.0);

    println!("=== {label}: worker sweep (evaluate_parallel) ===");
    let mut workers_one_median = None;
    for workers in [1usize, 2, 4, 8, 16] {
        let samples = time_evaluate_parallel(program, lhs, rhs_t, workers, 3);
        for (index, sample) in samples.iter().enumerate() {
            println!(
                "  workers={workers} run {index}: {:.3} ms",
                sample.as_secs_f64() * 1000.0
            );
        }
        let this_median = median(samples);
        if workers == 1 {
            workers_one_median = Some(this_median);
            let gap = this_median.as_secs_f64() - serial_median.as_secs_f64();
            println!(
                "  workers=1 median: {:.3} ms  (vs evaluate median {:.3} ms, gap {:+.3} ms = fixed parallel-machinery overhead at zero parallelism)",
                this_median.as_secs_f64() * 1000.0,
                serial_median.as_secs_f64() * 1000.0,
                gap * 1000.0
            );
        } else {
            let baseline = match workers_one_median {
                Some(value) => value,
                None => panic!("workers=1 must run first in the sweep"),
            };
            let speedup = baseline.as_secs_f64() / this_median.as_secs_f64();
            println!(
                "  workers={workers} median: {:.3} ms  speedup-vs-workers=1: {:.2}x",
                this_median.as_secs_f64() * 1000.0,
                speedup
            );
        }
    }
}

fn report_chunk_distribution(m: u32, k: u32, n: u32) {
    println!("\n=== chunk distribution: workers=8 at {m}x{k}x{n} ===");
    let (program, sum) = matmul_program_rhs_transposed(m, k, n);
    let shapes = match infer(&program, &[]) {
        Ok(value) => value,
        Err(error) => panic!("infer failed: {error:?}"),
    };
    let bound = match bind(&program, &shapes, &[sum]) {
        Ok(value) => value,
        Err(error) => panic!("bind failed: {error:?}"),
    };
    let gemm_node = match bound.iter().find(|op| op.node == sum) {
        Some(value) => value,
        None => panic!("no bound op resolves to the GEMM's root node"),
    };
    let chunks = match gemm_node.split(8) {
        Some(value) => value,
        None => panic!("BoundOp::split(8) returned None for a 1024-row GEMM — unexpected"),
    };
    let lens: Vec<usize> = chunks.iter().map(reduce_output_len).collect();
    let min = *lens.iter().min().unwrap_or(&0);
    let max = *lens.iter().max().unwrap_or(&0);
    println!("  chunk count: {}", chunks.len());
    for (index, len) in lens.iter().enumerate() {
        println!("  chunk {index}: {len} output elements");
    }
    println!("  min={min} max={max} spread={}", max - min);
}

#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
fn report_tile_fallback(m: u32, k: u32, n: u32) {
    println!("\n=== NEON tile counters (gate passes, invocations, fallback elements): {m}x{k}x{n} ===");
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs: Vec<f32> = (0..(m * k)).map(|value| (value % 13) as f32).collect();
    let rhs: Vec<f32> = (0..(k * n)).map(|value| (value % 7) as f32).collect();
    let rhs_t = transpose(&rhs, k as usize, n as usize);

    for workers in [1usize, 8] {
        let before = proxima_tensor::cpu::neon_tile_counters();
        let row_remainder_invocations_before = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
        let row_remainder_elements_before = proxima_tensor::cpu::neon_tile_row_remainder_elements();
        let workers_nonzero = match NonZeroUsize::new(workers) {
            Some(value) => value,
            None => panic!("workers must be nonzero"),
        };
        match evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers_nonzero) {
            Ok(_) => {}
            Err(error) => panic!("evaluate_parallel failed: {error:?}"),
        }
        let after = proxima_tensor::cpu::neon_tile_counters();
        let row_remainder_invocations_after = proxima_tensor::cpu::neon_tile_row_remainder_invocations();
        let row_remainder_elements_after = proxima_tensor::cpu::neon_tile_row_remainder_elements();
        println!(
            "  workers={workers}: gate_passes={} invocations={} row_remainder_invocations={} \
             row_remainder_elements={} fallback_elements={}",
            after.0 - before.0,
            after.1 - before.1,
            row_remainder_invocations_after - row_remainder_invocations_before,
            row_remainder_elements_after - row_remainder_elements_before,
            after.2 - before.2
        );
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "instrument")))]
fn report_tile_fallback(_m: u32, _k: u32, _n: u32) {
    println!("\n=== NEON tile counters: skipped, requires aarch64 + instrument feature ===");
}

fn main() {
    let (m, k, n) = (1024u32, 1024u32, 1024u32);
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs: Vec<f32> = (0..(m * k)).map(|value| (value % 13) as f32).collect();
    let rhs: Vec<f32> = (0..(k * n)).map(|value| (value % 7) as f32).collect();
    let rhs_t = transpose(&rhs, k as usize, n as usize);
    report_worker_sweep("gemm_1024x1024x1024", &program, &lhs, &rhs_t);
    report_tile_fallback(1024, 1024, 1024);
}
