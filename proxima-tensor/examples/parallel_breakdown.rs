#![allow(clippy::expect_used)]
//! Measurement-only harness: instruments where `evaluate_parallel`'s
//! wall-clock actually goes (thread-spawn cost, join/teardown cost, and
//! per-chunk compute balance) via the crate's `instrument` feature counters
//! added to `cpu::run_chunks_threaded`. `matmul_program_rhs_transposed` and
//! `Lcg` copied verbatim from `examples/sweep_gemm.rs`. Not part of the
//! crate's public surface; throwaway for this measurement task.

use std::env;
use std::num::NonZeroUsize;
use std::time::Instant;

use proxima_tensor::instrument;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate_parallel, map};

fn random_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..n).map(|_| lcg.next_unit() * scale).collect()
}

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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <size> <threads> [iters]", args[0]);
        std::process::exit(1);
    }
    let size: u32 = args[1].parse().expect("size must be a positive integer");
    let threads: usize = args[2].parse().expect("threads must be a positive integer");
    let iters: usize = match args.get(3) {
        Some(value) => value.parse().expect("iters must be a positive integer"),
        None => 5,
    };

    let (m, k, n) = (size, size, size);
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs = random_vec(1, (m * k) as usize, 1.0);
    let rhs_t = random_vec(2, (n * k) as usize, 1.0);

    let workers = NonZeroUsize::new(threads).expect("threads must be nonzero");

    // one untimed warm-up
    let _ = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers).expect("warmup gemm evaluates");

    for iter in 0..iters {
        instrument::reset_parallel();
        instrument::reset_serial();
        instrument::reset_worker_busy();
        instrument::reset();
        let start = Instant::now();
        let evaluated = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers).expect("gemm evaluates");
        let wall_ns = start.elapsed().as_nanos() as u64;
        let checksum = evaluated.root()[0];
        let totals = instrument::parallel_totals();
        let serial = instrument::serial_totals();
        let kernel = instrument::totals();
        let chunk_mean_ns = if totals.chunk_count == 0 {
            0.0
        } else {
            totals.chunk_nanos_sum as f64 / totals.chunk_count as f64
        };
        let busy = instrument::worker_busy_snapshot();
        let busy_count = busy.len();
        let busy_sum: u64 = busy.iter().sum();
        let busy_min = busy.iter().copied().min().unwrap_or(0);
        let busy_max = busy.iter().copied().max().unwrap_or(0);
        let busy_mean = if busy_count == 0 { 0.0 } else { busy_sum as f64 / busy_count as f64 };
        let busy_variance = if busy_count == 0 {
            0.0
        } else {
            busy.iter().map(|value| (*value as f64 - busy_mean).powi(2)).sum::<f64>() / busy_count as f64
        };
        let busy_stddev = busy_variance.sqrt();
        let busy_spread = if busy_min == 0 { 0.0 } else { busy_max as f64 / busy_min as f64 };
        // utilization: summed busy time versus the region wall clock stretched
        // across every worker that claimed a chunk — 1.0 means every worker was
        // busy for the entire parallel region, no idling.
        let utilization = if totals.node_nanos == 0 || busy_count == 0 {
            0.0
        } else {
            busy_sum as f64 / (totals.node_nanos as f64 * busy_count as f64)
        };
        let busy_per_mac = if kernel.mac_ops == 0 { 0.0 } else { busy_sum as f64 / kernel.mac_ops as f64 };
        println!(
            "size={size} threads={threads} iter={iter} checksum={checksum:.5} wall_ns={wall_ns} \
             parallel_nodes={} node_ns={} spawn_ns={} join_ns={} chunk_count={} \
             chunk_min_ns={} chunk_max_ns={} chunk_mean_ns={chunk_mean_ns:.1} \
             busy_workers={busy_count} busy_sum_ns={busy_sum} busy_min_ns={busy_min} busy_max_ns={busy_max} \
             busy_mean_ns={busy_mean:.1} busy_stddev_ns={busy_stddev:.1} busy_spread={busy_spread:.3} \
             utilization={utilization:.4} mac_ops={} operand_loads={} busy_per_mac={busy_per_mac:.6} \
             serial_prepare_ns={} serial_alloc_ns={} serial_split_ns={} serial_slice_carve_ns={} \
             serial_finish_ns={} serial_bookkeeping_ns={} serial_sequential_compute_ns={} \
             serial_evaluate_parallel_ns={} serial_evaluate_parallel_calls={}",
            totals.parallel_nodes,
            totals.node_nanos,
            totals.spawn_nanos,
            totals.join_nanos,
            totals.chunk_count,
            totals.chunk_nanos_min,
            totals.chunk_nanos_max,
            kernel.mac_ops,
            kernel.operand_loads,
            serial.prepare_nanos,
            serial.alloc_nanos,
            serial.split_nanos,
            serial.slice_carve_nanos,
            serial.finish_nanos,
            serial.bookkeeping_nanos,
            serial.sequential_compute_nanos,
            serial.evaluate_parallel_nanos,
            serial.evaluate_parallel_calls,
        );
    }
}
