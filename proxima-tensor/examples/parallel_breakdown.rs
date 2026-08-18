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
use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate_parallel, map};

struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

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
        let start = Instant::now();
        let evaluated = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers).expect("gemm evaluates");
        let wall_ns = start.elapsed().as_nanos() as u64;
        let checksum = evaluated.root()[0];
        let totals = instrument::parallel_totals();
        let serial = instrument::serial_totals();
        let chunk_mean_ns = if totals.chunk_count == 0 {
            0.0
        } else {
            totals.chunk_nanos_sum as f64 / totals.chunk_count as f64
        };
        println!(
            "size={size} threads={threads} iter={iter} checksum={checksum:.5} wall_ns={wall_ns} \
             parallel_nodes={} node_ns={} spawn_ns={} join_ns={} chunk_count={} \
             chunk_min_ns={} chunk_max_ns={} chunk_mean_ns={chunk_mean_ns:.1} \
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
