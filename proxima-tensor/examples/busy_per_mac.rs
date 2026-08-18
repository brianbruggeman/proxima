#![allow(clippy::expect_used)]
//! Lever A x lever B interaction measurement: one line of CSV-ish text per
//! iteration, `size`/`threads` from argv, `iters` (default 9) repeats. Reuses
//! `matmul_program_rhs_transposed`/`Lcg`/`random_vec` verbatim from
//! `sweep_gemm.rs` so the checksum lines up with that harness's reference
//! values (512 -> 135.87619, 1024 -> 260.24106, 2048 -> 513.10425).
//!
//! `busy_ns` is total CPU time spent inside the compute kernel, summed across
//! every OS thread that ran a chunk this iteration — not wall-clock. At
//! `threads=1` `BoundOp::split_aligned(1, _)` returns `None` (a single chunk
//! is not worth wrapping in `thread::scope`), so dispatch takes the direct
//! sequential arm and there is no per-worker busy sample to sum; `busy_ns`
//! falls back to `serial_sequential_compute_nanos`, the same wall-clock
//! region under a different name. `busy_per_mac = busy_ns / mac_ops` is
//! deliberately CPU-time, not wall-time — it isolates kernel-level cost per
//! unit of work from parallel-dispatch overhead (imbalance, spawn/join),
//! which is why it can be compared directly across thread counts.

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

fn mean_and_stddev(values: &[u64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
    let variance =
        values.iter().map(|value| (*value as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, variance.sqrt())
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
        None => 9,
    };

    let (m, k, n) = (size, size, size);
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs = random_vec(1, (m * k) as usize, 1.0);
    let rhs_t = random_vec(2, (n * k) as usize, 1.0);
    let workers = NonZeroUsize::new(threads).expect("threads must be nonzero");

    let _ = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers).expect("warmup gemm evaluates");

    for iter in 0..iters {
        instrument::reset();
        instrument::reset_parallel();
        instrument::reset_serial();
        instrument::reset_worker_busy();

        let wall_start = Instant::now();
        let evaluated =
            evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers).expect("gemm evaluates");
        let wall_ns = wall_start.elapsed().as_nanos() as u64;
        let checksum = evaluated.root()[0];

        let totals = instrument::totals();
        let serial = instrument::serial_totals();
        let busy_samples = instrument::worker_busy_snapshot();

        let (busy_ns, busy_workers, busy_min, busy_max, busy_mean, busy_stddev) = if busy_samples.is_empty() {
            let sequential = serial.sequential_compute_nanos;
            (sequential, 1u64, sequential, sequential, sequential as f64, 0.0)
        } else {
            let sum: u64 = busy_samples.iter().sum();
            let min = *busy_samples.iter().min().expect("nonempty");
            let max = *busy_samples.iter().max().expect("nonempty");
            let (mean, stddev) = mean_and_stddev(&busy_samples);
            (sum, busy_samples.len() as u64, min, max, mean, stddev)
        };
        let busy_per_mac = busy_ns as f64 / totals.mac_ops.max(1) as f64;

        println!(
            "size={size} threads={threads} iter={iter} checksum={checksum:.5} wall_ns={wall_ns} \
             mac_ops={} operand_loads={} busy_workers={busy_workers} busy_ns={busy_ns} \
             busy_min_ns={busy_min} busy_max_ns={busy_max} busy_mean_ns={busy_mean:.1} \
             busy_stddev_ns={busy_stddev:.1} busy_per_mac={busy_per_mac:.6} \
             serial_sequential_compute_ns={}",
            totals.mac_ops, totals.operand_loads, serial.sequential_compute_nanos,
        );
    }
}
