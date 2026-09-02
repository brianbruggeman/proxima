#![allow(clippy::expect_used)]
//! PRESERVED COPY — do not delete. This lived untracked in a detached-HEAD
//! worktree (`scratchpad/sweep-wt`) and was created and deleted repeatedly by
//! agents. Belongs in the repo permanently — see the landing recommendation.
//!
//! Measurement-only sweep harness: size and thread count taken from argv,
//! reports mean GFLOPS and a checksum. `Lcg` copied verbatim from
//! `benches/bench_vs_ggml.rs`, `matmul_program_rhs_transposed` copied
//! verbatim from `examples/profile_hot.rs`.
//!
//! Reference checksums verified on 2026-08-18 (`sweep_gemm 1024 4 5` printed
//! `checksum=260.24106`; 512 -> 135.87619, 2048 -> 513.10425) do NOT
//! reproduce as of `e0310ff` (2026-08-30, `fix(tensor): correct
//! Lcg::next_unit's halved-range bug`) — that commit fixed
//! `test_support::Lcg::next_unit` to draw from `[-1, 1)` as documented
//! instead of the `[-1, 0)` it silently drew from before, which is a real
//! correctness fix, not a regression: the old checksums were computed
//! against biased-mean inputs. Current values (verified against an
//! independent from-scratch Rust + numpy oracle, see
//! `omega/benches/metal_vs_cpu.rs`'s `reference_checksum` doc for the full
//! derivation): `sweep_gemm 1024 4 5` now prints `checksum=16.38366`;
//! 512 -> 7.67010, 2048 -> 4.68941.

use std::env;
use std::num::NonZeroUsize;
use std::time::Instant;

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate_parallel, map,
};

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
        None => match size {
            512 => 30,
            1024 => 30,
            2048 => 10,
            _ => 10,
        },
    };

    let (m, k, n) = (size, size, size);
    let (program, _sum) = matmul_program_rhs_transposed(m, k, n);
    let lhs = random_vec(1, (m * k) as usize, 1.0);
    let rhs_t = random_vec(2, (n * k) as usize, 1.0);

    let workers = NonZeroUsize::new(threads).expect("threads must be nonzero");

    // one untimed warm-up
    let _ = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers)
        .expect("warmup gemm evaluates");

    let mut checksum = 0.0f32;
    let start = Instant::now();
    for _ in 0..iters {
        let evaluated = evaluate_parallel(&program, &[], &[&lhs, &rhs_t], &[], workers)
            .expect("gemm evaluates");
        checksum = evaluated.root()[0];
    }
    let elapsed = start.elapsed();

    let mean_s = elapsed.as_secs_f64() / iters as f64;
    let flops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n);
    let gflops = (flops / mean_s) / 1e9;

    println!(
        "size={size} threads={threads} iters={iters} mean_gflops={gflops:.6} checksum={checksum:.5}"
    );
}
