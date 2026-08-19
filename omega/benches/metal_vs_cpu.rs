//! Metal vs CPU on the SAME `&[Op]` program — `omega::execute` (GPU) against
//! `proxima_tensor::cpu::evaluate` (single-threaded CPU), the two backends
//! `omega/src/metal.rs`'s own module doc says share one descriptor.
//!
//! Two physically different regimes, per the discipline log's own framing
//! (`project_tensor_decode_is_bandwidth_not_compute.md`: 4.00 vs 0.012
//! bytes/mac):
//!
//! - `gemm_square_f32`: compute-bound square GEMM at 512/1024/2048, the
//!   `rhs`-transposed contraction `proxima-tensor/examples/sweep_gemm.rs`
//!   already checksums (512 -> 135.87619, 1024 -> 260.24106,
//!   2048 -> 513.10425) — `root()[0]` is asserted against those numbers
//!   before any timing arm runs, so a GPU checksum drift is caught before it
//!   could be misread as a perf win.
//! - `matvec_batch1_f32`: bandwidth-bound batch-1 matvec at the model's real
//!   weight shapes (4096x4096, 4096x14336, 14336x4096 — `attn_q`/`attn_output`,
//!   `ffn_gate`/`ffn_up`, `ffn_down` in `spec::mistral_forward_program`'s
//!   `EMBEDDING=4096`/`FEED_FORWARD=14336`). M1 Max GPU and CPU share unified
//!   memory, so this is the arm that tells whether Metal wins the decode
//!   regime at all, not just the arm that flatters it.
//!
//! `Criterion::throughput` is set to `Throughput::Elements(2*m*k*n)` for the
//! GEMM group (criterion reports this as `Melem/s`; multiply by 1e-3 for
//! GFLOP/s since each element is one multiply-add = 2 FLOP already folded
//! into the count) and `Throughput::Bytes(weight_bytes)` for the matvec
//! group (criterion reports this directly as GiB/s) — see each group's own
//! comment for the exact conversion, since criterion only ever displays one
//! derived unit per group and this bench needs both per the task brief.
//!
//! Left UNRUN per the coordinator's directive
//! (`feedback_own_agents_contaminate_the_bench.md`: own agents building
//! `proxima-tensor/src/cpu.rs` concurrently would contaminate any number
//! taken right now, CoV 0.3% -> 53% is the measured cost of that). Run with:
//!
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target-metal cargo bench -p omega --bench metal_vs_cpu --features metal
//! ```
//!
//! on a quiet tree (check `uptime` / `ps -eo pcpu,comm` first — no other
//! cargo/rustc process above ~20% CPU).

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use omega::execute;
use proxima_tensor::cpu::evaluate;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, map};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `lhs [m,k]` times `rhs^T [n,k]` — byte-for-byte the same program shape as
/// `proxima-tensor/examples/sweep_gemm.rs`'s `matmul_program_rhs_transposed`,
/// duplicated here (rather than imported) because that file is a `PRESERVED
/// COPY` pinned to the `260.24106` checksum and is not a library target this
/// crate can depend on.
fn matmul_rhs_transposed_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
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
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_rhs_transposed".into()),
        }),
    );
    (program, sum)
}

/// `weight [out, in]` times `activation [in]` -> `[out]` — batch-1 decode's
/// actual access pattern: one activation vector, the weight matrix read
/// exactly once. This is the "4.00 bytes/mac" shape
/// (`project_tensor_decode_is_bandwidth_not_compute.md`), unlike the square
/// GEMM above which reuses each operand `O(n)` times.
fn matvec_program(out_dim: u32, in_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(in_dim)],
            name: None,
        },
    );
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(out_dim), Extent::Static(in_dim)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (activation, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("matvec".into()),
        }),
    );
    (program, sum)
}

/// Asserts the two backends agree before any timing arm runs (task
/// principle 18 / 14: a perf claim on a program that never proved
/// correctness is not a claim). `epsilon` widens with `k` the same way
/// `omega/tests/metal_parity.rs`'s own
/// `matmul_parity_holds_over_a_contraction_spanning_multiple_simd_lanes`
/// documents (observed 7.629395e-6 at k=97, float reassociation, not a bug)
/// — scaled `sqrt(k)` here since none of these `k` values have been run yet
/// to pin an observed worst case the way that test's comment does.
fn assert_checksum_agrees(case: &str, cpu: f32, metal: f32, k: u32) {
    let epsilon = 1e-5 * (k as f32).sqrt();
    let diff = (cpu - metal).abs();
    assert!(
        diff <= epsilon,
        "{case}: cpu={cpu} metal={metal} diff={diff} exceeds {epsilon} (k={k})"
    );
}

/// Pinned reference checksums from `sweep_gemm.rs`'s module doc — a second,
/// independent anchor beyond "cpu and metal agree with each other" (both
/// backends could agree on a shared regression).
fn reference_checksum(size: u32) -> Option<f32> {
    match size {
        512 => Some(135.87619),
        1024 => Some(260.24106),
        2048 => Some(513.10425),
        _ => None,
    }
}

fn bench_gemm_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_square_f32");
    for size in [512u32, 1024, 2048] {
        let (program, _root) = matmul_rhs_transposed_program(size, size, size);
        let lhs = random_vec(1, (size * size) as usize);
        let rhs_t = random_vec(2, (size * size) as usize);
        let blocks: [&[f32]; 2] = [&lhs, &rhs_t];

        let cpu = evaluate(&program, &[], &blocks, &[]).expect("cpu gemm evaluates");
        let metal = execute(&program, &[], &blocks, &[]).expect("metal gemm executes on a real device");
        let cpu_checksum = cpu.root()[0];
        let metal_checksum = metal.root()[0];
        assert_checksum_agrees("gemm_square", cpu_checksum, metal_checksum, size);
        if let Some(reference) = reference_checksum(size) {
            // sweep_gemm.rs's own random_vec uses `scale=1.0` the same way
            // this file's `random_vec` does (no separate scale factor), and
            // the same LCG seeds (1, 2) in the same rhs-transposed layout —
            // matching the exact program the reference checksum was pinned
            // against, not merely a same-shaped one.
            assert_checksum_agrees("gemm_square_vs_pinned_reference", reference, cpu_checksum, size);
        }

        // 2 FLOP per multiply-add, m=k=n=size.
        let flops = 2u64 * u64::from(size) * u64::from(size) * u64::from(size);
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |bencher, _| {
            bencher.iter(|| black_box(evaluate(&program, &[], &blocks, &[]).expect("cpu gemm evaluates")));
        });
        group.bench_with_input(BenchmarkId::new("metal", size), &size, |bencher, _| {
            bencher.iter(|| black_box(execute(&program, &[], &blocks, &[]).expect("metal gemm executes")));
        });
    }
    group.finish();
}

fn bench_matvec_batch1(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_batch1_f32");
    // (out_dim, in_dim, label) — the three real weight shapes
    // `mistral_forward_program` carries: `attn_q`/`attn_output`
    // (4096x4096), `ffn_gate`/`ffn_up` (14336x4096), `ffn_down`
    // (4096x14336).
    let shapes: [(u32, u32, &str); 3] =
        [(4096, 4096, "4096x4096"), (14336, 4096, "4096x14336"), (4096, 14336, "14336x4096")];

    for (out_dim, in_dim, label) in shapes {
        let (program, _root) = matvec_program(out_dim, in_dim);
        let activation = random_vec(3, in_dim as usize);
        let weight = random_vec(4, (out_dim as u64 * in_dim as u64) as usize);
        let blocks: [&[f32]; 2] = [&activation, &weight];

        let cpu = evaluate(&program, &[], &blocks, &[]).expect("cpu matvec evaluates");
        let metal = execute(&program, &[], &blocks, &[]).expect("metal matvec executes on a real device");
        assert_checksum_agrees("matvec", cpu.root()[0], metal.root()[0], in_dim);

        // bandwidth-bound: the weight matrix is read exactly once per call
        // and dominates (activation + output are `in_dim`/`out_dim`
        // elements, the weight is `out_dim * in_dim`) — this is the
        // "4.00 bytes/mac" shape, so GB/s (criterion's native
        // `Throughput::Bytes` unit) is the axis that tells compute-bound
        // apart from bandwidth-bound here, not GFLOP/s alone.
        let weight_bytes = u64::from(out_dim) * u64::from(in_dim) * 4;
        group.throughput(Throughput::Bytes(weight_bytes));

        group.bench_with_input(BenchmarkId::new("cpu", label), &label, |bencher, _| {
            bencher.iter(|| black_box(evaluate(&program, &[], &blocks, &[]).expect("cpu matvec evaluates")));
        });
        group.bench_with_input(BenchmarkId::new("metal", label), &label, |bencher, _| {
            bencher.iter(|| black_box(execute(&program, &[], &blocks, &[]).expect("metal matvec executes")));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_square, bench_matvec_batch1);
criterion_main!(benches);
