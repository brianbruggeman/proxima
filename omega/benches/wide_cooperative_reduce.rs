//! Cooperative-reduce dispatch width -- `SIMD_WIDTH` (32, incumbent shape)
//! vs `metal-wide-cooperative-reduce`'s reduction-extent-scaled width, on
//! the exact op class the sealed decode-step measurement named
//! (`reduce-cooperative`, `docs/discipline.md`'s row for this initiative):
//! `op_count=385 gpu_ms=8.625 gpu_ns_per_op=22401.9 operand_bytes=12,509,184`
//! -- 22.4 us/op, 1.45 GB/s, against a ~4 us zero-byte dispatch floor.
//!
//! ONE bench file, run TWICE with different feature sets (this crate's
//! dispatch width is a compile-time `cfg`, not a runtime switch -- see
//! `msl::cooperative_reduce_width`'s two `#[cfg(feature = ...)]` arms) and
//! compared via saved criterion baselines:
//!
//! ```sh
//! CARGO_TARGET_DIR=<scratch> cargo bench -p omega --bench wide_cooperative_reduce \
//!   --features metal -- --save-baseline off
//! CARGO_TARGET_DIR=<scratch> cargo bench -p omega --bench wide_cooperative_reduce \
//!   --features metal,metal-wide-cooperative-reduce -- --save-baseline on --baseline off
//! ```
//!
//! TWO-SIZE MARGINAL method (`proxima-tensor/docs/discipline.md` ROW 71):
//! `rms_norm_small` (64 elements) and `rms_norm_large` (4096, the real
//! RMS-norm width) isolate the per-call fixed dispatch cost from the
//! per-element slope -- `(t_large - t_small) / (4096 - 64)` is the marginal
//! per-element cost with the ~4us dispatch floor cancelled out of both
//! arms identically.
//!
//! DEGENERATE-CONTROL REQUIREMENT: before reading any arm as a result,
//! check the `off` arm's ns/op against the sealed 22401.9 ns/op (1.45
//! GB/s) real-decode-loop number above. If `off` does not land in that
//! neighborhood, this harness is measuring something the real decode loop
//! is not (a different dispatch shape, a warm-cache artifact, a
//! non-representative op count) and no arm below may be reported as a
//! result until that is fixed.
//!
//! Home-turf incumbent arm: `rms_norm_large` at `off` -- `SIMD_WIDTH` (32)
//! is the shape every cooperative reduce in this crate dispatched at before
//! this initiative, on the exact 4096-wide extent the sealed measurement
//! names. `design-favors: incumbent`.
//!
//! Adversarial / malformed arm: `rms_norm_ragged` (4095, one below a whole
//! number of super-blocks and one below the real width) -- exercises the
//! per-lane identity-seed guard (`fold_init_tokens`/
//! `cooperative_identity_token`) and, with the feature on, the two-level
//! fold's exact-simdgroup-count guard (`push_cooperative_reduce_tail`) on
//! the same size class the correctness gate's ragged tests
//! (`omega/tests/wide_cooperative_reduce_ragged.rs`) already proved correct
//! -- this arm is about COST, not correctness.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use omega::execute;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `(1, cols)` -> `(1,)` sum reduce -- the RMS-norm shape the sealed
/// measurement's `reduce-cooperative` op class names, isolated with no
/// elementwise fusion in the way (mirrors `omega/tests/
/// wide_cooperative_reduce_ragged.rs::single_row_reduce_program`, kept as
/// its own copy here since bench targets cannot depend on a sibling test
/// binary).
fn rms_norm_sum_program(cols: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(cols)],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: input,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("rms_norm_sum".into()),
        }),
    );
    (program, sum)
}

/// One softmax-shaped max-then-sum pair over `cols` scores -- the
/// attention-shaped case the task brief names alongside the RMS-norm width,
/// sized at 512 (a realistic single-head attention sequence length) so the
/// bench sweeps a reduction extent between `rms_norm_small` and
/// `rms_norm_large` rather than only the two endpoints.
fn attention_softmax_max_program(cols: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let scores = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(cols)],
            name: None,
        },
    );
    let max = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: scores,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("attention_softmax_max".into()),
        }),
    );
    (program, max)
}

fn bench_reduce_cooperative(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reduce_cooperative");

    let sizes: [(&str, u32); 5] = [
        ("rms_norm_small_64", 64),
        ("rms_norm_medium_128", 128),
        ("attention_softmax_512", 512),
        ("rms_norm_ragged_4095", 4095),
        ("rms_norm_large_4096", 4096),
    ];

    for (label, cols) in sizes {
        let (program, _root) = if label.starts_with("attention") {
            attention_softmax_max_program(cols)
        } else {
            rms_norm_sum_program(cols)
        };
        let input = random_vec(u64::from(cols), cols as usize);
        let blocks = [QuantizedBlock::Float32(input.as_slice())];

        group.throughput(Throughput::Bytes(u64::from(cols) * 4));
        group.bench_with_input(BenchmarkId::new(label, cols), &cols, |bencher, _cols| {
            bencher.iter(|| {
                let result = execute(&program, &[], &blocks, &[])
                    .expect("cooperative reduce executes on a real device");
                black_box(result.root()[0]);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_reduce_cooperative);
criterion_main!(benches);
