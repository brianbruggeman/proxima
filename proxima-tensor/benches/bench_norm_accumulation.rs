//! The rms norm's sum of squares accumulated two ways on the same CPU
//! evaluator and the same `[rows, width]` input: `f32` -- the f32 fold and
//! the `1/width` multiply this crate used before -- against `f64` -- the
//! squares widened into an f64 fold, divided by an f64 count and narrowed,
//! ggml-cpu's `rms_norm_f32` (ops.cpp:3957-3964). Both arms bind through the
//! same `evaluate_quantized_with_scratch` with a warm pool, so the delta is
//! the accumulation path alone. Run with and without `--all-features`: with
//! `reduce-epilogue-fusion` each arm is one fused reduce, without it the f64
//! arm runs through the island.
//!
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-tensor --bench bench_norm_accumulation`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::hint::black_box;

use criterion::Criterion;
use proxima_tensor::cpu::{BufferPools, QuantizedBlock, evaluate_quantized_with_scratch};
use proxima_tensor::spec::{
    count_constant, elementwise, input_leaf, reduce, rmsnorm, scalar_constant,
};
use proxima_tensor::{Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_numeric::{DType};

const SHAPES: [(u32, u32); 3] = [(1, 1024), (1, 4096), (32, 1024)];

fn inputs(program: &mut Vec<Op>, rows: u32, width: u32) -> (NodeId, NodeId, NodeId) {
    let x = input_leaf(
        program,
        DType::FLOAT32,
        vec![Extent::Static(rows), Extent::Static(width)],
        "x",
    );
    let gamma = input_leaf(
        program,
        DType::FLOAT32,
        vec![Extent::Static(width)],
        "gamma",
    );
    let eps = scalar_constant(program, 1e-6);
    (x, gamma, eps)
}

fn single_precision_norm(rows: u32, width: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let (x, gamma, eps) = inputs(&mut program, rows, width);
    let inverse_width = scalar_constant(&mut program, 1.0 / width as f32);
    let squared = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(x, "sd->sd"), (x, "sd->sd")],
    )
    .unwrap();
    let sum = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "sd->sd",
        "s->sd",
    )
    .unwrap();
    let mean = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(sum, "s->s"), (inverse_width, "->s")],
    )
    .unwrap();
    let shifted = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        &[(mean, "s->s"), (eps, "->s")],
    )
    .unwrap();
    let root = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::SquareRoot,
        &[(shifted, "s->s")],
    )
    .unwrap();
    let inverse = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Reciprocal,
        &[(root, "s->s")],
    )
    .unwrap();
    let normed = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(x, "sd->sd"), (inverse, "s->sd")],
    )
    .unwrap();
    let out = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (gamma, "d->sd")],
    )
    .unwrap();
    (program, out)
}

fn double_precision_norm(rows: u32, width: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let (x, gamma, eps) = inputs(&mut program, rows, width);
    let count = count_constant(&mut program, width);
    let out = rmsnorm(&mut program, x, gamma, count, eps).unwrap();
    (program, out)
}

fn evaluate(
    program: &[Op],
    out: NodeId,
    data: &[f32],
    gamma: &[f32],
    pools: &mut BufferPools,
    validated: &mut Option<BTreeSet<NodeId>>,
) -> Vec<f32> {
    let blocks = [
        QuantizedBlock::Float32(data),
        QuantizedBlock::Float32(gamma),
    ];
    let evaluated =
        evaluate_quantized_with_scratch(program, &[], &blocks, &[out], pools, validated).unwrap();
    let values = evaluated.get(out).unwrap().0.to_vec();
    evaluated.into_scratch(&mut pools.0);
    values
}

fn bench_norm_accumulation(criterion: &mut Criterion) {
    for (rows, width) in SHAPES {
        let data: Vec<f32> = (0..rows * width)
            .map(|index| ((index * 7919) % 1013) as f32 / 97.0 - 5.0)
            .collect();
        let gamma: Vec<f32> = (0..width)
            .map(|index| 0.5 + (index % 17) as f32 / 16.0)
            .collect();
        let mut group = criterion.benchmark_group(format!("rms_norm_{rows}x{width}"));
        for (label, (program, out)) in [
            ("f32", single_precision_norm(rows, width)),
            ("f64", double_precision_norm(rows, width)),
        ] {
            let mut pools = BufferPools::default();
            let mut validated = None;
            let warm = evaluate(&program, out, &data, &gamma, &mut pools, &mut validated);
            assert_eq!(
                warm.len(),
                (rows * width) as usize,
                "{label} norm output length"
            );
            group.bench_function(label, |bencher| {
                bencher.iter(|| {
                    black_box(evaluate(
                        &program,
                        out,
                        &data,
                        &gamma,
                        &mut pools,
                        &mut validated,
                    ))
                });
            });
        }
        group.finish();
    }
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_norm_accumulation(&mut criterion);
    criterion.final_summary();
}
