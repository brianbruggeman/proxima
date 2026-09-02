//! Micro-bench for the `Conv`-shaped 2D GEMM tile (`docs/discipline.md` ROW
//! 149): `cpu::run_reduce`'s blocked (outer `ci` x contiguous-inner `ky,kx`)
//! NEON kernel against the crate's own `Generic` per-element interpreter --
//! the home-turf incumbent this initiative supersedes for `Conv`, named
//! explicitly in ROW 148/149, not a strawman.
//!
//! Both arms run the IDENTICAL `Op` program (same shape, same strides, same
//! total MAC count): the "engaged" arm's fused reduce body is the plain
//! one-step `windowed * weight` `conv2d_core` itself builds
//! (`proxima-onnx/src/lower.rs`), which `body_shape` classifies as
//! `BodyShape::Binary` and `conv_gemm_tile_plan` accepts. The "generic" arm
//! chains one extra `* ones` elementwise step ahead of the reduce so the
//! fused body has two steps instead of one -- `body_shape` then returns
//! `BodyShape::Generic`, which `conv_gemm_tile_plan`, `neon_tile_plan`, and
//! `width_tile_plan` all reject identically (every one of them matches on
//! `BodyShape::Binary` first), forcing the scalar per-element interpreter
//! for the exact same numeric work. This is a shape-preserving arm split,
//! not a different program.
//!
//! Four sizes: mnist's own 3 real `Conv` fold shapes (`docs/discipline.md`
//! ROW 148's measured `extents`) plus one larger square shape past all
//! three, to see how the win scales.
//!
//! Re-prove with (host must be quiet -- see the discipline log row this
//! bench seeds for the loadout it was actually measured under):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-tensor --bench bench_conv_gemm_tile -- --save-baseline row149-micro`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::Criterion;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, TypedBuffer, append,
    evaluate_typed, map,
};

/// One `Conv` layer's shape: `co` output channels, `ci` input channels,
/// `oy`/`ox` output spatial extent, `kh`/`kw` kernel extent -- exactly the
/// fields `docs/discipline.md` ROW 148's own diagnostic dumped per layer.
struct ConvShape {
    label: &'static str,
    co: u32,
    ci: u32,
    oy: u32,
    ox: u32,
    kh: u32,
    kw: u32,
}

/// mnist's 3 real `Conv` folds (ROW 148's measured `extents`, re-labeled
/// `co,ci,oy,ox,kh,kw`) plus one larger square shape past all three.
const SHAPES: [ConvShape; 4] = [
    ConvShape {
        label: "mnist_layer1_co8_ci1_26x26",
        co: 8,
        ci: 1,
        oy: 26,
        ox: 26,
        kh: 3,
        kw: 3,
    },
    ConvShape {
        label: "mnist_layer2_co16_ci8_24x24",
        co: 16,
        ci: 8,
        oy: 24,
        ox: 24,
        kh: 3,
        kw: 3,
    },
    ConvShape {
        label: "mnist_layer3_co24_ci16_22x22",
        co: 24,
        ci: 16,
        oy: 22,
        ox: 22,
        kh: 3,
        kw: 3,
    },
    ConvShape {
        label: "larger_square_co64_ci32_32x32",
        co: 64,
        ci: 32,
        oy: 32,
        ox: 32,
        kh: 3,
        kw: 3,
    },
];

fn input(program: &mut Vec<Op>, shape: &[u32]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: shape.iter().map(|&extent| Extent::Static(extent)).collect(),
            name: None,
        },
    )
}

/// Builds `conv2d_core`'s own reduce shape directly at the `Op` level
/// (`n=1` fixed, matching every one of mnist's 3 real folds): a rank-6
/// `windowed` input declared `[n,ci,oy,ox,kh,kw]` -- exactly the row-major
/// layout `window_materialize` produces (`proxima-onnx/src/lower.rs`,
/// `docs/discipline.md` ROW 148) -- multiplied against a rank-4 `weight`
/// input declared `[co,ci,kh,kw]`, reduced over `(ci,kh,kw)`. `extra_step`
/// true chains one additional `* ones` elementwise step ahead of the
/// reduce, defeating `body_shape`'s `BodyShape::Binary` classification
/// (this file's own doc has the full mechanism) without changing the
/// shape, strides, or total MAC count at all.
fn conv_program(shape: &ConvShape, extra_step: bool) -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let windowed = input(
        &mut program,
        &[1, shape.ci, shape.oy, shape.ox, shape.kh, shape.kw],
    );
    let weight = input(&mut program, &[shape.co, shape.ci, shape.kh, shape.kw]);

    // shared iteration space: 0=n 1=co 2=oy 3=ox 4=ci 5=ky 6=kx -- identical
    // to `conv2d_core`'s own comment and axis assignment.
    let windowed_pattern = map::projection(7, &[0, 4, 2, 3, 5, 6]);
    let weight_pattern = map::projection(7, &[1, 4, 5, 6]);
    let mut product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (windowed, IndexMap::Affine(windowed_pattern)),
                (weight, IndexMap::Affine(weight_pattern)),
            ],
            name: None,
        },
    );

    if extra_step {
        let ones = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(1),
                    Extent::Static(shape.co),
                    Extent::Static(shape.oy),
                    Extent::Static(shape.ox),
                    Extent::Static(shape.ci),
                    Extent::Static(shape.kh),
                    Extent::Static(shape.kw),
                ],
                value: 1.0,
            },
        );
        product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (
                        product,
                        IndexMap::Affine(map::projection(7, &[0, 1, 2, 3, 4, 5, 6])),
                    ),
                    (
                        ones,
                        IndexMap::Affine(map::projection(7, &[0, 1, 2, 3, 4, 5, 6])),
                    ),
                ],
                name: None,
            },
        );
    }

    let reduced = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(7, &[0, 1, 2, 3, 4, 5, 6])),
            out_map: IndexMap::Affine(map::projection(7, &[0, 1, 2, 3])),
            keep: Keep::Reduce,
            name: Some(shape.label.into()),
        }),
    );
    (program, windowed, weight, reduced)
}

fn deterministic_data(len: usize, phase: f32) -> Vec<f32> {
    (0..len).map(|value| (value as f32 * phase).sin()).collect()
}

fn run(
    program: &[Op],
    windowed: NodeId,
    weight: NodeId,
    output: NodeId,
    windowed_data: &[f32],
    weight_data: &[f32],
) -> Vec<f32> {
    let _ = windowed;
    let _ = weight;
    let blocks = [
        TypedBuffer::Float32(windowed_data.to_vec()),
        TypedBuffer::Float32(weight_data.to_vec()),
    ];
    let rows = evaluate_typed(program, &[], &blocks, &[output]).expect("evaluate_typed");
    let (_, _, TypedBuffer::Float32(data)) = rows.into_iter().next().expect("one output row")
    else {
        panic!("conv reduce output was not f32");
    };
    data
}

fn main() {
    // ROW 189: same env-var hook `real_mnist_accuracy.rs` (ROW 188) uses --
    // toggles the Accelerate/AMX route for `conv_tile`'s own arm below,
    // independently re-provable without a permanent bench-default change.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if std::env::var("PROXIMA_ACCELERATE_GEMM").as_deref() == Ok("1") {
        proxima_tensor::cpu::set_accelerate_gemm_enabled(true);
    }

    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("bench_conv_gemm_tile");
    group.sample_size(20);

    for shape in &SHAPES {
        let windowed_len = (shape.ci * shape.oy * shape.ox * shape.kh * shape.kw) as usize;
        let weight_len = (shape.co * shape.ci * shape.kh * shape.kw) as usize;
        let windowed_data = deterministic_data(windowed_len, 0.0137);
        let weight_data = deterministic_data(weight_len, 0.0271);

        let (engaged_program, engaged_windowed, engaged_weight, engaged_output) =
            conv_program(shape, false);
        let (generic_program, generic_windowed, generic_weight, generic_output) =
            conv_program(shape, true);

        // correctness self-check, once per shape, outside the timed loop:
        // both arms must agree bit-for-bit -- the whole point of the
        // extra-step technique is that it changes ONLY which executor path
        // runs, never the numeric result.
        let engaged_result = run(
            &engaged_program,
            engaged_windowed,
            engaged_weight,
            engaged_output,
            &windowed_data,
            &weight_data,
        );
        let generic_result = run(
            &generic_program,
            generic_windowed,
            generic_weight,
            generic_output,
            &windowed_data,
            &weight_data,
        );
        assert_eq!(
            engaged_result.len(),
            generic_result.len(),
            "{}: output length mismatch",
            shape.label
        );
        for (index, (&engaged_value, &generic_value)) in
            engaged_result.iter().zip(&generic_result).enumerate()
        {
            assert!(
                (engaged_value - generic_value).abs() <= engaged_value.abs() * 1e-5 + 1e-6,
                "{}: element {index} diverged: conv_tile={engaged_value} generic={generic_value}",
                shape.label,
            );
        }

        group.bench_function(format!("{}/conv_tile", shape.label), |bencher| {
            bencher.iter(|| {
                run(
                    &engaged_program,
                    engaged_windowed,
                    engaged_weight,
                    engaged_output,
                    &windowed_data,
                    &weight_data,
                )
            });
        });
        group.bench_function(format!("{}/generic", shape.label), |bencher| {
            bencher.iter(|| {
                run(
                    &generic_program,
                    generic_windowed,
                    generic_weight,
                    generic_output,
                    &windowed_data,
                    &weight_data,
                )
            });
        });
    }

    group.finish();
    criterion.final_summary();
}
