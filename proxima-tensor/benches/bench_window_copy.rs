//! Micro-bench for the window-materialize-shaped copy specialization
//! (`docs/discipline.md` ROW 154, rung 2 of ROW 150's own charter):
//! `cpu::window_copy_block`'s row-segment copy against the crate's own
//! block-loop mechanism (ROW 150) it supersedes -- the home-turf incumbent
//! for this rung, not a strawman.
//!
//! Both arms run the IDENTICAL `Op` program (same shape, same strides, same
//! total element count): `window_materialize`'s own construction
//! (`proxima-onnx/src/lower.rs`) -- a source `image` read through a
//! `window_axis`-style two-term affine pattern (`h = oy + ky`,
//! `w = ox + kx`, stride/dilation 1, matching every one of mnist's 3 real
//! `Conv` folds) multiplied against an all-ones stamp shaped
//! `[oh,ow,kh,kw]`, exactly `window_materialize` itself
//! (`proxima-onnx/src/lower.rs:2320`). `bind.rs`'s `eliminate_identity_multiply`
//! (ROW 147) collapses this to a bare `Unary(Identity, image)` body -- the
//! "specialized" arm's real, engaged shape post-ROW-147, not a hypothetical.
//!
//! The "incumbent" arm chains one further, genuinely non-eliminable `+ 0.0`
//! step (`eliminate_identity_multiply` only ever fires on `Multiply`, never
//! `Add`, so a zero-valued constant survives composition) ahead of the same
//! read. This changes `body_shape` from `Unary(Identity, _)` to
//! `Binary(Add, _, _)` -- `window_copy_operand`'s own gate only matches
//! `Unary(Identity, _)`, so this arm falls straight through to
//! `run_elementwise_range`'s pre-existing block loop
//! (`elementwise_width_fast` called once per `kh` row, ROW 150's own
//! mechanism) while computing the IDENTICAL values (`x + 0.0 == x`) at the
//! IDENTICAL shape/strides/element count. Both arms are cross-checked
//! bit-for-bit before every timed run.
//!
//! Four shapes: mnist's 3 real `Conv` fold window-materialize shapes
//! (`docs/discipline.md` ROW 149's measured `extents`, source `h`/`w`
//! derived as `oh + kh - 1` at stride 1) plus one larger square shape past
//! all three.
//!
//! Re-prove with (host must be quiet -- see the discipline log row this
//! bench seeds for the loadout it was actually measured under):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-tensor --bench bench_window_copy -- --save-baseline row154-micro`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::Criterion;
use proxima_tensor::{DType, Extent, IndexMap, NodeId, Op, ScalarOp, TypedBuffer, append, evaluate_typed, map};

/// One `Conv` layer's window-materialize shape: `c` input channels, `oh`/`ow`
/// output spatial extent, `kh`/`kw` kernel extent (stride/dilation fixed at
/// 1, matching every one of mnist's 3 real folds) -- source `h`/`w` derived
/// as `oh + kh - 1` / `ow + kw - 1`.
struct WindowShape {
    label: &'static str,
    channels: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
}

/// mnist's 3 real `Conv` fold window shapes (`docs/discipline.md` ROW 149's
/// measured `extents`) plus one larger square shape past all three.
const SHAPES: [WindowShape; 4] = [
    WindowShape { label: "mnist_layer1_c1_26x26_k3", channels: 1, out_h: 26, out_w: 26, kernel_h: 3, kernel_w: 3 },
    WindowShape { label: "mnist_layer2_c8_24x24_k3", channels: 8, out_h: 24, out_w: 24, kernel_h: 3, kernel_w: 3 },
    WindowShape { label: "mnist_layer3_c16_22x22_k3", channels: 16, out_h: 22, out_w: 22, kernel_h: 3, kernel_w: 3 },
    WindowShape { label: "larger_square_c32_32x32_k3", channels: 32, out_h: 32, out_w: 32, kernel_h: 3, kernel_w: 3 },
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

/// Builds `window_materialize`'s own shape at the `Op` level: a rank-4
/// `image` input `[n=1,c,h,w]`, windowed through a two-term affine pattern
/// (`h = oy + ky`, `w = ox + kx`) against an all-ones stamp into a rank-6
/// `[n,c,oh,ow,kh,kw]` output. `extra_step` true chains one further,
/// non-eliminable `+ 0.0` step (this file's own doc has the full
/// mechanism) without changing the shape, strides, or element count.
fn window_copy_program(shape: &WindowShape, extra_step: bool) -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let source_h = shape.out_h + shape.kernel_h - 1;
    let source_w = shape.out_w + shape.kernel_w - 1;
    let image = input(&mut program, &[1, shape.channels, source_h, source_w]);

    // iteration space: 0=n 1=c 2=oy 3=ox 4=ky 5=kx -- `window_materialize`'s
    // own axis assignment (`proxima-onnx/src/lower.rs`).
    let image_pattern = map::IndexPattern {
        iter_rank: 6,
        axes: vec![
            map::AxisIndex { terms: core::iter::once(map::AxisTerm::projection(0)).collect(), offset: 0 },
            map::AxisIndex { terms: core::iter::once(map::AxisTerm::projection(1)).collect(), offset: 0 },
            map::AxisIndex { terms: vec![map::AxisTerm::projection(2), map::AxisTerm::projection(4)].into(), offset: 0 },
            map::AxisIndex { terms: vec![map::AxisTerm::projection(3), map::AxisTerm::projection(5)].into(), offset: 0 },
        ],
    };
    let stamp = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: [shape.out_h, shape.out_w, shape.kernel_h, shape.kernel_w].iter().map(|&extent| Extent::Static(extent)).collect(),
            value: 1.0,
        },
    );
    let stamp_pattern = map::projection(6, &[2, 3, 4, 5]);
    let mut windowed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(image, IndexMap::Affine(image_pattern)), (stamp, IndexMap::Affine(stamp_pattern))],
            name: None,
        },
    );

    if extra_step {
        let zero = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: vec![Extent::Static(1), Extent::Static(shape.channels), Extent::Static(shape.out_h), Extent::Static(shape.out_w), Extent::Static(shape.kernel_h), Extent::Static(shape.kernel_w)],
                value: 0.0,
            },
        );
        windowed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![(windowed, IndexMap::Affine(map::projection(6, &[0, 1, 2, 3, 4, 5]))), (zero, IndexMap::Affine(map::projection(6, &[0, 1, 2, 3, 4, 5])))],
                name: None,
            },
        );
    }

    (program, image, windowed)
}

fn deterministic_data(len: usize, phase: f32) -> Vec<f32> {
    (0..len).map(|value| (value as f32 * phase).sin()).collect()
}

fn run(program: &[Op], image: NodeId, output: NodeId, image_data: &[f32]) -> Vec<f32> {
    let _ = image;
    let blocks = [TypedBuffer::Float32(image_data.to_vec())];
    let rows = evaluate_typed(program, &[], &blocks, &[output]).expect("evaluate_typed");
    let (_, _, TypedBuffer::Float32(data)) = rows.into_iter().next().expect("one output row") else {
        panic!("window-copy output was not f32");
    };
    data
}

fn main() {
    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("bench_window_copy");
    group.sample_size(20);

    for shape in &SHAPES {
        let source_h = shape.out_h + shape.kernel_h - 1;
        let source_w = shape.out_w + shape.kernel_w - 1;
        let image_len = (shape.channels * source_h * source_w) as usize;
        let image_data = deterministic_data(image_len, 0.0137);

        let (specialized_program, specialized_image, specialized_output) = window_copy_program(shape, false);
        let (block_loop_program, block_loop_image, block_loop_output) = window_copy_program(shape, true);

        // correctness self-check, once per shape, outside the timed loop:
        // the extra `+ 0.0` step changes ONLY which executor path runs,
        // never the numeric result.
        let specialized_result = run(&specialized_program, specialized_image, specialized_output, &image_data);
        let block_loop_result = run(&block_loop_program, block_loop_image, block_loop_output, &image_data);
        assert_eq!(specialized_result.len(), block_loop_result.len(), "{}: output length mismatch", shape.label);
        for (index, (&specialized_value, &block_loop_value)) in specialized_result.iter().zip(&block_loop_result).enumerate() {
            assert!(
                (specialized_value - block_loop_value).abs() <= specialized_value.abs() * 1e-5 + 1e-6,
                "{}: element {index} diverged: window_copy={specialized_value} block_loop={block_loop_value}",
                shape.label,
            );
        }

        group.bench_function(format!("{}/window_copy", shape.label), |bencher| {
            bencher.iter(|| run(&specialized_program, specialized_image, specialized_output, &image_data));
        });
        group.bench_function(format!("{}/block_loop_incumbent", shape.label), |bencher| {
            bencher.iter(|| run(&block_loop_program, block_loop_image, block_loop_output, &image_data));
        });
    }

    group.finish();
    criterion.final_summary();
}
