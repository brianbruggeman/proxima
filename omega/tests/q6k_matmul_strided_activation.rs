//! The strided-activation arm `push_packed_row_blocked_body`'s generic
//! (non-plain-product) path never had: every existing packed-row test
//! (`q4k_matmul_layout.rs`, `q6k_real_checkpoint_parity.rs`) binds the
//! activation as a plain contiguous 1-D vector, so the reduce-axis stride is
//! trivially 1 and the STRIDE-FREE SPECIALIZATION in
//! `push_packed_row_blocked_body` (see that function's own doc,
//! `docs/discipline.md` perf/packed-row-addressing row) never has a
//! non-unit-stride input to prove its `else` branch against.
//!
//! This binds the activation INTERLEAVED -- physically `[in_dim * STRIDE]`,
//! real values every `STRIDE`-th element, `f32::NAN` poison everywhere else
//! -- via an explicit `AxisTerm::scaled` affine map, the same "slice /
//! stride / dilation" grammar `proxima_tensor::map`'s own doc table names.
//! A wrong-stride read (e.g. a regression that always takes the stride-free
//! form) lands on a poisoned NaN slot and the comparison against the
//! independent dequantize+dot reference fails loudly instead of silently.
//!
//! `Q6_K`'s pair-dot arm is selected by structure
//! (`PackedCodec::supports_pair_dot`), not a cargo feature, so this op
//! always exercises the plain-product `y4` pointer's own stride-aware path
//! (`other_stride_is_one` false here) rather than the GENERIC arm's
//! `acts_row` hoist -- both arms share the same `other_stride`-aware
//! addressing fix this file's own module doc names, so either one proves
//! the claim; this fixture happens to land on the plain-product one.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, affine, append, projection,
};

/// How far apart consecutive activation elements sit in the physical
/// buffer -- the reduce-axis stride `push_packed_row_blocked_body`'s
/// `other_stride_is_one` gate must see as `false` for this fixture.
const ACTIVATION_STRIDE: i32 = 3;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// One `Multiply`-then-`Add` matmul, weight declared `[in_dim, out_dim]`
/// (`q4k_matmul_layout.rs`'s own reduction-axis-first convention) with the
/// activation read through a non-unit reduce-axis stride instead of a plain
/// projection.
fn matmul_program_strided_activation(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(in_dim * ACTIVATION_STRIDE as u32)],
            name: None,
        },
    );
    // iteration space (o, i): axis 0 = out (survives), axis 1 = in (reduced).
    // The activation's one physical axis reads iteration axis 1 scaled by
    // `ACTIVATION_STRIDE` -- `proxima_tensor::map`'s own "stride / dilation"
    // affine form, not a projection -- so its `Layout` stride at the reduce
    // dim comes out as `ACTIVATION_STRIDE`, not 1.
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                (
                    activation,
                    IndexMap::Affine(affine(2, &[(&[AxisTerm::scaled(1, ACTIVATION_STRIDE)], 0)])),
                ),
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
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

/// Packs `out_dim` independent rows of `in_dim` elements each into GGUF's
/// native `[out_dim, in_dim]` row-major byte layout, exactly
/// `q4k_matmul_layout.rs`'s own `pack_rows` one codec over.
fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let blocks_per_row = in_dim / QK_K;
    let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row, row_packed).expect("in_dim is a whole multiple of QK_K");
    }
    packed
}

/// Interleaves `values` into a `[len * ACTIVATION_STRIDE]` buffer, every
/// other slot poisoned with `f32::NAN` -- a wrong-stride read (reading
/// element `i` instead of `i * ACTIVATION_STRIDE`) lands on a poisoned slot
/// and every downstream sum becomes `NaN`, which `assert!` below catches
/// unconditionally regardless of the relative-error threshold.
fn interleave_with_poison(values: &[f32]) -> Vec<f32> {
    let mut buffer = vec![f32::NAN; values.len() * ACTIVATION_STRIDE as usize];
    for (index, value) in values.iter().enumerate() {
        buffer[index * ACTIVATION_STRIDE as usize] = *value;
    }
    buffer
}

/// Dequantizes `packed` row by row and dots each row against the REAL
/// (non-poisoned) `activation` values -- computed independently of both
/// `proxima_tensor::cpu`'s quantized matmul path and omega's Metal emitter.
fn expected_output(packed: &[u8], in_dim: usize, out_dim: usize, activation: &[f32]) -> Vec<f32> {
    let blocks_per_row = in_dim / QK_K;
    let mut expected = Vec::with_capacity(out_dim);
    for row_packed in packed.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut row = vec![0.0f32; in_dim];
        dequantize(row_packed, &mut row).expect("packed row dequantizes");
        let dot: f32 = row
            .iter()
            .zip(activation.iter())
            .map(|(weight, value)| weight * value)
            .sum();
        expected.push(dot);
    }
    assert_eq!(
        expected.len(),
        out_dim,
        "degenerate fixture: one row per output element"
    );
    expected
}

#[test]
fn metal_agrees_with_the_independent_reference_on_a_q6k_weight_against_an_interleaved_non_unit_stride_activation()
 {
    const IN_DIM: usize = 256;
    const OUT_DIM: usize = 3;

    let rows: Vec<Vec<f32>> = (0..OUT_DIM)
        .map(|row| random_vec(31 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(&rows, IN_DIM);
    let real_activation = random_vec(101, IN_DIM);
    let interleaved_activation = interleave_with_poison(&real_activation);
    let expected = expected_output(&packed, IN_DIM, OUT_DIM, &real_activation);

    let (program, sum) = matmul_program_strided_activation(IN_DIM as u32, OUT_DIM as u32);
    let blocks = [
        QuantizedBlock::Q6K(&packed),
        QuantizedBlock::Float32(&interleaved_activation),
    ];

    // `proxima_tensor::cpu::evaluate_quantized` is deliberately NOT
    // exercised as a second oracle here: its own specialized quantized-
    // matmul dot-product fast path (`build_matmul_stage_plan`/
    // `run_reduce_quantized`) derives the activation's batch count from the
    // RAW buffer length divided by the packed weight's row width
    // (`cpu.rs`'s "quantized matmul batch shape does not evenly divide by
    // its packed weight rows" `NotLowerable`), a CPU-side contiguity
    // assumption independent of this landing's Metal addressing and out of
    // this row's scope. The independent dequantize+dot reference below is
    // oracle enough for the Metal claim this test exists to make.
    let plan = omega::plan(&program, &[], &blocks, &[sum]).expect("metal plans the matmul");
    let metal =
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");

    let metal_root = metal.root();
    assert_eq!(
        metal_root.len(),
        OUT_DIM,
        "degenerate gate: metal produced no output"
    );

    for (index, (&metal_value, &reference)) in metal_root.iter().zip(expected.iter()).enumerate() {
        assert!(
            metal_value.is_finite(),
            "row {index}: metal={metal_value} is not finite -- a wrong-stride read landed on a \
             poisoned NaN activation slot (this is the stride-free-specialization defect this \
             test exists to catch if it fires)"
        );
        let scale = reference.abs().max(f32::MIN_POSITIVE);
        let metal_relative = (metal_value - reference).abs() / scale;
        assert!(
            metal_relative < 1e-2,
            "row {index}: metal={metal_value} disagrees with the independent dequantize+dot reference={reference} \
             (relative={metal_relative}) -- non-unit-stride activation addressing defect if it fires"
        );
    }
}
