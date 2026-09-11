//! `push_q6k_ggml_port_body`'s activation address shares
//! `push_packed_row_plain_product_y4_address` with `push_q4k_ggml_port_body`,
//! so it takes the SAME `other_stride_is_one` stride-free specialization --
//! see that function's own doc. One codec over from
//! `q4k_ggml_port_strided_activation.rs`: a unit-stride activation (the
//! specialization's own path) and a `NaN`-poisoned non-unit-stride
//! activation (`q6k_matmul_strided_activation.rs`'s own technique) -- a
//! wrong-stride read on either branch lands on a poisoned slot or disagrees
//! with the independent dequantize+dot reference.

#![cfg(all(feature = "metal-q4k-ggml-port", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, affine, append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// One `Multiply`-then-`Add` matmul (`q6k_matmul_strided_activation.rs`'s own
/// shape) with the activation read through a reduce-axis stride of `stride`
/// rather than a plain projection -- `stride == 1` degenerates to a plain
/// contiguous read (`other_stride_is_one` true), any larger value forces the
/// generic multiply-by-`other_stride` arm.
fn matmul_program_strided_activation(in_dim: u32, out_dim: u32, stride: i32) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(in_dim * stride as u32)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                (
                    activation,
                    IndexMap::Affine(affine(2, &[(&[AxisTerm::scaled(1, stride)], 0)])),
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

/// Interleaves `values` into a `[len * stride]` buffer, every other slot
/// poisoned with `f32::NAN` -- a wrong-stride read lands on a poisoned slot
/// and every downstream sum becomes `NaN`.
fn interleave_with_poison(values: &[f32], stride: usize) -> Vec<f32> {
    let mut buffer = vec![f32::NAN; values.len() * stride];
    for (index, value) in values.iter().enumerate() {
        buffer[index * stride] = *value;
    }
    buffer
}

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

fn assert_ggml_port_matches_reference_at_stride(label: &str, stride: i32) {
    const IN_DIM: usize = 512;
    const OUT_DIM: usize = 3;

    let rows: Vec<Vec<f32>> = (0..OUT_DIM)
        .map(|row| random_vec(31 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(&rows, IN_DIM);
    let real_activation = random_vec(101, IN_DIM);
    let physical_activation = if stride == 1 {
        real_activation.clone()
    } else {
        interleave_with_poison(&real_activation, stride as usize)
    };
    let expected = expected_output(&packed, IN_DIM, OUT_DIM, &real_activation);

    let (program, sum) = matmul_program_strided_activation(IN_DIM as u32, OUT_DIM as u32, stride);
    let blocks = [
        QuantizedBlock::Q6K(&packed),
        QuantizedBlock::Float32(&physical_activation),
    ];

    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
        .expect("metal plans the matmul");
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
            "{label} row {index}: metal={metal_value} is not finite -- a wrong-stride read landed \
             on a poisoned NaN activation slot in `push_q6k_ggml_port_body`'s stride-{stride} arm"
        );
        let scale = reference.abs().max(f32::MIN_POSITIVE);
        let relative = (metal_value - reference).abs() / scale;
        assert!(
            relative < 1e-2,
            "{label} row {index}: metal={metal_value} disagrees with the independent \
             dequantize+dot reference={reference} (relative={relative}) -- \
             `push_q6k_ggml_port_body`'s stride-{stride} activation addressing defect if it fires"
        );
    }
}

#[test]
fn ggml_port_matches_the_reference_with_a_unit_stride_contiguous_activation() {
    assert_ggml_port_matches_reference_at_stride("unit-stride", 1);
}

#[test]
fn ggml_port_matches_the_reference_against_an_interleaved_non_unit_stride_activation() {
    assert_ggml_port_matches_reference_at_stride("stride-3", 3);
}
