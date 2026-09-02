//! The gate that was missing: every existing omega test binds `Float32`
//! operands only (`support::real_forward_fixture`, `metal_parity.rs`,
//! `backend_parity.rs`), and `q4k_matvec_probe.rs`'s own synthetic weight
//! shape (`[rows, k]`, output axis first) happens to already match a packed
//! weight's physical byte order -- so none of them can catch a Q4_K weight
//! bound in the OTHER declared axis order, reduction-axis-first, which is
//! the order every real matmul weight in
//! `proxima_tensor::spec::mistral_cached_forward_program` actually uses
//! (`wq`'s own declared shape is `[embedding, heads, head_dim]`: the
//! contraction axis `embedding` comes first, not last).
//!
//! This binds ONE `Q4_K` matmul weight in exactly that declared order and
//! checks both backends against an independent reference (dequantize each
//! packed row by hand, dot it against the activation) -- not just against
//! each other, so two backends agreeing on the same wrong answer cannot
//! pass this gate.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::cpu::evaluate_quantized;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// One `Multiply`-then-`Add` matmul, weight declared `[in_dim, out_dim]` --
/// the mistral spec's own convention (reduction axis first, output axis
/// after), the opposite of `q4k_matvec_probe.rs`'s `[rows, k]`.
fn matmul_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(in_dim)],
            name: None,
        },
    );
    // iteration space (o, i): axis 0 = out (survives), axis 1 = in (reduced).
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                (activation, IndexMap::Affine(projection(2, &[1]))),
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
/// native `[out_dim, in_dim]` row-major byte layout (`out_dim` rows, each a
/// contiguous run of `in_dim` elements) -- exactly what
/// `proxima-model-interop::bind_matmul_weight` leaves a `Q4_K` tensor's
/// bytes in, untransposed.
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

/// Dequantizes `packed` row by row and dots each row against `activation` --
/// the reference this test holds both backends to, computed independently
/// of both `proxima_tensor::cpu`'s quantized matmul path and omega's Metal
/// emitter.
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
fn metal_agrees_with_cpu_and_the_independent_reference_on_a_q4k_weight_declared_reduction_axis_first()
 {
    const IN_DIM: usize = 512;
    const OUT_DIM: usize = 3;

    let rows: Vec<Vec<f32>> = (0..OUT_DIM)
        .map(|row| random_vec(17 + row as u64, IN_DIM))
        .collect();
    let packed = pack_rows(&rows, IN_DIM);
    let activation = random_vec(97, IN_DIM);
    let expected = expected_output(&packed, IN_DIM, OUT_DIM, &activation);

    let (program, sum) = matmul_program(IN_DIM as u32, OUT_DIM as u32);
    let blocks = [
        QuantizedBlock::Q4K(&packed),
        QuantizedBlock::Float32(&activation),
    ];

    let cpu = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("cpu runs the matmul");
    let plan = omega::plan(&program, &[], &blocks, &[sum]).expect("metal plans the matmul");
    let metal =
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");

    let cpu_root = cpu.root();
    let metal_root = metal.root();
    assert_eq!(
        cpu_root.len(),
        OUT_DIM,
        "degenerate gate: cpu produced no output"
    );
    assert_eq!(
        metal_root.len(),
        OUT_DIM,
        "degenerate gate: metal produced no output"
    );

    // relative, not absolute: both quantized-matmul paths fold 512 terms in a
    // different order than this test's own plain `f32` dot (int8-dot SIMD
    // lanes vs a straight-line sum), so a few ULPs of fp32 accumulation drift
    // is expected and is not the layout defect this test exists to catch --
    // that defect reads the WRONG BYTES, which misses by orders of
    // magnitude, not fractions of a percent.
    for (index, ((&cpu_value, &metal_value), &reference)) in cpu_root
        .iter()
        .zip(metal_root.iter())
        .zip(expected.iter())
        .enumerate()
    {
        let scale = reference.abs().max(f32::MIN_POSITIVE);
        let cpu_relative = (cpu_value - reference).abs() / scale;
        let metal_relative = (metal_value - reference).abs() / scale;
        assert!(
            cpu_relative < 1e-2,
            "row {index}: cpu={cpu_value} disagrees with the independent dequantize+dot reference={reference} \
             (relative={cpu_relative})"
        );
        assert!(
            metal_relative < 1e-2,
            "row {index}: metal={metal_value} disagrees with the independent dequantize+dot reference={reference} \
             (relative={metal_relative}) -- this is the reduction-axis-first Q4_K layout defect if it fires"
        );
    }
}
