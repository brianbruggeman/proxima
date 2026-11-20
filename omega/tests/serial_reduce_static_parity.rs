//! The serial reduce body bakes its reduced extents as literals and walks each
//! operand by a hoisted base offset plus a per-step stride. The fold order is
//! still `k = 0..n-1`, so a Metal run must equal `cpu::evaluate` bit for bit.
//! The shape is flint's MoE combine: `down[s, k, o] * w[s, k]` summed over the
//! six selected experts, 512 tokens by 2048 columns.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};

const TOKENS: u32 = 512;
const EXPERTS: u32 = 6;
const COLUMNS: u32 = 2048;

fn combine_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let down = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(TOKENS), Extent::Static(EXPERTS), Extent::Static(COLUMNS)],
            name: None,
        },
    );
    let weights = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(TOKENS), Extent::Static(EXPERTS)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (down, IndexMap::Affine(projection(3, &[0, 1, 2]))),
                (weights, IndexMap::Affine(projection(3, &[0, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

#[test]
fn moe_combine_fold_over_six_experts_equals_the_cpu_oracle_bit_for_bit() {
    let down = random_vec(11, (TOKENS * EXPERTS * COLUMNS) as usize);
    let weights = random_vec(23, (TOKENS * EXPERTS) as usize);
    let (program, sum) = combine_program();
    let blocks = [QuantizedBlock::Float32(&down), QuantizedBlock::Float32(&weights)];

    let metal = omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::llama_relaxed())
        .expect("metal runs the combine fold");
    let cpu = proxima_tensor::cpu::evaluate(&program, &[], &[&down, &weights], &[sum])
        .expect("cpu evaluates the combine fold");

    let actual: Vec<u32> = metal.root().iter().map(|value| value.to_bits()).collect();
    let expected: Vec<u32> = cpu.root().iter().map(|value| value.to_bits()).collect();
    assert_eq!(expected.len(), (TOKENS * COLUMNS) as usize, "the oracle must produce every output");
    let mismatches = actual.iter().zip(&expected).filter(|(left, right)| left != right).count();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(mismatches, 0, "metal differs from the cpu oracle in {mismatches} outputs");
}
