//! The int8 activation path lowers every codec through `block_dot_i8` or `sub_block_dot`; the codecs that lowered
//! before the sub-block lowering existed (Q8_0, Q5_0, Pq2_0) must keep emitting the same bytes.
//! The fixtures under `tests/fixtures/int8_emit` were captured from the commit that preceded
//! `sub_block_dot`, through the same derive-then-emit sequence a plan runs; the k-quant fixtures were
//! captured from the emitter at the commit that introduced them and pin their kernel text.

use proxima_tensor::{
    BoundOp, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append,
    bind_with_fusion, correct_packed_matmul_layouts, infer, projection,
};

use crate::backend::GpuDriver;
use crate::cuda::emit_cuda_with_policy;
use crate::derived_quantize::derive_block_quantize;
use crate::lowering::{Codec, PackedOperands};

const ROWS: u32 = 8;
const DEPTH: u32 = 512;
const TOKENS: u32 = 64;

fn gemm_program(tokens: u32) -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let activation = append(&mut program, input(DType::FLOAT32, tokens));
    let weight = append(&mut program, input(DType::UINT8, ROWS));
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
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
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, weight, sum)
}

fn input(dtype: DType, rows: u32) -> Op {
    Op::Input { dtype, shape: vec![Extent::Static(rows), Extent::Static(DEPTH)], name: None }
}

fn emitted_sources(codec: Codec, tokens: u32) -> String {
    let (program, weight, sum) = gemm_program(tokens);
    let policy = NumericPolicy::llama_relaxed().with_activation_quantization(true);
    let shapes = infer(&program, &[]).expect("snapshot program infers");
    let mut packed = PackedOperands::from([(weight, codec)]);
    let mut resolved = bind_with_fusion(&program, &shapes, &[sum], policy.reassociation, policy).expect("binds");
    correct_packed_matmul_layouts(&mut resolved, &packed.keys().copied().collect());
    let derived = derive_block_quantize(resolved, &program, &shapes, &mut packed, policy, GpuDriver::Cuda);
    derived.iter().map(|bound| emitted(bound, &packed, policy)).collect()
}

fn emitted(bound: &BoundOp, packed: &PackedOperands, policy: NumericPolicy) -> String {
    let kernel = emit_cuda_with_policy(bound, packed, policy).expect("snapshot op emits");
    let prologue: String = kernel.prologue.iter().map(|stage| stage.source.as_str()).collect();
    format!("{prologue}{}", kernel.source)
}

#[test]
fn pre_sub_block_codecs_emit_the_bytes_the_fixtures_captured() {
    let cases = [
        ("q8_0_matvec", Codec::Q8_0, 1, include_str!("../tests/fixtures/int8_emit/q8_0_matvec.cu")),
        ("q8_0_gemm", Codec::Q8_0, TOKENS, include_str!("../tests/fixtures/int8_emit/q8_0_gemm.cu")),
        ("q5_0_matvec", Codec::Q5_0, 1, include_str!("../tests/fixtures/int8_emit/q5_0_matvec.cu")),
        ("q5_0_gemm", Codec::Q5_0, TOKENS, include_str!("../tests/fixtures/int8_emit/q5_0_gemm.cu")),
        ("pq2_0_matvec", Codec::Pq2_0, 1, include_str!("../tests/fixtures/int8_emit/pq2_0_matvec.cu")),
        ("pq2_0_gemm", Codec::Pq2_0, TOKENS, include_str!("../tests/fixtures/int8_emit/pq2_0_gemm.cu")),
    ];
    for (name, codec, tokens, expected) in cases {
        assert_eq!(emitted_sources(codec, tokens), expected, "{name} drifted from its fixture");
    }
}

#[test]
fn k_quant_codecs_emit_the_bytes_the_fixtures_captured() {
    let cases = [
        ("q4_k_matvec", Codec::Q4K, 1, include_str!("../tests/fixtures/int8_emit/q4_k_matvec.cu")),
        ("q4_k_gemm", Codec::Q4K, TOKENS, include_str!("../tests/fixtures/int8_emit/q4_k_gemm.cu")),
        ("q5_k_matvec", Codec::Q5K, 1, include_str!("../tests/fixtures/int8_emit/q5_k_matvec.cu")),
        ("q5_k_gemm", Codec::Q5K, TOKENS, include_str!("../tests/fixtures/int8_emit/q5_k_gemm.cu")),
        ("q6_k_matvec", Codec::Q6K, 1, include_str!("../tests/fixtures/int8_emit/q6_k_matvec.cu")),
        ("q6_k_gemm", Codec::Q6K, TOKENS, include_str!("../tests/fixtures/int8_emit/q6_k_gemm.cu")),
    ];
    for (name, codec, tokens, expected) in cases {
        assert_eq!(emitted_sources(codec, tokens), expected, "{name} drifted from its fixture");
    }
}

#[test]
fn the_snapshot_forms_are_the_int8_forms() {
    for codec in [Codec::Q8_0, Codec::Q5_0, Codec::Pq2_0, Codec::Q4K, Codec::Q5K, Codec::Q6K] {
        assert!(emitted_sources(codec, 1).contains("__dp4a"), "{codec:?} matvec is not the dp4a form");
        assert!(emitted_sources(codec, TOKENS).contains("mma.sync"), "{codec:?} gemm is not the int8 mma form");
    }
}
