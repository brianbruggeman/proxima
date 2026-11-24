//! The int8 activation path lowers every codec through `block_dot_i8` or `sub_block_dot`; the codecs that lowered
//! before the sub-block lowering existed (Q8_0, Q5_0, Pq2_0) must keep emitting the same bytes.
//! The fixtures under `tests/fixtures/int8_emit` pin the emitter text: the k-quant fixtures were captured
//! at the commit that introduced them, and every fixture was regenerated on 2026-11-21 when the card-C
//! uniform word and 64-bit index types became `long long`. Re-generating them is the expected change for
//! a deliberate text change: run the module with `OMEGA_REGENERATE_INT8_EMIT=1` to rewrite them from the emitter;
//! the tests still catch any other drift.

use proxima_tensor::{BoundOp, BoundOpKind, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, bind_with_fusion, correct_packed_matmul_layouts, infer, projection};
use proxima_numeric::{DType, NumericPolicy, ScaleArithmetic};

use crate::cuda::emit_cuda_with_policy;
use crate::derived_quantize::derive_through_tensor_pass;
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

const REGENERATE: &str = "OMEGA_REGENERATE_INT8_EMIT";

fn default_policy() -> NumericPolicy {
    NumericPolicy::llama_relaxed().with_activation_quantization(true)
}

fn derived_ops(codec: Codec, tokens: u32, policy: NumericPolicy) -> (Vec<BoundOp>, PackedOperands) {
    let (program, weight, sum) = gemm_program(tokens);
    let shapes = infer(&program, &[]).expect("snapshot program infers");
    let mut packed = PackedOperands::from([(weight, codec)]);
    let mut resolved = bind_with_fusion(&program, &shapes, &[sum], policy.reassociation, policy).expect("binds");
    correct_packed_matmul_layouts(&mut resolved, &packed.keys().copied().collect());
    let derived = derive_through_tensor_pass(resolved, &program, &shapes, &mut packed, policy, None);
    (derived, packed)
}

fn emitted_sources(codec: Codec, tokens: u32, policy: NumericPolicy) -> String {
    let (derived, packed) = derived_ops(codec, tokens, policy);
    derived.iter().map(|bound| emitted(bound, &packed, policy)).collect()
}

fn emitted(bound: &BoundOp, packed: &PackedOperands, policy: NumericPolicy) -> String {
    let kernel = emit_cuda_with_policy(bound, packed, policy).expect("snapshot op emits");
    let prologue: String = kernel.prologue.iter().map(|stage| stage.source.as_str()).collect();
    format!("{prologue}{}", kernel.source)
}

fn check_fixtures(cases: &[(&str, Codec, u32)]) {
    for (name, codec, tokens) in cases {
        let path = format!("{}/tests/fixtures/int8_emit/{name}.cu", env!("CARGO_MANIFEST_DIR"));
        let actual = emitted_sources(*codec, *tokens, default_policy());
        if std::env::var_os(REGENERATE).is_some() {
            std::fs::write(&path, &actual).expect("fixture rewrites");
            continue;
        }
        let expected = std::fs::read_to_string(&path).expect("fixture reads");
        assert_eq!(actual, expected, "{name} drifted from its fixture");
    }
}

#[test]
fn pre_sub_block_codecs_emit_the_bytes_the_fixtures_captured() {
    check_fixtures(&[
        ("q8_0_matvec", Codec::Q8_0, 1),
        ("q8_0_gemm", Codec::Q8_0, TOKENS),
        ("q5_0_matvec", Codec::Q5_0, 1),
        ("q5_0_gemm", Codec::Q5_0, TOKENS),
        ("pq2_0_matvec", Codec::Pq2_0, 1),
        ("pq2_0_gemm", Codec::Pq2_0, TOKENS),
    ]);
}

#[test]
fn k_quant_codecs_emit_the_bytes_the_fixtures_captured() {
    check_fixtures(&k_quant_cases());
}

fn k_quant_cases() -> [(&'static str, Codec, u32); 6] {
    [
        ("q4_k_matvec", Codec::Q4K, 1),
        ("q4_k_gemm", Codec::Q4K, TOKENS),
        ("q5_k_matvec", Codec::Q5K, 1),
        ("q5_k_gemm", Codec::Q5K, TOKENS),
        ("q6_k_matvec", Codec::Q6K, 1),
        ("q6_k_gemm", Codec::Q6K, TOKENS),
    ]
}

#[test]
fn the_snapshot_forms_are_the_int8_forms() {
    for codec in [Codec::Q8_0, Codec::Q5_0, Codec::Pq2_0, Codec::Q4K, Codec::Q5K, Codec::Q6K] {
        assert!(emitted_sources(codec, 1, default_policy()).contains("__dp4a"), "{codec:?} matvec is not the dp4a form");
        assert!(emitted_sources(codec, TOKENS, default_policy()).contains("mma.sync"), "{codec:?} gemm is not the int8 mma form");
    }
}

fn quantize_arithmetic(derived: &[BoundOp]) -> ScaleArithmetic {
    let found: Vec<ScaleArithmetic> = derived
        .iter()
        .filter_map(|bound| match &bound.kind {
            BoundOpKind::BlockQuantize { arithmetic, .. } => Some(*arithmetic),
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "one activation quantize op is derived");
    found[0]
}

#[test]
fn no_configuration_binds_the_layout_rules_arithmetic() {
    for tokens in [1, TOKENS] {
        let (derived, _) = derived_ops(k_quant_cases()[0].1, tokens, default_policy());
        assert_eq!(quantize_arithmetic(&derived), ScaleArithmetic::InverseOfScale, "{tokens} rows");
        assert_eq!(ScaleArithmetic::for_rows(u64::from(tokens), default_policy()), ScaleArithmetic::InverseOfScale);
    }
}

#[test]
fn the_emitted_quantize_text_follows_the_bound_arithmetic() {
    let signatures = [
        (ScaleArithmetic::InverseOfScale, "d_inv = d == 0.0f ? 0.0f", "x / d"),
        (ScaleArithmetic::Quotient, "x / d", "d_inv"),
        (ScaleArithmetic::ReciprocalMultiply, "d = 1.0f / d_inv;", "d_inv = d == 0.0f"),
    ];
    for (arithmetic, present, absent) in signatures {
        let policy = default_policy().with_activation_scale_arithmetic(arithmetic);
        let (derived, packed) = derived_ops(k_quant_cases()[0].1, TOKENS, policy);
        assert_eq!(quantize_arithmetic(&derived), arithmetic, "bound arithmetic follows the policy");
        let quantize = derived.iter().find(|bound| matches!(bound.kind, BoundOpKind::BlockQuantize { .. })).expect("quantize op");
        let text = emitted(quantize, &packed, policy);
        assert!(text.contains(present), "{arithmetic:?} text lacks `{present}`");
        assert!(!text.contains(absent), "{arithmetic:?} text carries `{absent}`");
    }
}
