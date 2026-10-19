//! BF16/F16 storage through the dense tiled Metal lowering, executed on the
//! device and checked against an f64 fold of the original f32 inputs.
//!
//! The bound is derived, not tuned. Each stored element carries relative
//! rounding error at most `u = 2^-p` (`p` = significand bits including the
//! implicit one: 8 for bf16, 11 for f16, 24 for f32). A product of stored
//! values then differs from the true product by at most
//! `(1+u_a)(1+u_b)-1` of `|a*b|`, and accumulating K terms in f32 in any
//! order adds at most `gamma_K = K*2^-24/(1-K*2^-24)` of `sum |stored a*b|`.
//! Inputs keep magnitude in [0.25, 1] so every value is a normal number in
//! both half formats and the relative bound holds.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::{bf16, f16};
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_primitives::Codec;
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, ReduceInit, ScalarOp,
};

const TOKENS: u32 = 197;
const FEATURES: u32 = 45;
const REDUCTION: u32 = 131;

fn weight_first_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(FEATURES), Extent::Static(REDUCTION)],
        "weight",
    );
    let activation = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(TOKENS), Extent::Static(REDUCTION)],
        "activation",
    );
    let product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weight, "fi->tfi"), (activation, "ti->tfi")],
    )
    .expect("weight-first product has compatible axes");
    let output = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tfi->tfi",
        "tf->tfi",
    )
    .expect("reduction keeps token and feature axes");
    (program, output)
}

fn unit_roundoff(codec: Option<Codec>) -> f64 {
    match codec {
        Some(Codec::BFloat16) => 2f64.powi(-8),
        Some(Codec::Float16) => 2f64.powi(-11),
        None => 0.0,
        Some(other) => panic!("fixture stores only bf16, f16 or f32, got {other:?}"),
    }
}

fn values(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| {
            let magnitude = 0.25 + 0.75 * lcg.next_unit().abs();
            if lcg.next_unit() < 0.5 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

fn pack(values: &[f32], codec: Codec) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| match codec {
            Codec::BFloat16 => bf16::from_f32(*value).to_bits().to_le_bytes(),
            Codec::Float16 => f16::from_f32(*value).to_bits().to_le_bytes(),
            other => panic!("fixture packs only bf16 and f16, got {other:?}"),
        })
        .collect()
}

fn error_bound(weight: &[f32], activation: &[f32], codecs: [Option<Codec>; 2]) -> Vec<f64> {
    let weight_u = unit_roundoff(codecs[0]);
    let activation_u = unit_roundoff(codecs[1]);
    let storage = (1.0 + weight_u) * (1.0 + activation_u) - 1.0;
    let accumulation_u = 2f64.powi(-24) * f64::from(REDUCTION);
    let gamma = accumulation_u / (1.0 - accumulation_u);
    let k = REDUCTION as usize;
    (0..TOKENS as usize)
        .flat_map(|token| (0..FEATURES as usize).map(move |feature| (token, feature)))
        .map(|(token, feature)| {
            let abs_sum: f64 = (0..k)
                .map(|index| {
                    f64::from(weight[feature * k + index] * activation[token * k + index]).abs()
                })
                .sum();
            abs_sum * (storage + gamma * (1.0 + storage))
        })
        .collect()
}

fn reference(weight: &[f32], activation: &[f32]) -> Vec<f64> {
    let k = REDUCTION as usize;
    (0..TOKENS as usize)
        .flat_map(|token| (0..FEATURES as usize).map(move |feature| (token, feature)))
        .map(|(token, feature)| {
            (0..k)
                .map(|index| {
                    f64::from(weight[feature * k + index])
                        * f64::from(activation[token * k + index])
                })
                .sum()
        })
        .collect()
}

fn worst_ratio(actual: &[f32], expected: &[f64], bound: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "coordinate counts must match");
    assert!(!actual.is_empty(), "comparison must see coordinates");
    actual
        .iter()
        .zip(expected)
        .zip(bound)
        .map(|((actual, expected), bound)| (f64::from(*actual) - expected).abs() / bound)
        .fold(0.0, f64::max)
}

fn block<'data>(
    codec: Option<Codec>,
    bytes: &'data Option<Vec<u8>>,
    floats: &'data [f32],
) -> QuantizedBlock<'data> {
    match (codec, bytes) {
        (Some(codec), Some(bytes)) => QuantizedBlock::Packed { codec, bytes },
        _ => QuantizedBlock::Float32(floats),
    }
}

fn run(policy: NumericPolicy, codecs: [Option<Codec>; 2]) -> (Vec<f32>, Vec<String>) {
    let (program, output) = weight_first_program();
    let weight = values(11, (FEATURES * REDUCTION) as usize);
    let activation = values(29, (TOKENS * REDUCTION) as usize);
    let weight_bytes = codecs[0].map(|codec| pack(&weight, codec));
    let activation_bytes = codecs[1].map(|codec| pack(&activation, codec));
    let bindings = [
        ("weight", block(codecs[0], &weight_bytes, &weight)),
        (
            "activation",
            block(codecs[1], &activation_bytes, &activation),
        ),
    ];
    let config = RuntimeConfig {
        engine: "gpu".into(),
        gpu_driver: "metal".into(),
        allow_fusion: true,
        numeric: policy,
        ..RuntimeConfig::default()
    };
    let (mut plan, records) = plan_named_with_config(&config, &program, &[], &bindings, &[output])
        .expect("metal prepares the weight-first matmul");
    let evaluated = execute_plan_named(&mut plan, &bindings).expect("metal executes the matmul");
    let (actual, shape) = evaluated.get(output).expect("root is present");
    assert_eq!(shape, &[u64::from(TOKENS), u64::from(FEATURES)]);
    (actual.to_vec(), records[0].kernel_identities.clone())
}

fn assert_within_derived_bound(codecs: [Option<Codec>; 2]) {
    let weight = values(11, (FEATURES * REDUCTION) as usize);
    let activation = values(29, (TOKENS * REDUCTION) as usize);
    let expected = reference(&weight, &activation);
    let effective_codecs = if codecs.contains(&Some(Codec::BFloat16)) {
        [Some(Codec::BFloat16), Some(Codec::BFloat16)]
    } else {
        codecs
    };
    let bound = error_bound(&weight, &activation, effective_codecs);

    let (actual, identities) = run(NumericPolicy::llama_relaxed(), codecs);
    let expected_fragment = if codecs.contains(&Some(Codec::BFloat16)) {
        "_mma_bf16"
    } else {
        "_mma_f32"
    };
    let dense_tiled = identities
        .iter()
        .any(|identity| identity.contains("_dbg") && identity.contains(expected_fragment));
    assert!(
        dense_tiled,
        "dense tiled admission must not depend on the build-time float-storage gate: {identities:?}"
    );
    let ratio = worst_ratio(&actual, &expected, &bound);
    println!(
        "codecs={codecs:?} dense_tiled={dense_tiled} worst_error_over_bound={ratio} kernels={identities:?}"
    );
    assert!(ratio <= 1.0, "error exceeded the derived bound: {ratio}");

    let tight_control = error_bound(&weight, &activation, [None, None]);
    if codecs != [None, None] {
        assert!(
            worst_ratio(&actual, &expected, &tight_control) > 1.0,
            "an f32-only bound must reject storage rounding, or the check measures nothing"
        );
    }

    let mut corrupted = actual.clone();
    corrupted[7] += (bound[7] * 4.0) as f32;
    assert!(
        worst_ratio(&corrupted, &expected, &bound) > 1.0,
        "a one-coordinate corruption must exceed the bound"
    );
}

#[test]
fn programmable_tiled_bf16_weight_matches_f32_within_derived_bound() {
    assert_within_derived_bound([Some(Codec::BFloat16), None]);
}

#[test]
fn programmable_tiled_f16_weight_matches_f32_within_derived_bound() {
    assert_within_derived_bound([Some(Codec::Float16), None]);
}

#[test]
fn programmable_tiled_bf16_both_operands_match_f32_within_derived_bound() {
    assert_within_derived_bound([Some(Codec::BFloat16), Some(Codec::BFloat16)]);
}

#[test]
fn programmable_tiled_mixed_storage_matches_f32_within_derived_bound() {
    assert_within_derived_bound([Some(Codec::Float16), Some(Codec::BFloat16)]);
}

#[test]
fn programmable_tiled_f32_operands_match_within_accumulation_bound() {
    assert_within_derived_bound([None, None]);
}

#[test]
fn programmable_tiled_bit_exact_policy_declines_the_tiled_kernel() {
    let codecs = [Some(Codec::BFloat16), None];
    let weight = values(11, (FEATURES * REDUCTION) as usize);
    let activation = values(29, (TOKENS * REDUCTION) as usize);
    let (actual, identities) = run(NumericPolicy::bit_exact(), codecs);
    assert!(
        !identities.iter().any(|identity| identity.contains("_dbg")),
        "bit-exact policy must not select the tiled kernel: {identities:?}"
    );
    let ratio = worst_ratio(
        &actual,
        &reference(&weight, &activation),
        &error_bound(&weight, &activation, codecs),
    );
    assert!(
        ratio <= 1.0,
        "fallback error exceeded the derived bound: {ratio}"
    );
}
