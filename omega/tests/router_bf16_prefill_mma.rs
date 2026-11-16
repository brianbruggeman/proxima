//! The MoE router matmul (`ffn_gate_inp`, BF16 weight) at the shape flint binds for one 512-row
//! prefill chunk -- extents `[tokens 512, reduce 2048, features 256]` -- planned with the serving
//! numeric policy (`llama_relaxed` + epilogue sources: no activation-rounding permission), executed
//! on Metal and checked against an f64 fold of the stored weight and the unrounded f32 activation.
//!
//! The bound is derived: the weight is read exactly (the reference decodes the stored bf16), the
//! activation is not rounded (f32 tiles), so the only error is f32 accumulation of K terms in any
//! order: `gamma_K * sum|w*a|` with `gamma_K = K*2^-24/(1-K*2^-24)`, K = 2048. Inputs keep magnitude
//! in [0.25, 1] so the sum of absolute products does not cancel to zero.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::bf16;
use omega::backend::{Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ReduceInit, ScalarOp};

const TOKENS: usize = 512;
const REDUCTION: usize = 2048;
const FEATURES: usize = 256;

fn router_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(FEATURES as u32), Extent::Static(REDUCTION as u32)],
        "weight",
    );
    let activation = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(TOKENS as u32), Extent::Static(REDUCTION as u32)],
        "activation",
    );
    let product = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(weight, "fi->tfi"), (activation, "ti->tfi")],
    )
    .expect("router product has compatible axes");
    let output = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tfi->tfi",
        "tf->tfi",
    )
    .expect("router reduction keeps token and feature axes");
    (program, output)
}

fn values(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| {
            let magnitude = 0.25 + 0.75 * lcg.next_unit().abs();
            if lcg.next_unit() < 0.5 { -magnitude } else { magnitude }
        })
        .collect()
}

fn fold(weight: &[f32], activation: &[f32], round_activation: bool) -> (Vec<f64>, Vec<f64>) {
    let mut sums = Vec::with_capacity(TOKENS * FEATURES);
    let mut abs_sums = Vec::with_capacity(TOKENS * FEATURES);
    for token in 0..TOKENS {
        for feature in 0..FEATURES {
            let (mut sum, mut abs_sum) = (0f64, 0f64);
            for index in 0..REDUCTION {
                let stored = activation[token * REDUCTION + index];
                let seen = if round_activation { bf16::from_f32(stored).to_f32() } else { stored };
                let product = f64::from(weight[feature * REDUCTION + index]) * f64::from(seen);
                sum += product;
                abs_sum += product.abs();
            }
            sums.push(sum);
            abs_sums.push(abs_sum);
        }
    }
    (sums, abs_sums)
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

#[test]
fn router_bf16_weight_runs_the_mma_and_matches_the_unrounded_f32_fold() {
    let (program, output) = router_program();
    let weight_f32 = values(11, FEATURES * REDUCTION);
    let weight_bytes: Vec<u8> = weight_f32.iter().flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes()).collect();
    let weight_stored: Vec<f32> = weight_f32.iter().map(|value| bf16::from_f32(*value).to_f32()).collect();
    let activation = values(29, TOKENS * REDUCTION);
    let bindings = [
        ("weight", QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: &weight_bytes }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let serving_policy = NumericPolicy::llama_relaxed().with_epilogue_sources(true);
    let mut plan = plan_named(Engine::Gpu, Some(GpuDriver::Metal), &program, &[], &bindings, &[output], serving_policy)
        .expect("metal plans the router matmul");
    let evaluated = execute_plan_named(&mut plan, &bindings).expect("metal executes the router matmul");
    let (actual, shape) = evaluated.get(output).expect("root is present");
    assert_eq!(shape, &[TOKENS as u64, FEATURES as u64]);

    let (expected, abs_sums) = fold(&weight_stored, &activation, false);
    let accumulation_u = 2f64.powi(-24) * REDUCTION as f64;
    let gamma = accumulation_u / (1.0 - accumulation_u);
    let bound: Vec<f64> = abs_sums.iter().map(|abs_sum| abs_sum * gamma).collect();
    let ratio = worst_ratio(actual, &expected, &bound);
    println!("router worst_error_over_bound={ratio}");
    assert!(ratio <= 1.0, "router MMA exceeded the f32 accumulation bound: {ratio}");

    let (rounded, _) = fold(&weight_stored, &activation, true);
    assert!(
        worst_ratio(actual, &rounded, &bound) > 1.0,
        "a bf16-rounded-activation fold must fall outside the bound, or the output is not the f32-tile result"
    );
    let mut corrupted = actual.to_vec();
    corrupted[7] += (bound[7] * 4.0) as f32;
    assert!(worst_ratio(&corrupted, &expected, &bound) > 1.0, "a one-coordinate corruption must exceed the bound");
}
