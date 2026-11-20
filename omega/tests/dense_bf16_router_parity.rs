//! The flint router GEMM (`ffn_gate_inp`: 512 tokens x 2048 K x 256 experts, a bf16 weight laid out
//! `[experts, K]` and an f32 activation) runs the dense bf16 MMA under the default serving config.
//! Its staging (all threads decode `uint4` weight runs into the bf16 tile, `float4` activation loads
//! converted with `bfloat(float)`) must not change the arithmetic: the output matches an f64
//! reduction over the bf16-rounded operands within the f32-accumulation tolerance, and its bit hash
//! is printed so a run before and after a staging change can be compared bit for bit.
#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::bf16;
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, QuantizedBlock, ReduceInit, ScalarOp};

const TOKENS: usize = 512;
const REDUCTION: usize = 2048;
const EXPERTS: usize = 256;
const SEED: u64 = 581;
const RELATIVE_TOLERANCE: f64 = 1.0e-5;

fn bit_hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0000_0100_0000_01b3))
}

#[test]
fn bf16_router_gemm_matches_the_bf16_rounded_reduction_at_the_flint_chunk_shape() {
    let mut lcg = Lcg(SEED);
    let weights: Vec<bf16> = (0..EXPERTS * REDUCTION).map(|_| bf16::from_f32(lcg.next_unit() * 0.2 - 0.1)).collect();
    let activations: Vec<f32> = (0..TOKENS * REDUCTION).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|weight| weight.to_bits().to_le_bytes()).collect();

    let mut program = Vec::new();
    let activation = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(TOKENS as u32), Extent::Static(REDUCTION as u32)], "activation");
    let weight = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(EXPERTS as u32), Extent::Static(REDUCTION as u32)], "weight");
    let product = elementwise(&mut program, DType::FLOAT32, ScalarOp::Multiply, &[(activation, "si->sio"), (weight, "oi->sio")]).expect("router product axes");
    let output = reduce(&mut program, DType::FLOAT32, ScalarOp::Add, ReduceInit::Zero, product, "sio->sio", "so->sio").expect("router reduction axes");

    let bindings = [
        ("activation", QuantizedBlock::Float32(&activations)),
        ("weight", QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: &weight_bytes }),
    ];
    let config = RuntimeConfig { engine: "gpu".into(), gpu_driver: "metal".into(), allow_fusion: true, numeric: NumericPolicy::llama_relaxed(), ..RuntimeConfig::default() };
    let (mut plan, records) = plan_named_with_config(&config, &program, &[], &bindings, &[output]).expect("metal plans the router GEMM");
    assert_eq!(records.len(), 1, "one candidate");
    assert!(records[0].kernel_identities.iter().any(|identity| identity.contains("_dbg")), "the router runs the dense tiled kernel: {:?}", records[0].kernel_identities);
    let evaluated = execute_plan_named(&mut plan, &bindings).expect("metal runs the router GEMM");
    let (actual, shape) = evaluated.get(output).expect("the router output is present");
    assert_eq!(shape, &[TOKENS as u64, EXPERTS as u64]);
    assert_eq!(actual.len(), TOKENS * EXPERTS, "every logit is compared");

    let mut worst = 0.0f64;
    let mut out_of_tolerance = 0usize;
    for token in 0..TOKENS {
        for expert in 0..EXPERTS {
            let (sum, magnitude) = (0..REDUCTION).fold((0.0f64, 0.0f64), |(sum, magnitude), step| {
                let term = f64::from(weights[expert * REDUCTION + step].to_f32()) * f64::from(bf16::from_f32(activations[token * REDUCTION + step]).to_f32());
                (sum + term, magnitude + term.abs())
            });
            let diff = (f64::from(actual[token * EXPERTS + expert]) - sum).abs();
            worst = worst.max(diff / (magnitude + 1.0e-9));
            if diff > RELATIVE_TOLERANCE * magnitude + 1.0e-6 {
                out_of_tolerance += 1;
            }
        }
    }
    println!("router_parity max_relative_diff={worst:e} out_of_tolerance={out_of_tolerance} bit_hash={:016x}", bit_hash(actual));
    assert_eq!(out_of_tolerance, 0, "worst relative diff {worst:e}");
}
