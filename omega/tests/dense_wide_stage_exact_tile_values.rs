//! The dense MMA weight stage reads a K run as `uint4` vectors and decodes it through the
//! renderer derived from the operand's float layout. A one-hot activation turns every output
//! coordinate into a single product `weight[feature][token] * 1.0` summed with exact zeros, so the
//! output IS the staged tile value: each must equal the host decode of the stored code, bit for
//! bit. The same check runs for an f16 and an FP8 weight: the stage is derived from the layout. `REDUCTION` 128 keeps every K step on the vector path; 131 leaves a ragged last step that
//! takes the guarded per-element path, and both feed the same decode.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use omega::msl::layout_render::one_element_float_layout;
use proxima_numeric::codec::Codec;
use proxima_numeric::float_layout_codec::decode;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, NodeId, Op, QuantizedBlock, ReduceInit, ScalarOp};

fn weight_first_program(tokens: u32, features: u32, reduction: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(features), Extent::Static(reduction)],
        "weight",
    );
    let activation = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(tokens), Extent::Static(reduction)],
        "activation",
    );
    let product = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(weight, "fi->tfi"), (activation, "ti->tfi")],
    )
    .expect("weight-first product has compatible axes");
    let output = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tfi->tfi",
        "tf->tfi",
    )
    .expect("reduction keeps token and feature axes");
    (program, output)
}

/// Every code pattern of the codec's width, in LCG order, with the non-finite ones replaced by
/// zero (an infinite weight times the one-hot's zeros would be NaN and hide the tile value it is
/// meant to check), and each code's host decode through the layout.
fn stored(codec: Codec, count: usize) -> (Vec<u8>, Vec<f32>) {
    let layout = one_element_float_layout(&codec).expect("the codec is a one-element whole-byte float");
    let width = usize::from(layout.width() / 8);
    let mut lcg = Lcg(0x5EED);
    let mut bytes = Vec::with_capacity(count * width);
    let mut decoded = Vec::with_capacity(count);
    for _ in 0..count {
        let raw = (lcg.next_unit().to_bits() >> 3) & ((1u32 << (width * 8)) - 1);
        let value = decode(raw, layout);
        let code = if value.is_finite() { raw } else { 0 };
        bytes.extend_from_slice(&code.to_le_bytes()[..width]);
        decoded.push(decode(code, layout));
    }
    (bytes, decoded)
}

fn assert_tile_values_are_the_decoded_codes(codec: Codec, reduction: u32) {
    let tokens = reduction;
    let token_tiles = u64::from(tokens).div_ceil(omega::sized::TILED_GEMM_BLOCK_N);
    let feature_tiles = omega::sized::DENSE_TILED_MIN_THREADGROUPS.div_ceil(token_tiles);
    let features = ((feature_tiles - 1) * omega::sized::TILED_GEMM_BLOCK_M + 1) as u32;
    let (program, output) = weight_first_program(tokens, features, reduction);
    let (weight_bytes, decoded) = stored(codec, (features * reduction) as usize);
    let activation: Vec<f32> = (0..tokens)
        .flat_map(|token| (0..reduction).map(move |index| f32::from(u8::from(index == token))))
        .collect();
    let bindings = [
        ("weight", QuantizedBlock::Packed { codec, bytes: &weight_bytes }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let config = RuntimeConfig {
        engine: "gpu".into(),
        gpu_driver: "metal".into(),
        allow_fusion: true,
        numeric: NumericPolicy::llama_relaxed(),
        ..RuntimeConfig::default()
    };
    let (mut plan, records) = plan_named_with_config(&config, &program, &[], &bindings, &[output])
        .expect("metal prepares the one-hot matmul");
    assert!(
        records[0]
            .kernel_identities
            .iter()
            .any(|identity| identity.contains("_dbg") && identity.contains("_mma_f32")),
        "the dense f32-tile kernel must run: {:?}",
        records[0].kernel_identities
    );
    let evaluated = execute_plan_named(&mut plan, &bindings).expect("metal executes the matmul");
    let (actual, _) = evaluated.get(output).expect("root is present");
    let mut compared = 0usize;
    for token in 0..tokens as usize {
        for feature in 0..features as usize {
            let expected = decoded[feature * reduction as usize + token];
            let got = actual[token * features as usize + feature];
            assert!(
                got == expected,
                "{codec:?} K={reduction} token {token} feature {feature}: staged {got} != decoded {expected}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, (tokens * features) as usize, "every coordinate was compared");
    assert!(compared > 0, "the comparison must see coordinates");
}

#[test]
fn f16_weight_tile_values_are_the_decoded_codes_on_the_vector_path() {
    assert_tile_values_are_the_decoded_codes(Codec::Float16, 128);
}

#[test]
fn f16_weight_tile_values_are_the_decoded_codes_across_a_ragged_last_step() {
    assert_tile_values_are_the_decoded_codes(Codec::Float16, 131);
}

#[test]
fn fp8_e4m3_weight_tile_values_are_the_decoded_codes_on_the_vector_path() {
    assert_tile_values_are_the_decoded_codes(Codec::Float8E4M3Fn, 128);
}

#[test]
fn fp8_e4m3_weight_tile_values_are_the_decoded_codes_across_a_ragged_last_step() {
    assert_tile_values_are_the_decoded_codes(Codec::Float8E4M3Fn, 131);
}
