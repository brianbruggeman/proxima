//! A half-output elementwise over an f32 operand on a real Metal device: the operand is read as
//! `float` (the dtype of the node that produced it) and the store rounds to binary16, bit for bit
//! with `half::f16::from_f32`. Needs a Metal device; a missing one fails loudly through `expect`.

#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_numeric::{DType, FormatCodec, NumericFormat, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, projection};

/// Mixed magnitudes, both signs, exact zeros and values below binary16's normal range, none of
/// them exactly representable as binary16 apart from the zeros.
fn real_valued_input() -> Vec<f32> {
    let magnitudes = [0.0f32, 3.1e-6, 0.0173, 0.4999, 1.0 / 3.0, 1.7, 12.37, 255.9, 1031.3];
    (0..27)
        .map(|index| {
            let magnitude = magnitudes[index % magnitudes.len()] * (1.0 + index as f32 * 0.0137);
            if index % 3 == 1 { -magnitude } else { magnitude }
        })
        .collect()
}

#[test]
fn identity_from_f32_to_half_rounds_to_binary16_bit_for_bit() {
    let input = real_valued_input();
    let extent = input.len() as u32;
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(extent)], name: None },
    );
    let narrowed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT16,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let blocks = [QuantizedBlock::Float32(&input)];
    let plan = omega::plan(&program, &[], &blocks, &[narrowed], NumericPolicy::default())
        .expect("the mixed-width identity plans");
    let bytes = input.len() * 2;
    let output = omega::allocate_placed_buffer(bytes).expect("allocates the half output");
    omega::zero_placed_buffer(&output, bytes).expect("zeroes the half output");

    omega::execute_plan_with_placements(&plan, &blocks, &[], &[(narrowed, &output, 0)], &mut Vec::new())
        .expect("the narrowing identity runs");

    let codec = FormatCodec::for_format(&NumericFormat::Float16).expect("binary16 resolves a codec");
    let written = omega::read_placed_buffer(&output, 0, &codec, input.len()).expect("the half rows decode");
    let expected: Vec<u32> = input.iter().map(|value| half::f16::from_f32(*value).to_f32().to_bits()).collect();
    let actual: Vec<u32> = written.iter().map(|value| value.to_bits()).collect();
    assert_eq!(actual, expected, "the stored values are binary16(f32) bit for bit");
    assert_ne!(
        expected,
        input.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        "the inputs are not binary16 exact, so equality proves the rounding ran"
    );
}
