//! A two-level block encode: ggml's `quantize_row_q4_K_ref` (ggml-quants.c:1467-1536, b12110) as the
//! arithmetic of an [`EncodeRule`] whose `sub` is set.
//!
//! Each sub-block is fitted by [`fit_range`] (the rule's [`crate::encode_rule::RangeFit`]); the
//! sub-block scales and mins are quantized against their maxima, the block scale and min are stored
//! as the binary16 of `max / levels`, and every element code is then refitted against the scales the
//! decoder will see. The sub-block codes are written through the planes of the description, the
//! inverse of the read in [`crate::decode::decode_blocks`]; no format is named here.
//!
//! Teaching pointer: [`BlockLayout::encode_rule`] derives the rule; [`crate::encode::encode_blocks`]
//! runs it for a layout the compile-time block table does not hold.
//!
//! Allocation: none; scratch is on the stack.

use crate::block_layout::{BlockLayout, Coded, SubBlocks};
use crate::block_codec::{MAX_SUB_COUNT, MAX_SUB_ELEMENTS};
use crate::encode::{cursors_of, format_of, put_bits_at, put_code};
use crate::encode_rule::{EncodeRule, SubRule};
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec::{decode, encode};
use crate::range_fit::{fit_range, nearest_even};

fn unsupported(layout: &BlockLayout, reason: &'static str) -> NumericError {
    NumericError::UnsupportedEncode { format: format_of(layout), reason }
}

/// Stores `code` as sub-block `index`'s value through the code's own low and high planes.
fn put_sub_code(coded: &Coded, index: usize, code: u32, block: &mut [u8]) {
    put_bits_at(coded.low.bits, coded.low.address(index), code, block);
    if let Some(high) = &coded.high {
        put_bits_at(high.bits, high.address(index), code >> coded.low.bits, block);
    }
}

/// Stores a float field of the block (the scale or the min) and returns what the decoder reads back.
fn put_float(layout: FloatLayout, offset: usize, value: f32, block: &mut [u8]) -> f32 {
    let code = encode(value, layout);
    let bytes = usize::from(layout.width() / 8);
    block[offset..offset + bytes].copy_from_slice(&code.to_le_bytes()[..bytes]);
    decode(code, layout)
}

/// Encodes one block under `rule`; `input` holds the block's elements.
///
/// # Errors
/// [`NumericError::NonFinite`] for a NaN or infinite element, [`NumericError::UnsupportedEncode`]
/// when the rule has no second level or the sub-block is not one the stack scratch holds.
pub(crate) fn encode_two_level(
    layout: &BlockLayout,
    rule: &EncodeRule,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    let (Some(sub_rule), Some(sub)) = (rule.sub, &layout.sub) else {
        return Err(unsupported(layout, "a rule without a second level"));
    };
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite { format: format_of(layout) });
    }
    match input.len() / usize::from(sub_rule.count.max(1)) {
        16 => fit_block::<16>(layout, rule, &sub_rule, sub, input, block),
        32 => fit_block::<32>(layout, rule, &sub_rule, sub, input, block),
        _ => Err(unsupported(layout, "a sub-block width the two-level fit does not hold")),
    }
}

fn fit_block<const SUB: usize>(
    layout: &BlockLayout,
    rule: &EncodeRule,
    sub_rule: &SubRule,
    sub: &SubBlocks,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    const { assert!(SUB <= MAX_SUB_ELEMENTS) };
    let (Some(min_field), Some(sub_min)) = (&layout.min, &sub.min) else {
        return Err(unsupported(layout, "a two-level layout without a min"));
    };
    let count = usize::from(sub_rule.count);
    let (inputs, _) = input.as_chunks::<SUB>();
    let mut codes = [0u8; MAX_SUB_ELEMENTS * MAX_SUB_COUNT];
    let (chunks, _) = codes.as_chunks_mut::<SUB>();
    let mut scales = [0.0f32; MAX_SUB_COUNT];
    let mut mins = [0.0f32; MAX_SUB_COUNT];
    for (index, (values, chunk)) in inputs.iter().zip(chunks.iter_mut()).enumerate().take(count) {
        let (scale, offset) = fit_range(&sub_rule.fit, rule.highest, values, chunk);
        scales[index] = scale;
        mins[index] = -offset;
    }
    let levels = sub_rule.levels as f32;
    let top = sub_rule.levels as i32;
    let (max_scale, max_min) = (peak_of(&scales[..count]), peak_of(&mins[..count]));
    let inverse_scale = if max_scale > 0.0 { levels / max_scale } else { 0.0 };
    let inverse_min = if max_min > 0.0 { levels / max_min } else { 0.0 };
    block.fill(0);
    let d = put_float(layout.scale.layout, usize::from(layout.scale.byte_offset), max_scale / levels, block);
    let dmin = put_float(min_field.layout, usize::from(min_field.byte_offset), max_min / levels, block);
    for index in 0..count {
        let scale_code = nearest_even(inverse_scale * scales[index]).min(top);
        let min_code = nearest_even(inverse_min * mins[index]).min(top);
        put_sub_code(&sub.scale, index, scale_code as u32, block);
        put_sub_code(sub_min, index, min_code as u32, block);
        let step = d * scale_code as f32;
        if step == 0.0 {
            continue;
        }
        let shift = dmin * min_code as f32;
        for (value, code) in inputs[index].iter().zip(chunks[index].iter_mut()) {
            *code = nearest_even((value + shift) / step).clamp(rule.lowest, rule.highest) as u8;
        }
    }
    let mut cursors = cursors_of(layout);
    for &code in codes.iter().take(input.len()) {
        put_code(layout, &mut cursors, u32::from(code), block);
    }
    Ok(())
}

/// The largest of non-negative values, taken with a strict comparison from zero as the C does.
fn peak_of(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |peak, &value| if value > peak { value } else { peak })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::decode_blocks;
    use crate::encode::encode_blocks;
    use crate::format::NumericFormat;

    const Q4_K_ORACLE: &str = "9c20cc29f4f4fff5bef4ffbae3cf6493a0c2f5072a5c7fa0c3f5083a5d8fa0d3f5083a5d8fa1d3f6183a6d8fb1d3f618e406285a7caec2e406385a7caec2f406385a8caec2f416385a8ca0d2f416386a274a6c8fa0c2e5274a6c8fa0c205284a6d8fa0c305284a6d8fa0c305284a6d8f4c7f90c2f5072a5c7fa0c3f5072a5c7fa0c3f5083a5d8fa0d3f5083a5d8fa1d3";

    const Q2_K_ORACLE: &str = "ffefcfcfafbeefdfcfbfbeefdfcfbfafc8181d6162b687c8195d62a2b787d8192d6176b68bd81d6d62b6878bd81d6d626176868bd81d6d6276869bd81d6162b6868bd81d616276869bd81d6166b687db00310032";
    const Q5_K_ORACLE: &str = "211cb629fafafffafff4febc09ef8ad94a5296b4a52d694a5296b4a52d494a5296b5ad6d4b5ad296b5ad694b5ad296b53085da0f54a9fe4095ea0f54a9fe4196eb1065ba0e4196eb1065ba0f52a7fb20d80c51a5fa4e94e80d51a6fa4f94e81d62b60b5fa4f91d62b60b50a5f92e72c74e84d91e60b5fa4f94d92e60b50a4f94e92e71b60b5095ea3f71c61b50a5ea3fa8fd4095ea0f54a9fe4095ea0f54a9fe4196eb1065ba0e51a6fb1065ba0f52a7";

    fn oracle_input() -> std::vec::Vec<f32> {
        (0..256).map(|index| (((index * 37) % 256) as f32 - 90.0) / 32.0).collect()
    }

    fn hex_of(bytes: &[u8]) -> std::string::String {
        bytes.iter().map(|byte| std::format!("{byte:02x}")).collect()
    }

    #[test]
    fn q4_k_matches_llama_cpp_b12110_quantize_row_q4_k_ref_on_the_permutation_block() {
        let input = oracle_input();
        let mut bytes = std::vec![0u8; 144];

        encode_blocks(&NumericFormat::Q4K, &input, &mut bytes).expect("q4_k encodes from its description");

        assert_eq!(hex_of(&bytes), Q4_K_ORACLE);
    }

    #[test]
    fn q4_k_decodes_back_within_the_oracles_max_abs_error() {
        let input = oracle_input();
        let mut bytes = std::vec![0u8; 144];
        encode_blocks(&NumericFormat::Q4K, &input, &mut bytes).expect("q4_k encodes");
        let mut back = std::vec![0.0f32; 256];

        decode_blocks(&NumericFormat::Q4K, &bytes, &mut back).expect("q4_k decodes");

        let worst = input.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert_eq!(worst, 0.232_788_09);
    }

    fn encoded_and_worst(format: &NumericFormat, bytes: usize) -> (std::string::String, f32) {
        let input = oracle_input();
        let mut packed = std::vec![0u8; bytes];
        encode_blocks(format, &input, &mut packed).expect("the format encodes from its description");
        let mut back = std::vec![0.0f32; 256];
        decode_blocks(format, &packed, &mut back).expect("the format decodes");
        let worst = input.iter().zip(&back).map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs()).fold(0.0f64, f64::max);
        (hex_of(&packed), worst as f32)
    }

    #[test]
    fn q5_k_matches_llama_cpp_b12110_quantize_row_q5_k_ref_on_the_permutation_block() {
        let (hex, worst) = encoded_and_worst(&NumericFormat::Q5K, 176);

        assert_eq!(hex, Q5_K_ORACLE);
        assert_eq!(worst, 0.125_137_33);
    }

    #[test]
    fn q2_k_matches_llama_cpp_b12110_quantize_row_q2_k_ref_on_the_permutation_block() {
        let (hex, worst) = encoded_and_worst(&NumericFormat::Q2K, 84);

        assert_eq!(hex, Q2_K_ORACLE);
        assert_eq!(worst, 1.15625);
    }

    #[test]
    fn a_non_finite_element_is_a_typed_error() {
        let mut input = oracle_input();
        input[3] = f32::NAN;

        let error = encode_blocks(&NumericFormat::Q4K, &input, &mut std::vec![0u8; 144]).expect_err("nan has no code");

        assert!(matches!(error, NumericError::NonFinite { .. }), "{error:?}");
    }
}
