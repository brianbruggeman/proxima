use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_primitives::float_layout::{FloatLayout, SignEncoding};

use super::layout_render::layout_load_msl;
use super::scaled_float_render::{
    E3M2_BLOCK, Q8_0_BLOCK, ScaledBlock, q8_0_unpack_msl, scaled_block_element_msl,
};

const SAMPLE_LEN: u32 = 1 << 20;
const SAMPLE_STRIDE: u32 = 4099;

fn mismatches(expected: &[f32], observed: &[f32]) -> usize {
    assert_eq!(
        expected.len(),
        observed.len(),
        "compared sequences differ in length"
    );
    expected
        .iter()
        .zip(observed)
        .filter(|(want, got)| want.to_bits() != got.to_bits())
        .count()
}

fn decoded(layout: FloatLayout, codes: &[u32]) -> Vec<f32> {
    codes.iter().map(|&code| decode(code, layout)).collect()
}

fn all_codes(width: u32) -> Vec<u32> {
    (0..1u32 << width).collect()
}

fn sampled_codes() -> Vec<u32> {
    (0..SAMPLE_LEN)
        .map(|index| index.wrapping_mul(SAMPLE_STRIDE))
        .collect()
}

#[proxima::test]
#[case::int8(FloatLayout::INT8, all_codes(8), |code| f32::from(code as u8 as i8))]
#[case::uint8(FloatLayout::UINT8, all_codes(8), |code| f32::from(code as u8))]
#[case::int16(FloatLayout::INT16, all_codes(16), |code| f32::from(code as u16 as i16))]
#[case::uint16(FloatLayout::UINT16, all_codes(16), |code| f32::from(code as u16))]
#[case::int32(FloatLayout::INT32, sampled_codes(), |code| code as i32 as f32)]
#[case::uint32(FloatLayout::UINT32, sampled_codes(), |code| code as f32)]
async fn float_layout_int_decode_equals_native_cast(
    #[case] layout: FloatLayout,
    #[case] codes: Vec<u32>,
    #[case] native: fn(u32) -> f32,
) {
    let expected: Vec<f32> = codes.iter().map(|&code| native(code)).collect();

    assert_eq!(mismatches(&expected, &decoded(layout, &codes)), 0);
}

#[test]
fn float_layout_int_twos_complement_has_asymmetric_range_and_no_negative_zero() {
    let layout = FloatLayout::INT8;
    let values = decoded(layout, &all_codes(8));

    let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    assert_eq!((lowest, highest), (-128.0, 127.0));
    assert!(
        values
            .iter()
            .all(|value| value.to_bits() != (-0.0f32).to_bits())
    );
    assert_eq!(encode(-0.0, layout), 0);
    assert_eq!(encode(-128.0, layout), 0x80);
    assert_eq!(encode(1000.0, layout), 0x7f);
    assert_eq!(encode(-1000.0, layout), 0x80);
    assert_eq!((encode(2.5, layout), encode(3.5, layout)), (2, 4));
}

#[test]
fn float_layout_int_sign_magnitude_has_symmetric_range_and_negative_zero() {
    let layout = FloatLayout::SIGN_MAGNITUDE_INT8;
    let values = decoded(layout, &all_codes(8));

    let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    assert_eq!((lowest, highest), (-127.0, 127.0));
    assert_eq!(decode(0x80, layout).to_bits(), (-0.0f32).to_bits());
    assert_eq!(decode(0x00, layout).to_bits(), 0.0f32.to_bits());
    assert_eq!(encode(-0.0, layout), 0x80);
    assert_eq!(encode(-1000.0, layout), 0xff);
}

#[test]
fn float_layout_int_q8_0_block_uses_the_same_scaled_block_emitter_as_fp6() {
    let skeleton = |block: &ScaledBlock, layout: FloatLayout| {
        let text = scaled_block_element_msl(block, layout).replace(block.prefix, "block");
        blank_numeric_literals(&text)
    };

    let q8_0 = skeleton(&Q8_0_BLOCK, FloatLayout::INT8);
    let e3m2 = skeleton(&E3M2_BLOCK, FloatLayout::E3M2);
    let unpack = q8_0_unpack_msl();

    assert_eq!(q8_0, e3m2, "one element emitter renders both blocks");
    assert!(unpack.contains("float q8_0_element(device const uchar *block, uint index)"));
    assert!(
        unpack.contains("float q8_0_super_element(device const uchar *superblock, uint index)")
    );
    assert!(unpack.contains("float q8_0_field(uint code)"));
}

#[test]
fn float_layout_int_control_native_cast_output_through_the_comparator_passes() {
    let codes = all_codes(8);
    let native: Vec<f32> = codes
        .iter()
        .map(|&code| f32::from(code as u8 as i8))
        .collect();

    assert_eq!(mismatches(&native, &native.clone()), 0);
    assert_eq!(mismatches(&native, &decoded(FloatLayout::INT8, &codes)), 0);
}

#[test]
fn float_layout_int_control_flipped_sign_encoding_differs() {
    let codes = all_codes(8);
    let native: Vec<f32> = codes
        .iter()
        .map(|&code| f32::from(code as u8 as i8))
        .collect();
    let flipped = FloatLayout {
        sign: SignEncoding::Magnitude,
        ..FloatLayout::INT8
    };

    assert!(mismatches(&native, &decoded(flipped, &codes)) > 0);
}

#[test]
fn float_layout_int_control_wrong_scale_layout_differs() {
    let wide_scale = ScaledBlock {
        block_bytes: Q8_0_BLOCK.block_bytes + 2,
        ..Q8_0_BLOCK
    };

    let right = scaled_block_element_msl(&Q8_0_BLOCK, FloatLayout::INT8);
    let wrong = scaled_block_element_msl(&wide_scale, FloatLayout::INT8);

    assert_ne!(right, wrong);
    assert_ne!(
        layout_load_msl(FloatLayout::INT8, "probe"),
        layout_load_msl(FloatLayout::UINT8, "probe")
    );
}

fn blank_numeric_literals(source: &str) -> String {
    let mut blanked = String::new();
    let mut previous = ' ';
    let mut chars = source.chars().peekable();
    while let Some(current) = chars.next() {
        let starts_literal =
            current.is_ascii_digit() && !(previous.is_ascii_alphanumeric() || previous == '_');
        if starts_literal {
            while chars
                .next_if(|next| next.is_ascii_alphanumeric() || *next == '.')
                .is_some()
            {}
            blanked.push('#');
            previous = '#';
        } else {
            blanked.push(current);
            previous = current;
        }
    }
    blanked
}
