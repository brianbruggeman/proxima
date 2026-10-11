use half::{bf16, f16};
use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_primitives::float_layout::{FloatLayout, Specials};

const SAMPLE_LEN: u32 = 1 << 20;
const SAMPLE_STRIDE: u32 = 4093;
const LAYOUT_F32: FloatLayout = FloatLayout::BF32;
const LAYOUT_F16: FloatLayout = FloatLayout {
    exponent_bits: 5,
    mantissa_bits: 10,
    specials: Specials::InfinityOnOverflow,
    bias_override: None,
};
const LAYOUT_E4M3: FloatLayout = FloatLayout {
    exponent_bits: 4,
    mantissa_bits: 3,
    specials: Specials::None,
    bias_override: None,
};

fn power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn reference_bias(layout: FloatLayout) -> i32 {
    layout
        .bias_override
        .map_or((1 << (layout.exponent_bits - 1)) - 1, i32::from)
}

fn exponent_all_ones(layout: FloatLayout) -> u64 {
    (1u64 << layout.exponent_bits) - 1
}

fn mantissa_ones(layout: FloatLayout) -> u64 {
    (1u64 << layout.mantissa_bits) - 1
}

fn sign_shift(layout: FloatLayout) -> u32 {
    u32::from(layout.exponent_bits) + u32::from(layout.mantissa_bits)
}

fn grid_value(code: u64, layout: FloatLayout) -> f64 {
    let mantissa_bits = i32::from(layout.mantissa_bits);
    let exponent = ((code >> mantissa_bits) & exponent_all_ones(layout)) as i32;
    let mantissa = (code & mantissa_ones(layout)) as f64;
    let bias = reference_bias(layout);
    match exponent {
        0 => mantissa * power_of_two(1 - bias - mantissa_bits),
        _ => {
            (power_of_two(mantissa_bits) + mantissa) * power_of_two(exponent - bias - mantissa_bits)
        }
    }
}

fn reference_decode(code: u64, layout: FloatLayout) -> f64 {
    let negative = (code >> sign_shift(layout)) & 1 == 1;
    let magnitude_code = code & ((1u64 << sign_shift(layout)) - 1);
    let reserved = layout.specials != Specials::None
        && (magnitude_code >> layout.mantissa_bits) == exponent_all_ones(layout);
    let magnitude = match (reserved, magnitude_code & mantissa_ones(layout)) {
        (true, 0) => f64::INFINITY,
        (true, _) => f64::NAN,
        (false, _) => grid_value(magnitude_code, layout),
    };
    if negative { -magnitude } else { magnitude }
}

fn top_code(layout: FloatLayout) -> u64 {
    let ones = exponent_all_ones(layout);
    match layout.specials {
        Specials::None => (ones << layout.mantissa_bits) | mantissa_ones(layout),
        Specials::SaturateOnOverflow => {
            ((ones - 1) << layout.mantissa_bits) | mantissa_ones(layout)
        }
        Specials::InfinityOnOverflow => ones << layout.mantissa_bits,
    }
}

fn largest_code_at_or_below(magnitude: f64, layout: FloatLayout) -> u64 {
    let (mut low, mut high) = (0u64, top_code(layout));
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if grid_value(middle, layout) <= magnitude {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn nearest_code_ties_to_even(magnitude: f64, layout: FloatLayout) -> u64 {
    let lower = largest_code_at_or_below(magnitude, layout);
    if lower == top_code(layout) {
        return lower;
    }
    let below = magnitude - grid_value(lower, layout);
    let above = grid_value(lower + 1, layout) - magnitude;
    match below.partial_cmp(&above) {
        Some(core::cmp::Ordering::Less) => lower,
        Some(core::cmp::Ordering::Greater) => lower + 1,
        _ if lower.is_multiple_of(2) => lower,
        _ => lower + 1,
    }
}

fn reference_encode(value: f32, layout: FloatLayout) -> u64 {
    let sign = u64::from(value.is_sign_negative()) << sign_shift(layout);
    sign | nearest_code_ties_to_even(f64::from(value).abs(), layout)
}

fn decode_sample(layout: FloatLayout) -> Vec<u64> {
    match layout.width() {
        width @ 1..=16 => (0..1u64 << width).collect(),
        _ => (0..SAMPLE_LEN)
            .map(|index| u64::from(index.wrapping_mul(SAMPLE_STRIDE | 0x1_0000)))
            .collect(),
    }
}

fn encode_sample(layout: FloatLayout) -> Vec<f32> {
    let fixed = [
        0.0,
        -0.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
    ];
    fixed
        .into_iter()
        .chain(sweep_patterns())
        .chain(tie_values(layout))
        .collect()
}

fn tie_values(layout: FloatLayout) -> Vec<f32> {
    if layout.width() > 16 {
        return Vec::new();
    }
    (0..top_code(layout))
        .flat_map(|code| {
            let midpoint = ((grid_value(code, layout) + grid_value(code + 1, layout)) / 2.0) as f32;
            let bits = midpoint.to_bits();
            [bits - 1, bits, bits + 1].map(f32::from_bits)
        })
        .collect()
}

fn encode_mismatches(layout: FloatLayout, produce: impl Fn(f32) -> u64) -> usize {
    encode_sample(layout)
        .into_iter()
        .filter(|value| !value.is_nan())
        .filter(|value| produce(*value) != reference_encode(*value, layout))
        .count()
}

fn decode_mismatches(layout: FloatLayout, produce: impl Fn(u64) -> f32) -> usize {
    decode_sample(layout)
        .into_iter()
        .filter(|&code| {
            let expected = reference_decode(code, layout);
            let actual = f64::from(produce(code));
            let same_nan = expected.is_nan() && actual.is_nan();
            !same_nan && expected.to_bits() != actual.to_bits()
        })
        .count()
}

fn check_decode_equals_reference(layout: FloatLayout) {
    let expected_count = if layout.width() <= 16 {
        1usize << layout.width()
    } else {
        SAMPLE_LEN as usize
    };
    assert_eq!(decode_sample(layout).len(), expected_count, "code count");
    let mismatches = decode_mismatches(layout, |code| decode(code as u32, layout));
    assert_eq!(mismatches, 0, "decode disagrees with the f64 reference");
}

fn check_encode_equals_reference(layout: FloatLayout) {
    let mismatches = encode_mismatches(layout, |value| u64::from(encode(value, layout)));
    assert_eq!(
        mismatches, 0,
        "encode disagrees with the f64 enumeration reference"
    );
}

fn check_round_trip(layout: FloatLayout) {
    let representable: Vec<u64> = decode_sample(layout)
        .into_iter()
        .filter(|&code| reference_decode(code, layout).is_finite())
        .collect();
    assert!(representable.len() > 1, "layout must hold finite codes");
    for code in representable {
        let value = decode(code as u32, layout);
        assert_eq!(u64::from(encode(value, layout)), code, "code {code:#x}");
    }
}

macro_rules! per_layout {
    ($check:ident; $($name:ident => $layout:expr),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                $check($layout);
            }
        )+
    };
}

per_layout! { check_decode_equals_reference;
    float_layout_codec_decode_equals_f64_reference_over_every_code_f32 => LAYOUT_F32,
    float_layout_codec_decode_equals_f64_reference_over_every_code_f16 => LAYOUT_F16,
    float_layout_codec_decode_equals_f64_reference_over_every_code_bf16 => FloatLayout::BF16,
    float_layout_codec_decode_equals_f64_reference_over_every_code_bf8_e5m2 => FloatLayout::E5M2_SATURATING,
    float_layout_codec_decode_equals_f64_reference_over_every_code_bf4_e2m1 => FloatLayout::E2M1,
    float_layout_codec_decode_equals_f64_reference_over_every_code_fp6_e3m2 => FloatLayout::E3M2,
    float_layout_codec_decode_equals_f64_reference_over_every_code_fp6_e2m3 => FloatLayout::E2M3,
    float_layout_codec_decode_equals_f64_reference_over_every_code_e4m3 => LAYOUT_E4M3,
}

per_layout! { check_encode_equals_reference;
    float_layout_codec_encode_equals_f64_enumeration_reference_f32 => LAYOUT_F32,
    float_layout_codec_encode_equals_f64_enumeration_reference_f16 => LAYOUT_F16,
    float_layout_codec_encode_equals_f64_enumeration_reference_bf16 => FloatLayout::BF16,
    float_layout_codec_encode_equals_f64_enumeration_reference_bf8_e5m2 => FloatLayout::E5M2_SATURATING,
    float_layout_codec_encode_equals_f64_enumeration_reference_bf4_e2m1 => FloatLayout::E2M1,
    float_layout_codec_encode_equals_f64_enumeration_reference_fp6_e3m2 => FloatLayout::E3M2,
    float_layout_codec_encode_equals_f64_enumeration_reference_fp6_e2m3 => FloatLayout::E2M3,
    float_layout_codec_encode_equals_f64_enumeration_reference_e4m3 => LAYOUT_E4M3,
}

per_layout! { check_round_trip;
    float_layout_codec_encode_round_trips_every_representable_code_f32 => LAYOUT_F32,
    float_layout_codec_encode_round_trips_every_representable_code_f16 => LAYOUT_F16,
    float_layout_codec_encode_round_trips_every_representable_code_bf16 => FloatLayout::BF16,
    float_layout_codec_encode_round_trips_every_representable_code_bf8_e5m2 => FloatLayout::E5M2_SATURATING,
    float_layout_codec_encode_round_trips_every_representable_code_bf4_e2m1 => FloatLayout::E2M1,
    float_layout_codec_encode_round_trips_every_representable_code_fp6_e3m2 => FloatLayout::E3M2,
    float_layout_codec_encode_round_trips_every_representable_code_fp6_e2m3 => FloatLayout::E2M3,
    float_layout_codec_encode_round_trips_every_representable_code_e4m3 => LAYOUT_E4M3,
}

fn sweep_patterns() -> impl Iterator<Item = f32> {
    (0..SAMPLE_LEN).map(|index| f32::from_bits(index.wrapping_mul(SAMPLE_STRIDE)))
}

#[test]
fn float_layout_codec_bf16_equals_half_over_1048576_patterns() {
    assert_eq!(sweep_patterns().count(), 1_048_576);
    for value in sweep_patterns() {
        let code = encode(value, FloatLayout::BF16);
        let expected = u32::from(bf16::from_f32(value).to_bits());
        assert_eq!(code, expected, "encode {:#010x}", value.to_bits());
    }
    for code in 0..=u16::MAX {
        let expected = bf16::from_bits(code).to_f32().to_bits();
        let actual = decode(u32::from(code), FloatLayout::BF16).to_bits();
        assert_eq!(actual, expected, "decode {code:#06x}");
    }
}

#[test]
fn float_layout_codec_f16_equals_half_over_1048576_patterns() {
    assert_eq!(sweep_patterns().count(), 1_048_576);
    for value in sweep_patterns() {
        let code = encode(value, LAYOUT_F16);
        let expected = u32::from(f16::from_f32(value).to_bits());
        assert_eq!(code, expected, "encode {:#010x}", value.to_bits());
    }
    for code in 0..=u16::MAX {
        let expected = f16::from_bits(code).to_f32().to_bits();
        let actual = decode(u32::from(code), LAYOUT_F16).to_bits();
        assert_eq!(actual, expected, "decode {code:#06x}");
    }
}

#[test]
fn float_layout_codec_control_half_output_through_the_reference_comparator_passes() {
    let layout = FloatLayout::BF16;
    let compared = encode_sample(layout)
        .into_iter()
        .filter(|value| !value.is_nan())
        .count();
    let mismatches = encode_mismatches(layout, |value| u64::from(bf16::from_f32(value).to_bits()));
    assert!(
        compared > SAMPLE_LEN as usize,
        "comparator must see the whole sample"
    );
    assert_eq!(mismatches, 0);
}

#[test]
fn float_layout_codec_control_wrong_bias_differs() {
    let layout = FloatLayout::E3M2;
    let shifted = FloatLayout {
        bias_override: Some(layout.bias() + 1),
        ..layout
    };
    let decode_differences = decode_mismatches(layout, |code| decode(code as u32, shifted));
    let encode_differences = encode_mismatches(layout, |value| u64::from(encode(value, shifted)));
    assert!(decode_differences > 0, "wrong bias must change decode");
    assert!(encode_differences > 0, "wrong bias must change encode");
}

#[test]
fn float_layout_codec_control_truncating_encode_differs() {
    let truncated = |value: f32| u64::from(value.to_bits() >> 16);
    let mismatches = encode_mismatches(FloatLayout::BF16, truncated);
    assert!(
        mismatches > 0,
        "truncation must disagree with round to nearest even"
    );
}
