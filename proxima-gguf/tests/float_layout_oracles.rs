use proxima_gguf::quant::float_layout::{decode, encode, encode_counting};
use proxima_primitives::float_layout::FloatLayout;

fn positive_finite_codes(layout: FloatLayout) -> Vec<u32> {
    (0..1u32 << (layout.width() - 1))
        .take_while(|&code| decode(code, layout).is_finite())
        .collect()
}

fn assert_midpoints_tie_to_even_code(layout: FloatLayout) {
    let codes = positive_finite_codes(layout);
    assert!(codes.len() > 2, "layout must have several finite codes");
    for pair in codes.windows(2) {
        let (lower, upper) = (pair[0], pair[1]);
        let midpoint = (decode(lower, layout) + decode(upper, layout)) / 2.0;
        let even_code = if lower % 2 == 0 { lower } else { upper };
        assert_eq!(
            encode(midpoint, layout),
            even_code,
            "midpoint {midpoint} of codes {lower:#x} and {upper:#x}"
        );
    }
}

#[test]
fn float_layout_midpoint_ties_to_even_e5m2() {
    assert_midpoints_tie_to_even_code(FloatLayout::E5M2);
}

#[test]
fn float_layout_midpoint_ties_to_even_e2m1() {
    assert_midpoints_tie_to_even_code(FloatLayout::E2M1);
}

#[test]
fn float_layout_midpoint_ties_to_even_e3m2() {
    assert_midpoints_tie_to_even_code(FloatLayout::E3M2);
}

#[test]
fn float_layout_midpoint_ties_to_even_e2m3() {
    assert_midpoints_tie_to_even_code(FloatLayout::E2M3);
}

#[test]
fn float_layout_range_saturates_e3m2() {
    assert_eq!(encode(1.0e9, FloatLayout::E3M2), 0x1f);
    assert_eq!(encode(-1.0e9, FloatLayout::E3M2), 0x3f);
}

#[test]
fn float_layout_range_saturates_infinity_e5m2() {
    assert_eq!(encode(f32::INFINITY, FloatLayout::E5M2), 0x7b);
    assert_eq!(encode(f32::NEG_INFINITY, FloatLayout::E5M2), 0xfb);
}

#[test]
fn float_layout_range_overflows_to_infinity_bf16() {
    assert_eq!(encode(f32::INFINITY, FloatLayout::BF16), 0x7f80);
    assert_eq!(encode(f32::MAX, FloatLayout::BF16), 0x7f80);
}

#[test]
fn float_layout_range_underflows_to_signed_zero_e3m2() {
    assert_eq!(encode(0.03, FloatLayout::E3M2), 0x00);
    assert_eq!(encode(-0.03, FloatLayout::E3M2), 0x20);
}

#[test]
fn float_layout_nan_counted_and_clamped_e3m2() {
    let activations = [
        0.5,
        f32::NAN,
        -3.25,
        12.0,
        f32::NAN,
        0.0625,
        -0.0,
        f32::NAN,
        28.0,
        -7.5,
    ];
    let mut nan_count = 0u64;

    let codes = activations.map(|value| encode_counting(value, FloatLayout::E3M2, &mut nan_count));

    assert_eq!(nan_count, 3);
    for (value, code) in activations.iter().zip(codes) {
        if value.is_nan() {
            assert_eq!(code, 0x1f);
        }
    }
}

#[test]
fn float_layout_nan_quiet_code_not_counted_e5m2() {
    let mut nan_count = 0u64;

    let code = encode_counting(f32::NAN, FloatLayout::E5M2, &mut nan_count);

    assert_eq!(code, 0x7e);
    assert_eq!(nan_count, 0);
}
