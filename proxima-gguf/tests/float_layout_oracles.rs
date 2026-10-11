use half::bf16;
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

const CORPUS_LEN: usize = 1_048_588;
const BF8_VECTORS_CSV: &str =
    include_str!("../../proxima-tensor/specs/granite-attention-numeric-matrix/bf8_vectors.csv");

fn corpus() -> Vec<f32> {
    let fixed = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        f32::NAN,
    ];
    let sweep = (0..1_048_576u32).map(|index| f32::from_bits(index.wrapping_mul(4093)));
    let values: Vec<f32> = fixed.into_iter().chain(sweep).collect();
    assert_eq!(values.len(), CORPUS_LEN, "corpus length");
    values
}

fn assert_same_float(actual: f32, expected: f32, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected nan, got {actual}");
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
    }
}

#[test]
fn float_layout_bf32_encode_is_f32_to_bits() {
    for value in corpus() {
        let code = encode(value, FloatLayout::BF32);
        if value.is_nan() {
            assert!(decode(code, FloatLayout::BF32).is_nan(), "nan encode");
        } else {
            assert_eq!(code, value.to_bits(), "value bits {:#x}", value.to_bits());
        }
    }
}

#[test]
fn float_layout_bf32_decode_is_f32_from_bits() {
    for value in corpus() {
        let bits = value.to_bits();
        let context = format!("code {bits:#x}");
        assert_same_float(
            decode(bits, FloatLayout::BF32),
            f32::from_bits(bits),
            &context,
        );
    }
}

#[test]
fn float_layout_bf32_control_bf16_layout_differs_from_to_bits() {
    let disagreements = corpus()
        .into_iter()
        .filter(|value| !value.is_nan())
        .filter(|value| encode(*value, FloatLayout::BF16) != value.to_bits())
        .count();
    assert!(disagreements > 0, "bf16 layout must disagree with to_bits");
}

#[test]
fn float_layout_bf16_encode_matches_half_from_f32() {
    for value in corpus() {
        let code = encode(value, FloatLayout::BF16);
        let oracle = bf16::from_f32(value);
        if value.is_nan() {
            assert!(decode(code, FloatLayout::BF16).is_nan(), "nan encode");
        } else {
            let context = format!("value bits {:#x}", value.to_bits());
            assert_eq!(code, u32::from(oracle.to_bits()), "{context}");
        }
    }
}

#[test]
fn float_layout_bf16_decode_matches_half_to_f32() {
    let all_codes = 0..=u16::MAX;
    assert_eq!(all_codes.clone().count(), 65_536, "bf16 code count");
    for code in all_codes {
        let oracle = bf16::from_bits(code).to_f32();
        let context = format!("code {code:#x}");
        assert_same_float(decode(u32::from(code), FloatLayout::BF16), oracle, &context);
    }
}

#[test]
fn float_layout_bf16_encode_ties_to_even_matches_half() {
    let ties = [0x3f80_8000u32, 0x3f81_8000, 0x4000_8000, 0x4001_8000];
    for bits in ties {
        let value = f32::from_bits(bits);
        let oracle = u32::from(bf16::from_f32(value).to_bits());
        assert_eq!(encode(value, FloatLayout::BF16), oracle, "tie {bits:#x}");
    }
}

#[test]
fn float_layout_bf16_control_truncation_differs_from_half() {
    let disagreements = corpus()
        .into_iter()
        .filter(|value| !value.is_nan())
        .filter(|value| value.to_bits() >> 16 != u32::from(bf16::from_f32(*value).to_bits()))
        .count();
    assert!(disagreements > 0, "truncation must disagree with half");
}

struct Bf8Row {
    case: &'static str,
    input: f32,
    code: u32,
}

fn parse_hex(text: &str) -> Option<u32> {
    u32::from_str_radix(text.trim_start_matches("0x"), 16).ok()
}

fn parse_row(line: &'static str) -> Option<Bf8Row> {
    let mut fields = line.split(',');
    Some(Bf8Row {
        case: fields.next()?,
        input: f32::from_bits(parse_hex(fields.next()?)?),
        code: parse_hex(fields.next()?)?,
    })
}

fn bf8_rows() -> Vec<Bf8Row> {
    let mut lines = BF8_VECTORS_CSV.lines();
    assert_eq!(lines.next(), Some("case,f32_bits,bf8_bits"), "csv header");
    let rows: Vec<Bf8Row> = lines.filter_map(parse_row).collect();
    assert_eq!(rows.len(), 16, "csv row count");
    rows
}

fn row_encodes_to_expected_code(row: &Bf8Row, expected_code: u32) -> bool {
    let actual = encode(row.input, FloatLayout::E5M2);
    if row.input.is_nan() {
        decode(actual, FloatLayout::E5M2).is_nan()
            && decode(expected_code, FloatLayout::E5M2).is_nan()
    } else {
        actual == expected_code
    }
}

#[test]
fn float_layout_e5m2_control_altered_row_fails() {
    let rows = bf8_rows();
    let ones: Vec<&Bf8Row> = rows.iter().filter(|row| row.case == "one").collect();
    assert_eq!(ones.len(), 1, "csv has exactly one row named one");
    for row in ones {
        assert!(
            row_encodes_to_expected_code(row, row.code),
            "unaltered row must pass"
        );
        assert!(
            !row_encodes_to_expected_code(row, row.code ^ 1),
            "altered row must fail"
        );
    }
}
