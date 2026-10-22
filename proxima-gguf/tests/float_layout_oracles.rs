use half::bf16;
use proxima_gguf::quant::float_layout::{decode, encode, encode_counting};
use proxima_gguf::quant::{bf4_e2m1, bf8_e5m2};
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
fn float_layout_midpoint_ties_to_even_e4m3fn() {
    assert_midpoints_tie_to_even_code(FloatLayout::E4M3FN);
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
fn float_layout_range_saturates_infinity_e5m2_saturating() {
    assert_eq!(encode(f32::INFINITY, FloatLayout::E5M2_SATURATING), 0x7b);
    assert_eq!(
        encode(f32::NEG_INFINITY, FloatLayout::E5M2_SATURATING),
        0xfb
    );
}

#[test]
fn float_layout_e4m3fn_reserves_only_the_top_nan_code() {
    assert_eq!(FloatLayout::E4M3FN.max_normal(), 448.0);
    assert_eq!(decode(0x7e, FloatLayout::E4M3FN), 448.0);
    assert!(decode(0x7f, FloatLayout::E4M3FN).is_nan());
    assert_eq!(decode(0x7f, FloatLayout::E4M3FN).to_bits(), 0x7ff0_0000);
    assert_eq!(decode(0xff, FloatLayout::E4M3FN).to_bits(), 0xfff0_0000);
    assert_eq!(decode(0x78, FloatLayout::E4M3FN), 256.0);
    assert_eq!(encode(f32::INFINITY, FloatLayout::E4M3FN), 0x7e);
    assert_eq!(encode(f32::NAN, FloatLayout::E4M3FN), 0x7f);
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

#[test]
fn float_layout_e5m2_matches_bf8_vectors_csv() {
    for row in bf8_rows() {
        assert!(
            row_encodes_to_expected_code(&row, row.code),
            "{}: encode",
            row.case
        );
        assert_row_decodes(&row);
    }
}

fn assert_row_decodes(row: &Bf8Row) {
    let decoded = decode(row.code, FloatLayout::E5M2);
    if row.input.is_nan() {
        assert!(decoded.is_nan(), "{}: expected nan", row.case);
    } else if row.input.to_bits() & 0x1f_ffff == 0 {
        assert_same_float(decoded, row.input, row.case);
    } else {
        assert_eq!(encode(decoded, FloatLayout::E5M2), row.code, "{}", row.case);
    }
}

fn wrapper_corpus() -> impl Iterator<Item = f32> {
    (0..65_536u32).map(|index| f32::from_bits(index.wrapping_mul(65_537)))
}

#[test]
fn float_layout_wrapper_bf8_equals_generic_e5m2_saturating() {
    for code in 0..=u8::MAX {
        let expected = decode(u32::from(code), FloatLayout::E5M2_SATURATING);
        assert_same_float(
            bf8_e5m2::decode(code),
            expected,
            &format!("decode {code:#04x}"),
        );
    }
    for value in wrapper_corpus() {
        let expected = encode(value, FloatLayout::E5M2_SATURATING);
        assert_eq!(
            u32::from(bf8_e5m2::encode(value)),
            expected,
            "encode {:#010x}",
            value.to_bits()
        );
    }
}

#[test]
fn float_layout_wrapper_bf4_equals_generic_e2m1() {
    for code in 0..16u8 {
        let expected = decode(u32::from(code), FloatLayout::E2M1);
        assert_same_float(
            bf4_e2m1::decode(code),
            expected,
            &format!("decode {code:#x}"),
        );
    }
    for value in wrapper_corpus().filter(|value| value.is_finite()) {
        let expected = encode(value, FloatLayout::E2M1);
        assert_eq!(
            bf4_e2m1::encode(value).map(u32::from),
            Ok(expected),
            "encode {:#010x}",
            value.to_bits()
        );
    }
}

const OCP_MX_TEXT: &str = include_str!("../src/quant/fixtures/ocp_mx_v1_0.txt");
const E3M2_SIGN_BIT: u32 = 0x20;
const E3M2_POSITIVE_GOLDEN: [f32; 32] = [
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
    1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0,
    28.0,
];

fn e3m2_golden_table(positive: &[f32; 32]) -> Vec<(u32, f32)> {
    let positives = positive.iter().zip(0u32..);
    let negatives = positive.iter().zip(E3M2_SIGN_BIT..);
    positives
        .map(|(&value, code)| (code, value))
        .chain(negatives.map(|(&value, code)| (code, -value)))
        .collect()
}

fn e3m2_decode_matches(table: &[(u32, f32)]) -> bool {
    table
        .iter()
        .all(|&(code, value)| decode(code, FloatLayout::E3M2).to_bits() == value.to_bits())
}

fn e3m2_encode_matches(table: &[(u32, f32)]) -> bool {
    table
        .iter()
        .all(|&(code, value)| encode(value, FloatLayout::E3M2) == code)
}

#[test]
fn float_layout_e3m2_golden_decode_all_64_codes() {
    let table = e3m2_golden_table(&E3M2_POSITIVE_GOLDEN);

    assert_eq!(table.len(), 64);
    assert!(e3m2_decode_matches(&table));
    assert!(decode(E3M2_SIGN_BIT, FloatLayout::E3M2).is_sign_negative());
    assert_eq!(decode(E3M2_SIGN_BIT, FloatLayout::E3M2), 0.0);
}

#[test]
fn float_layout_e3m2_golden_encode_each_value() {
    let table = e3m2_golden_table(&E3M2_POSITIVE_GOLDEN);

    assert_eq!(table.len(), 64);
    assert!(e3m2_encode_matches(&table));
}

#[test]
fn float_layout_e3m2_golden_control_altered_row_fails() {
    let mut altered = E3M2_POSITIVE_GOLDEN;
    altered[31] = 24.0;
    let table = e3m2_golden_table(&altered);

    assert!(!e3m2_decode_matches(&table));
    assert!(!e3m2_encode_matches(&table));
}

#[test]
fn float_layout_e3m2_golden_fixture_carries_table4() {
    let newline_count = OCP_MX_TEXT.bytes().filter(|&byte| byte == b'\n').count();

    assert_eq!(newline_count, 592);
    assert!(OCP_MX_TEXT.contains("S 111 112 = ± 24 × 1.75 = ± 28.0"));
    assert!(OCP_MX_TEXT.contains("S 000 012 = ± 2-2 × 0.25 = ± 0.0625"));
}
