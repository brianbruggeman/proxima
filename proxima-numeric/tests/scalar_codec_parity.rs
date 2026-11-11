//! `FormatCodec` (scalar layouts) against independent oracles, under nextest.
//!
//! The oracles are `half` (f16, bf16), a transcription of onnxruntime's `Float8E5M2::ToFloat`
//! (bf8 decode) and a copy of the encode in `proxima-tensor/src/bfloat8.rs` (bf8 encode, the code this
//! crate's scalar path replaced; onnxruntime narrows NaN to `sign | 0x7f` where this narrows to
//! `sign | 0x7e`, and the current behaviour stays). f32 is the identity. The `kv_like` corpus is synthetic (principle 9): a Gaussian-ish sum of twelve uniforms
//! with two x24 outlier channels per 64-wide head, the shape of an attention key row, not a capture
//! of one; the random-bit corpus is deliberately not kv-shaped, it exists to reach NaN, sNaN, inf
//! and subnormal inputs a real row never carries.

// tests state their failures through expect messages
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::mem::MaybeUninit;

use half::{bf16, f16};

use proxima_numeric::float_layout::NAMED_LAYOUTS;
use proxima_numeric::float_layout_codec::{decode as decode_one, encode as encode_one};
use proxima_numeric::{FloatLayout, FormatCodec, NumericFormat, decode_blocks};

const RANDOM_PATTERNS: usize = 1 << 20;
const KV_LIKE_VALUES: usize = 1 << 18;
const F32_QUIET_BIT: u32 = 0x0040_0000;

#[derive(Clone, Copy, Debug)]
enum Scalar {
    Float32,
    Float16,
    BFloat16,
    BFloat8,
}

impl Scalar {
    const ALL: [Self; 4] = [Self::Float32, Self::Float16, Self::BFloat16, Self::BFloat8];

    fn layout(self) -> FloatLayout {
        match self {
            Self::Float32 => FloatLayout::BF32,
            Self::Float16 => FloatLayout::F16,
            Self::BFloat16 => FloatLayout::BF16,
            Self::BFloat8 => FloatLayout::E5M2,
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::BFloat8 => 1,
        }
    }

    fn code_count(self) -> Option<usize> {
        match self {
            Self::Float32 => None,
            Self::Float16 | Self::BFloat16 => Some(1 << 16),
            Self::BFloat8 => Some(1 << 8),
        }
    }

    fn oracle_encode(self, value: f32) -> u32 {
        match self {
            Self::Float32 => value.to_bits(),
            Self::Float16 => u32::from(f16::from_f32(value).to_bits()),
            Self::BFloat16 => u32::from(bf16::from_f32(value).to_bits()),
            Self::BFloat8 => u32::from(bfloat8_encode(value.to_bits())),
        }
    }

    fn oracle_decode(self, code: u32) -> u32 {
        match self {
            Self::Float32 => code,
            Self::Float16 => f16::from_bits(code as u16).to_f32().to_bits(),
            Self::BFloat16 => bf16::from_bits(code as u16).to_f32().to_bits(),
            Self::BFloat8 => bfloat8_decode(code as u8),
        }
    }
}

fn round_shift_to_even(significand: u32, shift: u32) -> u32 {
    if shift == 0 {
        return significand;
    }
    if shift > 24 {
        return 0;
    }
    let quotient = significand >> shift;
    let remainder = significand & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

fn bfloat8_encode(source: u32) -> u8 {
    let sign = ((source >> 24) & 0x80) as u8;
    let exponent = (source >> 23) & 0xff;
    let mantissa = source & 0x7f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 { sign | 0x7c } else { sign | 0x7e };
    }
    if exponent == 0 && mantissa == 0 {
        return sign;
    }

    let (significand, binary_power, leading_exponent) = if exponent == 0 {
        (mantissa, -149, (u32::BITS - mantissa.leading_zeros() - 1) as i32 - 149)
    } else {
        (0x80_0000 | mantissa, exponent as i32 - 150, exponent as i32 - 127)
    };

    if leading_exponent < -14 {
        let shift = -(binary_power + 16) as u32;
        let units = round_shift_to_even(significand, shift);
        return sign | units.min(4) as u8;
    }

    let mut rounded_significand = round_shift_to_even(significand, 21);
    let mut rounded_exponent = leading_exponent;
    if rounded_significand == 8 {
        rounded_exponent += 1;
        rounded_significand = 4;
    }
    if rounded_exponent > 15 {
        return sign | 0x7c;
    }

    sign | (((rounded_exponent + 15) as u8) << 2) | (rounded_significand - 4) as u8
}

/// Transcribed from onnxruntime `include/onnxruntime/core/common/float8.h:454-492`,
/// `Float8E5M2::ToFloat`, the CPU path. Codes 125..=127 widen to `0x7fc0_0000` and codes from 253 to
/// `0xffc0_0000`: the canonical quiet NaN, sign kept, payload dropped. The KV bf8 path that preceded
/// the scalar codec kept NaN payloads, but it was test-only (`#[cfg(test)] Bf8`), so no production
/// behaviour changed.
fn bfloat8_decode(val: u8) -> u32 {
    if val >= 253 {
        return 0xffc0_0000;
    }
    if (125..=127).contains(&val) {
        return 0x7fc0_0000;
    }
    if val == 252 {
        return 0xff80_0000;
    }
    if val == 124 {
        return 0x7f80_0000;
    }
    let mut expo = u32::from((val & 0x7c) >> 2);
    let mut mant = u32::from(val & 0x03);
    let sign = u32::from(val & 0x80);
    let mut res = sign << 24;
    if expo == 0 {
        if mant > 0 {
            expo = 0x7f - 15;
            if mant & 0x2 == 0 {
                mant &= 0x1;
                mant <<= 1;
                expo -= 1;
            }
            res |= (mant & 0x1) << 22;
            res |= expo << 23;
        }
    } else {
        res |= mant << 21;
        expo = expo.wrapping_sub(15).wrapping_add(0x7f);
        res |= expo << 23;
    }
    res
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn kv_like_values(count: usize) -> Vec<f32> {
    let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
    (0..count)
        .map(|index| {
            let gaussian: f32 = (0..12).map(|_| rng.unit()).sum::<f32>() - 6.0;
            if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian }
        })
        .collect()
}

fn random_bit_patterns(count: usize) -> Vec<f32> {
    let mut rng = Xorshift(0xdead_beef_cafe_f00d);
    (0..count).map(|_| f32::from_bits((rng.next() >> 16) as u32)).collect()
}

fn is_signaling_nan(value: f32) -> bool {
    value.is_nan() && value.to_bits() & F32_QUIET_BIT == 0
}

fn codec_of(scalar: Scalar) -> FormatCodec {
    FormatCodec::scalar(&scalar.layout()).expect("a named scalar layout resolves")
}

fn encode_all(scalar: Scalar, values: &[f32]) -> Vec<u8> {
    let codec = codec_of(scalar);
    let mut out = vec![0u8; values.len() * scalar.width()];
    codec.encode(values, &mut out).expect("output sized to the element count");
    out
}

fn decode_all(scalar: Scalar, packed: &[u8]) -> Vec<u32> {
    let codec = codec_of(scalar);
    let count = packed.len() / scalar.width();
    let mut slots = vec![MaybeUninit::<f32>::uninit(); count];
    codec.decode(packed, &mut slots).expect("slots sized to the element count");
    // SAFETY: decode initialises every slot when it returns Ok.
    slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect()
}

fn code_at(packed: &[u8], width: usize, index: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes[..width].copy_from_slice(&packed[index * width..(index + 1) * width]);
    u32::from_le_bytes(bytes)
}

fn encode_disagreements(scalar: Scalar, values: &[f32]) -> Vec<String> {
    let packed = encode_all(scalar, values);
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let got = code_at(&packed, scalar.width(), index);
            let want = scalar.oracle_encode(*value);
            (got != want).then(|| format!("{scalar:?} in={:#010x} got={got:#x} want={want:#x}", value.to_bits()))
        })
        .collect()
}

fn assert_encode_matches_oracle(label: &str, values: &[f32]) {
    for scalar in Scalar::ALL {
        let disagreements = encode_disagreements(scalar, values);
        assert!(
            disagreements.is_empty(),
            "{label}: {} of {} encodes disagree with the oracle, first {:?}",
            disagreements.len(),
            values.len(),
            &disagreements[..disagreements.len().min(3)]
        );
    }
}

#[test]
fn exhaustive_decode_of_every_f16_code_matches_half_bit_for_bit() {
    let packed: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();

    let decoded = decode_all(Scalar::Float16, &packed);

    let mismatched = (0..=u16::MAX).filter(|&code| decoded[usize::from(code)] != Scalar::Float16.oracle_decode(u32::from(code))).count();
    let nan_codes = decoded.iter().filter(|bits| f32::from_bits(**bits).is_nan()).count();
    assert_eq!(decoded.len(), 65536);
    assert_eq!(mismatched, 0);
    assert!(nan_codes > 0, "the f16 code space holds NaN codes; the census must see them");
}

#[test]
fn exhaustive_decode_of_every_bf16_code_matches_half_bit_for_bit() {
    let packed: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();

    let decoded = decode_all(Scalar::BFloat16, &packed);

    let mismatched = (0..=u16::MAX).filter(|&code| decoded[usize::from(code)] != Scalar::BFloat16.oracle_decode(u32::from(code))).count();
    let nan_codes = decoded.iter().filter(|bits| f32::from_bits(**bits).is_nan()).count();
    assert_eq!(decoded.len(), 65536);
    assert_eq!(mismatched, 0);
    assert!(nan_codes > 0, "the bf16 code space holds NaN codes; the census must see them");
}

#[test]
fn exhaustive_decode_of_every_bf8_code_matches_onnxruntime_float8_e5m2_to_float() {
    let packed: Vec<u8> = (0..=u8::MAX).collect();

    let decoded = decode_all(Scalar::BFloat8, &packed);

    let mismatched = (0..=u8::MAX).filter(|&code| decoded[usize::from(code)] != Scalar::BFloat8.oracle_decode(u32::from(code))).count();
    let nan_codes = decoded.iter().filter(|bits| f32::from_bits(**bits).is_nan()).count();
    assert_eq!(decoded.len(), 256);
    assert_eq!(mismatched, 0);
    assert!(nan_codes > 0, "the bf8 code space holds NaN codes; the census must see them");
    assert_eq!(
        [0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff].map(|code| decoded[code]),
        [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000],
        "every bf8 NaN code widens to the canonical quiet NaN with its sign"
    );
}

#[test]
fn random_f32_bit_patterns_encode_like_the_oracles_with_nan_and_snan_present() {
    let inputs = random_bit_patterns(RANDOM_PATTERNS);
    let nan_inputs = inputs.iter().filter(|value| value.is_nan()).count();
    let snan_inputs = inputs.iter().filter(|value| is_signaling_nan(**value)).count();

    assert_eq!(inputs.len(), 1_048_576);
    assert!(nan_inputs > 0, "no NaN inputs: the parity check is blind to NaN handling");
    assert!(snan_inputs > 0, "no sNaN inputs: the parity check is blind to sNaN quieting");
    assert_encode_matches_oracle("random f32 bit patterns", &inputs);
}

#[test]
fn random_f32_bit_patterns_roundtrip_through_decode_like_the_oracles() {
    let inputs = random_bit_patterns(RANDOM_PATTERNS);

    for scalar in Scalar::ALL {
        let packed: Vec<u8> = inputs.iter().flat_map(|value| scalar.oracle_encode(*value).to_le_bytes()[..scalar.width()].to_vec()).collect();
        let decoded = decode_all(scalar, &packed);
        let mismatched = decoded
            .iter()
            .enumerate()
            .filter(|(index, bits)| **bits != scalar.oracle_decode(code_at(&packed, scalar.width(), *index)))
            .count();
        assert_eq!(mismatched, 0, "{scalar:?} decode of oracle-encoded random patterns");
    }
}

#[test]
fn synthetic_kv_like_rows_encode_like_the_oracles() {
    let inputs = kv_like_values(KV_LIKE_VALUES);

    assert_encode_matches_oracle("synthetic kv_like", &inputs);
}

fn signed(values: impl IntoIterator<Item = f32>) -> Vec<f32> {
    values.into_iter().flat_map(|value| [value, -value]).collect()
}

fn one_ulp_around(midpoint: f32) -> [f32; 3] {
    let bits = midpoint.to_bits();
    [f32::from_bits(bits - 1), midpoint, f32::from_bits(bits + 1)]
}

fn neighbour_value(scalar: Scalar, code: u32) -> f64 {
    f64::from(f32::from_bits(scalar.oracle_decode(code)))
}

fn tie_midpoints(scalar: Scalar) -> Vec<f32> {
    let count = scalar.code_count().expect("tie midpoints are defined for narrow formats");
    let finite = |code: u32| neighbour_value(scalar, code).is_finite();
    let positive_codes = (count / 2) as u32;
    let mut midpoints = Vec::new();
    for code in 1..positive_codes - 1 {
        if !finite(code) {
            break;
        }
        let here = neighbour_value(scalar, code);
        let step = if finite(code + 1) { neighbour_value(scalar, code + 1) - here } else { here - neighbour_value(scalar, code - 1) };
        midpoints.extend(one_ulp_around(((here + step / 2.0) as f32).abs()));
    }
    midpoints
}

fn specials_of(scalar: Scalar) -> Vec<(&'static str, Vec<f32>)> {
    let mut classes = vec![
        ("zeros", signed([0.0])),
        ("infinities", signed([f32::INFINITY])),
        ("f32 extremes", signed([f32::MAX])),
        ("min subnormal", signed([f32::from_bits(1)])),
        ("min normal", signed([f32::MIN_POSITIVE])),
    ];
    if scalar.code_count().is_some() {
        classes.push(("tie midpoints and one f32 ulp either side", signed(tie_midpoints(scalar))));
    }
    classes
}

#[test]
fn specials_zero_inf_extremes_subnormal_and_ties_encode_like_the_oracles() {
    for scalar in Scalar::ALL {
        for (class, values) in specials_of(scalar) {
            assert!(values.len() >= 2, "{scalar:?} {class}: census is empty");
            let disagreements = encode_disagreements(scalar, &values);
            assert!(
                disagreements.is_empty(),
                "{scalar:?} {class}: {} of {} disagree, first {:?}",
                disagreements.len(),
                values.len(),
                &disagreements[..disagreements.len().min(3)]
            );
        }
    }
}

#[test]
fn tie_census_reaches_every_narrow_format_in_volume() {
    let f16_ties = tie_midpoints(Scalar::Float16).len();
    let bf16_ties = tie_midpoints(Scalar::BFloat16).len();
    let bf8_ties = tie_midpoints(Scalar::BFloat8).len();

    assert!(f16_ties > 30_000, "f16 ties {f16_ties}");
    assert!(bf16_ties > 30_000, "bf16 ties {bf16_ties}");
    assert!(bf8_ties > 100, "bf8 ties {bf8_ties}");
}

#[test]
fn an_exact_halfway_f32_rounds_to_even_for_each_narrow_format() {
    let ties = [
        (Scalar::BFloat8, 1.125_f32, 0x3c_u32),
        (Scalar::BFloat8, 1.375, 0x3e),
        (Scalar::BFloat8, 61440.0, 0x7c),
        (Scalar::Float16, f32::from_bits(0x3f80_1000), 0x3c00),
        (Scalar::BFloat16, f32::from_bits(0x3f80_8000), 0x3f80),
    ];

    for (scalar, input, expected) in ties {
        let packed = encode_all(scalar, &[input]);
        assert_eq!(code_at(&packed, scalar.width(), 0), expected, "{scalar:?} {input}");
    }
}

const STRIDE: usize = 4093;

fn respelled(layout: &FloatLayout) -> FloatLayout {
    FloatLayout { bias_override: Some(layout.bias()), ..*layout }
}

fn layout_bytes(layout: &FloatLayout) -> usize {
    usize::from(layout.width().div_ceil(8))
}

fn slice_encode(layout: &FloatLayout, values: &[f32]) -> Vec<u8> {
    let codec = FormatCodec::scalar(layout).expect("a supported layout resolves");
    let mut out = vec![0u8; values.len() * layout_bytes(layout)];
    codec.encode(values, &mut out).expect("output sized to the element count");
    out
}

fn slice_decode(layout: &FloatLayout, packed: &[u8]) -> Vec<u32> {
    let codec = FormatCodec::scalar(layout).expect("a supported layout resolves");
    let mut slots = vec![MaybeUninit::<f32>::uninit(); packed.len() / layout_bytes(layout)];
    codec.decode(packed, &mut slots).expect("slots sized to the element count");
    // SAFETY: decode initialises every slot when it returns Ok.
    slots.iter().map(|slot| unsafe { slot.assume_init() }.to_bits()).collect()
}

fn stride_sweep() -> Vec<f32> {
    (0..=u32::MAX).step_by(STRIDE).map(f32::from_bits).collect()
}

fn code_mask(layout: &FloatLayout) -> u32 {
    u32::MAX >> (32 - u32::from(layout.width()))
}

#[test]
fn per_element_encode_of_a_negative_into_a_sign_less_layout_stays_inside_the_layout_width() {
    let layout = FloatLayout::F8_E8M0;

    let code = encode_one(-1.0, layout);

    assert!(code <= code_mask(&layout), "float8_e8m0 has no sign bit; encode(-1.0) returned {code:#x}");
    assert_eq!(code, encode_one(1.0, layout), "a sign-less layout stores the magnitude only");
}

#[test]
fn a_negative_infinity_and_a_negative_nan_into_a_sign_less_layout_store_their_magnitude_only() {
    let layout = FloatLayout::F8_E8M0;

    assert_eq!(encode_one(f32::NEG_INFINITY, layout), encode_one(f32::INFINITY, layout));
    assert_eq!(encode_one(f32::from_bits(0xffc0_0000), layout), 0xff);
}

#[test]
fn every_named_layout_resolves_except_the_five_wider_than_32_bits() {
    let unresolved: Vec<&str> = NAMED_LAYOUTS
        .iter()
        .filter(|(layout, _)| FormatCodec::scalar(layout).is_none())
        .map(|(_, name)| *name)
        .collect();

    assert_eq!(NAMED_LAYOUTS.len(), 29);
    assert_eq!(unresolved, ["int64", "uint64", "int128", "uint128", "float64"]);
}

#[test]
fn slice_encode_equals_per_element_encode_for_every_supported_named_layout_and_its_respelling() {
    let inputs = stride_sweep();
    let mut checked_layouts = 0;

    for (layout, name) in NAMED_LAYOUTS.iter().filter(|(layout, _)| layout.width() <= 32) {
        for spelling in [*layout, respelled(layout)] {
            let packed = slice_encode(&spelling, &inputs);
            let mismatched = inputs
                .iter()
                .enumerate()
                .filter(|(index, value)| {
                    code_at(&packed, layout_bytes(&spelling), *index) != encode_one(**value, spelling)
                })
                .count();
            assert_eq!(mismatched, 0, "{name} encode, respelled={}", spelling != *layout);
        }
        checked_layouts += 1;
    }

    assert_eq!(checked_layouts, 24);
    assert!(inputs.len() > 1_000_000, "the stride sweep covers about 2^20 patterns, got {}", inputs.len());
}

#[test]
fn slice_decode_equals_per_element_decode_for_every_supported_named_layout_and_its_respelling() {
    let mut checked_layouts = 0;

    for (layout, name) in NAMED_LAYOUTS.iter().filter(|(layout, _)| layout.width() <= 32) {
        let mask = code_mask(layout);
        let codes: Vec<u32> = if layout.width() <= 16 {
            (0..=mask).collect()
        } else {
            (0..=u32::MAX).step_by(STRIDE).map(|code| code & mask).collect()
        };
        let packed: Vec<u8> = codes.iter().flat_map(|code| code.to_le_bytes()[..layout_bytes(layout)].to_vec()).collect();
        for spelling in [*layout, respelled(layout)] {
            let decoded = slice_decode(&spelling, &packed);
            let mismatched = codes
                .iter()
                .zip(&decoded)
                .filter(|(code, bits)| decode_one(**code, spelling).to_bits() != **bits)
                .count();
            assert_eq!(mismatched, 0, "{name} decode, respelled={}", spelling != *layout);
        }
        checked_layouts += 1;
    }

    assert_eq!(checked_layouts, 24);
}

#[test]
fn resolved_decode_equals_decode_blocks_bit_for_bit_for_every_named_scalar_including_binary32() {
    let inputs = random_bit_patterns(1 << 16);
    let signalling = inputs.iter().filter(|value| is_signaling_nan(**value)).count();

    for layout in [
        FloatLayout::BF32,
        FloatLayout::F16,
        FloatLayout::BF16,
        FloatLayout::E5M2_SATURATING,
        FloatLayout::E5M2,
        FloatLayout::E4M3FN,
        FloatLayout::E3M2,
        FloatLayout::E2M1,
    ] {
        let packed = slice_encode(&layout, &inputs);
        let resolved = slice_decode(&layout, &packed);
        let mut routed = vec![0.0f32; inputs.len()];
        decode_blocks(&NumericFormat::Scalar(layout), &packed, &mut routed).expect("a scalar format decodes");

        let mismatched = resolved.iter().zip(&routed).filter(|(left, right)| **left != right.to_bits()).count();

        assert_eq!(mismatched, 0, "{layout:?}");
    }
    assert!(signalling > 0, "the binary32 identity case is only proved if sNaN inputs were present");
}
