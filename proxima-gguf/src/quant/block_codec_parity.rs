//! `proxima_numeric::FormatCodec` (block layouts) against this crate's per-format routines, the incumbents
//! (principle 14): encode bytes and decode bits must be identical, NaN, infinity and zero blocks
//! included, with the incumbent's handling as the oracle.

extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use proxima_numeric::{BlockLayout, FormatCodec};

use crate::quant::QuantError;
use crate::quant::llama_reference::{dequantize_row_mxfp4, dequantize_row_tq2_0, quantize_row_tq2_0_ref};
use crate::quant::{q1_0, q2_0, pq2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, q8_k};

type Encode = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;
type Decode = fn(&[u8], &mut [f32]) -> Result<(), QuantError>;

struct Case {
    name: &'static str,
    layout: BlockLayout,
    encode: Option<Encode>,
    decode: Decode,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "q4_0", layout: BlockLayout::Q4_0, encode: Some(q4_0::quantize), decode: q4_0::dequantize },
        Case { name: "q4_1", layout: BlockLayout::Q4_1, encode: None, decode: q4_1::dequantize },
        Case { name: "q5_0", layout: BlockLayout::Q5_0, encode: Some(q5_0::quantize), decode: q5_0::dequantize },
        Case { name: "q5_1", layout: BlockLayout::Q5_1, encode: None, decode: q5_1::dequantize },
        Case { name: "q8_0", layout: BlockLayout::Q8_0, encode: Some(q8_0::quantize), decode: q8_0::dequantize },
        Case { name: "q8_1", layout: BlockLayout::Q8_1, encode: None, decode: q8_1::dequantize },
        Case { name: "q8_k", layout: BlockLayout::Q8_K, encode: None, decode: q8_k::dequantize },
        Case { name: "q1_0", layout: BlockLayout::Q1_0, encode: Some(q1_0::quantize), decode: q1_0::dequantize },
        Case { name: "q2_0", layout: BlockLayout::Q2_0, encode: Some(q2_0::quantize), decode: q2_0::dequantize },
        Case { name: "pq2_0", layout: BlockLayout::PQ2_0, encode: Some(pq2_0::quantize), decode: pq2_0::dequantize },
    ]
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f32 {
    (next(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// Gaussian-ish values with two outlier channels per 64, the shape of a real key row.
fn kv_like(count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..count)
        .map(|index| {
            let gaussian: f32 = (0..12).map(|_| unit(&mut state)).sum::<f32>() - 6.0;
            if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian }
        })
        .collect()
}

/// One block's worth per row: all zero, a lone NaN, a lone +inf, a lone -inf, NaN beside the peak,
/// signed zeros, an exact +/- tie, and a block of one repeated value.
fn special_blocks(elements: usize) -> Vec<f32> {
    let base = kv_like(elements);
    let mut blocks = Vec::new();
    blocks.extend(vec![0.0f32; elements]);
    for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::from_bits(0x7fa0_0001)] {
        let mut block = base.clone();
        block[elements / 3] = special;
        blocks.extend(block);
    }
    let mut both = base.clone();
    both[1] = f32::NAN;
    both[2] = f32::INFINITY;
    blocks.extend(both);
    blocks.extend((0..elements).map(|index| if index % 2 == 0 { 4.0 } else { -4.0 }));
    blocks.extend(vec![0.75f32; elements]);
    blocks.extend(vec![f32::MAX; elements]);
    blocks
}

fn bytes_of(state: &mut u64, count: usize) -> Vec<u8> {
    (0..count).map(|_| (next(state) >> 24) as u8).collect()
}

fn new_decode(codec: &FormatCodec, data: &[u8]) -> Vec<f32> {
    let mut slots = vec![MaybeUninit::<f32>::uninit(); data.len() / codec.block_bytes() * codec.elements()];
    codec.decode(data, &mut slots).expect("whole blocks decode");
    // decode returned ok, which initialises every slot
    slots.iter().map(|slot| unsafe { slot.assume_init() }).collect()
}

/// The indices of the blocks whose bytes differ between `reference` and the template's encode.
fn differing_blocks(layout: &BlockLayout, reference: impl Fn(&[f32], &mut [u8]), values: &[f32]) -> Vec<usize> {
    let codec = FormatCodec::block(layout).expect("a simple layout resolves");
    let blocks = values.len() / codec.elements();
    let mut expected = vec![0u8; blocks * codec.block_bytes()];
    let mut got = vec![0u8; expected.len()];
    reference(values, &mut expected);
    codec.encode(values, &mut got).expect("new encodes");
    expected
        .chunks(codec.block_bytes())
        .zip(got.chunks(codec.block_bytes()))
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .map(|(index, (left, right))| {
            std::eprintln!("block {index} differs\n  incumbent {left:02x?}\n  template  {right:02x?}");
            index
        })
        .collect()
}

fn encode_mismatches(case: &Case, values: &[f32]) -> usize {
    let encode = case.encode.expect("this case has an incumbent encode");
    differing_blocks(&case.layout, |input, out| encode(input, out).expect("incumbent encodes"), values).len()
}

fn is_f16_nan(low: u8, high: u8) -> bool {
    u16::from_le_bytes([low, high]) & 0x7fff > 0x7c00
}

/// Per block: whether a layout with a min stores a NaN scale beside a NaN min (scale at bytes 0..2,
/// min at 2..4 for every layout here that has one).
fn both_nan_blocks(layout: &BlockLayout, data: &[u8], block_bytes: usize) -> Vec<bool> {
    data.chunks(block_bytes)
        .map(|block| layout.min.is_some() && is_f16_nan(block[0], block[1]) && is_f16_nan(block[2], block[3]))
        .collect()
}

/// Every element compares by bits, a NaN's sign and payload included: the layout codec and the
/// incumbent port share one multiply-add rule (`proxima_numeric::fused`), so a block holding a NaN
/// scale, a NaN min or an infinite scale decodes to the same bits in both.
fn decode_mismatches(case: &Case, data: &[u8]) -> usize {
    let decode = case.decode;
    decode_differences(case.name, &case.layout, |bytes, out| decode(bytes, out).expect("incumbent decodes"), data)
}

fn decode_differences(
    name: &str,
    layout: &BlockLayout,
    reference: impl Fn(&[u8], &mut [f32]),
    data: &[u8],
) -> usize {
    let codec = FormatCodec::block(layout).expect("a simple layout resolves");
    let mut expected = vec![0.0f32; data.len() / codec.block_bytes() * codec.elements()];
    reference(data, &mut expected);
    let got = new_decode(&codec, data);
    let mut differing = expected.iter().zip(&got).filter(|(left, right)| left.to_bits() != right.to_bits());
    let first = differing.next().map(|(left, right)| (left.to_bits(), right.to_bits()));
    let count = usize::from(first.is_some()) + differing.count();
    if let Some((incumbent, new)) = first {
        std::eprintln!("{name} first decode mismatch incumbent={incumbent:#010x} new={new:#010x} of {count}");
    }
    count
}

#[test]
fn encode_bytes_equal_the_incumbent_quantize_on_kv_like_values() {
    for case in cases().iter().filter(|case| case.encode.is_some()) {
        let elements = FormatCodec::block(&case.layout).expect("resolves").elements();
        let values = kv_like(65_536 / elements * elements);

        assert_eq!(encode_mismatches(case, &values), 0, "{}", case.name);
    }
}

#[test]
fn encode_bytes_equal_the_incumbent_quantize_on_nan_inf_zero_and_tie_blocks() {
    for case in cases().iter().filter(|case| case.encode.is_some()) {
        let elements = FormatCodec::block(&case.layout).expect("resolves").elements();

        assert_eq!(encode_mismatches(case, &special_blocks(elements)), 0, "{}", case.name);
    }
}

#[test]
fn decode_bits_equal_the_incumbent_dequantize_on_arbitrary_block_bytes() {
    let mut state = 0xdead_beef_cafe_f00du64;
    for case in cases() {
        let codec = FormatCodec::block(&case.layout).expect("resolves");
        let data = bytes_of(&mut state, 4096 * codec.block_bytes());
        let both_nan = both_nan_blocks(&case.layout, &data, codec.block_bytes());
        if case.layout.min.is_some() && matches!(case.name, "q4_1" | "q5_1") {
            std::eprintln!("{} both-nan scale+min blocks: {}", case.name, both_nan.iter().filter(|&&flag| flag).count());
            assert!(both_nan.contains(&true), "{} seed yields no both-nan block", case.name);
        }

        assert_eq!(decode_mismatches(&case, &data), 0, "{}", case.name);
    }
}

#[test]
fn decode_bits_equal_the_incumbent_dequantize_on_encoded_kv_values() {
    for case in cases() {
        let Some(encode) = case.encode else { continue };
        let codec = FormatCodec::block(&case.layout).expect("resolves");
        let values = kv_like(16_384 / codec.elements() * codec.elements());
        let mut packed = vec![0u8; values.len() / codec.elements() * codec.block_bytes()];
        encode(&values, &mut packed).expect("incumbent encodes");

        assert_eq!(decode_mismatches(&case, &packed), 0, "{}", case.name);
    }
}

#[test]
fn a_block_codec_refuses_a_partial_block_and_a_wrong_output_size() {
    let codec = FormatCodec::block(&BlockLayout::Q4_0).expect("q4_0 resolves");
    let mut out = [0u8; 18];

    assert!(codec.encode(&[0.0; 31], &mut out).is_err());
    assert!(codec.encode(&[0.0; 32], &mut out[..17]).is_err());
    assert!(codec.decode(&[0u8; 17], &mut [MaybeUninit::uninit(); 32]).is_err());
}

fn tq2_0_encode_differences(values: &[f32]) -> Vec<usize> {
    differing_blocks(&BlockLayout::TQ2_0, quantize_row_tq2_0_ref, values)
}

fn tq2_0_decode_differences(data: &[u8]) -> usize {
    decode_differences("tq2_0", &BlockLayout::TQ2_0, dequantize_row_tq2_0, data)
}

#[test]
fn tq2_0_encode_bytes_equal_llama_cpp_quantize_row_tq2_0_ref_on_kv_like_values() {
    let values = kv_like(256 * 8192);

    assert_eq!(tq2_0_encode_differences(&values), Vec::<usize>::new());
}

#[test]
fn tq2_0_encode_bytes_equal_llama_cpp_quantize_row_tq2_0_ref_on_nan_inf_zero_and_tie_blocks() {
    let values = special_blocks(256);
    let blocks = values.len() / 256;

    let differing = tq2_0_encode_differences(&values);

    std::eprintln!("tq2_0 special blocks {blocks}, differing {}", differing.len());
    assert_eq!(differing, Vec::<usize>::new());
}

#[test]
fn tq2_0_encode_bytes_equal_llama_cpp_quantize_row_tq2_0_ref_on_a_nan_first_and_a_nan_last_element() {
    let mut first = kv_like(256);
    first[0] = f32::NAN;
    let mut last = kv_like(256);
    last[255] = f32::NAN;
    let values = [first, last].concat();

    assert_eq!(tq2_0_encode_differences(&values), Vec::<usize>::new());
}

#[test]
fn tq2_0_decode_bits_equal_llama_cpp_dequantize_row_tq2_0_on_arbitrary_block_bytes() {
    let mut state = 0x7a52_0000_1234_5678u64;
    let data = bytes_of(&mut state, 4096 * 66);
    let nan_scales = data.chunks(66).filter(|block| is_f16_nan(block[64], block[65])).count();

    std::eprintln!("tq2_0 random blocks with nan scale: {nan_scales} of 4096");
    assert!(nan_scales > 0, "the seed must yield nan-scale blocks");
    assert_eq!(tq2_0_decode_differences(&data), 0);
}

#[test]
fn tq2_0_decode_bits_equal_llama_cpp_dequantize_row_tq2_0_on_reference_encoded_kv_blocks() {
    let values = kv_like(256 * 64);
    let mut packed = vec![0u8; 64 * 66];
    quantize_row_tq2_0_ref(&values, &mut packed);

    assert_eq!(tq2_0_decode_differences(&packed), 0);
}

const MXFP4_NAN_SCALE_BYTE: u8 = 0xff;

/// One block per shared-exponent byte, each carrying every nibble value twice (low and high).
fn mxfp4_block_per_scale_byte() -> Vec<u8> {
    (0..=255u8)
        .flat_map(|scale| core::iter::once(scale).chain((0..16u8).map(|pair| ((15 - pair) << 4) | pair)))
        .collect()
}

/// Splits 17-byte blocks into those whose scale byte is not E8M0 NaN and those whose is.
fn split_by_nan_scale(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let (nan, finite): (Vec<&[u8]>, Vec<&[u8]>) = data.chunks(17).partition(|block| block[0] == MXFP4_NAN_SCALE_BYTE);
    (finite.concat(), nan.concat())
}

const E2M1_NEGATIVE_ZERO_NIBBLE: u8 = 0x8;

/// The E2M1 code nibble of element `index` in a 17-byte mxfp4 block stream: elements 0..16 sit in the
/// low nibbles of bytes 1..17, elements 16..32 in the high nibbles.
fn mxfp4_nibble(data: &[u8], index: usize) -> u8 {
    let byte = data[index / 32 * 17 + 1 + index % 16];
    if index % 32 < 16 { byte & 0x0f } else { byte >> 4 }
}

/// Elements of `finite` blocks where llama.cpp and the template disagree, plus how many elements carry
/// the negative-zero nibble. Those are compared by value (both 0.0) with the template's sign bit
/// required set; every other element is compared by bits.
fn mxfp4_decode_report(codec: &FormatCodec, finite: &[u8]) -> (Vec<(usize, u32, u32)>, usize) {
    let mut expected = vec![0.0f32; finite.len() / 17 * 32];
    dequantize_row_mxfp4(finite, &mut expected);
    let got = new_decode(codec, finite);
    let mut negative_zeros = 0;
    let mut mismatches = Vec::new();
    for (index, (left, right)) in expected.iter().zip(&got).enumerate() {
        let agrees = if mxfp4_nibble(finite, index) == E2M1_NEGATIVE_ZERO_NIBBLE {
            negative_zeros += 1;
            *left == 0.0 && *right == 0.0 && right.is_sign_negative()
        } else {
            left.to_bits() == right.to_bits()
        };
        if !agrees {
            mismatches.push((index, left.to_bits(), right.to_bits()));
        }
    }
    (mismatches, negative_zeros)
}

/// llama.cpp's `ggml_e8m0_to_fp32_half` documents "NaNs are not handled here" (ggml-impl.h:487-510),
/// so 0xff decodes there to a finite 2^127; OCP MX v1.0 reserves E8M0 0xff as NaN
/// (fixtures/ocp_mx_v1_0.txt:503). Blocks off 0xff are compared by bits against
/// `dequantize_row_mxfp4`; blocks on 0xff must decode to NaN per the spec line.
fn assert_mxfp4_decode(data: &[u8]) -> (usize, usize, usize) {
    let (finite, nan) = split_by_nan_scale(data);
    let codec = FormatCodec::block(&BlockLayout::MXFP4).expect("mxfp4 resolves");

    let (mismatches, negative_zeros) = mxfp4_decode_report(&codec, &finite);
    assert!(mismatches.is_empty(), "mxfp4 mismatches (element, incumbent bits, new bits): {} first {:x?}", mismatches.len(), &mismatches[..mismatches.len().min(8)]);
    assert!(new_decode(&codec, &nan).iter().all(|value| value.is_nan()), "ocp mx: e8m0 0xff is nan");
    (finite.len() / 17, nan.len() / 17, negative_zeros)
}

/// E2M1 code 0x8 is -0.0 in the template: OCP MX v1.0 gives sign bit S with E=00 M=0 as a zero
/// (fixtures/ocp_mx_v1_0.txt:443-452), onnxruntime `float4.h` `Fp4ToFloatConversionCpuHelper` computes
/// `sign * (mantissa / 2.f)`, and ml_dtypes `float4_e2m1fn` has the same negative zero. llama.cpp
/// decodes +0 for that nibble only because `kvalues_fp4` (ggml-common.h:1126-1128) is an int8 table,
/// which has no -0, so those elements are compared by value with the template's sign bit asserted.
#[test]
fn mxfp4_decode_bits_equal_llama_cpp_dequantize_row_mxfp4_except_the_spec_reserved_nan_scale() {
    let mut state = 0x1717_2024_dead_5eedu64;
    let random = bytes_of(&mut state, 4096 * 17);
    let hand_built = mxfp4_block_per_scale_byte();

    let (random_finite, random_nan, random_zeros) = assert_mxfp4_decode(&random);
    let (built_finite, built_nan, built_zeros) = assert_mxfp4_decode(&hand_built);

    std::eprintln!("mxfp4 random blocks finite {random_finite} nan-scale {random_nan} negative-zero elements {random_zeros}; hand-built finite {built_finite} nan-scale {built_nan} negative-zero elements {built_zeros}");
    assert!(random_finite > 0 && random_nan > 0 && built_finite > 0 && built_nan > 0);
    assert!(random_zeros > 0 && built_zeros > 0, "the inputs must contain nibble 0x8");
}

#[test]
fn mxfp4_nibbles_decode_to_the_ocp_e2m1_values_at_unit_scale() {
    let mut block = vec![127u8];
    block.extend((0..16u8).map(|pair| ((15 - pair) << 4) | pair));
    let codec = FormatCodec::block(&BlockLayout::MXFP4).expect("mxfp4 resolves");

    let decoded = new_decode(&codec, &block);

    let positive = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    assert_eq!(&decoded[..8], &positive);
    assert_eq!(decoded[8], -0.0);
    assert_eq!(decoded[15], -6.0);
}

const F16_SPECIAL_BITS: [(&str, u16); 7] = [
    ("+0", 0x0000),
    ("-0", 0x8000),
    ("+inf", 0x7c00),
    ("-inf", 0xfc00),
    ("nan", 0x7e00),
    ("max", 0x7bff),
    ("min subnormal", 0x0001),
];

/// Every (scale, min) pair of the special f16 values in the first four bytes of an otherwise random
/// block, as q4_1 and q5_1 store them.
fn special_scale_and_min_blocks(block_bytes: usize) -> Vec<u8> {
    let mut state = 0x5151_0001_0002_0003u64;
    let mut data = Vec::new();
    for (_, scale) in F16_SPECIAL_BITS {
        for (_, min) in F16_SPECIAL_BITS {
            let mut block = bytes_of(&mut state, block_bytes);
            block[..2].copy_from_slice(&scale.to_le_bytes());
            block[2..4].copy_from_slice(&min.to_le_bytes());
            data.extend(block);
        }
    }
    data
}

#[test]
fn formats_with_a_min_decode_special_scale_and_min_blocks_like_the_incumbent() {
    for case in cases().iter().filter(|case| case.layout.min.is_some()) {
        let codec = FormatCodec::block(&case.layout).expect("resolves");
        let data = special_scale_and_min_blocks(codec.block_bytes());
        let both_nan = both_nan_blocks(&case.layout, &data, codec.block_bytes());
        let both_nan_count = both_nan.iter().filter(|&&flag| flag).count();

        std::eprintln!("{} special blocks {} both-nan {both_nan_count}", case.name, both_nan.len());
        assert_eq!(both_nan.len(), 49, "{}", case.name);
        assert_eq!(both_nan_count, 1, "{}", case.name);
        assert_eq!(decode_mismatches(case, &data), 0, "{}", case.name);
    }
}

/// Where a fit's rounding step flips: `PeakRound` rounds `value * inverse` half away from zero, so
/// the boundary is a fraction of exactly one half; `SignedMax` truncates `value * inverse + bias +
/// 0.5`, so the boundary is that sum being exactly an integer.
enum Boundary {
    PeakRound { reach: f32 },
    SignedMax { bias: f32 },
}

fn boundary_hits(values: &[f32], boundary: &Boundary) -> usize {
    match *boundary {
        Boundary::PeakRound { reach } => {
            let peak = values.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
            let inverse = 1.0 / (peak / reach);
            values.iter().filter(|&&value| (value * inverse).abs().fract() == 0.5).count()
        }
        Boundary::SignedMax { bias } => {
            let signed_max = values.iter().fold(0.0f32, |held, &value| if held.abs() < value.abs() { value } else { held });
            let inverse = 1.0 / (signed_max / -bias);
            values.iter().filter(|&&value| (value * inverse + bias + 0.5).fract() == 0.0).count()
        }
    }
}

fn zero_padded(elements: usize, leading: &[f32]) -> Vec<f32> {
    let mut block = vec![0.0f32; elements];
    block[..leading.len()].copy_from_slice(leading);
    block
}

fn incumbent_encode(name: &str, values: &[f32], out: &mut [u8]) {
    match name {
        "q4_0" => q4_0::quantize(values, out).expect("incumbent encodes"),
        "q5_0" => q5_0::quantize(values, out).expect("incumbent encodes"),
        "q8_0" => q8_0::quantize(values, out).expect("incumbent encodes"),
        "q2_0" => q2_0::quantize(values, out).expect("incumbent encodes"),
        "pq2_0" => pq2_0::quantize(values, out).expect("incumbent encodes"),
        "tq2_0" => quantize_row_tq2_0_ref(values, out),
        other => unreachable!("{other} has no incumbent encode in this test"),
    }
}

/// One block per format with elements on the exact rounding boundary of its fit, built from power of
/// two peaks so every product is exact. q1_0's code is `value >= 0`, which has no rounding tie.
fn tie_blocks() -> Vec<(&'static str, BlockLayout, Vec<f32>, Boundary)> {
    let two_bit_ties = [4.0, 2.0, -2.0, -4.0];
    vec![
        ("q4_0", BlockLayout::Q4_0, zero_padded(32, &[-8.0, 0.5, -0.5, 1.5, 7.5, -7.5]), Boundary::SignedMax { bias: 8.0 }),
        ("q5_0", BlockLayout::Q5_0, zero_padded(32, &[-16.0, 0.5, -0.5, 15.5, -15.5, 2.5]), Boundary::SignedMax { bias: 16.0 }),
        ("q8_0", BlockLayout::Q8_0, zero_padded(32, &[127.0, 0.5, -0.5, 126.5, -126.5, 1.5, -2.5]), Boundary::PeakRound { reach: 127.0 }),
        ("q2_0", BlockLayout::Q2_0, zero_padded(64, &two_bit_ties), Boundary::PeakRound { reach: 1.0 }),
        ("pq2_0", BlockLayout::PQ2_0, zero_padded(128, &two_bit_ties), Boundary::PeakRound { reach: 1.0 }),
        ("tq2_0", BlockLayout::TQ2_0, zero_padded(256, &two_bit_ties), Boundary::PeakRound { reach: 1.0 }),
    ]
}

#[test]
fn encode_bytes_equal_the_incumbent_on_blocks_whose_elements_sit_exactly_on_the_rounding_boundary() {
    for (name, layout, values, boundary) in tie_blocks() {
        let hits = boundary_hits(&values, &boundary);

        std::eprintln!("{name} elements on the rounding boundary: {hits}");
        assert!(hits > 0, "{name} block must hit its boundary exactly");
        let differing = differing_blocks(&layout, |input, out| incumbent_encode(name, input, out), &values);
        assert_eq!(differing, Vec::<usize>::new(), "{name}");
    }
}

#[test]
fn layouts_outside_the_simple_family_do_not_resolve() {
    for layout in [BlockLayout::Q4_K, BlockLayout::Q6_K, BlockLayout::NVFP4, BlockLayout::Q8_1_GROUPED, BlockLayout::TQ1_0] {
        assert!(FormatCodec::block(&layout).is_none());
    }
}
