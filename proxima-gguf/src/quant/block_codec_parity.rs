//! `proxima_numeric::BlockCodec` against this crate's per-format routines, the incumbents
//! (principle 14): encode bytes and decode bits must be identical, NaN, infinity and zero blocks
//! included, with the incumbent's handling as the oracle.

extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use proxima_numeric::{BlockCodec, BlockLayout};

use crate::quant::QuantError;
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

fn new_decode(codec: &BlockCodec, data: &[u8]) -> Vec<f32> {
    let mut slots = vec![MaybeUninit::<f32>::uninit(); data.len() / codec.block_bytes() * codec.elements()];
    codec.decode(data, &mut slots).expect("whole blocks decode");
    slots.iter().map(|slot| unsafe { slot.assume_init() }).collect()
}

fn encode_mismatches(case: &Case, values: &[f32]) -> usize {
    let codec = BlockCodec::resolve(&case.layout).expect("a simple layout resolves");
    let encode = case.encode.expect("this case has an incumbent encode");
    let blocks = values.len() / codec.elements();
    let mut expected = vec![0u8; blocks * codec.block_bytes()];
    let mut got = vec![0u8; expected.len()];
    encode(values, &mut expected).expect("incumbent encodes");
    codec.encode(values, &mut got).expect("new encodes");
    expected
        .chunks(codec.block_bytes())
        .zip(got.chunks(codec.block_bytes()))
        .filter(|(left, right)| left != right)
        .count()
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

/// IEEE 754 leaves open which operand's payload a two-NaN add returns, so a both-NaN block's
/// elements compare by NaN-ness; every other element compares by bits.
fn decode_mismatches(case: &Case, data: &[u8], both_nan: &[bool]) -> usize {
    let codec = BlockCodec::resolve(&case.layout).expect("a simple layout resolves");
    let mut expected = vec![0.0f32; data.len() / codec.block_bytes() * codec.elements()];
    (case.decode)(data, &mut expected).expect("incumbent decodes");
    let got = new_decode(&codec, data);
    let elements = codec.elements();
    let mut differing = expected.iter().zip(&got).enumerate().filter(|(index, (left, right))| {
        if both_nan.get(index / elements).copied().unwrap_or(false) {
            !(left.is_nan() && right.is_nan())
        } else {
            left.to_bits() != right.to_bits()
        }
    }).map(|(_, pair)| pair);
    let first = differing.next().map(|(left, right)| (left.to_bits(), right.to_bits()));
    let count = usize::from(first.is_some()) + differing.count();
    if let Some((incumbent, new)) = first {
        std::eprintln!("{} first decode mismatch incumbent={incumbent:#010x} new={new:#010x} of {count}", case.name);
    }
    count
}

#[test]
fn encode_bytes_equal_the_incumbent_quantize_on_kv_like_values() {
    for case in cases().iter().filter(|case| case.encode.is_some()) {
        let elements = BlockCodec::resolve(&case.layout).expect("resolves").elements();
        let values = kv_like(65_536 / elements * elements);

        assert_eq!(encode_mismatches(case, &values), 0, "{}", case.name);
    }
}

#[test]
fn encode_bytes_equal_the_incumbent_quantize_on_nan_inf_zero_and_tie_blocks() {
    for case in cases().iter().filter(|case| case.encode.is_some()) {
        let elements = BlockCodec::resolve(&case.layout).expect("resolves").elements();

        assert_eq!(encode_mismatches(case, &special_blocks(elements)), 0, "{}", case.name);
    }
}

#[test]
fn decode_bits_equal_the_incumbent_dequantize_on_arbitrary_block_bytes() {
    let mut state = 0xdead_beef_cafe_f00du64;
    for case in cases() {
        let codec = BlockCodec::resolve(&case.layout).expect("resolves");
        let data = bytes_of(&mut state, 4096 * codec.block_bytes());
        let both_nan = both_nan_blocks(&case.layout, &data, codec.block_bytes());
        if case.layout.min.is_some() && matches!(case.name, "q4_1" | "q5_1") {
            std::eprintln!("{} both-nan scale+min blocks: {}", case.name, both_nan.iter().filter(|&&flag| flag).count());
            assert!(both_nan.contains(&true), "{} seed yields no both-nan block", case.name);
        }

        assert_eq!(decode_mismatches(&case, &data, &both_nan), 0, "{}", case.name);
    }
}

#[test]
fn decode_bits_equal_the_incumbent_dequantize_on_encoded_kv_values() {
    for case in cases() {
        let Some(encode) = case.encode else { continue };
        let codec = BlockCodec::resolve(&case.layout).expect("resolves");
        let values = kv_like(16_384 / codec.elements() * codec.elements());
        let mut packed = vec![0u8; values.len() / codec.elements() * codec.block_bytes()];
        encode(&values, &mut packed).expect("incumbent encodes");

        assert_eq!(decode_mismatches(&case, &packed, &[]), 0, "{}", case.name);
    }
}

#[test]
fn a_block_codec_refuses_a_partial_block_and_a_wrong_output_size() {
    let codec = BlockCodec::resolve(&BlockLayout::Q4_0).expect("q4_0 resolves");
    let mut out = [0u8; 18];

    assert!(codec.encode(&[0.0; 31], &mut out).is_err());
    assert!(codec.encode(&[0.0; 32], &mut out[..17]).is_err());
    assert!(codec.decode(&[0u8; 17], &mut [MaybeUninit::uninit(); 32]).is_err());
}

#[test]
fn layouts_outside_the_simple_family_do_not_resolve() {
    for layout in [BlockLayout::Q4_K, BlockLayout::Q6_K, BlockLayout::NVFP4, BlockLayout::Q8_1_GROUPED, BlockLayout::TQ1_0] {
        assert!(BlockCodec::resolve(&layout).is_none());
    }
}
