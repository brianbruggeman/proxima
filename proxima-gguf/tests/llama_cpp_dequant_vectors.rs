#![allow(clippy::expect_used)]
//! Both proxima decoders against golden dequantize vectors produced by llama.cpp itself.
//!
//! `tests/data/llama_cpp_dequant_vectors.bin` holds, per format, block bytes and the hashes of what
//! llama.cpp's own `dequantize_row_<t>` wrote for them (build, commit and regeneration notes in
//! `tests/data/llama_cpp_dequant_vectors.md`). The other parity tests in this crate compare the
//! codec with the gguf ports, which share one multiply-add rule (`proxima_numeric::fused`) and so
//! cannot say anything about llama.cpp; this one can.
//!
//! File layout, little-endian: `"LCDV"`, `u32` version, `u32` format count; then per format a `u8`
//! name length and the name, `u32` block bytes, `u32` elements per block, `u32` record count, and
//! the records, each the block bytes then `H_all` (`u64`) then `H_canon` (`u64`). Records 0..32
//! are random blocks, 32..64 random blocks with every f16 scale/min forced to a finite normal, and
//! the rest every combination of the special f16 values (+nan, -nan with payload, +inf, -inf, -0,
//! three subnormals) in the scale/min fields, so 64 + 8 records for a one-field format and
//! 64 + 64 for a two-field one.
//!
//! `H_all` is FNV-1a-64 over the little-endian bytes of every output `f32`'s bits. `H_canon` is
//! the same with every NaN replaced by `0x7fc00000`. Finite results are the same on every target,
//! so `H_canon` must match everywhere; NaN bits are what the arm64 fused build of llama.cpp
//! produced, so `H_all` is asserted only where the fused rule is in force on aarch64.

use std::mem::MaybeUninit;

use proxima_gguf::quant::{QuantError, q2_k, q3_k, q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0};
use proxima_numeric::{FormatCodec, NumericFormat};

const VECTORS: &[u8] = include_bytes!("data/llama_cpp_dequant_vectors.bin");
const PLAIN_RECORDS: usize = 64;
const CANON_NAN: u32 = 0x7fc0_0000;

type PortFn = fn(&[u8], &mut [f32]) -> Result<(), QuantError>;

struct Record<'a> {
    block: &'a [u8],
    hash_all: u64,
    hash_canon: u64,
}

struct Section<'a> {
    elements: usize,
    records: Vec<Record<'a>>,
}

fn take<'a>(cursor: &mut &'a [u8], length: usize) -> &'a [u8] {
    let (head, rest) = cursor.split_at(length);
    *cursor = rest;
    head
}

fn take_u32(cursor: &mut &[u8]) -> u32 {
    u32::from_le_bytes(take(cursor, 4).try_into().expect("four bytes make a u32"))
}

fn take_u64(cursor: &mut &[u8]) -> u64 {
    u64::from_le_bytes(take(cursor, 8).try_into().expect("eight bytes make a u64"))
}

fn section(wanted: &str) -> Section<'static> {
    let mut cursor = VECTORS;
    assert_eq!(take(&mut cursor, 4), b"LCDV", "the vector file starts with its magic");
    assert_eq!(take_u32(&mut cursor), 1, "the vector file is version 1");
    for _ in 0..take_u32(&mut cursor) {
        let name_length = usize::from(take(&mut cursor, 1)[0]);
        let name = take(&mut cursor, name_length);
        let block_bytes = take_u32(&mut cursor) as usize;
        let elements = take_u32(&mut cursor) as usize;
        let count = take_u32(&mut cursor) as usize;
        let records: Vec<Record<'static>> = (0..count)
            .map(|_| Record {
                block: take(&mut cursor, block_bytes),
                hash_all: take_u64(&mut cursor),
                hash_canon: take_u64(&mut cursor),
            })
            .collect();
        if name == wanted.as_bytes() {
            return Section { elements, records };
        }
    }
    panic!("format {wanted} is not in the vector file");
}

fn fnv1a(output: &[f32], canonical: bool) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in output {
        let bits = if canonical && value.is_nan() { CANON_NAN } else { value.to_bits() };
        for byte in bits.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn decode_with_codec(codec: &FormatCodec, block: &[u8], elements: usize) -> Vec<f32> {
    let mut slots = vec![MaybeUninit::<f32>::uninit(); elements];
    codec.decode(block, &mut slots).expect("the codec decodes one whole block");
    // decode returned ok, which initialises every slot
    slots.into_iter().map(|slot| unsafe { slot.assume_init() }).collect()
}

fn check(name: &str, format: &NumericFormat, port: PortFn, combos: usize) {
    let section = section(name);
    assert_eq!(section.records.len(), PLAIN_RECORDS + combos, "{name}: record count");
    // the layout codec resolves the five legacy block formats only; the K formats have no codec
    // and are checked through the gguf port alone
    let codec = FormatCodec::for_format(format);
    let exact_nan_bits = proxima_numeric::fused::FUSED && cfg!(target_arch = "aarch64");
    let mut failures = Vec::new();
    for (index, record) in section.records.iter().enumerate() {
        let special = index >= PLAIN_RECORDS;
        let mut ported = vec![0.0f32; section.elements];
        port(record.block, &mut ported).expect("the port decodes one whole block");
        let mut outputs = vec![("gguf port", ported)];
        if let Some(active) = &codec {
            outputs.push(("codec", decode_with_codec(active, record.block, section.elements)));
        }
        for (who, output) in &outputs {
            if fnv1a(output, true) != record.hash_canon {
                failures.push(format!("{name} record {index} special={special}: {who} H_canon differs from llama.cpp"));
            }
            if exact_nan_bits && fnv1a(output, false) != record.hash_all {
                failures.push(format!("{name} record {index} special={special}: {who} H_all differs from llama.cpp"));
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches, first {:?}", failures.len(), &failures[..failures.len().min(8)]);
}

#[proxima::test]
async fn q4_0_matches_llama_cpp() {
    check("q4_0", &NumericFormat::Q4_0, q4_0::dequantize, 8);
}

#[proxima::test]
async fn q4_1_matches_llama_cpp() {
    check("q4_1", &NumericFormat::Q4_1, q4_1::dequantize, 64);
}

#[proxima::test]
async fn q5_0_matches_llama_cpp() {
    check("q5_0", &NumericFormat::Q5_0, q5_0::dequantize, 8);
}

#[proxima::test]
async fn q5_1_matches_llama_cpp() {
    check("q5_1", &NumericFormat::Q5_1, q5_1::dequantize, 64);
}

#[proxima::test]
async fn q8_0_matches_llama_cpp() {
    check("q8_0", &NumericFormat::Q8_0, q8_0::dequantize, 8);
}

#[proxima::test]
async fn q2_k_matches_llama_cpp() {
    check("q2_K", &NumericFormat::Q2K, q2_k::dequantize, 64);
}

#[proxima::test]
async fn q3_k_matches_llama_cpp() {
    check("q3_K", &NumericFormat::Q3K, q3_k::dequantize, 8);
}

#[proxima::test]
async fn q4_k_matches_llama_cpp() {
    check("q4_K", &NumericFormat::Q4K, q4_k::dequantize, 64);
}

#[proxima::test]
async fn q5_k_matches_llama_cpp() {
    check("q5_K", &NumericFormat::Q5K, q5_k::dequantize, 64);
}

#[proxima::test]
async fn q6_k_matches_llama_cpp() {
    check("q6_K", &NumericFormat::Q6K, q6_k::dequantize, 8);
}
