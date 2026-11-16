//! `proxima_numeric::FormatCodec` against `proxima_gguf::quant`, the encoder and decoder
//! `bind::quantize_to_kind` and `expert_slab::dequantize_expert` call today (principle 14: the
//! incumbent is the oracle). Encode bytes and decode f32 bits must be identical for every format
//! both sides support; the per-format coverage table is asserted, not just printed.

#![allow(clippy::expect_used)]

use core::mem::MaybeUninit;
use std::format;
use std::string::String;
use std::vec;
use std::vec::Vec;

use proxima_gguf::GgmlType;
use proxima_gguf::quant::{QuantError, dispatch, f16, bf16, pq2_0, ptq1_0, q1_0, q2_0, q2_k, q3_k, q4_0, q4_k, q5_0, q5_k, q6_k, q8_0};
use proxima_numeric::{FormatCodec, NumericFormat};

use crate::bind::quantize_to_kind;

type GgufEncode = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;

const ROW_ELEMENTS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    Both,
    GgufOnly,
    NumericOnly,
    Neither,
}

struct FormatRow {
    format: NumericFormat,
    name: String,
    gguf_encode: Option<GgufEncode>,
    gguf_decode: bool,
    codec: Option<FormatCodec>,
}

impl FormatRow {
    fn numeric_encode(&self) -> bool {
        self.codec.as_ref().is_some_and(FormatCodec::can_encode)
    }

    fn gguf_any(&self) -> bool {
        self.gguf_encode.is_some() || self.gguf_decode
    }

    fn coverage(&self) -> Coverage {
        let gguf = self.gguf_any();
        let numeric = self.codec.is_some();
        match (gguf, numeric) {
            (true, true) => Coverage::Both,
            (true, false) => Coverage::GgufOnly,
            (false, true) => Coverage::NumericOnly,
            (false, false) => Coverage::Neither,
        }
    }
}

fn float32_encode(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    quantize_to_kind(NumericFormat::Float32, input, output)
}

fn gguf_encoder(format: NumericFormat) -> Option<GgufEncode> {
    Some(match format {
        NumericFormat::Float32 => float32_encode,
        NumericFormat::Q8_0 => q8_0::quantize,
        NumericFormat::Q4K => q4_k::quantize,
        NumericFormat::Q5K => q5_k::quantize,
        NumericFormat::Q6K => q6_k::quantize,
        NumericFormat::Q3K => q3_k::quantize,
        NumericFormat::Q2K => q2_k::quantize,
        NumericFormat::Q4_0 => q4_0::quantize,
        NumericFormat::Q5_0 => q5_0::quantize,
        NumericFormat::Float16 => f16::quantize,
        NumericFormat::BFloat16 => bf16::quantize,
        NumericFormat::Q1_0 => q1_0::quantize,
        NumericFormat::Q2_0 => q2_0::quantize,
        NumericFormat::Pq2_0 => pq2_0::quantize,
        NumericFormat::Ptq1_0 => ptq1_0::quantize,
        _ => return None,
    })
}

fn gguf_decode(format: NumericFormat, data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if format == NumericFormat::Float32 {
        for (slot, chunk) in output.iter_mut().zip(data.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*chunk);
        }
        return Ok(());
    }
    match GgmlType::from_numeric_format(&format) {
        Some(kind) => dispatch::dequantize(kind, data, output),
        None => Err(QuantError::UnsupportedCodec { codec: "no ggml wire type" }),
    }
}

fn all_formats() -> Vec<NumericFormat> {
    let mut formats = vec![NumericFormat::Float32];
    formats.extend(NumericFormat::ALL);
    formats
}

fn probe_decodes(format: NumericFormat) -> bool {
    let elements = usize::from(format.elements_per_block()) * 2;
    let bytes = vec![0u8; usize::from(format.bytes_per_block()) * 2];
    let mut output = vec![0f32; elements];
    gguf_decode(format, &bytes, &mut output).is_ok()
}

fn format_rows() -> Vec<FormatRow> {
    all_formats()
        .into_iter()
        .map(|format| FormatRow {
            format,
            name: format!("{format:?}"),
            gguf_encode: gguf_encoder(format),
            gguf_decode: probe_decodes(format),
            codec: FormatCodec::for_format(&format),
        })
        .collect()
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

/// Gaussian-ish weights with an outlier channel every 64, the shape of a real weight or key row.
fn gaussian_row_with_outliers() -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..ROW_ELEMENTS)
        .map(|index| {
            let gaussian: f32 = (0..12).map(|_| unit(&mut state)).sum::<f32>() - 6.0;
            if index % 64 == 5 { gaussian * 24.0 } else { gaussian * 0.05 }
        })
        .collect()
}

/// Two blocks of zeros, then negatives, then a single large-magnitude outlier.
fn zeros_negatives_and_one_outlier_row() -> Vec<f32> {
    let mut row = vec![0.0f32; ROW_ELEMENTS];
    for (index, slot) in row.iter_mut().enumerate().skip(ROW_ELEMENTS / 2) {
        *slot = -0.01 * ((index % 31) as f32 + 1.0);
    }
    row[ROW_ELEMENTS - 7] = 1.0e4;
    row
}

fn input_rows() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("gaussian_with_outliers", gaussian_row_with_outliers()),
        ("zeros_negatives_one_outlier", zeros_negatives_and_one_outlier_row()),
    ]
}

fn numeric_encode(codec: &FormatCodec, values: &[f32]) -> Vec<u8> {
    let packed = codec.packed_len(values.len()).expect("row is whole blocks");
    let mut out = vec![0u8; packed];
    let written = codec.encode(values, &mut out).expect("numeric encode of a whole-block row");
    assert_eq!(written, packed);
    out
}

fn numeric_decode(codec: &FormatCodec, data: &[u8], elements: usize) -> Vec<f32> {
    let mut out: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); elements];
    let written = codec.decode(data, &mut out).expect("numeric decode of whole blocks");
    assert_eq!(written, elements / codec.elements() * codec.block_bytes());
    out.into_iter().map(|slot| unsafe { slot.assume_init() }).collect()
}

fn first_byte_difference(left: &[u8], right: &[u8]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!("length gguf={} numeric={}", left.len(), right.len()));
    }
    left.iter().zip(right).position(|(a, b)| a != b).map(|at| format!("byte {at}: gguf={:#04x} numeric={:#04x}", left[at], right[at]))
}

fn first_bit_difference(left: &[f32], right: &[f32]) -> Option<String> {
    if left.len() != right.len() {
        return Some(format!("length gguf={} numeric={}", left.len(), right.len()));
    }
    left.iter().zip(right).position(|(a, b)| a.to_bits() != b.to_bits()).map(|at| {
        format!("element {at}: gguf={} ({:#010x}) numeric={} ({:#010x})", left[at], left[at].to_bits(), right[at], right[at].to_bits())
    })
}

fn pseudo_random_blocks(codec: &FormatCodec, blocks: usize) -> Vec<u8> {
    let mut state = 0xdead_beef_cafe_f00du64 ^ codec.block_bytes() as u64;
    (0..blocks * codec.block_bytes()).map(|_| (next(&mut state) >> 33) as u8).collect()
}

fn encode_differences(rows: &[FormatRow]) -> (usize, Vec<String>) {
    let mut compared = 0;
    let mut differences = Vec::new();
    for row in rows {
        let (Some(encode), Some(codec)) = (row.gguf_encode, row.codec.as_ref()) else { continue };
        if !codec.can_encode() {
            continue;
        }
        for (label, values) in input_rows() {
            let mut gguf = vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
            encode(&values, &mut gguf).expect("gguf encode of a whole-block row");
            let numeric = numeric_encode(codec, &values);
            compared += 1;
            if let Some(detail) = first_byte_difference(&gguf, &numeric) {
                differences.push(format!("ENCODE {} on {label}: {detail}", row.name));
            }
        }
    }
    (compared, differences)
}

fn decode_source_bytes(row: &FormatRow, codec: &FormatCodec, values: &[f32]) -> Vec<u8> {
    if let Some(encode) = row.gguf_encode {
        let mut bytes = vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
        encode(values, &mut bytes).expect("gguf encode");
        return bytes;
    }
    if codec.can_encode() {
        return numeric_encode(codec, values);
    }
    pseudo_random_blocks(codec, values.len() / codec.elements())
}

fn decode_differences(rows: &[FormatRow]) -> (usize, Vec<String>) {
    let mut compared = 0;
    let mut differences = Vec::new();
    for row in rows {
        let Some(codec) = row.codec.as_ref() else { continue };
        if !row.gguf_decode {
            continue;
        }
        for (label, values) in input_rows() {
            let bytes = decode_source_bytes(row, codec, &values);
            let mut gguf = vec![0f32; values.len()];
            gguf_decode(row.format, &bytes, &mut gguf).expect("gguf decode of whole blocks");
            let numeric = numeric_decode(codec, &bytes, values.len());
            compared += 1;
            if let Some(detail) = first_bit_difference(&gguf, &numeric) {
                differences.push(format!("DECODE {} on {label}: {detail}", row.name));
            }
        }
    }
    (compared, differences)
}

#[test]
fn per_format_coverage_has_formats_supported_by_both_sides() {
    let rows = format_rows();
    let both = rows.iter().filter(|row| row.coverage() == Coverage::Both).count();
    for row in &rows {
        std::println!(
            "COVERAGE {:<12} {:?} gguf_encode={} gguf_decode={} numeric_encode={} numeric_decode={}",
            row.name,
            row.coverage(),
            row.gguf_encode.is_some(),
            row.gguf_decode,
            row.numeric_encode(),
            row.codec.is_some(),
        );
    }
    std::println!("COVERAGE-COUNTS both={both} total={}", rows.len());
    assert!(both > 0, "no format is supported by both gguf and FormatCodec; the parity test would compare nothing");
}

#[test]
fn encode_bytes_equal_gguf_quantize_for_every_format_both_sides_encode() {
    let rows = format_rows();
    let (compared, differences) = encode_differences(&rows);
    std::println!("ENCODE-COUNTS compared={compared} differences={}", differences.len());
    assert!(compared > 0, "no (format, row) pair was compared");
    assert!(differences.is_empty(), "encode differs from the incumbent:\n{}", differences.join("\n"));
}

#[test]
fn decode_f32_bits_equal_gguf_dequantize_for_every_format_both_sides_decode() {
    let rows = format_rows();
    let (compared, differences) = decode_differences(&rows);
    std::println!("DECODE-COUNTS compared={compared} differences={}", differences.len());
    assert!(compared > 0, "no (format, row) pair was compared");
    assert!(differences.is_empty(), "decode differs from the incumbent:\n{}", differences.join("\n"));
}

#[test]
fn quantize_to_kind_reports_q5_0_and_q5_1_unsupported_while_numeric_codec_encodes_both() {
    let values = gaussian_row_with_outliers();
    let mut out = vec![0u8; 512 / 32 * 22];
    for format in [NumericFormat::Q5_0, NumericFormat::Q5_1] {
        let result = quantize_to_kind(format, &values, &mut out);
        assert!(matches!(result, Err(QuantError::UnsupportedCodec { .. })), "{format:?}: {result:?}");
        let codec = FormatCodec::for_format(&format).expect("numeric resolves q5 formats");
        assert!(codec.can_encode(), "{format:?}: FormatCodec resolves but cannot encode");
    }
}
