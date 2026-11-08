//! Parity of proxima's `Q1_0` / `Q2_0` / `PQ2_0` decoders with PrismML's published
//! block spec, on blocks copied byte-for-byte from the downloaded Bonsai checkpoints.
//!
//! Spec: github.com/PrismML-Eng/llama.cpp, branch `prism`.
//! - `ggml/include/ggml.h`: `GGML_TYPE_Q1_0 = 41`, `GGML_TYPE_Q2_0 = 42`, `GGML_TYPE_PQ2_0 = 142`.
//! - `ggml/src/ggml-common.h`: `QK1_0 128` (`block_q1_0 { ggml_half d; uint8_t qs[16]; }`),
//!   `QK2_0 64` (`block_q2_0 { ggml_half d; uint8_t qs[16]; }`),
//!   `QK_PQ2_0 128` (`block_pq2_0 { ggml_half d; uint8_t qs[32]; }`).
//! - `ggml/src/ggml-quants.c`: `dequantize_row_q1_0` (`bit ? d : -d`, bit `j % 8` of byte
//!   `j / 8`), `dequantize_row_q2_0` / `dequantize_row_pq2_0`
//!   (`((qs[j / 4] >> ((j % 4) * 2)) & 3) - 1) * d`).
//!
//! Source blocks, each the first block of `blk.0.attn_k.weight`:
//! - `Bonsai-1.7B-Q1_0.gguf` (sha256 3d7c6c90...f7e3), data section at 5945120, tensor at +43688864.
//! - `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` (sha256 6d0ecb3d...5c91), data at 5945280, tensor at +87369536.
//! - `Ternary-Bonsai-1.7B-PQ2_0.gguf` (sha256 de68ba48...ae50), data at 5945280, tensor at +82516128.

#![allow(clippy::expect_used)]

use proxima_gguf::quant::QuantError;
use proxima_gguf::quant::dispatch::dequantize;
use proxima_gguf::types::GgmlType;

const Q1_0_BLOCK: [u8; 18] = [
    0x08, 0x28, 0xfa, 0x30, 0x11, 0xde, 0x71, 0x34, 0xd5, 0x5d, 0x22, 0x28, 0xfb, 0xfc, 0xbf, 0xac,
    0x57, 0x00,
];

const Q2_0_BLOCKS: [u8; 36] = [
    0x78, 0x27, 0x55, 0x89, 0x5a, 0x4a, 0x41, 0x52, 0xa4, 0x95, 0x11, 0x65, 0x54, 0x0a, 0x42, 0xa1,
    0x55, 0x25, 0x78, 0x27, 0x59, 0x05, 0x59, 0x29, 0x99, 0x16, 0x64, 0x98, 0x99, 0x46, 0x90, 0x84,
    0x2a, 0x64, 0x51, 0x06,
];

const PQ2_0_BLOCK: [u8; 34] = [
    0x78, 0x27, 0x55, 0x89, 0x5a, 0x4a, 0x41, 0x52, 0xa4, 0x95, 0x11, 0x65, 0x54, 0x0a, 0x42, 0xa1,
    0x55, 0x25, 0x59, 0x05, 0x59, 0x29, 0x99, 0x16, 0x64, 0x98, 0x99, 0x46, 0x90, 0x84, 0x2a, 0x64,
    0x51, 0x06,
];

/// 0x2778 is f16 sign 0, exponent 9, mantissa 0x378: 2^(9-15) * (1 + 888/1024).
const TERNARY_SCALE: f32 = 1.867_187_5 / 64.0;

/// 0x2808 is f16 sign 0, exponent 10, mantissa 8: 2^(10-15) * (1 + 8/1024).
const BINARY_SCALE: f32 = 1.007_812_5 / 32.0;

fn spec_dequantize_row_q1_0(block: &[u8]) -> Vec<f32> {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    (0..128)
        .map(|j| {
            let bit = (block[2 + j / 8] >> (j % 8)) & 1;
            if bit == 1 { scale } else { -scale }
        })
        .collect()
}

fn spec_dequantize_row_two_bit(block: &[u8], group: usize) -> Vec<f32> {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    (0..group)
        .map(|j| {
            let code = (block[2 + j / 4] >> ((j % 4) * 2)) & 3;
            (i32::from(code) - 1) as f32 * scale
        })
        .collect()
}

fn proxima_decode(kind: GgmlType, bytes: &[u8], elements: usize) -> Vec<f32> {
    let mut output = vec![0.0; elements];
    dequantize(kind, bytes, &mut output).expect("registered codec decodes a whole block");
    output
}

#[test]
fn q1_0_block_from_bonsai_1_7b_hand_walks_to_the_spec_first_sixteen_values() {
    let scale = BINARY_SCALE;
    let decoded = proxima_decode(GgmlType::Q1_0, &Q1_0_BLOCK, 128);

    let hand_walked_first_sixteen = [
        -scale, scale, -scale, scale, scale, scale, scale, scale, -scale, -scale, -scale, -scale,
        scale, scale, -scale, -scale,
    ];
    assert_eq!(&decoded[..16], &hand_walked_first_sixteen);
}

#[test]
fn q1_0_block_from_bonsai_1_7b_matches_the_transcribed_spec_function_on_all_128_values() {
    let decoded = proxima_decode(GgmlType::Q1_0, &Q1_0_BLOCK, 128);
    assert_eq!(decoded, spec_dequantize_row_q1_0(&Q1_0_BLOCK));
}

#[test]
fn pq2_0_block_from_ternary_bonsai_1_7b_hand_walks_to_the_spec_first_sixteen_values() {
    let scale = TERNARY_SCALE;
    let decoded = proxima_decode(GgmlType::Pq2_0, &PQ2_0_BLOCK, 128);

    let hand_walked_first_sixteen = [
        0.0, 0.0, 0.0, 0.0, 0.0, scale, -scale, scale, scale, scale, 0.0, 0.0, scale, scale,
        -scale, 0.0,
    ];
    assert_eq!(&decoded[..16], &hand_walked_first_sixteen);
}

#[test]
fn pq2_0_block_from_ternary_bonsai_1_7b_matches_the_transcribed_spec_function_on_all_128_values() {
    let decoded = proxima_decode(GgmlType::Pq2_0, &PQ2_0_BLOCK, 128);
    assert_eq!(decoded, spec_dequantize_row_two_bit(&PQ2_0_BLOCK, 128));
}

#[test]
fn q2_0_group_64_blocks_from_ternary_bonsai_1_7b_match_the_transcribed_spec_function() {
    let decoded = proxima_decode(GgmlType::Q2_0, &Q2_0_BLOCKS, 128);

    let mut expected = spec_dequantize_row_two_bit(&Q2_0_BLOCKS[..18], 64);
    expected.extend(spec_dequantize_row_two_bit(&Q2_0_BLOCKS[18..], 64));
    assert_eq!(decoded, expected);
}

#[test]
fn group_128_and_group_64_checkpoints_of_one_model_decode_the_same_weights() {
    let group_64 = proxima_decode(GgmlType::Q2_0, &Q2_0_BLOCKS, 128);
    let group_128 = proxima_decode(GgmlType::Pq2_0, &PQ2_0_BLOCK, 128);
    assert_eq!(group_64, group_128);
}

#[test]
fn ternary_checkpoint_blocks_use_only_the_three_ternary_code_points() {
    let decoded = proxima_decode(GgmlType::Pq2_0, &PQ2_0_BLOCK, 128);
    assert!(
        decoded
            .iter()
            .all(|value| [-TERNARY_SCALE, 0.0, TERNARY_SCALE].contains(value))
    );
}

#[test]
fn prismml_wire_ids_and_block_sizes_match_ggml_h_and_ggml_common_h() {
    let spec = [
        (GgmlType::Q1_0, 41, 128, 18),
        (GgmlType::Q2_0, 42, 64, 18),
        (GgmlType::Pq2_0, 142, 128, 34),
    ];
    for (kind, wire, elements, bytes) in spec {
        let layout = kind.wire_shape();
        assert_eq!(kind.to_wire(), wire, "{kind:?} wire id");
        assert_eq!(GgmlType::from_wire(wire), Some(kind), "{kind:?} decodes");
        assert_eq!(layout.block_elements, elements, "{kind:?} block elements");
        assert_eq!(layout.block_bytes, bytes, "{kind:?} block bytes");
    }
}

#[test]
fn truncated_pq2_0_block_is_a_typed_error_not_a_misread() {
    let mut output = vec![0.0; 128];
    let error = dequantize(GgmlType::Pq2_0, &PQ2_0_BLOCK[..33], &mut output)
        .expect_err("33 bytes is not a whole 34-byte block");
    assert!(matches!(
        error,
        QuantError::InputNotBlockMultiple {
            found: 33,
            block_bytes: 34,
            ..
        }
    ));
}

fn proxima_encode(quantize: QuantizeFn, values: &[f32], block_bytes: usize) -> Vec<u8> {
    let mut output = vec![0u8; values.len() / 128 * block_bytes];
    quantize(values, &mut output).expect("whole blocks quantize");
    output
}

type QuantizeFn = fn(&[f32], &mut [u8]) -> Result<(), QuantError>;

#[test]
fn quantizing_the_decoded_real_pq2_0_block_reproduces_its_bytes() {
    let decoded = proxima_decode(GgmlType::Pq2_0, &PQ2_0_BLOCK, 128);
    let encoded = proxima_encode(proxima_gguf::quant::pq2_0::quantize, &decoded, 34);
    assert_eq!(encoded, PQ2_0_BLOCK);
}

#[test]
fn quantizing_the_decoded_real_q2_0_blocks_reproduces_their_bytes() {
    let decoded = proxima_decode(GgmlType::Q2_0, &Q2_0_BLOCKS, 128);
    let mut encoded = vec![0u8; 36];
    proxima_gguf::quant::q2_0::quantize(&decoded, &mut encoded).expect("two whole blocks");
    assert_eq!(encoded, Q2_0_BLOCKS);
}

#[test]
fn quantizing_the_decoded_real_q1_0_block_reproduces_its_bytes() {
    let decoded = proxima_decode(GgmlType::Q1_0, &Q1_0_BLOCK, 128);
    let encoded = proxima_encode(proxima_gguf::quant::q1_0::quantize, &decoded, 18);
    assert_eq!(encoded, Q1_0_BLOCK);
}

#[test]
fn quantizing_a_ragged_run_is_a_typed_error() {
    let mut output = vec![0u8; 34];
    let error = proxima_gguf::quant::pq2_0::quantize(&[0.5; 100], &mut output)
        .expect_err("100 elements is not a whole 128-element block");
    assert!(matches!(
        error,
        QuantError::InputNotElementMultiple {
            found: 100,
            block_elements: 128,
            ..
        }
    ));
}
