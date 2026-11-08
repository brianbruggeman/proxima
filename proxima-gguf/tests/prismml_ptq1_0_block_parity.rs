//! Parity of proxima's `PTQ1_0` decoder with PrismML's published block spec, on blocks copied
//! byte-for-byte from the downloaded `Ternary-Bonsai-2-27B-PTQ1_0.gguf`.
//!
//! Spec: github.com/PrismML-Eng/llama.cpp, branch `prism`.
//! - `ggml/include/ggml.h`: `GGML_TYPE_PTQ1_0 = 143`.
//! - `ggml/src/ggml-common.h`: `QK_PTQ1_0 128`,
//!   `block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; }` (28 bytes).
//! - `ggml/src/ggml-quants.c`: `dequantize_row_ptq1_0` with `ptq1_0_stages = {32, 16, 8}`.
//!
//! Source blocks: the first two blocks of `output.weight` in `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
//! (sha256 recorded in the bonsai-27b run log), data section at 11120992, tensor at +0.

#![allow(clippy::expect_used)]

use proxima_gguf::quant::QuantError;
use proxima_gguf::quant::dispatch::dequantize;
use proxima_gguf::types::GgmlType;

const PTQ1_0_BLOCKS: [u8; 56] = [
    0x2f, 0x38, 0xb1, 0xbc, 0x39, 0x43, 0x1c, 0xb6, 0x57, 0x87, 0xdc, 0x47, 0xee, 0xca, 0xd6, 0xb8,
    0x92, 0x24, 0x65, 0x4a, 0x2e, 0x3f, 0x39, 0xf2, 0xf7, 0x33, 0x48, 0x24, 0x85, 0x2e, 0xa8, 0x81,
    0xae, 0x05, 0xe9, 0xe7, 0x43, 0x15, 0x09, 0x3d, 0x4c, 0x0e, 0x60, 0x38, 0x25, 0x10, 0x7e, 0x2b,
    0x5d, 0xa9, 0xb5, 0x16, 0x5f, 0xb5, 0x35, 0x24,
];

/// 0x2448 is f16 sign 0, exponent 9, mantissa 0x048: 2^(9-15) * (1 + 72/1024).
const FIRST_BLOCK_SCALE: f32 = 1.070_312_5 / 64.0;

fn spec_dequantize_row_ptq1_0(block: &[u8; 28]) -> Vec<f32> {
    const STAGES: [usize; 3] = [32, 16, 8];
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let scale = half::f16::from_le_bytes([block[26], block[27]]).to_f32();
    let mut output = Vec::new();
    let mut consumed = 0;
    for stage in STAGES {
        while consumed + stage <= 24 {
            for power in &POW3[..5] {
                for lane in 0..stage {
                    let scaled = block[consumed + lane].wrapping_mul(*power);
                    let trit = (u16::from(scaled) * 3) >> 8;
                    output.push((trit as i16 - 1) as f32 * scale);
                }
            }
            consumed += stage;
        }
    }
    for power in &POW3[..4] {
        for lane in 0..2 {
            let scaled = block[24 + lane].wrapping_mul(*power);
            let trit = (u16::from(scaled) * 3) >> 8;
            output.push((trit as i16 - 1) as f32 * scale);
        }
    }
    output
}

fn proxima_decode(bytes: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0; bytes.len() / 28 * 128];
    dequantize(GgmlType::Ptq1_0, bytes, &mut output).expect("registered codec decodes whole blocks");
    output
}

fn first_block() -> [u8; 28] {
    PTQ1_0_BLOCKS[..28].try_into().expect("28 byte block")
}

fn second_block() -> [u8; 28] {
    PTQ1_0_BLOCKS[28..].try_into().expect("28 byte block")
}

#[test]
fn first_block_hand_walks_to_the_spec_trits_of_qs_byte_zero_and_qh() {
    let scale = FIRST_BLOCK_SCALE;
    let decoded = proxima_decode(&PTQ1_0_BLOCKS[..28]);

    assert_eq!(decoded[0], -scale);
    assert_eq!(decoded[16], 0.0);
    assert_eq!(decoded[32], 0.0);
    assert_eq!(decoded[48], scale);
    assert_eq!(decoded[64], scale);
    assert_eq!(decoded[120], scale);
    assert_eq!(decoded[121], -scale);
    assert_eq!(decoded[122], scale);
}

#[test]
fn layout_walker_matches_the_transcribed_spec_function_on_both_blocks() {
    for block in [first_block(), second_block()] {
        assert_eq!(proxima_decode(&block), spec_dequantize_row_ptq1_0(&block));
    }
}

#[test]
fn scalar_decoder_matches_the_transcribed_spec_function_on_both_blocks() {
    for block in [first_block(), second_block()] {
        let mut output = vec![0.0; 128];
        proxima_gguf::quant::ptq1_0::dequantize(&block, &mut output).expect("whole block");
        assert_eq!(output, spec_dequantize_row_ptq1_0(&block));
    }
}

#[test]
fn real_blocks_use_only_the_three_ternary_code_points() {
    let scale = FIRST_BLOCK_SCALE;
    let decoded = proxima_decode(&PTQ1_0_BLOCKS[..28]);
    assert!(decoded.iter().all(|value| [-scale, 0.0, scale].contains(value)));
}

#[test]
fn prismml_wire_id_and_block_size_match_ggml_h_and_ggml_common_h() {
    let layout = GgmlType::Ptq1_0.wire_shape();
    assert_eq!(GgmlType::Ptq1_0.to_wire(), 143);
    assert_eq!(GgmlType::from_wire(143), Some(GgmlType::Ptq1_0));
    assert_eq!(layout.block_elements, 128);
    assert_eq!(layout.block_bytes, 28);
}

#[test]
fn truncated_block_is_a_typed_error_not_a_misread() {
    let mut output = vec![0.0; 128];
    let error = dequantize(GgmlType::Ptq1_0, &PTQ1_0_BLOCKS[..27], &mut output)
        .expect_err("27 bytes is not a whole 28-byte block");
    assert!(matches!(
        error,
        QuantError::InputNotBlockMultiple {
            found: 27,
            block_bytes: 28,
            ..
        }
    ));
}

#[test]
fn quantizing_the_decoded_real_blocks_reproduces_their_bytes() {
    let decoded = proxima_decode(&PTQ1_0_BLOCKS);
    let mut encoded = vec![0u8; PTQ1_0_BLOCKS.len()];
    proxima_gguf::quant::ptq1_0::quantize(&decoded, &mut encoded).expect("two whole blocks");
    assert_eq!(encoded, PTQ1_0_BLOCKS);
}

#[test]
fn quantizing_a_ragged_run_is_a_typed_error() {
    let mut output = vec![0u8; 28];
    let error = proxima_gguf::quant::ptq1_0::quantize(&[0.5; 100], &mut output)
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
