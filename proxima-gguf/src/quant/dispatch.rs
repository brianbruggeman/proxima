//! One codec dispatch seam for GGUF tensor decoding.
//!
//! The wire enum is broader than the codecs that have landed in every
//! backend. Keeping this match here means binders, format transforms, and
//! validation tools agree on which formats have a correctness decoder. A
//! missing arm is an explicit capability result, never an accidental byte
//! reinterpretation.

use super::{
    QuantError, bf16, f16, iq2_xs, iq3_xxs, iq4_nl, q1_0, q2_0, q2_k, q3_k, q4_0, q4_1, q4_k, q5_0,
    q5_1, q5_k, q6_k, q8_0, q8_1, q8_k,
};
use crate::types::GgmlType;

/// Decode a complete GGML tensor payload into logical `f32` values.
pub fn dequantize(kind: GgmlType, data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    match kind {
        GgmlType::F16 => f16::dequantize(data, output),
        GgmlType::Bf16 => bf16::dequantize(data, output),
        GgmlType::Q2_K => q2_k::dequantize(data, output),
        GgmlType::Q1_0 => q1_0::dequantize(data, output),
        GgmlType::Q2_0 => q2_0::dequantize(data, output),
        GgmlType::Q3_K => q3_k::dequantize(data, output),
        GgmlType::Q4_0 => q4_0::dequantize(data, output),
        GgmlType::Q4_1 => q4_1::dequantize(data, output),
        GgmlType::Q4_K => q4_k::dequantize(data, output),
        GgmlType::Q5_0 => q5_0::dequantize(data, output),
        GgmlType::Q5_1 => q5_1::dequantize(data, output),
        GgmlType::Q5_K => q5_k::dequantize(data, output),
        GgmlType::Q6_K => q6_k::dequantize(data, output),
        GgmlType::Q8_0 => q8_0::dequantize(data, output),
        GgmlType::Q8_1 => q8_1::dequantize(data, output),
        GgmlType::Q8_K => q8_k::dequantize(data, output),
        GgmlType::Iq2Xs => iq2_xs::dequantize(data, output),
        GgmlType::Iq3Xxs => iq3_xxs::dequantize(data, output),
        GgmlType::Iq4Nl => iq4_nl::dequantize(data, output),
        other => Err(QuantError::UnsupportedCodec {
            codec: other.name(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatches_a_registered_decoder_without_format_specific_caller_logic() {
        let mut block = [0_u8; q4_0::BLOCK_BYTES];
        block[..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut output = [0.0_f32; q4_0::QK4_0];
        dequantize(GgmlType::Q4_0, &block, &mut output).expect("q4_0 is registered");
        assert!(output.iter().all(|value| *value == -8.0));
    }

    #[test]
    fn unsupported_current_format_is_a_typed_capability_result() {
        let error = dequantize(GgmlType::Mxfp4, &[0; 17], &mut [0.0; 32])
            .expect_err("mxfp4 has no scalar decoder yet");
        assert_eq!(error, QuantError::UnsupportedCodec { codec: "mxfp4" });
    }

    #[test]
    fn decodes_legacy_scalar_formats_from_hand_packed_blocks() {
        let mut q1 = [0_u8; q1_0::BLOCK_BYTES];
        q1[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        q1[2] = 0b0000_0001;
        let mut q1_out = [0.0; q1_0::QK_Q1_0];
        dequantize(GgmlType::Q1_0, &q1, &mut q1_out).unwrap();
        assert_eq!(&q1_out[..2], &[2.0, -2.0]);

        let mut q2 = [0_u8; q2_0::BLOCK_BYTES];
        q2[..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        q2[2] = 0b11_10_01_00;
        let mut q2_out = [0.0; q2_0::QK_Q2_0];
        dequantize(GgmlType::Q2_0, &q2, &mut q2_out).unwrap();
        assert_eq!(&q2_out[..4], &[-1.0, 0.0, 1.0, 2.0]);

        let mut q41 = [0_u8; q4_1::BLOCK_BYTES];
        q41[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        q41[2..4].copy_from_slice(&half::f16::from_f32(-1.0).to_le_bytes());
        q41[4] = 0xf0;
        let mut q41_out = [0.0; q4_1::QK_Q4_1];
        dequantize(GgmlType::Q4_1, &q41, &mut q41_out).unwrap();
        assert_eq!(q41_out[0], -1.0);
        assert_eq!(q41_out[16], 6.5);

        let mut q81 = [0_u8; q8_1::BLOCK_BYTES];
        q81[..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        q81[4] = 4;
        q81[5] = 252;
        let mut q81_out = [0.0; q8_1::QK_Q8_1];
        dequantize(GgmlType::Q8_1, &q81, &mut q81_out).unwrap();
        assert_eq!(&q81_out[..2], &[1.0, -1.0]);

        let mut q8k = [0_u8; q8_k::BLOCK_BYTES];
        q8k[..4].copy_from_slice(&0.5f32.to_le_bytes());
        q8k[4] = 6;
        q8k[5] = 250;
        let mut q8k_out = [0.0; q8_k::QK_Q8_K];
        dequantize(GgmlType::Q8_K, &q8k, &mut q8k_out).unwrap();
        assert_eq!(&q8k_out[..2], &[3.0, -3.0]);
    }
}
