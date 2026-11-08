//! Scalar legacy `Q1_0` decoder (128 values / 18 bytes).

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "q1_0";
pub const QK_Q1_0: usize = 128;
pub const BLOCK_BYTES: usize = 18;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_Q1_0;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<QK_Q1_0>().0.iter_mut())
    {
        let mut d = [0; 2];
        d.copy_from_slice(&block[..2]);
        let scale = half::f16::from_le_bytes(d).to_f32();
        for (i, value) in out.iter_mut().enumerate() {
            let bit = (block[2 + i / 8] >> (i % 8)) & 1;
            *value = if bit == 0 { -scale } else { scale };
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Q1_0.wire_shape().block_bytes as usize == BLOCK_BYTES);
};

fn quantize_block(input: &[f32; QK_Q1_0], block: &mut [u8; BLOCK_BYTES]) {
    let scale = input.iter().map(|value| value.abs()).sum::<f32>() / QK_Q1_0 as f32;
    block[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    block[2..].fill(0);
    for (index, value) in input.iter().enumerate() {
        if *value >= 0.0 {
            block[2 + index / 8] |= 1 << (index % 8);
        }
    }
}

/// Quantizes whole 128-element runs of `f32` into `Q1_0` blocks, following
/// PrismML's `quantize_row_q1_0_ref`: scale is the mean absolute value, each bit the sign.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(QK_Q1_0) {
        return Err(QuantError::InputNotElementMultiple {
            codec: CODEC,
            unit: "block",
            found: input.len(),
            block_elements: QK_Q1_0,
        });
    }
    let expected = input.len() / QK_Q1_0 * BLOCK_BYTES;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, block) in input
        .as_chunks::<QK_Q1_0>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        quantize_block(chunk, block);
    }
    Ok(())
}
