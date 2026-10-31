//! Scalar legacy `Q2_0` decoder (64 values / 18 bytes).

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "q2_0";
pub const QK_Q2_0: usize = 64;
pub const BLOCK_BYTES: usize = 18;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_Q2_0;
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
        .zip(output.as_chunks_mut::<QK_Q2_0>().0.iter_mut())
    {
        let mut d = [0; 2];
        d.copy_from_slice(&block[..2]);
        let scale = half::f16::from_le_bytes(d).to_f32();
        for (i, value) in out.iter_mut().enumerate() {
            let q = (block[2 + i / 4] >> (2 * (i % 4))) & 3;
            *value = (i32::from(q) - 1) as f32 * scale;
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Q2_0.block_layout().block_bytes as usize == BLOCK_BYTES);
};

/// PrismML's `quantize_row_q2_0_ref` / `quantize_row_pq2_0_ref` body, shared by the two group
/// sizes: scale is the absolute maximum, code is `round(w / scale) + 1` clamped to `0..=3`.
pub(super) fn quantize_groups<const GROUP: usize, const BYTES: usize>(
    codec: &'static str,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(GROUP) {
        return Err(QuantError::InputNotElementMultiple {
            codec,
            unit: "block",
            found: input.len(),
            block_elements: GROUP,
        });
    }
    let expected = input.len() / GROUP * BYTES;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, block) in input
        .as_chunks::<GROUP>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<BYTES>().0)
    {
        let scale = chunk.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        block[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        block[2..].fill(0);
        for (index, value) in chunk.iter().enumerate() {
            let code = ((value * inverse).round() as i32 + 1).clamp(0, 3) as u8;
            block[2 + index / 4] |= code << (2 * (index % 4));
        }
    }
    Ok(())
}

/// Quantizes whole 64-element runs of `f32` into `Q2_0` blocks.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    quantize_groups::<QK_Q2_0, BLOCK_BYTES>(CODEC, input, output)
}
