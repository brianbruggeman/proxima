//! `Q5_0`: 5-bit signed weights in 32-element blocks, one `f16` scale and
//! four packed high-bit flags. This is the legacy format used by several
//! token-embedding tensors in otherwise `Q4_K_M` checkpoints.

use crate::quant::QuantError;
use crate::types::GgmlType;

pub const QK5_0: usize = 32;
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q5_0.block_layout();
    assert!(layout.block_elements as usize == QK5_0);
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QH_OFFSET: usize = 2;
const QS_OFFSET: usize = 6;
const HALF_BLOCK: usize = QK5_0 / 2;

#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK5_0
}

fn f16_at(block: &[u8]) -> half::f16 {
    half::f16::from_le_bytes([block[D_OFFSET], block[D_OFFSET + 1]])
}

fn qh_at(block: &[u8]) -> u32 {
    u32::from_le_bytes([
        block[QH_OFFSET],
        block[QH_OFFSET + 1],
        block[QH_OFFSET + 2],
        block[QH_OFFSET + 3],
    ])
}

/// Decode one block according to `dequantize_row_q5_0` in llama.cpp:
/// `d * (q - 16)`, where the low/high nibbles supply bits 0..3 and `qh`
/// supplies bit 4 for elements 0..31.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let scale = f16_at(block).to_f32();
    let qh = qh_at(block);
    for index in 0..HALF_BLOCK {
        let byte = block[QS_OFFSET + index];
        let low = u32::from(byte & 0x0f) | (((qh >> index) & 1) << 4);
        let high = u32::from(byte >> 4) | (((qh >> (index + HALF_BLOCK)) & 1) << 4);
        output[index] = (low as i32 - 16) as f32 * scale;
        output[HALF_BLOCK + index] = (high as i32 - 16) as f32 * scale;
    }
}

/// Dequantize a complete Q5_0 byte range into caller-owned f32 storage.
pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    let block_count = blocks_for_bytes(data.len()).ok_or(QuantError::InputNotBlockMultiple {
        codec: "q5_0",
        found: data.len(),
        block_bytes: BLOCK_BYTES,
    })?;
    let expected = elements_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, output) in data
        .chunks_exact(BLOCK_BYTES)
        .zip(output.chunks_exact_mut(QK5_0))
    {
        dequantize_block(block, output);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_low_and_high_five_bit_values() {
        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..6].copy_from_slice(&0x0003_0002u32.to_le_bytes());
        block[6] = 0xf1;
        let mut output = [0.0; QK5_0];
        dequantize(&block, &mut output).expect("one complete q5_0 block");
        assert_eq!(output[0], -15.0);
        assert_eq!(output[16], 15.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[17], 0.0);
    }

    #[test]
    fn rejects_partial_block() {
        let error = dequantize(&[0; BLOCK_BYTES - 1], &mut []).expect_err("partial block");
        assert!(matches!(error, QuantError::InputNotBlockMultiple { .. }));
    }
}
