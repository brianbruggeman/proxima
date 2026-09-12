//! Affine scalar `Q4_1` decoder (32 values / 20 bytes).

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "q4_1";
pub const QK_Q4_1: usize = 32;
pub const BLOCK_BYTES: usize = 20;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_Q4_1;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out) in data
        .chunks_exact(BLOCK_BYTES)
        .zip(output.chunks_exact_mut(QK_Q4_1))
    {
        let mut d = [0; 2];
        let mut m = [0; 2];
        d.copy_from_slice(&block[..2]);
        m.copy_from_slice(&block[2..4]);
        let scale = half::f16::from_le_bytes(d).to_f32();
        let min = half::f16::from_le_bytes(m).to_f32();
        for i in 0..16 {
            let byte = block[4 + i];
            out[i] = f32::from(byte & 0x0f) * scale + min;
            out[16 + i] = f32::from(byte >> 4) * scale + min;
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Q4_1.block_layout().block_bytes as usize == BLOCK_BYTES);
};
