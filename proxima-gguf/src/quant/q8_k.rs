//! `Q8_K`: 256 signed int8 values with one `f32` scale and auxiliary sums.

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "q8_k";
pub const QK_Q8_K: usize = 256;
pub const BLOCK_BYTES: usize = 292;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_Q8_K;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out) in data
        .chunks_exact(BLOCK_BYTES)
        .zip(output.chunks_exact_mut(QK_Q8_K))
    {
        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for (i, value) in out.iter_mut().enumerate() {
            *value = i8::from_le_bytes([block[4 + i]]) as f32 * scale;
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Q8_K.block_layout().block_bytes as usize == BLOCK_BYTES);
};
