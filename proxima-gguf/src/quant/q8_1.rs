//! `Q8_1`: 32 signed int8 values with an `f16` scale and stored sum.
//!
//! `s` is an auxiliary quantization sum, not another per-element scale. The
//! logical values are therefore `q * d`; the stored sum is intentionally not
//! exposed by this value decoder.

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "q8_1";
pub const QK_Q8_1: usize = 32;
pub const BLOCK_BYTES: usize = 36;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_Q8_1;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out) in data
        .chunks_exact(BLOCK_BYTES)
        .zip(output.chunks_exact_mut(QK_Q8_1))
    {
        let mut d = [0; 2];
        d.copy_from_slice(&block[..2]);
        let scale = half::f16::from_le_bytes(d).to_f32();
        for (i, value) in out.iter_mut().enumerate() {
            *value = i8::from_le_bytes([block[4 + i]]) as f32 * scale;
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Q8_1.block_layout().block_bytes as usize == BLOCK_BYTES);
};
