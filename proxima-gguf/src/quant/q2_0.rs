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
        .chunks_exact(BLOCK_BYTES)
        .zip(output.chunks_exact_mut(QK_Q2_0))
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
