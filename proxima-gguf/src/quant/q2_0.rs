//! Scalar legacy `Q2_0` decoder (64 values / 18 bytes).

use proxima_numeric::BlockLayout;

use crate::quant::QuantError;
use crate::quant::block_layout_encode::quantize_by_rule;
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
    assert!(GgmlType::Q2_0.wire_shape().block_bytes as usize == BLOCK_BYTES);
};

/// Quantizes whole 64-element runs of `f32` into `Q2_0` blocks by `BlockLayout::Q2_0`'s encode rule
/// (PrismML's `quantize_row_q2_0_ref`): scale is the absolute maximum over the code reach, code is
/// `round(w / scale) + 1` clamped to `0..=3`. The loop this replaced is frozen in
/// `reference_encoders` as the byte oracle.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    quantize_by_rule(&BlockLayout::Q2_0, CODEC, input, output)
}
