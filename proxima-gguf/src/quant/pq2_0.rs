//! Scalar PrismML `PQ2_0` decoder (128 values / 34 bytes): `dequantize_row_pq2_0`
//! in PrismML-Eng/llama.cpp (`ggml-quants.c`, branch `prism`).

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "pq2_0";
pub const QK_PQ2_0: usize = 128;
pub const BLOCK_BYTES: usize = 34;

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_PQ2_0;
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
        .zip(output.as_chunks_mut::<QK_PQ2_0>().0.iter_mut())
    {
        let mut scale_bytes = [0; 2];
        scale_bytes.copy_from_slice(&block[..2]);
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
        for (index, value) in out.iter_mut().enumerate() {
            let code = (block[2 + index / 4] >> (2 * (index % 4))) & 3;
            *value = (i32::from(code) - 1) as f32 * scale;
        }
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Pq2_0.block_layout().block_bytes as usize == BLOCK_BYTES);
};
