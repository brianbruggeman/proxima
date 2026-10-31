//! Scalar PrismML `PTQ1_0` codec (128 values / 28 bytes): `dequantize_row_ptq1_0` and
//! `quantize_row_ptq1_0_ref` in PrismML-Eng/llama.cpp (`ggml-quants.c`, branch `prism`).
//!
//! Block: `qs[24]` holds 120 trits at 5 per byte, `qh[2]` holds 8 trits at 4 per byte, then the
//! f16 scale. The stages walk `qs` in runs of 16 then 8 bytes; within a run, trit `n` of byte
//! `m` is element `n * run + m`.

use crate::quant::QuantError;
use crate::types::GgmlType;

const CODEC: &str = "ptq1_0";
pub const QK_PTQ1_0: usize = 128;
pub const BLOCK_BYTES: usize = 28;
const QS_BYTES: usize = 24;
const QH_BYTES: usize = 2;
const SCALE_OFFSET: usize = QS_BYTES + QH_BYTES;
const STAGES: [usize; 3] = [32, 16, 8];
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

fn trit(byte: u8, position: usize) -> f32 {
    let scaled = byte.wrapping_mul(POW3[position]);
    ((u16::from(scaled) * 3) >> 8) as f32 - 1.0
}

fn check_decode_shapes(data: &[u8], output: &[f32]) -> Result<(), QuantError> {
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes: BLOCK_BYTES,
        });
    }
    let expected = data.len() / BLOCK_BYTES * QK_PTQ1_0;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    Ok(())
}

pub fn dequantize_block(block: &[u8; BLOCK_BYTES], output: &mut [f32; QK_PTQ1_0]) {
    let scale = half::f16::from_le_bytes([block[SCALE_OFFSET], block[SCALE_OFFSET + 1]]).to_f32();
    let mut written = 0;
    let mut consumed = 0;
    for run in STAGES {
        while consumed + run <= QS_BYTES {
            for position in 0..5 {
                for lane in 0..run {
                    output[written] = trit(block[consumed + lane], position) * scale;
                    written += 1;
                }
            }
            consumed += run;
        }
    }
    for position in 0..4 {
        for lane in 0..QH_BYTES {
            output[written] = trit(block[QS_BYTES + lane], position) * scale;
            written += 1;
        }
    }
}

pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    check_decode_shapes(data, output)?;
    for (block, out) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<QK_PTQ1_0>().0.iter_mut())
    {
        dequantize_block(block, out);
    }
    Ok(())
}

const _: () = {
    assert!(GgmlType::Ptq1_0.block_layout().block_bytes as usize == BLOCK_BYTES);
};

fn ternary(value: f32, inverse: f32) -> u32 {
    (libm::roundf(value * inverse) as i32 + 1) as u32
}

fn pack_byte(trits: u32) -> u8 {
    ((trits * 256 + 242) / 243) as u8
}

fn quantize_block(input: &[f32; QK_PTQ1_0], block: &mut [u8; BLOCK_BYTES]) {
    let scale = input.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    block[SCALE_OFFSET..].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    let mut read = 0;
    let mut written = 0;
    for run in STAGES {
        while written + run <= QS_BYTES {
            for lane in 0..run {
                let trits = (0..5).fold(0, |acc, position| {
                    acc * 3 + ternary(input[read + lane + position * run], inverse)
                });
                block[written + lane] = pack_byte(trits);
            }
            read += 5 * run;
            written += run;
        }
    }
    for lane in 0..QH_BYTES {
        let trits = (0..4).fold(0, |acc, position| {
            acc * 3 + ternary(input[read + lane + position * QH_BYTES], inverse)
        });
        block[QS_BYTES + lane] = pack_byte(trits * 3);
    }
}

/// Quantizes whole 128-element runs of `f32` into `PTQ1_0` blocks, following PrismML's
/// `quantize_row_ptq1_0_ref`: scale is the absolute maximum, each weight `round(w / scale)`
/// in `-1..=1` packed five (or four) trits to the byte.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(QK_PTQ1_0) {
        return Err(QuantError::InputNotElementMultiple {
            codec: CODEC,
            unit: "block",
            found: input.len(),
            block_elements: QK_PTQ1_0,
        });
    }
    let expected = input.len() / QK_PTQ1_0 * BLOCK_BYTES;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, block) in input
        .as_chunks::<QK_PTQ1_0>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        quantize_block(chunk, block);
    }
    Ok(())
}
