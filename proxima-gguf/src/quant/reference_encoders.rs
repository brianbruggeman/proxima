//! The hand-written block encoders that `q4_0`, `q5_0`, `q8_0`, `q1_0`, `q2_0` and `pq2_0` carried
//! before their `quantize` became [`proxima_numeric::BlockLayout::encode_rule`] run by
//! [`proxima_numeric::FormatCodec`], frozen byte for byte as the oracle the rule is checked against
//! (principle 14: the incumbent wins). Each is llama.cpp's or PrismML's `quantize_row_*_ref`, one
//! loop per format. Whole blocks only: the callers in the parity tests size the buffers.

use proxima_numeric::float_layout::FloatLayout;

use crate::quant::{QuantError, float_layout};

pub(crate) fn q4_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    const HALF: usize = 16;
    for (x, block) in input.as_chunks::<32>().0.iter().zip(output.as_chunks_mut::<18>().0) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &value in x {
            if amax < value.abs() {
                amax = value.abs();
                max = value;
            }
        }
        let d = max / -8.0;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for index in 0..HALF {
            let level0 = 15i8.min((x[index] * inv_d + 8.5) as i8);
            let level1 = 15i8.min((x[HALF + index] * inv_d + 8.5) as i8);
            block[2 + index] = (level0 as u8) | ((level1 as u8) << 4);
        }
    }
    Ok(())
}

pub(crate) fn q5_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    const HALF: usize = 16;
    for (x, block) in input.as_chunks::<32>().0.iter().zip(output.as_chunks_mut::<22>().0) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &value in x {
            if amax < value.abs() {
                amax = value.abs();
                max = value;
            }
        }
        let scale = max / -16.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        block[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        let mut high_bits = 0u32;
        for index in 0..HALF {
            let low = 31i8.min((x[index] * inverse + 16.5) as i8) as u8;
            let high = 31i8.min((x[HALF + index] * inverse + 16.5) as i8) as u8;
            block[6 + index] = (low & 0x0f) | ((high & 0x0f) << 4);
            high_bits |= u32::from((low & 0x10) >> 4) << index;
            high_bits |= u32::from((high & 0x10) >> 4) << (index + HALF);
        }
        block[2..6].copy_from_slice(&high_bits.to_le_bytes());
    }
    Ok(())
}

pub(crate) fn q8_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    for (x, block) in input.as_chunks::<32>().0.iter().zip(output.as_chunks_mut::<34>().0) {
        let mut amax = 0.0f32;
        for &value in x {
            amax = amax.max(value.abs());
        }
        let d = amax / 127.0;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for (out, &value) in block[2..].iter_mut().zip(x) {
            *out = float_layout::encode(libm::roundf(value * inv_d), FloatLayout::INT8) as u8;
        }
    }
    Ok(())
}

pub(crate) fn q1_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    for (x, block) in input.as_chunks::<128>().0.iter().zip(output.as_chunks_mut::<18>().0) {
        let scale = x.iter().map(|value| value.abs()).sum::<f32>() / 128.0;
        block[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        block[2..].fill(0);
        for (index, value) in x.iter().enumerate() {
            if *value >= 0.0 {
                block[2 + index / 8] |= 1 << (index % 8);
            }
        }
    }
    Ok(())
}

fn two_bit_groups<const GROUP: usize, const BYTES: usize>(input: &[f32], output: &mut [u8]) {
    for (x, block) in input.as_chunks::<GROUP>().0.iter().zip(output.as_chunks_mut::<BYTES>().0) {
        let scale = x.iter().fold(0.0f32, |peak, value| peak.max(value.abs()));
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        block[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        block[2..].fill(0);
        for (index, value) in x.iter().enumerate() {
            let code = (libm::roundf(value * inverse) as i32).wrapping_add(1).clamp(0, 3) as u8;
            block[2 + index / 4] |= code << (2 * (index % 4));
        }
    }
}

pub(crate) fn q2_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    two_bit_groups::<64, 18>(input, output);
    Ok(())
}

pub(crate) fn pq2_0(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    two_bit_groups::<128, 34>(input, output);
    Ok(())
}
