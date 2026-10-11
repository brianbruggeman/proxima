use crate::error::RecordError;

pub fn put_f16(block: &mut [u8], at: usize, value: f32) {
    block[at..at + 2].copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
}

pub fn absolute_max(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()))
}

fn signed_absolute_max(values: &[f32]) -> f32 {
    values.iter().fold(0.0_f32, |peak, value| {
        if value.abs() > peak.abs() {
            *value
        } else {
            peak
        }
    })
}

fn extremes(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((f32::MAX, f32::MIN), |(low, high), value| {
            (low.min(*value), high.max(*value))
        })
}

fn inverse(scale: f32) -> f32 {
    if scale == 0.0 { 0.0 } else { 1.0 / scale }
}

fn level(value: f32, offset: f32, inverse_scale: f32, top: f32) -> u8 {
    (value * inverse_scale + offset).clamp(0.0, top) as u8
}

fn nibble_levels(values: &[f32], offset: f32, inverse_scale: f32, top: f32) -> Vec<u8> {
    values
        .iter()
        .map(|value| level(*value, offset, inverse_scale, top))
        .collect()
}

fn pack_nibbles(levels: &[u8], block: &mut [u8]) {
    (0..16).for_each(|index| {
        block[index] = (levels[index] & 0x0F) | ((levels[16 + index] & 0x0F) << 4)
    });
}

fn high_bits(levels: &[u8]) -> u32 {
    (0..16).fold(0_u32, |bits, index| {
        bits | (u32::from(levels[index] >> 4) << index)
            | (u32::from(levels[16 + index] >> 4) << (index + 16))
    })
}

pub fn q4_1_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let (low, high) = extremes(values);
    let scale = (high - low) / 15.0;
    put_f16(block, 0, scale);
    put_f16(block, 2, low);
    let levels = nibble_levels(values, -low * inverse(scale) + 0.5, inverse(scale), 15.0);
    pack_nibbles(&levels, &mut block[4..20]);
    Ok(())
}

pub fn q5_0_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let scale = signed_absolute_max(values) / -16.0;
    put_f16(block, 0, scale);
    let levels = nibble_levels(values, 16.5, inverse(scale), 31.0);
    block[2..6].copy_from_slice(&high_bits(&levels).to_le_bytes());
    pack_nibbles(&levels, &mut block[6..22]);
    Ok(())
}

pub fn q5_1_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let (low, high) = extremes(values);
    let scale = (high - low) / 31.0;
    put_f16(block, 0, scale);
    put_f16(block, 2, low);
    let levels = nibble_levels(values, -low * inverse(scale) + 0.5, inverse(scale), 31.0);
    block[4..8].copy_from_slice(&high_bits(&levels).to_le_bytes());
    pack_nibbles(&levels, &mut block[8..24]);
    Ok(())
}

pub fn q8_1_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let scale = absolute_max(values) / 127.0;
    let levels: Vec<i8> = values
        .iter()
        .map(|value| (value * inverse(scale)).round() as i8)
        .collect();
    put_f16(block, 0, scale);
    put_f16(
        block,
        2,
        levels.iter().map(|level| f32::from(*level)).sum::<f32>() * scale,
    );
    levels
        .iter()
        .enumerate()
        .for_each(|(index, level)| block[4 + index] = level.cast_unsigned());
    Ok(())
}

pub fn q1_0_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    put_f16(
        block,
        0,
        values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32,
    );
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= 0.0)
        .for_each(|(index, _)| block[2 + index / 8] |= 1 << (index % 8));
    Ok(())
}

pub fn q2_0_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let scale = absolute_max(values) / 2.0;
    put_f16(block, 0, scale);
    values.iter().enumerate().for_each(|(index, value)| {
        let code = ((value * inverse(scale)).round() + 1.0).clamp(0.0, 3.0) as u8;
        block[2 + index / 4] |= code << (2 * (index % 4));
    });
    Ok(())
}
