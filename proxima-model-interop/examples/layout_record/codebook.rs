use proxima_gguf::quant::iq4_nl::KVALUES_IQ4NL;
use proxima_gguf::quant::tables::iq2xs_grid::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use proxima_gguf::quant::tables::iq3xxs_grid::IQ3XXS_GRID;

use crate::error::RecordError;
use crate::linear::{absolute_max, put_f16};

const SUB_BLOCKS: usize = 8;
const SCALE_LEVELS: usize = 16;

fn rounded_scale(block: &mut [u8], value: f32) -> f32 {
    put_f16(block, 0, value);
    half::f16::from_le_bytes([block[0], block[1]]).to_f32()
}

fn lane_signs(sign_index: usize) -> [f32; 8] {
    std::array::from_fn(|lane| {
        if KSIGNS_IQ2XS[sign_index] & KMASK_IQ2XS[lane] != 0 {
            -1.0
        } else {
            1.0
        }
    })
}

fn sign_mismatch(values: &[f32], sign_index: usize) -> f32 {
    let signs = lane_signs(sign_index);
    values
        .iter()
        .zip(signs)
        .filter(|(value, sign)| (**value < 0.0) != (*sign < 0.0))
        .map(|(value, _)| value.abs())
        .sum()
}

fn best_sign_index(values: &[f32]) -> usize {
    (0..KSIGNS_IQ2XS.len())
        .min_by(|left, right| {
            sign_mismatch(values, *left).total_cmp(&sign_mismatch(values, *right))
        })
        .unwrap_or(0)
}

fn squared_error(values: &[f32], signs: &[f32; 8], scale: f32, magnitudes: &[u8]) -> f32 {
    values
        .iter()
        .zip(signs)
        .zip(magnitudes)
        .map(|((value, sign), magnitude)| (scale * f32::from(*magnitude) * sign - value).powi(2))
        .sum()
}

fn nearest_entry(
    values: &[f32],
    signs: &[f32; 8],
    scale: f32,
    entries: usize,
    magnitudes: impl Fn(usize) -> [u8; 8],
) -> (usize, f32) {
    (0..entries)
        .map(|entry| {
            (
                entry,
                squared_error(values, signs, scale, &magnitudes(entry)),
            )
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, f32::MAX))
}

fn iq2_magnitudes(entry: usize) -> [u8; 8] {
    IQ2XS_GRID[entry].to_le_bytes()
}

fn iq3_magnitudes(entry: usize) -> [u8; 8] {
    let mut lanes = [0_u8; 8];
    lanes[..4].copy_from_slice(&IQ3XXS_GRID[entry].to_le_bytes());
    lanes
}

fn grid_peak(magnitudes: impl Iterator<Item = u8>) -> f32 {
    f32::from(magnitudes.max().unwrap_or(1))
}

pub fn iq4_nl_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let scale = absolute_max(values) / 127.0;
    put_f16(block, 0, scale);
    let nearest = |value: f32| -> u8 {
        let target = if scale == 0.0 { 0.0 } else { value / scale };
        (0..16_u8)
            .min_by(|left, right| {
                (f32::from(KVALUES_IQ4NL[usize::from(*left)]) - target)
                    .abs()
                    .total_cmp(&(f32::from(KVALUES_IQ4NL[usize::from(*right)]) - target).abs())
            })
            .unwrap_or(0)
    };
    (0..16).for_each(|index| {
        block[2 + index] = nearest(values[index]) | (nearest(values[16 + index]) << 4)
    });
    Ok(())
}

fn iq2_pair(pair: &[f32], delta: f32) -> (u8, [u16; 2], f32) {
    (0..SCALE_LEVELS as u8)
        .map(|nibble| {
            let scale = delta * (0.5 + f32::from(nibble)) * 0.25;
            let halves: Vec<(u16, f32)> = pair
                .as_chunks::<8>()
                .0
                .iter()
                .map(|group| iq2_entry(group, scale))
                .collect();
            (
                nibble,
                [halves[0].0, halves[1].0],
                halves[0].1 + halves[1].1,
            )
        })
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .unwrap_or((0, [0, 0], f32::MAX))
}

fn iq2_entry(group: &[f32], scale: f32) -> (u16, f32) {
    let sign_index = best_sign_index(group);
    let (grid, error) = nearest_entry(group, &lane_signs(sign_index), scale, 512, iq2_magnitudes);
    ((grid | (sign_index << 9)) as u16, error)
}

pub fn iq2_xs_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let peak = grid_peak(IQ2XS_GRID.iter().flat_map(|word| word.to_le_bytes()));
    let delta = rounded_scale(block, absolute_max(values) / (peak * (0.5 + 15.0) * 0.25));
    for sub_block in 0..SUB_BLOCKS {
        let sub = &values[sub_block * 32..sub_block * 32 + 32];
        let (low, low_entries, _) = iq2_pair(&sub[..16], delta);
        let (high, high_entries, _) = iq2_pair(&sub[16..], delta);
        block[66 + sub_block] = low | (high << 4);
        [
            low_entries[0],
            low_entries[1],
            high_entries[0],
            high_entries[1],
        ]
        .iter()
        .enumerate()
        .for_each(|(group, entry)| {
            let at = 2 + (sub_block * 4 + group) * 2;
            block[at..at + 2].copy_from_slice(&entry.to_le_bytes());
        });
    }
    Ok(())
}

fn iq3_sub_block(sub: &[f32], delta: f32, level: u8) -> (u32, [u8; 8], f32) {
    let scale = delta * (0.5 + f32::from(level)) * 0.5;
    let mut packed = u32::from(level) << 28;
    let mut indexes = [0_u8; 8];
    let mut error = 0.0;
    for (group, values) in sub.as_chunks::<8>().0.iter().enumerate() {
        let sign_index = best_sign_index(values);
        let signs = lane_signs(sign_index);
        packed |= (sign_index as u32) << (7 * group);
        for half_index in 0..2 {
            let part = &values[half_index * 4..half_index * 4 + 4];
            let (entry, part_error) = nearest_entry(
                part,
                &[
                    signs[half_index * 4],
                    signs[half_index * 4 + 1],
                    signs[half_index * 4 + 2],
                    signs[half_index * 4 + 3],
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                scale,
                256,
                iq3_magnitudes,
            );
            indexes[group * 2 + half_index] = entry as u8;
            error += part_error;
        }
    }
    (packed, indexes, error)
}

pub fn iq3_xxs_block(values: &[f32], block: &mut [u8]) -> Result<(), RecordError> {
    let peak = grid_peak(IQ3XXS_GRID.iter().flat_map(|word| word.to_le_bytes()));
    let delta = rounded_scale(block, absolute_max(values) / (peak * (0.5 + 15.0) * 0.5));
    for sub_block in 0..SUB_BLOCKS {
        let sub = &values[sub_block * 32..sub_block * 32 + 32];
        let (packed, indexes, _) = (0..SCALE_LEVELS as u8)
            .map(|level| iq3_sub_block(sub, delta, level))
            .min_by(|left, right| left.2.total_cmp(&right.2))
            .unwrap_or((0, [0; 8], f32::MAX));
        block[2 + sub_block * 8..2 + sub_block * 8 + 8].copy_from_slice(&indexes);
        block[66 + sub_block * 4..66 + sub_block * 4 + 4].copy_from_slice(&packed.to_le_bytes());
    }
    Ok(())
}
