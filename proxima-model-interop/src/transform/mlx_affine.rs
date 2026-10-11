//! mlx-lm affine quantization: pack a float matrix into the `weight`,
//! `scales` and `biases` tensors `mlx_lm` writes for `mode="affine"`.
//!
//! Sans-IO and bytes-first: [`pack_affine`] fills caller-owned `&mut [u32]` and
//! `&mut [f32]` buffers and allocates nothing; [`export_affine`] is the owned
//! convenience that sizes those buffers and casts scales and biases to the
//! export float dtype. Element `k` of a row sits at bit offset `k * bits` of a
//! little-endian bit stream, so 3, 5 and 6 bit elements straddle `u32` words.
//! Scale and bias follow mlx `quantize` (`quantized.cpp:1165-1233` at mlx
//! 06eb7483): the group edge of larger magnitude anchors the bias, the scale
//! is negative when that edge is the maximum, and dequantization is
//! `w = scale * q + bias`.

use alloc::vec::Vec;

use half::{bf16, f16};
use proxima_tensor::DType;

use crate::error::InteropError;

const SCALE_FLOOR: f32 = 1e-7;
const SUPPORTED_BITS: [u8; 6] = [2, 3, 4, 5, 6, 8];
const SUPPORTED_GROUP_SIZES: [usize; 3] = [32, 64, 128];

/// Bit width and group size of an affine export; mlx-lm's default is 4 bits
/// at group size 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineParams {
    bits: u8,
    group_size: usize,
}

impl AffineParams {
    /// # Errors
    ///
    /// [`InteropError::AffineBitsUnsupported`] or
    /// [`InteropError::AffineGroupSizeUnsupported`] outside mlx-lm's grid.
    pub fn new(bits: u8, group_size: usize) -> Result<Self, InteropError> {
        if !SUPPORTED_BITS.contains(&bits) {
            return Err(InteropError::AffineBitsUnsupported { bits });
        }
        if !SUPPORTED_GROUP_SIZES.contains(&group_size) {
            return Err(InteropError::AffineGroupSizeUnsupported { group_size });
        }
        Ok(Self { bits, group_size })
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    #[must_use]
    pub const fn group_size(self) -> usize {
        self.group_size
    }
}

impl Default for AffineParams {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 64,
        }
    }
}

/// A packed tensor triple in mlx-lm's layout; `scales` and `biases` hold
/// little-endian `dtype` bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct AffinePacked {
    pub weight: Vec<u32>,
    pub scales: Vec<u8>,
    pub biases: Vec<u8>,
    pub dtype: DType,
    pub weight_shape: [usize; 2],
    pub scales_shape: [usize; 2],
}

/// Result of [`export_affine`]: a tensor whose last dimension is not a
/// multiple of the group size stays float, as in mlx-lm.
#[derive(Debug, Clone, PartialEq)]
pub enum AffineExport {
    Packed(AffinePacked),
    Float,
}

fn shape_error(rows: usize, columns: usize, reason: &'static str) -> InteropError {
    InteropError::AffineShapeMismatch {
        rows,
        columns,
        reason,
    }
}

const fn is_power_of_two(bits: u8) -> bool {
    bits & (bits - 1) == 0
}

const fn get_pack_factor(bits: u8, word_bits: u8) -> usize {
    if bits == 3 || bits == 5 {
        8
    } else if bits == 6 {
        4
    } else {
        (word_bits / bits) as usize
    }
}

const fn get_bytes_per_pack(bits: u8, word_bits: u8) -> usize {
    if is_power_of_two(bits) {
        (word_bits / 8) as usize
    } else if bits == 5 {
        5
    } else {
        3
    }
}

const fn written_bytes_per_pack(bits: u8) -> usize {
    if is_power_of_two(bits) {
        4
    } else {
        get_bytes_per_pack(bits, 8)
    }
}

fn rint(value: f32) -> f32 {
    let magnitude = value.abs();
    let floor = magnitude as u64;
    let fraction = magnitude - floor as f32;
    let round_up = fraction > 0.5 || (fraction == 0.5 && floor % 2 == 1);
    let rounded = (floor + u64::from(round_up)) as f32;
    if value < 0.0 { -rounded } else { rounded }
}

fn group_scale_bias(chunk: &[f32], top_level: f32) -> (f32, f32) {
    let low = chunk.iter().copied().fold(f32::INFINITY, f32::min);
    let high = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mask = low.abs() > high.abs();
    let floored = ((high - low) / top_level).max(SCALE_FLOOR);
    let scale = if mask { floored } else { -floored };
    let edge = if mask { low } else { high };
    let edge_level = rint(edge / scale);
    if edge_level == 0.0 {
        (scale, 0.0)
    } else {
        (edge / edge_level, edge)
    }
}

fn write_pack(weight: &mut [u32], byte_offset: usize, byte_count: usize, out_el: u64) {
    for index in 0..byte_count {
        let byte = ((out_el >> (8 * index)) & 0xff) as u32;
        let position = byte_offset + index;
        weight[position / 4] |= byte << (8 * (position % 4));
    }
}

fn pack_group(chunk: &[f32], bits: u8, first_pack: usize, weight: &mut [u32]) -> (f32, f32) {
    let top_level = ((1u32 << bits) - 1) as f32;
    let (scale, bias) = group_scale_bias(chunk, top_level);
    let packs = chunk.chunks_exact(get_pack_factor(bits, 32));
    for (offset, elements) in packs.enumerate() {
        let out_el = elements
            .iter()
            .enumerate()
            .fold(0u64, |packed, (k, value)| {
                let level = rint((value - bias) / scale).clamp(0.0, top_level) as u64;
                packed | (level << (k * usize::from(bits)))
            });
        let byte_count = written_bytes_per_pack(bits);
        write_pack(
            weight,
            (first_pack + offset) * byte_count,
            byte_count,
            out_el,
        );
    }
    (scale, bias)
}

fn check_buffers(
    values: &[f32],
    rows: usize,
    params: AffineParams,
    sizes: [usize; 3],
) -> Result<usize, InteropError> {
    if rows == 0 || !values.len().is_multiple_of(rows) {
        return Err(shape_error(
            rows,
            values.len(),
            "values do not divide into rows",
        ));
    }
    let columns = values.len() / rows;
    let groups = values.len() / params.group_size;
    let words = values.len() * usize::from(params.bits) / 32;
    if !columns.is_multiple_of(params.group_size) {
        return Err(shape_error(
            rows,
            columns,
            "columns are not a multiple of the group size",
        ));
    }
    if sizes != [words, groups, groups] {
        return Err(shape_error(
            rows,
            columns,
            "weight, scales or biases length is wrong",
        ));
    }
    Ok(columns)
}

/// Pack `values` (`rows` x columns, row-major) into caller buffers.
///
/// `weight` holds `rows * columns * bits / 32` words, `scales` and `biases`
/// hold `rows * columns / group_size` entries each. Teaching pointer:
/// [`export_affine`] sizes these buffers and casts to the export dtype.
///
/// # Errors
///
/// [`InteropError::AffineShapeMismatch`] when columns are not a multiple of
/// the group size or a buffer length disagrees with the shape.
pub fn pack_affine(
    values: &[f32],
    rows: usize,
    params: AffineParams,
    weight: &mut [u32],
    scales: &mut [f32],
    biases: &mut [f32],
) -> Result<(), InteropError> {
    check_buffers(
        values,
        rows,
        params,
        [weight.len(), scales.len(), biases.len()],
    )?;
    weight.fill(0);
    let packs_per_group = params.group_size / get_pack_factor(params.bits, 32);
    let groups = values
        .chunks_exact(params.group_size)
        .zip(scales.iter_mut().zip(biases.iter_mut()));
    for (group_index, (chunk, (scale, bias))) in groups.enumerate() {
        let first_pack = group_index * packs_per_group;
        (*scale, *bias) = pack_group(chunk, params.bits, first_pack, weight);
    }
    Ok(())
}

fn float_bytes(values: &[f32], dtype: DType) -> Result<Vec<u8>, InteropError> {
    match dtype {
        DType::BFloat16 => Ok(values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_le_bytes())
            .collect()),
        DType::Float16 => Ok(values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect()),
        DType::Float32 => Ok(values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()),
        other => Err(InteropError::AffineDTypeUnsupported { dtype: other }),
    }
}

/// Quantize a `[rows, columns]` float tensor the way mlx-lm does, with
/// `scales` and `biases` cast to `dtype` (bf16, f16 or f32).
///
/// Composes [`pack_affine`]; use that directly when the caller owns the
/// buffers. A `columns` that is not a multiple of the group size yields
/// [`AffineExport::Float`] and the caller writes the tensor as plain float.
///
/// # Errors
///
/// [`InteropError::AffineDTypeUnsupported`] for a non-float `dtype`, and the
/// shape errors of [`pack_affine`].
pub fn export_affine(
    values: &[f32],
    shape: [usize; 2],
    params: AffineParams,
    dtype: DType,
) -> Result<AffineExport, InteropError> {
    float_bytes(&[], dtype)?;
    let [rows, columns] = shape;
    if rows.checked_mul(columns) != Some(values.len()) {
        return Err(shape_error(
            rows,
            columns,
            "values length is not rows times columns",
        ));
    }
    if !columns.is_multiple_of(params.group_size) {
        return Ok(AffineExport::Float);
    }
    let groups = values.len() / params.group_size;
    let mut weight = alloc::vec![0u32; values.len() * usize::from(params.bits) / 32];
    let mut scales = alloc::vec![0f32; groups];
    let mut biases = alloc::vec![0f32; groups];
    pack_affine(values, rows, params, &mut weight, &mut scales, &mut biases)?;
    Ok(AffineExport::Packed(AffinePacked {
        weight,
        scales: float_bytes(&scales, dtype)?,
        biases: float_bytes(&biases, dtype)?,
        dtype,
        weight_shape: [rows, columns * usize::from(params.bits) / 32],
        scales_shape: [rows, columns / params.group_size],
    }))
}

#[cfg(test)]
mod tests;
