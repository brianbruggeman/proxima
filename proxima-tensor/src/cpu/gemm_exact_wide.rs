use super::*;

use super::dequant_x86::{
    dequantize_block_q4_k, dequantize_block_q5_0, dequantize_block_q6_k, dequantize_block_q8_0,
};
use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};

const TILE_ROWS: usize = 2;
const TILE_POSITIONS: usize = 4;
const LANES: usize = 8;
const UNIT_BYTES_BUDGET: usize = 96 * 1024;
const POSITION_BYTES_BUDGET: usize = 64 * 1024;

pub(super) struct ExactOperands<'a> {
    pub(super) activation: &'a [f32],
    pub(super) k: usize,
    pub(super) positions: usize,
    pub(super) block_elements: usize,
}

// One decoded weight block is shared by TILE_POSITIONS positions; each output keeps
// `dot_fold_fused_multiply_add_fma`'s sequence exactly: 8-lane fma over the block,
// then `acc = acc + lane0 + .. + lane7` in lane order, block after block.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile<const ROWS: usize, const TILE: usize>(
    operands: &ExactOperands<'_>,
    decoded: &[f32],
    first_row: usize,
    position: usize,
    out: &mut [f32],
) {
    let k = operands.k;
    let mut totals = [[0.0f32; ROWS]; TILE];
    for block in 0..k / operands.block_elements {
        let mut lanes = [[_mm256_setzero_ps(); ROWS]; TILE];
        for step in 0..operands.block_elements / LANES {
            let offset = block * operands.block_elements + step * LANES;
            // SAFETY: avx2 and fma are enabled; `offset + 8 <= k` for every row of `decoded`
            // and every position of `activation` (shapes checked by the caller).
            unsafe {
                let mut weights = [_mm256_setzero_ps(); ROWS];
                for (row, weight) in weights.iter_mut().enumerate() {
                    *weight = _mm256_loadu_ps(decoded.as_ptr().add((first_row + row) * k + offset));
                }
                for (lane, row_lanes) in lanes.iter_mut().enumerate() {
                    let activation = _mm256_loadu_ps(operands.activation.as_ptr().add((position + lane) * k + offset));
                    for (accumulator, weight) in row_lanes.iter_mut().zip(weights) {
                        *accumulator = _mm256_fmadd_ps(weight, activation, *accumulator);
                    }
                }
            }
        }
        for (lane_totals, row_lanes) in totals.iter_mut().zip(lanes) {
            for (total, accumulator) in lane_totals.iter_mut().zip(row_lanes) {
                let mut values = [0.0f32; LANES];
                // SAFETY: `values` is 8 f32; storeu has no alignment requirement.
                unsafe { _mm256_storeu_ps(values.as_mut_ptr(), accumulator) };
                for value in values {
                    *total += value;
                }
            }
        }
    }
    for (lane, lane_totals) in totals.into_iter().enumerate() {
        for (row, total) in lane_totals.into_iter().enumerate() {
            out[(first_row + row) * operands.positions + position + lane] = total;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn group<const ROWS: usize>(
    operands: &ExactOperands<'_>,
    decoded: &[f32],
    first_row: usize,
    positions: core::ops::Range<usize>,
    out: &mut [f32],
) {
    let mut position = positions.start;
    while position + TILE_POSITIONS <= positions.end {
        // SAFETY: avx2 and fma are enabled on this function.
        unsafe { tile::<ROWS, TILE_POSITIONS>(operands, decoded, first_row, position, out) };
        position += TILE_POSITIONS;
    }
    while position < positions.end {
        // SAFETY: avx2 and fma are enabled on this function.
        unsafe { tile::<ROWS, 1>(operands, decoded, first_row, position, out) };
        position += 1;
    }
}

fn unit_row_budget(k: usize) -> usize {
    (UNIT_BYTES_BUDGET / (4 * k)).clamp(TILE_ROWS, 16) / TILE_ROWS * TILE_ROWS
}

fn position_block(k: usize) -> usize {
    (POSITION_BYTES_BUDGET / (4 * k)).clamp(TILE_POSITIONS, 64) / TILE_POSITIONS * TILE_POSITIONS
}

#[target_feature(enable = "avx2,fma")]
unsafe fn unit(
    operands: &ExactOperands<'_>,
    weight_rows: &[u8],
    row_bytes: usize,
    block_bytes: usize,
    decode: fn(&[u8], &mut [f32]),
    out: &mut [f32],
) {
    let rows = weight_rows.len() / row_bytes;
    let k = operands.k;
    let mut decoded = vec![0.0f32; rows * k];
    for (weight_block, destination) in weight_rows
        .chunks_exact(block_bytes)
        .zip(decoded.chunks_exact_mut(operands.block_elements))
    {
        decode(weight_block, destination);
    }
    let grouped = rows / TILE_ROWS * TILE_ROWS;
    let block = position_block(k);
    for block_start in (0..operands.positions).step_by(block) {
        let range = block_start..(block_start + block).min(operands.positions);
        for first_row in (0..grouped).step_by(TILE_ROWS) {
            // SAFETY: avx2 and fma are enabled on this function.
            unsafe { group::<TILE_ROWS>(operands, &decoded, first_row, range.clone(), out) };
        }
        for first_row in grouped..rows {
            // SAFETY: avx2 and fma are enabled on this function.
            unsafe { group::<1>(operands, &decoded, first_row, range.clone(), out) };
        }
    }
}

/// Wide (`[row][position]`) result of `rows` quantized weight rows against `positions`
/// f32 activation rows, bit-identical to calling the codec's per-position
/// `dot_*_f32` kernel for every output. `None` when the host lacks avx2 or fma.
pub(super) fn matmul_exact_wide(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    positions: usize,
    session: Option<&MatmulSession<'_>>,
    shape: (usize, usize, fn(&[u8], &mut [f32])),
) -> Option<Result<Vec<f32>, TensorError>> {
    let (block_bytes, block_elements, decode) = shape;
    if !fma_runtime_available() {
        return None;
    }
    let k = activation.len() / positions;
    let row_bytes = weights.len() / rows;
    if k == 0
        || row_bytes % block_bytes != 0
        || row_bytes / block_bytes * block_elements != k
        || activation.len() != k * positions
    {
        return Some(Err(TensorError::QuantizedShapeMismatch {
            reason: "weight rows and f32 activation do not match the codec's block layout",
        }));
    }
    let operands = ExactOperands { activation, k, positions, block_elements };
    let unit_rows = unit_row_budget(k);
    let run_unit = |first_row: usize, out: &mut [f32]| {
        let group_rows = unit_rows.min(rows - first_row);
        let unit_weights = &weights[first_row * row_bytes..(first_row + group_rows) * row_bytes];
        // SAFETY: avx2 and fma were detected above; shapes were checked above.
        unsafe { unit(&operands, unit_weights, row_bytes, block_bytes, decode, &mut out[..group_rows * positions]) };
    };
    Some(match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => matmul_rows_threaded(
            rows.div_ceil(unit_rows),
            unit_rows * positions,
            workers,
            session,
            k,
            |unit_index, slot| {
                run_unit(unit_index * unit_rows, slot);
                Ok(())
            },
        )
        .map(|mut output| {
            output.truncate(rows * positions);
            output
        }),
        None => {
            let mut output = vec![0.0f32; rows * positions];
            for first_row in (0..rows).step_by(unit_rows) {
                run_unit(first_row, &mut output[first_row * positions..]);
            }
            Ok(output)
        }
    })
}

pub(super) fn codec_shape(codec: Codec) -> Option<(usize, usize, fn(&[u8], &mut [f32]))> {
    match codec {
        Codec::Q4K => Some((Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, dequantize_block_q4_k)),
        Codec::Q6K => Some((Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, dequantize_block_q6_k)),
        Codec::Q5_0 => Some((Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, dequantize_block_q5_0)),
        Codec::Q8_0 => Some((Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, dequantize_block_q8_0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_bytes(seed: u64, length: usize) -> Vec<u8> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect()
    }

    fn weight_rows(codec: Codec, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let (block_bytes, _, _) = codec_shape(codec).expect("codec has an exact wide shape");
        let mut bytes = lcg_bytes(seed, rows * blocks * block_bytes);
        for (index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
            let scale = half::f16::from_f32((seed + index as u64) as f32 % 97.0 / 211.0 + 0.003).to_le_bytes();
            let minimum = half::f16::from_f32((seed * 3 + index as u64) as f32 % 89.0 / 197.0 + 0.002).to_le_bytes();
            match codec {
                Codec::Q6K => block[Q6K_D_OFFSET..Q6K_D_OFFSET + 2].copy_from_slice(&scale),
                Codec::Q4K => {
                    block[Q4K_D_OFFSET..Q4K_D_OFFSET + 2].copy_from_slice(&scale);
                    block[Q4K_DMIN_OFFSET..Q4K_DMIN_OFFSET + 2].copy_from_slice(&minimum);
                }
                _ => block[..2].copy_from_slice(&scale),
            }
        }
        bytes
    }

    #[test]
    fn exact_wide_tile_is_bit_identical_to_the_per_position_f32_kernels() {
        let kernels: [(Codec, usize, fn(&[u8], &[f32]) -> Result<f32, TensorError>); 4] = [
            (Codec::Q4K, 256, dot_q4k_f32),
            (Codec::Q6K, 256, dot_q6k_f32),
            (Codec::Q5_0, 32, dot_q5_0_f32),
            (Codec::Q8_0, 32, dot_q8_0_f32),
        ];
        let shapes = [(1usize, 2usize), (3, 5), (2, 4), (5, 7), (16, 33), (17, 66), (9, 129)];
        for (codec, block_elements, dot) in kernels {
            for (rows, positions) in shapes {
                let blocks = if block_elements == 256 { 3 } else { 24 };
                let k = blocks * block_elements;
                let weights = weight_rows(codec, rows, blocks, 5 + rows as u64);
                let activation: Vec<f32> = lcg_bytes(17 + positions as u64, positions * k)
                    .into_iter()
                    .map(|byte| (f32::from(byte) - 127.5) / 37.0)
                    .collect();
                let shape = codec_shape(codec).expect("codec has an exact wide shape");
                let wide = matmul_exact_wide(&weights, rows, &activation, positions, None, shape)
                    .expect("avx2 and fma are present on the test host")
                    .expect("wide matmul runs");
                let row_bytes = weights.len() / rows;
                for row in 0..rows {
                    for position in 0..positions {
                        let expected = dot(
                            &weights[row * row_bytes..(row + 1) * row_bytes],
                            &activation[position * k..(position + 1) * k],
                        )
                        .expect("per-position kernel runs");
                        assert_eq!(
                            wide[row * positions + position].to_bits(),
                            expected.to_bits(),
                            "{codec:?} rows={rows} positions={positions} row={row} position={position}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exact_wide_rejects_an_activation_that_does_not_match_the_row_layout() {
        let weights = weight_rows(Codec::Q8_0, 2, 3, 1);
        let activation = vec![0.5f32; 2 * (3 * 32 + 1)];
        let shape = codec_shape(Codec::Q8_0).expect("q8_0 has an exact wide shape");
        let result = matmul_exact_wide(&weights, 2, &activation, 2, None, shape)
            .expect("avx2 and fma are present on the test host");
        assert!(result.is_err());
    }
}
