use super::*;

use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_srli_epi16,
    _mm256_storeu_si256,
};

use gemm_q6k_avx2::lane_levels;
use proxima_gguf::quant::q4_k::get_scale_min_k4;

const SUB_BLOCKS: usize = 8;
const SUB_BLOCK_ELEMENTS: usize = 32;
const TILE_ROWS: usize = 4;
const TILE_POSITIONS: usize = 2;
const POSITION_BLOCK: usize = 16;
pub(super) const KQUANT_UNIT_ROWS: usize = 8;

// a super-block with its bit-packing undone once per unit: u8 levels per 32-element
// sub-block, the per-16-element scale pairs widened to i16 lanes, and the i16 weights
// that multiply the activation's `bsums` (Q4_K mins / Q6_K scales).
pub(super) struct UnpackedKQuantRows {
    levels: Vec<u8>,
    scales: Vec<i16>,
    bias: Vec<i16>,
    weight_d: Vec<f32>,
    weight_dmin: Vec<f32>,
}

impl UnpackedKQuantRows {
    fn new(rows: usize, blocks: usize) -> Self {
        let block_count = rows * blocks;
        Self {
            levels: vec![0; block_count * SUB_BLOCKS * SUB_BLOCK_ELEMENTS],
            scales: vec![0; block_count * SUB_BLOCKS * 16],
            bias: vec![0; block_count * 16],
            weight_d: vec![0.0; block_count],
            weight_dmin: vec![0.0; block_count],
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unpack_q4k_block(weight_block: &[u8], block_index: usize, unpacked: &mut UnpackedKQuantRows) {
    let mut scale_bytes = [0u8; Q4K_SCALE_BYTES];
    scale_bytes.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    unpacked.weight_d[block_index] = f16_le_at(weight_block, Q4K_D_OFFSET);
    unpacked.weight_dmin[block_index] = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    for sub_block in 0..SUB_BLOCKS {
        let (scale, min) = get_scale_min_k4(sub_block, &scale_bytes);
        let lanes = (block_index * SUB_BLOCKS + sub_block) * 16;
        unpacked.scales[lanes..lanes + 16].fill(i16::from(scale));
        unpacked.bias[block_index * 16 + sub_block * 2] = i16::from(min);
        unpacked.bias[block_index * 16 + sub_block * 2 + 1] = i16::from(min);
    }
    // SAFETY: avx2 is enabled; each load reads 32 bytes of the 128-byte `qs` plane and each
    // store writes one 32-byte level vector inside this block's slot.
    unsafe {
        let low_mask = _mm256_set1_epi8(0x0f);
        for pair in 0..SUB_BLOCKS / 2 {
            let packed = _mm256_loadu_si256(weight_block[Q4K_QS_OFFSET + pair * 32..].as_ptr().cast());
            let destination = (block_index * SUB_BLOCKS + pair * 2) * SUB_BLOCK_ELEMENTS;
            let low = _mm256_and_si256(packed, low_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
            _mm256_storeu_si256(unpacked.levels[destination..].as_mut_ptr().cast(), low);
            _mm256_storeu_si256(unpacked.levels[destination + 32..].as_mut_ptr().cast(), high);
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unpack_q6k_block(weight_block: &[u8], block_index: usize, unpacked: &mut UnpackedKQuantRows) {
    let scales = &weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + 16];
    unpacked.weight_d[block_index] = f16_le_at(weight_block, Q6K_D_OFFSET);
    for (index, &scale) in scales.iter().enumerate() {
        unpacked.bias[block_index * 16 + index] = i16::from(scale.cast_signed());
    }
    for sub_block in 0..SUB_BLOCKS {
        let lanes = (block_index * SUB_BLOCKS + sub_block) * 16;
        unpacked.scales[lanes..lanes + 8].fill(i16::from(scales[sub_block * 2].cast_signed()));
        unpacked.scales[lanes + 8..lanes + 16].fill(i16::from(scales[sub_block * 2 + 1].cast_signed()));
    }
    // SAFETY: avx2 is enabled; offsets stay inside the 210-byte block and the level slot.
    unsafe {
        for half in 0..2 {
            let ql_base = weight_block[Q6K_QL_OFFSET + half * 64..].as_ptr();
            let ql_near = _mm256_loadu_si256(ql_base.cast());
            let ql_far = _mm256_loadu_si256(ql_base.add(32).cast());
            let high_pairs = _mm256_loadu_si256(weight_block[Q6K_QH_OFFSET + half * 32..].as_ptr().cast());
            for lane in 0..4 {
                let ql_pair = if lane % 2 == 0 { ql_near } else { ql_far };
                let levels = lane_levels(ql_pair, high_pairs, lane);
                let destination = (block_index * SUB_BLOCKS + half * 4 + lane) * SUB_BLOCK_ELEMENTS;
                _mm256_storeu_si256(unpacked.levels[destination..].as_mut_ptr().cast(), levels);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unpack_rows<const SIX_BIT: bool>(
    weight_rows: &[u8],
    row_bytes: usize,
    blocks: usize,
    unpacked: &mut UnpackedKQuantRows,
) {
    let block_bytes = row_bytes / blocks;
    for (block_index, weight_block) in weight_rows.chunks_exact(block_bytes).enumerate() {
        // SAFETY: avx2 is enabled on this function.
        unsafe {
            if SIX_BIT {
                unpack_q6k_block(weight_block, block_index, unpacked);
            } else {
                unpack_q4k_block(weight_block, block_index, unpacked);
            }
        }
    }
}

pub(super) struct KQuantOperands<'a> {
    pub(super) activation_q8k: &'a [u8],
    pub(super) positions: usize,
    pub(super) blocks: usize,
}

// ROWS weight rows x TILE positions. Integer sums are exact in any order, so the
// per-sub-block horizontal sums collapse into one vector accumulator per super-block;
// the float step and the sequential `total += block` match `dot_q*k_q8k` exactly.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile<const SIX_BIT: bool, const ROWS: usize, const TILE: usize>(
    operands: &KQuantOperands<'_>,
    unpacked: &UnpackedKQuantRows,
    first_row: usize,
    position: usize,
    out: &mut [f32],
) {
    let row_bytes = operands.blocks * Q8K_BLOCK_BYTES;
    let mut totals = [[0.0f32; TILE]; ROWS];
    for block in 0..operands.blocks {
        let mut accumulators = [[_mm256_setzero_si256(); TILE]; ROWS];
        // SAFETY: avx2 is enabled; every pointer addresses a whole vector inside the
        // `Q8K_BLOCK_BYTES` activation block or the unpacked slot of `(row, block)`.
        unsafe {
            let mut activation_blocks = [core::ptr::null::<u8>(); TILE];
            for (lane, slot) in activation_blocks.iter_mut().enumerate() {
                *slot = operands.activation_q8k.as_ptr().add((position + lane) * row_bytes + block * Q8K_BLOCK_BYTES);
            }
            for sub_block in 0..SUB_BLOCKS {
                for (row, row_accumulators) in accumulators.iter_mut().enumerate() {
                    let row_block = (first_row + row) * operands.blocks + block;
                    let levels = _mm256_loadu_si256(
                        unpacked.levels.as_ptr().add((row_block * SUB_BLOCKS + sub_block) * SUB_BLOCK_ELEMENTS).cast(),
                    );
                    let scales = _mm256_loadu_si256(
                        unpacked.scales.as_ptr().add((row_block * SUB_BLOCKS + sub_block) * 16).cast(),
                    );
                    for (lane, accumulator) in row_accumulators.iter_mut().enumerate() {
                        let activations = _mm256_loadu_si256(
                            activation_blocks[lane].add(Q8K_QS_OFFSET + sub_block * SUB_BLOCK_ELEMENTS).cast(),
                        );
                        *accumulator = _mm256_add_epi32(
                            *accumulator,
                            _mm256_madd_epi16(_mm256_maddubs_epi16(levels, activations), scales),
                        );
                    }
                }
            }
            for lane in 0..TILE {
                let bsums: __m256i = _mm256_loadu_si256(activation_blocks[lane].add(Q8K_BSUMS_OFFSET).cast());
                let activation_scale = f32::from_le_bytes([
                    *activation_blocks[lane].add(Q8K_D_OFFSET),
                    *activation_blocks[lane].add(Q8K_D_OFFSET + 1),
                    *activation_blocks[lane].add(Q8K_D_OFFSET + 2),
                    *activation_blocks[lane].add(Q8K_D_OFFSET + 3),
                ]);
                for row in 0..ROWS {
                    let row_block = (first_row + row) * operands.blocks + block;
                    let bias = _mm256_loadu_si256(unpacked.bias.as_ptr().add(row_block * 16).cast());
                    let correction = hsum_epi32_avx2(_mm256_madd_epi16(bsums, bias));
                    let sumi = hsum_epi32_avx2(accumulators[row][lane]);
                    let d = activation_scale * unpacked.weight_d[row_block];
                    totals[row][lane] += if SIX_BIT {
                        d * (sumi - 32 * correction) as f32
                    } else {
                        let dmin = activation_scale * unpacked.weight_dmin[row_block];
                        d.mul_add(sumi as f32, -(dmin * correction as f32))
                    };
                }
            }
        }
    }
    for (row, row_totals) in totals.into_iter().enumerate() {
        for (lane, total) in row_totals.into_iter().enumerate() {
            out[(first_row + row) * operands.positions + position + lane] = total;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn group<const SIX_BIT: bool, const ROWS: usize>(
    operands: &KQuantOperands<'_>,
    unpacked: &UnpackedKQuantRows,
    first_row: usize,
    positions: core::ops::Range<usize>,
    out: &mut [f32],
) {
    let mut position = positions.start;
    while position + TILE_POSITIONS <= positions.end {
        // SAFETY: avx2 and fma are enabled on this function.
        unsafe { tile::<SIX_BIT, ROWS, TILE_POSITIONS>(operands, unpacked, first_row, position, out) };
        position += TILE_POSITIONS;
    }
    while position < positions.end {
        // SAFETY: avx2 and fma are enabled on this function.
        unsafe { tile::<SIX_BIT, ROWS, 1>(operands, unpacked, first_row, position, out) };
        position += 1;
    }
}

// `out` is `[row][position]` for this unit's rows. Position blocks keep a slice of the
// activation matrix in L2 while every row group of the unit sweeps it.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn unit<const SIX_BIT: bool>(
    operands: &KQuantOperands<'_>,
    weight_rows: &[u8],
    row_bytes: usize,
    out: &mut [f32],
) {
    let rows = weight_rows.len() / row_bytes;
    let mut unpacked = UnpackedKQuantRows::new(rows, operands.blocks);
    // SAFETY: avx2 and fma are enabled on this function.
    unsafe { unpack_rows::<SIX_BIT>(weight_rows, row_bytes, operands.blocks, &mut unpacked) };
    let grouped = rows / TILE_ROWS * TILE_ROWS;
    for block_start in (0..operands.positions).step_by(POSITION_BLOCK) {
        let block_range = block_start..(block_start + POSITION_BLOCK).min(operands.positions);
        for first_row in (0..grouped).step_by(TILE_ROWS) {
            // SAFETY: avx2 and fma are enabled on this function.
            unsafe { group::<SIX_BIT, TILE_ROWS>(operands, &unpacked, first_row, block_range.clone(), out) };
        }
        for first_row in grouped..rows {
            // SAFETY: avx2 and fma are enabled on this function.
            unsafe { group::<SIX_BIT, 1>(operands, &unpacked, first_row, block_range.clone(), out) };
        }
    }
}

pub(super) fn matmul_tiled<const SIX_BIT: bool>(
    weights: &[u8],
    rows: usize,
    activation_q8k: &[u8],
    leading_total: usize,
    workers: usize,
    session: Option<&MatmulSession<'_>>,
) -> Option<Result<Vec<f32>, TensorError>> {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
        return None;
    }
    let row_bytes = weights.len() / rows;
    let block_bytes = if SIX_BIT { Q6K_BLOCK_BYTES } else { Q4K_BLOCK_BYTES };
    let operands = KQuantOperands {
        activation_q8k,
        positions: leading_total,
        blocks: row_bytes / block_bytes,
    };
    let units = rows.div_ceil(KQUANT_UNIT_ROWS);
    let tiled = matmul_rows_threaded(
        units,
        KQUANT_UNIT_ROWS * leading_total,
        workers,
        session,
        operands.blocks * Q4K_BLOCK_ELEMENTS,
        |unit_index, slot| {
            let first_row = unit_index * KQUANT_UNIT_ROWS;
            let group_rows = KQUANT_UNIT_ROWS.min(rows - first_row);
            let unit_weights = &weights[first_row * row_bytes..(first_row + group_rows) * row_bytes];
            // SAFETY: avx2 and fma were detected at runtime above.
            unsafe { unit::<SIX_BIT>(&operands, unit_weights, row_bytes, &mut slot[..group_rows * leading_total]) };
            Ok(())
        },
    );
    Some(tiled.map(|mut output| {
        output.truncate(rows * leading_total);
        output
    }))
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

    fn finite_f16(seed: u64) -> [u8; 2] {
        half::f16::from_f32((seed % 97) as f32 / 211.0 + 0.003).to_le_bytes()
    }

    fn weight_rows(six_bit: bool, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let block_bytes = if six_bit { Q6K_BLOCK_BYTES } else { Q4K_BLOCK_BYTES };
        let mut bytes = lcg_bytes(seed, rows * blocks * block_bytes);
        for (index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
            let scale = finite_f16(seed + index as u64);
            let minimum = finite_f16(seed * 3 + index as u64);
            if six_bit {
                block[Q6K_D_OFFSET..Q6K_D_OFFSET + 2].copy_from_slice(&scale);
            } else {
                block[Q4K_D_OFFSET..Q4K_D_OFFSET + 2].copy_from_slice(&scale);
                block[Q4K_DMIN_OFFSET..Q4K_DMIN_OFFSET + 2].copy_from_slice(&minimum);
            }
        }
        bytes
    }

    fn activation_of(seed: u64, length: usize) -> Vec<f32> {
        lcg_bytes(seed, length)
            .into_iter()
            .map(|byte| (f32::from(byte) - 127.5) / 37.0)
            .collect()
    }

    fn tiled_equals_row_dots(six_bit: bool) {
        let shapes = [(1usize, 2usize), (3, 5), (4, 2), (9, 17), (16, 33), (17, 40), (8, 129)];
        let dot: fn(&[u8], &[u8]) -> Result<f32, TensorError> = if six_bit { dot_q6k_q8k } else { dot_q4k_q8k };
        for (rows, positions) in shapes {
            let blocks = 3;
            let weights = weight_rows(six_bit, rows, blocks, 5 + rows as u64);
            let activation = activation_of(17 + positions as u64, positions * blocks * Q4K_BLOCK_ELEMENTS);
            let mut q8k = vec![0u8; positions * blocks * Q8K_BLOCK_BYTES];
            quantize_row_q8k(&activation, &mut q8k).expect("activation quantizes");
            let tiled = if six_bit {
                matmul_tiled::<true>(&weights, rows, &q8k, positions, 4, None)
            } else {
                matmul_tiled::<false>(&weights, rows, &q8k, positions, 4, None)
            }
            .expect("avx2 and fma are present on the test host")
            .expect("tiled matmul runs");
            let row_bytes = weights.len() / rows;
            let q8k_row_bytes = blocks * Q8K_BLOCK_BYTES;
            for row in 0..rows {
                for position in 0..positions {
                    let expected = dot(
                        &weights[row * row_bytes..(row + 1) * row_bytes],
                        &q8k[position * q8k_row_bytes..(position + 1) * q8k_row_bytes],
                    )
                    .expect("row dot runs");
                    assert_eq!(
                        tiled[row * positions + position].to_bits(),
                        expected.to_bits(),
                        "six_bit={six_bit} rows={rows} positions={positions} row={row} position={position}"
                    );
                }
            }
        }
    }

    #[test]
    fn q4_k_tile_is_bit_identical_to_the_per_position_row_dots() {
        tiled_equals_row_dots(false);
    }

    #[test]
    fn q6_k_tile_is_bit_identical_to_the_per_position_row_dots() {
        tiled_equals_row_dots(true);
    }
}
