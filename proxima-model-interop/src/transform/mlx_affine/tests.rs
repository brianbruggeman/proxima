//! Oracle: the unpack expressions below are transcribed from mlx
//! `mlx/backend/cpu/quantized.cpp` at commit
//! `06eb7483f03bca49a80e596f4089d02a507a2cf9`:
//!
//! - lines 65-93 `extract_bits` for 3, 5 and 6 bits (bytes in, one pack out)
//! - lines 99-160 `_qmm` for the power-of-two path (`wi & bitmask`, `wi >>= bits`
//!   per byte) and the 3, 5, 6 bit path (`extract_bits` per `bytes_per_pack`)
//! - lines 1165-1233 `quantize` for the packing side (`out_el |= w_el << (k *
//!   bits)`, bytes written least significant first)
//!
//! and `mlx/backend/common/quantized.h` lines 5-12 for `get_pack_factor` and
//! `get_bytes_per_pack`. The C++ carries no worked example values, so the
//! vectors here are generated weights plus one hand-derived group.
//! `python/mlx/nn/layers/quantized.py` at the same commit fixes the `weight`
//! shape `(out, in * bits / 32)` (line 266 `in_dims = (in_dims * 32) // bits`).

use super::*;

type Outcome = Result<(), InteropError>;
type PackedFixture = (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>);

const ROWS: usize = 4;
const COLUMNS: usize = 128;
const EXPORT_BITS: [u8; 6] = [2, 3, 4, 5, 6, 8];
const FLOAT_DTYPES: [DType; 3] = [DType::BFLOAT16, DType::FLOAT16, DType::FLOAT32];

fn weights(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| ((index * 7919 + 13) % 1000) as f32 / 1000.0 * 0.08 - 0.04)
        .collect()
}

fn packed(bits: u8, group_size: usize) -> Result<PackedFixture, InteropError> {
    let values = weights(ROWS, COLUMNS);
    let params = AffineParams::new(bits, group_size)?;
    let mut weight = alloc::vec![0u32; values.len() * usize::from(bits) / 32];
    let mut scales = alloc::vec![0f32; values.len() / group_size];
    let mut biases = alloc::vec![0f32; values.len() / group_size];
    pack_affine(&values, ROWS, params, &mut weight, &mut scales, &mut biases)?;
    Ok((values, weight, scales, biases))
}

fn source_levels(
    values: &[f32],
    scales: &[f32],
    biases: &[f32],
    bits: u8,
    group_size: usize,
) -> Vec<u32> {
    let top_level = ((1u32 << bits) - 1) as f32;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let group = index / group_size;
            ((value - biases[group]) / scales[group])
                .round_ties_even()
                .clamp(0.0, top_level) as u32
        })
        .collect()
}

fn extract_3_bits(pack: &[u8; 3]) -> Vec<u32> {
    let byte = |index: usize| u32::from(pack[index]);
    alloc::vec![
        byte(0) & 0x7,
        (byte(0) & 0x38) >> 3,
        ((byte(0) & 0xc0) >> 6) + ((byte(1) & 0x1) << 2),
        (byte(1) & 0xe) >> 1,
        (byte(1) & 0x70) >> 4,
        ((byte(1) & 0x80) >> 7) + ((byte(2) & 0x3) << 1),
        (byte(2) & 0x1c) >> 2,
        (byte(2) & 0xe0) >> 5,
    ]
}

fn extract_5_bits(pack: &[u8; 5]) -> Vec<u32> {
    let byte = |index: usize| u32::from(pack[index]);
    alloc::vec![
        byte(0) & 0x1f,
        ((byte(0) & 0xe0) >> 5) + ((byte(1) & 0x3) << 3),
        (byte(1) & 0x7c) >> 2,
        ((byte(1) & 0x80) >> 7) + ((byte(2) & 0xf) << 1),
        ((byte(2) & 0xf0) >> 4) + ((byte(3) & 0x1) << 4),
        (byte(3) & 0x3e) >> 1,
        ((byte(3) & 0xc0) >> 6) + ((byte(4) & 0x7) << 2),
        (byte(4) & 0xf8) >> 3,
    ]
}

fn extract_6_bits(pack: &[u8; 3]) -> Vec<u32> {
    let byte = |index: usize| u32::from(pack[index]);
    alloc::vec![
        byte(0) & 0x3f,
        ((byte(0) >> 6) & 0x03) + ((byte(1) & 0x0f) << 2),
        ((byte(1) >> 4) & 0x0f) + ((byte(2) & 0x03) << 4),
        (byte(2) >> 2) & 0x3f,
    ]
}

fn extract_power_of_two(byte: u8, bits: u8) -> Vec<u32> {
    let bitmask = (1u32 << bits) - 1;
    let mut remaining = byte;
    (0..8 / bits)
        .map(|_| {
            let level = u32::from(remaining) & bitmask;
            if bits != 8 {
                remaining >>= bits;
            }
            level
        })
        .collect()
}

fn unpack_like_mlx(weight: &[u32], bits: u8) -> Vec<u32> {
    let bytes: Vec<u8> = weight.iter().flat_map(|word| word.to_le_bytes()).collect();
    match bits {
        3 => bytes
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(extract_3_bits)
            .collect(),
        5 => bytes
            .as_chunks::<5>()
            .0
            .iter()
            .flat_map(extract_5_bits)
            .collect(),
        6 => bytes
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(extract_6_bits)
            .collect(),
        _ => bytes
            .iter()
            .flat_map(|byte| extract_power_of_two(*byte, bits))
            .collect(),
    }
}

fn assert_unpacks_to_source(bits: u8) -> Outcome {
    let (values, weight, scales, biases) = packed(bits, 64)?;
    let expected = source_levels(&values, &scales, &biases, bits, 64);
    assert_eq!(unpack_like_mlx(&weight, bits), expected, "bits {bits}");
    Ok(())
}

fn export_packed(bits: u8, dtype: DType) -> Result<AffinePacked, InteropError> {
    let values = weights(ROWS, COLUMNS);
    match export_affine(
        &values,
        [ROWS, COLUMNS],
        AffineParams::new(bits, 64)?,
        dtype,
    )? {
        AffineExport::Packed(tensor) => Ok(tensor),
        AffineExport::Float => Err(shape_error(ROWS, COLUMNS, "unexpected float fallback")),
    }
}

fn hand_derived_group() -> Vec<f32> {
    (0..32)
        .map(|index| -1.0 + 0.25 * (index % 16) as f32)
        .collect()
}

fn pack_hand_group() -> Result<(Vec<u32>, f32, f32), InteropError> {
    let values = hand_derived_group();
    let (mut weight, mut scales, mut biases) = (alloc::vec![0u32; 4], [0f32], [0f32]);
    pack_affine(
        &values,
        1,
        AffineParams::new(4, 32)?,
        &mut weight,
        &mut scales,
        &mut biases,
    )?;
    Ok((weight, scales[0], biases[0]))
}

fn reconstruction_error(value: f32, scale: f32, bias: f32, level: u32) -> f32 {
    (value - (scale * level as f32 + bias)).abs()
}

#[test]
fn mlx_affine_2_bit_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(2)
}

#[test]
fn mlx_affine_4_bit_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(4)
}

#[test]
fn mlx_affine_8_bit_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(8)
}

#[test]
fn mlx_affine_3_bit_straddling_pack_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(3)
}

#[test]
fn mlx_affine_5_bit_straddling_pack_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(5)
}

#[test]
fn mlx_affine_6_bit_straddling_pack_unpacks_to_source_levels() -> Outcome {
    assert_unpacks_to_source(6)
}

#[test]
fn mlx_affine_reversed_element_order_does_not_unpack() -> Outcome {
    let (values, weight, scales, biases) = packed(4, 64)?;
    let mut reversed = source_levels(&values, &scales, &biases, 4, 64);
    reversed.reverse();
    assert_ne!(unpack_like_mlx(&weight, 4), reversed);
    Ok(())
}

#[test]
fn mlx_affine_weight_shape_is_out_by_in_times_bits_over_32() -> Outcome {
    for bits in EXPORT_BITS {
        let tensor = export_packed(bits, DType::FLOAT32)?;
        let expected = [ROWS, COLUMNS * usize::from(bits) / 32];
        assert_eq!(tensor.weight_shape, expected, "bits {bits}");
        assert_eq!(
            tensor.weight.len(),
            expected[0] * expected[1],
            "bits {bits}"
        );
    }
    Ok(())
}

#[test]
fn mlx_affine_scales_shape_is_out_by_in_over_group_size() -> Outcome {
    let tensor = export_packed(4, DType::FLOAT32)?;
    assert_eq!(tensor.scales_shape, [ROWS, COLUMNS / 64]);
    assert_eq!(tensor.scales.len(), ROWS * (COLUMNS / 64) * 4);
    Ok(())
}

#[test]
fn mlx_affine_biases_shape_is_out_by_in_over_group_size() -> Outcome {
    let tensor = export_packed(4, DType::FLOAT32)?;
    assert_eq!(
        tensor.biases.len(),
        tensor.scales_shape[0] * tensor.scales_shape[1] * 4
    );
    Ok(())
}

#[test]
fn mlx_affine_scales_and_biases_dtype_equals_export_float_dtype() -> Outcome {
    for dtype in FLOAT_DTYPES {
        let tensor = export_packed(4, dtype)?;
        let group_count = ROWS * COLUMNS / 64;
        assert_eq!(tensor.dtype, dtype);
        assert_eq!(tensor.scales.len(), group_count * dtype.size_bytes());
        assert_eq!(tensor.biases.len(), group_count * dtype.size_bytes());
    }
    Ok(())
}

/// Hand walk of `quantized.cpp:1165-1233` for the group `-1.0 + 0.25 * (i % 16)`,
/// 32 elements, 4 bits:
///
/// - `n_bins = 15`, `w_min = -1.0`, `w_max = 2.75`
/// - `mask = |w_min| > |w_max| = false`, so the max edge anchors
/// - `scale = max(3.75 / 15, 1e-7) = 0.25`, negated by `mask == false` to `-0.25`
/// - `edge = w_max = 2.75`, `q0 = rint(2.75 / -0.25) = -11`, nonzero
/// - so `scale = edge / q0 = 2.75 / -11 = -0.25` and `bias = edge = 2.75`
/// - `q = rint((w - 2.75) / -0.25) = 15 - (i % 16)`
/// - dequant: `-0.25 * (15 - k) + 2.75 = -1.0 + 0.25 * k`, the source value
#[test]
fn mlx_affine_hand_derived_group_gives_expected_scale_bias_and_levels() -> Outcome {
    let (weight, scale, bias) = pack_hand_group()?;
    let expected: Vec<u32> = (0..32).map(|index| 15 - index % 16).collect();
    assert_eq!((scale, bias), (-0.25, 2.75));
    assert_eq!(unpack_like_mlx(&weight, 4), expected);
    Ok(())
}

#[test]
fn mlx_affine_hand_derived_group_control_shifted_levels_do_not_match() -> Outcome {
    let (weight, _, _) = pack_hand_group()?;
    let shifted: Vec<u32> = (0..32).map(|index| 15 - index % 16 + 1).collect();
    assert_ne!(unpack_like_mlx(&weight, 4), shifted);
    Ok(())
}

/// The incumbent anchors the grid on one edge (`quantized.cpp:1203-1211`), so
/// the opposite edge can fall outside `0..=n_bins` and clamp: an element that
/// does not clamp reconstructs within half a scale, one that clamps within one.
#[test]
fn mlx_affine_reconstruction_is_within_half_a_scale() -> Outcome {
    for bits in EXPORT_BITS {
        let (values, weight, scales, biases) = packed(bits, 64)?;
        let levels = unpack_like_mlx(&weight, bits);
        let top_level = ((1u32 << bits) - 1) as f32;
        for (index, value) in values.iter().enumerate() {
            let group = index / 64;
            let (scale, bias) = (scales[group], biases[group]);
            let quotient = ((value - bias) / scale).round_ties_even();
            let error = reconstruction_error(*value, scale, bias, levels[index]) / scale.abs();
            let clamped = !(0.0..=top_level).contains(&quotient);
            let bound = if clamped { 1.0 } else { 0.5 };
            assert!(
                error <= bound + 1e-4,
                "bits {bits} element {index} error {error}"
            );
        }
    }
    Ok(())
}

#[test]
fn mlx_affine_reconstruction_control_level_off_by_one_exceeds_half_a_scale() -> Outcome {
    let (weight, scale, bias) = pack_hand_group()?;
    let off_by_one = unpack_like_mlx(&weight, 4)[5] + 1;
    let error = reconstruction_error(hand_derived_group()[5], scale, bias, off_by_one);
    assert!(error > scale.abs() * 0.5);
    Ok(())
}

#[test]
fn mlx_affine_accepts_all_eighteen_bits_and_group_size_pairs() -> Outcome {
    let values = weights(2, COLUMNS);
    for (bits, group_size) in EXPORT_BITS
        .iter()
        .flat_map(|bits| [32, 64, 128].map(|group| (*bits, group)))
    {
        let params = AffineParams::new(bits, group_size)?;
        let exported = export_affine(&values, [2, COLUMNS], params, DType::BFLOAT16)?;
        assert!(
            matches!(exported, AffineExport::Packed(_)),
            "bits {bits} group {group_size}"
        );
    }
    Ok(())
}

#[test]
fn mlx_affine_refuses_seven_bits() {
    let refused = AffineParams::new(7, 64);
    assert!(matches!(
        refused,
        Err(InteropError::AffineBitsUnsupported { bits: 7 })
    ));
}

#[test]
fn mlx_affine_refuses_group_size_forty_eight() {
    let refused = AffineParams::new(4, 48);
    assert!(matches!(
        refused,
        Err(InteropError::AffineGroupSizeUnsupported { group_size: 48 })
    ));
}

#[test]
fn mlx_affine_default_params_are_four_bits_group_size_sixty_four() {
    let params = AffineParams::default();
    assert_eq!((params.bits(), params.group_size()), (4, 64));
}

#[test]
fn mlx_affine_last_dim_not_a_group_multiple_stays_float() -> Outcome {
    let values = weights(ROWS, 100);
    let exported = export_affine(
        &values,
        [ROWS, 100],
        AffineParams::default(),
        DType::BFLOAT16,
    )?;
    assert_eq!(exported, AffineExport::Float);
    Ok(())
}

#[test]
fn mlx_affine_last_dim_of_one_group_multiple_is_quantized_control() -> Outcome {
    let values = weights(ROWS, 128);
    let exported = export_affine(
        &values,
        [ROWS, 128],
        AffineParams::default(),
        DType::BFLOAT16,
    )?;
    assert!(matches!(exported, AffineExport::Packed(_)));
    Ok(())
}
