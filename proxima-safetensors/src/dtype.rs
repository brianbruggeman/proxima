//! Safetensors `dtype` string -> [`proxima_tensor::DType`] mapping.
//!
//! The full dtype vocabulary (`BOOL, F4, F6_E2M3, F6_E3M2, U8, I8, F8_E5M2,
//! F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ, I16, U16, F16, BF16, I32,
//! U32, F32, C64, F64, I64, U64`) was checked against the reference
//! `huggingface/safetensors` crate's `Dtype` enum
//! (`safetensors/src/tensor.rs` on `main`). `proxima_tensor::DType` was
//! widened on `a0f5f97` to add `Int16, UInt16, Int64, UInt64, Int128,
//! UInt128, Float64`, so `I16, U16, I64, U64, F64` now have counterparts
//! and are mapped below. `C64` (complex) and the sub-byte / 8-bit float
//! family (`F4, F6_E2M3, F6_E3M2, F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ,
//! F8_E5M2FNUZ`) still have no `DType` counterpart — there is no
//! fixed-width machine scalar to represent them — so those still return
//! [`SafetensorsError::UnsupportedDtype`] rather than guessing a lossy
//! substitute.

use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::DType;

use crate::error::SafetensorsError;

/// Maps a safetensors `dtype` string onto the `DType` this crate compiles
/// against. Returns [`SafetensorsError::UnsupportedDtype`] for any
/// safetensors dtype `proxima_tensor::DType` cannot represent yet.
pub fn map_dtype(tensor: &str, dtype: &str) -> Result<DType, SafetensorsError> {
    match dtype {
        "BOOL" => Ok(DType::Bool),
        "I8" => Ok(DType::Int8),
        "U8" => Ok(DType::UInt8),
        "I16" => Ok(DType::Int16),
        "U16" => Ok(DType::UInt16),
        "I32" => Ok(DType::Int32),
        "U32" => Ok(DType::UInt32),
        "I64" => Ok(DType::Int64),
        "U64" => Ok(DType::UInt64),
        "BF16" => Ok(DType::BFloat16),
        "F16" => Ok(DType::Float16),
        "F32" => Ok(DType::Float32),
        "F64" => Ok(DType::Float64),
        // still no DType counterpart: complex, and the sub-byte / 8-bit
        // micro-float family — none of these is a fixed-width machine
        // scalar `DType` can represent.
        "F4" | "F6_E2M3" | "F6_E3M2" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" | "C64" => Err(SafetensorsError::UnsupportedDtype {
            tensor: tensor.into(),
            dtype: dtype.into(),
        }),
        _ => Err(SafetensorsError::UnsupportedDtype {
            tensor: tensor.into(),
            dtype: dtype.into(),
        }),
    }
}

/// The writer's half of [`map_dtype`]: `DType` back to the safetensors wire
/// string. `None` for `Int128`/`UInt128` — safetensors has no 128-bit
/// integer dtype, so these have no wire representation to write.
#[must_use]
pub fn dtype_to_wire(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Bool => Some("BOOL"),
        DType::Int8 => Some("I8"),
        DType::UInt8 => Some("U8"),
        DType::Int16 => Some("I16"),
        DType::UInt16 => Some("U16"),
        DType::Int32 => Some("I32"),
        DType::UInt32 => Some("U32"),
        DType::Int64 => Some("I64"),
        DType::UInt64 => Some("U64"),
        DType::BFloat16 => Some("BF16"),
        DType::Float16 => Some("F16"),
        DType::Float32 => Some("F32"),
        DType::Float64 => Some("F64"),
        DType::Int128 | DType::UInt128 => None,
    }
}

/// Decodes one `F8_E4M3` (OCP E4M3FN: 1 sign, 4 exponent bias-7, 3
/// mantissa, no infinities, `S.1111.111` reserved for NaN) byte to `f32`.
/// `DType` has no fixed-width counterpart for this format (see the module
/// doc), so this bypasses `DType` entirely and decodes straight from the
/// raw byte — the same reason [`f8_block_dequant`] takes raw bytes rather
/// than a `DType`-typed buffer.
#[must_use]
pub fn f8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0_f32 } else { -1.0_f32 };
    let exponent = (byte >> 3) & 0x0F;
    let mantissa = byte & 0x07;

    if exponent == 0x0F && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        // subnormal: 2^(1-bias) * (mantissa / 8)
        return sign * libm_ldexp(f32::from(mantissa) / 8.0, -6);
    }
    // normal: 2^(exponent-bias) * (1 + mantissa/8)
    let significand = 1.0 + f32::from(mantissa) / 8.0;
    sign * libm_ldexp(significand, i32::from(exponent) - 7)
}

/// `f32 * 2^exponent` without pulling in `libm`/`std::f32::exp2` — `no_std`
/// safe, exact for the small integer exponents E4M3 ever produces
/// (`-6..=8`).
fn libm_ldexp(value: f32, exponent: i32) -> f32 {
    if exponent >= 0 {
        value * f32::from_bits(((127 + exponent) as u32) << 23)
    } else {
        value / f32::from_bits(((127 - exponent) as u32) << 23)
    }
}

/// Dequantizes a block-wise-scaled `F8_E4M3` tensor to `f32`.
///
/// `bytes` is the raw row-major `F8_E4M3` buffer for a tensor of shape
/// `shape` (last two dims are `[rows, cols]`; anything higher-rank is
/// flattened into leading blocks of `rows x cols`, matching how the
/// official FP8 checkpoint's `weight` / `weight_scale_inv` pair is laid
/// out — DeepSeek-style block FP8, `weight_block_size: [128, 128]` per
/// `config.json`). `scales` is `weight_scale_inv`'s own row-major buffer
/// over the `ceil(rows/block) x ceil(cols/block)` scale grid: one `f32`
/// scale per `block x block` tile, applied as `value = raw_f32 *
/// scale[row / block, col / block]`.
///
/// # Errors
///
/// [`SafetensorsError::TensorDataLengthMismatch`] if `bytes.len()` doesn't
/// match `shape`'s element count, or if `scales.len()` doesn't match the
/// expected scale-grid size.
pub fn f8_block_dequant(
    bytes: &[u8],
    scales: &[f32],
    shape: &[u64],
    block: usize,
) -> Result<Vec<f32>, SafetensorsError> {
    let element_count: u64 = shape.iter().product();
    if bytes.len() as u64 != element_count {
        return Err(SafetensorsError::TensorDataLengthMismatch {
            tensor: "f8_block_dequant".into(),
            expected: element_count,
            found: bytes.len() as u64,
        });
    }
    let (rows, cols) = trailing_two_dims(shape);
    let scale_cols = cols.div_ceil(block);
    let scale_rows = rows.div_ceil(block);
    let expected_scales = (scale_rows * scale_cols) as u64;
    if scales.len() as u64 != expected_scales {
        return Err(SafetensorsError::TensorDataLengthMismatch {
            tensor: "f8_block_dequant.weight_scale_inv".into(),
            expected: expected_scales,
            found: scales.len() as u64,
        });
    }

    let matrix_size = rows * cols;
    let matrix_count = bytes.len().checked_div(matrix_size).unwrap_or(0);
    let mut out = vec![0.0_f32; bytes.len()];
    for matrix in 0..matrix_count {
        let base = matrix * matrix_size;
        for row in 0..rows {
            let scale_row = row / block;
            for col in 0..cols {
                let scale_col = col / block;
                let scale = scales[scale_row * scale_cols + scale_col];
                let index = base + row * cols + col;
                out[index] = f8_e4m3_to_f32(bytes[index]) * scale;
            }
        }
    }
    Ok(out)
}

/// The last two dimensions of `shape`, treating a 0-D or 1-D shape as a
/// single row (`rows = 1`, `cols = product of shape`) — the block grid is
/// only ever meaningful over a matrix, and every FP8 weight tensor in the
/// official checkpoint is rank >= 2.
fn trailing_two_dims(shape: &[u64]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0] as usize),
        rank => (shape[rank - 2] as usize, shape[rank - 1] as usize),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test]
    #[case::bool_dtype("BOOL", DType::Bool)]
    #[case::int8("I8", DType::Int8)]
    #[case::uint8("U8", DType::UInt8)]
    #[case::int32("I32", DType::Int32)]
    #[case::uint32("U32", DType::UInt32)]
    #[case::bfloat16("BF16", DType::BFloat16)]
    #[case::float16("F16", DType::Float16)]
    #[case::float32("F32", DType::Float32)]
    async fn known_dtype_strings_map(#[case] wire: &str, #[case] expected: DType) {
        assert_eq!(map_dtype("t", wire), Ok(expected));
    }

    /// The dtype widening in `a0f5f97` added `Int16, UInt16, Int64,
    /// UInt64, Float64` to `proxima_tensor::DType`, so these five wire
    /// strings — previously `UnsupportedDtype` — now map to real variants.
    #[proxima::test]
    #[case::int16("I16", DType::Int16)]
    #[case::uint16("U16", DType::UInt16)]
    #[case::int64("I64", DType::Int64)]
    #[case::uint64("U64", DType::UInt64)]
    #[case::float64("F64", DType::Float64)]
    async fn widened_dtype_strings_now_map(#[case] wire: &str, #[case] expected: DType) {
        assert_eq!(map_dtype("t", wire), Ok(expected));
    }

    /// Dtypes the widening did NOT reach: complex has no fixed-width
    /// machine scalar counterpart at all, and the micro-float family
    /// (sub-byte / 8-bit) has no `DType` variant either. These must keep
    /// returning a typed error rather than a silent, wrong-width guess.
    #[proxima::test]
    #[case::complex("C64")]
    #[case::fp6_e2m3("F6_E2M3")]
    #[case::fp6_e3m2("F6_E3M2")]
    #[case::fp8_e4m3("F8_E4M3")]
    #[case::fp8_e5m2("F8_E5M2")]
    #[case::fp8_e8m0("F8_E8M0")]
    #[case::fp8_e4m3fnuz("F8_E4M3FNUZ")]
    #[case::fp8_e5m2fnuz("F8_E5M2FNUZ")]
    #[case::sub_byte("F4")]
    #[case::unknown_junk("NOT_A_DTYPE")]
    async fn unsupported_dtype_strings_return_typed_error(#[case] wire: &str) {
        let error = map_dtype("t", wire).expect_err("dtype has no DType counterpart");
        assert!(matches!(error, SafetensorsError::UnsupportedDtype { .. }));
    }

    #[proxima::test]
    #[case::bool_dtype(DType::Bool)]
    #[case::int8(DType::Int8)]
    #[case::uint8(DType::UInt8)]
    #[case::int16(DType::Int16)]
    #[case::uint16(DType::UInt16)]
    #[case::int32(DType::Int32)]
    #[case::uint32(DType::UInt32)]
    #[case::int64(DType::Int64)]
    #[case::uint64(DType::UInt64)]
    #[case::bfloat16(DType::BFloat16)]
    #[case::float16(DType::Float16)]
    #[case::float32(DType::Float32)]
    #[case::float64(DType::Float64)]
    async fn dtype_to_wire_round_trips_through_map_dtype(#[case] dtype: DType) {
        let wire = dtype_to_wire(dtype).expect("dtype has a wire representation");
        assert_eq!(map_dtype("t", wire), Ok(dtype));
    }

    #[proxima::test]
    #[case::int128(DType::Int128)]
    #[case::uint128(DType::UInt128)]
    async fn dtype_to_wire_returns_none_for_dtypes_safetensors_cannot_represent(
        #[case] dtype: DType,
    ) {
        assert_eq!(dtype_to_wire(dtype), None);
    }

    /// Hand-computed E4M3 values, byte laid out `S EEEE MMM`: `0x00` is
    /// positive zero; `0x38` = `0_0111_000` -- exponent field `0111` (bias
    /// 7, so `2^0`), mantissa `000` -> `1.0`; `0x40` = `0_1000_000` bumps
    /// the exponent field to `1000` (`2^1`) -> `2.0`; `0x7E` =
    /// `0_1111_110` is the maximum normal, exponent `1111` (`2^8`),
    /// mantissa `110` -> `256 * 1.75 = 448.0`; `0xB8` is `-1.0` (sign bit
    /// set on `0x38`).
    #[proxima::test]
    #[case::positive_zero(0x00, 0.0)]
    #[case::one(0x38, 1.0)]
    #[case::two(0x40, 2.0)]
    #[case::max_normal(0x7E, 448.0)]
    #[case::negative_one(0xB8, -1.0)]
    async fn f8_e4m3_decodes_hand_computed_values(#[case] byte: u8, #[case] expected: f32) {
        assert_eq!(f8_e4m3_to_f32(byte), expected);
    }

    #[proxima::test]
    async fn f8_e4m3_reserved_bit_pattern_is_nan() {
        assert!(f8_e4m3_to_f32(0x7F).is_nan());
        assert!(f8_e4m3_to_f32(0xFF).is_nan());
    }

    #[proxima::test]
    async fn f8_e4m3_subnormal_decodes_below_the_smallest_normal() {
        // exponent field 0, mantissa 1: 2^-6 * (1/8) = 2^-9.
        let value = f8_e4m3_to_f32(0x01);
        assert!((value - 2.0_f32.powi(-9)).abs() < 1e-9);
    }

    /// A 4x4 tensor split into four 2x2 blocks (`block = 2`), each block
    /// carrying its own scale. Every element in a block is `0x38` (raw
    /// `1.0`), so the dequantized value is exactly that block's scale --
    /// hand-computed, not derived from the function under test.
    #[proxima::test]
    async fn f8_block_dequant_applies_per_block_scale() {
        let bytes = [0x38_u8; 16];
        let shape = [4_u64, 4];
        let scales = [1.0_f32, 2.0, 3.0, 4.0];

        let dequantized = f8_block_dequant(&bytes, &scales, &shape, 2).expect("valid shapes");

        let expected_block = |row: usize, col: usize| -> f32 {
            match (row / 2, col / 2) {
                (0, 0) => 1.0,
                (0, 1) => 2.0,
                (1, 0) => 3.0,
                (1, 1) => 4.0,
                _ => unreachable!("only four 2x2 blocks in a 4x4 tensor"),
            }
        };
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    dequantized[row * 4 + col],
                    expected_block(row, col),
                    "row {row} col {col}"
                );
            }
        }
    }

    #[proxima::test]
    async fn f8_block_dequant_rejects_byte_shape_mismatch() {
        let bytes = [0x38_u8; 15];
        let shape = [4_u64, 4];
        let scales = [1.0_f32; 4];
        let outcome = f8_block_dequant(&bytes, &scales, &shape, 2);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TensorDataLengthMismatch { .. })
        ));
    }

    #[proxima::test]
    async fn f8_block_dequant_rejects_scale_grid_mismatch() {
        let bytes = [0x38_u8; 16];
        let shape = [4_u64, 4];
        let scales = [1.0_f32; 3];
        let outcome = f8_block_dequant(&bytes, &scales, &shape, 2);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TensorDataLengthMismatch { .. })
        ));
    }
}
