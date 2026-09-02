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
}
