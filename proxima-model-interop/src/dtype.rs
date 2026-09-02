//! `GgmlType <-> DType` mapping — the element-type half of the common
//! denominator both formats share. Only fixed-width, non-quantized scalar
//! types have a counterpart on both sides: GGUF's block-quantized types
//! (`Q4_0`, `Q4_K`, the `IQ*` family, ...) pack multiple elements into a
//! shared scale/bias, which safetensors' flat typed-array model has no way
//! to express, and safetensors' `Bool`/unsigned-integer dtypes have no
//! `GgmlType` counterpart since ggml never defined them.

use proxima_gguf::GgmlType;
use proxima_tensor::DType;

/// `None` for any block-quantized or otherwise non-scalar `GgmlType` —
/// there is no flat, per-element safetensors dtype that could represent
/// packed block data without dequantizing it, and dequantizing is out of
/// scope for a format transform.
#[must_use]
pub fn ggml_to_dtype(ggml_type: GgmlType) -> Option<DType> {
    match ggml_type {
        GgmlType::F32 => Some(DType::Float32),
        GgmlType::F16 => Some(DType::Float16),
        GgmlType::Bf16 => Some(DType::BFloat16),
        GgmlType::I8 => Some(DType::Int8),
        GgmlType::I16 => Some(DType::Int16),
        GgmlType::I32 => Some(DType::Int32),
        GgmlType::I64 => Some(DType::Int64),
        GgmlType::F64 => Some(DType::Float64),
        _ => None,
    }
}

/// `None` for `Bool` and the unsigned-integer/128-bit `DType`s — ggml has
/// no wire type for any of them.
#[must_use]
pub fn dtype_to_ggml(dtype: DType) -> Option<GgmlType> {
    match dtype {
        DType::Float32 => Some(GgmlType::F32),
        DType::Float16 => Some(GgmlType::F16),
        DType::BFloat16 => Some(GgmlType::Bf16),
        DType::Int8 => Some(GgmlType::I8),
        DType::Int16 => Some(GgmlType::I16),
        DType::Int32 => Some(GgmlType::I32),
        DType::Int64 => Some(GgmlType::I64),
        DType::Float64 => Some(GgmlType::F64),
        DType::Bool
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64
        | DType::Int128
        | DType::UInt128 => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mapped_ggml_type_round_trips_through_dtype() {
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
            GgmlType::F64,
        ];
        for ggml_type in types {
            let dtype =
                ggml_to_dtype(ggml_type).unwrap_or_else(|| panic!("{ggml_type:?} should map"));
            assert_eq!(
                dtype_to_ggml(dtype),
                Some(ggml_type),
                "{ggml_type:?} round trip"
            );
        }
    }

    #[test]
    fn quantized_ggml_types_have_no_dtype_counterpart() {
        for ggml_type in [
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
            GgmlType::Iq2Xxs,
        ] {
            assert_eq!(ggml_to_dtype(ggml_type), None, "{ggml_type:?}");
        }
    }

    #[test]
    fn safetensors_only_dtypes_have_no_ggml_counterpart() {
        for dtype in [DType::Bool, DType::UInt8, DType::UInt32, DType::Int128] {
            assert_eq!(dtype_to_ggml(dtype), None, "{dtype:?}");
        }
    }
}
