//! `GgmlType <-> DType` mapping — the element-type half of the common
//! denominator both formats share. Only fixed-width, non-quantized scalar
//! types have a counterpart on both sides: GGUF's block-quantized types
//! (`Q4_0`, `Q4_K`, the `IQ*` family, ...) pack multiple elements into a
//! shared scale/bias, which safetensors' flat typed-array model has no way
//! to express, and safetensors' `Bool`/unsigned-integer dtypes have no
//! `GgmlType` counterpart since ggml never defined them.

use proxima_gguf::GgmlType;
use proxima_primitives::float_layout::FloatLayout;
use proxima_tensor::DType;

/// `None` for any block-quantized or otherwise non-scalar `GgmlType` —
/// there is no flat, per-element safetensors dtype that could represent
/// packed block data without dequantizing it, and dequantizing is out of
/// scope for a format transform.
#[must_use]
pub fn ggml_to_dtype(ggml_type: GgmlType) -> Option<DType> {
    match ggml_type {
        GgmlType::F32 => Some(DType::FLOAT32),
        GgmlType::F16 => Some(DType::FLOAT16),
        GgmlType::Bf16 => Some(DType::BFLOAT16),
        GgmlType::I8 => Some(DType::INT8),
        GgmlType::I16 => Some(DType::INT16),
        GgmlType::I32 => Some(DType::INT32),
        GgmlType::I64 => Some(DType::INT64),
        GgmlType::F64 => Some(DType::FLOAT64),
        _ => None,
    }
}

/// The wire type of a layout is the ggml scalar whose fields equal it; any other layout
/// (bf8, bf4, fp6, a layout nobody named) has no ggml scalar and travels as a proxima codec.
#[must_use]
pub fn dtype_to_ggml(dtype: DType) -> Option<GgmlType> {
    match dtype {
        DType::Layout(layout) => [
            (FloatLayout::BF32, GgmlType::F32),
            (FloatLayout::F16, GgmlType::F16),
            (FloatLayout::BF16, GgmlType::Bf16),
            (FloatLayout::INT8, GgmlType::I8),
            (FloatLayout::INT16, GgmlType::I16),
            (FloatLayout::INT32, GgmlType::I32),
            (FloatLayout::INT64, GgmlType::I64),
            (FloatLayout::F64, GgmlType::F64),
        ]
        .into_iter()
        .find_map(|(scalar, wire)| scalar.same_format(&layout).then_some(wire)),
        DType::Bool => None,
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
        for dtype in [
            DType::Bool,
            DType::BFLOAT8,
            DType::UINT8,
            DType::UINT32,
            DType::INT128,
        ] {
            assert_eq!(dtype_to_ggml(dtype), None, "{dtype:?}");
        }
    }
}
