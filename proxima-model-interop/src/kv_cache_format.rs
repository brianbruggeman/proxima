//! Which `-ctk`/`-ctv` cache types the serving path admits, and how the decode loop treats them.
//!
//! Every answer comes from the cache type's own [`NumericFormat`] (`GgmlType::numeric_format`) or
//! from the kernel's own capability query (`omega::cached_attention_kv_codec_for`, and
//! `omega::cached_attention_row_tiled_kv_codec_for` for the verify kernel), never from a list of
//! format names kept here. A build without the cached-attention kernels (no `gpu` feature) reads
//! only the binary32 cache the graph itself produces.

use alloc::format;
#[cfg(feature = "gpu")]
use alloc::string::ToString;
use alloc::string::String;

use proxima_gguf::types::GgmlType;
use proxima_numeric::NumericFormat;

use crate::error::InteropError;

fn cache_format(element: GgmlType) -> Result<NumericFormat, String> {
    element
        .numeric_format()
        .ok_or_else(|| format!("{element:?} names no numeric format"))
}

fn is_binary32_format(format: &NumericFormat) -> bool {
    format.is_identity_of(&NumericFormat::Float32)
}

#[cfg(feature = "gpu")]
fn kernel_codec(format: &NumericFormat) -> Result<Option<NumericFormat>, String> {
    omega::cached_attention_kv_codec_for(format).map_err(|error| error.to_string())
}

#[cfg(not(feature = "gpu"))]
fn kernel_codec(format: &NumericFormat) -> Result<Option<NumericFormat>, String> {
    if is_binary32_format(format) {
        Ok(None)
    } else {
        Err(String::from("this build has no cached-attention kernel that reads a narrow cache"))
    }
}

#[cfg(all(feature = "std", feature = "gpu"))]
fn row_tiled_reads_format(format: &NumericFormat) -> bool {
    omega::cached_attention_row_tiled_kv_codec_for(format).is_ok()
}

#[cfg(all(feature = "std", not(feature = "gpu")))]
fn row_tiled_reads_format(format: &NumericFormat) -> bool {
    is_binary32_format(format)
}

/// Admits a `-ctk`/`-ctv` pair: key and value in one type that is binary32 or that the
/// cached-attention kernel reads. A mixed pair, or a type no kernel reads (Q8_0, Q4_0, ...), is
/// refused with the format named.
///
/// # Errors
/// [`InteropError::UnsupportedServingConfig`] naming the refused pair and the reason.
pub(crate) fn admit_pair(key: GgmlType, value: GgmlType) -> Result<(), InteropError> {
    let refuse = |reason: String| {
        InteropError::UnsupportedServingConfig(format!(
            "kv_cache_key_quant={key:?} kv_cache_value_quant={value:?} (-ctk/-ctv): {reason}"
        ))
    };
    if key != value {
        return Err(refuse(String::from(
            "key and value must be stored in the same type; the cached attention kernels read one cache codec for both",
        )));
    }
    let Ok(format) = cache_format(key) else {
        return Err(refuse(format!("{key:?} names no numeric format")));
    };
    kernel_codec(&format).map(|_| ()).map_err(|reason| {
        refuse(format!(
            "cache format {format:?} has no read path: {reason}; a quantized cache (Q8_0, Q4_0, ...) \
             is not readable because the quantized matmul dispatch only handles a flat \
             `weight[rows, k] x activation[batch, k]` matmul, while the cached-attention reduces \
             are batched reduces over a shared kv-head axis"
        ))
    })
}

/// Whether the cache is the graph's own binary32 output, so the step's rows land in it directly.
#[cfg(feature = "std")]
pub(crate) fn is_binary32(element: GgmlType) -> bool {
    cache_format(element).is_ok_and(|format| is_binary32_format(&format))
}

/// Whether the cache is narrower than the graph's binary32 output: the kernel reads it through a
/// codec, so the step's rows are narrowed into a device-resident cache.
#[cfg(feature = "std")]
pub(crate) fn is_narrow(element: GgmlType) -> bool {
    cache_format(element)
        .ok()
        .and_then(|format| kernel_codec(&format).ok())
        .is_some_and(|codec| codec.is_some())
}

/// Whether the row-tiled kernel (speculative verify, several query rows against the cached range)
/// reads this cache.
#[cfg(feature = "std")]
pub(crate) fn row_tiled_reads(element: GgmlType) -> bool {
    cache_format(element).is_ok_and(|format| row_tiled_reads_format(&format))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_binary32_and_narrow_pairs_the_kernel_reads_are_admitted() {
        let admitted = [GgmlType::F32, GgmlType::F16, GgmlType::Bf16];

        for element in admitted {
            assert!(admit_pair(element, element).is_ok(), "{element:?}");
        }
    }

    #[test]
    fn a_mixed_pair_is_refused_naming_both_types() {
        let error = admit_pair(GgmlType::F16, GgmlType::Bf16).expect_err("mixed pair");

        let message = error.to_string();
        assert!(message.contains("F16") && message.contains("Bf16"), "{message}");
        assert!(message.contains("same type"), "{message}");
    }

    #[test]
    fn a_quantized_cache_is_refused_naming_its_format() {
        for element in [GgmlType::Q8_0, GgmlType::Q4_0] {
            let error = admit_pair(element, element).expect_err("quantized cache");

            let message = error.to_string();
            assert!(message.contains(&format!("{element:?}")), "{message}");
            assert!(message.contains("no read path"), "{message}");
        }
    }

    #[test]
    fn a_type_with_no_numeric_format_is_refused() {
        let error = admit_pair(GgmlType::I8, GgmlType::I8).expect_err("integer cache");

        assert!(error.to_string().contains("I8"), "{error}");
    }

    #[test]
    fn only_f32_is_binary32() {
        assert!(is_binary32(GgmlType::F32));
        for element in [GgmlType::F16, GgmlType::Bf16, GgmlType::Q8_0] {
            assert!(!is_binary32(element), "{element:?}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn f16_and_bf16_are_narrow_and_f32_is_not() {
        assert!(!is_narrow(GgmlType::F32));
        assert!(is_narrow(GgmlType::F16));
        assert!(is_narrow(GgmlType::Bf16));
        assert!(!is_narrow(GgmlType::Q8_0));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_row_tiled_verify_kernel_reads_bf16_and_not_f16() {
        assert!(row_tiled_reads(GgmlType::F32));
        assert!(row_tiled_reads(GgmlType::Bf16));
        assert!(!row_tiled_reads(GgmlType::F16));
    }
}
