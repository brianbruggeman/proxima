//! Which `-ctk`/`-ctv` cache types the serving path admits, and how the decode loop treats them.
//!
//! Every answer comes from the cache's own [`NumericFormat`] (the type the serving config holds) or
//! from the kernel's own capability query (`omega::cached_attention_kv_codec_for`, and
//! `omega::cached_attention_row_tiled_kv_codec_for` for the verify kernel), never from a list of
//! format names kept here. A build without the cached-attention kernels (no `gpu` feature) reads
//! only the binary32 cache the graph itself produces.

use alloc::format;
#[cfg(feature = "gpu")]
use alloc::string::ToString;
use alloc::string::String;

use proxima_numeric::NumericFormat;

use crate::error::InteropError;

fn is_binary32_format(format: &NumericFormat) -> bool {
    format.is_identity_of(&NumericFormat::Float32)
}

/// The codec the kernel reads a K cache stored as `format` with, whatever the V cache is.
#[cfg(all(feature = "std", feature = "gpu"))]
fn kernel_codec(format: &NumericFormat) -> Result<Option<NumericFormat>, String> {
    omega::cached_attention_key_codec_for(format).map_err(|error| error.to_string())
}

#[cfg(all(feature = "std", not(feature = "gpu")))]
fn kernel_codec(format: &NumericFormat) -> Result<Option<NumericFormat>, String> {
    if is_binary32_format(format) {
        Ok(None)
    } else {
        Err(String::from("this build has no cached-attention kernel that reads a narrow cache"))
    }
}

/// The kernel's answer for a K and V stored as different formats.
#[cfg(feature = "gpu")]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    omega::cached_attention_kv_codecs_for(key, value)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(not(feature = "gpu"))]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    if is_binary32_format(key) && is_binary32_format(value) {
        Ok(())
    } else {
        Err(String::from("this build has no cached-attention kernel that reads a narrow cache"))
    }
}

#[cfg(all(feature = "std", feature = "gpu"))]
fn row_tiled_reads_format(format: &NumericFormat, head_dim: u64) -> bool {
    omega::cached_attention_row_tiled_reads(format, head_dim).is_ok()
}

#[cfg(all(feature = "std", not(feature = "gpu")))]
fn row_tiled_reads_format(format: &NumericFormat, _head_dim: u64) -> bool {
    is_binary32_format(format)
}

/// Admits a `-ctk`/`-ctv` pair: key and value in one type that is binary32 or that the
/// cached-attention kernel reads, or the one mixed pair the decode kernel reads: a block key
/// layout beside a scalar narrow value. Any other pair is refused with the formats named.
///
/// # Errors
/// [`InteropError::UnsupportedServingConfig`] naming the refused pair and the reason.
pub(crate) fn admit_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), InteropError> {
    let refuse = |reason: String| {
        InteropError::UnsupportedServingConfig(format!(
            "kv_cache_key_quant={key:?} kv_cache_value_quant={value:?} (-ctk/-ctv): {reason}"
        ))
    };
    if key != value {
        return kernel_pair(key, value).map_err(|reason| {
            refuse(format!(
                "key and value must be stored in the same type unless the decode kernel reads the pair: {reason}"
            ))
        });
    }
    kernel_pair(key, key).map_err(|reason| {
        refuse(format!(
            "cache format {key:?} has no read path: {reason}; a block layout is read as the key \
             cache (-ctk) beside a scalar or block value cache (-ctv)"
        ))
    })
}

/// Whether the cache is the graph's own binary32 output, so the step's rows land in it directly.
#[cfg(feature = "std")]
pub(crate) fn is_binary32(format: &NumericFormat) -> bool {
    is_binary32_format(format)
}

/// Whether the cache is narrower than the graph's binary32 output: the kernel reads it through a
/// codec, so the step's rows are narrowed into a device-resident cache.
#[cfg(feature = "std")]
pub(crate) fn is_narrow(format: &NumericFormat) -> bool {
    kernel_codec(format).is_ok_and(|codec| codec.is_some())
}

/// Whether the row-tiled kernel (speculative verify, several query rows against the cached range)
/// reads this cache: the K side and the V side each, for a head of `head_dim` elements.
#[cfg(feature = "std")]
pub(crate) fn row_tiled_reads(key: &NumericFormat, value: &NumericFormat, head_dim: u64) -> bool {
    row_tiled_reads_format(key, head_dim) && row_tiled_reads_format(value, head_dim)
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn a_binary32_cache_is_admitted() {
        assert!(admit_pair(&NumericFormat::Float32, &NumericFormat::Float32).is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_narrow_pairs_the_kernel_reads_are_admitted() {
        for element in [NumericFormat::Float16, NumericFormat::BFloat16] {
            assert!(admit_pair(&element, &element).is_ok(), "{element:?}");
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn a_narrow_cache_is_refused_at_config_time_when_no_kernel_reads_it() {
        for element in [NumericFormat::Float16, NumericFormat::BFloat16] {
            let error = admit_pair(&element, &element).expect_err("no kernel reads a narrow cache");

            let message = error.to_string();
            assert!(message.contains(&format!("{element:?}")), "{message}");
            assert!(message.contains("no cached-attention kernel that reads a narrow cache"), "{message}");
        }
    }

    #[test]
    fn a_mixed_pair_is_refused_naming_both_types() {
        let error = admit_pair(&NumericFormat::Float16, &NumericFormat::BFloat16).expect_err("mixed pair");

        let message = error.to_string();
        assert!(message.contains("Float16") && message.contains("BFloat16"), "{message}");
        assert!(message.contains("same type"), "{message}");
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn a_quantized_cache_is_refused_naming_its_format() {
        for element in [NumericFormat::Q8_0, NumericFormat::Q4_0] {
            let error = admit_pair(&element, &element).expect_err("quantized cache");

            let message = error.to_string();
            assert!(message.contains(&format!("{element:?}")), "{message}");
            assert!(message.contains("no read path"), "{message}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_block_key_beside_a_scalar_or_block_value_is_admitted_and_every_other_block_pairing_is_refused() {
        for (key, value) in [
            (NumericFormat::Q4_0, NumericFormat::Float16),
            (NumericFormat::Q8_0, NumericFormat::Float16),
            (NumericFormat::Q4_0, NumericFormat::Q4_0),
            (NumericFormat::Q8_0, NumericFormat::Q8_0),
            (NumericFormat::Q8_0, NumericFormat::Q4_0),
        ] {
            assert!(admit_pair(&key, &value).is_ok(), "{key:?} key, {value:?} value");
        }
        for (key, value) in [
            (NumericFormat::Float16, NumericFormat::Q4_0),
            (NumericFormat::Q4_0, NumericFormat::Float32),
            (NumericFormat::Float32, NumericFormat::Float16),
        ] {
            let error = admit_pair(&key, &value).expect_err("the kernel reads no such pair");

            let message = error.to_string();
            assert!(message.contains(&format!("{key:?}")) && message.contains(&format!("{value:?}")), "{message}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn only_f32_is_binary32() {
        assert!(is_binary32(&NumericFormat::Float32));
        for element in [NumericFormat::Float16, NumericFormat::BFloat16, NumericFormat::Q8_0] {
            assert!(!is_binary32(&element), "{element:?}");
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn f16_and_bf16_are_narrow_and_f32_is_not() {
        assert!(!is_narrow(&NumericFormat::Float32));
        assert!(is_narrow(&NumericFormat::Float16));
        assert!(is_narrow(&NumericFormat::BFloat16));
        assert!(is_narrow(&NumericFormat::Q4_0), "a block key layout is a narrow cache");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_row_tiled_verify_kernel_reads_what_omega_describes_for_k_and_v() {
        const HEAD_DIM: u64 = 128;
        let formats = [
            NumericFormat::Float32,
            NumericFormat::Float16,
            NumericFormat::BFloat16,
            NumericFormat::BFloat8,
            NumericFormat::Q8_0,
            NumericFormat::Q4_0,
        ];
        for format in &formats {
            let expected = format.is_identity_of(&NumericFormat::Float32)
                || omega::msl::layout_render::readable_scalar_cache(format).is_some()
                || omega::msl::layout_render::readable_block_layout(format).is_some();
            assert_eq!(row_tiled_reads(format, &NumericFormat::Float32, HEAD_DIM), expected, "K {format:?}");
            assert_eq!(row_tiled_reads(&NumericFormat::Float32, format, HEAD_DIM), expected, "V {format:?}");
        }
        assert!(row_tiled_reads(&NumericFormat::Float16, &NumericFormat::Float16, HEAD_DIM), "binary16 is the flint default");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_block_layout_whose_blocks_do_not_tile_the_head_is_refused_for_k_and_v() {
        let head_dim = 48;
        let q4_elements = omega::msl::layout_render::readable_block_layout(&NumericFormat::Q4_0)
            .map(|layout| u64::from(layout.elements))
            .expect("q4_0 is a readable block layout");
        assert_ne!((head_dim / 2) % q4_elements, 0, "fixture head must not tile the block");
        assert!(!row_tiled_reads(&NumericFormat::Q4_0, &NumericFormat::Float32, head_dim));
        assert!(!row_tiled_reads(&NumericFormat::Float32, &NumericFormat::Q4_0, head_dim));
    }
}
