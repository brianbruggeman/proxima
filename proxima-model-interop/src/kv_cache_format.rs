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

/// The kernel's answer for a K and V stored as different formats.
#[cfg(feature = "gpu")]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    omega::cached_attention_kv_codecs_for(key, value)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Without a cached-attention kernel the host reads the cache: each row is rounded through the
/// format's own codec at the write (`generate::host_kv_rounding`), so any format that encodes is read.
#[cfg(not(feature = "gpu"))]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    for format in [key, value] {
        let encodes = is_binary32_format(format)
            || proxima_numeric::FormatCodec::for_format(format).is_some_and(|codec| codec.can_encode());
        if !encodes {
            return Err(String::from("the format has no codec that encodes"));
        }
    }
    Ok(())
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
#[cfg(all(feature = "std", feature = "gpu"))]
pub(crate) fn is_narrow(format: &NumericFormat) -> bool {
    kernel_codec(format).is_ok_and(|codec| codec.is_some())
}

/// Whether the row-tiled kernel (speculative verify, several query rows against the cached range)
/// reads this cache: the K side and the V side each, for a head of `head_dim` elements.
#[cfg(feature = "std")]
pub(crate) fn row_tiled_reads(key: &NumericFormat, value: &NumericFormat, head_dim: u64) -> bool {
    row_tiled_reads_format(key, head_dim) && row_tiled_reads_format(value, head_dim)
}

/// Bytes `elements` values occupy stored in `format`: the format's own packed size, with a partial
/// trailing block counted whole. Binary16 is 2 bytes an element, binary32 4, a 4-bit block format
/// its block bytes over its block elements.
#[cfg(all(feature = "instrument", any(test, all(feature = "metal", target_os = "macos"))))]
pub(crate) fn stored_bytes(format: &NumericFormat, elements: u64) -> u64 {
    let count = elements as usize;
    let block_elements = usize::from(format.elements_per_block()).max(1);
    let blocks = count.div_ceil(block_elements);
    format
        .packed_len(count)
        .unwrap_or(blocks * usize::from(format.bytes_per_block())) as u64
}

/// Bytes one decode step writes into the KV cache: `new_rows` rows, each layer contributing its two
/// K planes of `even_odd_row` elements in `key_format` and its V row of `v_row` elements in
/// `value_format`. `layer_rows` is `(even_odd_row, v_row)` for every layer that holds the cache.
#[cfg(all(feature = "instrument", any(test, all(feature = "metal", target_os = "macos"))))]
pub(crate) fn step_write_bytes(
    (key_format, value_format): (&NumericFormat, &NumericFormat),
    new_rows: u64,
    layer_rows: impl IntoIterator<Item = (u64, u64)>,
) -> u64 {
    layer_rows
        .into_iter()
        .map(|(even_odd_row, v_row)| {
            2 * stored_bytes(key_format, even_odd_row) + stored_bytes(value_format, v_row)
        })
        .sum::<u64>()
        * new_rows
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu")]
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
    fn a_cpu_only_build_admits_every_format_that_encodes_including_mixed_pairs() {
        for element in [NumericFormat::Float16, NumericFormat::BFloat16, NumericFormat::Q8_0, NumericFormat::Q4_0] {
            assert!(admit_pair(&element, &element).is_ok(), "{element:?}");
        }
        assert!(admit_pair(&NumericFormat::Float16, &NumericFormat::BFloat16).is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_mixed_pair_is_refused_naming_both_types() {
        let error = admit_pair(&NumericFormat::Float16, &NumericFormat::BFloat16).expect_err("mixed pair");

        let message = error.to_string();
        assert!(message.contains("Float16") && message.contains("BFloat16"), "{message}");
        assert!(message.contains("same type"), "{message}");
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
    #[cfg(feature = "instrument")]
    #[test]
    fn binary16_is_two_bytes_an_element_and_binary32_four() {
        assert_eq!(stored_bytes(&NumericFormat::Float16, 512), 1024);
        assert_eq!(stored_bytes(&NumericFormat::Float32, 512), 2048);
    }

    #[cfg(feature = "instrument")]
    #[test]
    fn a_four_bit_block_format_counts_its_packed_size() {
        let block_elements = u64::from(NumericFormat::Q4_0.elements_per_block());
        let block_bytes = u64::from(NumericFormat::Q4_0.bytes_per_block());
        assert_eq!((block_elements, block_bytes), (32, 18), "q4_0 packs 32 nibbles plus a scale into 18 bytes");
        assert_eq!(stored_bytes(&NumericFormat::Q4_0, 64), 36);
        assert_eq!(stored_bytes(&NumericFormat::Q4_0, 33), 36, "a partial block is stored whole");
    }


    #[cfg(feature = "instrument")]
    #[test]
    fn gemma4_fixture_layers_write_the_hand_computed_bytes_per_step() {
        use crate::gemma4_fixture::{FULL_HEAD_DIM, GEMMA4_SLIDING_PATTERN, SLIDING_HEAD_DIM};
        // attn_k and attn_v are [embedding, head_dim] (gemma4_fixture.rs:130-155): one kv head, so each
        // layer's K planes are head_dim / 2 elements apiece and its V row head_dim
        let layer_rows: Vec<(u64, u64)> = GEMMA4_SLIDING_PATTERN
            .iter()
            .map(|sliding| {
                let head_dim = if *sliding { SLIDING_HEAD_DIM } else { FULL_HEAD_DIM };
                (head_dim / 2, head_dim)
            })
            .collect();
        let half = (&NumericFormat::Float16, &NumericFormat::Float16);
        // sliding: (128 + 128 + 256) * 2 = 1024 twice, full: (256 + 256 + 512) * 2 = 2048
        assert_eq!(step_write_bytes(half, 1, layer_rows.iter().copied()), 1024 + 1024 + 2048);
        assert_eq!(step_write_bytes(half, 5, layer_rows.iter().copied()), 5 * 4096);
        let full = (&NumericFormat::Float32, &NumericFormat::Float32);
        assert_eq!(step_write_bytes(full, 1, layer_rows.iter().copied()), 8192);
    }
}
