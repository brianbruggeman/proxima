//! Which `-ctk`/`-ctv` cache types the serving path admits, and how the decode loop treats them.
//!
//! Every answer comes from the cache's own [`NumericFormat`] (the type the serving config holds) or
//! from the kernel's own capability query (`omega::cached_attention_kv_codec_for`, and
//! `omega::cached_attention_row_tiled_kv_codec_for` for the verify kernel), never from a list of
//! format names kept here. A build without the cached-attention kernels (no `gpu` feature) reads
//! only the binary32 cache the graph itself produces.

use alloc::format;
#[cfg(all(feature = "std", feature = "gpu", feature = "metal-output-placement", target_os = "macos"))]
use alloc::string::ToString;
use alloc::string::String;

use proxima_numeric::NumericFormat;

use crate::error::InteropError;

/// The format the graph evaluates its K and V rows in: the activation format every run produces
/// until a config names another, which every driver and the host read.
fn graph_output_format() -> NumericFormat {
    NumericFormat::default()
}

/// The cache format the serving configuration declares for `-ctk`, resolved by conflaguration from
/// `ServingSettings` (its field default, then `PROXIMA_SERVING_KV_CACHE_KEY_QUANT`).
#[cfg(feature = "std")]
fn configured_cache_format() -> Option<NumericFormat> {
    use conflaguration::Settings;

    crate::serving_settings::ServingSettings::from_env().ok().map(|settings| settings.kv_cache_key_quant)
}

#[cfg(not(feature = "std"))]
fn configured_cache_format() -> Option<NumericFormat> {
    None
}

/// The cache format the serving default stores: the configured one when the active driver's cached
/// attention reads it, otherwise the graph's own output format. Never a format the active backend
/// would refuse at admission.
pub(crate) fn default_cache_format() -> NumericFormat {
    match configured_cache_format() {
        Some(configured) if kernel_pair(&configured, &configured).is_ok() => configured,
        _ => graph_output_format(),
    }
}

/// Whether `format` is the one the graph evaluates its K and V rows in, which a placed cache buffer
/// or a device carry receives without a conversion step.
fn stores_graph_output_format(format: &NumericFormat) -> bool {
    format.is_identity_of(&graph_output_format())
}

/// The codec the kernel reads a K cache stored as `format` with, whatever the V cache is.
#[cfg(all(
    feature = "std",
    feature = "gpu",
    feature = "metal-output-placement",
    target_os = "macos"
))]
fn kernel_codec(format: &NumericFormat) -> Result<Option<NumericFormat>, String> {
    omega::cached_attention_key_codec_for(format).map_err(|error| error.to_string())
}

/// The active driver's answer for a K and V stored as different formats; a build with no driver
/// compiled reads the cache on the host.
#[cfg(feature = "gpu")]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    match omega::backend::GpuDriver::for_target() {
        Some(driver) => driver.admits_kv_cache(key, value),
        None => host_pair(key, value),
    }
}

/// Without a cached-attention kernel the host reads the cache: each row is encoded through the
/// format's own codec at the write (the host store is `PackedRows`), so any format that encodes is read.
#[cfg(not(feature = "gpu"))]
fn kernel_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    host_pair(key, value)
}

fn host_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    for format in [key, value] {
        let encodes = stores_graph_output_format(format)
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
    stores_graph_output_format(format)
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

/// Whether the cache is held in the format the graph evaluates in, so the step's rows land in it directly.
#[cfg(feature = "std")]
pub(crate) fn stores_graph_output(format: &NumericFormat) -> bool {
    stores_graph_output_format(format)
}

/// Whether the active backend keeps a cache stored as `key` and `value` on the device between
/// steps: it admits the pair and its encode render accepts both formats. A build with no driver
/// keeps only the graph-output format.
#[cfg(feature = "std")]
pub(crate) fn carry_admitted(key: &NumericFormat, value: &NumericFormat) -> bool {
    carry_pair(key, value).is_ok()
}

#[cfg(all(feature = "std", feature = "gpu"))]
fn carry_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    match omega::backend::GpuDriver::for_target() {
        Some(driver) => driver.admits_kv_carry(key, value),
        None => graph_output_pair(key, value),
    }
}

#[cfg(all(feature = "std", not(feature = "gpu")))]
fn carry_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    graph_output_pair(key, value)
}

#[cfg(feature = "std")]
fn graph_output_pair(key: &NumericFormat, value: &NumericFormat) -> Result<(), String> {
    if stores_graph_output_format(key) && stores_graph_output_format(value) {
        Ok(())
    } else {
        Err(String::from("only the graph output format is carried"))
    }
}

/// Whether the cache is narrower than the graph's binary32 output: the kernel reads it through a
/// codec, so the step's rows are narrowed into a device-resident cache.
#[cfg(all(
    feature = "std",
    feature = "gpu",
    feature = "metal-output-placement",
    target_os = "macos"
))]
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
    fn admission_is_exactly_the_active_drivers_answer_over_every_format_pair() {
        let Some(driver) = omega::backend::GpuDriver::for_target() else {
            return;
        };
        for key in &NumericFormat::ALL {
            for value in &NumericFormat::ALL {
                assert_eq!(
                    admit_pair(key, value).is_ok(),
                    driver.admits_kv_cache(key, value).is_ok(),
                    "{key:?} key, {value:?} value"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn the_default_cache_format_is_one_the_active_backend_admits() {
        let format = default_cache_format();
        assert!(admit_pair(&format, &format).is_ok(), "{format:?}");
    }

    #[cfg(all(feature = "gpu", feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
    #[test]
    fn a_cuda_build_admits_only_the_graph_output_format_and_defaults_to_it() {
        for format in &NumericFormat::ALL {
            assert_eq!(
                admit_pair(format, format).is_ok(),
                stores_graph_output_format(format),
                "{format:?}"
            );
        }
        assert!(stores_graph_output_format(&default_cache_format()));
    }

    #[cfg(all(feature = "gpu", feature = "cuda", not(all(feature = "metal", target_os = "macos"))))]
    #[test]
    fn a_refused_cache_format_names_the_backend_and_the_format() {
        let refused = NumericFormat::ALL
            .iter()
            .find(|format| !stores_graph_output_format(format))
            .expect("a format other than the graph output exists");
        let message = admit_pair(refused, refused).expect_err("refused").to_string();
        assert!(message.contains("cuda") && message.contains(&format!("{refused:?}")), "{message}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn only_the_graph_output_format_stores_graph_output() {
        assert!(stores_graph_output(&NumericFormat::Float32));
        for element in [NumericFormat::Float16, NumericFormat::BFloat16, NumericFormat::Q8_0] {
            assert!(!stores_graph_output(&element), "{element:?}");
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "gpu",
        feature = "metal-output-placement",
        target_os = "macos"
    ))]
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
