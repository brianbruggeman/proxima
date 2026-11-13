//! What a narrow `-ctk`/`-ctv` cache holds, on the host.
//!
//! The host KV store ([`LayerCache`]) keeps binary32. A cache described as another
//! [`NumericFormat`] is read back through that format's own [`FormatCodec`], so each row the
//! host appends is encoded and decoded once, here, at the write: the stored binary32 are exactly
//! the values the narrow cache would give back, and every later read (the pad scratch, the
//! CPU `CachedAttention` evaluation) sees them unchanged. Rounding is idempotent, so a layer the
//! device cache later adopts encodes the same bytes. This composes
//! [`FormatCodec::encode`]/[`FormatCodec::decode`], the codecs `DeviceKv` narrows with, rather
//! than a second table of formats; a binary32 side holds no codec and appends untouched.
//!
//! Chosen over narrow bytes with a decode on every read: the cache vectors are shared by prefix
//! state, the prompt cache, ring checkpoints and sealing, all binary32 consumers, and a decode of
//! the whole cached range per step is O(cached rows) where the write is O(new rows). The cost is
//! host memory at binary32 width; the narrow width lives on the device.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::{FormatCodec, NumericFormat};

use super::residency_caches::{DenseAttentionCache, LayerCache};
use crate::error::InteropError;
use crate::kv_cache_format;

const PLANES: usize = 4;

/// Per-call rounding of the rows the host appends: resolved once per decode call, scratch reused
/// across steps so the steady state allocates nothing once the widest step has been seen.
pub(super) struct HostCacheRounding {
    key: Option<(FormatCodec, String)>,
    value: Option<(FormatCodec, String)>,
    packed: Vec<u8>,
    staged: [Vec<f32>; PLANES],
}

fn side(format: &NumericFormat) -> Result<Option<(FormatCodec, String)>, InteropError> {
    if kv_cache_format::is_binary32(format) {
        return Ok(None);
    }
    match FormatCodec::for_format(format) {
        Some(codec) if codec.can_encode() => Ok(Some((codec, alloc::format!("{format:?}")))),
        _ => Err(InteropError::UnsupportedServingConfig(alloc::format!(
            "kv cache format {format:?} has no slice codec the host can round rows through"
        ))),
    }
}

fn round_plane(
    side: &(FormatCodec, String),
    packed: &mut Vec<u8>,
    staged: &mut Vec<f32>,
    values: &[f32],
) -> Result<(), InteropError> {
    let (codec, name) = side;
    let refuse = |reason: String| {
        InteropError::UnsupportedServingConfig(alloc::format!("kv cache format {name}: {reason}"))
    };
    let bytes = codec
        .packed_len(values.len())
        .ok_or_else(|| refuse(alloc::format!("{} elements are not whole blocks", values.len())))?;
    packed.clear();
    packed.resize(bytes, 0);
    codec
        .encode(values, packed)
        .map_err(|error| refuse(alloc::format!("{error}")))?;
    staged.clear();
    staged.reserve(values.len());
    let slots = &mut staged.spare_capacity_mut()[..values.len()];
    codec
        .decode(packed, slots)
        .map_err(|error| refuse(alloc::format!("{error}")))?;
    // SAFETY: decode returned Ok, which initialises every slot of its output, and the first
    // `values.len()` slots are the ones it was handed.
    unsafe { staged.set_len(values.len()) };
    Ok(())
}

impl HostCacheRounding {
    /// # Errors
    /// [`InteropError::UnsupportedServingConfig`] when a non-binary32 side has no encoding codec.
    pub(super) fn new(key: &NumericFormat, value: &NumericFormat) -> Result<Self, InteropError> {
        Ok(Self {
            key: side(key)?,
            value: side(value)?,
            packed: Vec::new(),
            staged: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        })
    }

    /// Whether any side rounds; `false` is the binary32 path, which appends the evaluated rows as-is.
    pub(super) fn is_active(&self) -> bool {
        self.key.is_some() || self.value.is_some()
    }

    fn stage(&mut self, planes: &[(&[f32], bool)]) -> Result<[bool; PLANES], InteropError> {
        let mut staged_flags = [false; PLANES];
        for (slot, (values, is_key)) in planes.iter().enumerate() {
            let side = if *is_key { &self.key } else { &self.value };
            if let Some(side) = side {
                round_plane(side, &mut self.packed, &mut self.staged[slot], values)?;
                staged_flags[slot] = true;
            }
        }
        Ok(staged_flags)
    }

    /// [`LayerCache::append_at`] with the rows rounded through the cache's formats.
    ///
    /// # Errors
    /// [`InteropError::UnsupportedServingConfig`] when a row is not whole blocks of its format.
    pub(super) fn append_attention(
        &mut self,
        cache: &mut LayerCache,
        start_position: usize,
        (even, odd, value): (&[f32], &[f32], &[f32]),
    ) -> Result<(), InteropError> {
        if !self.is_active() {
            cache.append_at(start_position, even, odd, value);
            return Ok(());
        }
        let used = self.stage(&[(even, true), (odd, true), (value, false)])?;
        let [first, second, third, _] = &self.staged;
        cache.append_at(
            start_position,
            if used[0] { first } else { even },
            if used[1] { second } else { odd },
            if used[2] { third } else { value },
        );
        Ok(())
    }

    /// [`DenseAttentionCache::append`] with the rows rounded through the cache's formats.
    ///
    /// # Errors
    /// [`InteropError::UnsupportedServingConfig`] when a row is not whole blocks of its format.
    pub(super) fn append_dense(
        &mut self,
        cache: &mut DenseAttentionCache,
        (first, second, pass, value): (&[f32], &[f32], &[f32], &[f32]),
    ) -> Result<(), InteropError> {
        if !self.is_active() {
            cache.append(first, second, pass, value);
            return Ok(());
        }
        let used = self.stage(&[(first, true), (second, true), (pass, true), (value, false)])?;
        let [staged_first, staged_second, staged_pass, staged_value] = &self.staged;
        cache.append(
            if used[0] { staged_first } else { first },
            if used[1] { staged_second } else { second },
            if used[2] { staged_pass } else { pass },
            if used[3] { staged_value } else { value },
        );
        Ok(())
    }
}

#[cfg(test)]
// workspace denies expect_used; a failed fixture must abort the test with its message
#[allow(clippy::expect_used)]
mod tests {
    use half::{bf16, f16};
    use proxima_tensor::physical::{
        AttentionExtents, CachedAttentionRotary, CachedAttentionScore, CausalBand,
        stream_cached_attention_split_gqa,
    };

    use super::*;

    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 32;
    const PAIR_DIM: usize = HEAD_DIM / 2;
    const CACHED_ROWS: usize = 3;
    const EVEN_ODD_ROW: usize = KV_HEADS * PAIR_DIM;
    const VALUE_ROW: usize = KV_HEADS * HEAD_DIM;

    fn activations(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as f32 + seed) * 0.37).sin() * 3.1 + ((index as f32) * 0.011).cos())
            .collect()
    }

    fn round_each(values: &[f32], round: impl Fn(f32) -> f32) -> Vec<f32> {
        values.iter().map(|value| round(*value)).collect()
    }

    fn rounded_through_codec(format: &NumericFormat, values: &[f32]) -> Vec<f32> {
        let codec = FormatCodec::for_format(format).expect("the format resolves a codec");
        let mut packed = alloc::vec![0u8; codec.packed_len(values.len()).expect("whole blocks")];
        codec.encode(values, &mut packed).expect("encodes");
        let mut slots = alloc::vec![core::mem::MaybeUninit::<f32>::uninit(); values.len()];
        codec.decode(&packed, &mut slots).expect("decodes");
        // SAFETY: decode returned Ok, which initialises every slot of its output
        slots.iter().map(|slot| unsafe { slot.assume_init() }).collect()
    }

    struct Rows {
        even: Vec<f32>,
        odd: Vec<f32>,
        value: Vec<f32>,
    }

    fn cached_rows() -> Rows {
        Rows {
            even: activations(CACHED_ROWS * EVEN_ODD_ROW, 1.0),
            odd: activations(CACHED_ROWS * EVEN_ODD_ROW, 2.0),
            value: activations(CACHED_ROWS * VALUE_ROW, 3.0),
        }
    }

    fn one_step(cache: &LayerCache) -> Vec<f32> {
        let query_even = activations(EVEN_ODD_ROW, 4.0);
        let query_odd = activations(EVEN_ODD_ROW, 5.0);
        let new_even = activations(EVEN_ODD_ROW, 6.0);
        let new_odd = activations(EVEN_ODD_ROW, 7.0);
        let new_value = activations(VALUE_ROW, 8.0);
        let mut output = alloc::vec![0.0f32; VALUE_ROW];
        let streamed = stream_cached_attention_split_gqa(
            [&query_even, &query_odd],
            [[&cache.k_even, &cache.k_odd], [&new_even, &new_odd]],
            [&cache.v, &new_value],
            &mut output,
            AttentionExtents {
                query_rows: 1,
                cached_key_rows: CACHED_ROWS as u64,
                new_key_rows: 1,
                kv_heads: KV_HEADS as u64,
                query_groups: 1,
                head_dim: HEAD_DIM as u64,
            },
            CachedAttentionRotary { rotary_dim: HEAD_DIM as u64, pass: None },
            CachedAttentionScore {
                sinks: None,
                scale: 0.25,
                bands: [
                    CausalBand { lower_inclusive: i64::MIN, upper_inclusive: i64::MAX },
                    CausalBand { lower_inclusive: 0, upper_inclusive: 0 },
                ],
            },
        );
        assert!(streamed, "the attention step accepts its extents");
        output
    }

    fn appended(rounding: &mut HostCacheRounding, rows: &Rows) -> LayerCache {
        let mut cache = LayerCache::new();
        rounding
            .append_attention(&mut cache, 0, (&rows.even, &rows.odd, &rows.value))
            .expect("rows append");
        cache
    }

    fn assert_step_matches_prerounded_binary32(
        format: &NumericFormat,
        round: impl Fn(&[f32]) -> Vec<f32>,
    ) {
        let rows = cached_rows();
        let narrow = appended(&mut HostCacheRounding::new(format, format).expect("codec"), &rows);
        let prerounded = Rows {
            even: round(&rows.even),
            odd: round(&rows.odd),
            value: round(&rows.value),
        };
        let reference = appended(
            &mut HostCacheRounding::new(&NumericFormat::Float32, &NumericFormat::Float32).expect("binary32"),
            &prerounded,
        );

        let narrow_step = one_step(&narrow);
        let reference_step = one_step(&reference);

        let differing = narrow_step
            .iter()
            .zip(&reference_step)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        assert_eq!(differing, 0, "{format:?}: attention over the narrow cache equals the prerounded binary32 cache");
        assert_ne!(narrow.k_even, rows.even, "{format:?}: the rows were really rounded, not passed through");
        assert_ne!(one_step(&appended(&mut HostCacheRounding::new(&NumericFormat::Float32, &NumericFormat::Float32).expect("binary32"), &rows)), narrow_step, "{format:?}: rounding moves the step");
    }

    #[test]
    fn binary16_cache_attention_equals_binary32_cache_of_binary16_rounded_values() {
        assert_step_matches_prerounded_binary32(&NumericFormat::Float16, |values| {
            round_each(values, |value| f16::from_f32(value).to_f32())
        });
    }

    #[test]
    fn bfloat16_cache_attention_equals_binary32_cache_of_bfloat16_rounded_values() {
        assert_step_matches_prerounded_binary32(&NumericFormat::BFloat16, |values| {
            round_each(values, |value| bf16::from_f32(value).to_f32())
        });
    }

    #[test]
    fn q8_0_cache_attention_equals_binary32_cache_of_q8_0_rounded_values() {
        assert_step_matches_prerounded_binary32(&NumericFormat::Q8_0, |values| {
            rounded_through_codec(&NumericFormat::Q8_0, values)
        });
    }

    #[test]
    fn a_binary32_cache_appends_the_evaluated_rows_untouched() {
        let rows = cached_rows();
        let rounding = HostCacheRounding::new(&NumericFormat::Float32, &NumericFormat::Float32).expect("binary32");
        let cache = appended(&mut { rounding }, &rows);

        assert_eq!(cache.k_even, rows.even);
        assert_eq!(cache.v, rows.value);
    }

    #[test]
    fn a_row_that_is_not_whole_blocks_is_refused_naming_the_format() {
        let mut rounding = HostCacheRounding::new(&NumericFormat::Q8_0, &NumericFormat::Q8_0).expect("codec");
        let mut cache = LayerCache::new();
        let ragged = activations(31, 1.0);

        let error = rounding
            .append_attention(&mut cache, 0, (&ragged, &ragged, &ragged))
            .expect_err("31 elements are not a Q8_0 block");

        assert!(alloc::format!("{error}").contains("Q8_0"), "{error}");
    }
}
