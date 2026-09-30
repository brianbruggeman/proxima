//! The sliding-window  behind a sliding layer's host KV cache.
//!
//! A sliding-window layer attends to the most recent `window` positions and
//! masks every older one, so rows older than the window never contribute.
//! [`KvRing`] stores only the rows that can: position `p` lives in row
//! `p % capacity`, and a read hands the program the most recent
//! `min(cached_len, window)` rows in chronological order
//! ([`LayerCache::unroll_live_rows`]), counted by the program's
//! `cached_len_swa` input ([`proxima_tensor::spec::SLIDING_CACHED_LEN_INPUT`]).
//! Dropping the evicted prefix does not move the query-to-key distance the
//! mask reads, which is why the ring is exact rather than approximate.
//!
//! This composes [`LayerCache`], the growing host cache every attention layer
//! already uses: a ring layer is a `LayerCache` whose `ring` field is set, so
//! full layers, prefix reuse ([`PrefixState`]) and the pad scratch
//! ([`KvPadScratch`]) keep one cache type. Reach for [`LayerCache::append`]
//! directly only for a full layer; [`LayerCache::append_at`] is the one write
//! that is right for both.

use proxima_tokenizer::draft::DEFAULT_N_DRAFT;

use super::*;
use crate::serving::{SpeculativeConfig, SpeculativeType};

/// Geometry of one sliding layer's ring.
///
/// `capacity >= window`: the surplus is the speculative-decode slack. A
/// verify step writes its drafted positions before it knows how many survive,
/// and a rewind must find the rows those writes would otherwise have
/// evicted, so the ring holds `window + draft_len` rows when speculation is
/// on and exactly `window` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvRing {
    pub(super) window: usize,
    pub(super) capacity: usize,
    pub(super) even_odd_row: usize,
    pub(super) v_row: usize,
    /// Rows the write position is displaced by. Zero in every real run; the
    /// ring-parity control sets it to prove the read really depends on the
    /// row mapping (`gemma4_ring_parity --ring-offset`).
    pub(super) write_offset: usize,
}

impl KvRing {
    pub(super) fn new(
        window: usize,
        slack: usize,
        even_odd_row: usize,
        v_row: usize,
        write_offset: usize,
    ) -> Self {
        Self {
            window,
            capacity: window + slack,
            even_odd_row,
            v_row,
            write_offset,
        }
    }

    /// Rows allocated for a call that reaches `positions_needed` positions:
    /// never more than the ring holds.
    pub(super) fn rows_for(&self, positions_needed: usize) -> usize {
        positions_needed.min(self.capacity)
    }

    /// Rows a read returns at `cached_len` cached positions.
    pub(super) fn live_rows(&self, cached_len: usize) -> usize {
        cached_len.min(self.window)
    }

    /// The program's sliding extent for a step whose full-attention extent is
    /// `kv_bound_extent`: bucketed like the full layers, capped at the
    /// window.
    pub(super) fn bound_extent(&self, kv_bound_extent: usize) -> usize {
        kv_bound_extent.min(self.window)
    }

    fn write_row(&self, position: usize) -> usize {
        (position + self.write_offset) % self.capacity
    }

    fn read_row(&self, position: usize) -> usize {
        position % self.capacity
    }
}

impl LayerCache {
    /// A ring cache with rows for `positions_needed` positions allocated up
    /// front (zeroed; a row is only ever read after it was written).
    pub(super) fn ring(ring: KvRing, positions_needed: usize) -> Self {
        let mut cache = Self::new();
        cache.ring = Some(ring);
        cache.reserve_ring_rows(positions_needed);
        cache
    }

    pub(super) fn ring_geometry(&self) -> Option<&KvRing> {
        self.ring.as_ref()
    }

    /// Grows a ring cache to hold rows for `positions_needed` positions.
    /// Never shrinks, and never grows past the ring's capacity: a cache
    /// carried across calls ([`PrefixState`]) reaches further on the next one.
    pub(super) fn reserve_ring_rows(&mut self, positions_needed: usize) {
        let Some(ring) = self.ring else {
            return;
        };
        let rows = ring.rows_for(positions_needed);
        self.k_even
            .resize(self.k_even.len().max(rows * ring.even_odd_row), 0.0);
        self.k_odd
            .resize(self.k_odd.len().max(rows * ring.even_odd_row), 0.0);
        self.v.resize(self.v.len().max(rows * ring.v_row), 0.0);
    }

    /// [`Self::append`] for a block whose first row is absolute position
    /// `start_position`. A full layer extends; a ring layer writes each row to
    /// its ring slot, skipping the leading rows of a block longer than the
    /// ring because the later rows overwrite them.
    pub(super) fn append_at(
        &mut self,
        start_position: usize,
        even: &[f32],
        odd: &[f32],
        value: &[f32],
    ) {
        let Some(ring) = self.ring else {
            self.append(even, odd, value);
            return;
        };
        let rows = even.len() / ring.even_odd_row.max(1);
        let skipped = rows.saturating_sub(ring.capacity);
        for row in skipped..rows {
            let slot = ring.write_row(start_position + row);
            copy_row(&mut self.k_even, slot, ring.even_odd_row, even, row);
            copy_row(&mut self.k_odd, slot, ring.even_odd_row, odd, row);
            copy_row(&mut self.v, slot, ring.v_row, value, row);
        }
    }

    /// Copies the `min(cached_len, window)` most recent rows, oldest first,
    /// into the front of the three scratch buffers and returns that count.
    ///
    /// # Errors
    ///
    /// [`InteropError::CacheScratchShapeMismatch`] when a scratch buffer is
    /// shorter than the live rows need, which means the bound extent the
    /// caller sized it to disagrees with the ring's window.
    pub(super) fn unroll_live_rows(
        &self,
        cached_len: usize,
        even_scratch: &mut [f32],
        odd_scratch: &mut [f32],
        value_scratch: &mut [f32],
        layer: usize,
    ) -> Result<usize, InteropError> {
        let Some(ring) = self.ring else {
            return Ok(0);
        };
        let live = ring.live_rows(cached_len);
        let first = cached_len - live;
        for (leaf, scratch, source, width) in [
            (
                "k_even",
                &mut *even_scratch,
                &self.k_even,
                ring.even_odd_row,
            ),
            ("k_odd", &mut *odd_scratch, &self.k_odd, ring.even_odd_row),
            ("v", &mut *value_scratch, &self.v, ring.v_row),
        ] {
            if scratch.len() < live * width {
                return Err(InteropError::CacheScratchShapeMismatch {
                    layer,
                    leaf,
                    expected: scratch.len(),
                    found: live * width,
                });
            }
            for offset in 0..live {
                let slot = ring.read_row(first + offset);
                scratch[offset * width..(offset + 1) * width]
                    .copy_from_slice(&source[slot * width..(slot + 1) * width]);
            }
        }
        Ok(live)
    }
}

/// One attention layer's cache: a ring of `window` rows plus `slack` when
/// `window` is `Some`, a growing full cache otherwise. The one constructor
/// `LoadedModel::attention_layer_cache` and the allocation test share.
pub(super) fn attention_cache(
    window: Option<usize>,
    slack: usize,
    even_odd_row: usize,
    v_row: usize,
    write_offset: usize,
    positions_needed: usize,
) -> LayerCache {
    match window {
        Some(window) => LayerCache::ring(
            KvRing::new(window, slack, even_odd_row, v_row, write_offset),
            positions_needed,
        ),
        None => LayerCache::new(),
    }
}

/// The most rows one speculative verify step can write past the last accepted
/// position: the widest draft any enabled drafter emits, or `forced_draft_width`
/// (`speculative_bench`'s verify-width sweep) when that is wider. `None` when
/// nothing speculates. One reader, so the ring's slack, the memory budget and
/// the decode loop's speculation gate cannot disagree.
///
/// Each bound is the drafter's own copy cap: ngram-simple
/// (`proxima-tokenizer/src/draft/ngram_simple.rs:174`), ngram-map-k and
/// ngram-map-k4v (`draft/ngram_map.rs:449`, `:527`) cap at `size_m`;
/// ngram-mod loops `0..n_max` (`draft/ngram_mod.rs:283`); ngram-cache stops at
/// `DEFAULT_N_DRAFT` (`draft/ngram_cache.rs:577`, passed at `generate/drafter.rs:83`).
pub(super) fn speculative_draft_limit(
    config: &SpeculativeConfig<'_>,
    forced_draft_width: Option<u16>,
) -> Option<usize> {
    let enabled = config
        .speculative_types
        .iter_priority_order()
        .filter_map(|drafter| match drafter {
            SpeculativeType::NgramSimple => Some(config.ngram_simple.size_m),
            SpeculativeType::NgramMapK => Some(config.ngram_map_k.size_m),
            SpeculativeType::NgramMapK4v => Some(config.ngram_map_k4v.size_m),
            SpeculativeType::NgramMod => Some(config.ngram_mod.n_max),
            SpeculativeType::NgramCache => Some(DEFAULT_N_DRAFT),
            _ => None,
        });
    enabled.chain(forced_draft_width).max().map(usize::from)
}

/// The ring geometry the sliding layers of `layer_caches` share, `None` when
/// no layer is a ring. A program binds one sliding slot, so every ring layer
/// has the same window.
pub(super) fn sliding_ring_geometry(layer_caches: &[LayerCacheState]) -> Option<KvRing> {
    layer_caches.iter().find_map(|state| match state {
        LayerCacheState::Attention(cache) => cache.ring_geometry().copied(),
        _ => None,
    })
}

/// The `cached_len_swa` scalar for a step at `cached_len` cached positions:
/// the rows the ring hands the program, `0` when there is no ring.
pub(super) fn sliding_cached_len_scalar(
    layer_caches: &[LayerCacheState],
    cached_len: usize,
) -> [f32; 1] {
    [sliding_ring_geometry(layer_caches).map_or(0, |ring| ring.live_rows(cached_len)) as f32]
}

/// Whether every ring in `layer_caches` holds `draft_limit` rows of slack, the
/// condition under which a speculative verify step's rewind is exact. A cache
/// carried in from a call that ran without speculation has none.
pub(super) fn rings_cover_speculation(
    layer_caches: &[LayerCacheState],
    draft_limit: usize,
) -> bool {
    layer_caches.iter().all(|state| match state {
        LayerCacheState::Attention(cache) => cache
            .ring_geometry()
            .is_none_or(|ring| ring.capacity - ring.window >= draft_limit),
        _ => true,
    })
}

fn copy_row(destination: &mut Vec<f32>, slot: usize, width: usize, source: &[f32], row: usize) {
    let end = (slot + 1) * width;
    if destination.len() < end {
        destination.resize(end, 0.0);
    }
    destination[slot * width..end].copy_from_slice(&source[row * width..(row + 1) * width]);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gemma4::GEMMA4;
    use crate::memory_fit::{MemoryBudget, WeightClassBytes};
    use crate::serving::{NgramModParams, SpeculativeTypeSet};
    use crate::test_support::gemma4_e2b_header;

    const EVEN_ODD_ROW: usize = 2;
    const V_ROW: usize = 3;
    const WINDOW: usize = 4;

    fn ring_cache(slack: usize, write_offset: usize, positions_needed: usize) -> LayerCache {
        LayerCache::ring(
            KvRing::new(WINDOW, slack, EVEN_ODD_ROW, V_ROW, write_offset),
            positions_needed,
        )
    }

    /// Position `position`'s three leaves, each element a distinct value so a
    /// misplaced row cannot pass by coincidence.
    fn allocated_rows(cache: &LayerCache) -> usize {
        cache.k_even.len() / EVEN_ODD_ROW
    }

    fn position_rows(position: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let base = position as f32 * 100.0;
        (
            (0..EVEN_ODD_ROW)
                .map(|column| base + column as f32)
                .collect(),
            (0..EVEN_ODD_ROW)
                .map(|column| base + 10.0 + column as f32)
                .collect(),
            (0..V_ROW)
                .map(|column| base + 20.0 + column as f32)
                .collect(),
        )
    }

    fn append_positions(cache: &mut LayerCache, positions: core::ops::Range<usize>) {
        let start = positions.start;
        let (mut even, mut odd, mut value) = (Vec::new(), Vec::new(), Vec::new());
        for position in positions {
            let (row_even, row_odd, row_value) = position_rows(position);
            even.extend(row_even);
            odd.extend(row_odd);
            value.extend(row_value);
        }
        cache.append_at(start, &even, &odd, &value);
    }

    fn unrolled(cache: &LayerCache, cached_len: usize) -> (usize, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut even = vec![0.0; WINDOW * EVEN_ODD_ROW];
        let mut odd = vec![0.0; WINDOW * EVEN_ODD_ROW];
        let mut value = vec![0.0; WINDOW * V_ROW];
        let live = cache
            .unroll_live_rows(cached_len, &mut even, &mut odd, &mut value, 0)
            .expect("the scratch buffers hold a full window");
        (live, even, odd, value)
    }

    fn expected_rows(first: usize, live: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut even, mut odd, mut value) = (Vec::new(), Vec::new(), Vec::new());
        for position in first..first + live {
            let (row_even, row_odd, row_value) = position_rows(position);
            even.extend(row_even);
            odd.extend(row_odd);
            value.extend(row_value);
        }
        (even, odd, value)
    }

    /// Ten positions written one at a time into a four-row ring: the read
    /// returns positions 6..10 oldest first, which is what a full cache's
    /// last `window` rows hold.
    #[test]
    fn one_token_appends_read_back_the_last_window_rows_in_order() {
        let mut cache = ring_cache(0, 0, 10);
        for position in 0..10 {
            append_positions(&mut cache, position..position + 1);
        }

        let (live, even, odd, value) = unrolled(&cache, 10);
        let (want_even, want_odd, want_value) = expected_rows(6, 4);

        assert_eq!(live, 4);
        assert_eq!(&even[..live * EVEN_ODD_ROW], want_even.as_slice());
        assert_eq!(&odd[..live * EVEN_ODD_ROW], want_odd.as_slice());
        assert_eq!(&value[..live * V_ROW], want_value.as_slice());
    }

    /// A prefill block longer than the ring keeps only its last `window`
    /// rows, the same rows one-token appends would have left.
    #[test]
    fn a_block_longer_than_the_ring_keeps_its_last_window_rows() {
        let mut cache = ring_cache(0, 0, 10);
        append_positions(&mut cache, 0..10);

        let (live, even, _odd, _value) = unrolled(&cache, 10);
        let (want_even, _, _) = expected_rows(6, 4);

        assert_eq!(live, 4);
        assert_eq!(&even[..live * EVEN_ODD_ROW], want_even.as_slice());
    }

    /// Before the window fills, the read is the whole history.
    #[test]
    fn before_the_window_fills_every_cached_row_is_read() {
        let mut cache = ring_cache(0, 0, 10);
        append_positions(&mut cache, 0..3);

        let (live, even, _odd, _value) = unrolled(&cache, 3);
        let (want_even, _, _) = expected_rows(0, 3);

        assert_eq!(live, 3);
        assert_eq!(&even[..live * EVEN_ODD_ROW], want_even.as_slice());
    }

    /// Allocation is `min(positions_needed, capacity)` rows, and a cache
    /// carried into a longer call grows to the new need but never past the
    /// ring.
    #[test]
    fn allocation_is_the_smaller_of_positions_needed_and_capacity() {
        let mut cache = ring_cache(0, 0, 3);
        assert_eq!(allocated_rows(&cache), 3);

        cache.reserve_ring_rows(100);
        assert_eq!(allocated_rows(&cache), WINDOW);

        cache.reserve_ring_rows(2);
        assert_eq!(allocated_rows(&cache), WINDOW, "a ring cache never shrinks");
    }

    /// The rows a query at `cached_len` can attend to are the newest
    /// `window - 1` of the `window` rows a read returns; the oldest read row is
    /// at exactly `window` distance and the mask hides it.
    fn attended_rows(rows: &[f32], live: usize) -> &[f32] {
        &rows[EVEN_ODD_ROW..live * EVEN_ODD_ROW]
    }

    /// Slack widens the ring, not the read: a speculative verify that writes
    /// `slack + 1` rows past the committed position and is then rewound (those
    /// rows are simply never read, `cached_len` going back) leaves every row
    /// the next query attends to intact.
    #[test]
    fn slack_keeps_the_attended_rows_intact_after_a_rewound_speculative_write() {
        let slack = 3;
        let mut cache = ring_cache(slack, 0, 20);
        append_positions(&mut cache, 0..8);
        append_positions(&mut cache, 8..8 + slack + 1);

        let (live, even, _odd, _value) = unrolled(&cache, 8);
        let (want_even, _, _) = expected_rows(4, 4);

        assert_eq!(live, 4);
        assert_eq!(attended_rows(&even, live), attended_rows(&want_even, live));
    }

    /// Control: without slack the same speculative write evicts rows the
    /// rewound query attends to, so the read no longer matches.
    #[test]
    fn without_slack_a_rewound_speculative_write_corrupts_the_attended_rows() {
        let mut cache = ring_cache(0, 0, 20);
        append_positions(&mut cache, 0..8);
        append_positions(&mut cache, 8..12);

        let (live, even, _odd, _value) = unrolled(&cache, 8);
        let (want_even, _, _) = expected_rows(4, 4);

        assert_eq!(live, 4);
        assert_ne!(attended_rows(&even, live), attended_rows(&want_even, live));
    }

    /// The parity control's mechanism: writing every row one slot off while
    /// reading by the true mapping returns the wrong rows.
    #[test]
    fn a_write_offset_breaks_the_read_mapping() {
        let mut cache = ring_cache(0, 1, 10);
        append_positions(&mut cache, 0..10);

        let (live, even, _odd, _value) = unrolled(&cache, 10);
        let (want_even, _, _) = expected_rows(6, 4);

        assert_eq!(live, 4);
        assert_ne!(&even[..live * EVEN_ODD_ROW], want_even.as_slice());
    }

    /// gemma4 E2B at 2,048 positions: the 12 sliding layers hold the
    /// checkpoint's own `attention.sliding_window` rows (read from the
    /// header, not assumed), the 3 full layers all 2,048. The 20 shared-KV
    /// layers own no cache, so `kv_layers` lists 15.
    #[test]
    fn gemma4_ring_rows() {
        let parsed = gemma4_e2b_header();
        let layers = GEMMA4
            .kv_layers(&parsed)
            .expect("the e2b header carries every key kv_layers reads");
        let window_from_header = layers
            .iter()
            .find_map(|&(_, _, window)| window)
            .expect("the e2b header has sliding layers") as usize;
        let positions_needed = 2048;

        let rows: Vec<(Option<u32>, usize)> = layers
            .iter()
            .map(|&(kv_heads, head_dim, window)| {
                let even_odd_row = (kv_heads * head_dim / 2) as usize;
                let value_row = (kv_heads * head_dim) as usize;
                let mut cache = attention_cache(
                    window.map(|width| width as usize),
                    0,
                    even_odd_row,
                    value_row,
                    0,
                    positions_needed,
                );
                cache.append_at(
                    0,
                    &vec![0.5; positions_needed * even_odd_row],
                    &vec![0.5; positions_needed * even_odd_row],
                    &vec![0.5; positions_needed * value_row],
                );
                (window, cache.k_even.len() / even_odd_row)
            })
            .collect();

        let sliding: Vec<usize> = rows
            .iter()
            .filter(|(window, _)| window.is_some())
            .map(|&(_, held)| held)
            .collect();
        let full: Vec<usize> = rows
            .iter()
            .filter(|(window, _)| window.is_none())
            .map(|&(_, held)| held)
            .collect();
        assert_eq!(window_from_header, 512);
        assert_eq!(sliding, vec![window_from_header; 12]);
        assert_eq!(full, vec![positions_needed; 3]);
    }

    /// The default config enables ngram-simple at `size_m = 48`; the ring
    /// holds that many rows past the 512-row window on each of the 12 sliding
    /// layers, 12 x 2048 B x 48 = 1,179,648 B over the slack-free budget.
    #[test]
    fn memory_budget_gemma4_ring_draft_slack() {
        const CONTEXT: u32 = 131_072;
        let parsed = gemma4_e2b_header();
        let layers = GEMMA4
            .kv_layers(&parsed)
            .expect("the e2b header carries every key kv_layers reads");
        let budget_for = |config: &SpeculativeConfig<'_>| {
            let slack = speculative_draft_limit(config, None).map_or(0, |limit| limit as u32);
            MemoryBudget::derive(WeightClassBytes::default(), &layers, CONTEXT, slack, 0)
                .kv_cache_bytes
        };

        assert_eq!(budget_for(&SpeculativeConfig::default()), 1_624_375_296);
        assert_eq!(budget_for(&SpeculativeConfig::none()), 1_623_195_648);
    }

    #[test]
    fn draft_limit_is_the_widest_enabled_drafters_bound() {
        let config = SpeculativeConfig {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple)
                .insert(SpeculativeType::NgramMod),
            ngram_mod: NgramModParams {
                n_match: 24,
                n_max: 64,
                n_min: 48,
            },
            ..SpeculativeConfig::none()
        };
        let cache_only = SpeculativeConfig {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramCache),
            ..SpeculativeConfig::none()
        };

        assert_eq!(speculative_draft_limit(&config, None), Some(64));
        assert_eq!(speculative_draft_limit(&cache_only, None), Some(8));
        assert_eq!(speculative_draft_limit(&cache_only, Some(100)), Some(100));
    }

    #[test]
    fn draft_limit_is_none_when_nothing_speculates() {
        let off = SpeculativeConfig::none();

        assert_eq!(speculative_draft_limit(&off, None), None);
        assert_eq!(speculative_draft_limit(&off, Some(6)), Some(6));
    }

    /// A scratch buffer sized for fewer rows than the live window is a typed
    /// error, not a silent truncation.
    #[test]
    fn a_scratch_shorter_than_the_live_rows_is_a_shape_mismatch() {
        let mut cache = ring_cache(0, 0, 10);
        append_positions(&mut cache, 0..10);

        let mut even = vec![0.0; EVEN_ODD_ROW];
        let mut odd = vec![0.0; WINDOW * EVEN_ODD_ROW];
        let mut value = vec![0.0; WINDOW * V_ROW];
        let result = cache.unroll_live_rows(10, &mut even, &mut odd, &mut value, 7);

        assert!(matches!(
            result,
            Err(InteropError::CacheScratchShapeMismatch {
                layer: 7,
                leaf: "k_even",
                ..
            })
        ));
    }
}
