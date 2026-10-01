//! The per-model prompt cache: a bounded, least-recently-used set of
//! [`PrefixState`]s keyed by the token ids they cover, so a request whose
//! prompt shares a prefix with an earlier one prefills only the tokens past
//! the shared part (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R1, R2,
//! R8).
//!
//! This composes [`PrefixState`] (the `(ids, layer_caches, cached_len)` triple
//! [`LoadedModel::prefill_prefix`] already returns) and the seeded decode loop
//! [`LoadedModel::run_decode_loop_observed_seeded`] already resumes from; it
//! adds only the matching and the bookkeeping. Reach for
//! [`LoadedModel::prefill_prefix`] + [`LoadedModel::generate_from_prefix`]
//! when the caller owns the one state and knows the split point; reach for
//! this cache, enabled through [`crate::ServingConfig::prompt_cache`], when
//! requests arrive whole and the shared prefix has to be found.
//!
//! Matching is on token ids, never text. llama-server (`tools/server/
//! server-context.cpp`, upstream `f1ea20621`) keeps `slot.prompt.tokens` as
//! the prompt AND the tokens generated from it and matches the next prompt
//! against that whole sequence (`server-common.cpp:697-709`); the entry stored
//! here is the same sequence ([`PrefixState::ids`] is the prompt plus every
//! generated token the model has forward-passed).
//!
//! The lock is held only to take an entry out, or put one back, never across
//! a decode: a request owns its taken state for the whole generation, so a
//! second request on the same model sees no entry and prefills in full rather
//! than waiting. [`super::prewarm`] fills the same entries ahead of a request
//! through the same lookup, and [`super::prewarm_gate`] decides who holds the
//! device when a request and a prewarm meet.

use core::ops::{ControlFlow, Range};
use std::collections::BTreeMap;
use std::sync::PoisonError;
use std::time::Duration;

use proxima_telemetry::debug;

use super::block_index::{BlockBloom, BlockIndex, content_hashes};
use super::*;

/// Length of the shared token prefix of `left` and `right`.
pub(super) fn longest_common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left_id, right_id)| left_id == right_id)
        .count()
}

/// Whether an entry whose stored sequence shares `lcp` tokens with a prompt
/// of `prompt_len` tokens may be reused. An entry the prompt extends whole
/// (`lcp == stored_len`) always may: nothing is rewound, so nothing is lost.
/// Otherwise the shared prefix must cover more than `min_similarity_milli`
/// thousandths of the prompt, llama-server's `f_sim_cur > slot_prompt_similarity`
/// with `f_sim_cur = lcp / prompt_len` (`server-context.cpp:1563-1571`);
/// `0` accepts any overlap past the first token.
const fn entry_is_reusable(
    lcp: usize,
    stored_len: usize,
    prompt_len: usize,
    min_similarity_milli: u32,
) -> bool {
    lcp > 0 && (lcp == stored_len || lcp * 1000 > min_similarity_milli as usize * prompt_len)
}

/// How a request used the cache (spec R8's `cache_path`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePath {
    /// The whole stored sequence is a prefix of the prompt: nothing rewound.
    Extend,
    /// The prompt diverges inside the stored sequence: every layer was
    /// rewound to the shared prefix first.
    Rewind,
    /// The prompt diverges further back than the ring layers can rewind: a
    /// checkpoint at or before the shared prefix restored their rows, and the
    /// request prefilled from the checkpoint onward.
    Checkpoint,
    /// No entry could be reused; the whole prompt was prefilled.
    Miss,
}

impl CachePath {
    /// The telemetry label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extend => "extend",
            Self::Rewind => "rewind",
            Self::Checkpoint => "checkpoint",
            Self::Miss => "miss",
        }
    }
}

/// Why a request found nothing to reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    /// The cache holds no entry.
    Empty,
    /// Every entry was built under a different [`CacheKey`]: its rows are not
    /// the rows this request would compute.
    ConfigMismatch,
    /// No entry shares even the first token with the prompt.
    NoCommonPrefix,
    /// Entries share a prefix with the prompt, but none covers enough of it
    /// ([`crate::PromptCacheConfig::min_similarity_milli`]) and none is
    /// extended whole: reusing one would rewind it to a few rows and destroy
    /// what it caches, so the request builds an entry of its own and every
    /// older one stays.
    BelowSimilarity,
    /// A sliding-window layer would need rows its ring has already
    /// overwritten: rewinding `rewind_rows` tokens exceeds the `slack_rows`
    /// the ring keeps past its window.
    RingSlackExceeded {
        /// Tokens the rewind would drop from the stored sequence.
        rewind_rows: usize,
        /// Rows the ring keeps past its window.
        slack_rows: usize,
    },
    /// The entry's ring rows before `restored_at - window` belong to a state a
    /// checkpoint replaced, so a rewind to before `restored_at` would read
    /// them.
    RingRowsStale {
        /// Position of the checkpoint the entry was last restored to.
        restored_at: usize,
    },
    /// A layer holds state that cannot be rewound (a recurrent layer, or a
    /// cache shape this build does not truncate).
    UnrewindableLayer,
}

impl MissReason {
    /// The telemetry label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::ConfigMismatch => "config_mismatch",
            Self::NoCommonPrefix => "no_common_prefix",
            Self::BelowSimilarity => "below_similarity",
            Self::RingSlackExceeded { .. } => "ring_slack_exceeded",
            Self::RingRowsStale { .. } => "ring_rows_stale",
            Self::UnrewindableLayer => "unrewindable_layer",
        }
    }
}

/// What one request took from the cache: the numbers behind spec R8's
/// `cache_lcp`, `cache_reused_tokens`, `cache_prefilled_tokens`,
/// `cache_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReport {
    /// Tokens the prompt shares with the entry it matched (capped one short
    /// of the prompt, so the last token is always forwarded).
    pub lcp: usize,
    /// Tokens served from the cache without a forward pass: `lcp`, or the
    /// position of the checkpoint restored when `path` is
    /// [`CachePath::Checkpoint`].
    pub reused_tokens: usize,
    /// Tokens of the prompt the forward pass had to prefill.
    pub prefilled_tokens: usize,
    /// How the entry was used.
    pub path: CachePath,
    /// Why nothing was reused, when `path` is [`CachePath::Miss`].
    pub miss: Option<MissReason>,
    /// Tokens of the reused rows an anticipatory prefill produced
    /// ([`LoadedModel::prewarm`], the end-of-answer trigger) rather than an
    /// earlier request: the part of `reused_tokens` the prewarm paid for.
    pub prewarm_hit_tokens: usize,
    /// How long the request waited for a running prewarm to yield the device
    /// before its own lookup could start; zero when none was running.
    pub prewarm_wait: Duration,
    /// Tokens of the reused rows past the answer, which a drafted follow-up
    /// branch ([`crate::PromptCacheConfig::follow_up_branches`]) had already
    /// prefilled; `0` when the entry was not a branch or the request left it
    /// at the shared answer.
    pub follow_up_hit_tokens: usize,
}

impl CacheReport {
    fn miss(prompt_tokens: usize, reason: MissReason) -> Self {
        Self {
            lcp: 0,
            reused_tokens: 0,
            prefilled_tokens: prompt_tokens,
            path: CachePath::Miss,
            miss: Some(reason),
            prewarm_hit_tokens: 0,
            prewarm_wait: Duration::ZERO,
            follow_up_hit_tokens: 0,
        }
    }
}

/// Whether a ring layer that has seen `stored_len` positions still holds
/// every row a window ending at `target_len` reads. Position `p` lives in row
/// `p % capacity`, so it is overwritten once position `p + capacity` has been
/// written; the window needs positions `target_len - window ..` onward, and
/// the first of them survives iff `first_needed + capacity >= stored_len`.
/// For `target_len >= window` that is `stored_len - target_len <= slack`.
///
/// `restored_at` is the checkpoint position the rows were last restored to
/// (`0` for a ring only ever written forward): a restore writes the window
/// ending there and nothing before it, so a window starting earlier than
/// `restored_at - window` would read rows of the state the restore replaced.
fn ring_rewind_fits(
    stored_len: usize,
    target_len: usize,
    restored_at: usize,
    ring: &KvRing,
) -> bool {
    let first_needed = target_len.saturating_sub(ring.window);
    let first_valid = restored_at.saturating_sub(ring.window);
    ring.write_offset == 0
        && first_needed >= first_valid
        && (target_len == 0 || first_needed + ring.capacity >= stored_len)
}

impl PrefixState {
    /// A full copy of this state's rows, for a branch that diverges from it.
    /// This clones every layer's cache, so its cost is the entry's own
    /// [`Self::byte_len`].
    pub(super) fn branch(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            layer_caches: self.layer_caches.clone(),
            cached_len: self.cached_len,
        }
    }

    /// The first reason this state cannot be rewound to `target_len` tokens,
    /// `None` when every layer can. `restored_at` is the checkpoint position
    /// the ring rows were last restored to, `0` when never.
    fn rewind_refusal(
        &self,
        target_len: usize,
        restored_at: usize,
        widths: &[LayerPadRowWidths],
    ) -> Option<MissReason> {
        if widths.len() != self.layer_caches.len() {
            return Some(MissReason::UnrewindableLayer);
        }
        self.layer_caches
            .iter()
            .zip(widths)
            .find_map(|(state, width)| match (state, width) {
                (LayerCacheState::SharedFromLayer, _) => None,
                (LayerCacheState::Attention(cache), _) if cache.ring_geometry().is_some() => cache
                    .ring_geometry()
                    .and_then(|ring| self.ring_refusal(target_len, restored_at, ring)),
                (LayerCacheState::Attention(_), LayerPadRowWidths::Attention { .. }) => None,
                _ => Some(MissReason::UnrewindableLayer),
            })
    }

    fn ring_refusal(
        &self,
        target_len: usize,
        restored_at: usize,
        ring: &KvRing,
    ) -> Option<MissReason> {
        if ring.write_offset != 0 {
            return Some(MissReason::UnrewindableLayer);
        }
        if ring_rewind_fits(self.cached_len, target_len, restored_at, ring) {
            return None;
        }
        let stale =
            target_len.saturating_sub(ring.window) < restored_at.saturating_sub(ring.window);
        Some(if stale {
            MissReason::RingRowsStale { restored_at }
        } else {
            MissReason::RingSlackExceeded {
                rewind_rows: self.cached_len - target_len,
                slack_rows: ring.capacity - ring.window,
            }
        })
    }

    /// Rewinds this state to its first `target_len` tokens: full-attention
    /// layers truncate their rows, ring layers keep theirs (their rows are
    /// addressed by position, so rewinding is `cached_len` going back --
    /// [`LayerCache::truncate`]'s own doc) and are checked to still hold the
    /// window, and a shared-KV layer follows its donor. All-or-nothing: on
    /// `Err` the state is untouched. Past what the ring holds,
    /// [`Self::restore_checkpoint`] is the way back.
    pub(super) fn rewind_to(
        &mut self,
        target_len: usize,
        restored_at: usize,
        widths: &[LayerPadRowWidths],
    ) -> Result<(), MissReason> {
        if target_len >= self.cached_len {
            return Ok(());
        }
        if let Some(reason) = self.rewind_refusal(target_len, restored_at, widths) {
            return Err(reason);
        }
        for (state, width) in self.layer_caches.iter_mut().zip(widths) {
            if let (
                LayerCacheState::Attention(cache),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) = (state, width)
            {
                cache.truncate(target_len, *even_odd_row, *v_row);
            }
        }
        self.ids.truncate(target_len);
        self.cached_len = target_len;
        Ok(())
    }

    /// Host bytes this state holds: its ids plus every layer's cache rows,
    /// counted by allocated capacity because that is what the heap keeps.
    pub(super) fn byte_len(&self) -> usize {
        let float_bytes = |rows: &[&Vec<f32>]| -> usize {
            rows.iter()
                .map(|rows| rows.capacity() * size_of::<f32>())
                .sum()
        };
        let layer_bytes: usize = self
            .layer_caches
            .iter()
            .map(|state| match state {
                LayerCacheState::Attention(cache) => {
                    float_bytes(&[&cache.k_even, &cache.k_odd, &cache.v])
                }
                LayerCacheState::DenseAttention(cache) => {
                    float_bytes(&[&cache.k_first, &cache.k_second, &cache.k_pass, &cache.v])
                }
                LayerCacheState::Ssm(cache) => float_bytes(&[&cache.conv_history, &cache.state]),
                LayerCacheState::SharedFromLayer => 0,
            })
            .sum();
        layer_bytes + self.ids.capacity() * size_of::<u32>()
    }
}

/// One cached conversation: the state, the checkpoints that let it rewind
/// past its ring's slack, and where the rings were last restored to.
pub(super) struct CacheEntry {
    pub(super) state: PrefixState,
    /// What `state` was built under; only a request with an equal key may
    /// resume it.
    pub(super) key: CacheKey,
    /// Ascending by position, every one a snapshot of `state.ids[..position]`.
    pub(super) checkpoints: Vec<RingCheckpoint>,
    /// Position of the checkpoint the rings were last restored to, `0` when
    /// they were only ever written forward ([`ring_rewind_fits`]).
    pub(super) restored_at: usize,
    /// The rows of `state` an anticipatory prefill produced, so the request
    /// that reuses them can report what the prewarm saved it.
    pub(super) prewarmed: Option<Range<usize>>,
    /// Set on a drafted follow-up branch: the length of the answer prefix it
    /// shares with the entry it was cloned from. An unused branch is evicted
    /// before any entry a request produced.
    pub(super) branch_base: Option<usize>,
    /// Which content blocks the stored ids hold, set when the cache stores
    /// the entry ([`PromptCache::bloom_candidates`]).
    pub(super) bloom: Option<BlockBloom>,
}

impl CacheEntry {
    pub(super) const fn new(state: PrefixState, key: CacheKey) -> Self {
        Self {
            state,
            key,
            checkpoints: Vec::new(),
            restored_at: 0,
            prewarmed: None,
            branch_base: None,
            bloom: None,
        }
    }

    /// An entry holding nothing yet, for a request or prewarm no entry served.
    pub(super) const fn empty(key: CacheKey) -> Self {
        Self::new(
            PrefixState {
                ids: Vec::new(),
                layer_caches: Vec::new(),
                cached_len: 0,
            },
            key,
        )
    }

    /// Records that rows `start..end` came from a prewarm, joining a range
    /// that ends where this one starts.
    pub(super) fn mark_prewarmed(&mut self, start: usize, end: usize) {
        if end <= start {
            return;
        }
        let joined = match self.prewarmed.take() {
            Some(held) if held.end >= start => held.start.min(start)..end,
            _ => start..end,
        };
        self.prewarmed = Some(joined);
    }

    /// Cuts the prewarmed range down to the rows the entry still holds.
    fn clamp_prewarmed(&mut self) {
        let held = self.state.cached_len;
        self.prewarmed = self
            .prewarmed
            .take()
            .map(|range| range.start..range.end.min(held))
            .filter(|range| range.start < range.end);
    }

    pub(super) fn prewarmed_len(&self) -> usize {
        self.prewarmed.as_ref().map_or(0, ExactSizeIterator::len)
    }

    pub(super) fn checkpoint_positions(&self) -> Vec<usize> {
        self.checkpoints
            .iter()
            .map(RingCheckpoint::position)
            .collect()
    }

    pub(super) fn byte_len(&self) -> usize {
        self.state.byte_len()
            + self
                .checkpoints
                .iter()
                .map(RingCheckpoint::byte_len)
                .sum::<usize>()
            + self.bloom.as_ref().map_or(0, BlockBloom::byte_len)
    }

    /// The state to hand a decode loop as its seed, `None` while nothing is
    /// cached yet (the loop builds fresh caches for `None`).
    pub(super) fn take_state(&mut self) -> Option<PrefixState> {
        (self.state.cached_len > 0).then(|| {
            core::mem::replace(
                &mut self.state,
                PrefixState {
                    ids: Vec::new(),
                    layer_caches: Vec::new(),
                    cached_len: 0,
                },
            )
        })
    }

    /// Adds `checkpoint` in position order, then evicts down to `max`
    /// ([`retained_positions`]).
    pub(super) fn insert_checkpoint(&mut self, checkpoint: RingCheckpoint, max: usize) {
        self.checkpoints
            .retain(|held| held.position() != checkpoint.position());
        self.checkpoints.push(checkpoint);
        self.checkpoints.sort_by_key(RingCheckpoint::position);
        let kept = retained_positions(&self.checkpoint_positions(), max);
        self.checkpoints
            .retain(|held| kept.contains(&held.position()));
    }

    /// Brings the entry to `resume_len` tokens: a rewind when the rings hold
    /// the window, otherwise the nearest checkpoint at or before `resume_len`.
    /// Checkpoints past the new length describe a sequence the prompt has left
    /// and are dropped (llama-server erases `pos_max > pos_next`,
    /// `server-context.cpp:3367-3378`). On `Err` the entry is untouched.
    fn resume(
        &mut self,
        resume_len: usize,
        widths: &[LayerPadRowWidths],
    ) -> Result<CachePath, MissReason> {
        let path = if resume_len >= self.state.cached_len {
            CachePath::Extend
        } else {
            CachePath::Rewind
        };
        match self.state.rewind_to(resume_len, self.restored_at, widths) {
            Ok(()) => {
                self.checkpoints
                    .retain(|held| held.position() <= resume_len);
                Ok(path)
            }
            Err(
                reason @ (MissReason::RingSlackExceeded { .. } | MissReason::RingRowsStale { .. }),
            ) => self.restore_nearest(resume_len, widths).ok_or(reason),
            Err(reason) => Err(reason),
        }
    }

    fn restore_nearest(
        &mut self,
        resume_len: usize,
        widths: &[LayerPadRowWidths],
    ) -> Option<CachePath> {
        let index = self
            .checkpoints
            .iter()
            .rposition(|held| held.position() <= resume_len)?;
        self.checkpoints.truncate(index + 1);
        let position = self.checkpoints[index].position();
        self.state
            .restore_checkpoint(&self.checkpoints[index], widths);
        self.restored_at = position;
        Some(CachePath::Checkpoint)
    }
}

/// The entries one [`LoadedModel`] keeps, keyed by the stamp each was stored
/// under, so the least recently used is the lowest stamp.
pub(super) struct PromptCache {
    entries: BTreeMap<u64, CacheEntry>,
    index: BlockIndex,
    next_stamp: u64,
    bloom_bits: u32,
    bloom_hashes: u32,
    last_report: Option<CacheReport>,
    prewarm_suffix: Vec<u32>,
    follow_up_closing: Vec<u32>,
}

/// What the per-entry bloom filters say about a prompt whose prefix stopped
/// matching: how many entries probably hold a block of its later content, and
/// at which of the prompt's blocks.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct BloomCandidates {
    /// Entries with at least one probable block.
    pub(super) entries: usize,
    /// `(entry stamp, prompt block index)` for every probable block.
    pub(super) blocks: Vec<(u64, usize)>,
}

impl PromptCache {
    pub(super) fn new() -> Self {
        let standard = PromptCacheConfig::standard();
        Self {
            entries: BTreeMap::new(),
            index: BlockIndex::new(standard.block_tokens as usize),
            next_stamp: 0,
            bloom_bits: standard.bloom_bits_per_entry,
            bloom_hashes: standard.bloom_hashes,
            last_report: None,
            prewarm_suffix: Vec::new(),
            follow_up_closing: Vec::new(),
        }
    }

    pub(super) fn set_follow_up_closing(&mut self, closing: &[u32]) {
        self.follow_up_closing = closing.to_vec();
    }

    pub(super) fn follow_up_closing(&self) -> &[u32] {
        &self.follow_up_closing
    }

    pub(super) fn set_prewarm_suffix(&mut self, suffix: &[u32]) {
        self.prewarm_suffix = suffix.to_vec();
    }

    pub(super) fn prewarm_suffix(&self) -> &[u32] {
        &self.prewarm_suffix
    }

    /// [`Self::take_best`] for a prewarm: the lookup is the same, but the
    /// request-facing report it leaves is the previous request's, not this
    /// one's.
    pub(super) fn take_for_prewarm(
        &mut self,
        prompt_ids: &[u32],
        key: &CacheKey,
        widths: &[LayerPadRowWidths],
        min_similarity_milli: u32,
    ) -> (Option<CacheEntry>, CacheReport) {
        let previous = self.last_report;
        let taken = self.take_best(prompt_ids, key, widths, min_similarity_milli);
        self.last_report = previous;
        taken
    }

    /// Takes the entry sharing the longest prefix with `prompt_ids` out of
    /// the cache, among the entries whose [`CacheKey`] equals `key`, brought
    /// to that prefix (or to the checkpoint behind it) and ready to be
    /// resumed from, or reports why none can be. `widths` are the model's
    /// per-layer row widths
    /// ([`LoadedModel::declared_layer_cache_names_and_widths`]).
    ///
    /// An entry is only a candidate when [`entry_is_reusable`] holds for it:
    /// llama-server's `slot_prompt_similarity` rule
    /// (`tools/server/server-context.cpp:1542-1571`), which takes a slot only
    /// when its common prefix covers more than the threshold of the incoming
    /// prompt and otherwise falls to a different slot, leaving the first
    /// slot's context alone (`:1587-1601`). Here the different slot is a new
    /// entry the request builds, and [`Self::store`]'s least-recently-used
    /// eviction decides what makes room for it.
    ///
    /// The candidates come from [`BlockIndex`], not from comparing every entry
    /// ([`Self::best_candidate`]).
    pub(super) fn take_best(
        &mut self,
        prompt_ids: &[u32],
        key: &CacheKey,
        widths: &[LayerPadRowWidths],
        min_similarity_milli: u32,
    ) -> (Option<CacheEntry>, CacheReport) {
        let best = self.best_candidate(prompt_ids, key, min_similarity_milli);
        let resume = best.map(|(stamp, lcp)| (stamp, lcp.min(prompt_ids.len().saturating_sub(1))));
        let outcome = match resume {
            None => Err(self.miss_reason(prompt_ids, key)),
            Some((_, 0)) => Err(MissReason::NoCommonPrefix),
            Some((stamp, resume_len)) => self
                .resume_at(stamp, resume_len, widths)
                .map(|(entry, path)| (entry, path, resume_len)),
        };
        let (entry, report) = match outcome {
            Ok((mut entry, path, lcp)) => {
                entry.clamp_prewarmed();
                let reused = entry.state.cached_len;
                let report = CacheReport {
                    lcp,
                    reused_tokens: reused,
                    prefilled_tokens: prompt_ids.len() - reused,
                    path,
                    miss: None,
                    prewarm_hit_tokens: entry.prewarmed_len(),
                    prewarm_wait: Duration::ZERO,
                    follow_up_hit_tokens: entry
                        .branch_base
                        .map_or(0, |base| reused.saturating_sub(base)),
                };
                (Some(entry), report)
            }
            Err(reason) => (None, CacheReport::miss(prompt_ids.len(), reason)),
        };
        self.last_report = Some(report);
        (entry, report)
    }

    /// The reusable entry sharing the longest prefix with `prompt_ids`, and
    /// that prefix's length; the most recently stored among equals. Walks the
    /// chained block hashes from the deepest block any entry still matches
    /// outward, so the first level with a reusable entry holds the longest
    /// reusable prefix: an entry that stopped matching at one level shares
    /// fewer tokens than any that went on to the next. Entries sharing less
    /// than a block are found through the first token, and only when such an
    /// overlap could still clear the similarity floor or the entry is itself
    /// shorter than a block.
    fn best_candidate(
        &self,
        prompt_ids: &[u32],
        key: &CacheKey,
        min_similarity_milli: u32,
    ) -> Option<(u64, usize)> {
        let block = self.index.block_tokens();
        let entry_of = |stamp: u64| self.entries.get(&stamp).filter(|entry| entry.key == *key);
        let walk = self.index.walk(prompt_ids, |stamp, depth| {
            entry_of(stamp).is_some_and(|entry| {
                let span = depth * block..(depth + 1) * block;
                entry.state.ids.get(span.clone()) == prompt_ids.get(span)
            })
        });
        let reusable = |stamp: u64, from: usize| {
            let entry = entry_of(stamp)?;
            let lcp = from + longest_common_prefix(&entry.state.ids[from..], &prompt_ids[from..]);
            entry_is_reusable(
                lcp,
                entry.state.ids.len(),
                prompt_ids.len(),
                min_similarity_milli,
            )
            .then_some((stamp, lcp))
        };
        for depth in (0..walk.levels.len()).rev() {
            let from = (depth + 1) * block;
            let best = walk.levels[depth]
                .iter()
                .filter(|stamp| {
                    walk.levels
                        .get(depth + 1)
                        .is_none_or(|deeper| !deeper.contains(stamp))
                })
                .filter_map(|stamp| reusable(*stamp, from))
                .max_by_key(|&(stamp, lcp)| (lcp, stamp));
            if best.is_some() {
                return best;
            }
        }
        let overlap_below_a_block_can_clear_the_floor =
            (block - 1) * 1000 > min_similarity_milli as usize * prompt_ids.len();
        let sub_block: &[u64] = if overlap_below_a_block_can_clear_the_floor {
            self.index.sharing_first_token(prompt_ids)
        } else {
            self.index.shorter_than_a_block()
        };
        sub_block
            .iter()
            .filter(|stamp| {
                walk.levels
                    .first()
                    .is_none_or(|first| !first.contains(stamp))
            })
            .filter_map(|stamp| reusable(*stamp, 0))
            .max_by_key(|&(stamp, lcp)| (lcp, stamp))
    }

    fn miss_reason(&self, prompt_ids: &[u32], key: &CacheKey) -> MissReason {
        if self.entries.is_empty() {
            return MissReason::Empty;
        }
        let mut same_key = self.entries.values().filter(|entry| entry.key == *key);
        let Some(first) = same_key.next() else {
            return MissReason::ConfigMismatch;
        };
        let shares_first_token = core::iter::once(first)
            .chain(same_key)
            .any(|entry| entry.state.ids.first() == prompt_ids.first());
        if shares_first_token {
            MissReason::BelowSimilarity
        } else {
            MissReason::NoCommonPrefix
        }
    }

    /// Takes entry `stamp` out and brings it to `resume_len`; an entry that
    /// cannot get there stays in the cache, under its stamp, for the next
    /// request.
    fn resume_at(
        &mut self,
        stamp: u64,
        resume_len: usize,
        widths: &[LayerPadRowWidths],
    ) -> Result<(CacheEntry, CachePath), MissReason> {
        let mut entry = self
            .drop_entry(stamp)
            .ok_or(MissReason::UnrewindableLayer)?;
        match entry.resume(resume_len, widths) {
            Ok(path) => Ok((entry, path)),
            Err(reason) => {
                self.index.insert(stamp, &entry.state.ids);
                self.entries.insert(stamp, entry);
                Err(reason)
            }
        }
    }

    fn drop_entry(&mut self, stamp: u64) -> Option<CacheEntry> {
        let entry = self.entries.remove(&stamp)?;
        self.index.remove(stamp, &entry.state.ids);
        Some(entry)
    }

    /// The bloom filters' answer for `prompt_ids`: the entries under `key`
    /// that probably hold a block of its content that starts at or past token
    /// `from_token` (the end of the prefix already reused), and which of its
    /// blocks. A filter can say yes wrongly,
    /// never no wrongly ([`BlockBloom`]).
    pub(super) fn bloom_candidates(
        &self,
        prompt_ids: &[u32],
        key: &CacheKey,
        from_token: usize,
    ) -> BloomCandidates {
        let block = self.index.block_tokens();
        let from_block = from_token.div_ceil(block);
        let hashes = content_hashes(prompt_ids, block);
        let blocks: Vec<(u64, usize)> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.key == *key)
            .filter_map(|(stamp, entry)| entry.bloom.as_ref().map(|bloom| (*stamp, bloom)))
            .flat_map(|(stamp, bloom)| {
                hashes
                    .iter()
                    .enumerate()
                    .skip(from_block)
                    .filter(|(_, hash)| bloom.maybe_contains(**hash))
                    .map(move |(block_index, _)| (stamp, block_index))
            })
            .collect();
        let mut stamps: Vec<u64> = blocks.iter().map(|(stamp, _)| *stamp).collect();
        stamps.dedup();
        BloomCandidates {
            entries: stamps.len(),
            blocks,
        }
    }

    /// Whole blocks of `prompt_ids` that some entry shares from the start.
    #[cfg(all(test, feature = "metal", target_os = "macos"))]
    pub(super) fn matched_blocks(&self, prompt_ids: &[u32]) -> usize {
        let block = self.index.block_tokens();
        self.index
            .walk(prompt_ids, |stamp, depth| {
                self.entries.get(&stamp).is_some_and(|entry| {
                    let span = depth * block..(depth + 1) * block;
                    entry.state.ids.get(span.clone()) == prompt_ids.get(span)
                })
            })
            .levels
            .len()
    }

    /// Re-cuts every entry into `config`'s blocks and re-sizes their filters
    /// when the config asks for other ones than the index holds.
    fn reconfigure(&mut self, config: &PromptCacheConfig) {
        let block = (config.block_tokens as usize).max(1);
        let unchanged = self.index.block_tokens() == block
            && self.bloom_bits == config.bloom_bits_per_entry
            && self.bloom_hashes == config.bloom_hashes;
        if unchanged {
            return;
        }
        self.bloom_bits = config.bloom_bits_per_entry;
        self.bloom_hashes = config.bloom_hashes;
        self.index = BlockIndex::new(block);
        let (bits, hashes) = (self.bloom_bits, self.bloom_hashes);
        for (stamp, entry) in &mut self.entries {
            entry.bloom = Some(BlockBloom::of(&entry.state.ids, block, bits, hashes));
            self.index.insert(*stamp, &entry.state.ids);
        }
    }

    /// Puts `entry` back as the most recently used entry, then evicts from
    /// the least recently used end until the entry count and byte budget in
    /// `config` hold, returning how many entries are held; `None` when `entry`
    /// was not stored (it holds nothing, or is larger than the whole budget).
    pub(super) fn store(
        &mut self,
        mut entry: CacheEntry,
        config: &PromptCacheConfig,
    ) -> Option<usize> {
        self.reconfigure(config);
        let budget = usize::try_from(config.byte_budget).unwrap_or(usize::MAX);
        entry.bloom = Some(BlockBloom::of(
            &entry.state.ids,
            self.index.block_tokens(),
            self.bloom_bits,
            self.bloom_hashes,
        ));
        let state = &entry.state;
        if state.cached_len == 0 || state.layer_caches.is_empty() || entry.byte_len() > budget {
            return None;
        }
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        self.index.insert(stamp, &entry.state.ids);
        self.entries.insert(stamp, entry);
        let max_entries = config.max_entries as usize;
        while self.entries.len() > max_entries || self.stored_bytes() > budget {
            let victim = self
                .entries
                .iter()
                .find(|(_, held)| held.branch_base.is_some())
                .or_else(|| self.entries.iter().next())
                .map(|(stamp, _)| *stamp)?;
            self.drop_entry(victim);
        }
        Some(self.entries.len())
    }

    pub(super) fn stored_bytes(&self) -> usize {
        self.entries.values().map(CacheEntry::byte_len).sum()
    }

    pub(super) const fn last_report(&self) -> Option<CacheReport> {
        self.last_report
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.index = BlockIndex::new(self.index.block_tokens());
    }
}

/// A forward that failed after the lookup has partly written the taken
/// entry's rows, so it is dropped rather than put back.
pub(super) fn log_entry_dropped(error: &InteropError) {
    debug!(
        cache_drop_reason = %error,
        "prompt cache entry dropped after a failed forward"
    );
}

impl LoadedModel<'_> {
    /// Drops every entry. A model-level input the key does not carry (the
    /// expert sidecar, [`Self::attach_expert_sidecar`]) just changed, so every
    /// stored row was built under a model this one no longer is.
    pub(super) fn clear_prompt_cache(&mut self) {
        self.prompt_cache
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Host bytes the prompt cache holds across all entries.
    #[must_use]
    pub fn prompt_cache_bytes(&self) -> usize {
        self.prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stored_bytes()
    }

    /// The report of the most recent request that went through the prompt
    /// cache, `None` before the first. Concurrent requests overwrite it; read
    /// it right after the request it describes.
    #[must_use]
    pub fn last_prompt_cache_report(&self) -> Option<CacheReport> {
        self.prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .last_report()
    }

    /// The key a request (or prewarm) under these inputs may reuse rows from.
    pub(super) fn cache_key(
        &self,
        serving_config: &ServingConfig,
        runtime: &BackendRuntime,
        forced_draft_width: Option<u16>,
    ) -> CacheKey {
        CacheKey::of(
            serving_config,
            runtime.uses_gpu(),
            self.effective_rope_scaling(serving_config),
            ring_slack_rows(serving_config, forced_draft_width),
            self.ring_write_offset,
        )
    }

    /// Looks `prompt_ids` up in the cache and emits spec R8's per-request
    /// telemetry for the outcome. `waited` is how long the request waited for
    /// a running prewarm; the entry it takes is no longer a prewarm's, so its
    /// prewarmed range is spent.
    pub(super) fn prompt_cache_lookup(
        &self,
        prompt_ids: &[u32],
        key: &CacheKey,
        widths: &[LayerPadRowWidths],
        waited: Duration,
        min_similarity_milli: u32,
    ) -> (Option<CacheEntry>, CacheReport) {
        let mut cache = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let (mut entry, mut report) =
            cache.take_best(prompt_ids, key, widths, min_similarity_milli);
        report.prewarm_wait = waited;
        cache.last_report = Some(report);
        let bloom = cache.bloom_candidates(prompt_ids, key, report.reused_tokens);
        drop(cache);
        if let Some(taken) = entry.as_mut() {
            taken.prewarmed = None;
            taken.branch_base = None;
        }
        debug!(
            cache_lcp = report.lcp as u64,
            cache_reused_tokens = report.reused_tokens as u64,
            cache_prefilled_tokens = report.prefilled_tokens as u64,
            cache_path = report.path.as_str(),
            cache_miss_reason = report.miss.map_or("none", MissReason::as_str),
            prewarm_hit_tokens = report.prewarm_hit_tokens as u64,
            follow_up_hit_tokens = report.follow_up_hit_tokens as u64,
            prewarm_wait_ns = u64::try_from(report.prewarm_wait.as_nanos()).unwrap_or(u64::MAX),
            bloom_candidate_entries = bloom.entries as u64,
            bloom_candidate_blocks = bloom.blocks.len() as u64,
            "prompt cache lookup"
        );
        (entry, report)
    }

    /// Hands a finished request's entry to the cache, `true` when it is held.
    pub(super) fn prompt_cache_store(&self, entry: CacheEntry, config: &PromptCacheConfig) -> bool {
        let cached_tokens = entry.state.cached_len;
        let checkpoints = entry.checkpoints.len();
        let entries = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .store(entry, config);
        debug!(
            cache_stored_tokens = cached_tokens as u64,
            cache_checkpoints = checkpoints as u64,
            cache_entries = entries.unwrap_or(0) as u64,
            "prompt cache store"
        );
        entries.is_some()
    }

    /// Prefills `ids` up to each of `stops` (ascending), snapshotting the
    /// ring layers at every stop that is also one of `checkpoints`, so the
    /// entry can come back to it later. `after_stop` runs at every stop with
    /// its position and may end the prefill there (`Break`): a request has no
    /// reason to, a prewarm yields to a waiting request. The entry returned
    /// holds whatever was reached.
    ///
    /// Each stretch is a seeded decode of `max_tokens = 1` over the tokens
    /// since the last stop -- the same shape [`Self::prefill_prefix`] uses to
    /// prefill without decoding -- and its sampled token is discarded. A
    /// sampled end-of-sequence token is forwarded and cached, so the state is
    /// rewound the one row back to the stop. That rewind fits the ring's
    /// slack because [`planned_positions`] plans nothing without any.
    #[allow(clippy::too_many_arguments)] // the seeded decode's own knobs, minus the sinks it fixes
    pub(super) fn prefill_through_stops(
        &self,
        ids: &[u32],
        mut entry: CacheEntry,
        stops: &[usize],
        checkpoints: &[usize],
        widths: &[LayerPadRowWidths],
        config: &PromptCacheConfig,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
        after_stop: &mut dyn FnMut(usize) -> ControlFlow<(), ()>,
    ) -> Result<CacheEntry, InteropError> {
        for &position in stops {
            let held = entry.state.cached_len;
            if position > held {
                let seed = entry.take_state();
                let (_, _, _, advanced) = self.run_decode_loop_from_ids(
                    ids[held..position].to_vec(),
                    1,
                    serving_config,
                    runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    seed,
                    true,
                    None,
                    forced_draft_width,
                )?;
                entry.state = advanced;
                entry
                    .state
                    .rewind_to(position, entry.restored_at, widths)
                    .map_err(|reason| InteropError::PromptCacheStopRewind {
                        position,
                        reason: reason.as_str(),
                    })?;
            }
            if checkpoints.contains(&position)
                && let Some(checkpoint) = RingCheckpoint::capture(&entry.state)
            {
                debug!(
                    cache_checkpoint_position = position as u64,
                    cache_checkpoint_bytes = checkpoint.byte_len() as u64,
                    "prompt cache checkpoint"
                );
                entry.insert_checkpoint(checkpoint, config.max_checkpoints as usize);
            }
            if after_stop(position).is_break() {
                break;
            }
        }
        Ok(entry)
    }

    /// The decode loop behind every entry point, with the prompt cache in
    /// front of it when `serving_config.prompt_cache` is enabled and nothing
    /// asks the loop to observe the prefill itself (a caller-owned seed, a
    /// forced token stream, or a logits/node-values sink all need the full
    /// prefill to run).
    ///
    /// `turn_ends` are token counts into `ids` where a turn ends (the index
    /// just past an end-of-turn token); the prefill stops there to snapshot
    /// the ring layers ([`RingCheckpoint`]), so a later request that rewrites
    /// from that turn on can restore it. Checkpoints also land every
    /// [`PromptCacheConfig::checkpoint_interval`] tokens and at the point the
    /// request resumed from.
    ///
    /// On the cached path the returned [`PrefixState`] is hollow: the real
    /// one now lives in the cache, and the only caller that wants the state
    /// back ([`Self::prefill_prefix`]) switches the cache off for its call.
    #[allow(clippy::too_many_arguments)] // mirrors `run_decode_loop_from_ids`, which it forwards to
    pub(super) fn run_decode_loop_through_cache(
        &self,
        ids: Vec<u32>,
        turn_ends: &[usize],
        max_tokens: usize,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        token_override: Option<&[u32]>,
        logits_sink: &mut LogitsSink,
        node_values_sink: &mut NodeValuesSink,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
        seed: Option<PrefixState>,
        force_two_range: bool,
        speculative_stats: Option<&mut SpeculativeDecodeStats>,
        forced_draft_width: Option<u16>,
    ) -> Result<(Vec<u32>, String, bool, PrefixState), InteropError> {
        let pending = self.prewarm_gate.enter_request();
        let config = serving_config.prompt_cache;
        let cacheable = config.is_enabled()
            && seed.is_none()
            && token_override.is_none()
            && !ids.is_empty()
            && matches!(logits_sink, LogitsSink::Discard)
            && matches!(node_values_sink, NodeValuesSink::Discard);
        if !cacheable {
            return self.run_decode_loop_from_ids(
                ids,
                max_tokens,
                serving_config,
                runtime,
                token_override,
                logits_sink,
                node_values_sink,
                on_token,
                seed,
                force_two_range,
                speculative_stats,
                forced_draft_width,
            );
        }
        // before the lookup: a request this rejects must not take an entry out
        apply_serving_config(serving_config, ids.len())?;
        let (_, widths) = self.declared_layer_cache_names_and_widths()?;
        let key = self.cache_key(serving_config, runtime, forced_draft_width);
        let (found, report) = self.prompt_cache_lookup(
            &ids,
            &key,
            &widths,
            pending.waited(),
            config.min_similarity_milli,
        );
        let entry = found.unwrap_or_else(|| CacheEntry::empty(key));
        let positions = planned_positions(
            &entry.checkpoint_positions(),
            report.reused_tokens,
            ids.len(),
            turn_ends,
            &config,
        );
        let mut entry = self
            .prefill_through_stops(
                &ids,
                entry,
                &positions,
                &positions,
                &widths,
                &config,
                serving_config,
                runtime,
                forced_draft_width,
                &mut |_position| ControlFlow::Continue(()),
            )
            .inspect_err(log_entry_dropped)?;
        let resumed_at = entry.state.cached_len;
        let (generated_ids, text, stopped_by_eos, final_state) = self
            .run_decode_loop_from_ids(
                ids[resumed_at..].to_vec(),
                max_tokens,
                serving_config,
                runtime,
                None,
                logits_sink,
                node_values_sink,
                on_token,
                entry.take_state(),
                true,
                speculative_stats,
                forced_draft_width,
            )
            .inspect_err(log_entry_dropped)?;
        entry.state = final_state;
        let stored = self.prompt_cache_store(entry, &config);
        drop(pending);
        if stored {
            self.queue_prewarm_after_answer(&ids, &generated_ids, key, forced_draft_width);
        }
        let hollow = PrefixState {
            ids: Vec::new(),
            layer_caches: Vec::new(),
            cached_len: 0,
        };
        Ok((generated_ids, text, stopped_by_eos, hollow))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::cell::Cell;

    use proptest::collection::vec;
    use proptest::test_runner::{Config, TestRunner};
    use proxima_tensor::NumericPolicy;

    use super::*;
    use crate::serving::{ContextLength, GdnPrefillBackend};

    const ANY_OVERLAP: u32 = 0;

    fn base_key() -> CacheKey {
        CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)
    }

    fn state_with_ids(ids: &[u32]) -> CacheEntry {
        CacheEntry::new(
            PrefixState {
                ids: ids.to_vec(),
                layer_caches: vec![LayerCacheState::SharedFromLayer],
                cached_len: ids.len(),
            },
            base_key(),
        )
    }

    fn shared_widths() -> Vec<LayerPadRowWidths> {
        vec![LayerPadRowWidths::SharedFromLayer]
    }

    fn enabled_config() -> PromptCacheConfig {
        PromptCacheConfig {
            byte_budget: 1 << 20,
            ..PromptCacheConfig::off()
        }
    }

    /// A conversation turn extends the previous prompt and its generated
    /// tokens: the whole stored sequence is the shared prefix.
    #[test]
    fn extension_of_the_stored_sequence_takes_the_entry_and_prefills_only_the_new_tokens() {
        let mut cache = PromptCache::new();
        cache.store(
            state_with_ids(&[2, 105, 2364, 107, 9259]),
            &enabled_config(),
        );

        let (state, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259, 106, 107, 105, 4368],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(state.expect("entry reused").state.cached_len, 5);
        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(report.lcp, 5);
        assert_eq!(report.prefilled_tokens, 4);
    }

    #[test]
    fn a_prompt_sharing_no_first_token_reports_no_common_prefix() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364]), &enabled_config());

        let (state, report) =
            cache.take_best(&[7, 105, 2364], &base_key(), &shared_widths(), ANY_OVERLAP);

        assert!(state.is_none());
        assert_eq!(report.miss, Some(MissReason::NoCommonPrefix));
        assert_eq!(report.prefilled_tokens, 3);
    }

    #[test]
    fn an_empty_cache_reports_empty_and_prefills_the_whole_prompt() {
        let mut cache = PromptCache::new();

        let (state, report) =
            cache.take_best(&[2, 105, 2364], &base_key(), &shared_widths(), ANY_OVERLAP);

        assert!(state.is_none());
        assert_eq!(report.miss, Some(MissReason::Empty));
        assert_eq!(cache.last_report(), Some(report));
    }

    /// An identical prompt still prefills its last token: the model needs
    /// that row's logits to sample the first new token.
    #[test]
    fn an_identical_prompt_leaves_one_token_to_prefill() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());

        let (state, report) = cache.take_best(
            &[2, 105, 2364, 107],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.path, CachePath::Rewind);
        assert_eq!(report.lcp, 3);
        assert_eq!(report.prefilled_tokens, 1);
        assert_eq!(state.expect("entry reused").state.ids, vec![2, 105, 2364]);
    }

    fn prewarmed_entry() -> CacheEntry {
        let mut entry = state_with_ids(&[2, 105, 2364, 107, 9259, 106, 107, 105]);
        entry.mark_prewarmed(5, 8);
        entry
    }

    /// The suffix a prewarm appended (end-of-turn, newline, the next user
    /// turn's opener) is what the next prompt shares in full.
    #[test]
    fn a_request_extending_prewarmed_rows_reports_the_rows_the_prewarm_paid_for() {
        let mut cache = PromptCache::new();
        cache.store(prewarmed_entry(), &enabled_config());

        let (_, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259, 106, 107, 105, 4368],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(report.prewarm_hit_tokens, 3);
        assert_eq!(report.prefilled_tokens, 1);
    }

    #[test]
    fn a_rewind_into_prewarmed_rows_counts_only_the_rows_it_kept() {
        let mut cache = PromptCache::new();
        cache.store(prewarmed_entry(), &enabled_config());

        let (_, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259, 106, 9, 9, 9],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.path, CachePath::Rewind);
        assert_eq!(report.prewarm_hit_tokens, 1);
    }

    #[test]
    fn a_rewind_before_the_prewarmed_rows_reports_no_hit() {
        let mut cache = PromptCache::new();
        cache.store(prewarmed_entry(), &enabled_config());

        let (_, report) = cache.take_best(
            &[2, 105, 2364, 107, 9, 9, 9, 9, 9],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.prewarm_hit_tokens, 0);
    }

    #[test]
    fn prewarmed_ranges_that_touch_join_and_ranges_that_do_not_replace() {
        let mut entry = state_with_ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        entry.mark_prewarmed(2, 4);
        entry.mark_prewarmed(4, 6);
        assert_eq!(entry.prewarmed, Some(2..6));

        entry.mark_prewarmed(8, 10);
        assert_eq!(entry.prewarmed, Some(8..10));

        entry.mark_prewarmed(10, 10);
        assert_eq!(entry.prewarmed, Some(8..10));
    }

    #[test]
    fn a_prewarm_lookup_leaves_the_last_requests_report_alone() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364]), &enabled_config());
        let (_, request_report) =
            cache.take_best(&[7, 105], &base_key(), &shared_widths(), ANY_OVERLAP);

        let (taken, _) = cache.take_for_prewarm(
            &[2, 105, 2364, 107],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert!(taken.is_some());
        assert_eq!(cache.last_report(), Some(request_report));
    }

    fn branch_entry(ids: &[u32], base: usize) -> CacheEntry {
        let mut entry = state_with_ids(ids);
        entry.branch_base = Some(base);
        entry
    }

    #[test]
    fn a_request_reusing_a_branch_reports_the_rows_it_took_past_the_answer() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());
        cache.store(
            branch_entry(&[2, 105, 2364, 107, 7, 8, 9], 4),
            &enabled_config(),
        );

        let (taken, report) = cache.take_best(
            &[2, 105, 2364, 107, 7, 8, 55, 56],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.lcp, 6);
        assert_eq!(report.follow_up_hit_tokens, 2);
        assert_eq!(
            taken.expect("the branch served").state.ids,
            vec![2, 105, 2364, 107, 7, 8]
        );
    }

    #[test]
    fn a_request_that_stops_at_the_shared_answer_has_no_follow_up_hit() {
        let mut cache = PromptCache::new();
        cache.store(
            branch_entry(&[2, 105, 2364, 107, 7, 8, 9], 4),
            &enabled_config(),
        );

        let (_, report) = cache.take_best(
            &[2, 105, 2364, 107, 55, 56],
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.lcp, 4);
        assert_eq!(report.follow_up_hit_tokens, 0);
    }

    #[test]
    fn unused_branches_are_evicted_before_entries_a_request_produced() {
        let config = PromptCacheConfig {
            max_entries: 3,
            ..enabled_config()
        };
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[1, 1, 1]), &config);
        cache.store(branch_entry(&[1, 1, 1, 5], 3), &config);
        cache.store(state_with_ids(&[2, 2, 2]), &config);

        cache.store(state_with_ids(&[3, 3, 3]), &config);

        let held = |cache: &mut PromptCache, prompt: &[u32]| {
            cache
                .take_best(prompt, &base_key(), &shared_widths(), ANY_OVERLAP)
                .1
                .lcp
        };
        assert_eq!(
            held(&mut cache, &[1, 1, 1, 5, 9]),
            3,
            "only the base kept the answer"
        );
        assert_eq!(held(&mut cache, &[2, 2, 2, 9]), 3);
        assert_eq!(held(&mut cache, &[3, 3, 3, 9]), 3);
    }

    #[test]
    fn the_least_recently_used_entry_is_evicted_past_max_entries() {
        let config = PromptCacheConfig {
            max_entries: 2,
            ..enabled_config()
        };
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[1, 1]), &config);
        cache.store(state_with_ids(&[2, 2]), &config);
        cache.store(state_with_ids(&[3, 3]), &config);

        assert_eq!(
            cache
                .take_best(&[1, 1, 9], &base_key(), &shared_widths(), ANY_OVERLAP)
                .1
                .miss,
            Some(MissReason::NoCommonPrefix)
        );
        assert_eq!(
            cache
                .take_best(&[2, 2, 9], &base_key(), &shared_widths(), ANY_OVERLAP)
                .1
                .path,
            CachePath::Extend
        );
        assert_eq!(
            cache
                .take_best(&[3, 3, 9], &base_key(), &shared_widths(), ANY_OVERLAP)
                .1
                .path,
            CachePath::Extend
        );
    }

    #[test]
    fn a_state_larger_than_the_byte_budget_is_not_stored() {
        let config = PromptCacheConfig {
            byte_budget: 8,
            ..enabled_config()
        };
        let mut cache = PromptCache::new();

        let entries = cache.store(state_with_ids(&[1, 2, 3, 4]), &config);

        assert_eq!(entries, None);
    }

    #[test]
    fn the_byte_budget_evicts_older_entries_to_fit_a_newer_one() {
        let bloom_bytes = u64::from(PromptCacheConfig::standard().bloom_bits_per_entry / 8);
        let config = PromptCacheConfig {
            byte_budget: 24 + bloom_bytes,
            ..enabled_config()
        };
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[1, 2, 3, 4]), &config);

        let entries = cache.store(state_with_ids(&[5, 6, 7, 8]), &config);

        assert_eq!(entries, Some(1));
        assert_eq!(
            cache
                .take_best(&[5, 6, 7, 8, 9], &base_key(), &shared_widths(), ANY_OVERLAP)
                .1
                .path,
            CachePath::Extend
        );
    }

    const WINDOW: usize = 8;
    const SLACK: usize = 4;
    const EVEN_ODD_ROW: usize = 2;
    const V_ROW: usize = 3;

    fn marker(position: usize) -> f32 {
        position as f32 + 1.0
    }

    fn full_layer(positions: usize) -> LayerCache {
        let mut cache = LayerCache::new();
        (0..positions).for_each(|position| {
            let value = marker(position);
            cache.append(
                &[value; EVEN_ODD_ROW],
                &[value; EVEN_ODD_ROW],
                &[value; V_ROW],
            );
        });
        cache
    }

    fn ring_layer(window: usize, slack: usize, positions: usize) -> LayerCache {
        let ring = KvRing::new(window, slack, EVEN_ODD_ROW, V_ROW, 0);
        let mut cache = LayerCache::ring(ring, positions);
        (0..positions).for_each(|position| {
            let value = marker(position);
            cache.append_at(
                position,
                &[value; EVEN_ODD_ROW],
                &[value; EVEN_ODD_ROW],
                &[value; V_ROW],
            );
        });
        cache
    }

    fn gemma_like_state(stored_len: usize) -> PrefixState {
        PrefixState {
            ids: (0..stored_len as u32).collect(),
            layer_caches: vec![
                LayerCacheState::Attention(ring_layer(WINDOW, SLACK, stored_len)),
                LayerCacheState::Attention(full_layer(stored_len)),
                LayerCacheState::SharedFromLayer,
            ],
            cached_len: stored_len,
        }
    }

    fn gemma_like_widths() -> Vec<LayerPadRowWidths> {
        let attention = || LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        };
        vec![attention(), attention(), LayerPadRowWidths::SharedFromLayer]
    }

    fn full_layer_rows(state: &PrefixState) -> usize {
        match &state.layer_caches[1] {
            LayerCacheState::Attention(cache) => cache.k_even.len() / EVEN_ODD_ROW,
            _ => 0,
        }
    }

    /// R3: a rewind inside the ring's slack truncates the full layer to the
    /// shared prefix and leaves the ring layer's rows untouched.
    #[test]
    fn rewind_within_ring_slack_truncates_the_full_layer_and_keeps_the_ring() {
        let mut state = gemma_like_state(40);
        let ring_before = match &state.layer_caches[0] {
            LayerCacheState::Attention(cache) => cache.k_even.clone(),
            _ => Vec::new(),
        };

        state
            .rewind_to(36, 0, &gemma_like_widths())
            .expect("a 4-token rewind fits a 4-row slack");

        assert_eq!(state.cached_len, 36);
        assert_eq!(state.ids.len(), 36);
        assert_eq!(full_layer_rows(&state), 36);
        match &state.layer_caches[0] {
            LayerCacheState::Attention(cache) => assert_eq!(cache.k_even, ring_before),
            _ => panic!("layer 0 is an attention layer"),
        }
    }

    /// R3: past the slack the ring has overwritten rows the window needs, so
    /// the rewind is refused with the numbers, and the state is untouched.
    #[test]
    fn rewind_beyond_ring_slack_is_refused_and_leaves_the_state_untouched() {
        let mut state = gemma_like_state(40);

        let refusal = state.rewind_to(35, 0, &gemma_like_widths());

        assert_eq!(
            refusal,
            Err(MissReason::RingSlackExceeded {
                rewind_rows: 5,
                slack_rows: SLACK,
            })
        );
        assert_eq!(state.cached_len, 40);
        assert_eq!(state.ids.len(), 40);
        assert_eq!(full_layer_rows(&state), 40);
    }

    /// A rewind into the first window of a ring that never wrapped keeps
    /// every row, however far back it goes.
    #[test]
    fn rewind_of_a_ring_that_never_wrapped_always_fits() {
        let mut state = gemma_like_state(WINDOW + SLACK);

        state
            .rewind_to(3, 0, &gemma_like_widths())
            .expect("a ring holding every position it ever saw can rewind anywhere");

        assert_eq!(state.cached_len, 3);
    }

    /// The closed-form check agrees with the ring's real rows: for every
    /// geometry and every stored/target pair in a small box, the check says
    /// "fits" exactly when every position in `target - window .. target`
    /// still reads back its own marker from the ring after the writes.
    #[test]
    fn ring_rewind_check_matches_the_rows_the_ring_actually_holds() {
        let mut cases = 0;
        for window in 1..=6_usize {
            for slack in 0..=6_usize {
                for stored_len in 0..=30_usize {
                    let cache = ring_layer(window, slack, stored_len);
                    let ring = KvRing::new(window, slack, EVEN_ODD_ROW, V_ROW, 0);
                    for target_len in 0..=stored_len {
                        let rows_survive =
                            (target_len.saturating_sub(window)..target_len).all(|position| {
                                cache.k_even[(position % ring.capacity) * EVEN_ODD_ROW]
                                    == marker(position)
                            });
                        assert_eq!(
                            ring_rewind_fits(stored_len, target_len, 0, &ring),
                            rows_survive,
                            "window={window} slack={slack} stored={stored_len} target={target_len}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(
            cases,
            6 * 7 * (0..=30).map(|stored| stored + 1).sum::<usize>()
        );
    }

    #[test]
    fn a_recurrent_layer_blocks_any_rewind_but_not_an_extension() {
        let mut state = PrefixState {
            ids: vec![1, 2, 3, 4],
            layer_caches: vec![LayerCacheState::Ssm(SsmLayerCache::new(4, 4))],
            cached_len: 4,
        };
        let widths = vec![LayerPadRowWidths::Ssm {
            conv_history_len: 4,
            state_len: 4,
        }];

        assert_eq!(
            state.rewind_to(3, 0, &widths),
            Err(MissReason::UnrewindableLayer)
        );
        assert_eq!(state.rewind_to(4, 0, &widths), Ok(()));
    }

    /// A refused rewind leaves the entry in the cache for the next request.
    #[test]
    fn a_refused_rewind_keeps_the_entry_cached() {
        let mut cache = PromptCache::new();
        cache.store(
            CacheEntry::new(gemma_like_state(40), base_key()),
            &enabled_config(),
        );
        let diverging_prompt: Vec<u32> = (0..35).chain([900, 901]).collect();

        let (state, report) = cache.take_best(
            &diverging_prompt,
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        assert!(state.is_none());
        assert_eq!(
            report.miss,
            Some(MissReason::RingSlackExceeded {
                rewind_rows: 5,
                slack_rows: SLACK,
            })
        );
        let extension: Vec<u32> = (0..40).chain([900]).collect();
        let (state, report) =
            cache.take_best(&extension, &base_key(), &gemma_like_widths(), ANY_OVERLAP);
        assert!(state.is_some());
        assert_eq!(report.path, CachePath::Extend);
    }

    fn ring_marker(state: &PrefixState, position: usize) -> f32 {
        match &state.layer_caches[0] {
            LayerCacheState::Attention(cache) => {
                let ring = cache.ring_geometry().expect("layer 0 is a ring");
                cache.k_even[(position % ring.capacity) * EVEN_ODD_ROW]
            }
            _ => 0.0,
        }
    }

    fn entry_with_checkpoints(stored_len: usize, positions: &[usize]) -> CacheEntry {
        let mut entry = CacheEntry::new(gemma_like_state(stored_len), base_key());
        for &position in positions {
            let checkpoint = RingCheckpoint::capture(&gemma_like_state(position))
                .expect("a state with a ring layer is captured");
            entry.insert_checkpoint(checkpoint, 8);
        }
        entry
    }

    fn prompt_diverging_after(shared: usize, new_tokens: usize) -> Vec<u32> {
        (0..shared as u32)
            .chain((0..new_tokens as u32).map(|offset| 900 + offset))
            .collect()
    }

    /// Writes real rows for positions `entry.cached_len..to`, the way a
    /// prefill after a restore would.
    fn prefill_to(state: &mut PrefixState, to: usize) {
        for position in state.cached_len..to {
            let value = marker(position);
            for layer in &mut state.layer_caches {
                if let LayerCacheState::Attention(cache) = layer {
                    cache.append_at(
                        position,
                        &[value; EVEN_ODD_ROW],
                        &[value; EVEN_ODD_ROW],
                        &[value; V_ROW],
                    );
                }
            }
            state.ids.push(position as u32);
        }
        state.cached_len = to;
    }

    /// R4: a rewind of 20 tokens does not fit the 4-row slack, so the entry
    /// restores the newest checkpoint at or before the shared prefix (32 of
    /// 16, 32, 48 for a prefix of 40) and the request prefills from there.
    #[test]
    fn a_rewind_past_the_slack_restores_the_nearest_checkpoint_at_or_before_the_prefix() {
        let mut cache = PromptCache::new();
        cache.store(entry_with_checkpoints(60, &[16, 32, 48]), &enabled_config());

        let (entry, report) = cache.take_best(
            &prompt_diverging_after(40, 2),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        let entry = entry.expect("a checkpoint at 32 serves a prefix of 40");
        assert_eq!(report.path, CachePath::Checkpoint);
        assert_eq!(report.lcp, 40);
        assert_eq!(report.reused_tokens, 32);
        assert_eq!(report.prefilled_tokens, 42 - 32);
        assert_eq!(entry.state.cached_len, 32);
        assert_eq!(entry.state.ids.len(), 32);
        assert_eq!(entry.checkpoint_positions(), vec![16, 32]);
        assert_eq!(entry.restored_at, 32);
        assert!(
            (24..32).all(|position| ring_marker(&entry.state, position) == marker(position)),
            "the window ending at the checkpoint reads back its own rows"
        );
        assert_eq!(full_layer_rows(&entry.state), 32);
    }

    #[test]
    fn a_checkpoint_exactly_at_the_prefix_is_used_whole() {
        let mut cache = PromptCache::new();
        cache.store(entry_with_checkpoints(60, &[16, 32, 48]), &enabled_config());

        let (entry, report) = cache.take_best(
            &prompt_diverging_after(48, 2),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        assert!(entry.is_some());
        assert_eq!(report.path, CachePath::Checkpoint);
        assert_eq!(report.reused_tokens, 48);
        assert_eq!(report.prefilled_tokens, 50 - 48);
    }

    /// With every checkpoint after the shared prefix nothing can stand in for
    /// the overwritten rows: the request prefills in full and the entry stays.
    #[test]
    fn a_prefix_before_every_checkpoint_is_a_miss_that_keeps_the_entry() {
        let mut cache = PromptCache::new();
        cache.store(entry_with_checkpoints(60, &[32, 48]), &enabled_config());

        let (entry, report) = cache.take_best(
            &prompt_diverging_after(20, 2),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        assert!(entry.is_none());
        assert_eq!(
            report.miss,
            Some(MissReason::RingSlackExceeded {
                rewind_rows: 40,
                slack_rows: SLACK,
            })
        );
        assert_eq!(report.prefilled_tokens, 22);
        let (entry, _) = cache.take_best(
            &prompt_diverging_after(48, 2),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );
        assert!(entry.is_some(), "the refused entry is still cached");
    }

    /// After a restore at 32 and a prefill to 44 the rings hold rows from 24
    /// on and nothing valid before: a prefix of 20 fits no slack and has no
    /// row to read, so it falls back to the checkpoint at 16, not to the
    /// stale rows a bare slack check would have accepted at a prefix of 30.
    #[test]
    fn rows_a_restore_replaced_are_never_read_by_a_later_rewind() {
        let mut restored = entry_with_checkpoints(60, &[16, 32, 48]);
        restored
            .resume(40, &gemma_like_widths())
            .expect("the checkpoint at 32 restores");
        prefill_to(&mut restored.state, 44);
        let mut cache = PromptCache::new();
        cache.store(restored, &enabled_config());

        let stale = prompt_diverging_after(30, 2);
        let (entry, report) =
            cache.take_best(&stale, &base_key(), &gemma_like_widths(), ANY_OVERLAP);

        let entry = entry.expect("the checkpoint at 16 serves a prefix of 30");
        assert_eq!(report.path, CachePath::Checkpoint);
        assert_eq!(report.reused_tokens, 16);
        assert_eq!(entry.restored_at, 16);
        assert_eq!(entry.checkpoint_positions(), vec![16]);
    }

    #[test]
    fn the_rows_after_a_restore_are_reported_stale_to_a_rewind_without_checkpoints() {
        let mut restored = entry_with_checkpoints(60, &[32]);
        restored
            .resume(40, &gemma_like_widths())
            .expect("the checkpoint at 32 restores");
        prefill_to(&mut restored.state, 36);

        let refusal = restored
            .state
            .rewind_to(30, restored.restored_at, &gemma_like_widths());

        assert_eq!(refusal, Err(MissReason::RingRowsStale { restored_at: 32 }));
        assert_eq!(restored.state.cached_len, 36);
    }

    /// A rewind that fits the slack keeps the checkpoints at or before the
    /// target and drops the ones that described the abandoned tail.
    #[test]
    fn a_rewind_inside_the_slack_drops_only_the_checkpoints_past_the_target() {
        let mut cache = PromptCache::new();
        cache.store(
            entry_with_checkpoints(60, &[16, 32, 48, 58]),
            &enabled_config(),
        );

        let (entry, report) = cache.take_best(
            &prompt_diverging_after(57, 3),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        let entry = entry.expect("a 3-row rewind fits the slack");
        assert_eq!(report.path, CachePath::Rewind);
        assert_eq!(report.reused_tokens, 57);
        assert_eq!(entry.checkpoint_positions(), vec![16, 32, 48]);
        assert_eq!(entry.restored_at, 0);
    }

    #[test]
    fn an_extension_keeps_every_checkpoint() {
        let mut cache = PromptCache::new();
        cache.store(entry_with_checkpoints(60, &[16, 32, 48]), &enabled_config());

        let (entry, report) = cache.take_best(
            &prompt_diverging_after(60, 0)
                .into_iter()
                .chain([7])
                .collect::<Vec<_>>(),
            &base_key(),
            &gemma_like_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(
            entry.expect("extended").checkpoint_positions(),
            vec![16, 32, 48]
        );
    }

    /// The cap pins the earliest checkpoint and evicts the oldest of the
    /// rest as newer ones arrive.
    #[test]
    fn checkpoints_past_the_cap_evict_the_oldest_but_the_first() {
        let mut entry = CacheEntry::new(gemma_like_state(60), base_key());
        for position in [8, 16, 24, 32, 40] {
            let checkpoint = RingCheckpoint::capture(&gemma_like_state(position))
                .expect("a state with a ring layer is captured");
            entry.insert_checkpoint(checkpoint, 3);
        }

        assert_eq!(entry.checkpoint_positions(), vec![8, 32, 40]);
    }

    #[test]
    fn checkpoint_bytes_count_against_the_entry_and_the_budget() {
        let plain = CacheEntry::new(gemma_like_state(60), base_key());
        let checkpointed = entry_with_checkpoints(60, &[16, 32]);
        let one_checkpoint = RingCheckpoint::capture(&gemma_like_state(32)).expect("captured");

        assert_eq!(
            checkpointed.byte_len(),
            plain.byte_len()
                + RingCheckpoint::capture(&gemma_like_state(16))
                    .expect("captured")
                    .byte_len()
                + one_checkpoint.byte_len()
        );
        let tight = PromptCacheConfig {
            byte_budget: plain.byte_len() as u64,
            ..enabled_config()
        };
        assert_eq!(PromptCache::new().store(checkpointed, &tight), None);
    }

    fn naive_common_prefix(left: &[u32], right: &[u32]) -> usize {
        let mut shared = 0;
        while shared < left.len() && shared < right.len() && left[shared] == right[shared] {
            shared += 1;
        }
        shared
    }

    /// AC1: over 10,000 generated pairs the prefix length equals a plain
    /// index scan. A six-token vocabulary and a generated shared head make
    /// every case a near-collision, the shape real prompts that differ late
    /// have; the case count is asserted so a zero-case run cannot pass.
    #[test]
    fn longest_common_prefix_equals_a_naive_scan_over_10000_generated_pairs() {
        const CASES: usize = 10_000;
        let mut runner = TestRunner::new(Config {
            cases: CASES as u32,
            failure_persistence: None,
            ..Config::default()
        });
        let executed = Cell::new(0_usize);
        let token_run = || vec(0_u32..6, 0..64);

        runner
            .run(
                &(token_run(), token_run(), token_run()),
                |(head, left_tail, right_tail)| {
                    let left: Vec<u32> = head.iter().chain(&left_tail).copied().collect();
                    let right: Vec<u32> = head.iter().chain(&right_tail).copied().collect();
                    let expected = naive_common_prefix(&left, &right);

                    assert_eq!(longest_common_prefix(&left, &right), expected);
                    assert_eq!(longest_common_prefix(&right, &left), expected);
                    assert!(expected >= head.len());
                    executed.set(executed.get() + 1);
                    Ok(())
                },
            )
            .expect("the shared-prefix scan must agree with the naive scan on every pair");

        assert_eq!(executed.get(), CASES);
    }
    const GEMMA_HEADER: [u32; 5] = [2, 105, 2364, 107, 9259];
    const LLAMA_DEFAULT_MILLI: u32 = 100;

    fn conversation(header: &[u32], body_start: u32, body_len: u32) -> Vec<u32> {
        header
            .iter()
            .copied()
            .chain((0..body_len).map(|offset| body_start + offset))
            .collect()
    }

    /// llama-server's rule (`server-context.cpp:1571`): the prefix must
    /// cover strictly more than the threshold of the incoming prompt.
    #[proxima::test]
    #[case::exactly_at_the_threshold_is_refused(5, 50, 100, false)]
    #[case::one_token_past_the_threshold_is_taken(6, 50, 100, true)]
    #[case::five_tokens_of_a_long_prompt_is_refused(5, 1000, 100, false)]
    #[case::zero_threshold_takes_any_overlap(1, 1000, 0, true)]
    #[case::half_the_prompt_under_a_high_threshold_is_refused(500, 1000, 600, false)]
    #[case::no_overlap_is_refused_at_any_threshold(0, 1000, 0, false)]
    async fn an_entry_is_reusable_only_above_the_similarity_threshold(
        #[case] lcp: usize,
        #[case] prompt_len: usize,
        #[case] min_similarity_milli: u32,
        #[case] reusable: bool,
    ) {
        let stored_len = lcp + 40;

        assert_eq!(
            entry_is_reusable(lcp, stored_len, prompt_len, min_similarity_milli),
            reusable
        );
    }

    #[test]
    fn an_extension_is_reused_whatever_its_share_of_the_prompt() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&GEMMA_HEADER), &enabled_config());
        let long_prompt = conversation(&GEMMA_HEADER, 5000, 995);

        let (taken, report) = cache.take_best(
            &long_prompt,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );

        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(report.reused_tokens, 5);
        assert_eq!(taken.expect("the extension is served").state.cached_len, 5);
    }

    /// The thrash: an unrelated prompt sharing five header tokens with a long
    /// conversation used to rewind it to those five rows.
    #[test]
    fn an_unrelated_prompt_sharing_a_header_misses_and_leaves_the_long_entry_intact() {
        let conversation_a = conversation(&GEMMA_HEADER, 1000, 1000);
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&conversation_a), &enabled_config());
        let unrelated = conversation(&GEMMA_HEADER, 7000, 200);

        let (taken, report) = cache.take_best(
            &unrelated,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );

        assert!(taken.is_none());
        assert_eq!(report.miss, Some(MissReason::BelowSimilarity));
        assert_eq!(report.prefilled_tokens, unrelated.len());
        let next_turn = conversation(&conversation_a, 3000, 20);
        let (kept, next_report) = cache.take_best(
            &next_turn,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );
        assert_eq!(next_report.path, CachePath::Extend);
        assert_eq!(
            kept.expect("conversation A survived").state.cached_len,
            1005
        );
    }

    /// Two conversations alternating through a cache: each misses once, builds
    /// its own entry, and from then on every turn extends its own.
    #[test]
    fn two_alternating_conversations_each_keep_their_own_entry() {
        let mut cache = PromptCache::new();
        let config = enabled_config();
        let mut turns = [
            conversation(&GEMMA_HEADER, 1000, 400),
            conversation(&GEMMA_HEADER, 7000, 400),
        ];
        let mut paths = Vec::new();

        for round in 0..3_u32 {
            for turn in &mut turns {
                let (taken, report) =
                    cache.take_best(turn, &base_key(), &shared_widths(), LLAMA_DEFAULT_MILLI);
                paths.push(report.path);
                let mut entry = taken.unwrap_or_else(|| CacheEntry::empty(base_key()));
                entry.state = PrefixState {
                    ids: turn.clone(),
                    layer_caches: vec![LayerCacheState::SharedFromLayer],
                    cached_len: turn.len(),
                };
                cache.store(entry, &config);
                turn.extend((0..50).map(|offset| 20_000 + round * 100 + offset));
            }
        }

        assert_eq!(
            paths,
            vec![
                CachePath::Miss,
                CachePath::Miss,
                CachePath::Extend,
                CachePath::Extend,
                CachePath::Extend,
                CachePath::Extend,
            ]
        );
    }

    #[test]
    fn a_zero_threshold_reproduces_the_unconditional_longest_prefix_reuse() {
        let mut cache = PromptCache::new();
        cache.store(
            state_with_ids(&conversation(&GEMMA_HEADER, 1000, 1000)),
            &enabled_config(),
        );

        let (taken, report) = cache.take_best(
            &conversation(&GEMMA_HEADER, 7000, 200),
            &base_key(),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert_eq!(report.path, CachePath::Rewind);
        assert_eq!(taken.expect("rewound").state.cached_len, 5);
    }

    #[test]
    fn the_most_similar_eligible_entry_serves_the_request() {
        let mut cache = PromptCache::new();
        cache.store(
            state_with_ids(&conversation(&GEMMA_HEADER, 1000, 60)),
            &enabled_config(),
        );
        let mut near = conversation(&GEMMA_HEADER, 2000, 20);
        near.extend(conversation(&[], 9000, 40));
        cache.store(state_with_ids(&near), &enabled_config());
        let prompt = conversation(&near[..45], 8000, 5);

        let (taken, report) =
            cache.take_best(&prompt, &base_key(), &shared_widths(), LLAMA_DEFAULT_MILLI);

        assert_eq!(report.lcp, 45);
        assert_eq!(
            taken.expect("the near entry").state.ids,
            near[..45].to_vec()
        );
    }

    /// A request and a prewarm each take their entry out of the cache for the
    /// whole of their forward, so two parties never write the same rows: the
    /// second sees nothing until the first stores its entry back.
    #[test]
    fn an_entry_taken_by_one_party_is_not_offered_to_another_until_stored_back() {
        let mut cache = PromptCache::new();
        let prompt = conversation(&GEMMA_HEADER, 1000, 200);
        cache.store(state_with_ids(&prompt), &enabled_config());
        let next_turn = conversation(&prompt, 3000, 20);

        let (request_entry, request_report) = cache.take_best(
            &next_turn,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );
        let (prewarm_entry, prewarm_report) = cache.take_for_prewarm(
            &next_turn,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );
        cache.store(
            request_entry.expect("the request held it"),
            &enabled_config(),
        );
        let (after_store, after_report) = cache.take_for_prewarm(
            &next_turn,
            &base_key(),
            &shared_widths(),
            LLAMA_DEFAULT_MILLI,
        );

        assert_eq!(request_report.path, CachePath::Extend);
        assert!(prewarm_entry.is_none());
        assert_eq!(prewarm_report.miss, Some(MissReason::Empty));
        assert!(after_store.is_some());
        assert_eq!(after_report.path, CachePath::Extend);
    }

    fn indexed_config(block_tokens: u32) -> PromptCacheConfig {
        PromptCacheConfig {
            block_tokens,
            max_entries: 64,
            ..enabled_config()
        }
    }

    /// The scan `take_best` replaced: every entry's common prefix against the
    /// prompt, the reusable ones, the longest, the most recently stored among
    /// equals.
    fn scan_reference(
        stored: &[Vec<u32>],
        prompt: &[u32],
        min_similarity_milli: u32,
    ) -> Option<(usize, usize)> {
        stored
            .iter()
            .enumerate()
            .map(|(position, ids)| (position, naive_common_prefix(ids, prompt), ids.len()))
            .filter(|&(_, lcp, stored_len)| {
                entry_is_reusable(lcp, stored_len, prompt.len(), min_similarity_milli)
            })
            .max_by_key(|&(position, lcp, _)| (lcp, position))
            .map(|(position, lcp, _)| (position, lcp))
    }

    /// AC17: over 10,000 generated caches and prompts the block index nominates
    /// the same entry, at the same prefix length, as the scan it replaced. A
    /// four-token vocabulary and four-token blocks make most cases share
    /// several whole blocks and diverge inside one, which is where an index
    /// that drops an entry between two levels would disagree.
    #[test]
    fn the_block_index_picks_the_entry_and_prefix_the_old_scan_picked_over_10000_cases() {
        const CASES: usize = 10_000;
        let mut runner = TestRunner::new(Config {
            cases: CASES as u32,
            failure_persistence: None,
            ..Config::default()
        });
        let executed = Cell::new(0_usize);
        let tokens = |shortest: usize, longest: usize| vec(0_u32..4, shortest..longest);

        runner
            .run(
                &(
                    vec(tokens(0, 40), 0..7),
                    tokens(0, 12),
                    tokens(1, 40),
                    0_u32..400,
                ),
                |(entries, shared_head, prompt_tail, min_similarity_milli)| {
                    let config = indexed_config(4);
                    let mut cache = PromptCache::new();
                    let stored: Vec<Vec<u32>> = entries
                        .iter()
                        .map(|tail| {
                            shared_head
                                .iter()
                                .chain(tail)
                                .copied()
                                .collect::<Vec<u32>>()
                        })
                        .filter(|ids| !ids.is_empty())
                        .collect();
                    stored.iter().for_each(|ids| {
                        cache.store(state_with_ids(ids), &config);
                    });
                    let prompt: Vec<u32> =
                        shared_head.iter().chain(&prompt_tail).copied().collect();
                    let expected = scan_reference(&stored, &prompt, min_similarity_milli)
                        .map(|(position, lcp)| (position, lcp.min(prompt.len() - 1)))
                        .filter(|&(_, resume)| resume > 0);

                    let (taken, report) = cache.take_best(
                        &prompt,
                        &base_key(),
                        &shared_widths(),
                        min_similarity_milli,
                    );

                    match expected {
                        Some((position, resume)) => {
                            let taken = taken.expect("the scan found a reusable entry");
                            assert_eq!(taken.state.ids, stored[position][..resume].to_vec());
                            assert_eq!(report.lcp, resume);
                        }
                        None => assert!(taken.is_none(), "{report:?}"),
                    }
                    executed.set(executed.get() + 1);
                    Ok(())
                },
            )
            .expect("the index must agree with the scan on every case");

        assert_eq!(executed.get(), CASES);
    }

    #[test]
    fn a_block_whose_hash_collides_is_never_reused_on_the_hash_alone() {
        let mut cache = PromptCache::new();
        cache.store(
            state_with_ids(&[1, 2, 3, 4, 5, 6, 7, 8]),
            &indexed_config(4),
        );
        let impostor = cache.index.walk(&[9, 9, 9, 9], |_, _| false);

        assert!(impostor.levels.is_empty());
    }

    /// The false-positive rate of the default filter against blocks no entry
    /// holds, and that it never misses a block it was given.
    #[test]
    fn the_default_bloom_filter_has_no_false_negatives_and_a_low_false_positive_rate() {
        let standard = PromptCacheConfig::standard();
        let mut rng = fastrand::Rng::with_seed(7);
        let block = standard.block_tokens as usize;
        let held: Vec<u32> = (0..128 * block).map(|_| rng.u32(0..262_144)).collect();
        let absent: Vec<u32> = (0..10_000 * block).map(|_| rng.u32(0..262_144)).collect();
        let bloom = BlockBloom::of(
            &held,
            block,
            standard.bloom_bits_per_entry,
            standard.bloom_hashes,
        );

        let missed = content_hashes(&held, block)
            .iter()
            .filter(|hash| !bloom.maybe_contains(**hash))
            .count();
        let false_positives = content_hashes(&absent, block)
            .iter()
            .filter(|hash| bloom.maybe_contains(**hash))
            .count();

        println!(
            "BLOOM held_blocks=128 absent_blocks=10000 false_positives={false_positives} bits={} hashes={}",
            standard.bloom_bits_per_entry, standard.bloom_hashes
        );
        assert_eq!(missed, 0);
        assert!(
            false_positives < 100,
            "{false_positives} of 10000 absent blocks matched"
        );
    }

    #[test]
    fn the_bloom_candidates_name_the_prompt_blocks_an_earlier_entry_holds_after_its_prefix_diverged()
     {
        let mut cache = PromptCache::new();
        let config = indexed_config(4);
        let history: Vec<u32> = (100..140).collect();
        cache.store(state_with_ids(&history), &config);
        let squashed: Vec<u32> = [1, 2, 3, 4]
            .iter()
            .chain(&history[8..24])
            .chain(&[7, 7, 7, 7])
            .copied()
            .collect();

        let candidates = cache.bloom_candidates(&squashed, &base_key(), 4);

        assert_eq!(candidates.entries, 1);
        let blocks: Vec<usize> = candidates.blocks.iter().map(|(_, block)| *block).collect();
        assert!(blocks.starts_with(&[1, 2, 3, 4]), "{blocks:?}");
    }

    fn random_conversation(rng: &mut fastrand::Rng, tokens: usize) -> Vec<u32> {
        std::iter::once(2)
            .chain((1..tokens).map(|_| rng.u32(4..262_144)))
            .collect()
    }

    fn median_nanos(mut samples: Vec<u128>) -> u128 {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    /// AC17, cost: the time one lookup takes with 4 and with 256 cached
    /// 1,024-token conversations (every one opening with the same BOS token),
    /// for a prompt that extends one of them, for one that shares nothing past
    /// BOS, and for the scan the index replaced. Median of 2,000 lookups each.
    #[test]
    #[ignore = "timing probe: run alone, it prints the numbers it measured"]
    fn block_index_lookup_cost_against_the_entry_count() {
        const LOOKUPS: usize = 2000;
        let config = PromptCacheConfig {
            byte_budget: 1 << 32,
            max_entries: 1024,
            ..indexed_config(64)
        };
        for entry_count in [4_usize, 256] {
            let mut rng = fastrand::Rng::with_seed(11);
            let stored: Vec<Vec<u32>> = (0..entry_count)
                .map(|_| random_conversation(&mut rng, 1024))
                .collect();
            let mut cache = PromptCache::new();
            stored
                .iter()
                .for_each(|ids| assert!(cache.store(state_with_ids(ids), &config).is_some()));
            let extends: Vec<u32> = stored[entry_count / 2]
                .iter()
                .copied()
                .chain((0..100).map(|_| rng.u32(4..262_144)))
                .collect();
            let unrelated = random_conversation(&mut rng, 1124);
            let timed_take = |cache: &mut PromptCache, prompt: &[u32]| -> Vec<u128> {
                (0..LOOKUPS)
                    .map(|_| {
                        let started = std::time::Instant::now();
                        let (taken, _) =
                            cache.take_best(prompt, &base_key(), &shared_widths(), 100);
                        let nanos = started.elapsed().as_nanos();
                        if let Some(entry) = taken {
                            let mut whole = entry;
                            whole.state.ids = stored[entry_count / 2].clone();
                            whole.state.cached_len = whole.state.ids.len();
                            cache.store(whole, &config);
                        }
                        nanos
                    })
                    .collect()
            };
            let hit = median_nanos(timed_take(&mut cache, &extends));
            let miss = median_nanos(timed_take(&mut cache, &unrelated));
            let scan = median_nanos(
                (0..LOOKUPS)
                    .map(|_| {
                        let started = std::time::Instant::now();
                        std::hint::black_box(scan_reference(&stored, &extends, 100));
                        started.elapsed().as_nanos()
                    })
                    .collect(),
            );
            println!(
                "BLOCK_INDEX entries={entry_count} index_hit_ns={hit} index_miss_ns={miss} scan_hit_ns={scan} profile={}",
                if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                }
            );
        }
    }

    fn key_under(
        config: &ServingConfig,
        rope: RopeScaling,
        slack: usize,
        offset: usize,
    ) -> CacheKey {
        CacheKey::of(config, false, rope, slack, offset)
    }

    /// Two requests that differ in one input the rows depend on never share an
    /// entry: the entry stored under the base key is not offered to the
    /// request, which reports `ConfigMismatch` and leaves the entry in place.
    #[proxima::test]
    #[case::rope_linear_scaling(|config| ServingConfig { rope_scaling: Some(RopeScaling::Linear { factor: 2.0 }), ..config })]
    #[case::numeric_policy_bit_exact(|config| ServingConfig { numeric_policy: NumericPolicy::bit_exact(), ..config })]
    #[case::numeric_policy_fast(|config| ServingConfig { numeric_policy: NumericPolicy::fast(), ..config })]
    #[case::exact_activations(|config| ServingConfig { exact_activations: !config.exact_activations, ..config })]
    #[case::cached_attention_fusion(|config| ServingConfig { cached_attention_fusion: !config.cached_attention_fusion, ..config })]
    #[case::gated_delta_net_fusion(|config| ServingConfig { gated_delta_net_fusion: !config.gated_delta_net_fusion, ..config })]
    #[case::moe_topk_fusion(|config| ServingConfig { moe_topk_fusion: !config.moe_topk_fusion, ..config })]
    #[case::gdn_prefill_backend(|config| ServingConfig { gdn_prefill_backend: GdnPrefillBackend::Mlx, ..config })]
    #[case::qwen35moe_pre_gather(|config| ServingConfig { qwen35moe_pre_gather: !config.qwen35moe_pre_gather, ..config })]
    #[case::qwen35moe_monolithic_all_low(|config| ServingConfig { qwen35moe_monolithic_all_low: !config.qwen35moe_monolithic_all_low, ..config })]
    #[case::qwen35moe_monolithic_high_mmap(|config| ServingConfig { qwen35moe_monolithic_high_mmap: !config.qwen35moe_monolithic_high_mmap, ..config })]
    #[case::qwen35moe_layer_window(|config| ServingConfig { qwen35moe_layer_window: config.qwen35moe_layer_window + 1, ..config })]
    #[case::qwen35moe_residency_budget(|config| ServingConfig { qwen35moe_residency_budget_bytes: config.qwen35moe_residency_budget_bytes + (1 << 30), ..config })]
    async fn a_request_differing_in_a_row_affecting_config_field_misses(
        #[case] change: fn(ServingConfig<'static>) -> ServingConfig<'static>,
    ) {
        let base_config = ServingConfig::default();
        let changed_config = change(base_config);
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());

        let (taken, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259],
            &key_under(
                &changed_config,
                changed_config.rope_scaling.unwrap_or(RopeScaling::None),
                0,
                0,
            ),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert!(
            taken.is_none(),
            "an entry built under other settings was offered"
        );
        assert_eq!(report.miss, Some(MissReason::ConfigMismatch));
        assert!(
            cache.stored_bytes() > 0,
            "the mismatched entry must stay for its own config"
        );
    }

    #[proxima::test]
    #[case::cpu_versus_metal_route(true, RopeScaling::None, 0, 0)]
    #[case::rope_yarn_scaling(false, RopeScaling::yarn(4.0, 32_768), 0, 0)]
    #[case::speculative_ring_slack(false, RopeScaling::None, 8, 0)]
    #[case::ring_write_offset(false, RopeScaling::None, 0, 3)]
    async fn a_request_differing_in_a_derived_input_misses(
        #[case] uses_gpu: bool,
        #[case] rope: RopeScaling,
        #[case] slack: usize,
        #[case] offset: usize,
    ) {
        let config = ServingConfig::default();
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());

        let (taken, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259],
            &CacheKey::of(&config, uses_gpu, rope, slack, offset),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert!(taken.is_none());
        assert_eq!(report.miss, Some(MissReason::ConfigMismatch));
    }

    /// Fields the rows do not depend on: sampling, chunk widths, the memory
    /// gates, scheduling. A request differing only in these reuses the entry.
    #[proxima::test]
    #[case::sampling_temperature_and_seed(|config| ServingConfig { temperature: 0.8, seed: 7, ..config })]
    #[case::ubatch_and_batch_widths(|config| ServingConfig { ubatch_size: 8, batch_size: 64, ..config })]
    #[case::kv_bucket_tokens(|config| ServingConfig { kv_bucket_tokens: 64, ..config })]
    #[case::command_buffer_chunks(|config| ServingConfig { command_buffer_chunks: 4, ..config })]
    #[case::context_length_resolution(|config| ServingConfig { context_length: ContextLength::Within(4096), ..config })]
    #[case::prompt_cache_policy(|config| ServingConfig { prompt_cache: PromptCacheConfig { max_entries: 9, ..config.prompt_cache }, ..config })]
    async fn a_request_differing_only_in_a_row_independent_field_shares_the_entry(
        #[case] change: fn(ServingConfig<'static>) -> ServingConfig<'static>,
    ) {
        let base_config = ServingConfig::default();
        let changed_config = change(base_config);
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());

        let (taken, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259],
            &key_under(&changed_config, RopeScaling::None, 0, 0),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert!(
            taken.is_some(),
            "a row-independent setting must not evict reuse"
        );
        assert_eq!(report.path, CachePath::Extend);
    }

    #[test]
    fn entries_under_two_configs_coexist_and_each_serves_its_own() {
        let linear = ServingConfig {
            rope_scaling: Some(RopeScaling::Linear { factor: 2.0 }),
            ..ServingConfig::default()
        };
        let linear_key = key_under(&linear, RopeScaling::Linear { factor: 2.0 }, 0, 0);
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364, 107]), &enabled_config());
        let mut linear_entry = state_with_ids(&[2, 105, 2364, 107]);
        linear_entry.key = linear_key;
        cache.store(linear_entry, &enabled_config());
        let prompt = [2, 105, 2364, 107, 9259];

        let (base_taken, base_report) =
            cache.take_best(&prompt, &base_key(), &shared_widths(), ANY_OVERLAP);
        let (linear_taken, linear_report) =
            cache.take_best(&prompt, &linear_key, &shared_widths(), ANY_OVERLAP);

        assert_eq!(base_taken.expect("base entry").key, base_key());
        assert_eq!(linear_taken.expect("linear entry").key, linear_key);
        assert_eq!(
            (base_report.path, linear_report.path),
            (CachePath::Extend, CachePath::Extend)
        );
    }

    #[test]
    fn an_empty_cache_reports_empty_not_config_mismatch() {
        let mut cache = PromptCache::new();

        let (_, report) = cache.take_best(&[2, 105], &base_key(), &shared_widths(), ANY_OVERLAP);

        assert_eq!(report.miss, Some(MissReason::Empty));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn a_request_under_another_metal_math_mode_misses() {
        let safe = ServingConfig {
            math_mode: omega::MathMode::Safe,
            ..ServingConfig::default()
        };
        let relaxed = ServingConfig {
            math_mode: omega::MathMode::Relaxed,
            ..ServingConfig::default()
        };
        let mut cache = PromptCache::new();
        let mut entry = state_with_ids(&[2, 105, 2364, 107]);
        entry.key = key_under(&safe, RopeScaling::None, 0, 0);
        cache.store(entry, &enabled_config());

        let (taken, report) = cache.take_best(
            &[2, 105, 2364, 107, 9259],
            &key_under(&relaxed, RopeScaling::None, 0, 0),
            &shared_widths(),
            ANY_OVERLAP,
        );

        assert!(taken.is_none());
        assert_eq!(report.miss, Some(MissReason::ConfigMismatch));
    }
}
