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
//! than waiting.

use core::ops::ControlFlow;
use std::sync::PoisonError;

use proxima_telemetry::debug;

use super::*;

/// Length of the shared token prefix of `left` and `right`.
pub(super) fn longest_common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left_id, right_id)| left_id == right_id)
        .count()
}

/// How a request used the cache (spec R8's `cache_path`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePath {
    /// The whole stored sequence is a prefix of the prompt: nothing rewound.
    Extend,
    /// The prompt diverges inside the stored sequence: every layer was
    /// rewound to the shared prefix first.
    Rewind,
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
            Self::Miss => "miss",
        }
    }
}

/// Why a request found nothing to reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    /// The cache holds no entry.
    Empty,
    /// No entry shares even the first token with the prompt.
    NoCommonPrefix,
    /// A sliding-window layer would need rows its ring has already
    /// overwritten: rewinding `rewind_rows` tokens exceeds the `slack_rows`
    /// the ring keeps past its window.
    RingSlackExceeded {
        /// Tokens the rewind would drop from the stored sequence.
        rewind_rows: usize,
        /// Rows the ring keeps past its window.
        slack_rows: usize,
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
            Self::NoCommonPrefix => "no_common_prefix",
            Self::RingSlackExceeded { .. } => "ring_slack_exceeded",
            Self::UnrewindableLayer => "unrewindable_layer",
        }
    }
}

/// What one request took from the cache: the numbers behind spec R8's
/// `cache_lcp`, `cache_reused_tokens`, `cache_prefilled_tokens`,
/// `cache_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReport {
    /// Tokens of the prompt that matched an entry and were reused.
    pub lcp: usize,
    /// Tokens of the prompt the forward pass had to prefill.
    pub prefilled_tokens: usize,
    /// How the entry was used.
    pub path: CachePath,
    /// Why nothing was reused, when `path` is [`CachePath::Miss`].
    pub miss: Option<MissReason>,
}

impl CacheReport {
    /// Tokens served from the cache without a forward pass.
    #[must_use]
    pub const fn reused_tokens(&self) -> usize {
        self.lcp
    }

    fn miss(prompt_tokens: usize, reason: MissReason) -> Self {
        Self {
            lcp: 0,
            prefilled_tokens: prompt_tokens,
            path: CachePath::Miss,
            miss: Some(reason),
        }
    }
}

/// Whether a ring layer that has seen `stored_len` positions still holds
/// every row a window ending at `target_len` reads. Position `p` lives in row
/// `p % capacity`, so it is overwritten once position `p + capacity` has been
/// written; the window needs positions `target_len - window ..` onward, and
/// the first of them survives iff `first_needed + capacity >= stored_len`.
/// For `target_len >= window` that is `stored_len - target_len <= slack`.
fn ring_rewind_fits(stored_len: usize, target_len: usize, ring: &KvRing) -> bool {
    let first_needed = target_len.saturating_sub(ring.window);
    ring.write_offset == 0 && (target_len == 0 || first_needed + ring.capacity >= stored_len)
}

impl PrefixState {
    /// The first reason this state cannot be rewound to `target_len` tokens,
    /// `None` when every layer can.
    fn rewind_refusal(
        &self,
        target_len: usize,
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
                    .and_then(|ring| self.ring_refusal(target_len, ring)),
                (LayerCacheState::Attention(_), LayerPadRowWidths::Attention { .. }) => None,
                _ => Some(MissReason::UnrewindableLayer),
            })
    }

    fn ring_refusal(&self, target_len: usize, ring: &KvRing) -> Option<MissReason> {
        if ring.write_offset != 0 {
            return Some(MissReason::UnrewindableLayer);
        }
        (!ring_rewind_fits(self.cached_len, target_len, ring)).then_some(
            MissReason::RingSlackExceeded {
                rewind_rows: self.cached_len - target_len,
                slack_rows: ring.capacity - ring.window,
            },
        )
    }

    /// Rewinds this state to its first `target_len` tokens: full-attention
    /// layers truncate their rows, ring layers keep theirs (their rows are
    /// addressed by position, so rewinding is `cached_len` going back --
    /// [`LayerCache::truncate`]'s own doc) and are checked to still hold the
    /// window, and a shared-KV layer follows its donor. All-or-nothing: on
    /// `Err` the state is untouched.
    pub(super) fn rewind_to(
        &mut self,
        target_len: usize,
        widths: &[LayerPadRowWidths],
    ) -> Result<(), MissReason> {
        if target_len >= self.cached_len {
            return Ok(());
        }
        if let Some(reason) = self.rewind_refusal(target_len, widths) {
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

    /// Host bytes this state holds: its ids plus every layer's cache rows.
    pub(super) fn byte_len(&self) -> usize {
        let float_bytes = |rows: &[&Vec<f32>]| -> usize {
            rows.iter().map(|rows| rows.len() * size_of::<f32>()).sum()
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
        layer_bytes + self.ids.len() * size_of::<u32>()
    }
}

/// The entries one [`LoadedModel`] keeps, least recently used first.
pub(super) struct PromptCache {
    entries: Vec<PrefixState>,
    last_report: Option<CacheReport>,
}

impl PromptCache {
    pub(super) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_report: None,
        }
    }

    /// Takes the entry sharing the longest prefix with `prompt_ids` out of
    /// the cache, rewound to that prefix and ready to be resumed from, or
    /// reports why none can be. `widths` are the model's per-layer row
    /// widths ([`LoadedModel::declared_layer_cache_names_and_widths`]).
    pub(super) fn take_best(
        &mut self,
        prompt_ids: &[u32],
        widths: &[LayerPadRowWidths],
    ) -> (Option<PrefixState>, CacheReport) {
        let best = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (index, longest_common_prefix(&entry.ids, prompt_ids)))
            .max_by_key(|(_, lcp)| *lcp);
        let resume = best.map(|(index, lcp)| (index, lcp.min(prompt_ids.len().saturating_sub(1))));
        let outcome = match resume {
            None => Err(MissReason::Empty),
            Some((_, 0)) => Err(MissReason::NoCommonPrefix),
            Some((index, resume_len)) => self.resume_at(index, resume_len, widths),
        };
        let (state, report) = match outcome {
            Ok((state, path)) => {
                let resume_len = state.cached_len;
                let report = CacheReport {
                    lcp: resume_len,
                    prefilled_tokens: prompt_ids.len() - resume_len,
                    path,
                    miss: None,
                };
                (Some(state), report)
            }
            Err(reason) => (None, CacheReport::miss(prompt_ids.len(), reason)),
        };
        self.last_report = Some(report);
        (state, report)
    }

    /// Takes entry `index` out and rewinds it to `resume_len`; an entry that
    /// cannot be rewound stays in the cache for the next request.
    fn resume_at(
        &mut self,
        index: usize,
        resume_len: usize,
        widths: &[LayerPadRowWidths],
    ) -> Result<(PrefixState, CachePath), MissReason> {
        let mut state = self.entries.remove(index);
        let path = if resume_len >= state.cached_len {
            CachePath::Extend
        } else {
            CachePath::Rewind
        };
        match state.rewind_to(resume_len, widths) {
            Ok(()) => Ok((state, path)),
            Err(reason) => {
                self.entries.insert(index, state);
                Err(reason)
            }
        }
    }

    /// Puts `state` back as the most recently used entry, then evicts from
    /// the least recently used end until the entry count and byte budget in
    /// `config` hold. A state larger than the whole budget is not stored.
    pub(super) fn store(&mut self, state: PrefixState, config: &PromptCacheConfig) -> usize {
        let budget = usize::try_from(config.byte_budget).unwrap_or(usize::MAX);
        if state.cached_len == 0 || state.layer_caches.is_empty() || state.byte_len() > budget {
            return self.entries.len();
        }
        self.entries.push(state);
        let max_entries = config.max_entries as usize;
        while self.entries.len() > max_entries || self.total_bytes() > budget {
            self.entries.remove(0);
        }
        self.entries.len()
    }

    fn total_bytes(&self) -> usize {
        self.entries.iter().map(PrefixState::byte_len).sum()
    }

    pub(super) const fn last_report(&self) -> Option<CacheReport> {
        self.last_report
    }
}

impl LoadedModel<'_> {
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

    /// Looks `prompt_ids` up in the cache and emits spec R8's per-request
    /// telemetry for the outcome.
    pub(super) fn prompt_cache_lookup(
        &self,
        prompt_ids: &[u32],
    ) -> Result<(Option<PrefixState>, CacheReport), InteropError> {
        let (_, widths) = self.declared_layer_cache_names_and_widths()?;
        let (state, report) = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take_best(prompt_ids, &widths);
        debug!(
            cache_lcp = report.lcp as u64,
            cache_reused_tokens = report.reused_tokens() as u64,
            cache_prefilled_tokens = report.prefilled_tokens as u64,
            cache_path = report.path.as_str(),
            cache_miss_reason = report.miss.map_or("none", MissReason::as_str),
            "prompt cache lookup"
        );
        Ok((state, report))
    }

    /// Hands a finished request's state to the cache.
    pub(super) fn prompt_cache_store(&self, state: PrefixState, config: &PromptCacheConfig) {
        let cached_tokens = state.cached_len;
        let entries = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .store(state, config);
        debug!(
            cache_stored_tokens = cached_tokens as u64,
            cache_entries = entries as u64,
            "prompt cache store"
        );
    }

    /// The decode loop behind every entry point, with the prompt cache in
    /// front of it when `serving_config.prompt_cache` is enabled and nothing
    /// asks the loop to observe the prefill itself (a caller-owned seed, a
    /// forced token stream, or a logits/node-values sink all need the full
    /// prefill to run).
    ///
    /// On the cached path the returned [`PrefixState`] is hollow: the real
    /// one now lives in the cache, and the only caller that wants the state
    /// back ([`Self::prefill_prefix`]) switches the cache off for its call.
    #[allow(clippy::too_many_arguments)] // mirrors `run_decode_loop_from_ids`, which it forwards to
    pub(super) fn run_decode_loop_through_cache(
        &self,
        ids: Vec<u32>,
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
        let (cache_seed, report) = self.prompt_cache_lookup(&ids)?;
        let (generated_ids, text, stopped_by_eos, final_state) = self.run_decode_loop_from_ids(
            ids[report.lcp..].to_vec(),
            max_tokens,
            serving_config,
            runtime,
            None,
            logits_sink,
            node_values_sink,
            on_token,
            cache_seed,
            true,
            speculative_stats,
            forced_draft_width,
        )?;
        self.prompt_cache_store(final_state, &config);
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
    use super::*;

    fn state_with_ids(ids: &[u32]) -> PrefixState {
        PrefixState {
            ids: ids.to_vec(),
            layer_caches: vec![LayerCacheState::SharedFromLayer],
            cached_len: ids.len(),
        }
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
            &shared_widths(),
        );

        assert_eq!(state.expect("entry reused").cached_len, 5);
        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(report.lcp, 5);
        assert_eq!(report.prefilled_tokens, 4);
    }

    #[test]
    fn a_prompt_sharing_no_first_token_reports_no_common_prefix() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364]), &enabled_config());

        let (state, report) = cache.take_best(&[7, 105, 2364], &shared_widths());

        assert!(state.is_none());
        assert_eq!(report.miss, Some(MissReason::NoCommonPrefix));
        assert_eq!(report.prefilled_tokens, 3);
    }

    #[test]
    fn an_empty_cache_reports_empty_and_prefills_the_whole_prompt() {
        let mut cache = PromptCache::new();

        let (state, report) = cache.take_best(&[2, 105, 2364], &shared_widths());

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

        let (state, report) = cache.take_best(&[2, 105, 2364, 107], &shared_widths());

        assert_eq!(report.path, CachePath::Rewind);
        assert_eq!(report.lcp, 3);
        assert_eq!(report.prefilled_tokens, 1);
        assert_eq!(state.expect("entry reused").ids, vec![2, 105, 2364]);
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
            cache.take_best(&[1, 1, 9], &shared_widths()).1.miss,
            Some(MissReason::NoCommonPrefix)
        );
        assert_eq!(
            cache.take_best(&[2, 2, 9], &shared_widths()).1.path,
            CachePath::Extend
        );
        assert_eq!(
            cache.take_best(&[3, 3, 9], &shared_widths()).1.path,
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

        assert_eq!(entries, 0);
    }

    #[test]
    fn the_byte_budget_evicts_older_entries_to_fit_a_newer_one() {
        let config = PromptCacheConfig {
            byte_budget: 24,
            ..enabled_config()
        };
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[1, 2, 3, 4]), &config);

        let entries = cache.store(state_with_ids(&[5, 6, 7, 8]), &config);

        assert_eq!(entries, 1);
        assert_eq!(
            cache.take_best(&[5, 6, 7, 8, 9], &shared_widths()).1.path,
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
            .rewind_to(36, &gemma_like_widths())
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

        let refusal = state.rewind_to(35, &gemma_like_widths());

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
            .rewind_to(3, &gemma_like_widths())
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
                            ring_rewind_fits(stored_len, target_len, &ring),
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
            state.rewind_to(3, &widths),
            Err(MissReason::UnrewindableLayer)
        );
        assert_eq!(state.rewind_to(4, &widths), Ok(()));
    }

    /// A refused rewind leaves the entry in the cache for the next request.
    #[test]
    fn a_refused_rewind_keeps_the_entry_cached() {
        let mut cache = PromptCache::new();
        cache.store(gemma_like_state(40), &enabled_config());
        let diverging_prompt: Vec<u32> = (0..35).chain([900, 901]).collect();

        let (state, report) = cache.take_best(&diverging_prompt, &gemma_like_widths());

        assert!(state.is_none());
        assert_eq!(
            report.miss,
            Some(MissReason::RingSlackExceeded {
                rewind_rows: 5,
                slack_rows: SLACK,
            })
        );
        let extension: Vec<u32> = (0..40).chain([900]).collect();
        let (state, report) = cache.take_best(&extension, &gemma_like_widths());
        assert!(state.is_some());
        assert_eq!(report.path, CachePath::Extend);
    }
}
