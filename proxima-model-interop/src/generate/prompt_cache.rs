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
    /// The best entry needs a rewind this build cannot apply yet.
    RewindUnsupported,
}

impl MissReason {
    /// The telemetry label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::NoCommonPrefix => "no_common_prefix",
            Self::RewindUnsupported => "rewind_unsupported",
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

impl PrefixState {
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
    /// the cache, ready to be resumed from, or reports why none can be.
    pub(super) fn take_best(&mut self, prompt_ids: &[u32]) -> (Option<PrefixState>, CacheReport) {
        let best = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (index, longest_common_prefix(&entry.ids, prompt_ids)))
            .max_by_key(|(_, lcp)| *lcp);
        let report = match best {
            None => CacheReport::miss(prompt_ids.len(), MissReason::Empty),
            Some((_, 0)) => CacheReport::miss(prompt_ids.len(), MissReason::NoCommonPrefix),
            Some((index, lcp)) => return self.resume_at(index, lcp, prompt_ids.len()),
        };
        self.last_report = Some(report);
        (None, report)
    }

    fn resume_at(
        &mut self,
        index: usize,
        lcp: usize,
        prompt_tokens: usize,
    ) -> (Option<PrefixState>, CacheReport) {
        let resume_len = lcp.min(prompt_tokens.saturating_sub(1));
        if resume_len != self.entries[index].cached_len {
            let report = CacheReport::miss(prompt_tokens, MissReason::RewindUnsupported);
            self.last_report = Some(report);
            return (None, report);
        }
        let state = self.entries.remove(index);
        let report = CacheReport {
            lcp: resume_len,
            prefilled_tokens: prompt_tokens - resume_len,
            path: CachePath::Extend,
            miss: None,
        };
        self.last_report = Some(report);
        (Some(state), report)
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
    ) -> (Option<PrefixState>, CacheReport) {
        let (state, report) = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take_best(prompt_ids);
        debug!(
            cache_lcp = report.lcp as u64,
            cache_reused_tokens = report.reused_tokens() as u64,
            cache_prefilled_tokens = report.prefilled_tokens as u64,
            cache_path = report.path.as_str(),
            cache_miss_reason = report.miss.map_or("none", MissReason::as_str),
            "prompt cache lookup"
        );
        (state, report)
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
        let (cache_seed, report) = self.prompt_cache_lookup(&ids);
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

        let (state, report) = cache.take_best(&[2, 105, 2364, 107, 9259, 106, 107, 105, 4368]);

        assert_eq!(state.expect("entry reused").cached_len, 5);
        assert_eq!(report.path, CachePath::Extend);
        assert_eq!(report.lcp, 5);
        assert_eq!(report.prefilled_tokens, 4);
    }

    #[test]
    fn a_prompt_sharing_no_first_token_reports_no_common_prefix() {
        let mut cache = PromptCache::new();
        cache.store(state_with_ids(&[2, 105, 2364]), &enabled_config());

        let (state, report) = cache.take_best(&[7, 105, 2364]);

        assert!(state.is_none());
        assert_eq!(report.miss, Some(MissReason::NoCommonPrefix));
        assert_eq!(report.prefilled_tokens, 3);
    }

    #[test]
    fn an_empty_cache_reports_empty_and_prefills_the_whole_prompt() {
        let mut cache = PromptCache::new();

        let (state, report) = cache.take_best(&[2, 105, 2364]);

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

        let (_, report) = cache.take_best(&[2, 105, 2364, 107]);

        assert_eq!(report.lcp + report.prefilled_tokens, 4);
        assert!(report.prefilled_tokens >= 1);
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
            cache.take_best(&[1, 1, 9]).1.miss,
            Some(MissReason::NoCommonPrefix)
        );
        assert_eq!(cache.take_best(&[2, 2, 9]).1.path, CachePath::Extend);
        assert_eq!(cache.take_best(&[3, 3, 9]).1.path, CachePath::Extend);
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
        assert_eq!(cache.take_best(&[5, 6, 7, 8, 9]).1.path, CachePath::Extend);
    }
}
