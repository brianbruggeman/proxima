//! The closed `Drafter` enum the decode loop drives (`begin`/`draft`/`accept`)
//! for every llama.cpp n-gram speculation type enabled in a
//! [`crate::serving::SpeculativeConfig`], composing `proxima_tokenizer::draft`'s
//! five sans-IO drafters (`speculative-decode-llama-parity/TASKS.md` slice 9)
//! -- box-free per `AGENTS.md`'s workspace default: a closed, compile-time-known
//! set of implementations is exactly the discriminated-enum-plus-match case,
//! never a `Box<dyn Trait>`.
//!
//! `common_speculative_draft` (`common/speculative.cpp:2802-2885`) is the
//! incumbent [`DrafterSet::draft`] reproduces: every enabled drafter's own
//! `draft()` runs in llama's fixed priority order
//! ([`crate::serving::SpeculativeTypeSet::iter_priority_order`]) until one
//! yields a non-empty draft; later drafters in the order are never called
//! that step, matching `common_speculative_draft`'s own early `break` once
//! `n_drafting` reaches zero. `common_speculative_accept` (`:2887-2921`) is
//! [`DrafterSet::accept`]: only the drafting implementation's own
//! `accept(n_accepted)` runs. Every real caller of `common_speculative_accept`
//! upstream (`tools/server/server-context.cpp:3885-3900`, guarded by
//! `!slot.spec_draft.empty()`) only invokes it when a draft actually happened,
//! so "notify every other enabled implementation with `is_other = true`" is
//! observationally a no-op for every impl this crate ports: `ngram-simple`
//! and `ngram-cache`'s own `accept` overrides ignore `is_other` entirely (dead
//! no-ops), and `ngram-map`/`ngram-mod`'s own overrides return immediately on
//! `is_other` before touching any state -- so "notify others" and "do nothing"
//! are the same observable behaviour for this crate's five wired types, and
//! [`DrafterSet`] does the latter directly rather than giving [`Drafter`] a
//! method whose body is always empty.

use alloc::vec::Vec;

use proxima_tokenizer::draft::{
    DEFAULT_N_DRAFT, NgramCacheState, NgramMap, NgramMapConfig, NgramMod, NgramModConfig,
    NgramSimpleConfig, ngram_cache_state_draft, ngram_map_accept, ngram_map_begin,
    ngram_map_draft, ngram_mod_accept, ngram_mod_begin, ngram_mod_draft, ngram_simple_draft,
};

use crate::serving::{SpeculativeConfig, SpeculativeType};

/// One llama.cpp n-gram speculation implementation, own state included.
/// See this module's own doc for the priority/accept semantics
/// [`DrafterSet`] drives it under.
pub(crate) enum Drafter {
    NgramSimple(NgramSimpleConfig),
    NgramMapK(NgramMap),
    NgramMapK4v(NgramMap),
    NgramMod(NgramMod),
    NgramCache(NgramCacheState),
}

impl Drafter {
    pub(crate) fn type_id(&self) -> SpeculativeType {
        match self {
            Self::NgramSimple(_) => SpeculativeType::NgramSimple,
            Self::NgramMapK(_) => SpeculativeType::NgramMapK,
            Self::NgramMapK4v(_) => SpeculativeType::NgramMapK4v,
            Self::NgramMod(_) => SpeculativeType::NgramMod,
            Self::NgramCache(_) => SpeculativeType::NgramCache,
        }
    }

    /// `common_speculative_begin` per impl (`:2765-2775`): `ngram-map`
    /// trains its key index and `ngram-mod` trains its hash table over the
    /// prompt; `ngram-simple` and `ngram-cache` are no-ops, matching each
    /// impl's own `begin` override in `common/speculative.cpp`.
    pub(crate) fn begin(&mut self, prompt: &[u32]) {
        match self {
            Self::NgramSimple(_) | Self::NgramCache(_) => {}
            Self::NgramMapK(map) | Self::NgramMapK4v(map) => ngram_map_begin(map, prompt),
            Self::NgramMod(state) => ngram_mod_begin(state, prompt),
        }
    }

    /// `history`/`sampled`/`out` follow every drafter in
    /// `proxima_tokenizer::draft`'s own caller-owned-buffer discipline.
    pub(crate) fn draft(&mut self, history: &[u32], sampled: u32, out: &mut Vec<u32>) {
        match self {
            Self::NgramSimple(config) => ngram_simple_draft(config, history, sampled, out),
            Self::NgramMapK(map) | Self::NgramMapK4v(map) => {
                ngram_map_draft(map, history, sampled, out);
            }
            Self::NgramMod(state) => ngram_mod_draft(state, history, sampled, out),
            Self::NgramCache(state) => {
                ngram_cache_state_draft(state, history, sampled, DEFAULT_N_DRAFT, out);
            }
        }
    }

    /// `common_speculative_accept(.., is_other = false)`: the drafting
    /// impl's own bookkeeping update.
    pub(crate) fn accept(&mut self, n_accepted: u16) {
        match self {
            Self::NgramSimple(_) | Self::NgramCache(_) => {}
            Self::NgramMapK(map) | Self::NgramMapK4v(map) => ngram_map_accept(map, n_accepted),
            Self::NgramMod(state) => ngram_mod_accept(state, n_accepted),
        }
    }
}

/// `slot.get_n_draft_max()` (`tools/server/server-context.cpp:453-472`):
/// `n_ctx - prompt.n_tokens() - 2`, then `min(.., n_remaining() - 1)`. The
/// `- 2` leaves room for the sampled token and a context shift; the
/// `- 1` is the sampled token itself, because a verify of `k` drafts emits
/// up to `k + 1` tokens. `cached_len` excludes the sampled token (llama's
/// "slot.prompt is not yet expanded with the `id`"); `generated` counts
/// tokens already produced, the sampled one included (`n_remaining() =
/// n_predict - n_gen`). Saturates at `0`, which means "do not speculate".
pub(crate) fn draft_limit_for_step(
    context_length: usize,
    cached_len: usize,
    max_tokens: usize,
    generated: usize,
) -> usize {
    let context_room = context_length.saturating_sub(cached_len).saturating_sub(2);
    let budget_room = max_tokens.saturating_sub(generated).saturating_sub(1);
    context_room.min(budget_room)
}

/// The set of [`Drafter`]s driven each generation, in llama's own fixed
/// priority order -- built once per decode call from
/// [`SpeculativeConfig::speculative_types`]
/// ([`crate::serving::apply_serving_config`] already rejects every member
/// that is not one of these five n-gram types before a call reaches here).
pub(crate) struct DrafterSet {
    drafters: Vec<Drafter>,
    active: Option<usize>,
}

impl DrafterSet {
    /// `max_context_len` sizes [`NgramMap::new`]/[`NgramCacheState::new`]'s
    /// own presized storage -- the caller-supplied bound both drafters' own
    /// docs document (this crate's `ServingConfig::context_length`).
    pub(crate) fn build(config: &SpeculativeConfig<'_>, max_context_len: usize) -> Self {
        let mut drafters = Vec::new();
        for type_id in config.speculative_types.iter_priority_order() {
            let drafter = match type_id {
                SpeculativeType::NgramSimple => Drafter::NgramSimple(NgramSimpleConfig {
                    size_n: config.ngram_simple.size_n,
                    size_m: config.ngram_simple.size_m,
                }),
                SpeculativeType::NgramMapK => Drafter::NgramMapK(NgramMap::new(
                    NgramMapConfig {
                        size_key: config.ngram_map_k.size_n,
                        size_value: config.ngram_map_k.size_m,
                        key_only: true,
                        min_hits: config.ngram_map_k.min_hits,
                    },
                    max_context_len,
                )),
                SpeculativeType::NgramMapK4v => Drafter::NgramMapK4v(NgramMap::new(
                    NgramMapConfig {
                        size_key: config.ngram_map_k4v.size_n,
                        size_value: config.ngram_map_k4v.size_m,
                        key_only: false,
                        min_hits: config.ngram_map_k4v.min_hits,
                    },
                    max_context_len,
                )),
                SpeculativeType::NgramMod => Drafter::NgramMod(NgramMod::new(NgramModConfig {
                    n_match: config.ngram_mod.n_match,
                    n_max: config.ngram_mod.n_max,
                    n_min: config.ngram_mod.n_min,
                })),
                SpeculativeType::NgramCache => {
                    Drafter::NgramCache(NgramCacheState::new(max_context_len))
                }
                // `apply_serving_config` rejects every other member before a
                // call reaches this constructor.
                _ => continue,
            };
            drafters.push(drafter);
        }
        Self {
            drafters,
            active: None,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.drafters.is_empty()
    }

    pub(crate) fn begin(&mut self, prompt: &[u32]) {
        for drafter in &mut self.drafters {
            drafter.begin(prompt);
        }
    }

    /// `common_speculative_draft`'s own priority walk: the first drafter,
    /// in priority order, to fill `out` non-empty wins; [`Self::active`]
    /// remembers which one so [`Self::accept`] can route to it. `out` is
    /// cleared by the losing drafters' own `clear()`-then-refill contract,
    /// so a caller sees either the winner's draft or an empty buffer.
    ///
    /// `n_max` is `dp.n_max` (`common/speculative.cpp:2847-2851`): the
    /// winning draft is truncated to it after the drafter ran, and
    /// `n_max == 0` never drafts at all (`server-context.cpp:2985`,
    /// `if (n_draft_max > 0)`).
    pub(crate) fn draft(
        &mut self,
        history: &[u32],
        sampled: u32,
        n_max: usize,
        out: &mut Vec<u32>,
    ) {
        self.active = None;
        out.clear();
        if n_max == 0 {
            return;
        }
        for (index, drafter) in self.drafters.iter_mut().enumerate() {
            drafter.draft(history, sampled, out);
            out.truncate(n_max);
            if !out.is_empty() {
                self.active = Some(index);
                return;
            }
        }
    }

    /// `common_speculative_accept`, called only when [`Self::draft`] set
    /// [`Self::active`] this step -- see this module's own doc for why that
    /// is the exact set of steps the incumbent's own callers invoke it on.
    pub(crate) fn accept(&mut self, n_accepted: u16) {
        if let Some(index) = self.active.take()
            && let Some(drafter) = self.drafters.get_mut(index)
        {
            drafter.accept(n_accepted);
        }
    }

    /// Which [`SpeculativeType`] drafted the span [`Self::accept`] is about
    /// to score -- read before calling [`Self::accept`], which consumes
    /// [`Self::active`].
    pub(crate) fn active_type(&self) -> Option<SpeculativeType> {
        self.active.map(|index| self.drafters[index].type_id())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::serving::{NgramMapParams, SpeculativeConfig, SpeculativeType, SpeculativeTypeSet};

    use super::{DrafterSet, draft_limit_for_step};

    /// A history both `ngram-simple` and `ngram-map-k` could match (small
    /// sizes so the default `size_n=12, size_m=48` guard does not need a
    /// 61-token history) proves the priority walk returns the FIRST
    /// drafter to produce a non-empty draft (`ngram-simple`, highest
    /// priority), not `ngram-map-k`, even though both were enabled and
    /// `begin` trained `ngram-map-k`'s own index over the same history.
    #[test]
    fn priority_walk_prefers_the_highest_priority_drafter_that_matches() {
        let small = NgramMapParams {
            size_n: 3,
            size_m: 3,
            min_hits: 1,
        };
        let config = SpeculativeConfig {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple)
                .insert(SpeculativeType::NgramMapK),
            ngram_simple: small,
            ngram_map_k: small,
            ..SpeculativeConfig::none()
        };
        let mut set = DrafterSet::build(&config, 4096);
        assert!(!set.is_empty());

        // `ngram_simple.rs`'s own happy-path fixture: `history`'s trailing
        // `[1, 2]` plus `sampled = 3` completes the pattern `[1, 2, 3]`,
        // which recurs earlier at index 1, followed by real tokens `[4, 5,
        // 9]` -- a genuine draft for both `ngram-simple` and (once trained
        // by `begin`) `ngram-map-k`.
        let history: Vec<u32> = vec![0u32, 1, 2, 3, 4, 5, 9, 9, 1, 2];
        let sampled = 3u32;

        set.begin(&history);
        let mut out = Vec::new();
        set.draft(&history, sampled, usize::MAX, &mut out);
        assert!(!out.is_empty(), "this history has a real repeated pattern to draft from");
        assert_eq!(
            set.active_type(),
            Some(SpeculativeType::NgramSimple),
            "ngram-simple is highest priority and this history matches its own pattern"
        );
    }

    /// An empty set (no speculator enabled) never drafts anything and
    /// [`DrafterSet::is_empty`] reports it truthfully.
    #[test]
    fn empty_config_yields_an_empty_drafter_set() {
        let set = DrafterSet::build(&SpeculativeConfig::none(), 4096);
        assert!(set.is_empty());
    }

    /// The shipped default builds exactly one drafter (`ngram-simple`), so
    /// the decode loop constructs a drafter by default; `none()` (test
    /// above) constructs none, the off switch.
    #[test]
    fn default_config_yields_a_non_empty_drafter_set_with_ngram_simple() {
        let mut set = DrafterSet::build(&SpeculativeConfig::default(), 4096);
        assert!(!set.is_empty());

        let history: Vec<u32> = (0..70u32).chain(0..12u32).collect();
        set.begin(&history);
        let mut out = Vec::new();
        set.draft(&history, 12, usize::MAX, &mut out);
        assert!(!out.is_empty(), "size_n 12 pattern recurs in this history");
        assert_eq!(set.active_type(), Some(SpeculativeType::NgramSimple));
    }

    /// [`DrafterSet::accept`] is a no-op when nothing drafted this step
    /// (`active` is `None`) -- must not panic.
    #[test]
    fn accept_with_no_active_drafter_does_not_panic() {
        let config = SpeculativeConfig {
            speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramMod),
            ..SpeculativeConfig::none()
        };
        let mut set = DrafterSet::build(&config, 4096);
        set.accept(0);
        assert_eq!(set.active_type(), None);
    }

    fn default_set_and_history() -> (DrafterSet, Vec<u32>) {
        let mut set = DrafterSet::build(&SpeculativeConfig::default(), 4096);
        let history: Vec<u32> = (0..70u32).chain(0..12u32).collect();
        set.begin(&history);
        (set, history)
    }

    #[test]
    fn drafter_offering_48_tokens_is_truncated_to_the_limit() {
        let (mut set, history) = default_set_and_history();
        let mut out = Vec::new();
        set.draft(&history, 12, usize::MAX, &mut out);
        assert_eq!(out.len(), 48, "ngram-simple offers size_m=48 unclamped");

        let limit = draft_limit_for_step(4096, 82, 48, 43);
        set.draft(&history, 12, limit, &mut out);

        assert_eq!(limit, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(set.active_type(), Some(SpeculativeType::NgramSimple));
    }

    #[test]
    fn one_token_remaining_never_speculates() {
        let (mut set, history) = default_set_and_history();
        let mut out = vec![99u32];

        let limit = draft_limit_for_step(4096, 82, 48, 47);
        set.draft(&history, 12, limit, &mut out);

        assert_eq!(limit, 0);
        assert!(out.is_empty());
        assert_eq!(set.active_type(), None);
    }

    #[test]
    fn draft_limit_for_step_takes_the_tighter_of_context_and_budget() {
        assert_eq!(draft_limit_for_step(4096, 100, 48, 0), 47);
        assert_eq!(draft_limit_for_step(4096, 100, 48, 42), 5);
        assert_eq!(draft_limit_for_step(4096, 100, 48, 48), 0);
        assert_eq!(draft_limit_for_step(4096, 100, 48, 60), 0);
        assert_eq!(draft_limit_for_step(110, 100, 48, 0), 8);
        assert_eq!(draft_limit_for_step(102, 100, 48, 0), 0);
        assert_eq!(draft_limit_for_step(100, 100, 48, 0), 0);
        assert_eq!(draft_limit_for_step(90, 100, 48, 0), 0);
    }
}
