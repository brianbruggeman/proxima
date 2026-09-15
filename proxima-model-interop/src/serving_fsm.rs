//! The serving control plane as one sans-IO state machine over the algebra,
//! replacing `generate.rs`'s env/flag-gated branches (`split_prefill`, the
//! pre-gather arm, `run_decode_loop_placed_kv` vs the two-range loop, the
//! one-evaluation flag) with transitions that consume the old variant.
//!
//! Every transition performs exactly one program evaluation --
//! [`ServingState::advance_prefill`]: `M` rows in one pass,
//! [`ServingState::advance_decode`]: one row,
//! [`ServingState::accept`] (post-[`ServingState::enter_verify`]): `K` rows
//! with logits at every row -- or placement copies only
//! ([`ServingState::rollback`]). `Cache` is the per-layer recurrent/KV state
//! (`generate.rs`'s `LayerCacheState`/`SsmLayerCache` once migrated onto
//! this); it lives inside the variant a transition returns, not in a cache
//! object patched from outside.
//!
//! `Verify`'s draft-scoring ([`ServingState::accept`]) and `Rollback`'s
//! restore body ([`ServingState::rollback`]) are real now: [`draft_prompt_lookup`]
//! proposes up to `K` tokens by n-gram match over the prompt+generated id
//! history (no second model), [`ServingState::accept`] scores them against
//! one `K`-row verify evaluation's per-row greedy argmax and per-row cache
//! placements (no replay -- the caller already produced one `Cache` per
//! row, `accept` only selects the right one), and [`ServingState::rollback`]
//! resumes decoding from whichever placement `accept` chose.
//!
//! Wiring this onto the real `qwen35moe` forward program is blocked on the
//! pre-existing, already-documented `prefill_width` bug (`proxima-tensor`'s
//! `spec.rs:8541`, tracked by the `#[ignore]`d
//! `one_shot_program_matches_sequential_decode_on_synthetic_layers` test in
//! `qwen35moe_one_shot_vs_sequential_parity.rs`): the M-row evaluation this
//! step was told to reuse as the verify pass does not yet match sequential
//! decode for M>1, so a real-program oracle here would pass or fail on a
//! kernel bug, not on this module's own logic. The tests below prove the
//! FSM's control flow (drafting, acceptance counting, cache placement,
//! rollback-to-resume) against a fake target function instead.

// not yet driven by generate.rs (the migration is the next step); the
// walkthrough test below is this module's only caller until then.
#![allow(dead_code)]

use alloc::vec::Vec;

use thiserror::Error;

/// The one control-plane type this plan admits: no existing proxima
/// primitive (pipe, cache, or config) expresses "which evaluation shape is
/// legal right now" cheaper than an enum whose transitions consume `self`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ServingState<Cache> {
    /// `positions`: the prompt token ids to place in one `M`-row program
    /// evaluation. `cache` starts empty -- prefill is the only state that
    /// does not inherit recurrent state from a predecessor.
    Prefill { positions: Vec<u32>, cache: Cache },
    /// `last`: the most recently accepted token id, fed back as the next
    /// single-row evaluation's input.
    Decode { last: u32, cache: Cache },
    /// `draft`: candidate token ids proposed by a speculator, scored in one
    /// `K`-row evaluation with logits returned at every row. `snapshot` is
    /// the pre-draft `cache`, restored verbatim on rejection.
    Verify {
        draft: Vec<u32>,
        snapshot: Cache,
        cache: Cache,
    },
    /// `n`: how many leading draft tokens the verifier accepted; `next` is
    /// the token id decoding resumes from (the confirmed final draft token
    /// when `n == draft.len()`, otherwise the target's own row-`n`
    /// prediction that replaced the rejected draft token).
    Accept { n: usize, next: u32, cache: Cache },
    /// `to`: the token id decoding resumes from after `snapshot` is
    /// restored in place of a rejected draft's advanced cache.
    Rollback { snapshot: Cache, to: u32 },
    /// Terminal: no further evaluation is legal.
    Done { cache: Cache },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ServingFsmError {
    /// A transition was invoked on a variant it does not apply to (for
    /// example, [`ServingState::advance_decode`] called on `Prefill`).
    #[error("serving fsm: {attempted} is not legal from the current state")]
    IllegalTransition { attempted: &'static str },

    /// [`ServingState::accept`] and [`ServingState::rollback`] (draft
    /// scoring and cache restore) land in a later step; every `Verify` this
    /// step reaches returns this rather than a fabricated result.
    #[error("serving fsm: {operation} is not implemented yet")]
    NotSupported { operation: &'static str },
}

/// Prompt-lookup / n-gram drafting: find the most recent earlier occurrence
/// of `history`'s own last `ngram` tokens elsewhere in `history` (the
/// prompt+generated id sequence so far), and draft up to `max_k` tokens
/// (`max_k` up to 5) that followed that occurrence -- no second model, pure
/// string match over ids already produced. Returns an empty draft (never
/// entering `Verify`) when `history` is too short or no earlier occurrence
/// exists.
pub(crate) fn draft_prompt_lookup(history: &[u32], ngram: usize, max_k: usize) -> Vec<u32> {
    if ngram == 0 || max_k == 0 || history.len() <= ngram {
        return Vec::new();
    }
    let needle = &history[history.len() - ngram..];
    let search_end = history.len() - ngram;
    for start in (0..search_end).rev() {
        if &history[start..start + ngram] == needle {
            let match_end = start + ngram;
            let available = history.len() - match_end;
            let take = available.min(max_k);
            return history[match_end..match_end + take].to_vec();
        }
    }
    Vec::new()
}

impl<Cache> ServingState<Cache> {
    /// Enter the state machine at `Prefill` with an empty `cache`.
    pub(crate) fn start(positions: Vec<u32>, cache: Cache) -> Self {
        Self::Prefill { positions, cache }
    }

    /// `Prefill { positions } -> Decode { last }`: the one `M`-row program
    /// evaluation prefill performs produced `next_token` and left `cache`
    /// holding the resulting recurrent/KV state.
    pub(crate) fn advance_prefill(
        self,
        next_token: u32,
        cache: Cache,
    ) -> Result<Self, ServingFsmError> {
        match self {
            Self::Prefill { .. } => Ok(Self::Decode {
                last: next_token,
                cache,
            }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "advance_prefill",
            }),
        }
    }

    /// `Decode { last } -> Decode { last }`: one single-row evaluation.
    pub(crate) fn advance_decode(
        self,
        next_token: u32,
        cache: Cache,
    ) -> Result<Self, ServingFsmError> {
        match self {
            Self::Decode { .. } => Ok(Self::Decode {
                last: next_token,
                cache,
            }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "advance_decode",
            }),
        }
    }

    /// `Decode { last } -> Verify { draft, snapshot }`: the current `cache`
    /// becomes `snapshot`, restorable verbatim if the draft is rejected.
    pub(crate) fn enter_verify(self, draft: Vec<u32>) -> Result<Self, ServingFsmError>
    where
        Cache: Clone,
    {
        match self {
            Self::Decode { cache, .. } => Ok(Self::Verify {
                draft,
                snapshot: cache.clone(),
                cache,
            }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "enter_verify",
            }),
        }
    }

    /// `Verify { draft, snapshot } -> Accept { n, next } | Rollback { snapshot, to }`:
    /// scores the `K`-row verify evaluation's per-row greedy argmax
    /// (`row_tokens[i]` predicts the token that follows having consumed
    /// input row `i`, where row `0` follows `last` and row `i>0` follows
    /// `draft[i - 1]`) against `draft` itself. `row_caches[i]` is the exact
    /// `Cache` the same evaluation produced at row `i` -- accepting `n`
    /// draft tokens means resuming from `row_caches[n]` verbatim (a
    /// placement copy the caller already computed), never replaying
    /// anything. `n == draft.len()`: every drafted token matched, `Accept`
    /// resumes from the confirmed final draft token. `n < draft.len()`:
    /// `draft[n]` was wrong, so this returns `Rollback` carrying the
    /// correct placement and the target's own token in its place.
    pub(crate) fn accept(self, row_tokens: &[u32], row_caches: Vec<Cache>) -> Result<Self, ServingFsmError> {
        match self {
            Self::Verify { draft, .. } if draft.is_empty() || row_tokens.len() != draft.len() || row_caches.len() != draft.len() => {
                Err(ServingFsmError::IllegalTransition { attempted: "accept" })
            }
            Self::Verify { draft, .. } => {
                let accepted = draft.iter().zip(row_tokens.iter()).take_while(|(drafted, predicted)| drafted == predicted).count();
                let mut placements = row_caches.into_iter();
                let placement_index = if accepted == draft.len() { accepted - 1 } else { accepted };
                let Some(cache) = placements.nth(placement_index) else {
                    // unreachable given the length guard above; a bad
                    // caller-supplied `row_caches` fails closed instead of panicking
                    return Err(ServingFsmError::IllegalTransition { attempted: "accept" });
                };
                if accepted == draft.len() {
                    Ok(Self::Accept { n: accepted, next: draft[accepted - 1], cache })
                } else {
                    Ok(Self::Rollback { snapshot: cache, to: row_tokens[accepted] })
                }
            }
            _ => Err(ServingFsmError::IllegalTransition { attempted: "accept" }),
        }
    }

    /// `Accept { n, next, cache } -> Decode { last }`: no evaluation, just
    /// unwrapping the placement `accept` already chose.
    pub(crate) fn resume(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Accept { next, cache, .. } => Ok(Self::Decode { last: next, cache }),
            _ => Err(ServingFsmError::IllegalTransition { attempted: "resume" }),
        }
    }

    /// `Rollback { snapshot, to } -> Decode { last }`: placement copy only,
    /// no program evaluation -- `snapshot` is whichever row's `Cache`
    /// `accept` selected as the correct resume point.
    pub(crate) fn rollback(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Rollback { snapshot, to } => Ok(Self::Decode { last: to, cache: snapshot }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "rollback",
            }),
        }
    }

    /// Any state `-> Done`: the caller has stopped requesting further
    /// evaluations (max tokens reached, eos observed, error surfaced).
    pub(crate) fn finish(self) -> Self {
        let cache = match self {
            Self::Prefill { cache, .. }
            | Self::Decode { cache, .. }
            | Self::Verify { cache, .. }
            | Self::Accept { cache, .. }
            | Self::Done { cache } => cache,
            Self::Rollback { snapshot, .. } => snapshot,
        };
        Self::Done { cache }
    }
}

#[cfg(test)]
// every transition here returns a typed error the assertions above it already
// prove unreachable; `expect` documents which one, `unwrap_used`/`expect_used`
// stay denied outside `#[cfg(test)]`
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A minimal stand-in for `generate.rs`'s `LayerCacheState`/
    /// `SsmLayerCache`: just enough to prove the FSM threads recurrent
    /// state through transitions rather than patching a cache from outside.
    #[derive(Debug, Clone, PartialEq)]
    struct FakeCache {
        conv_history_len: usize,
        kv_len: usize,
    }

    impl FakeCache {
        fn empty() -> Self {
            Self {
                conv_history_len: 0,
                kv_len: 0,
            }
        }

        fn advanced_by(&self, rows: usize) -> Self {
            Self {
                conv_history_len: self.conv_history_len + rows,
                kv_len: self.kv_len + rows,
            }
        }
    }

    /// Drives every legal transition the FSM defines: `Prefill -> Decode ->
    /// Decode -> Verify -> Accept -> Decode -> Verify -> Rollback -> Decode
    /// -> Done`, and proves each illegal call at the wrong state is
    /// rejected rather than silently accepted.
    #[test]
    fn walkthrough_drives_every_legal_transition() {
        let prompt = alloc::vec![1_u32, 2, 3];
        let prompt_rows = prompt.len();
        let state = ServingState::start(prompt, FakeCache::empty());
        assert_eq!(
            state,
            ServingState::Prefill {
                positions: alloc::vec![1, 2, 3],
                cache: FakeCache::empty(),
            }
        );

        let after_prefill_cache = FakeCache::empty().advanced_by(prompt_rows);
        let state = state
            .advance_prefill(4, after_prefill_cache.clone())
            .expect("prefill -> decode is legal");
        assert_eq!(
            state,
            ServingState::Decode {
                last: 4,
                cache: after_prefill_cache.clone(),
            }
        );

        let after_first_decode_cache = after_prefill_cache.advanced_by(1);
        let state = state
            .advance_decode(5, after_first_decode_cache.clone())
            .expect("decode -> decode is legal");
        assert_eq!(
            state,
            ServingState::Decode {
                last: 5,
                cache: after_first_decode_cache.clone(),
            }
        );

        let illegal_prefill = state
            .clone()
            .advance_prefill(6, after_first_decode_cache.clone());
        assert_eq!(
            illegal_prefill,
            Err(ServingFsmError::IllegalTransition {
                attempted: "advance_prefill"
            })
        );

        let draft = alloc::vec![6_u32, 7];
        let state = state
            .enter_verify(draft.clone())
            .expect("decode -> verify is legal");
        assert_eq!(
            state,
            ServingState::Verify {
                draft,
                snapshot: after_first_decode_cache.clone(),
                cache: after_first_decode_cache.clone(),
            }
        );

        // full acceptance: both draft tokens (6, 7) match the target's own
        // predictions, so accept -> resume lands back in Decode with the
        // last draft token and row 1's placement.
        let row_one_cache = after_first_decode_cache.advanced_by(1);
        let row_two_cache = after_first_decode_cache.advanced_by(2);
        let state = state
            .accept(&[6, 7], alloc::vec![row_one_cache, row_two_cache.clone()])
            .expect("verify -> accept is legal");
        assert_eq!(
            state,
            ServingState::Accept {
                n: 2,
                next: 7,
                cache: row_two_cache.clone(),
            }
        );
        let state = state.resume().expect("accept -> decode is legal");
        assert_eq!(state, ServingState::Decode { last: 7, cache: row_two_cache.clone() });

        let illegal_resume = state.clone().resume();
        assert_eq!(illegal_resume, Err(ServingFsmError::IllegalTransition { attempted: "resume" }));

        // partial acceptance: draft proposes (100, 101), target predicts
        // 100 (confirming draft[0]) then 55 instead of 101 -- row_caches[0]
        // is the state after consuming `last` (used to predict draft[0]),
        // row_caches[1] is the state after consuming draft[0] (used to
        // predict row_tokens[1]=55); rejection at index 1 resumes from
        // row_caches[1], since that is the state row_tokens[1] came from.
        let draft = alloc::vec![100_u32, 101];
        let state = state.enter_verify(draft).expect("decode -> verify is legal");
        let cache_after_last = row_two_cache.advanced_by(1);
        let cache_after_draft_zero = row_two_cache.advanced_by(2);
        let state = state
            .accept(&[100, 55], alloc::vec![cache_after_last, cache_after_draft_zero.clone()])
            .expect("verify -> rollback is legal on partial acceptance");
        assert_eq!(
            state,
            ServingState::Rollback {
                snapshot: cache_after_draft_zero.clone(),
                to: 55,
            }
        );
        let state = state.rollback().expect("rollback -> decode is legal");
        assert_eq!(state, ServingState::Decode { last: 55, cache: cache_after_draft_zero });

        let done = state.finish();
        assert!(matches!(done, ServingState::Done { .. }));

        let past_done = done.advance_decode(8, FakeCache::empty());
        assert_eq!(
            past_done,
            Err(ServingFsmError::IllegalTransition {
                attempted: "advance_decode"
            })
        );
    }

    /// `accept` rejects a call whose `row_tokens`/`row_caches` lengths
    /// don't match the draft it is scoring, and an empty draft never
    /// reaches `accept` (verified separately since `enter_verify` accepts
    /// any `Vec`, including empty, but `accept` must not panic on it).
    #[test]
    fn accept_rejects_mismatched_or_empty_draft() {
        let cache = FakeCache::empty();
        let state = ServingState::start(alloc::vec![1], cache.clone())
            .advance_prefill(2, cache.clone())
            .expect("prefill -> decode")
            .enter_verify(alloc::vec![3, 4])
            .expect("decode -> verify");

        let wrong_lengths = state.clone().accept(&[3], alloc::vec![cache.clone()]);
        assert_eq!(wrong_lengths, Err(ServingFsmError::IllegalTransition { attempted: "accept" }));

        let empty_draft = ServingState::start(alloc::vec![1], cache.clone())
            .advance_prefill(2, cache.clone())
            .expect("prefill -> decode")
            .enter_verify(Vec::new())
            .expect("decode -> verify")
            .accept(&[], Vec::new());
        assert_eq!(empty_draft, Err(ServingFsmError::IllegalTransition { attempted: "accept" }));
    }

    /// Worked example for [`draft_prompt_lookup`]: history `[5, 1, 2, 3, 9,
    /// 1, 2, 3]`'s last `ngram=2` tokens are `[2, 3]`; scanning backward
    /// from the trailing occurrence, the first EARLIER `[2, 3]` is at index
    /// 2, so the draft is whatever followed it there -- `[9, 1, 2, 3]`, the
    /// four tokens between that earlier match and the end of history.
    /// `max_k=2` caps that same draft to its first two tokens; `max_k=0`,
    /// too-short history, and no earlier occurrence all draft nothing.
    #[test]
    fn draft_prompt_lookup_finds_the_earlier_occurrence() {
        let history = alloc::vec![5_u32, 1, 2, 3, 9, 1, 2, 3];
        assert_eq!(draft_prompt_lookup(&history, 2, 5), alloc::vec![9, 1, 2, 3]);
        assert_eq!(draft_prompt_lookup(&history, 2, 2), alloc::vec![9, 1]);
        assert_eq!(draft_prompt_lookup(&history, 2, 0), Vec::<u32>::new());
        assert_eq!(draft_prompt_lookup(&[1, 2], 2, 5), Vec::<u32>::new());
        assert_eq!(draft_prompt_lookup(&[9, 9, 9], 1, 1), alloc::vec![9]);
    }

    /// A pure, deterministic stand-in for the target model: `state` is the
    /// running token count (`FakeCache`'s `kv_len` doubles as it), and the
    /// "greedy argmax" for any input token is `input + 1` up to a `vocab`
    /// ceiling, wrapping to keep every generated id in range -- enough to
    /// drive both a plain sequential loop and a speculative loop through
    /// identical arithmetic so their outputs are comparable byte-for-byte.
    fn fake_greedy_next(input: u32, vocab: u32) -> u32 {
        (input + 1) % vocab
    }

    /// The oracle this step asks for, at the FSM level: plain greedy decode
    /// (one `advance_decode` per token, no drafting) produces the exact
    /// same 32-token sequence as speculative decode (prompt-lookup drafts
    /// up to `K=5`, scored via `accept`/`resume`/`rollback`) against the
    /// same deterministic target function -- proving this module's own
    /// control flow never changes what greedy decoding would have produced,
    /// independent of the real `qwen35moe` program (see the module-level
    /// doc for why that oracle is blocked on a different, pre-existing bug).
    #[test]
    fn speculation_matches_plain_greedy_for_thirty_two_tokens() {
        // a small vocab wraps `fake_greedy_next`'s output within the
        // 32-token run, so its own 2-grams recur and `draft_prompt_lookup`
        // actually finds matches -- proving the Accept/resume path, not
        // just the empty-draft fallback (a wide vocab like 97 never repeats
        // in 32 tokens from a single seed and would silently never draft).
        const VOCAB: u32 = 11;
        const TOKENS: usize = 32;
        const K: usize = 5;
        const NGRAM: usize = 2;
        let seed_prompt = alloc::vec![3_u32, 5];

        let mut plain_history = seed_prompt.clone();
        let mut plain_state = ServingState::start(seed_prompt.clone(), FakeCache::empty());
        let mut last = *plain_history.last().expect("seed prompt is non-empty");
        plain_state = plain_state.advance_prefill(last, FakeCache::empty()).expect("prefill -> decode");
        for _ in 0..TOKENS {
            let next = fake_greedy_next(last, VOCAB);
            plain_state = plain_state.advance_decode(next, FakeCache::empty()).expect("decode -> decode");
            plain_history.push(next);
            last = next;
        }

        let mut spec_history = seed_prompt.clone();
        let mut spec_state = ServingState::start(seed_prompt.clone(), FakeCache::empty());
        let mut last = *spec_history.last().expect("seed prompt is non-empty");
        spec_state = spec_state.advance_prefill(last, FakeCache::empty()).expect("prefill -> decode");
        let mut verify_rounds = 0usize;
        let mut draft_tokens_accepted = 0usize;
        while spec_history.len() - seed_prompt.len() < TOKENS {
            let draft = draft_prompt_lookup(&spec_history, NGRAM, K);
            if draft.is_empty() {
                let next = fake_greedy_next(last, VOCAB);
                spec_state = spec_state.advance_decode(next, FakeCache::empty()).expect("decode -> decode");
                spec_history.push(next);
                last = next;
                continue;
            }
            let mut row_tokens = Vec::with_capacity(draft.len());
            let mut feed = last;
            for &drafted in &draft {
                row_tokens.push(fake_greedy_next(feed, VOCAB));
                feed = drafted;
            }
            let row_caches = alloc::vec![FakeCache::empty(); draft.len()];
            verify_rounds += 1;
            let verifying = spec_state.enter_verify(draft.clone()).expect("decode -> verify");
            let outcome = verifying.accept(&row_tokens, row_caches).expect("verify -> accept | rollback");
            spec_state = match outcome {
                ServingState::Accept { n, next, cache } => {
                    draft_tokens_accepted += n;
                    for token in &draft {
                        spec_history.push(*token);
                    }
                    last = next;
                    ServingState::Accept { n, next, cache }.resume().expect("accept -> decode")
                }
                ServingState::Rollback { snapshot, to } => {
                    let accepted = draft.iter().zip(row_tokens.iter()).take_while(|(drafted, predicted)| drafted == predicted).count();
                    for token in &draft[..accepted] {
                        spec_history.push(*token);
                    }
                    spec_history.push(to);
                    last = to;
                    ServingState::Rollback { snapshot, to }.rollback().expect("rollback -> decode")
                }
                _ => unreachable!("accept only ever returns Accept or Rollback"),
            };
            spec_history.truncate(seed_prompt.len() + TOKENS.min(spec_history.len() - seed_prompt.len()));
        }

        assert!(
            verify_rounds > 0 && draft_tokens_accepted > 0,
            "the small vocab must make draft_prompt_lookup find real matches -- a zero count here means \
             this run degenerated to the empty-draft fallback and never exercised accept/resume at all"
        );
        assert_eq!(spec_history, plain_history, "speculative and plain greedy must produce identical token ids");
        assert!(matches!(plain_state.finish(), ServingState::Done { .. }));
        assert!(matches!(spec_state.finish(), ServingState::Done { .. }));
    }
}
