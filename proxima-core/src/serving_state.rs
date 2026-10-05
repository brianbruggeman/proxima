//! The serving control plane as one sans-IO state machine over the algebra: every transition performs exactly one program evaluation, or placement copies only, and the per-layer cache lives inside the variant a transition returns rather than in a cache object patched from outside.
//!
//! The tests below prove the control flow (drafting, acceptance counting, cache placement, rollback-to-resume) against a fake target function instead of a model program.

use alloc::vec::Vec;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ServingFsmError {
    /// A transition was invoked on a variant it does not apply to (for
    /// example, `ServingState::advance_decode` called on `Prefill`).
    #[error("serving fsm: {attempted} is not legal from the current state")]
    IllegalTransition { attempted: &'static str },
}

/// Which evaluation shape is legal right now. An enum whose transitions
/// consume `self` states that directly; no existing proxima primitive (pipe,
/// cache or config) expresses it more cheaply.
#[derive(Debug, Clone, PartialEq)]
pub enum ServingState<Entry, Cache> {
    /// `positions`: the prompt token ids to place in one `M`-row program
    /// evaluation. `cache` starts empty -- prefill is the only state that
    /// does not inherit recurrent state from a predecessor.
    Prefill { positions: Vec<Entry>, cache: Cache },
    /// `last`: the most recently accepted token id, fed back as the next
    /// single-row evaluation's input.
    Decode { last: Entry, cache: Cache },
    /// `draft`: candidate token ids proposed by a speculator, scored in one
    /// `K`-row evaluation with logits returned at every row. `snapshot` is
    /// the pre-draft `cache`, restored verbatim on rejection.
    Verify {
        draft: Vec<Entry>,
        snapshot: Cache,
        cache: Cache,
    },
    /// `n`: how many leading draft tokens the verifier accepted; `next` is
    /// the token id decoding resumes from (the confirmed final draft token
    /// when `n == draft.len()`, otherwise the target's own row-`n`
    /// prediction that replaced the rejected draft token).
    Accept { n: usize, next: Entry, cache: Cache },
    /// `to`: the token id decoding resumes from after `snapshot` is
    /// restored in place of a rejected draft's advanced cache.
    Rollback { snapshot: Cache, to: Entry },
    /// Terminal: no further evaluation is legal.
    Done { cache: Cache },
}

impl<Entry, Cache> ServingState<Entry, Cache> {
    /// Enter the state machine at `Prefill` with an empty `cache`.
    pub fn start(positions: Vec<Entry>, cache: Cache) -> Self {
        Self::Prefill { positions, cache }
    }

    /// `Prefill { positions } -> Decode { last }`: the one `M`-row program
    /// evaluation prefill performs produced `next_token` and left `cache`
    /// holding the resulting recurrent/KV state.
    pub fn advance_prefill(
        self,
        next_token: Entry,
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
    pub fn advance_decode(
        self,
        next_token: Entry,
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
    pub fn enter_verify(self, draft: Vec<Entry>) -> Result<Self, ServingFsmError>
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

    /// `Accept { n, next, cache } -> Decode { last }`: no evaluation, just
    /// unwrapping the placement `accept` already chose.
    pub fn resume(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Accept { next, cache, .. } => Ok(Self::Decode { last: next, cache }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "resume",
            }),
        }
    }

    /// `Rollback { snapshot, to } -> Decode { last }`: placement copy only,
    /// no program evaluation -- `snapshot` is whichever row's `Cache`
    /// `accept` selected as the correct resume point.
    pub fn rollback(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Rollback { snapshot, to } => Ok(Self::Decode {
                last: to,
                cache: snapshot,
            }),
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "rollback",
            }),
        }
    }

    /// Any state `-> Done`: the caller has stopped requesting further
    /// evaluations (max tokens reached, eos observed, error surfaced).
    pub fn finish(self) -> Self {
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

impl<Entry: PartialEq + Clone, Cache> ServingState<Entry, Cache> {
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
    pub fn accept(
        self,
        row_tokens: &[Entry],
        row_caches: Vec<Cache>,
    ) -> Result<Self, ServingFsmError> {
        match self {
            Self::Verify { draft, .. }
                if draft.is_empty()
                    || row_tokens.len() != draft.len()
                    || row_caches.len() != draft.len() =>
            {
                Err(ServingFsmError::IllegalTransition {
                    attempted: "accept",
                })
            }
            Self::Verify { draft, .. } => {
                let accepted = draft
                    .iter()
                    .zip(row_tokens.iter())
                    .take_while(|(drafted, predicted)| drafted == predicted)
                    .count();
                let mut placements = row_caches.into_iter();
                let placement_index = if accepted == draft.len() {
                    accepted - 1
                } else {
                    accepted
                };
                let Some(cache) = placements.nth(placement_index) else {
                    // unreachable given the length guard above; a bad
                    // caller-supplied `row_caches` fails closed instead of panicking
                    return Err(ServingFsmError::IllegalTransition {
                        attempted: "accept",
                    });
                };
                if accepted == draft.len() {
                    Ok(Self::Accept {
                        n: accepted,
                        next: draft[accepted - 1].clone(),
                        cache,
                    })
                } else {
                    Ok(Self::Rollback {
                        snapshot: cache,
                        to: row_tokens[accepted].clone(),
                    })
                }
            }
            _ => Err(ServingFsmError::IllegalTransition {
                attempted: "accept",
            }),
        }
    }
}

#[cfg(test)]
// every transition here returns a typed error the assertions above it already
// prove unreachable; `expect` documents which one, `unwrap_used`/`expect_used`
// stay denied outside `#[cfg(test)]`
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn serving_fsm_error_names_the_attempted_transition() {
        let accept = ServingFsmError::IllegalTransition { attempted: "accept" };
        let resume = ServingFsmError::IllegalTransition { attempted: "resume" };

        assert_eq!(
            alloc::format!("{}", accept),
            "serving fsm: accept is not legal from the current state"
        );
        assert_ne!(accept, resume);
    }

    #[test]
    fn walk_prefill_decode_verify_finish() {
        let prefilled = ServingState::start(alloc::vec![1_u32, 2], 0_u8);
        let decoding = prefilled.advance_prefill(3, 1_u8).expect("legal");
        let decoding = decoding.advance_decode(4, 2_u8).expect("legal");
        let verifying = decoding.enter_verify(alloc::vec![5]).expect("legal");

        assert_eq!(
            verifying,
            ServingState::Verify { draft: alloc::vec![5], snapshot: 2, cache: 2 }
        );
        let done = verifying.finish();
        assert_eq!(done, ServingState::Done { cache: 2 });
        assert_eq!(
            done.advance_decode(6, 3),
            Err(ServingFsmError::IllegalTransition { attempted: "advance_decode" })
        );
    }

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
    fn serving_state_walkthrough_drives_every_legal_transition() {
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
        assert_eq!(
            state,
            ServingState::Decode {
                last: 7,
                cache: row_two_cache.clone()
            }
        );

        let illegal_resume = state.clone().resume();
        assert_eq!(
            illegal_resume,
            Err(ServingFsmError::IllegalTransition {
                attempted: "resume"
            })
        );

        // partial acceptance: draft proposes (100, 101), target predicts
        // 100 (confirming draft[0]) then 55 instead of 101 -- row_caches[0]
        // is the state after consuming `last` (used to predict draft[0]),
        // row_caches[1] is the state after consuming draft[0] (used to
        // predict row_tokens[1]=55); rejection at index 1 resumes from
        // row_caches[1], since that is the state row_tokens[1] came from.
        let draft = alloc::vec![100_u32, 101];
        let state = state
            .enter_verify(draft)
            .expect("decode -> verify is legal");
        let cache_after_last = row_two_cache.advanced_by(1);
        let cache_after_draft_zero = row_two_cache.advanced_by(2);
        let state = state
            .accept(
                &[100, 55],
                alloc::vec![cache_after_last, cache_after_draft_zero.clone()],
            )
            .expect("verify -> rollback is legal on partial acceptance");
        assert_eq!(
            state,
            ServingState::Rollback {
                snapshot: cache_after_draft_zero.clone(),
                to: 55,
            }
        );
        let state = state.rollback().expect("rollback -> decode is legal");
        assert_eq!(
            state,
            ServingState::Decode {
                last: 55,
                cache: cache_after_draft_zero
            }
        );

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
    fn serving_state_accept_rejects_mismatched_or_empty_draft() {
        let cache = FakeCache::empty();
        let state = ServingState::start(alloc::vec![1], cache.clone())
            .advance_prefill(2, cache.clone())
            .expect("prefill -> decode")
            .enter_verify(alloc::vec![3, 4])
            .expect("decode -> verify");

        let wrong_lengths = state.clone().accept(&[3], alloc::vec![cache.clone()]);
        assert_eq!(
            wrong_lengths,
            Err(ServingFsmError::IllegalTransition {
                attempted: "accept"
            })
        );

        let empty_draft = ServingState::start(alloc::vec![1], cache.clone())
            .advance_prefill(2, cache.clone())
            .expect("prefill -> decode")
            .enter_verify(Vec::new())
            .expect("decode -> verify")
            .accept(&[], Vec::new());
        assert_eq!(
            empty_draft,
            Err(ServingFsmError::IllegalTransition {
                attempted: "accept"
            })
        );
    }
}
