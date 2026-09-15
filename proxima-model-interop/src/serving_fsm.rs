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
//! `Verify`'s own draft-scoring and `Rollback`'s own restore body are the
//! next step (owner review before landing speculative decoding); both
//! return [`ServingFsmError::NotSupported`] here so the enum's shape is
//! provable today without pretending the algebra is wired to a draft model.

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
    /// `n`: how many leading draft tokens the verifier accepted.
    Accept { n: usize, cache: Cache },
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

    /// `Verify { draft, snapshot } -> Accept { n }`: the `K`-row evaluation
    /// scoring the draft against the target model, with logits at every
    /// row. Not implemented -- draft scoring is the next step.
    pub(crate) fn accept(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Verify { .. } => Err(ServingFsmError::NotSupported {
                operation: "verify::accept",
            }),
            _ => Err(ServingFsmError::IllegalTransition { attempted: "accept" }),
        }
    }

    /// `Rollback { snapshot, to } -> Decode { last }`: placement copies
    /// only, no program evaluation. Not implemented -- the restore body is
    /// the next step alongside [`Self::accept`].
    pub(crate) fn rollback(self) -> Result<Self, ServingFsmError> {
        match self {
            Self::Rollback { .. } => Err(ServingFsmError::NotSupported {
                operation: "rollback",
            }),
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
    /// Decode -> Verify -> (Accept | Rollback stubbed) -> Done`, and proves
    /// each illegal call at the wrong state is rejected rather than
    /// silently accepted.
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
                cache: after_first_decode_cache,
            }
        );

        let accept_result = state.clone().accept();
        assert_eq!(
            accept_result,
            Err(ServingFsmError::NotSupported {
                operation: "verify::accept"
            })
        );

        let rollback_result = state.clone().rollback();
        assert_eq!(
            rollback_result,
            Err(ServingFsmError::IllegalTransition {
                attempted: "rollback"
            })
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

    /// `Rollback { snapshot, to } -> Decode`: not implemented yet, but the
    /// variant itself is constructible and rejects the wrong transition.
    #[test]
    fn rollback_state_is_constructible_and_stub_rejects_wrong_call() {
        let snapshot = FakeCache::empty();
        let state = ServingState::<FakeCache>::Rollback {
            snapshot: snapshot.clone(),
            to: 3,
        };

        let result = state.clone().rollback();
        assert_eq!(
            result,
            Err(ServingFsmError::NotSupported {
                operation: "rollback"
            })
        );

        let illegal = state.advance_decode(9, snapshot);
        assert_eq!(
            illegal,
            Err(ServingFsmError::IllegalTransition {
                attempted: "advance_decode"
            })
        );
    }
}
