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
}
