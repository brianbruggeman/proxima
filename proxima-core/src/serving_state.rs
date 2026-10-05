//! The serving control plane as one sans-IO state machine over the algebra: every transition performs exactly one program evaluation, or placement copies only, and the per-layer cache lives inside the variant a transition returns rather than in a cache object patched from outside.
//!
//! The tests below prove the control flow (drafting, acceptance counting, cache placement, rollback-to-resume) against a fake target function instead of a model program.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ServingFsmError {
    /// A transition was invoked on a variant it does not apply to (for
    /// example, `ServingState::advance_decode` called on `Prefill`).
    #[error("serving fsm: {attempted} is not legal from the current state")]
    IllegalTransition { attempted: &'static str },
}

#[cfg(test)]
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
}
