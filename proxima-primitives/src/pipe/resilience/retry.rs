use core::time::Duration;

use super::backoff::{Backoff, Jitter};
use super::deadline::Deadline;
use crate::pipe::capabilities::Retryable;
use crate::pipe::retry_rules::RetryRules;

/// The decision returned by [`RetryController::on_outcome`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetryAction {
    /// The outcome was terminal (success or non-retryable error) — stop.
    Done,
    /// Retryable outcome and budget remains — wait `after` then try again.
    Retry { after: Duration },
    /// Retryable outcome but budget (attempts or deadline) is exhausted.
    Exhausted,
}

/// Sans-IO retry controller: pure decision logic, no I/O or sleeping.
///
/// The caller drives the attempt loop; `on_outcome` says what to do next.
#[derive(Debug, Clone)]
pub struct RetryController {
    pub rules: RetryRules,
    pub backoff: Backoff,
    pub jitter: Jitter,
    pub max_attempts: u32,
    pub deadline: Option<Deadline>,
}

impl RetryController {
    /// Decide what to do after `attempt` (0-based) produced `outcome`.
    ///
    /// `now_nanos` and `rand` are caller-provided for deterministic replay.
    /// `prev` is the delay used on the immediately preceding attempt.
    #[must_use]
    pub fn on_outcome<Out: Retryable, Err>(
        &self,
        attempt: u32,
        outcome: &Result<Out, Err>,
        now_nanos: u64,
        rand: u64,
        prev: Duration,
    ) -> RetryAction {
        if !self.rules.should_retry(outcome) {
            return RetryAction::Done;
        }

        if attempt.saturating_add(1) >= self.max_attempts {
            return RetryAction::Exhausted;
        }

        if let Some(deadline) = &self.deadline
            && deadline.expired(now_nanos)
        {
            return RetryAction::Exhausted;
        }

        RetryAction::Retry {
            after: self.backoff.delay(attempt, self.jitter, prev, rand),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::retry_rules::RetryRules;

    struct RetryableStatus(Option<u16>);

    impl Retryable for RetryableStatus {
        fn retry_status(&self) -> Option<u16> {
            self.0
        }
        fn is_success(&self) -> bool {
            self.0.is_none()
        }
    }

    fn fixture() -> RetryController {
        RetryController {
            rules: RetryRules::default(),
            backoff: Backoff::Exponential {
                initial: Duration::from_millis(100),
                factor: 2,
                max: Duration::from_millis(2000),
            },
            jitter: Jitter::None,
            max_attempts: 3,
            deadline: None,
        }
    }

    #[test]
    fn success_outcome_returns_done() {
        let controller = fixture();
        let outcome: Result<RetryableStatus, ()> = Ok(RetryableStatus(None));
        let action = controller.on_outcome(0, &outcome, 0, 0, Duration::ZERO);
        assert_eq!(action, RetryAction::Done);
    }

    #[test]
    fn retryable_status_at_first_attempt_returns_retry() {
        let controller = fixture();
        let outcome: Result<RetryableStatus, ()> = Ok(RetryableStatus(Some(503)));
        let action = controller.on_outcome(0, &outcome, 0, 0, Duration::ZERO);
        let expected_delay = controller.backoff.delay(0, Jitter::None, Duration::ZERO, 0);
        assert_eq!(
            action,
            RetryAction::Retry {
                after: expected_delay
            }
        );
    }

    #[test]
    fn retryable_outcome_at_last_attempt_returns_exhausted() {
        let controller = fixture();
        let outcome: Result<RetryableStatus, ()> = Ok(RetryableStatus(Some(503)));
        let action = controller.on_outcome(2, &outcome, 0, 0, Duration::ZERO);
        assert_eq!(
            action,
            RetryAction::Exhausted,
            "attempt 2 of max 3 → exhausted"
        );
    }

    #[test]
    fn expired_deadline_returns_exhausted() {
        use super::super::deadline::Deadline;
        let mut controller = fixture();
        controller.deadline = Some(Deadline::new(0, Duration::from_secs(1)));
        let outcome: Result<RetryableStatus, ()> = Ok(RetryableStatus(Some(503)));
        let now_past = 2_000_000_000u64;
        let action = controller.on_outcome(0, &outcome, now_past, 0, Duration::ZERO);
        assert_eq!(action, RetryAction::Exhausted, "past deadline → exhausted");
    }

    #[test]
    fn non_retryable_error_with_retry_on_error_false_returns_done() {
        let controller = RetryController {
            rules: RetryRules {
                retry_on_error: false,
                ..RetryRules::default()
            },
            ..fixture()
        };
        let outcome: Result<RetryableStatus, ()> = Err(());
        let action = controller.on_outcome(0, &outcome, 0, 0, Duration::ZERO);
        assert_eq!(action, RetryAction::Done);
    }

    #[test]
    fn retryable_error_returns_retry_when_budget_remains() {
        let controller = fixture();
        let outcome: Result<RetryableStatus, ()> = Err(());
        let action = controller.on_outcome(0, &outcome, 0, 0, Duration::ZERO);
        assert!(matches!(action, RetryAction::Retry { .. }));
    }
}
