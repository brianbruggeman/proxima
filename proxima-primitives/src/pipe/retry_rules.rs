//! Retry decision as pure config. Tier-split backing store for the set of
//! retryable statuses: an unbounded `alloc::collections::BTreeSet` under
//! `proxima_alloc`, a fixed-capacity sorted `heapless::Vec` under no-alloc. The
//! cap (`RETRY_STATUS_CAP`) is baked from `proxima-primitives.toml` by build.rs.

#[cfg(proxima_alloc)]
use alloc::collections::BTreeSet;

use crate::pipe::capabilities::Retryable;
#[cfg(not(proxima_alloc))]
use crate::pipe::sized::RETRY_STATUS_CAP;

/// The retry decision as pure config — which statuses/errors to retry, and
/// whether to restrict to idempotent inputs. This is payload-agnostic data
/// evaluated against the capability traits (`Retryable`, `Idempotent`).
#[derive(Debug, Clone)]
pub struct RetryRules {
    /// Statuses that trigger a retry. Sorted, deduplicated. Unbounded under
    /// `proxima_alloc`; capped at `RETRY_STATUS_CAP` under no-alloc.
    pub retry_on_status: StatusSet,
    pub retry_on_error: bool,
    pub idempotent_only: bool,
}

/// The backing store for [`RetryRules::retry_on_status`], tier-selected at
/// build time: a heap `BTreeSet` when an allocator is present, a sorted const-cap
/// `heapless::Vec` when not. Both expose `insert` / `contains` over `u16`.
#[cfg(proxima_alloc)]
pub type StatusSet = BTreeSet<u16>;

/// No-alloc backing store: a sorted, deduplicated fixed-capacity vector. Sorted
/// invariant lets `contains` binary-search; `insert` past the cap is dropped
/// (the default set fits, and an over-cap config is a sizing misconfiguration,
/// not a runtime fault).
#[cfg(not(proxima_alloc))]
#[derive(Debug, Clone, Default)]
pub struct StatusSet {
    inner: heapless::Vec<u16, RETRY_STATUS_CAP>,
}

#[cfg(not(proxima_alloc))]
impl StatusSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: heapless::Vec::new(),
        }
    }

    /// Insert `status`, keeping the vector sorted and deduplicated. A status that
    /// would exceed `RETRY_STATUS_CAP` is dropped.
    pub fn insert(&mut self, status: u16) -> bool {
        match self.inner.binary_search(&status) {
            Ok(_) => false,
            Err(position) => self.inner.insert(position, status).is_ok(),
        }
    }

    /// Whether `status` is in the set. Binary search over the sorted vector.
    #[must_use]
    pub fn contains(&self, status: &u16) -> bool {
        self.inner.binary_search(status).is_ok()
    }
}

impl Default for RetryRules {
    fn default() -> Self {
        let mut retry_on_status = StatusSet::default();
        for status in [502u16, 503, 504] {
            retry_on_status.insert(status);
        }
        Self {
            retry_on_status,
            retry_on_error: true,
            idempotent_only: false,
        }
    }
}

impl RetryRules {
    pub fn should_retry<Out: Retryable, Err>(&self, outcome: &Result<Out, Err>) -> bool {
        match outcome {
            Ok(out) => out
                .retry_status()
                .is_some_and(|status| self.retry_on_status.contains(&status)),
            Err(_) => self.retry_on_error,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Statusful outcome probe — its `retry_status` decides whether the rules fire.
    struct Outcome(Option<u16>);

    impl Retryable for Outcome {
        fn retry_status(&self) -> Option<u16> {
            self.0
        }
        fn is_success(&self) -> bool {
            self.0.is_none()
        }
    }

    #[test]
    fn default_retries_the_gateway_statuses() {
        let rules = RetryRules::default();
        for status in [502u16, 503, 504] {
            assert!(
                rules.retry_on_status.contains(&status),
                "{status} should retry"
            );
        }
        assert!(
            !rules.retry_on_status.contains(&200),
            "200 should not retry"
        );
    }

    #[test]
    fn should_retry_matches_status_membership() {
        let rules = RetryRules::default();
        let ok: Result<Outcome, ()> = Ok(Outcome(Some(503)));
        let not_listed: Result<Outcome, ()> = Ok(Outcome(Some(418)));
        let statusless: Result<Outcome, ()> = Ok(Outcome(None));
        assert!(rules.should_retry(&ok), "503 is in the default set");
        assert!(!rules.should_retry(&not_listed), "418 is not");
        assert!(
            !rules.should_retry(&statusless),
            "no status => no status retry"
        );
    }

    #[test]
    fn should_retry_on_error_follows_the_flag() {
        let err: Result<Outcome, ()> = Err(());
        assert!(
            RetryRules::default().should_retry(&err),
            "default retries on error"
        );
        let no_error_retry = RetryRules {
            retry_on_error: false,
            ..RetryRules::default()
        };
        assert!(
            !no_error_retry.should_retry(&err),
            "flag off => no error retry"
        );
    }

    #[test]
    fn insert_is_idempotent_and_sorted() {
        let mut set = StatusSet::default();
        set.insert(503);
        set.insert(502);
        set.insert(503);
        assert!(set.contains(&502) && set.contains(&503));
        assert!(!set.contains(&504));
    }
}
