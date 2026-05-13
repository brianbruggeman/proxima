use core::time::Duration;

/// A budget-relative cutoff expressed as an absolute nanos timestamp.
///
/// Caller-provided `now_nanos` throughout — no wall-clock reads inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deadline {
    deadline_nanos: u64,
}

impl Deadline {
    /// Construct a deadline `budget` nanoseconds from `now_nanos`.
    #[must_use]
    pub fn new(now_nanos: u64, budget: Duration) -> Self {
        let budget_nanos = budget.as_nanos().min(u64::MAX as u128) as u64;
        let deadline_nanos = now_nanos.saturating_add(budget_nanos);
        Self { deadline_nanos }
    }

    /// Time remaining until the deadline. Saturates to zero when past.
    #[must_use]
    pub fn remaining(&self, now_nanos: u64) -> Duration {
        if now_nanos >= self.deadline_nanos {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.deadline_nanos - now_nanos)
    }

    /// Whether the deadline has passed.
    #[must_use]
    pub fn expired(&self, now_nanos: u64) -> bool {
        now_nanos >= self.deadline_nanos
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const ONE_SECOND_NS: u64 = 1_000_000_000;

    #[test]
    fn deadline_not_expired_before_budget_elapses() {
        let deadline = Deadline::new(0, Duration::from_secs(1));
        assert!(!deadline.expired(ONE_SECOND_NS - 1));
        assert_eq!(
            deadline.remaining(ONE_SECOND_NS - 1),
            Duration::from_nanos(1)
        );
    }

    #[test]
    fn deadline_expired_exactly_at_the_instant() {
        let deadline = Deadline::new(0, Duration::from_secs(1));
        assert!(deadline.expired(ONE_SECOND_NS));
        assert_eq!(deadline.remaining(ONE_SECOND_NS), Duration::ZERO);
    }

    #[test]
    fn deadline_expired_after_the_instant() {
        let deadline = Deadline::new(0, Duration::from_secs(1));
        assert!(deadline.expired(ONE_SECOND_NS + 1));
        assert_eq!(deadline.remaining(ONE_SECOND_NS + 1), Duration::ZERO);
    }

    #[test]
    fn deadline_with_nonzero_origin() {
        let now = 5 * ONE_SECOND_NS;
        let deadline = Deadline::new(now, Duration::from_secs(2));
        assert!(!deadline.expired(now + ONE_SECOND_NS));
        assert!(deadline.expired(now + 2 * ONE_SECOND_NS));
        assert_eq!(deadline.remaining(now), Duration::from_secs(2));
    }

    #[test]
    fn deadline_remaining_saturates_to_zero() {
        let deadline = Deadline::new(0, Duration::from_secs(1));
        assert_eq!(deadline.remaining(u64::MAX), Duration::ZERO);
    }
}
