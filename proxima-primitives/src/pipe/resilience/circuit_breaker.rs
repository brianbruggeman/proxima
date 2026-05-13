use core::time::Duration;

/// Observable state of a [`CircuitBreaker`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    /// Normal operation — all calls pass through.
    Closed,
    /// Too many failures — calls are rejected until the cooldown expires.
    Open,
    /// Cooldown elapsed — limited probe calls allowed to test recovery.
    HalfOpen,
}

/// Sans-IO failure-rate circuit breaker.
///
/// State is mutated by three methods; all time-sensitive paths take a
/// caller-provided `now_nanos: u64` — no wall-clock reads inside.
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    half_open_max_probes: u32,

    consecutive_failures: u32,
    state: CircuitState,
    opened_at_nanos: u64,
    half_open_in_flight: u32,
    half_open_successes: u32,
}

impl CircuitBreaker {
    #[must_use]
    pub fn new(failure_threshold: u32, cooldown: Duration, half_open_max_probes: u32) -> Self {
        Self {
            failure_threshold,
            cooldown,
            half_open_max_probes,
            consecutive_failures: 0,
            state: CircuitState::Closed,
            opened_at_nanos: 0,
            half_open_in_flight: 0,
            half_open_successes: 0,
        }
    }

    /// Whether this call is permitted to proceed.
    ///
    /// - Closed: always true.
    /// - Open: false until the cooldown expires; then transitions to HalfOpen
    ///   and allows the first probe.
    /// - HalfOpen: true while probe count < `half_open_max_probes`.
    pub fn allow(&mut self, now_nanos: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,

            CircuitState::Open => {
                let cooldown_nanos = self.cooldown.as_nanos().min(u64::MAX as u128) as u64;
                let open_until = self.opened_at_nanos.saturating_add(cooldown_nanos);
                if now_nanos >= open_until {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_in_flight = 1;
                    self.half_open_successes = 0;
                    true
                } else {
                    false
                }
            }

            CircuitState::HalfOpen => {
                if self.half_open_in_flight < self.half_open_max_probes {
                    self.half_open_in_flight += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful outcome.
    ///
    /// - Closed: resets the consecutive-failure counter.
    /// - HalfOpen: counts the probe; if enough successes, transitions to Closed.
    pub fn on_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.half_open_max_probes {
                    self.state = CircuitState::Closed;
                    self.consecutive_failures = 0;
                    self.half_open_in_flight = 0;
                    self.half_open_successes = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed outcome.
    ///
    /// - Closed: increments the counter; if it reaches the threshold, opens.
    /// - HalfOpen: any failure re-opens immediately.
    pub fn on_failure(&mut self, now_nanos: u64) {
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at_nanos = now_nanos;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.opened_at_nanos = now_nanos;
                self.half_open_in_flight = 0;
                self.half_open_successes = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Current state (read-only snapshot).
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const ONE_SECOND_NS: u64 = 1_000_000_000;

    fn fixture() -> CircuitBreaker {
        CircuitBreaker::new(3, Duration::from_secs(1), 1)
    }

    #[test]
    fn starts_closed_and_allows_calls() {
        let mut cb = fixture();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow(0));
    }

    #[test]
    fn three_failures_open_the_circuit() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        assert_eq!(cb.state(), CircuitState::Closed, "two failures not enough");
        cb.on_failure(0);
        assert_eq!(cb.state(), CircuitState::Open, "third failure opens");
    }

    #[test]
    fn open_circuit_rejects_before_cooldown() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        cb.on_failure(0);
        assert!(!cb.allow(500_000_000), "0.5s < 1s cooldown → rejected");
    }

    #[test]
    fn open_circuit_transitions_to_half_open_after_cooldown() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        cb.on_failure(0);
        let allowed = cb.allow(ONE_SECOND_NS);
        assert!(allowed, "cooldown elapsed → HalfOpen probe allowed");
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn half_open_probe_success_closes_the_circuit() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        cb.on_failure(0);
        cb.allow(ONE_SECOND_NS);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.on_success();
        assert_eq!(
            cb.state(),
            CircuitState::Closed,
            "one success (=max_probes) closes"
        );
    }

    #[test]
    fn half_open_probe_failure_reopens_the_circuit() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        cb.on_failure(0);
        cb.allow(ONE_SECOND_NS);
        cb.on_failure(ONE_SECOND_NS);
        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "probe failure → back to Open"
        );
        assert!(
            !cb.allow(ONE_SECOND_NS + 500_000_000),
            "within new cooldown → rejected"
        );
    }

    #[test]
    fn success_in_closed_state_resets_failure_counter() {
        let mut cb = fixture();
        cb.on_failure(0);
        cb.on_failure(0);
        cb.on_success();
        cb.on_failure(0);
        cb.on_failure(0);
        assert_eq!(
            cb.state(),
            CircuitState::Closed,
            "counter reset → two more not enough"
        );
    }

    #[test]
    fn half_open_limits_probe_count() {
        let mut cb = CircuitBreaker::new(1, Duration::from_secs(1), 2);
        cb.on_failure(0);
        assert!(cb.allow(ONE_SECOND_NS), "first probe allowed");
        assert!(cb.allow(ONE_SECOND_NS), "second probe allowed");
        assert!(!cb.allow(ONE_SECOND_NS), "third probe rejected — at max");
    }
}
