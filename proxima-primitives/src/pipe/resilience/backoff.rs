use core::time::Duration;

/// Delay growth strategy for retry attempts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backoff {
    /// Fixed delay on every attempt.
    Constant(Duration),
    /// Delay grows by `factor` each attempt, saturating at `max`.
    Exponential {
        initial: Duration,
        factor: u32,
        max: Duration,
    },
}

/// Randomisation applied on top of the base delay.
///
/// All variants take a caller-provided `rand: u64` — no global RNG, no
/// wall-clock. Same `(attempt, rand, prev)` always produces the same output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Jitter {
    /// No randomisation — return the base delay exactly.
    None,
    /// Uniform random in `[0, base]`.
    Full,
    /// Half the base plus a uniform random in `[0, base/2]`.
    Equal,
    /// AWS decorrelated: `min(max, uniform[initial, prev*3])`.
    Decorrelated,
}

impl Backoff {
    /// Base delay for the given 0-based `attempt`, before jitter.
    ///
    /// Exponential: `initial * factor^attempt`, saturating at `max`.
    #[must_use]
    pub fn base_delay(&self, attempt: u32) -> Duration {
        match self {
            Self::Constant(duration) => *duration,
            Self::Exponential {
                initial,
                factor,
                max,
            } => {
                let initial_ms = initial.as_millis().min(u64::MAX as u128) as u64;
                let max_ms = max.as_millis().min(u64::MAX as u128) as u64;
                let multiplier = (*factor as u64).saturating_pow(attempt);
                let delay_ms = initial_ms.saturating_mul(multiplier).min(max_ms);
                Duration::from_millis(delay_ms)
            }
        }
    }

    /// Delay for `attempt` with jitter applied.
    ///
    /// `prev` is the delay used on the previous attempt (needed for Decorrelated).
    /// `rand` is caller-provided entropy — seeded per call for deterministic replay.
    #[must_use]
    pub fn delay(&self, attempt: u32, jitter: Jitter, prev: Duration, rand: u64) -> Duration {
        let base = self.base_delay(attempt);
        let base_ms = base.as_millis().min(u64::MAX as u128) as u64;

        match jitter {
            Jitter::None => base,

            Jitter::Full => Duration::from_millis(rand % base_ms.saturating_add(1)),

            Jitter::Equal => {
                let half = base_ms / 2;
                Duration::from_millis(half + rand % half.saturating_add(1))
            }

            Jitter::Decorrelated => {
                let (initial_ms, max_ms) = match self {
                    Self::Constant(duration) => {
                        let ms = duration.as_millis().min(u64::MAX as u128) as u64;
                        (ms, ms)
                    }
                    Self::Exponential { initial, max, .. } => (
                        initial.as_millis().min(u64::MAX as u128) as u64,
                        max.as_millis().min(u64::MAX as u128) as u64,
                    ),
                };
                let prev_ms = prev.as_millis().min(u64::MAX as u128) as u64;
                let prev_times_3 = prev_ms.saturating_mul(3);

                if prev_times_3 < initial_ms {
                    return Duration::from_millis(initial_ms);
                }

                let range = prev_times_3 - initial_ms + 1;
                let value = initial_ms + rand % range;
                Duration::from_millis(value.min(max_ms))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn exponential_fixture() -> Backoff {
        Backoff::Exponential {
            initial: Duration::from_millis(100),
            factor: 2,
            max: Duration::from_millis(2000),
        }
    }

    #[test]
    fn exponential_base_delays_follow_doubling_sequence_and_cap() {
        let backoff = exponential_fixture();
        let expected_ms = [100u64, 200, 400, 800, 1600, 2000, 2000];
        for (attempt, expected) in expected_ms.iter().enumerate() {
            assert_eq!(
                backoff.base_delay(attempt as u32),
                Duration::from_millis(*expected),
                "attempt {attempt}"
            );
        }
    }

    #[test]
    fn constant_base_delay_is_always_the_constant() {
        let backoff = Backoff::Constant(Duration::from_millis(500));
        for attempt in [0u32, 1, 5, 100] {
            assert_eq!(backoff.base_delay(attempt), Duration::from_millis(500));
        }
    }

    #[test]
    fn jitter_none_returns_base_exactly() {
        let backoff = exponential_fixture();
        let base = backoff.base_delay(2);
        let result = backoff.delay(2, Jitter::None, Duration::ZERO, 0xDEAD_BEEF);
        assert_eq!(result, base);
    }

    #[test]
    fn jitter_full_bit_exact_rand_zero() {
        let backoff = exponential_fixture();
        let base_ms = 400u64;
        let expected_ms = 0u64 % (base_ms + 1);
        let result_ms = backoff
            .delay(2, Jitter::Full, Duration::ZERO, 0)
            .as_millis() as u64;
        assert_eq!(result_ms, expected_ms, "Full(rand=0, base=400ms)");
    }

    #[test]
    fn jitter_full_result_is_within_zero_to_base() {
        let backoff = exponential_fixture();
        let base_ms = 400u64;
        for rand in [0u64, 1, base_ms - 1, base_ms, base_ms + 7, u64::MAX] {
            let result_ms = backoff
                .delay(2, Jitter::Full, Duration::ZERO, rand)
                .as_millis() as u64;
            assert!(
                result_ms <= base_ms,
                "Full rand={rand}: {result_ms} > {base_ms}"
            );
        }
    }

    #[test]
    fn jitter_equal_bit_exact_rand_zero() {
        let backoff = exponential_fixture();
        let base_ms = 400u64;
        let half = base_ms / 2;
        let expected_ms = half + 0 % (half + 1);
        let result_ms = backoff
            .delay(2, Jitter::Equal, Duration::ZERO, 0)
            .as_millis() as u64;
        assert_eq!(result_ms, expected_ms, "Equal(rand=0, base=400ms)");
    }

    #[test]
    fn jitter_equal_result_is_within_half_to_base() {
        let backoff = exponential_fixture();
        let base_ms = 400u64;
        let half = base_ms / 2;
        for rand in [0u64, 1, half, u64::MAX] {
            let result_ms = backoff
                .delay(2, Jitter::Equal, Duration::ZERO, rand)
                .as_millis() as u64;
            assert!(
                result_ms >= half && result_ms <= base_ms,
                "Equal rand={rand}: {result_ms} not in [{half}, {base_ms}]"
            );
        }
    }

    #[test]
    fn jitter_decorrelated_bit_exact_worked_case() {
        let backoff = exponential_fixture();
        let initial_ms = 100u64;
        let max_ms = 2000u64;
        let prev = Duration::from_millis(200);
        let prev_ms = 200u64;
        let rand = 42u64;
        let prev_times_3 = prev_ms * 3;
        let range = prev_times_3 - initial_ms + 1;
        let expected_ms = (initial_ms + rand % range).min(max_ms);
        let result_ms = backoff
            .delay(0, Jitter::Decorrelated, prev, rand)
            .as_millis() as u64;
        assert_eq!(result_ms, expected_ms);
        assert!(result_ms >= initial_ms && result_ms <= max_ms);
    }

    #[test]
    fn jitter_decorrelated_returns_initial_when_prev_too_small() {
        let backoff = exponential_fixture();
        let result_ms = backoff
            .delay(0, Jitter::Decorrelated, Duration::ZERO, u64::MAX)
            .as_millis() as u64;
        assert_eq!(
            result_ms, 100,
            "prev=0 → prev*3=0 < initial=100 → return initial"
        );
    }

    #[test]
    fn jitter_decorrelated_caps_at_max() {
        let backoff = exponential_fixture();
        let prev = Duration::from_millis(10_000);
        let result_ms = backoff
            .delay(0, Jitter::Decorrelated, prev, u64::MAX)
            .as_millis() as u64;
        assert_eq!(result_ms, 2000, "capped at max=2000ms");
    }
}
