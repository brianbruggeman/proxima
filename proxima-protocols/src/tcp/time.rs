//! TCP's view of the sans-IO time primitives.
//!
//! Re-exported from [`crate::time`], shared with `quic`. These were
//! previously reproduced here to keep a then-separate `proxima-tcp` from
//! depending on the QUIC crate; both are now modules of this one crate,
//! so the copy had no boundary left to protect.

pub use crate::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn instant_add_is_saturating() {
        let near_max = Instant::from_micros(u64::MAX - 5);
        assert_eq!(near_max + Duration::from_micros(100), Instant::MAX);
    }

    #[test]
    fn instant_sub_saturates_at_zero() {
        assert_eq!(
            Instant::from_micros(5) - Duration::from_micros(100),
            Instant::ZERO
        );
    }

    #[test]
    fn duration_since_handles_backward_clock() {
        let later = Instant::from_micros(100);
        let earlier = Instant::from_micros(50);
        assert_eq!(
            later.duration_since(earlier),
            Some(Duration::from_micros(50))
        );
        assert_eq!(earlier.duration_since(later), None);
        assert_eq!(later.duration_since(later), Some(Duration::ZERO));
    }

    #[test]
    fn duration_constructors_saturate() {
        assert_eq!(Duration::from_millis(u64::MAX / 500), Duration::MAX);
        assert_eq!(Duration::from_secs(2).as_millis(), 2_000);
    }

    #[test]
    fn duration_min_max() {
        let small = Duration::from_micros(10);
        let large = Duration::from_micros(20);
        assert_eq!(small.min(large), small);
        assert_eq!(small.max(large), large);
    }
}
