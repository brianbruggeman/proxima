//! QUIC's view of the sans-IO time primitives.
//!
//! [`Instant`] / [`Duration`] / [`DurationError`] are the crate-shared
//! newtypes from [`crate::time`], re-exported so `quic::time::Instant`
//! keeps resolving. Only [`AckDelayExponent`] is QUIC-specific: it is an
//! RFC 9000 §18.2 transport parameter, not general time arithmetic.

use core::fmt;

pub use crate::time::{Duration, DurationError, Instant};

/// The per-RFC-9000 §18.2 `ack_delay_exponent` transport parameter,
/// validated at construction.
///
/// QUIC encodes peer ACK delays as `ack_delay << exponent` microseconds.
/// The exponent is a 3-bit transport parameter capped by RFC 9000 §18.2
/// at 20 (values above 20 are a transport-parameter protocol error).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct AckDelayExponent(u8);

/// Error from attempting to construct an [`AckDelayExponent`] outside
/// the RFC 9000 §18.2-defined range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AckDelayExponentError {
    /// The supplied exponent exceeds the RFC 9000 §18.2 max of 20.
    OutOfRange { supplied: u8 },
}

impl fmt::Display for AckDelayExponentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { supplied } => write!(
                f,
                "ack_delay_exponent {supplied} exceeds RFC 9000 §18.2 max of 20"
            ),
        }
    }
}

impl AckDelayExponent {
    /// RFC 9000 §18.2: maximum exponent any peer may advertise is 20.
    pub const MAX: u8 = 20;

    /// RFC 9000 §18.2 default `ack_delay_exponent` when the TP is absent.
    pub const DEFAULT: Self = Self(3);

    /// Construct an exponent, rejecting values above the RFC §18.2 max.
    ///
    /// # Errors
    ///
    /// Returns [`AckDelayExponentError::OutOfRange`] when `value > 20`.
    pub fn new(value: u8) -> Result<Self, AckDelayExponentError> {
        if value > Self::MAX {
            Err(AckDelayExponentError::OutOfRange { supplied: value })
        } else {
            Ok(Self(value))
        }
    }

    /// Construct without validation. **Only use for compile-time constants
    /// that are themselves provably ≤ 20.** Tests + the RFC default
    /// constructor are the only callers.
    #[must_use]
    pub const fn from_validated(value: u8) -> Self {
        Self(value)
    }

    /// The raw exponent (0..=20).
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Scale a peer-encoded ack-delay (microseconds-divided-by-2^exp)
    /// back to absolute microseconds, saturating at [`Duration::MAX`].
    #[must_use]
    pub const fn decode_ack_delay(self, encoded: u64) -> Duration {
        let scale: u64 = 1u64 << self.0;
        match encoded.checked_mul(scale) {
            Some(value) => Duration::from_micros(value),
            None => Duration::MAX,
        }
    }

    /// Scale an absolute ack-delay duration down to the peer-decodeable
    /// integer per the exponent (truncating, never panicking).
    #[must_use]
    pub const fn encode_ack_delay(self, delay: Duration) -> u64 {
        delay.as_micros() >> self.0
    }
}

impl fmt::Debug for AckDelayExponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AckDelayExponent({})", self.0)
    }
}

/// Test-only manual clock; advance steps deterministically.
///
/// Not gated behind a Cargo feature on purpose — Cargo feature unification
/// would leak this type onto the public API of any crate downstream of
/// `proxima-quic-proto` that happened to enable a `testing` feature on
/// any other consumer. `#[cfg(test)]` keeps it strictly internal.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub(crate) struct ManualClock {
    now: Instant,
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
impl ManualClock {
    pub const fn new(origin: Instant) -> Self {
        Self { now: origin }
    }

    pub fn now(&self) -> Instant {
        self.now
    }

    /// Advance the clock by `delta`. Panics on overflow — loud is
    /// correct in tests; a wrapped clock is a test-harness bug.
    pub fn advance(&mut self, delta: Duration) {
        self.now = self
            .now
            .checked_add(delta)
            .expect("ManualClock overflow — test fixture bug");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;


    #[test]
    fn ack_delay_exponent_rejects_above_20() {
        assert!(matches!(
            AckDelayExponent::new(21),
            Err(AckDelayExponentError::OutOfRange { supplied: 21 })
        ));
        assert!(AckDelayExponent::new(20).is_ok());
        assert_eq!(AckDelayExponent::new(20).expect("20 is valid").value(), 20);
    }

    #[test]
    fn ack_delay_exponent_default_is_rfc_default_3() {
        assert_eq!(AckDelayExponent::DEFAULT.value(), 3);
        assert_eq!(AckDelayExponent::default().value(), 0);
    }

    #[test]
    fn ack_delay_decode_scales_by_2_pow_exp() {
        let exp = AckDelayExponent::new(3).expect("3 is valid");
        let decoded = exp.decode_ack_delay(100);
        assert_eq!(decoded.as_micros(), 800);
    }

    #[test]
    fn ack_delay_encode_round_trips_with_truncation() {
        let exp = AckDelayExponent::new(3).expect("3 is valid");
        let original = Duration::from_micros(800);
        let encoded = exp.encode_ack_delay(original);
        assert_eq!(encoded, 100);
        let decoded = exp.decode_ack_delay(encoded);
        assert_eq!(decoded.as_micros(), 800);
    }

    #[test]
    fn ack_delay_decode_saturates_on_large_shift() {
        let exp = AckDelayExponent::new(20).expect("20 is valid");
        let decoded = exp.decode_ack_delay(u64::MAX >> 4);
        assert_eq!(decoded, Duration::MAX);
    }

    #[test]
    fn manual_clock_advances_monotonically() {
        let mut clock = ManualClock::new(Instant::from_micros(1_000));
        assert_eq!(clock.now(), Instant::from_micros(1_000));
        clock.advance(Duration::from_micros(500));
        assert_eq!(clock.now(), Instant::from_micros(1_500));
        clock.advance(Duration::from_millis(1));
        assert_eq!(clock.now(), Instant::from_micros(2_500));
    }
}
