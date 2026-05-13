//! Monotonic time primitives for the sans-IO protocol state machines.
//!
//! Sans-IO means the protocol layer has no access to a clock — it must
//! be told what time it is at every ingress entry point. The newtypes
//! in this module package monotonic-clock arithmetic in a shape that:
//!
//! - is tier-3 (`#[repr(transparent)]` over `u64`, no `std::time`);
//! - distinguishes a wall-clock instant from a duration at the type level;
//! - is **saturating-monotonic** at protocol arithmetic — `Instant + Duration`
//!   and `Instant - Duration` cannot panic or wrap (they pin at
//!   [`Instant::MAX`] / [`Instant::ZERO`] respectively);
//! - exposes a `checked_*` family for the rare caller that needs to
//!   distinguish saturation from a real value (e.g. probe-timeout budgeting);
//! - removes [`core::ops::Sub`] for `Instant - Instant` so callers reach
//!   for the explicit [`Instant::duration_since`] which returns
//!   `Option<Duration>` — a clock skew producing `None` is semantically
//!   meaningful and must be handled deliberately;
//! - has **no [`Default`]** for [`Instant`] or [`Duration`] — a
//!   `Default::default()`-ed instant would bypass the per-connection
//!   monotonicity gate on first use. Containers must construct explicitly.
//!
//! Shared by the `quic` and `tcp` data paths, which are independently
//! feature-gated — hence this sits at the crate root rather than under
//! either one. QUIC's `AckDelayExponent` stays in `quic::time`: it is an
//! RFC 9000 transport parameter, not general time arithmetic.
//!
//! # Choices grounded in the C11 design pass
//!
//! Resolution recorded in [`docs/proxima-quic/edges.md`]. Mac's pipeline
//! research-rigor self-play tournament (3 rounds, 4-round cap, unanimous
//! round-3-synthesis win each round; final emitted as best-of-tournament).
//!
//! [`docs/proxima-quic/edges.md`]: ../../docs/proxima-quic/edges.md

use core::fmt;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::time::Duration as CoreDuration;

/// A monotonic instant, measured as microseconds since a caller-defined origin.
///
/// `Instant` carries no notion of wall-clock time, time zone, or epoch.
/// It is whatever the caller's monotonic source produces, scaled to
/// microseconds. The proto layer never compares two [`Instant`] values
/// across different connections — each connection's time line is
/// independent. The proto layer DOES require that every successive
/// `now` passed into a single connection is monotonically non-decreasing.
///
/// At `u64` microseconds the representable horizon is ~584 000 years
/// per-connection. Saturation in `Add`/`Sub` is a theoretical concern
/// only; real connections live for seconds to hours.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(u64);

/// A monotonic duration, measured in microseconds.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(u64);

/// Error from converting a [`CoreDuration`] into a proto [`Duration`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DurationError {
    /// The supplied [`CoreDuration`] does not fit in a `u64` of microseconds.
    Overflow,
}

impl fmt::Display for DurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("duration overflows u64 microseconds"),
        }
    }
}

impl Instant {
    /// The instant `0` microseconds since the caller's origin.
    pub const ZERO: Self = Self(0);

    /// The largest representable instant.
    pub const MAX: Self = Self(u64::MAX);

    /// Construct an instant directly from microseconds since the caller's origin.
    ///
    /// The caller is responsible for ensuring the value is monotonically
    /// non-decreasing across successive calls into the same connection.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Return the raw microseconds since the caller's origin.
    #[must_use]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Compute the duration between `self` and an `earlier` instant.
    ///
    /// Returns `None` when `earlier > self` — a backward-going clock
    /// is semantically meaningful in a sans-IO state machine and must
    /// be handled deliberately by the caller rather than collapsed to
    /// a saturating zero.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration)
    }

    /// Saturating addition of a duration.
    ///
    /// Returns [`Instant::MAX`] on overflow (unreachable in practice;
    /// `u64` microseconds is ~584 000 years).
    #[must_use]
    pub const fn saturating_add(self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.0))
    }

    /// Checked addition of a duration. Returns `None` on overflow.
    #[must_use]
    pub const fn checked_add(self, duration: Duration) -> Option<Self> {
        match self.0.checked_add(duration.0) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Saturating subtraction of a duration.
    ///
    /// Returns [`Instant::ZERO`] when the duration exceeds the instant's
    /// microseconds-since-origin.
    #[must_use]
    pub const fn saturating_sub(self, duration: Duration) -> Self {
        Self(self.0.saturating_sub(duration.0))
    }

    /// Checked subtraction of a duration. Returns `None` on underflow.
    #[must_use]
    pub const fn checked_sub(self, duration: Duration) -> Option<Self> {
        match self.0.checked_sub(duration.0) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl Duration {
    /// The zero duration.
    pub const ZERO: Self = Self(0);

    /// The largest representable duration.
    pub const MAX: Self = Self(u64::MAX);

    /// Construct from microseconds.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Construct from milliseconds, saturating at [`Duration::MAX`].
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1_000))
    }

    /// Construct from seconds, saturating at [`Duration::MAX`].
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs.saturating_mul(1_000_000))
    }

    /// Construct from a [`core::time::Duration`].
    ///
    /// # Errors
    ///
    /// Returns [`DurationError::Overflow`] when the supplied duration
    /// does not fit in `u64` microseconds. A saturated infinite timer
    /// would be a hung connection, not graceful recovery — boundary
    /// conversions that lose information return `Result`.
    pub fn from_core(duration: CoreDuration) -> Result<Self, DurationError> {
        u64::try_from(duration.as_micros())
            .map(Self)
            .map_err(|_| DurationError::Overflow)
    }

    /// Return the raw microseconds.
    #[must_use]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Return the duration in (rounded-down) milliseconds.
    #[must_use]
    pub const fn as_millis(self) -> u64 {
        self.0 / 1_000
    }

    /// Saturating multiplication by a positive integer factor.
    #[must_use]
    pub const fn saturating_mul(self, factor: u64) -> Self {
        Self(self.0.saturating_mul(factor))
    }

    /// Checked multiplication by a positive integer factor.
    #[must_use]
    pub const fn checked_mul(self, factor: u64) -> Option<Self> {
        match self.0.checked_mul(factor) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Saturating addition of two durations.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction of two durations.
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Take the minimum of two durations.
    #[must_use]
    pub const fn min(self, other: Self) -> Self {
        if self.0 <= other.0 { self } else { other }
    }

    /// Take the maximum of two durations.
    #[must_use]
    pub const fn max(self, other: Self) -> Self {
        if self.0 >= other.0 { self } else { other }
    }
}

impl Add<Duration> for Instant {
    type Output = Self;

    /// Saturating addition; see [`Instant::saturating_add`].
    fn add(self, rhs: Duration) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, rhs: Duration) {
        *self = self.saturating_add(rhs);
    }
}

impl Sub<Duration> for Instant {
    type Output = Self;

    /// Saturating subtraction; see [`Instant::saturating_sub`].
    fn sub(self, rhs: Duration) -> Self::Output {
        self.saturating_sub(rhs)
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, rhs: Duration) {
        *self = self.saturating_sub(rhs);
    }
}

impl Add<Duration> for Duration {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        self.saturating_add(rhs)
    }
}

impl AddAssign<Duration> for Duration {
    fn add_assign(&mut self, rhs: Duration) {
        *self = self.saturating_add(rhs);
    }
}

impl Sub<Duration> for Duration {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        self.saturating_sub(rhs)
    }
}

impl SubAssign<Duration> for Duration {
    fn sub_assign(&mut self, rhs: Duration) {
        *self = self.saturating_sub(rhs);
    }
}

impl fmt::Debug for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Instant({} µs)", self.0)
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Duration({} µs)", self.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn instant_add_duration_is_saturating() {
        let near_max = Instant::from_micros(u64::MAX - 5);
        let big = Duration::from_micros(100);
        assert_eq!(near_max + big, Instant::MAX);
    }

    #[test]
    fn instant_sub_duration_is_saturating_at_zero() {
        let small = Instant::from_micros(5);
        let big = Duration::from_micros(100);
        assert_eq!(small - big, Instant::ZERO);
    }

    #[test]
    fn instant_checked_add_returns_none_on_overflow() {
        assert_eq!(Instant::MAX.checked_add(Duration::from_micros(1)), None);
    }

    #[test]
    fn instant_checked_sub_returns_none_on_underflow() {
        assert_eq!(Instant::ZERO.checked_sub(Duration::from_micros(1)), None);
    }

    #[test]
    fn duration_since_returns_none_for_backward_clock() {
        let later = Instant::from_micros(100);
        let earlier = Instant::from_micros(50);
        assert_eq!(
            later.duration_since(earlier),
            Some(Duration::from_micros(50))
        );
        assert_eq!(earlier.duration_since(later), None);
    }

    #[test]
    fn duration_since_equal_instants_is_zero() {
        let now = Instant::from_micros(42);
        assert_eq!(now.duration_since(now), Some(Duration::ZERO));
    }

    #[test]
    fn duration_from_millis_saturates_on_overflow() {
        let huge = u64::MAX / 500;
        let scaled = Duration::from_millis(huge);
        assert_eq!(scaled, Duration::MAX);
    }

    #[test]
    fn duration_from_core_overflows_above_u64_micros() {
        let too_big = CoreDuration::from_secs(u64::MAX);
        assert!(matches!(
            Duration::from_core(too_big),
            Err(DurationError::Overflow)
        ));
    }

    #[test]
    fn duration_from_core_round_trips_small_value() {
        let core_dur = CoreDuration::from_millis(250);
        let proto_dur = Duration::from_core(core_dur).expect("250 ms fits");
        assert_eq!(proto_dur, Duration::from_millis(250));
        assert_eq!(proto_dur.as_millis(), 250);
    }
}
