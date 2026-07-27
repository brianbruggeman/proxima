/// Wall-clock nanoseconds since the Unix epoch (1970-01-01T00:00:00Z).
///
/// The output of [`crate::anchor::ToUnixNanos`] — a [`crate::ticks::Ticks`]
/// reading converted at the edge, once, into a value a caller can render as
/// a timestamp. Never fed back into tick-domain arithmetic; the two units
/// are kept distinct at the type level so a caller cannot accidentally
/// subtract a `Ticks` from a `UnixNanos` and get a number that looks
/// plausible and means nothing.
///
/// `#[repr(transparent)]` over `u64`, matching
/// `proxima_protocols::Instant`/`Duration`'s convention for tier-3 value
/// types.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct UnixNanos(u64);

impl UnixNanos {
    /// The Unix epoch itself.
    pub const EPOCH: Self = Self(0);

    /// Construct from raw nanoseconds since the Unix epoch.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// The raw nanosecond count.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// The value rounded down to whole milliseconds — the common shape for
    /// a wire timestamp or a log field.
    #[must_use]
    pub const fn as_millis(self) -> u64 {
        self.0 / 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::UnixNanos;

    #[test]
    fn as_millis_truncates_toward_zero() {
        let reading = UnixNanos::from_nanos(1_753_500_000_999_999);

        assert_eq!(reading.as_millis(), 1_753_500_000);
    }

    #[test]
    fn epoch_round_trips_through_raw_nanos() {
        assert_eq!(UnixNanos::EPOCH.as_nanos(), 0);
        assert_eq!(UnixNanos::from_nanos(0), UnixNanos::EPOCH);
    }
}
