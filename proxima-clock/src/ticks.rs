/// A hardware counter reading: whatever a monotonic tick source produced,
/// at that source's own frequency.
///
/// Ticks, not nanoseconds. Deriving nanoseconds costs a multiply and a
/// divide per read — see [`crate::anchor::ToUnixNanos`] for where that
/// conversion belongs (the edge, once per anchor/export), never on the hot
/// path where a caller just wants `now_ticks - then_ticks`. A counter's
/// rate is whatever the hardware runs at: 24 MHz for the ARM generic timer
/// (`CNTVCT_EL0`), ~1 GHz nominal for `RDTSC`, 100 MHz for a PTP hardware
/// clock, 1000 Hz for `prime`'s timer wheel (see
/// `prime::core::timer::Clock`, the sibling tick source this type is
/// deliberately compatible with).
///
/// `#[repr(transparent)]` over `u64`, matching
/// `proxima_protocols::Instant`/`Duration`'s convention for tier-3 value
/// types.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ticks(u64);

impl Ticks {
    /// The zero tick.
    pub const ZERO: Self = Self(0);

    /// Construct from a raw tick count. Hardware source impls use this;
    /// the value's meaning (which counter, which frequency) travels
    /// separately, alongside the source that produced it.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw tick count.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// `self - earlier`, wrapping on counter rollover (a free-running
    /// hardware counter wraps at `u64::MAX`; the elapsed count is still
    /// correct via modular arithmetic as long as fewer than `u64::MAX`
    /// ticks elapsed between the two reads, true for any counter this
    /// side of the heat death of the universe).
    #[must_use]
    pub const fn wrapping_sub(self, earlier: Self) -> u64 {
        self.0.wrapping_sub(earlier.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Ticks;

    #[test]
    fn wrapping_sub_handles_counter_rollover() {
        let before_wrap = Ticks::from_raw(u64::MAX - 5);
        let after_wrap = Ticks::from_raw(4);

        let elapsed = after_wrap.wrapping_sub(before_wrap);

        assert_eq!(
            elapsed, 10,
            "5 ticks to wrap plus 5 ticks past it, via modular arithmetic"
        );
    }

    #[test]
    fn wrapping_sub_is_zero_for_identical_reads() {
        let reading = Ticks::from_raw(24_000_000);

        assert_eq!(reading.wrapping_sub(reading), 0);
    }
}
