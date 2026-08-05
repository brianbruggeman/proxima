use core::convert::Infallible;
use core::fmt;
use core::future::Future;

use proxima_primitives::pipe::primitives::Pipe;

use crate::seq_u64s::SeqU64s;
use crate::ticks::Ticks;
use crate::unix_nanos::UnixNanos;

/// The `(ticks, unix_nanos)` correlation point a monotonic tick source is
/// pinned to wall-clock time by — anchored once, re-anchorable later (an
/// NTP/PTP discipline loop calling [`AnchorCell::set`] on a schedule).
///
/// This is not speculative: it replaces two independent hand-rolled
/// versions already in the tree — `src/upstreams/record.rs` (main,
/// `OffsetDateTime::now_utc()` and `Instant::now()` read as two unrelated
/// values with no shared anchor) and `feat/determinism-proof`'s
/// `RecordUpstream::with_clock` (`wall_epoch`/`epoch_nanos` pair, anchored
/// once at construction, `wall_epoch + Duration::from_nanos(now_nanos -
/// epoch_nanos)` computed per read) — promoted here as the general,
/// tick-domain, `no_std` shape both were converging on.
///
/// Seqlock-protected rather than set-once: a
/// caller that never re-anchors pays nothing extra (one seqlock read costs
/// the same as a plain load on the happy path); a caller running an NTP/PTP
/// discipline loop gets a real re-anchor primitive instead of needing to
/// rebuild the whole pipeline. What is deliberately NOT here yet: a
/// frequency-drift/skew-slope correction term (the discipline loop's
/// *rate* correction, as opposed to its periodic *offset* correction).
/// Re-anchoring alone already tracks a disciplined external time source at
/// the loop's correction cadence; adding a continuous skew term is the
/// next increment, and this shape does not need to change to add it — a
/// third seqlock-protected field (parts-per-billion drift) would join
/// `ticks`/`unix_nanos` in the pair without touching [`ToUnixNanos`]'s
/// call site.
pub struct AnchorCell {
    pair: SeqU64s<2>,
}

impl AnchorCell {
    /// Anchor `ticks` to `unix_nanos` now.
    #[must_use]
    pub fn new(ticks: Ticks, unix_nanos: UnixNanos) -> Self {
        Self {
            pair: SeqU64s::new([ticks.as_raw(), unix_nanos.as_nanos()]),
        }
    }

    /// Re-anchor: pin a fresh `(ticks, unix_nanos)` correlation point. The
    /// discipline operation — call this from an NTP/PTP correction loop
    /// each time it resolves a fresh offset.
    pub fn set(&self, ticks: Ticks, unix_nanos: UnixNanos) {
        self.pair.store([ticks.as_raw(), unix_nanos.as_nanos()]);
    }

    /// The current `(ticks, unix_nanos)` anchor pair, read as one
    /// consistent unit.
    #[must_use]
    pub fn get(&self) -> (Ticks, UnixNanos) {
        let [ticks, unix_nanos] = self.pair.load();
        (Ticks::from_raw(ticks), UnixNanos::from_nanos(unix_nanos))
    }
}

// not derived: a derived impl reads each atomic half on its own, so a
// `{:?}` racing a re-anchor can print a pair that was never stored.
impl fmt::Debug for AnchorCell {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ticks, unix_nanos) = self.get();
        formatter
            .debug_struct("AnchorCell")
            .field("ticks", &ticks)
            .field("unix_nanos", &unix_nanos)
            .finish()
    }
}

/// Converts [`Ticks`] to [`UnixNanos`] against a live, re-anchorable
/// [`AnchorCell`] — the monotonic-to-wall-clock bridge, expressed as a
/// plain [`Pipe`] transform (`In = Ticks, Out = UnixNanos`) instead of a
/// bespoke wrapper type.
///
/// Compose it with any tick source via `.and_then`:
///
/// ```
/// use proxima_clock::anchor::{AnchorCell, ToUnixNanos};
/// use proxima_clock::ticks::Ticks;
/// use proxima_clock::unix_nanos::UnixNanos;
/// use proxima_primitives::pipe::ext::PipeExt;
/// use proxima_primitives::pipe::primitives::Pipe;
/// use core::convert::Infallible;
/// use core::future::Future;
///
/// struct HardwareTicks(Ticks);
/// impl Pipe for HardwareTicks {
///     type In = ();
///     type Out = Ticks;
///     type Err = Infallible;
///     fn call(&self, (): ()) -> impl Future<Output = Result<Ticks, Infallible>> {
///         let ticks = self.0;
///         async move { Ok(ticks) }
///     }
/// }
///
/// let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(1_753_500_000_000_000_000));
/// let wall_clock = HardwareTicks(Ticks::from_raw(24_000_000))
///     .and_then(ToUnixNanos::new(&anchor, 24_000_000));
/// // `wall_clock` is itself `impl Pipe<In = (), Out = UnixNanos, Err = Infallible>` —
/// // no `WallClock<C>` wrapper type; the transform stage IS the bridge.
/// ```
///
/// `frequency_hz` is supplied once at construction (a hardware clock's
/// nominal rate is fixed for its lifetime — a PTP discipline loop adjusts
/// [`AnchorCell`]'s offset, not this rate); it is not re-read per call.
#[derive(Debug, Clone, Copy)]
pub struct ToUnixNanos<'anchor> {
    anchor: &'anchor AnchorCell,
    frequency_hz: u64,
}

impl<'anchor> ToUnixNanos<'anchor> {
    /// Build the conversion stage against `anchor`, at `frequency_hz`
    /// (must be > 0 — see [`ToUnixNanos::call`]'s doc for the zero-frequency
    /// clamp).
    #[must_use]
    pub const fn new(anchor: &'anchor AnchorCell, frequency_hz: u64) -> Self {
        Self {
            anchor,
            frequency_hz,
        }
    }
}

impl Pipe for ToUnixNanos<'_> {
    type In = Ticks;
    type Out = UnixNanos;
    type Err = Infallible;

    /// `anchor_unix_nanos + (delta_ticks * 1_000_000_000 / frequency_hz)`,
    /// computed in `u128` so a large `delta_ticks * 1e9` product cannot
    /// overflow `u64` before the division narrows it back down. `u128`
    /// division lowers to a `compiler_builtins` soft-division call on
    /// 32-bit MCU targets (no native 64x64-bit or wider divide instruction)
    /// — accepted here because this conversion runs once per read at the
    /// edge (an export, a log timestamp, an NTP correction check), never
    /// in the hot tick-counting loop where ticks stay raw `u64` all the
    /// way through.
    ///
    /// `delta_ticks` uses [`Ticks::wrapping_sub`], so a free-running
    /// counter wrapping past `u64::MAX` between anchor and read still
    /// converts correctly. A `frequency_hz` of `0` (an implementor
    /// contract violation — no real oscillator runs at 0 Hz) is clamped
    /// to `1` rather than dividing by zero and panicking: a bare-metal
    /// caller may have no panic handler configured to unwind or abort
    /// safely, so this conversion stays total.
    fn call(&self, ticks: Ticks) -> impl Future<Output = Result<UnixNanos, Infallible>> {
        let (anchor_ticks, anchor_unix_nanos) = self.anchor.get();
        let delta_ticks = ticks.wrapping_sub(anchor_ticks);
        let frequency_hz = u128::from(self.frequency_hz.max(1));
        let delta_nanos = u128::from(delta_ticks) * 1_000_000_000 / frequency_hz;
        let unix_nanos = u128::from(anchor_unix_nanos.as_nanos()).saturating_add(delta_nanos);
        let unix_nanos = UnixNanos::from_nanos(u64::try_from(unix_nanos).unwrap_or(u64::MAX));
        async move { Ok(unix_nanos) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{AnchorCell, ToUnixNanos};
    use crate::ticks::Ticks;
    use crate::unix_nanos::UnixNanos;
    use proxima_primitives::block_on;
    use proxima_primitives::pipe::primitives::Pipe;

    // 24 MHz: the ARM generic timer's real rate, and one that does not
    // divide evenly into 1e9 (1_000_000_000 / 24_000_000 = 41.66...) — the
    // conversion's rounding is genuinely exercised, not hidden by a tidy
    // power-of-ten frequency.
    const ARM_GENERIC_TIMER_HZ: u64 = 24_000_000;
    const PTP_HARDWARE_CLOCK_HZ: u64 = 100_000_000;
    const TSC_NOMINAL_HZ: u64 = 1_000_000_000;
    const PRIME_WHEEL_HZ: u64 = 1_000;

    fn convert(anchor: &AnchorCell, frequency_hz: u64, ticks: Ticks) -> UnixNanos {
        block_on(Pipe::call(&ToUnixNanos::new(anchor, frequency_hz), ticks))
            .expect("conversion never fails")
    }

    #[test]
    fn at_the_anchor_point_wall_clock_equals_the_anchor() {
        let anchor = AnchorCell::new(
            Ticks::from_raw(0),
            UnixNanos::from_nanos(1_753_500_000_000_000_000),
        );

        let now = convert(&anchor, ARM_GENERIC_TIMER_HZ, Ticks::from_raw(0));

        assert_eq!(now, UnixNanos::from_nanos(1_753_500_000_000_000_000));
    }

    #[test]
    fn arm_generic_timer_24mhz_one_second_of_ticks_advances_one_second() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let one_second_later = convert(
            &anchor,
            ARM_GENERIC_TIMER_HZ,
            Ticks::from_raw(ARM_GENERIC_TIMER_HZ),
        );

        assert_eq!(one_second_later, UnixNanos::from_nanos(1_000_000_000));
    }

    #[test]
    fn arm_generic_timer_24mhz_rounding_is_exercised_not_hidden() {
        // one tick at 24 MHz is 1_000_000_000 / 24_000_000 = 41.666... ns;
        // integer division truncates to 41, proving the truncation
        // actually happens instead of a power-of-ten frequency (where the
        // division is always exact) masking it.
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let one_tick_later = convert(&anchor, ARM_GENERIC_TIMER_HZ, Ticks::from_raw(1));

        assert_eq!(one_tick_later, UnixNanos::from_nanos(41));
    }

    #[test]
    fn ptp_hardware_clock_100mhz_one_millisecond_of_ticks() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let one_millisecond_later =
            convert(&anchor, PTP_HARDWARE_CLOCK_HZ, Ticks::from_raw(100_000));

        assert_eq!(one_millisecond_later, UnixNanos::from_nanos(1_000_000));
    }

    #[test]
    fn tsc_nominal_1ghz_ticks_equal_nanos_1to1() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let converted = convert(&anchor, TSC_NOMINAL_HZ, Ticks::from_raw(1_234_567));

        assert_eq!(converted, UnixNanos::from_nanos(1_234_567));
    }

    #[test]
    fn prime_wheel_1khz_matches_the_wheels_millisecond_tick() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let converted = convert(&anchor, PRIME_WHEEL_HZ, Ticks::from_raw(5_000));

        assert_eq!(
            converted,
            UnixNanos::from_nanos(5_000_000_000),
            "5000 ticks at 1000 Hz is 5 seconds, matching prime's ms-resolution wheel"
        );
    }

    #[test]
    fn counter_wraparound_still_converts_correctly() {
        let anchor = AnchorCell::new(Ticks::from_raw(u64::MAX - 9), UnixNanos::from_nanos(0));

        // the counter wraps past u64::MAX and comes back around to 10:
        // 10 ticks to wrap (9 -> u64::MAX then +1) plus 10 more = 20 ticks
        // elapsed, at 1 GHz nominal that is 20 ns.
        let after_wrap = convert(&anchor, TSC_NOMINAL_HZ, Ticks::from_raw(10));

        assert_eq!(after_wrap, UnixNanos::from_nanos(20));
    }

    #[test]
    fn re_anchoring_disciplines_subsequent_reads_without_rebuilding_the_pipeline() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));
        let stage = ToUnixNanos::new(&anchor, TSC_NOMINAL_HZ);

        let before_discipline =
            block_on(Pipe::call(&stage, Ticks::from_raw(1_000))).expect("conversion never fails");
        assert_eq!(before_discipline, UnixNanos::from_nanos(1_000));

        // an NTP-style correction resolves a fresh offset: the same ticks
        // domain is now known to correspond to a different wall-clock
        // instant. re-anchoring is exactly this: no new pipeline, no new
        // `ToUnixNanos` — `stage` observes the discipline immediately.
        anchor.set(Ticks::from_raw(1_000), UnixNanos::from_nanos(50_000_000));

        let after_discipline =
            block_on(Pipe::call(&stage, Ticks::from_raw(2_000))).expect("conversion never fails");
        assert_eq!(after_discipline, UnixNanos::from_nanos(50_001_000));
    }

    #[test]
    fn zero_frequency_clamps_instead_of_dividing_by_zero() {
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

        let converted = convert(&anchor, 0, Ticks::from_raw(3));

        assert_eq!(
            converted,
            UnixNanos::from_nanos(3_000_000_000),
            "frequency_hz=0 clamps to 1 Hz rather than panicking on divide-by-zero"
        );
    }

    // proptest needs an allocator and a test harness, so the properties are
    // std-tier; the arithmetic they cover is `core`-only.
    #[cfg(feature = "std")]
    mod properties {
        use super::{
            AnchorCell, Pipe, TSC_NOMINAL_HZ, Ticks, ToUnixNanos, UnixNanos, block_on, convert,
        };
        use proptest::prelude::{ProptestConfig, any, prop_assert, prop_assert_eq, proptest};

        /// 1 Hz up to ten times a nominal TSC. Drawing a frequency from the
        /// whole `u64` space instead makes every property vacuous: at ~9e18 Hz
        /// the elapsed nanosecond term rounds to 0 and the conversion
        /// degenerates to the anchor, which a mutation of the saturation arm
        /// survives unnoticed (measured — three properties went green against
        /// a `u64::MAX` -> `0` mutation until this range replaced `any`). The
        /// 0 Hz contract violation is owned deterministically by
        /// [`super::zero_frequency_clamps_instead_of_dividing_by_zero`]
        /// rather than left to a draw.
        const PLAUSIBLE_COUNTER_HZ: core::ops::RangeInclusive<u64> = 1..=10_000_000_000;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(512))]

            /// Reading the anchor's own tick value must reproduce the anchor's
            /// own wall-clock value exactly, whatever the counter frequency —
            /// the fixed point the whole conversion is defined around.
            #[test]
            fn the_anchor_point_is_a_fixed_point(
                anchor_ticks in any::<u64>(),
                anchor_nanos in any::<u64>(),
                frequency_hz in PLAUSIBLE_COUNTER_HZ,
            ) {
                let anchor = AnchorCell::new(
                    Ticks::from_raw(anchor_ticks),
                    UnixNanos::from_nanos(anchor_nanos),
                );

                let converted = convert(&anchor, frequency_hz, Ticks::from_raw(anchor_ticks));

                prop_assert_eq!(converted, UnixNanos::from_nanos(anchor_nanos));
            }

            /// A clock never reads earlier than its own anchor: the `u128`
            /// widening, the `saturating_add` and the `u64::MAX` narrowing
            /// fallback all have to preserve that, for every input including
            /// the ones that saturate.
            #[test]
            fn a_reading_never_precedes_the_anchor(
                anchor_ticks in any::<u64>(),
                anchor_nanos in any::<u64>(),
                frequency_hz in PLAUSIBLE_COUNTER_HZ,
                elapsed_ticks in any::<u64>(),
            ) {
                let anchor = AnchorCell::new(
                    Ticks::from_raw(anchor_ticks),
                    UnixNanos::from_nanos(anchor_nanos),
                );
                let read = Ticks::from_raw(anchor_ticks.wrapping_add(elapsed_ticks));

                let converted = convert(&anchor, frequency_hz, read);

                prop_assert!(converted >= UnixNanos::from_nanos(anchor_nanos));
            }

            /// More elapsed ticks can never convert to an earlier instant, so a
            /// span measured across two reads can never come out negative.
            #[test]
            fn more_elapsed_ticks_never_convert_backwards(
                anchor_ticks in any::<u64>(),
                anchor_nanos in any::<u64>(),
                frequency_hz in PLAUSIBLE_COUNTER_HZ,
                first_elapsed in any::<u64>(),
                second_elapsed in any::<u64>(),
            ) {
                let (earlier, later) = if first_elapsed <= second_elapsed {
                    (first_elapsed, second_elapsed)
                } else {
                    (second_elapsed, first_elapsed)
                };
                let anchor = AnchorCell::new(
                    Ticks::from_raw(anchor_ticks),
                    UnixNanos::from_nanos(anchor_nanos),
                );

                let earlier_reading = convert(
                    &anchor,
                    frequency_hz,
                    Ticks::from_raw(anchor_ticks.wrapping_add(earlier)),
                );
                let later_reading = convert(
                    &anchor,
                    frequency_hz,
                    Ticks::from_raw(anchor_ticks.wrapping_add(later)),
                );

                prop_assert!(earlier_reading <= later_reading);
            }

            /// At 1 GHz a tick IS a nanosecond, so the whole conversion must
            /// reduce to `anchor.saturating_add(elapsed)` with no rounding
            /// term anywhere — the oracle for the general case's arithmetic.
            #[test]
            fn a_1ghz_counter_converts_by_plain_saturating_addition(
                anchor_ticks in any::<u64>(),
                anchor_nanos in any::<u64>(),
                elapsed_ticks in any::<u64>(),
            ) {
                let anchor = AnchorCell::new(
                    Ticks::from_raw(anchor_ticks),
                    UnixNanos::from_nanos(anchor_nanos),
                );
                let read = Ticks::from_raw(anchor_ticks.wrapping_add(elapsed_ticks));

                let converted = convert(&anchor, TSC_NOMINAL_HZ, read);

                prop_assert_eq!(
                    converted,
                    UnixNanos::from_nanos(anchor_nanos.saturating_add(elapsed_ticks))
                );
            }

            /// The stage holds a borrow, not a snapshot: whatever a discipline
            /// loop last stored is what the next reading is measured against.
            #[test]
            fn re_anchoring_is_observed_by_a_stage_built_beforehand(
                first_nanos in any::<u64>(),
                second_nanos in any::<u64>(),
            ) {
                let anchor = AnchorCell::new(Ticks::ZERO, UnixNanos::from_nanos(first_nanos));
                let stage = ToUnixNanos::new(&anchor, TSC_NOMINAL_HZ);

                anchor.set(Ticks::ZERO, UnixNanos::from_nanos(second_nanos));
                let reading = block_on(Pipe::call(&stage, Ticks::ZERO))
                    .expect("conversion never fails");

                prop_assert_eq!(reading, UnixNanos::from_nanos(second_nanos));
            }
        }
    }

    // `format!` needs an allocator, so the rendering assertions are std-tier;
    // the `Debug` impl itself is `core`-only and compiles on the floor.
    #[cfg(feature = "std")]
    mod rendering {
        use super::{AnchorCell, Ticks, UnixNanos};

        #[test]
        fn debug_renders_the_live_anchor_not_the_constructed_one() {
            let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));

            anchor.set(Ticks::from_raw(1_000), UnixNanos::from_nanos(50_000_000));

            assert_eq!(
                format!("{anchor:?}"),
                "AnchorCell { ticks: Ticks(1000), unix_nanos: UnixNanos(50000000) }"
            );
        }
    }
}
