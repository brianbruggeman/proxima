pub mod counter;
#[cfg(feature = "instrument-metrics")]
pub mod exemplar;
pub mod gauge;
#[cfg(feature = "histogram")]
pub mod histogram;
#[cfg(feature = "instrument-metrics")]
pub mod instrument_config;
pub mod sample;
pub mod updown;

pub use counter::Counter;
pub use gauge::Gauge;
pub use sample::{MetricSample, NumberDataPoint};
pub use updown::UpDownCounter;

#[cfg(feature = "histogram")]
pub use histogram::Histogram;
#[cfg(feature = "histogram")]
pub use sample::HistogramDataPoint;

/// Increment a `Counter` instrument.
///
/// Forms:
/// - `counter!(INSTRUMENT, delta)` — no tags
/// - `counter!(INSTRUMENT, delta, "key" = value, ...)` — with tags
/// - `counter!(INSTRUMENT, delta, recorder = rec)` — also mirror the delta
///   into `rec`'s same-named registered counter (by [`Counter::name`]), so an
///   installed/explicit [`crate::recorder::Recorder`]'s drain captures this
///   observation alongside the local accumulator. `rec` must be a `&Recorder`
///   (or deref to one) — mirrors `#[instrument(recorder = rec)]`'s explicit
///   seam, EXPLICIT by design (see `capture::capture`): no ambient fallback,
///   so a test recorder never depends on process-global state.
/// - `counter!(INSTRUMENT, delta, recorder = rec, "key" = value, ...)` — both
#[macro_export]
macro_rules! counter {
    ($instrument:expr, $value:expr $(,)?) => {{
        $instrument.add($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr $(,)?) => {{
        let __instrument = &$instrument;
        __instrument.add($value, &[]);
        $recorder.counter(__instrument.name).add($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        let __instrument = &$instrument;
        __instrument.add($value, &__tags);
        $recorder.counter(__instrument.name).add($value, &__tags);
    }};
    ($instrument:expr, $value:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        $instrument.add($value, &__tags);
    }};
}

/// Record a `Gauge` observation.
///
/// Forms:
/// - `gauge!(INSTRUMENT, value)` — no tags (stores as u64 bits)
/// - `gauge!(INSTRUMENT, value, "key" = v, ...)` — with tags
/// - `gauge!(INSTRUMENT, value, recorder = rec)` — also mirror into `rec`'s
///   same-named registered gauge; see [`counter!`] for the explicit-seam contract.
/// - `gauge!(INSTRUMENT, value, recorder = rec, "key" = v, ...)` — both
#[macro_export]
macro_rules! gauge {
    ($instrument:expr, $value:expr $(,)?) => {{
        $instrument.set_u64($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr $(,)?) => {{
        let __instrument = &$instrument;
        __instrument.set_u64($value, &[]);
        $recorder.gauge(__instrument.name).set_u64($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        let __instrument = &$instrument;
        __instrument.set_u64($value, &__tags);
        $recorder.gauge(__instrument.name).set_u64($value, &__tags);
    }};
    ($instrument:expr, $value:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        $instrument.set_u64($value, &__tags);
    }};
}

/// Record an observation into a `Histogram` instrument.
///
/// Forms:
/// - `histogram!(INSTRUMENT, value)` — no tags
/// - `histogram!(INSTRUMENT, value, "key" = v, ...)` — with tags
/// - `histogram!(INSTRUMENT, value, recorder = rec)` — also mirror into `rec`'s
///   same-named registered histogram; see [`counter!`] for the explicit-seam
///   contract. `rec`'s drain snapshots bucket counts, so this is how a static
///   histogram instrument becomes assertable via `capture`.
#[cfg(feature = "histogram")]
#[macro_export]
macro_rules! histogram {
    ($instrument:expr, $value:expr $(,)?) => {{
        $instrument.record($value);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr $(,)?) => {{
        let __instrument = &$instrument;
        __instrument.record($value);
        $recorder.histogram(__instrument.name).record($value);
        // correlate: a histogram recorded inside a span points its exemplar at
        // that span's trace, the same current-span read the log path uses.
        $recorder.record_current_exemplar(__instrument.name, $value as f64);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        // v1: attrs attach at the export layer (C9 attr-set sharding), not the
        // histogram primitive; validate the tag syntax then record tagless.
        let _ = __tags;
        let __instrument = &$instrument;
        __instrument.record($value);
        $recorder.histogram(__instrument.name).record($value);
        $recorder.record_current_exemplar(__instrument.name, $value as f64);
    }};
    ($instrument:expr, $value:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        // v1: attrs attach at the export layer (C9 attr-set sharding), not the
        // histogram primitive; validate the tag syntax then record tagless.
        let _ = __tags;
        $instrument.record($value);
    }};
}

/// Add a signed delta to an `UpDownCounter` instrument.
///
/// Forms:
/// - `updown!(INSTRUMENT, delta)` — no tags
/// - `updown!(INSTRUMENT, delta, "key" = value, ...)` — with tags
/// - `updown!(INSTRUMENT, delta, recorder = rec)` — also mirror into `rec`'s
///   same-named registered up-down counter; see [`counter!`] for the
///   explicit-seam contract.
/// - `updown!(INSTRUMENT, delta, recorder = rec, "key" = value, ...)` — both
#[macro_export]
macro_rules! updown {
    ($instrument:expr, $value:expr $(,)?) => {{
        $instrument.add($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr $(,)?) => {{
        let __instrument = &$instrument;
        __instrument.add($value, &[]);
        $recorder.updown_counter(__instrument.name).add($value, &[]);
    }};
    ($instrument:expr, $value:expr, recorder = $recorder:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        let __instrument = &$instrument;
        __instrument.add($value, &__tags);
        $recorder.updown_counter(__instrument.name).add($value, &__tags);
    }};
    ($instrument:expr, $value:expr, $($rest:tt)+) => {{
        let mut __tags: alloc::vec::Vec<$crate::tag::Tag> = alloc::vec::Vec::new();
        $crate::tag!(__tags, $($rest)+);
        $instrument.add($value, &__tags);
    }};
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use rstest::rstest;

    use crate::metric::{Counter, Gauge, MetricSample, NumberDataPoint, UpDownCounter};
    use crate::tag::ScalarValue;

    // 1. Counter::add increments the internal value
    #[test]
    fn counter_add_increments() {
        let counter = Counter::new("hits");
        counter.add(5, &[]);
        assert_eq!(counter.get(), 5);
    }

    // 2. Gauge::set updates the internal value
    #[test]
    fn gauge_set_updates_value() {
        let gauge = Gauge::new("load");
        gauge.set_u64(10, &[]);
        assert_eq!(gauge.get_u64(), 10);
        gauge.set_u64(20, &[]);
        assert_eq!(gauge.get_u64(), 20);
    }

    // 3. UpDownCounter positive increments, negative decrements
    #[rstest]
    #[case::increment(5i64, 3i64, 8i64)]
    #[case::decrement(10i64, -4i64, 6i64)]
    #[case::net_zero(5i64, -5i64, 0i64)]
    fn updown_counter_positive_and_negative(
        #[case] first: i64,
        #[case] second: i64,
        #[case] expected: i64,
    ) {
        let counter = UpDownCounter::new("queue");
        counter.add(first, &[]);
        counter.add(second, &[]);
        assert_eq!(counter.get(), expected);
    }

    // 4. counter! macro adds value with no tags
    #[test]
    fn counter_macro_no_tags() {
        static COUNTER: Counter = Counter::new("macro_test");
        counter!(COUNTER, 5u64);
        assert!(COUNTER.get() >= 5);
    }

    // 5. counter! macro adds value with one tag
    #[test]
    fn counter_macro_with_tag() {
        static TAGGED: Counter = Counter::new("tagged_test");
        counter!(TAGGED, 5u64, "route" = "/v1/x");
        assert!(TAGGED.get() >= 5);
    }

    // 6. NumberDataPoint construction round-trips
    #[test]
    fn number_data_point_round_trips() {
        let point = NumberDataPoint {
            value: ScalarValue::U64(42),
            attrs: smallvec::SmallVec::new(),
            ts_ns: 999,
            start_ts_ns: 0,
        };
        assert!(matches!(point.value, ScalarValue::U64(42)));
        assert_eq!(point.ts_ns, 999);
        assert_eq!(point.start_ts_ns, 0);
    }

    // 7. MetricSample enum variants are distinct
    #[test]
    fn metric_sample_variants_distinct() {
        let make = |value| NumberDataPoint {
            value,
            attrs: smallvec::SmallVec::new(),
            ts_ns: 0,
            start_ts_ns: 0,
        };
        let counter = MetricSample::Counter(make(ScalarValue::U64(1)));
        let gauge = MetricSample::Gauge(make(ScalarValue::F64(2.0)));
        let updown = MetricSample::UpDownCounter(make(ScalarValue::I64(-1)));
        assert!(matches!(counter, MetricSample::Counter(_)));
        assert!(matches!(gauge, MetricSample::Gauge(_)));
        assert!(matches!(updown, MetricSample::UpDownCounter(_)));
    }

    // 8. concurrent: 8 threads each calling Counter::add(1, &[]) results in total = 8
    #[test]
    fn counter_concurrent_8_threads() {
        extern crate std;
        use std::thread;

        static CONCURRENT: Counter = Counter::new("concurrent");
        let threads: alloc::vec::Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(|| {
                    CONCURRENT.add(1, &[]);
                })
            })
            .collect();
        for handle in threads {
            handle.join().expect("thread panicked");
        }
        assert_eq!(CONCURRENT.get(), 8);
    }

    // 9. size assertions — pin sizeof(Counter); NumberDataPoint/MetricSample sizes
    // updated in P8 opt-sweep (attrs: Vec<Tag>→SmallVec<[Tag;4]>) — inline buffer grows
    // struct size but eliminates heap alloc on ≤4-attr typical path.
    #[test]
    fn sizes_are_pinned() {
        assert_eq!(core::mem::size_of::<Counter>(), 56);
        assert!(core::mem::size_of::<NumberDataPoint>() > 0);
        assert!(core::mem::size_of::<MetricSample>() > 0);
    }

    // 6 (c7). macro: histogram!(H, 1.5) calls record with no tags
    #[cfg(feature = "histogram")]
    #[test]
    fn histogram_macro_no_tags() {
        use crate::metric::Histogram;
        static HIST: Histogram<f64> = Histogram::new("macro_hist");
        histogram!(HIST, 1.5f64);
        assert!(HIST.count() >= 1);
    }

    // 7 (c7). macro: histogram!(H, 1.5, "route" = "/v1/x") calls record with one tag
    #[cfg(feature = "histogram")]
    #[test]
    fn histogram_macro_with_tag() {
        use crate::metric::Histogram;
        static TAGGED_HIST: Histogram<f64> = Histogram::new("tagged_hist");
        histogram!(TAGGED_HIST, 1.5f64, "route" = "/v1/x");
        assert!(TAGGED_HIST.count() >= 1);
    }

    // 8 (c7). MetricSample::Histogram variant constructs and round-trips through enum
    #[cfg(feature = "histogram")]
    #[test]
    fn metric_sample_histogram_variant_round_trips() {
        use crate::metric::HistogramDataPoint;

        static BOUNDS: &[f64] = &[1.0, 2.0, 4.0, 8.0];
        let point = HistogramDataPoint {
            count: 5,
            sum: 12.5,
            bucket_counts: vec![1, 2, 1, 1, 0],
            bounds: BOUNDS,
            attrs: smallvec::SmallVec::new(),
            ts_ns: 2_000,
            start_ts_ns: 0,
        };
        let sample = MetricSample::Histogram(point.clone());
        assert!(matches!(sample, MetricSample::Histogram(_)));
        if let MetricSample::Histogram(dp) = sample {
            assert_eq!(dp.count, 5);
            assert!((dp.sum - 12.5).abs() < f64::EPSILON);
            assert_eq!(dp.bounds.len(), 4);
        }
    }

    // 9b (c7). size assertions for HistogramDataPoint and Histogram<f64>
    // HistogramDataPoint.attrs changed to SmallVec<[Tag;4]> in P8 opt-sweep —
    // exact byte size updated below after measuring; Histogram<f64> is unchanged.
    #[cfg(feature = "histogram")]
    #[test]
    fn histogram_sizes_are_pinned() {
        use crate::metric::{Histogram, HistogramDataPoint};
        assert!(core::mem::size_of::<HistogramDataPoint>() > 0);
        // 3×&str(48)+base(8)+min(8)+bucket_count(8)+buckets(256)+count(8)+sum(8) = 344
        assert_eq!(core::mem::size_of::<Histogram<f64>>(), 344);
    }

    // C9: a static instrument's local atomic is one view; `recorder = rec` is
    // the OTHER view -- an explicit, non-ambient seam into a recorder's own
    // same-named registered instrument, drainable into a pipe a test can read
    // back. This is what makes `counter!`/`gauge!`/`updown!`/`histogram!` (the
    // documented static-instrument idiom) assertable via `capture`, not just
    // the separate `Recorder::counter(name)` accessor.
    // capture() builds a real Recorder, which uses proxima-core's Ring/
    // StaticRing internally -- cfg-swapped to loom under `--features loom`
    // (forwarded via proxima-core/loom), only usable inside an actual
    // loom::model(...) closure, which these plain #[test] functions don't
    // provide.
    #[cfg(all(feature = "std", not(feature = "loom")))]
    mod recorder_routing {
        use crate::capture::capture;
        use crate::metric::{Counter, Gauge, UpDownCounter};

        // counter! with `recorder = rec`: the local static AND the recorder's
        // same-named registered counter both see the delta.
        #[test]
        fn counter_macro_routes_to_explicit_recorder() {
            static ROUTED: Counter = Counter::new("routed_counter");
            let captured = capture(|rec| {
                counter!(ROUTED, 5u64, recorder = rec);
                counter!(ROUTED, 3u64, recorder = rec);
            });
            assert_eq!(ROUTED.get(), 8, "local atomic still accumulates");
            let total: u64 = captured
                .metrics()
                .iter()
                .filter_map(|sample| match sample {
                    crate::metric::MetricSample::Counter(point) => {
                        if let crate::tag::ScalarValue::U64(value) = point.value {
                            Some(value)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .sum();
            assert_eq!(
                total,
                8,
                "recorder-side counter drains the same total; dump: {}",
                captured.dump()
            );
        }

        // gauge! with `recorder = rec`: last-value semantics preserved on both sides.
        #[test]
        fn gauge_macro_routes_to_explicit_recorder() {
            static ROUTED: Gauge = Gauge::new("routed_gauge");
            let captured = capture(|rec| {
                gauge!(ROUTED, 10u64, recorder = rec);
                gauge!(ROUTED, 42u64, recorder = rec);
            });
            assert_eq!(ROUTED.get_u64(), 42, "local gauge holds the last value");
            assert!(
                captured.metrics().iter().any(|sample| matches!(
                    sample,
                    crate::metric::MetricSample::Gauge(point)
                        if matches!(point.value, crate::tag::ScalarValue::U64(42))
                )),
                "recorder-side gauge drains the last value too; dump: {}",
                captured.dump()
            );
        }

        // updown! with `recorder = rec`: signed delta mirrored to the recorder.
        #[test]
        fn updown_macro_routes_to_explicit_recorder() {
            static ROUTED: UpDownCounter = UpDownCounter::new("routed_updown");
            let captured = capture(|rec| {
                updown!(ROUTED, 5i64, recorder = rec);
                updown!(ROUTED, -2i64, recorder = rec);
            });
            assert_eq!(ROUTED.get(), 3, "local up-down counter nets the deltas");
            assert!(
                captured.metrics().iter().any(|sample| matches!(
                    sample,
                    crate::metric::MetricSample::UpDownCounter(point)
                        if matches!(point.value, crate::tag::ScalarValue::I64(3))
                )),
                "recorder-side up-down counter drains the net total; dump: {}",
                captured.dump()
            );
        }

        // histogram! with `recorder = rec`: both the local bucket slab and the
        // recorder-owned histogram observe every value.
        #[cfg(feature = "histogram")]
        #[test]
        fn histogram_macro_routes_to_explicit_recorder() {
            use crate::metric::Histogram;

            static ROUTED: Histogram<f64> = Histogram::new("routed_histogram");
            let captured = capture(|rec| {
                histogram!(ROUTED, 1.5f64, recorder = rec);
                histogram!(ROUTED, 2.5f64, recorder = rec);
            });
            assert_eq!(
                ROUTED.count(),
                2,
                "local histogram counted both observations"
            );
            assert!(
                captured.metrics().iter().any(|sample| matches!(
                    sample,
                    crate::metric::MetricSample::Histogram(point) if point.count == 2
                )),
                "recorder-side histogram drains the same count; dump: {}",
                captured.dump()
            );
        }

        // with tags AND an explicit recorder together (the fourth macro arm).
        #[test]
        fn counter_macro_routes_with_tags_and_recorder() {
            static ROUTED: Counter = Counter::new("routed_counter_tagged");
            let captured = capture(|rec| {
                counter!(ROUTED, 1u64, recorder = rec, "route" = "/v1/x");
            });
            assert_eq!(ROUTED.get(), 1);
            assert_eq!(
                captured.metrics().len(),
                1,
                "tagged form still routes to the recorder; dump: {}",
                captured.dump()
            );
        }

        // two independent `capture()` calls -- each its own private recorder,
        // no process-global touched -- never see each other's routed metrics.
        // This is the isolation guarantee: a static instrument is a shared
        // process-wide accumulator, but the RECORDER side of `recorder = rec`
        // is scoped to whichever recorder the caller explicitly handed it.
        #[test]
        fn explicit_recorder_routing_is_isolated_across_captures() {
            static SHARED: Counter = Counter::new("isolation_counter");

            let first = capture(|rec| {
                counter!(SHARED, 10u64, recorder = rec);
            });
            let second = capture(|rec| {
                counter!(SHARED, 100u64, recorder = rec);
            });

            let first_total: u64 = first
                .metrics()
                .iter()
                .filter_map(|sample| match sample {
                    crate::metric::MetricSample::Counter(point) => {
                        if let crate::tag::ScalarValue::U64(value) = point.value {
                            Some(value)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .sum();
            let second_total: u64 = second
                .metrics()
                .iter()
                .filter_map(|sample| match sample {
                    crate::metric::MetricSample::Counter(point) => {
                        if let crate::tag::ScalarValue::U64(value) = point.value {
                            Some(value)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .sum();

            assert_eq!(first_total, 10, "first capture sees only its own delta");
            assert_eq!(
                second_total, 100,
                "second capture sees only its own delta, not 110"
            );
        }
    }
}
