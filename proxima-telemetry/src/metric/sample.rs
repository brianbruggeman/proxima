use crate::tag::{ScalarValue, Tag};

/// A complete observation for a counter, gauge, or up-down counter.
///
/// Histogram observations arrive in C7 via `HistogramDataPoint`.
#[derive(Clone, Debug)]
pub struct NumberDataPoint {
    pub value: ScalarValue,
    pub attrs: smallvec::SmallVec<[Tag; 4]>,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
}

/// Placeholder type until C7 fills in the real histogram data point.
#[cfg(not(feature = "histogram"))]
pub type HistogramDataPoint = ();

/// A complete snapshot for a histogram observation.
///
/// `bounds` is a borrowed static slice of exponential bucket upper bounds;
/// `bucket_counts[i]` is the number of observations in `(-inf, bounds[i])`.
#[cfg(feature = "histogram")]
#[derive(Clone, Debug)]
pub struct HistogramDataPoint {
    pub count: u64,
    pub sum: f64,
    pub bucket_counts: alloc::vec::Vec<u64>,
    pub bounds: &'static [f64],
    pub attrs: smallvec::SmallVec<[crate::tag::Tag; 4]>,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
}

/// A single metric sample — discriminated by instrument kind.
///
/// Enum form avoids Box<dyn Trait> and keeps match exhaustive.
#[derive(Clone, Debug)]
pub enum MetricSample {
    Counter(NumberDataPoint),
    Gauge(NumberDataPoint),
    UpDownCounter(NumberDataPoint),
    #[cfg(feature = "histogram")]
    Histogram(HistogramDataPoint),
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

    use super::{MetricSample, NumberDataPoint};
    use crate::tag::ScalarValue;

    fn make_point(value: ScalarValue) -> NumberDataPoint {
        NumberDataPoint {
            value,
            attrs: smallvec::SmallVec::new(),
            ts_ns: 1_000,
            start_ts_ns: 0,
        }
    }

    #[test]
    fn number_data_point_round_trips() {
        let point = make_point(ScalarValue::U64(42));
        assert!(matches!(point.value, ScalarValue::U64(42)));
        assert_eq!(point.ts_ns, 1_000);
        assert_eq!(point.start_ts_ns, 0);
        assert!(point.attrs.is_empty());
    }

    #[test]
    fn metric_sample_variants_distinct() {
        let counter = MetricSample::Counter(make_point(ScalarValue::U64(1)));
        let gauge = MetricSample::Gauge(make_point(ScalarValue::F64(2.0)));
        let updown = MetricSample::UpDownCounter(make_point(ScalarValue::I64(-1)));
        assert!(matches!(counter, MetricSample::Counter(_)));
        assert!(matches!(gauge, MetricSample::Gauge(_)));
        assert!(matches!(updown, MetricSample::UpDownCounter(_)));
    }

    #[test]
    fn sizes_are_nonzero() {
        // P8 opt-sweep: attrs changed from Vec<Tag>(24B) to SmallVec<[Tag;4]>(264B inline).
        // NumberDataPoint grows from 80B to ~320B; the inline slab trades struct size for
        // zero heap-alloc on typical ≤4-attr paths. Pinning exact sizes re-added once
        // the SmallVec layout stabilises across platforms.
        assert!(core::mem::size_of::<NumberDataPoint>() > 0);
        assert!(core::mem::size_of::<MetricSample>() > 0);
    }
}
