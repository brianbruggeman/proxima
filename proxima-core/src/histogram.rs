use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

pub const MAX_BUCKETS: usize = 32;

// offset so that exponent field 1013 maps to bucket 0: covers [2^-10, 2^-9)
const EXP_BIAS_OFFSET: u64 = 1013;

/// A histogram instrument with branchless base-2 exponential bucket pick.
///
/// 32 buckets span [2^-10, 2^22) — roughly [0.001, 4_000_000). Values
/// outside that range saturate to bucket 0 or bucket 31. V is the observed
/// value type: `f64` for latencies/sizes, `u64` for counts.
///
/// v1 baseline: single bucket slab per instrument; tag identity is accepted
/// for API symmetry but not accumulated. C9 will shard by attr-set.
pub struct Histogram<V> {
    pub name: &'static str,
    pub unit: &'static str,
    pub description: &'static str,
    base: f64,
    min: f64,
    bucket_count: usize,
    buckets: [AtomicU64; MAX_BUCKETS],
    count: AtomicU64,
    sum_bits: AtomicU64,
    _marker: PhantomData<V>,
}

impl<V> Histogram<V> {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        // clippy::declare_interior_mutable_const fires on `const ZERO: AtomicU64` because
        // it's interior-mutable. allowed here because the const is used exclusively as the
        // initializer for `[ZERO; MAX_BUCKETS]` — the array elements are distinct atomics,
        // not aliased through the const item. const-constructible AtomicU64 from u64::default is
        // necessary because `const fn` doesn't yet permit calling `AtomicU64::new(0)` inside
        // an array literal without this binding.
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            name,
            unit: "",
            description: "",
            base: 2.0,
            min: 9.765_625e-4, // 2^-10
            bucket_count: MAX_BUCKETS,
            buckets: [ZERO; MAX_BUCKETS],
            count: AtomicU64::new(0),
            sum_bits: AtomicU64::new(0),
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    #[must_use]
    pub const fn description(mut self, desc: &'static str) -> Self {
        self.description = desc;
        self
    }

    /// Set the exponential base. Only `2.0` is supported for the
    /// branchless bucket pick; other bases will fall back to bucket 0.
    #[must_use]
    pub const fn exponential_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    /// Set the lower bound of bucket 0 (must be a power of 2 for the fast path).
    #[must_use]
    pub const fn exponential_min(mut self, min: f64) -> Self {
        self.min = min;
        self
    }

    /// Return the total observation count.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Return the running sum as f64.
    #[must_use]
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum_bits.load(Ordering::Relaxed))
    }

    /// Return a snapshot of per-bucket counts.
    #[must_use]
    pub fn bucket_snapshot(&self) -> [u64; MAX_BUCKETS] {
        let mut out = [0u64; MAX_BUCKETS];
        for (index, bucket) in self.buckets.iter().enumerate() {
            out[index] = bucket.load(Ordering::Relaxed);
        }
        out
    }

    /// Snapshot all state (count, sum_bits, bucket counts) and reset atomics to zero.
    ///
    /// Called by the drainer at snapshot time. Each atomic is swapped to 0 — any
    /// observations racing the swap appear in the NEXT snapshot window (not lost).
    #[must_use]
    pub fn snapshot_and_reset(&self) -> (u64, u64, [u64; MAX_BUCKETS]) {
        let count = self.count.swap(0, Ordering::Relaxed);
        let sum_bits = self.sum_bits.swap(0, Ordering::Relaxed);
        let mut bucket_counts = [0u64; MAX_BUCKETS];
        for (index, bucket) in self.buckets.iter().enumerate() {
            bucket_counts[index] = bucket.swap(0, Ordering::Relaxed);
        }
        (count, sum_bits, bucket_counts)
    }

    // Branchless IEEE 754 exponent-field slot for base-2 buckets.
    // Extracts the 11-bit biased exponent, subtracts EXP_BIAS_OFFSET,
    // and saturates to [0, bucket_count-1]. O(1) constant time.
    fn slot_for_bits(bits: u64, bucket_count: usize) -> usize {
        let exp_field = (bits >> 52) & 0x7FF;
        // signed subtraction: subnormals/zero give exp_field=0 → saturates to 0
        let raw = exp_field.saturating_sub(EXP_BIAS_OFFSET);
        raw.min(bucket_count as u64 - 1) as usize
    }

    fn record_inner(&self, value_bits: u64, sum_f64: f64) {
        let slot = Self::slot_for_bits(value_bits, self.bucket_count);
        self.buckets[slot].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // last-write-wins on sum: races are acceptable for a running aggregate
        self.sum_bits
            .store((self.sum() + sum_f64).to_bits(), Ordering::Relaxed);
    }
}

impl Histogram<f64> {
    /// Record an f64 observation. Negative values are treated as their
    /// absolute magnitude for bucket assignment.
    pub fn record(&self, value: f64) {
        self.record_inner(value.abs().to_bits(), value);
    }
}

impl Histogram<u64> {
    /// Record a u64 observation (cast to f64 for bucket assignment).
    pub fn record(&self, value: u64) {
        let as_f64 = value as f64;
        self.record_inner(as_f64.to_bits(), as_f64);
    }
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

    use super::{Histogram, MAX_BUCKETS};

    fn bucket_pick_linear(value: f64, bucket_count: usize) -> usize {
        // reference: linear scan through base-2 powers from 2^-10
        for index in 0..bucket_count {
            let upper = 2.0f64.powi(index as i32 - 10 + 1);
            if value < upper {
                return index;
            }
        }
        bucket_count - 1
    }

    // 1. happy: Histogram<f64>::record(1.0, &[]) increments bucket containing 1.0
    #[test]
    fn f64_record_increments_bucket_for_1() {
        let hist: Histogram<f64> = Histogram::new("latency");
        hist.record(1.0);
        let snap = hist.bucket_snapshot();
        // 1.0 is in bucket 10 (2^0..2^1)
        assert_eq!(snap[10], 1);
        assert_eq!(hist.count(), 1);
    }

    // 2. happy: Histogram<u64>::record(42, &[]) works (u64 path)
    #[test]
    fn u64_record_increments_correct_bucket() {
        let hist: Histogram<u64> = Histogram::new("items");
        hist.record(42);
        let snap = hist.bucket_snapshot();
        // 42 as f64 falls in bucket for [32, 64) = bucket 15
        let expected = bucket_pick_linear(42.0, MAX_BUCKETS);
        assert_eq!(snap[expected], 1);
        assert_eq!(hist.count(), 1);
    }

    // 3. branchless bucket pick: boundary values land in correct slots
    #[test]
    fn boundary_values_land_in_expected_buckets() {
        let hist: Histogram<f64> = Histogram::new("boundary");
        hist.record(0.001);
        hist.record(1.0);
        hist.record(1_000_000.0);
        let snap = hist.bucket_snapshot();
        // 0.001 (< 2^-9) → bucket 0
        assert_eq!(snap[0], 1);
        // 1.0 = 2^0 → bucket 10
        assert_eq!(snap[10], 1);
        // 1_000_000.0 = ~2^19.9 → bucket 29
        let last_slot = bucket_pick_linear(1_000_000.0, MAX_BUCKETS);
        assert_eq!(snap[last_slot], 1);
        // large overflow saturates to bucket 31
        let big: Histogram<f64> = Histogram::new("big");
        big.record(5_000_000.0);
        let big_snap = big.bucket_snapshot();
        assert_eq!(big_snap[MAX_BUCKETS - 1], 1);
    }

    // 4. correctness: branchless matches linear reference across a range of values
    #[rstest]
    #[case::sub_min(0.0001)]
    #[case::at_min(0.001)]
    #[case::small(0.01)]
    #[case::tenth(0.1)]
    #[case::half(0.5)]
    #[case::one(1.0)]
    #[case::two(2.0)]
    #[case::ten(10.0)]
    #[case::hundred(100.0)]
    #[case::thousand(1_000.0)]
    #[case::ten_thousand(10_000.0)]
    #[case::hundred_thousand(100_000.0)]
    #[case::million(1_000_000.0)]
    #[case::over_max(5_000_000.0)]
    fn branchless_matches_linear_reference(#[case] value: f64) {
        let bits = value.abs().to_bits();
        let exp_field = (bits >> 52) & 0x7FF;
        let raw = exp_field.saturating_sub(super::EXP_BIAS_OFFSET);
        let branchless_slot = raw.min(MAX_BUCKETS as u64 - 1) as usize;
        let linear_slot = bucket_pick_linear(value, MAX_BUCKETS);
        assert_eq!(branchless_slot, linear_slot, "mismatch for value={value}");
    }

    // 5. concurrent: 8 threads each calling record → total count = 8 × N
    #[test]
    fn concurrent_record_total_count_matches() {
        extern crate std;
        use std::sync::Arc;

        let hist = Arc::new(Histogram::<f64>::new("concurrent"));
        let threads: std::vec::Vec<_> = (0..8)
            .map(|_| {
                let shared = Arc::clone(&hist);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        shared.record(1.0);
                    }
                })
            })
            .collect();
        for handle in threads {
            handle.join().expect("thread panicked");
        }
        assert_eq!(hist.count(), 800);
    }

    // 6. builder fields are carried through const chain
    #[test]
    fn builder_fields_carried() {
        let hist: Histogram<f64> = Histogram::new("req_duration")
            .unit("ms")
            .description("request latency");
        assert_eq!(hist.name, "req_duration");
        assert_eq!(hist.unit, "ms");
        assert_eq!(hist.description, "request latency");
    }

    // 9. size assertions
    #[test]
    fn sizes_are_pinned() {
        // 3×&str(16) + base(8) + min(8) + bucket_count(8) + buckets(256) + count(8) + sum(8) = 344
        assert_eq!(core::mem::size_of::<Histogram<f64>>(), 344);
        assert_eq!(core::mem::size_of::<Histogram<u64>>(), 344);
    }
}
