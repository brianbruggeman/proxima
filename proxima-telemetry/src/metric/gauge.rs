use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::tag::{ScalarValue, Tag};

// value-kind tag so the drainer interprets the bits correctly — the three
// setters store raw/bit-cast u64, so the kind is otherwise lost at snapshot.
const KIND_U64: u8 = 0;
const KIND_F64: u8 = 1;
const KIND_I64: u8 = 2;

/// A gauge instrument — last-write-wins, supports set/get.
///
/// Stored as a bit-cast u64 to keep the accumulator lock-free, plus a `kind` tag
/// so the drainer exports the right `ScalarValue`. F64 gauges lose NaN identity
/// but real telemetry never produces NaN. Default kind is f64 (the common gauge).
pub struct Gauge {
    pub name: &'static str,
    pub unit: &'static str,
    pub description: &'static str,
    bits: AtomicU64,
    // last value reported to a drain, so a steady gauge goes quiet (otherwise a
    // never-emptying instrument spins the managed drainer's drain-until-empty loop).
    last_bits: AtomicU64,
    kind: AtomicU8,
}

impl Gauge {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            unit: "",
            description: "",
            bits: AtomicU64::new(0),
            last_bits: AtomicU64::new(0),
            kind: AtomicU8::new(KIND_F64),
        }
    }

    #[must_use]
    pub const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    #[must_use]
    pub const fn description(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    /// Store a u64 observation.
    pub fn set_u64(&self, value: u64, _tags: &[Tag]) {
        self.kind.store(KIND_U64, Ordering::Relaxed);
        self.bits.store(value, Ordering::Relaxed);
    }

    /// Store an f64 observation (bit-cast to u64).
    pub fn set_f64(&self, value: f64, _tags: &[Tag]) {
        self.kind.store(KIND_F64, Ordering::Relaxed);
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Store an i64 observation (bit-cast to u64).
    pub fn set_i64(&self, value: i64, _tags: &[Tag]) {
        self.kind.store(KIND_I64, Ordering::Relaxed);
        self.bits.store(value as u64, Ordering::Relaxed);
    }

    /// Read back the raw bits as u64.
    #[must_use]
    pub fn get_u64(&self) -> u64 {
        self.bits.load(Ordering::Relaxed)
    }

    /// Read back as f64 (bit-cast).
    #[must_use]
    pub fn get_f64(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Read back as i64 (bit-cast).
    #[must_use]
    pub fn get_i64(&self) -> i64 {
        self.bits.load(Ordering::Relaxed) as i64
    }

    /// Snapshot the current value as the typed `ScalarValue` the last setter
    /// implied. Gauges are last-value, so there is no reset. (A snapshot racing a
    /// concurrent `set_*` may briefly read the new kind with old bits or vice
    /// versa — a benign one-window misread the next snapshot corrects.)
    #[must_use]
    pub fn snapshot_value(&self) -> ScalarValue {
        self.bits_to_value(self.bits.load(Ordering::Relaxed))
    }

    /// Drain-time snapshot: the typed value IF it changed since the last drain,
    /// else `None`. Report-on-change keeps a steady gauge from re-publishing every
    /// pass (which would spin the managed drainer). A value set back to its init 0
    /// reads as unchanged — acceptable for v1.
    #[must_use]
    pub fn snapshot_if_changed(&self) -> Option<ScalarValue> {
        let bits = self.bits.load(Ordering::Relaxed);
        if bits == self.last_bits.swap(bits, Ordering::Relaxed) {
            return None;
        }
        Some(self.bits_to_value(bits))
    }

    fn bits_to_value(&self, bits: u64) -> ScalarValue {
        match self.kind.load(Ordering::Relaxed) {
            KIND_U64 => ScalarValue::U64(bits),
            KIND_I64 => ScalarValue::I64(bits as i64),
            _ => ScalarValue::F64(f64::from_bits(bits)),
        }
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

    use super::Gauge;

    #[test]
    fn set_u64_updates_value() {
        let gauge = Gauge::new("cpu");
        gauge.set_u64(77, &[]);
        assert_eq!(gauge.get_u64(), 77);
        gauge.set_u64(99, &[]);
        assert_eq!(gauge.get_u64(), 99);
    }

    #[test]
    fn set_f64_round_trips() {
        let gauge = Gauge::new("temp");
        gauge.set_f64(36.6, &[]);
        assert!((gauge.get_f64() - 36.6).abs() < f64::EPSILON);
    }

    #[test]
    fn set_i64_round_trips_negative() {
        let gauge = Gauge::new("offset");
        gauge.set_i64(-5, &[]);
        assert_eq!(gauge.get_i64(), -5);
    }

    #[test]
    fn size_is_pinned() {
        // name+unit+desc(48) + bits(8) + last_bits(8) + kind(1) -> padded to 72B.
        assert_eq!(core::mem::size_of::<Gauge>(), 72);
    }

    #[test]
    fn snapshot_if_changed_quiets_when_steady() {
        let gauge = Gauge::new("g");
        gauge.set_f64(36.6, &[]);
        assert!(gauge.snapshot_if_changed().is_some(), "first set reports");
        assert!(
            gauge.snapshot_if_changed().is_none(),
            "steady value is quiet"
        );
        gauge.set_f64(40.0, &[]);
        assert!(
            gauge.snapshot_if_changed().is_some(),
            "change reports again"
        );
    }

    #[test]
    fn snapshot_value_tracks_last_setter_kind() {
        use crate::tag::ScalarValue;
        let gauge = Gauge::new("g");
        gauge.set_u64(42, &[]);
        assert!(matches!(gauge.snapshot_value(), ScalarValue::U64(42)));
        gauge.set_i64(-7, &[]);
        assert!(matches!(gauge.snapshot_value(), ScalarValue::I64(-7)));
        gauge.set_f64(3.5, &[]);
        assert!(
            matches!(gauge.snapshot_value(), ScalarValue::F64(v) if (v - 3.5).abs() < f64::EPSILON)
        );
    }
}
