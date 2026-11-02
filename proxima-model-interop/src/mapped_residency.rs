use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use memmap2::Mmap;

use crate::InteropError;
use crate::memory_reclaim::{
    ReclaimSettings, RegionLedger, Released, monotonic_now, read_used_fraction, release_mapping,
};

pub const WEIGHTS_REGION: u64 = 0;
pub const EXPERT_SIDECAR_REGION: u64 = 1;
const REGIONS: usize = 2;

#[derive(Default)]
struct RegionState {
    pins: AtomicU32,
    last_use_nanos: AtomicU64,
    released: AtomicBool,
}

/// Ties the weights and expert sidecar mappings to idle and pressure reclaim; pin, unpin and touch are atomics so an evaluation never takes a lock, and only the cold `reclaim` scan builds a `RegionLedger`.
pub struct MappedResidency<'map> {
    weights: OnceLock<&'map Mmap>,
    sidecar: OnceLock<Arc<Mmap>>,
    states: [RegionState; REGIONS],
    settings: ReclaimSettings,
}

/// Holds every region pinned for one evaluation; dropping it unpins on every exit path.
pub struct Pinned<'residency, 'map> {
    residency: &'residency MappedResidency<'map>,
}

impl Drop for Pinned<'_, '_> {
    fn drop(&mut self) {
        self.residency.unpin_all(monotonic_now());
    }
}

fn nanos(now: Duration) -> u64 {
    u64::try_from(now.as_nanos()).unwrap_or(u64::MAX)
}

impl<'map> MappedResidency<'map> {
    #[must_use]
    pub fn new(settings: ReclaimSettings) -> Self {
        Self {
            weights: OnceLock::new(),
            sidecar: OnceLock::new(),
            states: Default::default(),
            settings,
        }
    }

    pub fn register_weights(&self, mapping: &'map Mmap, now: Duration) {
        if self.weights.set(mapping).is_ok() {
            self.stamp(WEIGHTS_REGION, now);
        }
    }

    pub fn register_sidecar(&self, mapping: Arc<Mmap>, now: Duration) {
        if self.sidecar.set(mapping).is_ok() {
            self.stamp(EXPERT_SIDECAR_REGION, now);
        }
    }

    fn state(&self, region: u64) -> Option<&RegionState> {
        self.states.get(usize::try_from(region).ok()?)
    }

    fn mapping(&self, region: u64) -> Option<&Mmap> {
        match region {
            WEIGHTS_REGION => self.weights.get().copied(),
            EXPERT_SIDECAR_REGION => self.sidecar.get().map(Arc::as_ref),
            _ => None,
        }
    }

    fn stamp(&self, region: u64, now: Duration) {
        if let Some(state) = self.state(region) {
            state
                .last_use_nanos
                .fetch_max(nanos(now), Ordering::Relaxed);
        }
    }

    /// Returns whether the region had been released, so the caller knows pages refault.
    pub fn touch(&self, region: u64, now: Duration) -> bool {
        self.stamp(region, now);
        self.state(region)
            .is_some_and(|state| state.released.swap(false, Ordering::AcqRel))
    }

    pub fn pin_all(&self, now: Duration) {
        (0..REGIONS as u64).for_each(|region| {
            self.touch(region, now);
            if let Some(state) = self.state(region) {
                state.pins.fetch_add(1, Ordering::AcqRel);
            }
        });
    }

    pub fn unpin_all(&self, now: Duration) {
        self.states.iter().enumerate().for_each(|(region, state)| {
            let _ = state
                .pins
                .try_update(Ordering::AcqRel, Ordering::Acquire, |pins| {
                    pins.checked_sub(1)
                });
            self.stamp(region as u64, now);
        });
    }

    #[must_use]
    pub fn pin(&self) -> Pinned<'_, 'map> {
        self.pin_all(monotonic_now());
        Pinned { residency: self }
    }

    #[must_use]
    pub fn pins(&self, region: u64) -> u32 {
        self.state(region)
            .map_or(0, |state| state.pins.load(Ordering::Acquire))
    }

    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        (0..REGIONS as u64)
            .filter(|region| {
                self.state(*region)
                    .is_some_and(|state| !state.released.load(Ordering::Acquire))
            })
            .filter_map(|region| self.mapping(region))
            .map(|mapping| mapping.len() as u64)
            .sum()
    }

    fn snapshot(&self) -> RegionLedger {
        let mut ledger = RegionLedger::new();
        (0..REGIONS as u64).for_each(|region| {
            let (Some(mapping), Some(state)) = (self.mapping(region), self.state(region)) else {
                return;
            };
            ledger.restore(
                region,
                mapping.len() as u64,
                Duration::from_nanos(state.last_use_nanos.load(Ordering::Acquire)),
                state.pins.load(Ordering::Acquire),
                !state.released.load(Ordering::Acquire),
            );
        });
        ledger
    }

    fn release(&self, region: u64) -> Result<(), InteropError> {
        let (Some(mapping), Some(state)) = (self.mapping(region), self.state(region)) else {
            return Ok(());
        };
        release_mapping(mapping)?;
        state.released.store(true, Ordering::Release);
        Ok(())
    }

    pub fn reclaim(
        &self,
        now: Duration,
        used_fraction: impl FnMut() -> f64,
    ) -> Result<Vec<Released>, InteropError> {
        self.snapshot().reclaim(
            now,
            self.settings.idle_secs,
            self.settings.pressure_milli,
            used_fraction,
            |region, _bytes| self.release(region),
        )
    }

    pub fn reclaim_now(&self) -> Result<Vec<Released>, InteropError> {
        self.reclaim(monotonic_now(), || read_used_fraction().unwrap_or(0.0))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn at(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    fn mapped_file(payload: &[u8]) -> Mmap {
        let mut file = tempfile::tempfile().expect("temp file");
        file.write_all(payload).expect("write payload");
        unsafe { Mmap::map(&file) }.expect("map payload")
    }

    fn payload() -> Vec<u8> {
        (0..8192_u32).map(|value| (value % 251) as u8).collect()
    }

    fn residency() -> MappedResidency<'static> {
        MappedResidency::new(ReclaimSettings::builder().build())
    }

    #[test]
    fn weights_release_at_300_seconds_idle_and_not_at_299() {
        let mapping = mapped_file(&payload());
        let tracked = residency();
        tracked.register_weights(Box::leak(Box::new(mapping)), at(0));
        let early = tracked.reclaim(at(299), || 0.1).expect("reclaim at 299s");
        let on_time = tracked.reclaim(at(300), || 0.1).expect("reclaim at 300s");
        assert!(early.is_empty());
        assert_eq!(on_time.len(), 1);
        assert_eq!(tracked.resident_bytes(), 0);
    }

    #[test]
    fn weights_pinned_for_an_evaluation_survive_idle_and_pressure() {
        let tracked = residency();
        tracked.register_weights(Box::leak(Box::new(mapped_file(&payload()))), at(0));
        tracked.pin_all(at(0));
        let during = tracked
            .reclaim(at(10_000), || 0.95)
            .expect("reclaim while pinned");
        tracked.unpin_all(at(10_000));
        let after = tracked
            .reclaim(at(10_001), || 0.95)
            .expect("reclaim after unpin");
        assert!(during.is_empty());
        assert_eq!(after.len(), 1);
    }

    #[test]
    fn pin_guard_unpins_when_dropped() {
        let tracked = residency();
        tracked.register_weights(Box::leak(Box::new(mapped_file(&payload()))), at(0));
        {
            let _held = tracked.pin();
            assert_eq!(tracked.pins(WEIGHTS_REGION), 1);
        }
        assert_eq!(tracked.pins(WEIGHTS_REGION), 0);
    }

    #[test]
    fn released_weights_refault_identical_bytes_and_report_reload() {
        let bytes = payload();
        let mapping: &'static Mmap = Box::leak(Box::new(mapped_file(&bytes)));
        let tracked = residency();
        tracked.register_weights(mapping, at(0));
        tracked.reclaim(at(300), || 0.1).expect("reclaim");
        assert!(tracked.touch(WEIGHTS_REGION, at(301)));
        assert_eq!(&mapping[..], &bytes[..]);
        assert_eq!(tracked.resident_bytes(), bytes.len() as u64);
    }

    #[test]
    fn touched_sidecar_survives_while_untouched_weights_release() {
        let tracked = residency();
        tracked.register_weights(Box::leak(Box::new(mapped_file(&payload()))), at(0));
        tracked.register_sidecar(Arc::new(mapped_file(&payload())), at(0));
        tracked.touch(EXPERT_SIDECAR_REGION, at(250));
        let released = tracked.reclaim(at(400), || 0.1).expect("reclaim");
        assert_eq!(
            released
                .iter()
                .map(|entry| entry.region)
                .collect::<Vec<_>>(),
            [WEIGHTS_REGION]
        );
    }
}
