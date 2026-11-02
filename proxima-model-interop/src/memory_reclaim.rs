//! Idle and pressure policy for mapped regions (weights, expert slabs, cold
//! KV): a region idle for `idle_secs` or the least recently used one while
//! system memory is at `pressure_milli` thousandths is released. The ledger
//! owns only timestamps and pin counts; the caller's `release` closure does
//! the `madvise`/drop, and time and pressure arrive as closures so tests
//! drive both.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_telemetry::debug;
use serde::{Deserialize, Serialize};

use crate::InteropError;

const MEMINFO_PATH: &str = "/proc/meminfo";

static EPOCH: OnceLock<Instant> = OnceLock::new();

#[must_use]
pub fn monotonic_now() -> Duration {
    EPOCH.get_or_init(Instant::now).elapsed()
}

pub fn release_mapping(mapping: &memmap2::Mmap) -> Result<(), InteropError> {
    // SAFETY: the map is read-only and file-backed, so discarding its pages loses no data and the next read refaults.
    unsafe { mapping.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) }.map_err(InteropError::SidecarIo)
}

#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_RECLAIM")]
#[builder(derive(Clone, Debug))]
pub struct ReclaimSettings {
    /// Used-memory fraction in thousandths at which the least recently used regions are released.
    #[setting(default = 900)]
    #[builder(default = 900)]
    pub pressure_milli: u32,
    /// Seconds a region may go unused before it is released.
    #[setting(default = 300)]
    #[builder(default = 300)]
    pub idle_secs: u64,
    /// Directory that holds demoted prompt-cache entries, e.g. `/mnt/data3/kv-cold`; unset keeps the cold tier off.
    pub cold_dir: Option<String>,
    /// Most cold entries the directory keeps.
    #[setting(default = 64)]
    #[builder(default = 64)]
    pub cold_max_entries: u32,
    /// Most bytes the cold entries may hold on disk.
    #[setting(default = 17179869184)]
    #[builder(default = 17179869184)]
    pub cold_byte_budget: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReclaimReason {
    Idle,
    Pressure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Released {
    pub region: u64,
    pub bytes: u64,
    pub reason: ReclaimReason,
}

struct Region {
    bytes: u64,
    last_used: Duration,
    pins: u32,
    resident: bool,
}

#[derive(Default)]
pub struct RegionLedger {
    regions: BTreeMap<u64, Region>,
}

impl RegionLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region_id: u64, bytes: u64, now: Duration) {
        let region = Region {
            bytes,
            last_used: now,
            pins: 0,
            resident: true,
        };
        self.regions.insert(region_id, region);
    }

    /// Returns whether the region had been released, so the caller reloads it.
    pub fn touch(&mut self, region: u64, now: Duration) -> bool {
        self.regions.get_mut(&region).is_some_and(|region| {
            region.last_used = now;
            !std::mem::replace(&mut region.resident, true)
        })
    }

    pub fn pin(&mut self, region: u64, now: Duration) {
        self.touch(region, now);
        if let Some(region) = self.regions.get_mut(&region) {
            region.pins += 1;
        }
    }

    pub fn unpin(&mut self, region: u64, now: Duration) {
        if let Some(region) = self.regions.get_mut(&region) {
            region.pins = region.pins.saturating_sub(1);
            region.last_used = now;
        }
    }

    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.regions
            .values()
            .filter(|region| region.resident)
            .map(|region| region.bytes)
            .sum()
    }

    pub fn forget(&mut self, region: u64) {
        self.regions.remove(&region);
    }

    pub fn reclaim(
        &mut self,
        now: Duration,
        idle_secs: u64,
        pressure_milli: u32,
        mut used_fraction: impl FnMut() -> f64,
        mut release: impl FnMut(u64, u64) -> Result<(), InteropError>,
    ) -> Result<Vec<Released>, InteropError> {
        let mut released = self.release_idle(now, idle_secs, &mut release)?;
        let threshold = f64::from(pressure_milli) / 1000.0;
        loop {
            let used = used_fraction();
            if used < threshold {
                return Ok(released);
            }
            let Some(name) = self.least_recently_used() else {
                return Ok(released);
            };
            debug!(region = name, reason = "pressure", used_fraction = used, "memory pressure at threshold, releasing least recently used region");
            released.push(self.release_region(name, ReclaimReason::Pressure, &mut release)?);
        }
    }

    fn release_idle(
        &mut self,
        now: Duration,
        idle_secs: u64,
        release: &mut impl FnMut(u64, u64) -> Result<(), InteropError>,
    ) -> Result<Vec<Released>, InteropError> {
        let idle_limit = Duration::from_secs(idle_secs);
        let idle_names: Vec<u64> = self
            .regions
            .iter()
            .filter(|(_, region)| {
                region.resident && region.pins == 0 && now - region.last_used >= idle_limit
            })
            .map(|(name, _)| *name)
            .collect();
        idle_names
            .into_iter()
            .map(|name| {
                let region = &self.regions[&name];
                debug!(region = name, bytes = region.bytes, reason = "idle", idle_secs = (now - region.last_used).as_secs(), "region unused past the idle limit, releasing it");
                self.release_region(name, ReclaimReason::Idle, release)
            })
            .collect()
    }

    fn least_recently_used(&self) -> Option<u64> {
        self.regions
            .iter()
            .filter(|(_, region)| region.resident && region.pins == 0)
            .min_by_key(|(_, region)| region.last_used)
            .map(|(name, _)| *name)
    }

    fn release_region(
        &mut self,
        name: u64,
        reason: ReclaimReason,
        release: &mut impl FnMut(u64, u64) -> Result<(), InteropError>,
    ) -> Result<Released, InteropError> {
        let region = &mut self.regions.get_mut(&name).ok_or(InteropError::BlockFileMalformed {
            reason: "released region is not registered",
        })?;
        release(name, region.bytes)?;
        region.resident = false;
        Ok(Released {
            region: name,
            bytes: region.bytes,
            reason,
        })
    }
}

#[must_use]
pub fn used_fraction_of_meminfo(text: &str) -> Option<f64> {
    let kib = |key: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(key)?.trim().strip_suffix("kB")?.trim().parse::<f64>().ok())
    };
    let total = kib("MemTotal:")?;
    let available = kib("MemAvailable:")?;
    (total > 0.0).then(|| 1.0 - available / total)
}

pub fn read_used_fraction() -> Result<f64, InteropError> {
    let text = std::fs::read_to_string(MEMINFO_PATH).map_err(|source| InteropError::BlockFileIo {
        path: Path::new(MEMINFO_PATH).to_path_buf(),
        source,
    })?;
    used_fraction_of_meminfo(&text).ok_or(InteropError::BlockFileMalformed {
        reason: "meminfo lacks MemTotal or MemAvailable",
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn settings() -> ReclaimSettings {
        ReclaimSettings::builder().build()
    }

    fn at(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    fn names(released: &[Released]) -> Vec<u64> {
        released.iter().map(|entry| entry.region).collect()
    }

    #[test]
    fn defaults_are_ninety_percent_and_five_minutes_with_no_cold_dir() {
        let defaults = settings();
        assert_eq!((defaults.pressure_milli, defaults.idle_secs), (900, 300));
        assert_eq!(defaults.cold_dir, None);
    }

    #[test]
    fn idle_region_is_evicted_at_300_seconds_and_not_at_299() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 100, at(0));
        let early = ledger.reclaim(at(299), 300, 900, || 0.1, |_, _| Ok(())).expect("reclaim at 299s");
        let on_time = ledger.reclaim(at(300), 300, 900, || 0.1, |_, _| Ok(())).expect("reclaim at 300s");
        assert!(early.is_empty());
        assert_eq!(on_time, vec![Released { region: 1, bytes: 100, reason: ReclaimReason::Idle }]);
    }

    #[test]
    fn recently_touched_region_survives_idle_pass() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 10, at(0));
        ledger.touch(1, at(250));
        let released = ledger.reclaim(at(400), 300, 900, || 0.1, |_, _| Ok(())).expect("reclaim");
        assert!(released.is_empty());
        assert_eq!(ledger.resident_bytes(), 10);
    }

    #[test]
    fn pressure_evicts_lru_first_until_under_threshold() {
        let mut ledger = RegionLedger::new();
        [1_u64, 2, 3].into_iter().for_each(|region| ledger.register(region, 1, at(region * 10)));
        let used = std::cell::Cell::new(0.91);
        let log = RefCell::new(Vec::new());
        let released = ledger
            .reclaim(
                at(40),
                300,
                900,
                || used.get(),
                |region, bytes| {
                    log.borrow_mut().push(region);
                    used.set(used.get() - 0.02 * bytes as f64);
                    Ok(())
                },
            )
            .expect("reclaim under pressure");
        assert_eq!(names(&released), [1]);
        assert!(released.iter().all(|entry| entry.reason == ReclaimReason::Pressure));
        assert_eq!(*log.borrow(), [1]);
    }

    #[test]
    fn pressure_keeps_releasing_while_still_at_threshold() {
        let mut ledger = RegionLedger::new();
        [1_u64, 2].into_iter().for_each(|region| ledger.register(region, 1, at(region)));
        let released = ledger.reclaim(at(5), 300, 900, || 0.95, |_, _| Ok(())).expect("reclaim");
        assert_eq!(names(&released), [1, 2]);
    }

    #[test]
    fn pinned_region_is_never_released_even_idle_or_under_pressure() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 5, at(0));
        ledger.pin(1, at(0));
        let released = ledger.reclaim(at(10_000), 300, 900, || 0.99, |_, _| Ok(())).expect("reclaim");
        ledger.unpin(1, at(10_000));
        let after_unpin = ledger.reclaim(at(10_001), 300, 900, || 0.1, |_, _| Ok(())).expect("reclaim after unpin");
        assert!(released.is_empty());
        assert!(after_unpin.is_empty());
    }

    #[test]
    fn touching_a_released_region_reports_reload_and_makes_it_resident() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 7, at(0));
        ledger.reclaim(at(300), 300, 900, || 0.1, |_, _| Ok(())).expect("reclaim");
        assert_eq!(ledger.resident_bytes(), 0);
        assert!(ledger.touch(1, at(301)));
        assert!(!ledger.touch(1, at(302)));
        assert_eq!(ledger.resident_bytes(), 7);
    }

    #[test]
    fn forgotten_region_is_never_offered_to_release() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 7, at(0));
        ledger.forget(1);
        let released = ledger.reclaim(at(999), 300, 900, || 0.99, |_, _| Ok(())).expect("reclaim");
        assert!(released.is_empty());
    }

    #[test]
    fn release_failure_propagates_and_keeps_region_resident() {
        let mut ledger = RegionLedger::new();
        ledger.register(1, 7, at(0));
        let failed = ledger.reclaim(
            at(300),
            300,
                900,
            || 0.1,
            |_, _| Err(InteropError::BlockFileMalformed { reason: "madvise refused" }),
        );
        assert!(failed.is_err());
        assert_eq!(ledger.resident_bytes(), 7);
    }

    #[test]
    fn meminfo_used_fraction_reads_available_over_total() {
        let text = "MemTotal:       1000 kB\nMemFree:  10 kB\nMemAvailable:    90 kB\n";
        assert_eq!(used_fraction_of_meminfo(text), Some(0.91));
        assert_eq!(used_fraction_of_meminfo("MemTotal: 5 kB\n"), None);
    }

    #[test]
    fn live_meminfo_is_a_fraction() {
        let used = read_used_fraction().expect("proc meminfo readable on linux");
        assert!((0.0..=1.0).contains(&used));
    }

    #[test]
    fn released_mapping_refaults_identical_bytes_from_the_file() {
        let mut file = tempfile::tempfile().expect("temp file");
        let payload: Vec<u8> = (0..8192_u32).map(|value| (value % 251) as u8).collect();
        std::io::Write::write_all(&mut file, &payload).expect("write payload");
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("map payload");
        assert_eq!(&mapping[..], &payload[..]);
        release_mapping(&mapping).expect("release pages");
        assert_eq!(&mapping[..], &payload[..]);
    }

    #[test]
    fn monotonic_now_never_goes_backwards() {
        let first = monotonic_now();
        assert!(monotonic_now() >= first);
    }
}
