use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::time::Duration;

use proxima_tensor::Lowered;

use crate::backend::Plan;

/// How much work a door has done, counted since it was built.
///
/// A repeated pin set adds nothing to `lowerings` or `plans`; weights are
/// decoded once per door, so `decodes` stays at 1 however many pin sets
/// arrive. The same eviction is also emitted as a structured telemetry event
/// when it happens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DoorStats {
    /// Times the model bytes were parsed, once per pin set that missed.
    pub parses: u64,
    /// Times initializers were decoded to f32.
    pub decodes: u64,
    /// f32 bytes those decodes produced, e.g. `133_000_000`.
    pub decoded_bytes: u64,
    /// Times a graph was lowered at a pin set.
    pub lowerings: u64,
    /// Times a lowered graph was planned.
    pub plans: u64,
    /// Pin sets dropped to stay within `max_pin_sets`.
    pub evictions: u64,
}

struct Entry {
    pins: BTreeMap<String, u64>,
    lowered: Lowered,
    plan: Plan,
}

/// Plans kept per pin set.
///
/// Entries are ordered by last use: a call takes its entry out, runs it, and
/// stores it at the back, so the front is the pin set unused for longest.
pub struct PlanCache {
    capacity: usize,
    entries: VecDeque<Entry>,
    stats: DoorStats,
}

impl PlanCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
            stats: DoorStats::default(),
        }
    }

    pub const fn stats(&self) -> DoorStats {
        self.stats
    }

    pub fn take(&mut self, pins: &BTreeMap<String, u64>) -> Option<(Lowered, Plan)> {
        let index = self.entries.iter().position(|entry| &entry.pins == pins)?;
        let entry = self.entries.remove(index)?;
        Some((entry.lowered, entry.plan))
    }

    pub fn store(&mut self, pins: BTreeMap<String, u64>, lowered: Lowered, plan: Plan) {
        if self.entries.len() >= self.capacity {
            self.evict_oldest();
        }
        self.entries.push_back(Entry {
            pins,
            lowered,
            plan,
        });
    }

    pub fn record_parse(&mut self, elapsed: Duration) {
        self.stats.parses += 1;
        proxima_telemetry::debug!(
            parses = self.stats.parses,
            parse_ms = elapsed.as_secs_f64() * 1e3,
            "door parsed the model for a pin set it had no plan for"
        );
    }

    pub const fn record_lowering(&mut self) {
        self.stats.lowerings += 1;
    }

    pub const fn record_plan(&mut self) {
        self.stats.plans += 1;
    }

    fn evict_oldest(&mut self) {
        let Some(evicted) = self.entries.pop_front() else {
            return;
        };
        self.stats.evictions += 1;
        proxima_telemetry::info!(
            evictions = self.stats.evictions,
            evicted_pins = ?evicted.pins,
            max_pin_sets = self.capacity,
            "door dropped the least recently used pin set to stay within its cap"
        );
    }
}

impl fmt::Debug for PlanCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PlanCache")
            .field("capacity", &self.capacity)
            .field("pin_sets", &self.entries.len())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}
