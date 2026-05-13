#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! phase-tagged tail histograms: warmup / steady / spike / spindown.
//! a single histogram per arm hides which phase a p999 came from. this
//! module records samples with their per-call iteration index, then post-
//! classifies into four histograms using percentage-based warmup/spindown
//! cuts and magnitude-based spike detection.
//!
//! TODO(follow-up): bench_compat_libraries.rs and bench_fairness_imbalanced.rs
//! need phased tail wiring — deferred to Wave 3C pass.

#![allow(dead_code)]

use hdrhistogram::Histogram;

pub const DEFAULT_WARMUP_PCT: u32 = 10;
pub const DEFAULT_SPINDOWN_PCT: u32 = 10;
pub const DEFAULT_SPIKE_K: f64 = 5.0;

pub struct PhaseConfig {
    pub warmup_pct: u32,
    pub spindown_pct: u32,
    pub spike_k: f64,
}

impl PhaseConfig {
    pub fn from_env() -> Self {
        let warmup_pct = read_env_u32("HDR_WARMUP_PCT", DEFAULT_WARMUP_PCT);
        let spindown_pct = read_env_u32("HDR_SPINDOWN_PCT", DEFAULT_SPINDOWN_PCT);
        let spike_k = read_env_f64("HDR_SPIKE_K", DEFAULT_SPIKE_K);
        Self {
            warmup_pct,
            spindown_pct,
            spike_k,
        }
    }
}

fn read_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn read_env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(default)
}

fn fresh_hist() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 1_000_000_000, 3).expect("hdr bounds")
}

/// four-phase histogram quartet — warmup, steady, spike, spindown.
pub struct HdrQuartet {
    pub warmup: Histogram<u64>,
    pub steady: Histogram<u64>,
    pub spike: Histogram<u64>,
    pub spindown: Histogram<u64>,
    samples: Vec<(u64, u64)>,
    cfg: PhaseConfig,
}

impl HdrQuartet {
    pub fn new() -> Self {
        Self {
            warmup: fresh_hist(),
            steady: fresh_hist(),
            spike: fresh_hist(),
            spindown: fresh_hist(),
            samples: Vec::with_capacity(1024),
            cfg: PhaseConfig::from_env(),
        }
    }

    pub fn record(&mut self, iter_idx: u64, latency_ns: u64) {
        self.samples.push((iter_idx, latency_ns.max(1)));
    }

    pub fn finalize(&mut self, total_iters_in_call: u64) {
        if total_iters_in_call == 0 {
            self.samples.clear();
            return;
        }

        let warmup_cutoff = total_iters_in_call * u64::from(self.cfg.warmup_pct) / 100;
        let spindown_cutoff =
            total_iters_in_call - total_iters_in_call * u64::from(self.cfg.spindown_pct) / 100;

        // collect middle latencies to find median for spike threshold
        let middle: Vec<u64> = self
            .samples
            .iter()
            .filter(|(idx, _)| *idx >= warmup_cutoff && *idx < spindown_cutoff)
            .map(|(_, latency)| *latency)
            .collect();

        let spike_threshold = if middle.is_empty() {
            u64::MAX
        } else {
            let median = percentile_of_sorted(&middle, 0.5);
            (median as f64 * self.cfg.spike_k) as u64
        };

        for (idx, latency) in self.samples.drain(..) {
            if idx < warmup_cutoff {
                let _ = self.warmup.record(latency);
            } else if idx >= spindown_cutoff {
                let _ = self.spindown.record(latency);
            } else if latency > spike_threshold {
                let _ = self.spike.record(latency);
            } else {
                let _ = self.steady.record(latency);
            }
        }
    }

    /// print four stdout lines — one per phase. format is stable for bash parsing:
    ///   arm=<name> phase=<phase> p50=Xns p90=Yns p99=Zns p999=Wns max=Mns count=K
    pub fn report(&self, arm_name: &str) {
        print_phase_line(arm_name, "warmup", &self.warmup);
        print_phase_line(arm_name, "steady", &self.steady);
        print_phase_line(arm_name, "spike", &self.spike);
        print_phase_line(arm_name, "spindown", &self.spindown);
    }
}

fn print_phase_line(arm: &str, phase: &str, hist: &Histogram<u64>) {
    let count = hist.len();
    if count == 0 {
        println!("arm={arm} phase={phase} p50=0ns p90=0ns p99=0ns p999=0ns max=0ns count=0");
        return;
    }
    let p50 = hist.value_at_quantile(0.50);
    let p90 = hist.value_at_quantile(0.90);
    let p99 = hist.value_at_quantile(0.99);
    let p999 = hist.value_at_quantile(0.999);
    let max = hist.max();
    println!(
        "arm={arm} phase={phase} p50={p50}ns p90={p90}ns p99={p99}ns p999={p999}ns max={max}ns count={count}"
    );
}

fn percentile_of_sorted(values: &[u64], pct: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx]
}
