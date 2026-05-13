//! Adaptive per-core concurrency control: one controller, composable levers.
//!
//! In a closed-loop per-core async workload (rekt's in-flight connections, a
//! proxima server's concurrent-handler limit) throughput-vs-concurrency has a
//! crest — too few in-flight under-utilises the core, too many adds executor /
//! poll-set overhead past the gain. By Little's Law the optimal in-flight
//! `≈ 1 + wait/work`, so the crest is workload-dependent: a fixed
//! connections-per-core is wrong for every workload but the one it was tuned on.
//!
//! Every strategy here is the same shape: measure a [`Signal`], drive it toward
//! an [`Objective`], by a [`law`](crate::concurrency::law), accept the step
//! through a CoV [`Gate`], inside [`Bounds`]. Named strategies
//! ([`Preset`](crate::concurrency::Preset)) are just `(signal, objective)`
//! bundles, mirroring proxima-h1 `ResponseHandling::Discard` = `Drain + Framing`.
//!
//! Three extension tiers — only `signal` + `law` are extensible (objective /
//! gate / bounds are closed, YAGNI):
//! - **tune**: operator sets preset + lever overrides in TOML/env via
//!   [`ConcurrencySettings`] — zero Rust.
//! - **swap a lever**: `.law(MyPid)` ([`ControlLaw`](crate::concurrency::law::ControlLaw))
//!   or `.signal_fn(|s| ..)` — reuse the measure→decide→apply loop, write only
//!   the decision.
//! - **replace**: impl [`ConcurrencyStrategy`](crate::concurrency::ConcurrencyStrategy)
//!   for a whole controller that doesn't fit the lever model.

pub mod builder;
pub mod controller;
pub mod law;
pub mod settings;
pub mod sim;
pub mod strategy;

use core::time::Duration;

use alloc::boxed::Box;

pub use builder::ConcurrencyBuilder;
pub use controller::{ConcurrencyController, WorkerPool};
pub use law::{ControlLaw, Law, LawKind};
pub use settings::{ConcurrencySettings, Window};
pub use strategy::{Concurrency, ConcurrencyStrategy, Preset, Strategy, StrategyDescriptor};

/// The measured signal vector for one control window — the public contract every
/// built-in, preset, and foreign strategy reads. `Copy`, allocation-free, and
/// `no_std`-able (only `core::time::Duration` + `f64`). A configured strategy
/// reads ONLY the field its [`Signal`] names; the rest are populated best-effort
/// by the caller's meter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// The in-flight limit that was in force while this window was measured.
    pub concurrency: usize,
    /// Completions per second over the window.
    pub throughput: f64,
    /// Measured coefficient of variation (stddev/mean, dimensionless) of
    /// throughput across the window's sub-buckets — the workload's run-to-run
    /// jitter. Selectable as the `Cov` control signal; the gate compares moves
    /// against the configured `coefficient_of_variation_threshold`, not this.
    pub cov: f64,
    /// Smallest round-trip seen — the latency floor (`work` with zero `wait`).
    pub rtt_min: Duration,
    /// Median round-trip.
    pub rtt_p50: Duration,
    /// 99th-percentile round-trip.
    pub rtt_p99: Duration,
    /// Core utilisation in `0.0..=1.0` (busy fraction of the window).
    pub util: f64,
}

impl Sample {
    /// A zeroed sample at `concurrency` — the seed before any window has run.
    #[must_use]
    pub fn seed(concurrency: usize) -> Self {
        Self {
            concurrency,
            throughput: 0.0,
            cov: 0.0,
            rtt_min: Duration::ZERO,
            rtt_p50: Duration::ZERO,
            rtt_p99: Duration::ZERO,
            util: 0.0,
        }
    }
}

/// Which scalar a strategy extracts from a [`Sample`]. `Builtin` names one of the
/// measured fields; `Custom` carries a closure so an app can drive the controller
/// off its own state (queue depth, admission backlog, …) without proxima knowing
/// the quantity. The custom closure receives the `Sample` so it can blend
/// measured and external state — it may ignore the argument entirely.
pub enum Signal {
    /// One of the measured [`Sample`] fields, nameable in config.
    Builtin(SignalKind),
    /// An app-supplied scalar. Fluent-only — not config-nameable (a foreign
    /// signal becomes config-nameable via the on-demand registry, not pre-built).
    Custom(Box<dyn Fn(&Sample) -> f64 + Send>),
}

impl core::fmt::Debug for Signal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Builtin(kind) => write!(formatter, "Signal::Builtin({kind:?})"),
            Self::Custom(_) => formatter.write_str("Signal::Custom(<fn>)"),
        }
    }
}

impl Signal {
    /// Extract this signal's scalar from the sample.
    #[must_use]
    pub fn read(&self, sample: &Sample) -> f64 {
        match self {
            Self::Builtin(kind) => kind.read(sample),
            Self::Custom(extract) => extract(sample),
        }
    }

    /// The static class used by coherence checking — distinguishes "no signal"
    /// (fixed/hold) from a builtin (known semantics) from a custom (unknown).
    #[must_use]
    pub fn class(&self) -> SignalClass {
        match self {
            Self::Builtin(kind) => SignalClass::Builtin(*kind),
            Self::Custom(_) => SignalClass::Custom,
        }
    }
}

/// The five measured signals, nameable in config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// Completions per second — what a load-gen maximises.
    Throughput,
    /// p99 round-trip in milliseconds — a latency SLO target.
    LatencyP99,
    /// `rtt_min / rtt_p99` in `(0, 1]` — the Vegas/Netflix queueing gradient.
    /// 1.0 = no queue (raise), small = heavy queue (lower).
    LatencyGradient,
    /// Core busy fraction in `0.0..=1.0` — the headroom signal.
    Utilization,
    /// Throughput coefficient of variation — predictability of the workload.
    Cov,
}

impl SignalKind {
    #[must_use]
    fn read(self, sample: &Sample) -> f64 {
        match self {
            Self::Throughput => sample.throughput,
            Self::LatencyP99 => duration_millis(sample.rtt_p99),
            Self::LatencyGradient => latency_gradient(sample),
            Self::Utilization => sample.util,
            Self::Cov => sample.cov,
        }
    }

    /// True for the latency-derived signals — used by `knee` coherence.
    #[must_use]
    pub fn is_latency(self) -> bool {
        matches!(self, Self::LatencyP99 | Self::LatencyGradient)
    }

    /// True for the signals a `maximize` objective can sensibly climb.
    #[must_use]
    pub fn is_maximizable(self) -> bool {
        matches!(self, Self::Throughput | Self::Utilization)
    }
}

/// Static signal class for coherence checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalClass {
    /// No signal — the fixed / hold case.
    None,
    /// A measured builtin with known semantics.
    Builtin(SignalKind),
    /// An app-supplied closure of unknown semantics.
    Custom,
}

/// What the law drives the signal toward. Closed set (YAGNI). The default law is
/// derived from the objective ([`Objective::default_law`]) and is overridable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Objective {
    /// Climb the signal (hillclimb).
    Maximize,
    /// Hold the signal at `v` (proportional control).
    Target(f64),
    /// Keep the signal at or below `v` (AIMD — grow under, back off over).
    Ceiling(f64),
    /// Sit at the knee of the latency-vs-concurrency curve (multiplicative,
    /// Vegas/Netflix).
    Knee,
    /// Hold concurrency at `n` regardless of signal — the fixed case.
    Hold(usize),
}

impl Objective {
    /// The law the objective implies when none is set explicitly.
    #[must_use]
    pub fn default_law(self) -> LawKind {
        match self {
            Self::Maximize => LawKind::HillClimb,
            Self::Target(_) => LawKind::Proportional,
            Self::Ceiling(_) => LawKind::Aimd,
            Self::Knee => LawKind::Multiplicative,
            Self::Hold(_) => LawKind::HillClimb,
        }
    }

    /// Reject incoherent `(signal, objective)` pairs (gate point 4). The two
    /// task examples — "maximize + fixed target" (maximize with no signal) and
    /// "a latency objective with no latency signal" (knee off a non-latency
    /// signal) — both fail here.
    pub fn coherent_with(self, signal: SignalClass) -> Result<(), &'static str> {
        match self {
            Self::Maximize => match signal {
                SignalClass::Builtin(kind) if kind.is_maximizable() => Ok(()),
                SignalClass::Custom => Ok(()),
                SignalClass::None => Err("maximize objective names no signal to climb"),
                SignalClass::Builtin(_) => Err("maximize needs a throughput/utilization signal"),
            },
            Self::Knee => match signal {
                SignalClass::Builtin(kind) if kind.is_latency() => Ok(()),
                _ => Err("knee objective requires a latency signal (gradient or p99)"),
            },
            Self::Target(value) | Self::Ceiling(value) => {
                if !value.is_finite() || value <= 0.0 {
                    return Err("target/ceiling value must be finite and positive");
                }
                match signal {
                    SignalClass::None => Err("target/ceiling objective names no signal to read"),
                    _ => Ok(()),
                }
            }
            Self::Hold(_) => match signal {
                SignalClass::None => Ok(()),
                _ => Err("hold objective must not name a signal"),
            },
        }
    }
}

/// The significance gate plus the window cadence — the lever every adaptive
/// preset shares.
///
/// `coefficient_of_variation_threshold` is the minimum *relative* move a signal
/// must make before the controller believes it is real rather than noise. The
/// **coefficient of variation** (CoV) is a workload's run-to-run jitter expressed
/// as a fraction of its own mean — `stddev / mean`. Throughput wobbles even at a
/// fixed concurrency; a CoV of `0.05` means it naturally varies ±5%. So the gate
/// accepts a step only when
///
/// ```text
/// |Δsignal| / |signal|  >  coefficient_of_variation_threshold
/// ```
///
/// The value is 1:1 with the quantity it gates: set `0.05` and a change must
/// exceed 5% to count; a 4% wobble is dismissed as noise and the controller
/// holds. Higher = more skeptical (steadier, slower); lower = twitchier. The
/// controller exposes it to every law via [`LawStep::significant`] so no law
/// reimplements the accept criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    /// Minimum relative signal change (a fraction of the signal, e.g. `0.05` =
    /// 5%) that counts as real rather than noise. Same units as the measured
    /// coefficient of variation it gates against.
    pub coefficient_of_variation_threshold: f64,
    /// How long each control window runs before a sample is taken.
    pub window: Duration,
    /// Re-measure this many windows before acting on a step (1 = act every window).
    pub reprobe: u32,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            coefficient_of_variation_threshold: 0.05,
            window: Duration::from_millis(150),
            reprobe: 1,
        }
    }
}

/// The concurrency search bounds: never below `min`, never above `max`, and the
/// `start` the controller seeds at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub min: usize,
    pub max: usize,
    pub start: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min: 1,
            max: 512,
            start: 16,
        }
    }
}

impl Bounds {
    /// `1 ≤ min ≤ start ≤ max`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min < 1 {
            return Err("bounds.min must be >= 1");
        }
        if self.min > self.max {
            return Err("bounds.min must be <= bounds.max");
        }
        if self.start < self.min || self.start > self.max {
            return Err("bounds.start must lie within [min, max]");
        }
        Ok(())
    }

    /// Clamp `n` into `[min, max]`.
    #[must_use]
    pub fn clamp(&self, n: usize) -> usize {
        n.clamp(self.min, self.max)
    }
}

/// The context one [`ControlLaw::step`](crate::concurrency::law::ControlLaw::step)
/// receives — current concurrency, the signal now and last window, the objective,
/// and the gate parameters. Carries the shared accept criterion via
/// [`significant`](Self::significant) / [`margin`](Self::margin) so every law
/// honours the same gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LawStep {
    /// The in-flight limit in force for the window just sampled.
    pub current: usize,
    /// This window's signal value.
    pub signal: f64,
    /// Last window's signal value (equals `signal` on the first step).
    pub prev_signal: f64,
    /// The objective being driven toward.
    pub objective: Objective,
    /// The MEASURED coefficient of variation of throughput this window
    /// (`stddev / mean`). Informational — available to custom laws and as the
    /// `Cov` signal; the builtin gate uses the configured threshold below, not
    /// this measured value.
    pub cov: f64,
    /// The configured significance threshold: minimum relative signal change (a
    /// fraction, `0.05` = 5%) that counts as real. See [`Gate`].
    pub coefficient_of_variation_threshold: f64,
    /// Lower concurrency bound.
    pub min: usize,
    /// Upper concurrency bound.
    pub max: usize,
}

impl LawStep {
    /// The shared gate: did the signal move by more than
    /// `coefficient_of_variation_threshold · |signal|` (i.e. a relative change
    /// bigger than the threshold)? A law that returns `current` when this is
    /// false refuses to chase noise — the discipline every adaptive preset shares.
    #[must_use]
    pub fn significant(&self) -> bool {
        (self.signal - self.prev_signal).abs() > self.margin()
    }

    /// The noise floor `threshold · |signal|` — also the band a threshold law
    /// (AIMD) treats as "at the ceiling" rather than over/under it.
    #[must_use]
    pub fn margin(&self) -> f64 {
        self.coefficient_of_variation_threshold * self.signal.abs()
    }

    /// `signal - prev_signal`.
    #[must_use]
    pub fn delta(&self) -> f64 {
        self.signal - self.prev_signal
    }

    /// Clamp a proposed level into `[min, max]`.
    #[must_use]
    pub fn clamp(&self, n: usize) -> usize {
        n.clamp(self.min, self.max)
    }

    /// Apply a signed step to `current`, saturating at 0 before the clamp.
    #[must_use]
    pub fn stepped(&self, delta: isize) -> usize {
        let raw = self.current as isize + delta;
        self.clamp(raw.max(0) as usize)
    }
}

/// Milliseconds as `f64` — the unit latency signals report in.
#[must_use]
pub fn duration_millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// `rtt_min / rtt_p99` clamped to `(0, 1]`. Returns 1.0 (no queue) when p99 is
/// not yet measured, so an unprimed controller grows rather than stalls.
#[must_use]
pub fn latency_gradient(sample: &Sample) -> f64 {
    let p99 = sample.rtt_p99.as_secs_f64();
    if p99 <= 0.0 {
        return 1.0;
    }
    (sample.rtt_min.as_secs_f64() / p99).clamp(0.0, 1.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn sample_with(throughput: f64, p99_ms: u64, min_ms: u64, util: f64) -> Sample {
        Sample {
            concurrency: 8,
            throughput,
            cov: 0.05,
            rtt_min: Duration::from_millis(min_ms),
            rtt_p50: Duration::from_millis(min_ms),
            rtt_p99: Duration::from_millis(p99_ms),
            util,
            ..Sample::seed(8)
        }
    }

    #[test]
    fn signal_reads_named_field_only() {
        let sample = sample_with(1_000.0, 10, 2, 0.7);
        assert_eq!(
            Signal::Builtin(SignalKind::Throughput).read(&sample),
            1_000.0
        );
        assert_eq!(Signal::Builtin(SignalKind::LatencyP99).read(&sample), 10.0);
        assert_eq!(Signal::Builtin(SignalKind::Utilization).read(&sample), 0.7);
        // gradient = 2/10 = 0.2
        assert!((Signal::Builtin(SignalKind::LatencyGradient).read(&sample) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn custom_signal_can_ignore_sample() {
        let signal = Signal::Custom(Box::new(|_| 42.0));
        assert_eq!(signal.read(&Sample::seed(1)), 42.0);
        assert_eq!(signal.class(), SignalClass::Custom);
    }

    #[test]
    fn gradient_defaults_to_one_when_unprimed() {
        assert_eq!(latency_gradient(&Sample::seed(4)), 1.0);
    }

    #[test]
    fn coherence_rejects_maximize_without_signal() {
        // "maximize + fixed target": maximize with no signal to climb.
        assert!(
            Objective::Maximize
                .coherent_with(SignalClass::None)
                .is_err()
        );
        assert!(
            Objective::Maximize
                .coherent_with(SignalClass::Builtin(SignalKind::LatencyP99))
                .is_err()
        );
        assert!(
            Objective::Maximize
                .coherent_with(SignalClass::Builtin(SignalKind::Throughput))
                .is_ok()
        );
    }

    #[test]
    fn coherence_rejects_knee_without_latency_signal() {
        assert!(
            Objective::Knee
                .coherent_with(SignalClass::Builtin(SignalKind::Throughput))
                .is_err()
        );
        assert!(
            Objective::Knee
                .coherent_with(SignalClass::Builtin(SignalKind::LatencyGradient))
                .is_ok()
        );
    }

    #[test]
    fn coherence_rejects_hold_with_signal_and_bad_target() {
        assert!(
            Objective::Hold(8)
                .coherent_with(SignalClass::Builtin(SignalKind::Throughput))
                .is_err()
        );
        assert!(Objective::Hold(8).coherent_with(SignalClass::None).is_ok());
        assert!(
            Objective::Target(f64::NAN)
                .coherent_with(SignalClass::Builtin(SignalKind::LatencyP99))
                .is_err()
        );
    }

    #[test]
    fn bounds_validate_and_clamp() {
        assert!(
            Bounds {
                min: 1,
                max: 8,
                start: 4
            }
            .validate()
            .is_ok()
        );
        assert!(
            Bounds {
                min: 8,
                max: 4,
                start: 4
            }
            .validate()
            .is_err()
        );
        assert!(
            Bounds {
                min: 1,
                max: 8,
                start: 16
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            Bounds {
                min: 2,
                max: 8,
                start: 4
            }
            .clamp(100),
            8
        );
        assert_eq!(
            Bounds {
                min: 2,
                max: 8,
                start: 4
            }
            .clamp(0),
            2
        );
    }

    #[test]
    fn law_step_gate_rejects_sub_threshold_change() {
        // threshold 0.10 → a change must exceed 10% of the signal to count.
        let step = LawStep {
            current: 8,
            signal: 101.0,
            prev_signal: 100.0,
            objective: Objective::Maximize,
            cov: 0.05,
            coefficient_of_variation_threshold: 0.10,
            min: 1,
            max: 64,
        };
        assert!(
            !step.significant(),
            "1% change under a 10% threshold is noise"
        );
        let big = LawStep {
            signal: 130.0,
            ..step
        };
        assert!(big.significant(), "30% change clears the 10% threshold");
    }

    #[test]
    fn law_step_stepped_saturates_and_clamps() {
        let step = LawStep {
            current: 1,
            signal: 0.0,
            prev_signal: 0.0,
            objective: Objective::Maximize,
            cov: 0.0,
            coefficient_of_variation_threshold: 0.05,
            min: 1,
            max: 8,
        };
        assert_eq!(step.stepped(-5), 1, "saturates then clamps to min");
        assert_eq!(step.stepped(100), 8, "clamps to max");
    }
}
