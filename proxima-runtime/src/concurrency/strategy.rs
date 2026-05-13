//! [`Strategy`] — the resolved runtime controller: a signal, an objective, a live
//! law, a gate, and bounds. Impls [`ConcurrencyStrategy`], the whole-controller
//! trait. [`Concurrency`] is the top knob: `Fixed(n)` (the back-compat fast path)
//! or `Adaptive(Strategy)`. [`Preset`] bundles `(signal, objective)` into the
//! named strategies.

use core::time::Duration;

use alloc::boxed::Box;

use super::law::{ControlLaw, LawKind};
use super::{
    Bounds, Gate, LawStep, Objective, Sample, Signal, SignalClass, SignalKind, duration_millis,
};

/// The whole-controller escape hatch: feed a window's [`Sample`], get the next
/// concurrency level. A foreign impl replaces signal × objective × law wholesale
/// for a controller that doesn't fit the lever model.
pub trait ConcurrencyStrategy: Send {
    fn next(&mut self, sample: Sample) -> usize;
}

/// A resolved adaptive controller. Reads only the field its `signal` names from
/// each sample, drives it toward `objective` via `law`, accepts the step through
/// the CoV `gate`, and clamps to `bounds`.
pub struct Strategy {
    signal: Signal,
    objective: Objective,
    law: Box<dyn ControlLaw>,
    gate: Gate,
    bounds: Bounds,
    /// The builtin law name for the descriptor; `None` for a foreign law.
    recorded_law_kind: Option<LawKind>,
    /// Last window's signal value — seeded to the first reading so the opening
    /// step sees `delta == 0` (the gate holds until a real change appears).
    prev_signal: Option<f64>,
}

impl core::fmt::Debug for Strategy {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Strategy")
            .field("signal", &self.signal)
            .field("objective", &self.objective)
            .field("law", &self.law_kind())
            .field("gate", &self.gate)
            .field("bounds", &self.bounds)
            .finish()
    }
}

impl Strategy {
    /// Assemble from resolved parts. `law_kind` records the builtin name for the
    /// descriptor (`None` for a foreign law). Validates coherence + bounds.
    pub fn new(
        signal: Signal,
        objective: Objective,
        law: Box<dyn ControlLaw>,
        law_kind: Option<LawKind>,
        gate: Gate,
        bounds: Bounds,
    ) -> Result<Self, &'static str> {
        objective.coherent_with(signal.class())?;
        bounds.validate()?;
        Ok(Self {
            signal,
            objective,
            law,
            gate,
            bounds,
            recorded_law_kind: law_kind,
            prev_signal: None,
        })
    }

    /// The configured window cadence.
    #[must_use]
    pub fn window(&self) -> Duration {
        self.gate.window
    }

    /// The seed concurrency.
    #[must_use]
    pub fn start(&self) -> usize {
        self.bounds.start
    }

    #[must_use]
    fn law_kind(&self) -> Option<LawKind> {
        self.recorded_law_kind
    }

    /// A `Copy + PartialEq` summary for parity comparison — the builtin-only
    /// view two construction paths must agree on (a foreign law/signal collapses
    /// to its class marker).
    #[must_use]
    pub fn descriptor(&self) -> StrategyDescriptor {
        StrategyDescriptor {
            signal: self.signal.class(),
            objective: self.objective,
            law: self.recorded_law_kind,
            gate: self.gate,
            bounds: self.bounds,
        }
    }
}

impl ConcurrencyStrategy for Strategy {
    fn next(&mut self, sample: Sample) -> usize {
        let signal = self.signal.read(&sample);
        let prev = self.prev_signal.unwrap_or(signal);
        let step = LawStep {
            current: sample.concurrency,
            signal,
            prev_signal: prev,
            objective: self.objective,
            cov: sample.cov,
            coefficient_of_variation_threshold: self.gate.coefficient_of_variation_threshold,
            min: self.bounds.min,
            max: self.bounds.max,
        };
        let next = self.bounds.clamp(self.law.step(step));
        self.prev_signal = Some(signal);
        next
    }
}

/// The builtin-only descriptor of a strategy — what config and fluent paths must
/// agree on for P4 parity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyDescriptor {
    pub signal: SignalClass,
    pub objective: Objective,
    pub law: Option<LawKind>,
    pub gate: Gate,
    pub bounds: Bounds,
}

/// Named strategies = `(signal, objective)` bundles. The parameterised ones carry
/// their objective value. `fixed(n)` is canonically [`Concurrency::Fixed`]; it
/// appears here only to round out the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Preset {
    /// `hold(n)` — no signal. Prefer [`Concurrency::fixed`].
    Fixed(usize),
    /// throughput / maximize — a load-gen's intent.
    HillClimb,
    /// latency_gradient / knee / multiplicative — Vegas/Netflix, latency-safe.
    Gradient,
    /// latency_p99 / target(T_ms) — hold a p99 SLO.
    LatencyTarget(Duration),
    /// utilization / ceiling(U) — the low-CoV/predictable dual of hillclimb.
    Headroom(f64),
}

impl Preset {
    /// The `(signal, objective)` this preset bundles. `Fixed` returns `None`
    /// (it is not an adaptive strategy).
    #[must_use]
    pub fn levers(self) -> Option<(SignalKind, Objective)> {
        match self {
            Self::Fixed(_) => None,
            Self::HillClimb => Some((SignalKind::Throughput, Objective::Maximize)),
            Self::Gradient => Some((SignalKind::LatencyGradient, Objective::Knee)),
            Self::LatencyTarget(target) => Some((
                SignalKind::LatencyP99,
                Objective::Target(duration_millis(target)),
            )),
            Self::Headroom(ceiling) => Some((SignalKind::Utilization, Objective::Ceiling(ceiling))),
        }
    }

    /// Parse a config token. Value-carrying presets take their value from the
    /// settings, not the token, so this only resolves the name (with defaults).
    pub fn parse(token: &str) -> Result<Self, &'static str> {
        match token.trim().to_ascii_lowercase().as_str() {
            "hillclimb" | "hill_climb" => Ok(Self::HillClimb),
            "gradient" => Ok(Self::Gradient),
            "latency_target" | "latency" => Ok(Self::LatencyTarget(Duration::from_millis(5))),
            "headroom" => Ok(Self::Headroom(0.85)),
            "fixed" => Ok(Self::Fixed(25)),
            _ => Err("preset must be fixed|hillclimb|gradient|latency_target|headroom"),
        }
    }
}

/// The per-core concurrency knob — one primitive for client (rekt) and server
/// (a proxima per-core handler limit). `Fixed(n)` keeps the old fixed-cap
/// behaviour exactly; `Adaptive` drives the controller.
pub enum Concurrency {
    /// A fixed in-flight cap — the back-compat fast path (no controller).
    Fixed(usize),
    /// An adaptive controller seeking the crest.
    Adaptive(Strategy),
}

impl core::fmt::Debug for Concurrency {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fixed(n) => write!(formatter, "Concurrency::Fixed({n})"),
            Self::Adaptive(strategy) => {
                write!(formatter, "Concurrency::Adaptive({strategy:?})")
            }
        }
    }
}

impl Concurrency {
    /// A fixed cap.
    #[must_use]
    pub fn fixed(n: usize) -> Self {
        Self::Fixed(n)
    }

    /// Build the adaptive controller for a named preset with default gate/bounds.
    /// `Fixed` resolves to [`Concurrency::Fixed`].
    pub fn from_preset(preset: Preset) -> Result<Self, &'static str> {
        Self::builder().preset(preset).build()
    }

    /// A fresh fluent builder.
    #[must_use]
    pub fn builder() -> super::ConcurrencyBuilder {
        super::ConcurrencyBuilder::new()
    }

    /// The library-level default for a bare `adaptive` request: **gradient** —
    /// latency-aware, safe for a real server. (rekt overrides this to hillclimb
    /// at its config layer; see [`ConcurrencySettings`](super::ConcurrencySettings).)
    pub fn adaptive() -> Result<Self, &'static str> {
        Self::from_preset(Preset::Gradient)
    }

    /// The seed concurrency: `n` for `Fixed`, the strategy's `start` for adaptive.
    #[must_use]
    pub fn initial(&self) -> usize {
        match self {
            Self::Fixed(n) => *n,
            Self::Adaptive(strategy) => strategy.start(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn preset_levers_bundle_signal_and_objective() {
        assert_eq!(
            Preset::HillClimb.levers(),
            Some((SignalKind::Throughput, Objective::Maximize))
        );
        assert_eq!(
            Preset::Gradient.levers(),
            Some((SignalKind::LatencyGradient, Objective::Knee))
        );
        assert!(Preset::Fixed(4).levers().is_none());
    }

    #[test]
    fn from_preset_fixed_is_fixed() {
        let concurrency = Concurrency::from_preset(Preset::Fixed(25)).unwrap();
        assert!(matches!(concurrency, Concurrency::Fixed(25)));
        assert_eq!(concurrency.initial(), 25);
    }

    #[test]
    fn adaptive_default_is_gradient() {
        let concurrency = Concurrency::adaptive().unwrap();
        let Concurrency::Adaptive(strategy) = concurrency else {
            panic!("expected adaptive");
        };
        let descriptor = strategy.descriptor();
        assert_eq!(
            descriptor.signal,
            SignalClass::Builtin(SignalKind::LatencyGradient)
        );
        assert_eq!(descriptor.objective, Objective::Knee);
        assert_eq!(descriptor.law, Some(LawKind::Multiplicative));
    }

    #[test]
    fn strategy_next_climbs_a_monotonic_workload() {
        // throughput strictly rises with concurrency (no crest) → a maximizing
        // strategy should walk up, not stall.
        let mut concurrency = Concurrency::from_preset(Preset::HillClimb).unwrap();
        let Concurrency::Adaptive(ref mut strategy) = concurrency else {
            panic!("adaptive");
        };
        let start = strategy.start();
        let mut current = start;
        for _ in 0..50 {
            let mut sample = Sample::seed(current);
            sample.cov = 0.0;
            sample.throughput = current as f64; // more in-flight = more throughput
            current = strategy.next(sample);
        }
        assert!(
            current > start,
            "monotonic gain → climbed from {start} to {current}"
        );
    }
}
