//! [`ConcurrencyBuilder`] — the fluent surface, first-class alongside
//! [`ConcurrencySettings`](super::ConcurrencySettings) (P4). Sugar collapses the
//! builder semantically (`.hillclimb().coefficient_of_variation_threshold(0.05)`) rather than offering
//! bespoke constructors, mirroring proxima-h1 `with_response(preset)` +
//! `with_response_*` per-lever overrides. Hand-rolled (not bon) because the
//! semantic-collapse sugar — preset selectors plus per-lever setters — does not
//! map onto bon's generated field setters; this follows prime's `Builder`.

use core::time::Duration;

use alloc::boxed::Box;

use super::law::{ControlLaw, Law, LawKind};
use super::strategy::{Concurrency, Preset, Strategy};
use super::{Bounds, Gate, Objective, Sample, Signal, SignalKind};

/// Fluent builder for [`Concurrency`]. Set a preset (or explicit signal +
/// objective), then override any lever; `.build()` resolves and coherence-checks.
pub struct ConcurrencyBuilder {
    fixed: Option<usize>,
    preset: Option<Preset>,
    signal: Option<Signal>,
    objective: Option<Objective>,
    law: Option<Law>,
    gate: Gate,
    bounds: Bounds,
}

impl Default for ConcurrencyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ConcurrencyBuilder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ConcurrencyBuilder")
            .field("fixed", &self.fixed)
            .field("preset", &self.preset)
            .field("signal", &self.signal)
            .field("objective", &self.objective)
            .field("law", &self.law)
            .field("gate", &self.gate)
            .field("bounds", &self.bounds)
            .finish()
    }
}

impl ConcurrencyBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fixed: None,
            preset: None,
            signal: None,
            objective: None,
            law: None,
            gate: Gate::default(),
            bounds: Bounds::default(),
        }
    }

    // ── presets (semantic-collapse sugar) ──────────────────────────────────

    /// A fixed in-flight cap — the back-compat fast path.
    #[must_use]
    pub fn fixed(mut self, n: usize) -> Self {
        self.fixed = Some(n);
        self
    }

    /// throughput / maximize — a load-gen's intent.
    #[must_use]
    pub fn hillclimb(mut self) -> Self {
        self.preset = Some(Preset::HillClimb);
        self
    }

    /// latency_gradient / knee / multiplicative — Vegas/Netflix, latency-safe.
    #[must_use]
    pub fn gradient(mut self) -> Self {
        self.preset = Some(Preset::Gradient);
        self
    }

    /// latency_p99 / target(T) — hold a p99 SLO.
    #[must_use]
    pub fn latency_target(mut self, target: Duration) -> Self {
        self.preset = Some(Preset::LatencyTarget(target));
        self
    }

    /// utilization / ceiling(U) — the predictable dual of hillclimb.
    #[must_use]
    pub fn headroom(mut self, ceiling: f64) -> Self {
        self.preset = Some(Preset::Headroom(ceiling));
        self
    }

    /// Apply a [`Preset`] from a typed value (config path).
    #[must_use]
    pub fn preset(mut self, preset: Preset) -> Self {
        match preset {
            Preset::Fixed(n) => self.fixed = Some(n),
            other => self.preset = Some(other),
        }
        self
    }

    // ── lever overrides ────────────────────────────────────────────────────

    /// Override the signal to a named builtin.
    #[must_use]
    pub fn signal(mut self, kind: SignalKind) -> Self {
        self.signal = Some(Signal::Builtin(kind));
        self
    }

    /// Override the signal with an app-supplied closure (swap-a-lever tier).
    #[must_use]
    pub fn signal_fn(mut self, extract: impl Fn(&Sample) -> f64 + Send + 'static) -> Self {
        self.signal = Some(Signal::Custom(Box::new(extract)));
        self
    }

    /// Override the objective directly.
    #[must_use]
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = Some(objective);
        self
    }

    /// Sugar: `objective(Maximize)`.
    #[must_use]
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Sugar: `objective(Target(v))` — `v` in the signal's unit.
    #[must_use]
    pub fn target(self, value: f64) -> Self {
        self.objective(Objective::Target(value))
    }

    /// Sugar: `objective(Ceiling(v))`.
    #[must_use]
    pub fn ceiling(self, value: f64) -> Self {
        self.objective(Objective::Ceiling(value))
    }

    /// Sugar: `objective(Knee)`.
    #[must_use]
    pub fn knee(self) -> Self {
        self.objective(Objective::Knee)
    }

    /// Override the law to a named builtin.
    #[must_use]
    pub fn law_kind(mut self, kind: LawKind) -> Self {
        self.law = Some(Law::Builtin(kind));
        self
    }

    /// Swap in a foreign law (swap-a-lever tier). Fluent-only.
    #[must_use]
    pub fn law(mut self, law: impl ControlLaw + 'static) -> Self {
        self.law = Some(Law::Custom(Box::new(law)));
        self
    }

    // ── gate + bounds ──────────────────────────────────────────────────────

    /// Minimum relative signal move to act on, a fraction (`0.05` = 5%); smaller
    /// moves are treated as noise and held. See [`Gate`].
    #[must_use]
    pub fn coefficient_of_variation_threshold(
        mut self,
        coefficient_of_variation_threshold: f64,
    ) -> Self {
        self.gate.coefficient_of_variation_threshold = coefficient_of_variation_threshold;
        self
    }

    /// The control window cadence.
    #[must_use]
    pub fn window(mut self, window: Duration) -> Self {
        self.gate.window = window;
        self
    }

    /// Re-measure this many windows before acting.
    #[must_use]
    pub fn reprobe(mut self, reprobe: u32) -> Self {
        self.gate.reprobe = reprobe;
        self
    }

    /// Set the full gate (config path).
    #[must_use]
    pub fn gate(mut self, gate: Gate) -> Self {
        self.gate = gate;
        self
    }

    /// Set `[min, max]`.
    #[must_use]
    pub fn bounds(mut self, min: usize, max: usize) -> Self {
        self.bounds.min = min;
        self.bounds.max = max;
        self
    }

    /// Set the seed concurrency.
    #[must_use]
    pub fn start(mut self, start: usize) -> Self {
        self.bounds.start = start;
        self
    }

    /// Set the full bounds (config path).
    #[must_use]
    pub fn bounds_full(mut self, bounds: Bounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Resolve and coherence-check. `Fixed` (set via `.fixed`/`Preset::Fixed`)
    /// short-circuits to the fast path. Otherwise the `(signal, objective)` come
    /// from the preset, lever overrides win on top, and the law derives from the
    /// objective unless overridden.
    pub fn build(self) -> Result<Concurrency, &'static str> {
        if let Some(n) = self.fixed {
            return Ok(Concurrency::Fixed(n));
        }

        let preset_levers = self.preset.and_then(Preset::levers);
        let objective = self
            .objective
            .or(preset_levers.map(|(_, objective)| objective))
            .ok_or("adaptive build needs a preset or an explicit objective")?;

        // signal: explicit override wins; else the preset's builtin signal.
        let signal = match self.signal {
            Some(signal) => signal,
            None => {
                let kind = preset_levers
                    .map(|(kind, _)| kind)
                    .ok_or("adaptive build needs a preset or an explicit signal")?;
                Signal::Builtin(kind)
            }
        };

        let (law_kind, law_box): (Option<LawKind>, Box<dyn ControlLaw>) = match self.law {
            Some(Law::Builtin(kind)) => (Some(kind), kind.build()),
            Some(Law::Custom(law)) => (None, law),
            None => {
                let kind = objective.default_law();
                (Some(kind), kind.build())
            }
        };

        let strategy = Strategy::new(signal, objective, law_box, law_kind, self.gate, self.bounds)?;
        Ok(Concurrency::Adaptive(strategy))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::{SignalClass, StrategyDescriptor};
    use super::*;

    fn descriptor(concurrency: Concurrency) -> StrategyDescriptor {
        match concurrency {
            Concurrency::Adaptive(strategy) => strategy.descriptor(),
            Concurrency::Fixed(_) => panic!("expected adaptive"),
        }
    }

    #[test]
    fn fixed_short_circuits() {
        let concurrency = ConcurrencyBuilder::new().fixed(25).build().unwrap();
        assert!(matches!(concurrency, Concurrency::Fixed(25)));
    }

    #[test]
    fn preset_then_lever_override() {
        // .hillclimb().coefficient_of_variation_threshold(0.05) — sugar + override.
        let concurrency = ConcurrencyBuilder::new()
            .hillclimb()
            .coefficient_of_variation_threshold(0.05)
            .bounds(1, 512)
            .build()
            .unwrap();
        let descriptor = descriptor(concurrency);
        assert_eq!(
            descriptor.signal,
            SignalClass::Builtin(SignalKind::Throughput)
        );
        assert_eq!(descriptor.objective, Objective::Maximize);
        assert_eq!(descriptor.law, Some(LawKind::HillClimb));
        assert_eq!(descriptor.gate.coefficient_of_variation_threshold, 0.05);
        assert_eq!(descriptor.bounds.max, 512);
    }

    #[test]
    fn law_override_changes_only_the_law() {
        let concurrency = ConcurrencyBuilder::new()
            .hillclimb()
            .law_kind(LawKind::Aimd)
            .build()
            .unwrap();
        assert_eq!(descriptor(concurrency).law, Some(LawKind::Aimd));
    }

    #[test]
    fn incoherent_pair_is_rejected() {
        // maximize with a latency signal → rejected.
        let result = ConcurrencyBuilder::new()
            .signal(SignalKind::LatencyP99)
            .maximize()
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn custom_law_descriptor_is_foreign() {
        struct Noop;
        impl ControlLaw for Noop {
            fn step(&mut self, ctx: super::super::LawStep) -> usize {
                ctx.current
            }
        }
        let concurrency = ConcurrencyBuilder::new()
            .hillclimb()
            .law(Noop)
            .build()
            .unwrap();
        assert_eq!(descriptor(concurrency).law, None);
    }
}
