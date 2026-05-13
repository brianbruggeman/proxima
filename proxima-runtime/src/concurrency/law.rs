//! Control laws — how a step is computed from a [`LawStep`]. The one extensible
//! lever: a third party impls [`ControlLaw`] and feeds it to the controller via
//! `.law(MyLaw)` without touching proxima. The four builtins are stateful structs
//! resolved from a [`LawKind`] name (the config-nameable form).
//!
//! Each builtin is paper-derived; the worked example in its test IS the spec.

use alloc::boxed::Box;

use super::{LawStep, Objective};

/// The extensible lever. Given the per-window context, return the next
/// concurrency level. Implementations are stateful (a hillclimb remembers its
/// direction) and `Send` so the controller can cross a core boundary.
pub trait ControlLaw: Send {
    fn step(&mut self, ctx: LawStep) -> usize;
}

/// Config-nameable builtin laws. Resolved into a live `Box<dyn ControlLaw>` at
/// build time. Derived from the objective by default ([`Objective::default_law`]),
/// overridable per the lever model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawKind {
    /// Climb in the direction that last improved the signal; reverse on a
    /// significant drop; hold within the noise floor.
    HillClimb,
    /// Additive-increase below the ceiling, multiplicative-decrease above it.
    Aimd,
    /// Step proportional to the fractional error against a target.
    Proportional,
    /// `current · gradient + sqrt(current)` — the Vegas/Netflix knee law.
    Multiplicative,
}

impl LawKind {
    /// Resolve the name into a fresh, live law.
    #[must_use]
    pub fn build(self) -> Box<dyn ControlLaw> {
        match self {
            Self::HillClimb => Box::new(HillClimb::default()),
            Self::Aimd => Box::new(Aimd::default()),
            Self::Proportional => Box::new(Proportional::default()),
            Self::Multiplicative => Box::new(Multiplicative),
        }
    }

    /// Parse a config token.
    pub fn parse(token: &str) -> Result<Self, &'static str> {
        match token.trim().to_ascii_lowercase().as_str() {
            "hillclimb" | "hill_climb" => Ok(Self::HillClimb),
            "aimd" => Ok(Self::Aimd),
            "proportional" | "pid" => Ok(Self::Proportional),
            "multiplicative" | "gradient" => Ok(Self::Multiplicative),
            _ => Err("law must be hillclimb|aimd|proportional|multiplicative"),
        }
    }
}

/// The law selector carried by config/builder before resolution. Config names a
/// `Builtin` only; the fluent builder additionally takes a `Custom` foreign law.
pub enum Law {
    Builtin(LawKind),
    Custom(Box<dyn ControlLaw>),
}

impl core::fmt::Debug for Law {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Builtin(kind) => write!(formatter, "Law::Builtin({kind:?})"),
            Self::Custom(_) => formatter.write_str("Law::Custom(<dyn>)"),
        }
    }
}

impl Law {
    /// Resolve to a live law: a builtin name becomes a fresh struct, a custom box
    /// passes through.
    #[must_use]
    pub fn resolve(self) -> Box<dyn ControlLaw> {
        match self {
            Self::Builtin(kind) => kind.build(),
            Self::Custom(law) => law,
        }
    }

    /// The builtin kind for the descriptor, or `None` for a foreign law.
    #[must_use]
    pub fn kind(&self) -> Option<LawKind> {
        match self {
            Self::Builtin(kind) => Some(*kind),
            Self::Custom(_) => None,
        }
    }
}

/// Hillclimb: maximise the signal by walking the direction that last helped. A
/// climber must probe to learn, so it always takes a ±1 step — the gate decides
/// the *direction*, never whether to move. A significant improvement keeps the
/// direction; a significant drop reverses it; a sub-threshold change carries no
/// information, so it reverses too — the climber dithers ±1 in place rather than
/// drifting on noise (that dither is how the gate "rejects" a noisy workload).
///
/// Worked example: at concurrency 8, `dir = +1`. throughput 100→120 (significant
/// up) → keep `+1` → 9. Next 120→110 (significant down) → reverse to `-1` → 8. A
/// change below the threshold → reverse and step (dither), not commit.
#[derive(Debug)]
pub struct HillClimb {
    direction: isize,
}

impl Default for HillClimb {
    fn default() -> Self {
        Self { direction: 1 }
    }
}

impl ControlLaw for HillClimb {
    fn step(&mut self, ctx: LawStep) -> usize {
        if ctx.significant() {
            // a real change: keep climbing if it helped, reverse if it hurt.
            if ctx.delta() < 0.0 {
                self.direction = -self.direction;
            }
        } else {
            // no signal from being here: dither in place, never drift.
            self.direction = -self.direction;
        }
        ctx.stepped(self.direction)
    }
}

/// AIMD: grow by one below the utilisation ceiling, cut by `beta` above it. The
/// CoV margin around the ceiling absorbs noise so a blip does not trigger a cut.
///
/// Worked example (`beta = 0.85`, ceiling carried in `Objective::Ceiling`):
/// current 20, util 0.60 < 0.85−margin → +1 → 21. util 0.95 > 0.85+margin →
/// `round(20·0.85)` → 17. util inside the margin band → hold.
#[derive(Debug)]
pub struct Aimd {
    beta: f64,
}

impl Default for Aimd {
    fn default() -> Self {
        Self { beta: 0.85 }
    }
}

impl ControlLaw for Aimd {
    fn step(&mut self, ctx: LawStep) -> usize {
        let Objective::Ceiling(ceiling) = ctx.objective else {
            // AIMD only makes sense against a ceiling; without one it holds.
            return ctx.current;
        };
        let margin = (ctx.coefficient_of_variation_threshold * ceiling).abs();
        if ctx.signal > ceiling + margin {
            ctx.clamp((ctx.current as f64 * self.beta).round() as usize)
        } else if ctx.signal < ceiling - margin {
            ctx.stepped(1)
        } else {
            ctx.current
        }
    }
}

/// Proportional: step proportional to the fractional error against a target.
/// The gate is a *deadband* around the target, not a Δ-test (the signal here is
/// an absolute level, not a trend): when the fractional error is within the
/// `coefficient_of_variation_threshold`, it is noise and the law holds.
///
/// Worked example (`gain = 0.5`, `Objective::Target(5.0)` ms): current 20, p99
/// 2.5 ms → error `(5−2.5)/5 = 0.5` → step `round(20·0.5·0.5) = 5` → 25. p99 ==
/// target → inside the deadband → hold. p99 > target → negative step → shrink.
#[derive(Debug)]
pub struct Proportional {
    gain: f64,
}

impl Default for Proportional {
    fn default() -> Self {
        Self { gain: 0.5 }
    }
}

impl ControlLaw for Proportional {
    fn step(&mut self, ctx: LawStep) -> usize {
        let Objective::Target(target) = ctx.objective else {
            return ctx.current;
        };
        if target <= 0.0 {
            return ctx.current;
        }
        let error = (target - ctx.signal) / target;
        // relative deadband: a fractional error within the threshold is noise.
        if error.abs() <= ctx.coefficient_of_variation_threshold {
            return ctx.current;
        }
        let step = (ctx.current as f64 * self.gain * error).round() as isize;
        ctx.stepped(step)
    }
}

/// Multiplicative (Vegas/Netflix knee): `current · gradient + sqrt(current)`.
/// The signal IS the gradient `rtt_min/rtt_p99 ∈ (0,1]`, clamped to `[0.5, 1]`
/// so a heavy queue halves rather than collapses the limit.
///
/// Worked example: current 16, gradient 1.0 (no queue) → `16·1 + 4 = 20` (grow
/// by sqrt headroom). gradient 0.5 (heavy queue) → `16·0.5 + 4 = 12` (back off).
/// Continuous fixed point `c = c·g + sqrt(c)` ⇒ `c = 1/(1−g)²`; the rounded map
/// settles a little below it where per-step growth drops under 0.5. The law acts
/// on the *absolute* gradient (a constant gradient is itself the signal to grow),
/// so it carries no Δ-gate — it self-stabilises.
#[derive(Debug, Default)]
pub struct Multiplicative;

impl ControlLaw for Multiplicative {
    fn step(&mut self, ctx: LawStep) -> usize {
        let gradient = ctx.signal.clamp(0.5, 1.0);
        let headroom = (ctx.current as f64).sqrt();
        let next = ctx.current as f64 * gradient + headroom;
        ctx.clamp(next.round() as usize)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn ctx(current: usize, signal: f64, prev: f64, objective: Objective) -> LawStep {
        LawStep {
            current,
            signal,
            prev_signal: prev,
            objective,
            cov: 0.0,
            // zero threshold → every change is significant; isolates the law.
            coefficient_of_variation_threshold: 0.0,
            min: 1,
            max: 1_024,
        }
    }

    #[test]
    fn hillclimb_continues_then_reverses() {
        let mut law = HillClimb::default();
        // 100→120 up, first step assumes dir +1 → continues +1 → 9
        assert_eq!(law.step(ctx(8, 120.0, 100.0, Objective::Maximize)), 9);
        // 120→110 down → reverse to -1 → 8
        assert_eq!(law.step(ctx(9, 110.0, 120.0, Objective::Maximize)), 8);
    }

    #[test]
    fn hillclimb_dithers_not_drifts_on_noise() {
        // a flat noisy signal carries no direction: the climber must stay in a
        // ±1 band around where it started, never marching off toward a bound.
        let mut law = HillClimb::default();
        let mut current = 8usize;
        for tick in 0..40 {
            // ±1% wobble under a 10% threshold → always insignificant
            let signal = if tick % 2 == 0 { 101.0 } else { 99.0 };
            let mut step = ctx(current, signal, 100.0, Objective::Maximize);
            step.coefficient_of_variation_threshold = 0.10;
            current = law.step(step);
            assert!(
                (7..=9).contains(&current),
                "stayed in ±1 band, got {current}"
            );
        }
    }

    #[test]
    fn aimd_grows_under_and_cuts_over_ceiling() {
        let mut law = Aimd::default();
        // util 0.60 < 0.85 → +1
        assert_eq!(law.step(ctx(20, 0.60, 0.60, Objective::Ceiling(0.85))), 21);
        // util 0.95 > 0.85 → round(20*0.85)=17
        assert_eq!(law.step(ctx(20, 0.95, 0.95, Objective::Ceiling(0.85))), 17);
    }

    #[test]
    fn proportional_steps_toward_target() {
        let mut law = Proportional::default();
        // p99 2.5ms vs 5ms target → error 0.5 → step round(20*0.5*0.5)=5 → 25
        assert_eq!(law.step(ctx(20, 2.5, 0.0, Objective::Target(5.0))), 25);
        // p99 at target → hold
        assert_eq!(law.step(ctx(20, 5.0, 0.0, Objective::Target(5.0))), 20);
        // p99 over target → shrink: error (5-10)/5=-1 → step round(20*0.5*-1)=-10 → 10
        assert_eq!(law.step(ctx(20, 10.0, 0.0, Objective::Target(5.0))), 10);
    }

    #[test]
    fn multiplicative_grows_at_floor_and_backs_off_under_queue() {
        let mut law = Multiplicative;
        // gradient 1.0, current 16 → 16 + sqrt(16)=4 → 20
        assert_eq!(law.step(ctx(16, 1.0, 0.5, Objective::Knee)), 20);
        // gradient 0.5, current 16 → 8 + 4 → 12
        assert_eq!(law.step(ctx(16, 0.5, 1.0, Objective::Knee)), 12);
    }

    #[test]
    fn multiplicative_is_stable_under_constant_gradient() {
        // Under a constant gradient < 1 the rounded map `c·g + sqrt(c)` settles:
        // growth per step is `sqrt(c) - (1-g)·c`, which falls below the 0.5
        // rounding threshold and the level stops moving. Prove it grows from
        // below, then holds — no runaway, no oscillation.
        let mut law = Multiplicative;
        let mut current = 4usize;
        let mut previous = 0usize;
        for _ in 0..50 {
            previous = current;
            current = law.step(ctx(current, 0.75, 0.5, Objective::Knee));
        }
        assert_eq!(current, previous, "settles to a stable level");
        assert!(current > 4, "grew from the seed");
        assert!(current < 40, "no runaway");
    }

    #[test]
    fn custom_law_is_a_drop_in() {
        struct AlwaysSeven;
        impl ControlLaw for AlwaysSeven {
            fn step(&mut self, _ctx: LawStep) -> usize {
                7
            }
        }
        let mut law: Box<dyn ControlLaw> = Box::new(AlwaysSeven);
        assert_eq!(law.step(ctx(1, 0.0, 0.0, Objective::Maximize)), 7);
    }

    #[test]
    fn law_kind_parses_and_resolves() {
        assert_eq!(LawKind::parse("gradient").unwrap(), LawKind::Multiplicative);
        assert_eq!(LawKind::parse("AIMD").unwrap(), LawKind::Aimd);
        assert!(LawKind::parse("bogus").is_err());
        let mut law = LawKind::HillClimb.build();
        let _ = law.step(ctx(4, 10.0, 1.0, Objective::Maximize));
    }
}
