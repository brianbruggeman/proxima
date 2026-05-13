//! [`ConcurrencyController`] — the measure→decide→apply loop, factored sans-IO.
//! `observe(Sample) -> target` is the pure decision (testable with no sleeps);
//! `reconcile(pool)` is the apply that spawns/retires workers against a
//! caller-owned [`WorkerPool`]. The caller owns measure (timing the window,
//! gathering stats) — the correct sans-IO factoring (P11): proxima-runtime knows
//! nothing about HTTP, sockets, or the work the pool performs.

use core::time::Duration;

use super::Sample;
use super::strategy::{Concurrency, ConcurrencyStrategy, Strategy};

/// A pool of identical in-flight workers whose count the controller raises and
/// lowers. The caller implements this over its real workers (rekt's per-core
/// keep-alive clients; a server's per-core handler slots).
pub trait WorkerPool {
    /// Add one in-flight worker.
    fn spawn_one(&mut self);
    /// Signal one worker to retire after its current operation.
    fn retire_one(&mut self);
    /// Current live worker count.
    fn live(&self) -> usize;
}

enum ControllerKind {
    Fixed(usize),
    Adaptive(Strategy),
}

/// Drives a [`Concurrency`] knob: holds the current target and turns each
/// window's [`Sample`] into the next target.
pub struct ConcurrencyController {
    kind: ControllerKind,
    target: usize,
}

impl core::fmt::Debug for ConcurrencyController {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ConcurrencyController")
            .field("target", &self.target)
            .field(
                "kind",
                &match self.kind {
                    ControllerKind::Fixed(_) => "fixed",
                    ControllerKind::Adaptive(_) => "adaptive",
                },
            )
            .finish()
    }
}

impl ConcurrencyController {
    /// Seed from a knob. The opening target is the fixed cap or the strategy's
    /// `start`.
    #[must_use]
    pub fn new(concurrency: Concurrency) -> Self {
        let target = concurrency.initial();
        let kind = match concurrency {
            Concurrency::Fixed(n) => ControllerKind::Fixed(n),
            Concurrency::Adaptive(strategy) => ControllerKind::Adaptive(strategy),
        };
        Self { kind, target }
    }

    /// The current target concurrency.
    #[must_use]
    pub fn target(&self) -> usize {
        self.target
    }

    /// The control window cadence — `None` for a fixed knob (nothing to sample).
    #[must_use]
    pub fn window(&self) -> Option<Duration> {
        match &self.kind {
            ControllerKind::Fixed(_) => None,
            ControllerKind::Adaptive(strategy) => Some(strategy.window()),
        }
    }

    /// True if there is anything to drive each window. A fixed knob is set-and-
    /// forget; the caller can skip the measure loop entirely.
    #[must_use]
    pub fn is_adaptive(&self) -> bool {
        matches!(self.kind, ControllerKind::Adaptive(_))
    }

    /// Feed a window's sample, get the next target. The decide step — pure, no IO.
    /// A fixed knob ignores the sample and returns its cap.
    pub fn observe(&mut self, sample: Sample) -> usize {
        self.target = match &mut self.kind {
            ControllerKind::Fixed(n) => *n,
            ControllerKind::Adaptive(strategy) => strategy.next(sample),
        };
        self.target
    }

    /// Apply the current target to a pool: spawn or retire workers until the live
    /// count matches. The apply step.
    pub fn reconcile<P: WorkerPool>(&self, pool: &mut P) {
        while pool.live() < self.target {
            pool.spawn_one();
        }
        while pool.live() > self.target {
            pool.retire_one();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::strategy::Preset;
    use super::*;

    /// A trivial counting pool — proves `reconcile` drives `live()` to `target`.
    #[derive(Default)]
    struct CountingPool {
        live: usize,
        spawned: usize,
        retired: usize,
    }

    impl WorkerPool for CountingPool {
        fn spawn_one(&mut self) {
            self.live += 1;
            self.spawned += 1;
        }
        fn retire_one(&mut self) {
            self.live -= 1;
            self.retired += 1;
        }
        fn live(&self) -> usize {
            self.live
        }
    }

    #[test]
    fn fixed_controller_holds_its_cap() {
        let mut controller = ConcurrencyController::new(Concurrency::fixed(25));
        assert_eq!(controller.target(), 25);
        assert!(!controller.is_adaptive());
        assert_eq!(controller.window(), None);
        // any sample is ignored
        assert_eq!(controller.observe(Sample::seed(99)), 25);
    }

    #[test]
    fn reconcile_spawns_up_to_target() {
        let controller = ConcurrencyController::new(Concurrency::fixed(8));
        let mut pool = CountingPool::default();
        controller.reconcile(&mut pool);
        assert_eq!(pool.live(), 8);
        assert_eq!(pool.spawned, 8);
        assert_eq!(pool.retired, 0);
    }

    #[test]
    fn reconcile_retires_down_to_target() {
        let controller = ConcurrencyController::new(Concurrency::fixed(3));
        let mut pool = CountingPool {
            live: 10,
            ..CountingPool::default()
        };
        controller.reconcile(&mut pool);
        assert_eq!(pool.live(), 3);
        assert_eq!(pool.retired, 7);
    }

    #[test]
    fn adaptive_controller_reports_window() {
        let controller =
            ConcurrencyController::new(Concurrency::from_preset(Preset::Gradient).unwrap());
        assert!(controller.is_adaptive());
        assert_eq!(controller.window(), Some(Duration::from_millis(150)));
    }

    use super::super::Concurrency as _Concurrency;
    use super::super::sim::CrestModel;

    /// Run a controller against a model for `windows` windows; return the final
    /// target and the cumulative throughput (the work the loop got done).
    fn drive(concurrency: _Concurrency, model: &CrestModel, windows: usize) -> (usize, f64) {
        let mut controller = ConcurrencyController::new(concurrency);
        let mut current = controller.target();
        let mut total = 0.0;
        for _ in 0..windows {
            let sample = model.sample(current);
            total += sample.throughput;
            current = controller.observe(sample);
        }
        (current, total)
    }

    #[test]
    fn hillclimb_converges_to_the_crest() {
        // peak 8; per-step throughput gain (~12%) clears the 4% noise floor while
        // the 1% ripple stays under it. Starts above the crest (start 16) and
        // walks down to it.
        let model = CrestModel::new(8);
        let knob = Concurrency::builder()
            .hillclimb()
            .start(16)
            .bounds(1, 64)
            .build()
            .unwrap();
        let (final_target, _) = drive(knob, &model, 300);
        assert!(
            (6..=10).contains(&final_target),
            "converged near 8, got {final_target}"
        );
    }

    #[test]
    fn gradient_converges_to_the_crest() {
        // the latency-gradient law finds the same crest: gradient is 1.0 below the
        // peak (grow) and < 1 above it (back off).
        let model = CrestModel::new(8);
        let knob = Concurrency::builder()
            .gradient()
            .start(2)
            .bounds(1, 64)
            .build()
            .unwrap();
        let (final_target, _) = drive(knob, &model, 300);
        assert!(
            (6..=12).contains(&final_target),
            "converged near 8, got {final_target}"
        );
    }

    #[test]
    fn gate_rejects_sub_cov_steps() {
        // a noisy-but-flat workload: throughput barely moves, CoV is huge. The
        // gate must hold position — chasing this noise is the failure mode.
        let mut controller =
            ConcurrencyController::new(Concurrency::from_preset(Preset::HillClimb).unwrap());
        let start = controller.target();
        for tick in 0..50 {
            let mut sample = Sample::seed(controller.target());
            // ±0.5% wobble around 1000, CoV 50% → floor is 100x the wobble.
            sample.throughput = 1_000.0 + if tick % 2 == 0 { 5.0 } else { -5.0 };
            sample.cov = 0.5;
            let target = controller.observe(sample);
            // a probing controller dithers ±1 but must never DRIFT on noise.
            assert!(
                (start - 1..=start + 1).contains(&target),
                "noise drifted the target to {target} from start {start}"
            );
        }
    }

    #[test]
    fn adaptive_beats_a_misset_fixed() {
        // the whole point: a fixed cap tuned for the wrong workload loses to the
        // controller that finds the crest. Mis-set Fixed at 4x the peak.
        let model = CrestModel::new(8);
        let adaptive = Concurrency::builder()
            .hillclimb()
            .start(16)
            .bounds(1, 64)
            .build()
            .unwrap();
        let (_, adaptive_total) = drive(adaptive, &model, 300);
        let (_, fixed_total) = drive(Concurrency::fixed(32), &model, 300);
        assert!(
            adaptive_total > fixed_total,
            "adaptive {adaptive_total:.0} must beat mis-set Fixed(32) {fixed_total:.0}"
        );
    }
}
