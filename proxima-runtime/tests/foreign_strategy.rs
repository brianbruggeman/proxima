//! Foreign-strategy fixture (disciplined-component gate, the extension proof).
//!
//! A third-party crate depends on `proxima-runtime` ALONE and extends the
//! concurrency controller two ways without forking proxima: a custom
//! [`ControlLaw`] (swap-the-law tier) and a custom `signal_fn` reading the app's
//! own state (swap-the-signal tier). This file uses ONLY the public API — if it
//! compiles and drives the controller, the extension points hold.

#![cfg(feature = "concurrency")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;

use proxima_runtime::concurrency::law::ControlLaw;
use proxima_runtime::concurrency::{
    Concurrency, ConcurrencyController, LawStep, Objective, Sample,
};

/// A foreign control law: drive concurrency toward whatever keeps the app's queue
/// depth near a setpoint. Additive — grow when the queue is shallow, shrink when
/// it is deep. Knows nothing of proxima's builtins.
struct QueueDepthLaw {
    setpoint: f64,
}

impl ControlLaw for QueueDepthLaw {
    fn step(&mut self, ctx: LawStep) -> usize {
        let error = self.setpoint - ctx.signal; // signal = measured queue depth
        if error.abs() < 1.0 {
            return ctx.current;
        }
        if error > 0.0 {
            ctx.stepped(1)
        } else {
            ctx.stepped(-1)
        }
    }
}

/// The app's own state — a queue whose depth depends on the in-flight count. The
/// custom signal reads this, not anything in `Sample`.
struct AppQueue {
    depth: AtomicUsize,
    target_concurrency: usize,
}

impl AppQueue {
    fn observe_concurrency(&self, concurrency: usize) {
        // queue depth = how far over the sweet spot we are pushed in-flight work.
        let depth = concurrency.saturating_sub(self.target_concurrency);
        self.depth.store(depth, Ordering::Relaxed);
    }
    fn depth(&self) -> f64 {
        self.depth.load(Ordering::Relaxed) as f64
    }
}

#[test]
fn foreign_law_and_signal_drive_the_controller() {
    let queue = std::sync::Arc::new(AppQueue {
        depth: AtomicUsize::new(0),
        target_concurrency: 12,
    });

    // build the controller entirely through the public fluent surface: a custom
    // signal closure (reads app state) + a custom law + a maximize objective
    // (coherent with a custom signal). Zero proxima edits.
    let signal_queue = std::sync::Arc::clone(&queue);
    let knob: Concurrency = Concurrency::builder()
        .signal_fn(move |_sample: &Sample| signal_queue.depth())
        .objective(Objective::Maximize)
        .law(QueueDepthLaw { setpoint: 1.0 })
        .window(Duration::from_millis(100))
        .bounds(1, 64)
        .start(40)
        .build()
        .expect("foreign strategy builds via public API");

    let mut controller = ConcurrencyController::new(knob);
    let mut current = controller.target();
    for _ in 0..100 {
        queue.observe_concurrency(current);
        // the custom signal ignores the Sample fields and reads app state, so a
        // seed sample at the current level is all the controller needs.
        current = controller.observe(Sample::seed(current));
    }

    // the foreign law pulled concurrency down from 40 toward the depth-1 setpoint
    // (target_concurrency 12 + 1).
    assert!(
        (11..=14).contains(&current),
        "foreign law converged near the queue setpoint, got {current}"
    );
}

#[test]
fn foreign_whole_controller_replaces_the_strategy() {
    use proxima_runtime::concurrency::strategy::ConcurrencyStrategy;

    // the deepest tier: replace ConcurrencyStrategy wholesale for a controller
    // that does not fit signal x objective x law at all.
    struct EveryOtherStrategy {
        flip: bool,
    }
    impl ConcurrencyStrategy for EveryOtherStrategy {
        fn next(&mut self, sample: Sample) -> usize {
            self.flip = !self.flip;
            if self.flip {
                sample.concurrency + 1
            } else {
                sample.concurrency
            }
        }
    }

    let mut strategy = EveryOtherStrategy { flip: false };
    assert_eq!(strategy.next(Sample::seed(8)), 9);
    assert_eq!(strategy.next(Sample::seed(9)), 9);
}
