use std::time::Duration;

use crate::fsm::{Phase, RunController};
use crate::outcome::Outcome;
use crate::report::Recorder;
use crate::scenario::{Scenario, Stage};

/// the seam. one impl per protocol; the real ones drive a proxima pipe
/// (http/grpc/ws, or raw tcp/udp) and return what happened. swapping the
/// engine never touches the driver or the fsm.
pub trait Target {
    fn fire(&mut self) -> Outcome;
}

pub fn run(scenario: &Scenario, target: &mut dyn Target) -> Recorder {
    let mut rec = Recorder::new();
    let mut ctl = RunController::new(scenario.stages.len());
    loop {
        match ctl.advance() {
            Phase::Idle => {}
            Phase::Stage(i) => {
                if let Some(stage) = scenario.stages.get(i) {
                    run_stage(i, stage, target, &mut rec);
                }
            }
            Phase::Report => break,
        }
    }
    rec
}

// Planned-count runner: rate * duration determines the number of arrivals.
// The scheduler module owns the open-loop wall-clock pacing primitives; this
// default-off mock path stays deterministic and fires the planned count back to
// back.
fn run_stage(idx: usize, stage: &Stage, target: &mut dyn Target, rec: &mut Recorder) {
    let planned = (stage.rate_per_sec * stage.duration.as_secs_f64()).round();
    let planned = planned.max(0.0) as u64;
    for _ in 0..planned {
        let outcome = target.fire();
        rec.record(idx, outcome);
    }
}

/// deterministic stand-in so the whole pipeline runs and tests offline.
#[derive(Default)]
pub struct MockTarget {
    n: u64,
}

impl MockTarget {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Target for MockTarget {
    fn fire(&mut self) -> Outcome {
        self.n += 1;
        let spread = self.n.wrapping_mul(2_654_435_761) % 50;
        let latency = Duration::from_secs_f64((1 + spread) as f64 / 1000.0);
        Outcome {
            latency,
            ok: !self.n.is_multiple_of(100),
            timed_out: false,
        }
    }
}
