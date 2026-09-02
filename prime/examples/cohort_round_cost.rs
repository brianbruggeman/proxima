//! stage-1 measurement: round cost of `ThreadCohort::run` against
//! `ProximaBackgroundPool::spawn` + `sync_channel::recv`, on identical
//! trivial work with the same member/worker count. reports median + CoV
//! over N samples per arm — the number cohort must dramatically beat is
//! the measured 23,072 ns/call of spawn+recv_wait cited in the design.

#![cfg(all(feature = "runtime-prime-cohort", feature = "runtime-prime-bgpool"))]
// bench harness: a setup/API-contract failure (bad config, poisoned lock,
// no NaNs in a timing sample) has no recovery that makes sense mid-run --
// panicking immediately with a message is correct here, not a shortcut.
#![allow(clippy::expect_used)]

use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use prime::os::background::ProximaBackgroundPool;
use prime::os::cohort::{ChunkIndex, CohortRound, ThreadCohort};

const MEMBERS: usize = 8;
const ROUNDS: usize = 2_000;
const SAMPLES: usize = 5;

struct TrivialRound {
    counter: AtomicUsize,
}

impl CohortRound<Infallible> for TrivialRound {
    fn chunks(&self) -> usize {
        MEMBERS
    }

    fn run_chunk(&self, _chunk: ChunkIndex) -> Result<(), Infallible> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn cohort_round_cost_ns() -> f64 {
    let config = ThreadCohort::<Infallible>::builder()
        .members(NonZeroUsize::new(MEMBERS).expect("nonzero"))
        .spin_polls(2_000)
        .build();
    let cohort = ThreadCohort::from_config(config).expect("build cohort");
    let session = cohort.enter().expect("open session");
    let round = TrivialRound {
        counter: AtomicUsize::new(0),
    };

    for _ in 0..200 {
        let _ = session.run(&round);
    }

    let start = Instant::now();
    for _ in 0..ROUNDS {
        let _ = session.run(&round);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ROUNDS as f64
}

fn bgpool_round_cost_ns() -> f64 {
    let pool = Arc::new(ProximaBackgroundPool::with_threads(MEMBERS).expect("build pool"));

    for _ in 0..200 {
        run_bgpool_round(&pool);
    }

    let start = Instant::now();
    for _ in 0..ROUNDS {
        run_bgpool_round(&pool);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ROUNDS as f64
}

fn run_bgpool_round(pool: &Arc<ProximaBackgroundPool>) {
    let (sender, receiver) = mpsc::sync_channel::<()>(MEMBERS);
    for _ in 0..MEMBERS {
        let sender = sender.clone();
        let future = pool.spawn(move || {
            let _ = sender.send(());
            Ok::<(), proxima_core::ProximaError>(())
        });
        drop(future);
    }
    for _ in 0..MEMBERS {
        receiver.recv().expect("worker reply");
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("no NaNs"));
    values[values.len() / 2]
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn main() {
    let uptime = std::process::Command::new("uptime")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());

    let mut cohort_samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        cohort_samples.push(cohort_round_cost_ns());
    }
    println!("uptime: {uptime}");
    println!("cohort samples (ns/round): {cohort_samples:?}");
    println!(
        "cohort median: {:.1} ns/round, CoV: {:.4}",
        median(&mut cohort_samples.clone()),
        coefficient_of_variation(&cohort_samples)
    );

    let mut bgpool_samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        bgpool_samples.push(bgpool_round_cost_ns());
    }
    println!("uptime: {uptime}");
    println!("bgpool samples (ns/round): {bgpool_samples:?}");
    println!(
        "bgpool median: {:.1} ns/round, CoV: {:.4}",
        median(&mut bgpool_samples.clone()),
        coefficient_of_variation(&bgpool_samples)
    );
}
