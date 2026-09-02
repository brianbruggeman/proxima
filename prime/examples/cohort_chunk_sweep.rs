//! stage-2 pre-tuning: round cost of `ThreadCohort::run` as a function of
//! chunk count, at a fixed serial-work budget calibrated to the real matmul
//! round (~150 us wall at 8 workers). answers three questions before
//! `proxima-tensor/src/sized.rs`'s `ROW_OVERSUBSCRIBE` / `MIN_MACS_PER_CHUNK`
//! get retuned against the cheaper cohort dispatch:
//!
//! 1. where does per-chunk overhead start to dominate (the uniform knee)?
//! 2. where does oversubscription stop paying on a straggler-shaped round?
//! 3. does barrier cost scale with member count, chunk count, or both?
//!
//! chunk work is synthetic (`do_work`), calibrated once via a warm-up
//! timing pass, then sized in loop iterations so the total serial work per
//! round holds constant across chunk counts -- isolating the effect of
//! granularity from the effect of total work.

#![cfg(feature = "runtime-prime-cohort")]
// bench harness: a setup/API-contract failure (bad config, poisoned lock,
// no NaNs in a timing sample) has no recovery that makes sense mid-sweep --
// panicking immediately with a message is the correct behavior here, not a
// shortcut. matches this crate's own `cohort_round_cost.rs` precedent.
#![allow(clippy::expect_used)]

use std::convert::Infallible;
use std::num::NonZeroUsize;
use std::time::Instant;

use prime::os::cohort::{ChunkIndex, CohortRound, ThreadCohort};

const MEMBERS: usize = 8;
const ROUNDS: usize = 2_000;
const WARMUP_ROUNDS: usize = 200;
const SAMPLES: usize = 5;
const CALIBRATION_ITERS: u64 = 4_000_000;
const CALIBRATION_SAMPLES: usize = 5;

/// target wall time of one round at `MEMBERS` workers with zero overhead,
/// matching the real matmul round cited in the task brief (~150 us).
const TARGET_WALL_NS: f64 = 150_000.0;
/// total serial (single-thread-equivalent) work budget per round, held
/// constant across every chunk count and both work profiles so the sweep
/// isolates granularity, not total work.
const TOTAL_SERIAL_NS: f64 = TARGET_WALL_NS * MEMBERS as f64;

const CHUNK_COUNTS: [usize; 6] = [8, 16, 32, 64, 128, 256];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Profile {
    Uniform,
    Imbalanced,
}

/// busy-work with a data dependency across iterations (`acc` feeds the next
/// term) so the compiler cannot hoist or fold the loop away, and a
/// `black_box` on the final value so the call itself is never eliminated.
#[inline(never)]
fn do_work(iterations: u64) -> f64 {
    let mut acc: f64 = 1.0;
    for index in 0..iterations {
        acc = std::hint::black_box(acc * 1.000_000_1 + (index as f64) * 1e-9);
    }
    acc
}

fn calibrate_ns_per_iter() -> f64 {
    let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
    for _ in 0..CALIBRATION_SAMPLES {
        let start = Instant::now();
        std::hint::black_box(do_work(CALIBRATION_ITERS));
        let elapsed = start.elapsed();
        samples.push(elapsed.as_nanos() as f64 / CALIBRATION_ITERS as f64);
    }
    median(&mut samples)
}

struct ChunkWork {
    iters: Box<[u64]>,
}

impl ChunkWork {
    fn uniform(chunks: usize, ns_per_iter: f64) -> Self {
        let total_iters = (TOTAL_SERIAL_NS / ns_per_iter).round() as u64;
        let per_chunk = (total_iters / chunks as u64).max(1);
        Self {
            iters: vec![per_chunk; chunks].into_boxed_slice(),
        }
    }

    /// chunk 0 costs 8x the uniform per-chunk average; the remaining
    /// `chunks - 1` chunks split what is left of the total budget evenly.
    /// mirrors `ROW_OVERSUBSCRIBE`'s documented straggler shape (a measured
    /// 2.04x spread across equal-row chunks) at a deliberately harsher 8x.
    fn imbalanced(chunks: usize, ns_per_iter: f64) -> Self {
        let total_iters = (TOTAL_SERIAL_NS / ns_per_iter).round() as u64;
        let uniform_per_chunk = (total_iters / chunks as u64).max(1);
        let straggler = uniform_per_chunk.saturating_mul(8);
        let remaining_total = total_iters.saturating_sub(straggler);
        let remaining_count = (chunks - 1).max(1) as u64;
        let small = (remaining_total / remaining_count).max(1);
        let mut iters = vec![small; chunks];
        iters[0] = straggler.max(1);
        Self {
            iters: iters.into_boxed_slice(),
        }
    }
}

impl CohortRound<Infallible> for ChunkWork {
    fn chunks(&self) -> usize {
        self.iters.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), Infallible> {
        let iterations = self.iters[chunk.0];
        std::hint::black_box(do_work(iterations));
        Ok(())
    }
}

struct CellResult {
    chunks: usize,
    members: usize,
    profile: Profile,
    samples_ns: Vec<f64>,
}

fn run_cell(members: usize, chunks: usize, profile: Profile, ns_per_iter: f64) -> CellResult {
    let config = ThreadCohort::<Infallible>::builder()
        .members(NonZeroUsize::new(members).expect("nonzero"))
        .spin_polls(2_000)
        .build();
    let cohort = ThreadCohort::from_config(config).expect("build cohort");
    let session = cohort.enter().expect("open session");
    let round = match profile {
        Profile::Uniform => ChunkWork::uniform(chunks, ns_per_iter),
        Profile::Imbalanced => ChunkWork::imbalanced(chunks, ns_per_iter),
    };

    for _ in 0..WARMUP_ROUNDS {
        let _ = session.run(&round);
    }

    let mut samples_ns = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for _ in 0..ROUNDS {
            let _ = session.run(&round);
        }
        let elapsed = start.elapsed();
        samples_ns.push(elapsed.as_nanos() as f64 / ROUNDS as f64);
    }

    CellResult {
        chunks,
        members,
        profile,
        samples_ns,
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

fn report(cell: &CellResult, ideal_ns: f64) {
    let mut sorted = cell.samples_ns.clone();
    let round_median = median(&mut sorted);
    let cov = coefficient_of_variation(&cell.samples_ns);
    let overhead_per_chunk = (round_median - ideal_ns) / cell.chunks as f64;
    let profile_name = match cell.profile {
        Profile::Uniform => "uniform",
        Profile::Imbalanced => "imbalanced",
    };

    if cov > 0.05 {
        let min = sorted.first().copied().unwrap_or(f64::NAN);
        let max = sorted.last().copied().unwrap_or(f64::NAN);
        println!(
            "members={:>2} chunks={:>3} {:<10} range={:>9.1}-{:<9.1} ns/round (CoV {:.1}% > 5%, reporting range) overhead/chunk={:>7.1} ns  samples={:?}",
            cell.members,
            cell.chunks,
            profile_name,
            min,
            max,
            cov * 100.0,
            overhead_per_chunk,
            sorted
        );
    } else {
        println!(
            "members={:>2} chunks={:>3} {:<10} median={:>9.1} ns/round  CoV={:.2}%  overhead/chunk={:>7.1} ns  samples={:?}",
            cell.members,
            cell.chunks,
            profile_name,
            round_median,
            cov * 100.0,
            overhead_per_chunk,
            sorted
        );
    }
}

fn main() {
    let uptime = std::process::Command::new("uptime")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string());
    println!("uptime: {uptime}");

    let ns_per_iter = calibrate_ns_per_iter();
    println!(
        "calibration: {:.4} ns/iter (do_work), total_serial_ns budget/round = {:.0}",
        ns_per_iter, TOTAL_SERIAL_NS
    );

    println!("\n-- chunk sweep at members={MEMBERS} --");
    for &chunks in &CHUNK_COUNTS {
        let ideal_ns = TOTAL_SERIAL_NS / MEMBERS as f64;
        let uniform = run_cell(MEMBERS, chunks, Profile::Uniform, ns_per_iter);
        report(&uniform, ideal_ns);
        let imbalanced = run_cell(MEMBERS, chunks, Profile::Imbalanced, ns_per_iter);
        report(&imbalanced, ideal_ns);
    }

    println!("\n-- member scaling check: chunks=64, uniform, members=4 vs members=8 --");
    for &members in &[4_usize, 8_usize] {
        let ideal_ns = TOTAL_SERIAL_NS / members as f64;
        let cell = run_cell(members, 64, Profile::Uniform, ns_per_iter);
        report(&cell, ideal_ns);
    }
}
