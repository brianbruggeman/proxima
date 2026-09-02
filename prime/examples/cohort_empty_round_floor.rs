//! stage-2 measurement: the floor cost of a cohort round that does NO work,
//! as a function of member count and chunks-per-round, plus the delta a
//! caller actually pays to parallelize one decode-sized (~1.6 us) chunk.
//!
//! neither `cohort_round_cost.rs` (fixed members=8, chunks=members, real
//! `fetch_add` chunk work) nor `cohort_chunk_sweep.rs` (chunks in
//! {8..256} at a fixed total-serial-work budget, members in {4,8} only)
//! sweeps members in {2,4,8} x chunks-per-round in {1, members, members*4}
//! on a truly empty chunk body, so this file exists to answer that
//! specific grid rather than duplicate either.
//!
//! every round is timed individually (`Instant::now()` around exactly one
//! `session.run(&round)` call) so the reported median/min/range describe
//! the round-cost distribution itself, not an average masking it.

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

const MEMBER_COUNTS: [usize; 3] = [2, 4, 8];
const WARMUP_ROUNDS: usize = 200;
const MEASURED_ROUNDS: usize = 5_000;

/// ~1.6 us of arithmetic, matching the decode-sized elementwise node cited
/// in the task brief (4096 elements at 0.38 ns/element = 1556 ns).
const TARGET_CHUNK_WORK_NS: f64 = 1_556.0;
const CALIBRATION_ITERS: u64 = 4_000_000;
const CALIBRATION_SAMPLES: usize = 5;

struct EmptyRound {
    chunk_count: usize,
}

impl CohortRound<Infallible> for EmptyRound {
    fn chunks(&self) -> usize {
        self.chunk_count
    }

    fn run_chunk(&self, _chunk: ChunkIndex) -> Result<(), Infallible> {
        Ok(())
    }
}

/// busy-work with a data dependency across iterations so the compiler
/// cannot hoist or fold the loop away. identical shape to
/// `cohort_chunk_sweep.rs::do_work` so the calibration is directly
/// comparable across both files.
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

struct OneChunkWork {
    iterations: u64,
}

impl CohortRound<Infallible> for OneChunkWork {
    fn chunks(&self) -> usize {
        1
    }

    fn run_chunk(&self, _chunk: ChunkIndex) -> Result<(), Infallible> {
        std::hint::black_box(do_work(self.iterations));
        Ok(())
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("no NaNs"));
    values[values.len() / 2]
}

struct CellStats {
    median_ns: f64,
    min_ns: f64,
    max_ns: f64,
}

fn time_rounds<Round: CohortRound<Infallible>>(
    session: &prime::os::cohort::CohortSession<'_, Infallible>,
    round: &Round,
) -> CellStats {
    for _ in 0..WARMUP_ROUNDS {
        let _ = session.run(round);
    }

    let mut samples_ns = Vec::with_capacity(MEASURED_ROUNDS);
    for _ in 0..MEASURED_ROUNDS {
        let start = Instant::now();
        let _ = session.run(round);
        samples_ns.push(start.elapsed().as_nanos() as f64);
    }

    let min_ns = samples_ns.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ns = samples_ns.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let median_ns = median(&mut samples_ns);
    CellStats {
        median_ns,
        min_ns,
        max_ns,
    }
}

fn print_stats(label: &str, members: usize, chunks: usize, stats: &CellStats) {
    println!(
        "{label:<12} members={members:>2} chunks={chunks:>4}  median={:>9.1} ns/round  min={:>9.1}  max={:>10.1}  range={:>9.1}",
        stats.median_ns,
        stats.min_ns,
        stats.max_ns,
        stats.max_ns - stats.min_ns,
    );
}

fn uptime_line() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "uptime unavailable".to_string())
}

fn main() {
    println!("uptime: {}", uptime_line());

    let ns_per_iter = calibrate_ns_per_iter();
    let work_iterations = (TARGET_CHUNK_WORK_NS / ns_per_iter).round().max(1.0) as u64;
    println!(
        "calibration: {:.4} ns/iter (do_work), target chunk work = {:.0} ns -> {} iterations",
        ns_per_iter, TARGET_CHUNK_WORK_NS, work_iterations
    );

    println!("\n-- empty round: members x chunks-per-round grid --");
    let mut empty_baseline_by_members: Vec<(usize, f64)> = Vec::new();
    for &members in &MEMBER_COUNTS {
        let config = ThreadCohort::<Infallible>::builder()
            .members(NonZeroUsize::new(members).expect("nonzero"))
            .build();
        let cohort = ThreadCohort::from_config(config).expect("build cohort");
        let session = cohort.enter().expect("open session");

        let chunk_multipliers: [usize; 3] = [1, members, members * 4];
        for &chunks in &chunk_multipliers {
            let round = EmptyRound {
                chunk_count: chunks,
            };
            let stats = time_rounds(&session, &round);
            print_stats("empty", members, chunks, &stats);
            if chunks == 1 {
                empty_baseline_by_members.push((members, stats.median_ns));
            }
        }
        println!("uptime: {}", uptime_line());
    }

    println!(
        "\n-- decode-sized round: one chunk, ~{:.0} ns of arithmetic --",
        TARGET_CHUNK_WORK_NS
    );
    for &members in &MEMBER_COUNTS {
        let config = ThreadCohort::<Infallible>::builder()
            .members(NonZeroUsize::new(members).expect("nonzero"))
            .build();
        let cohort = ThreadCohort::from_config(config).expect("build cohort");
        let session = cohort.enter().expect("open session");

        let round = OneChunkWork {
            iterations: work_iterations,
        };
        let stats = time_rounds(&session, &round);
        print_stats("decode-1chunk", members, 1, &stats);

        let baseline_ns = empty_baseline_by_members
            .iter()
            .find(|(baseline_members, _)| *baseline_members == members)
            .map(|(_, ns)| *ns)
            .expect("baseline recorded for this member count");
        println!(
            "             delta vs empty(members={members},chunks=1): {:>9.1} ns/round",
            stats.median_ns - baseline_ns
        );
    }
    println!("uptime: {}", uptime_line());
}
