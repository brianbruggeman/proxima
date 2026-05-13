// custom-harness bench (no criterion dep, to keep the load-tester dep-light).
// prints ns/op + CoV per pacer, plus the load-bearing arm: offered arrivals
// held under a simulated target stall.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rekt::sched::{GridPacer, IntervalPacer, Pacer};

const ITERS: u64 = 5_000_000;
const RUNS: usize = 7;

fn ns_per_due(mut make: impl FnMut() -> Box<dyn Pacer>) -> (f64, f64) {
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let mut pacer = make();
        let step = Duration::from_micros(1);
        let mut now = Duration::ZERO;
        let mut sink = 0u64;
        let start = Instant::now();
        for _ in 0..ITERS {
            now += step;
            sink = sink.wrapping_add(pacer.due(black_box(now)));
        }
        let elapsed = start.elapsed();
        black_box(sink);
        samples.push(elapsed.as_secs_f64() * 1e9 / ITERS as f64);
    }
    mean_cov(&samples)
}

fn mean_cov(samples: &[f64]) -> (f64, f64) {
    let count = samples.len();
    if count <= 1 {
        return (samples.first().copied().unwrap_or(0.0), 0.0);
    }
    let mean = samples.iter().sum::<f64>() / count as f64;
    // sample variance (n-1): these runs are a sample, not the population
    let var = samples
        .iter()
        .map(|s| (s - mean).powi(2))
        .sum::<f64>()
        / (count as f64 - 1.0);
    let cov = if mean > 0.0 { var.sqrt() / mean * 100.0 } else { 0.0 };
    (mean, cov)
}

fn offered_under_stall(pacer: &mut dyn Pacer, window: Duration, step: Duration, stall: Duration) -> u64 {
    let mut now = Duration::ZERO;
    let mut total = 0u64;
    let half = window / 2;
    let mut stalled = false;
    while now <= window {
        total += pacer.due(now);
        if !stalled && now >= half {
            stalled = true;
            now += stall;
        } else {
            now += step;
        }
    }
    total
}

fn main() {
    let rate = 10_000.0;

    let (grid_ns, grid_cov) = ns_per_due(|| Box::new(GridPacer::new(rate)));
    let (intv_ns, intv_cov) = ns_per_due(|| Box::new(IntervalPacer::new(rate)));

    println!("ns/due  (design-favors: neutral, {RUNS} runs x {ITERS} iters)");
    println!("  grid pacer      {grid_ns:7.2} ns/op   cov {grid_cov:.1}%");
    println!("  interval pacer  {intv_ns:7.2} ns/op   cov {intv_cov:.1}%");

    let window = Duration::from_secs(10);
    let step = Duration::from_micros(50);
    let stall = Duration::from_millis(500);
    let expected = (window.as_secs_f64() * rate) as u64;

    let mut grid = GridPacer::new(rate);
    let mut intv = IntervalPacer::new(rate);
    let grid_offered = offered_under_stall(&mut grid, window, step, stall);
    let intv_offered = offered_under_stall(&mut intv, window, step, stall);

    let grid_held = grid_offered as f64 / expected as f64 * 100.0;
    let intv_held = intv_offered as f64 / expected as f64 * 100.0;

    println!("\noffered-rate held under a {}ms stall (design-favors: incumbent home turf)", stall.as_millis());
    println!("  expected arrivals  {expected}");
    println!("  grid pacer         {grid_offered}  ({grid_held:.2}% held)");
    println!("  interval pacer     {intv_offered}  ({intv_held:.2}% held)  <- coordinated omission");
}
