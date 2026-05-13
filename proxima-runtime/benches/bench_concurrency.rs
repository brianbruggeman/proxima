//! Benches for the adaptive concurrency controller.
//!
//! Two things to measure:
//! 1. **The control tax** — `observe()` per window, per law. This runs once per
//!    window (~150 ms), never on the request hot path, so the bar is only "is it
//!    negligibly cheap?" (nanoseconds). `design-favors: neutral`.
//! 2. **Delivered throughput vs the incumbent** — a fixed connections-per-core
//!    cap is what this replaces. On the `CrestModel`, the controller is compared
//!    to `Fixed` tuned correctly (the incumbent's home turf — adaptive should
//!    *match*, not beat) and `Fixed` mis-set (where adaptive wins). This is a
//!    quality number (completions delivered), printed once; criterion times the
//!    full drive.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_runtime::concurrency::sim::CrestModel;
use proxima_runtime::concurrency::strategy::ConcurrencyStrategy;
use proxima_runtime::concurrency::{Concurrency, ConcurrencyController, Preset};

const WINDOWS: usize = 300;
const PEAK: usize = 8;

/// Drive a controller against the model; return (final target, total completions).
fn drive(concurrency: Concurrency, model: &CrestModel) -> (usize, f64) {
    let mut controller = ConcurrencyController::new(concurrency);
    let mut current = controller.target();
    let mut total = 0.0;
    for _ in 0..WINDOWS {
        let sample = model.sample(current);
        total += sample.throughput;
        current = controller.observe(sample);
    }
    (current, total)
}

fn bench_decision_tax(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("decision_tax");
    let sample = CrestModel::new(PEAK).sample(16);

    for (label, preset) in [
        ("hillclimb", Preset::HillClimb),
        ("gradient", Preset::Gradient),
        (
            "latency_target",
            Preset::LatencyTarget(std::time::Duration::from_millis(5)),
        ),
        ("headroom", Preset::Headroom(0.85)),
    ] {
        let Concurrency::Adaptive(mut strategy) = Concurrency::from_preset(preset).unwrap() else {
            unreachable!("preset is adaptive");
        };
        group.bench_function(label, |bencher| {
            bencher.iter(|| black_box(strategy.next(black_box(sample))));
        });
    }
    group.finish();
}

fn bench_full_drive(criterion: &mut Criterion) {
    let model = CrestModel::new(PEAK);
    let mut group = criterion.benchmark_group("full_drive");
    group.bench_function("hillclimb_300_windows", |bencher| {
        bencher.iter(|| {
            let knob = Concurrency::builder()
                .hillclimb()
                .start(16)
                .bounds(1, 64)
                .build()
                .unwrap();
            black_box(drive(knob, &model))
        });
    });
    group.finish();

    // the headline quality number, printed once (not a timing).
    let adaptive = Concurrency::builder()
        .hillclimb()
        .start(16)
        .bounds(1, 64)
        .build()
        .unwrap();
    let (adaptive_final, adaptive_total) = drive(adaptive, &model);
    let (_, fixed_correct) = drive(Concurrency::fixed(PEAK), &model);
    let (_, fixed_misset_high) = drive(Concurrency::fixed(PEAK * 4), &model);
    let (_, fixed_misset_low) = drive(Concurrency::fixed(1), &model);
    eprintln!("\ndelivered throughput over {WINDOWS} windows (peak = {PEAK})");
    eprintln!("adaptive(hillclimb)   final={adaptive_final:>3}  total={adaptive_total:>12.0}");
    eprintln!(
        "Fixed(peak={PEAK})        [home turf]  total={fixed_correct:>12.0}  ratio={:.3}",
        adaptive_total / fixed_correct
    );
    eprintln!(
        "Fixed({})  [mis-set 4x] total={fixed_misset_high:>12.0}  ratio={:.3}",
        PEAK * 4,
        adaptive_total / fixed_misset_high
    );
    eprintln!(
        "Fixed(1)              [mis-set lo] total={fixed_misset_low:>12.0}  ratio={:.3}",
        adaptive_total / fixed_misset_low
    );
}

criterion_group!(benches, bench_decision_tax, bench_full_drive);
criterion_main!(benches);
