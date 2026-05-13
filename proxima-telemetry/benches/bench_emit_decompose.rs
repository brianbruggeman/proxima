#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Decompose the emit hot path. `recorder.span().tag().start()` + drop does:
//! 3 Arc clones (cores/clock/sampler), 2 ID generations, 2 `clock.now_ns()`
//! (start + duration), a tag push, build a SpanRecord, ring push.
//!
//! This isolates the clock cost: the default `SystemClock` calls
//! `SystemTime::now()` twice per span; a `MonotonicCounter` is a single relaxed
//! atomic add. The gap is the syscall/VDSO time-read cost per span. Drain is
//! cleared in setup so every push lands (we measure successful emit, not
//! drop-on-full).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_telemetry::clock::MonotonicCounter;
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;

const BATCH: usize = 2000;

fn emit_batch(recorder: &Recorder) {
    for _ in 0..BATCH {
        let guard = recorder
            .span(black_box("process"))
            .tag("route", black_box("/v1"))
            .tag("status", black_box(200u64))
            .start();
        black_box(&guard);
        drop(guard);
    }
}

fn bench_emit_system_clock(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("emit_system_clock");
    group.throughput(Throughput::Elements(BATCH as u64));
    let recorder = Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .ring_capacity(4096)
        .start()
        .expect("recorder");
    group.bench_function("span_2attr", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| emit_batch(&recorder),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_emit_monotonic_clock(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("emit_monotonic_clock");
    group.throughput(Throughput::Elements(BATCH as u64));
    let recorder = Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .ring_capacity(4096)
        .clock(MonotonicCounter::new(1))
        .start()
        .expect("recorder");
    group.bench_function("span_2attr", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| emit_batch(&recorder),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    emit_decompose,
    bench_emit_system_clock,
    bench_emit_monotonic_clock
);
criterion_main!(emit_decompose);
