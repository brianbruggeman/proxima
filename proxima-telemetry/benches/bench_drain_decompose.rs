#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Decompose the single-drainer cost: where does the ~430 ns/span drain go?
//!
//! Two pipes isolate the layers. NullPipe is drain machinery only (drain_owned
//! Vec + sentinels + copy out of the ring, batch-request Carry, block_on) with
//! NO encode. NativePipe is the same plus the real native frame encode per
//! record. So NativePipe minus NullPipe is the encode share, and NullPipe alone
//! is the pure drain overhead.
//!
//! Across batch sizes 1 / 64 / 512: batch=1 exposes the FIXED per-pass cost
//! (one Vec + one Carry + one block_on amortised over a single span); batch=512
//! exposes the steady-state PER-SPAN cost. The gap tells us whether to attack
//! per-pass overhead or per-record work.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_telemetry::out::native::{FrameSink, NATIVE_FRAME_SIZE};
use proxima_telemetry::pipes::{NativePipe, NullPipe};
use proxima_telemetry::recorder::Recorder;

const BATCHES: [usize; 3] = [1, 64, 512];

struct NullSink;
impl FrameSink for NullSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
    }
}

fn emit(recorder: &Recorder, count: usize) {
    for _ in 0..count {
        drop(
            recorder
                .span(black_box("process"))
                .tag("route", black_box("/v1"))
                .tag("status", black_box(200u64))
                .start(),
        );
    }
}

fn bench_drain_null(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("drain_null_no_encode");
    let recorder = Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start()
        .expect("recorder");
    for batch in BATCHES {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |bencher, &batch| {
                bencher.iter_batched(
                    || emit(&recorder, batch),
                    |()| black_box(recorder.drain()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_drain_native(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("drain_native_with_encode");
    let recorder = Recorder::builder()
        .pipe(NativePipe::new(NullSink))
        .core_count(1)
        .start()
        .expect("recorder");
    for batch in BATCHES {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |bencher, &batch| {
                bencher.iter_batched(
                    || emit(&recorder, batch),
                    |()| black_box(recorder.drain()),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(drain_decompose, bench_drain_null, bench_drain_native);
criterion_main!(drain_decompose);
