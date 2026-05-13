// Bench: the syscall amortization the batched terminal flush captures.
//
// The fix changed the terminal sink from one write() per record to one write()
// per drain batch. The write-count reduction is proven deterministically by
// `log_batch_flushes_in_a_single_write` / `console_split_batches_per_writer`
// (src/pipes.rs); this bench quantifies its wall-time value.
//
// `io::sink()` is a no-op (no syscall) and would hide the win — these arms write
// to a REAL fd (/dev/null) so write() is a real syscall:
//
//   amortization/one_write/N  — N formatted lines flushed in ONE write_all (the
//                               path the batched dispatch now takes)
//   amortization/n_writes/N   — the same N lines as N write_all calls (the
//                               per-record path the fix removed)
//   formatter_batched/N       — the real FormatterPipe via the recorder: emit N
//                               spans, drain once -> one write to /dev/null
//
// one_write vs n_writes is the amortization factor; formatter_batched is the
// real path's absolute throughput. Unit: records/sec.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_telemetry::pipes::{FormatterPipe, LogFormat};
use proxima_telemetry::recorder::Recorder;

const SIZES: [usize; 3] = [64, 512, 4096];

// a representative debug line under load (~70 B), the shape that drove the spiral.
const LINE: &str = "DEBUG proxima::serve: handling datagram seq=12345 peer=10.0.0.1:443\n";

fn devnull() -> File {
    OpenOptions::new()
        .write(true)
        .open("/dev/null")
        .expect("open /dev/null")
}

fn amortization_one_write(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_sink_flush");
    for &count in &SIZES {
        group.throughput(Throughput::Elements(count as u64));
        let blob = LINE.repeat(count);
        let mut sink = devnull();
        group.bench_function(BenchmarkId::new("one_write", count), |bencher| {
            bencher.iter(|| {
                std::io::Write::write_all(&mut sink, black_box(blob.as_bytes())).unwrap();
            });
        });
    }
    group.finish();
}

fn amortization_n_writes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_sink_flush");
    for &count in &SIZES {
        group.throughput(Throughput::Elements(count as u64));
        let mut sink = devnull();
        group.bench_function(BenchmarkId::new("n_writes", count), |bencher| {
            bencher.iter(|| {
                for _ in 0..count {
                    std::io::Write::write_all(&mut sink, black_box(LINE.as_bytes())).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn formatter_batched(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_sink_flush");
    for &count in &SIZES {
        group.throughput(Throughput::Elements(count as u64));
        let recorder = Arc::new(
            Recorder::builder()
                .pipe(FormatterPipe::new(devnull(), LogFormat::Human))
                .core_count(1)
                .ring_capacity(8192)
                .start()
                .expect("recorder build"),
        );
        group.bench_function(BenchmarkId::new("formatter_batched", count), |bencher| {
            bencher.iter(|| {
                for _ in 0..count {
                    let guard = recorder
                        .span(black_box("serve"))
                        .tag("route", black_box("/v1"))
                        .start();
                    black_box(&guard);
                    drop(guard);
                }
                recorder.drain();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    amortization_one_write,
    amortization_n_writes,
    formatter_batched,
);
criterion_main!(benches);
