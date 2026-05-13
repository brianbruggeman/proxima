// P13 SumByPipe bench.
//
// Three arms:
//   1. baseline_pass_through_via_in_memory  — 10k counter records → InMemoryPipe directly
//   2. sum_by_1s_window_2_groups            — 10k counters → SumByPipe(1s, ["route"]) → InMemoryPipe
//                                             window does not expire; all 10k accumulate into 2 groups
//   3. sum_by_1ms_window_frequent_flush     — 10k counters with 1ms window; window expires often;
//                                             each flush triggers emit to inner

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;

use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::metric::MetricSample;
use proxima_telemetry::metric::sample::NumberDataPoint;
use proxima_telemetry::pipes::{InMemoryPipe, SumByPipe, metric_batch_request};
use proxima_telemetry::tag::{ScalarValue, Tag};

extern crate alloc;

const BATCH_SIZE: usize = 10_000;

fn make_counter(route: &'static str, seq: usize) -> MetricSample {
    let mut point = NumberDataPoint {
        value: ScalarValue::U64(1),
        attrs: smallvec::SmallVec::new(),
        ts_ns: seq as u64 * 1_000,
        start_ts_ns: 0,
    };
    point.attrs.push(Tag::Scalar {
        key: "route",
        value: ScalarValue::Str(route),
    });
    MetricSample::Counter(point)
}

fn make_batch_two_routes() -> alloc::vec::Vec<MetricSample> {
    (0..BATCH_SIZE)
        .map(|i| {
            let route = if i % 2 == 0 { "/a" } else { "/b" };
            make_counter(route, i)
        })
        .collect()
}

fn bench_baseline_pass_through_via_in_memory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p13_sum_by_pipe");
    let inner = Arc::new(InMemoryPipe::new());

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("baseline_pass_through_via_in_memory", |bencher| {
        bencher.iter(|| {
            inner.clear();
            let request = metric_batch_request(black_box(make_batch_two_routes()));
            let _ = block_on(SendPipe::call(inner.as_ref(), request));
        });
    });
    group.finish();
}

fn bench_sum_by_1s_window_2_groups(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p13_sum_by_pipe");
    let inner = Arc::new(InMemoryPipe::new());

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("sum_by_1s_window_2_groups", |bencher| {
        bencher.iter(|| {
            inner.clear();
            let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_secs(1), ["route"]);
            let request = metric_batch_request(black_box(make_batch_two_routes()));
            let _ = block_on(SendPipe::call(&pipe, request));
            let _ = block_on(pipe.flush());
        });
    });
    group.finish();
}

fn bench_sum_by_1ms_window_frequent_flush(criterion: &mut Criterion) {
    // measures flush-path cost: send 10 batches × 100 records with a 1ms window,
    // sleeping 2ms between batches so window expires on every batch boundary.
    // throughput is 100 records × 10 batches = 1_000 records per iteration.
    const FLUSH_BATCHES: usize = 10;
    const RECORDS_PER_BATCH: usize = 100;
    const TOTAL: usize = FLUSH_BATCHES * RECORDS_PER_BATCH;

    let mut group = criterion.benchmark_group("p13_sum_by_pipe");
    let inner = Arc::new(InMemoryPipe::new());

    group.throughput(Throughput::Elements(TOTAL as u64));
    group.bench_function("sum_by_1ms_window_frequent_flush", |bencher| {
        bencher.iter(|| {
            inner.clear();
            let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_millis(1), ["route"]);
            for batch_idx in 0..FLUSH_BATCHES {
                let batch: alloc::vec::Vec<MetricSample> = (0..RECORDS_PER_BATCH)
                    .map(|i| {
                        let route = if i % 2 == 0 { "/a" } else { "/b" };
                        make_counter(route, batch_idx * RECORDS_PER_BATCH + i)
                    })
                    .collect();
                let request = metric_batch_request(black_box(batch));
                let _ = block_on(SendPipe::call(&pipe, request));
                // sleep > window so next batch triggers a flush
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = block_on(pipe.flush());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_pass_through_via_in_memory,
    bench_sum_by_1s_window_2_groups,
    bench_sum_by_1ms_window_frequent_flush,
);
criterion_main!(benches);
