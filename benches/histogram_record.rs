#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::telemetry::{Labels, Metrics, Telemetry};
use tokio::runtime::Runtime;

fn build_runtime(workers: usize) -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn single_thread_record(criterion: &mut Criterion) {
    let metrics = Metrics::default();
    let labels = Labels::from_pairs(&[("pipe", "single")]);
    let mut group = criterion.benchmark_group("histogram_record");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("single_thread", |bencher| {
        let counter = AtomicU64::new(0);
        bencher.iter(|| {
            let value = counter.fetch_add(1, Ordering::Relaxed) as f64;
            metrics.histogram_record("latency_ms", &labels, value % 60_000.0);
        });
    });
    group.finish();
}

fn concurrent_record(criterion: &mut Criterion) {
    let runtime = build_runtime(8);
    let metrics: Arc<Metrics> = Arc::new(Metrics::default());
    let labels = Labels::from_pairs(&[("pipe", "concurrent")]);
    let mut group = criterion.benchmark_group("histogram_record");
    group.throughput(Throughput::Elements(8));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("8_workers_one_record_each", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let metrics = metrics.clone();
            let labels = labels.clone();
            async move {
                let mut handles = Vec::with_capacity(8);
                for index in 0..8_u64 {
                    let metrics = metrics.clone();
                    let labels = labels.clone();
                    handles.push(tokio::spawn(async move {
                        metrics.histogram_record("latency_ms", &labels, index as f64);
                    }));
                }
                for handle in handles {
                    let _ = handle.await;
                }
            }
        });
    });
    group.finish();
}

criterion_group!(benches, single_thread_record, concurrent_record);
criterion_main!(benches);
