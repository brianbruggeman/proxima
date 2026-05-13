//! Micro-bench for the open-loop scenario driver (`drive_workload_open`)
//! against the closed-loop driver (`drive_workload`). Both run the same
//! synth pipe so the comparison isolates driver overhead from
//! per-request work.
//!
//! Groups:
//!   - `load_driver_throughput` — fixed-request-count completion time.
//!     `open_loop_rate_paced` paces 100 requests at 200 rps (≈500ms);
//!     `closed_loop_saturating` runs 100 requests at concurrency 8
//!     against the same pipe.
//!
//! The throughput comparison is intentionally one-sided: open-loop is
//! rate-paced (deliberately slower), closed-loop saturates. Comparing
//! per-iter time tells you the driver's pacing fidelity AND the
//! closed-loop's saturation ceiling.
//!
//! requires-features: runtime-tokio (for TokioPerCoreRuntime as the
//! Runtime impl backing Runtime::timer_at).

#![cfg(feature = "runtime-tokio")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
use proxima::request::{Request, Response};
use proxima::runtime::{Runtime, TokioPerCoreRuntime};
use proxima::scenarios::DurationSpec;
use proxima::{LoadContext, WorkloadSpec, into_handle};
use proxima_primitives::pipe::SendPipe;

const TARGET_REQUESTS: u64 = 100;
// open-loop: drive at 100 rps for 1s → ~100 requests rate-paced (≈1s
// per iter, dominated by the cadence wait, exercises the full driver
// path including periodic snapshot ticker and drain). closed-loop
// drives the same 100 requests at concurrency 8 against an in-memory
// synth pipe — saturates and finishes in microseconds, isolating
// per-call dispatch overhead.
const OPEN_LOOP_RPS: u64 = 100;
const CLOSED_LOOP_CONCURRENCY: usize = 8;

fn configure_group<MeasurementImpl: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, MeasurementImpl>,
) {
    // open-loop arm holds ~1s per iter; sample_size=10 over 12s budget
    // gives ~10-12 measurements per arm. closed-loop is microseconds
    // per iter so the same budget yields tens of thousands.
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(12));
    group.throughput(Throughput::Elements(TARGET_REQUESTS));
}

struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(b"ok"))) }
    }
}


// yields once per request so workers genuinely interleave — `EchoPipe` never
// suspends, so the first-polled worker drains everything and concurrency is
// illusory. this exposes how the closed-loop driver's `join_all` scales with
// concurrency under real wakeups (O(1)-per-wake vs O(N)-per-wake polling).
struct YieldPipe;

impl SendPipe for YieldPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            tokio::task::yield_now().await;
            Ok(Response::ok(Bytes::from_static(b"ok")))
        }
    }
}


fn build_runtime() -> Arc<dyn Runtime> {
    Arc::new(TokioPerCoreRuntime::new(1).expect("build TokioPerCoreRuntime"))
}

fn bench_load_driver_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("load_driver_throughput");
    configure_group(&mut group);

    group.bench_function("open_loop_rate_paced", |bencher| {
        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio current_thread");
        // hoist construction outside the timed loop — LoadContext setup
        // is heavy and not what this bench measures.
        let context = LoadContext::with_default_registry().expect("context");
        let runtime = build_runtime();
        let workload =
            WorkloadSpec::new_open_loop("echo", OPEN_LOOP_RPS, DurationSpec::from_secs(1))
                .with_concurrency(32);
        bencher.iter(|| {
            tokio_rt.block_on(async {
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(async {
                        let pipe = into_handle(EchoPipe);
                        let metrics = context.metrics.as_ref().cloned().expect("metrics handle");
                        proxima::scenarios::orchestrator::drive_workload_open(
                            pipe,
                            &workload,
                            context.telemetry.clone(),
                            metrics,
                            runtime.clone(),
                            None,
                        )
                        .await
                        .expect("open-loop drive")
                    })
                    .await
            });
        });
    });

    group.bench_function("closed_loop_saturating", |bencher| {
        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio current_thread");
        let context = LoadContext::with_default_registry().expect("context");
        let workload = WorkloadSpec::new("echo", TARGET_REQUESTS as usize)
            .with_concurrency(CLOSED_LOOP_CONCURRENCY);
        bencher.iter(|| {
            tokio_rt.block_on(async {
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(async {
                        let pipe = into_handle(EchoPipe);
                        proxima::scenarios::orchestrator::drive_workload(
                            pipe,
                            &workload,
                            context.telemetry.clone(),
                        )
                        .await
                        .expect("closed-loop drive")
                    })
                    .await
            });
        });
    });

    group.finish();
}

/// Targets the telemetry hot path directly: 10_000 `histogram_record`
/// calls against the same metric name + labels. C1 (metric-key
/// interning) is measured here, where the driver bench can't move the
/// needle because sleep / saturation dominates that workload.
fn bench_histogram_record_hot_path(criterion: &mut Criterion) {
    use proxima::{Metrics, Telemetry};

    const RECORDS_PER_ITER: usize = 10_000;
    let metrics = Metrics::default();
    let labels = proxima::Labels::from_pairs(&[("pipe", "echo"), ("region", "us")]);

    let mut group = criterion.benchmark_group("telemetry_hot_path");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(RECORDS_PER_ITER as u64));

    group.bench_function("histogram_record_10k", |bencher| {
        bencher.iter(|| {
            for value in 1..=RECORDS_PER_ITER {
                metrics.histogram_record("proxima.workload.co_latency_ms", &labels, value as f64);
            }
        });
    });

    group.finish();
}

/// Scaling guard for the closed-loop driver's `join_all`: drives a fixed
/// request count through `YieldPipe` (one real wakeup per request) at rising
/// concurrency. If per-request cost stays flat as concurrency climbs, the
/// driver polls O(1)-per-wake; if it climbs with concurrency, `join_all` is
/// re-polling all N futures per wakeup (O(N)-per-wake) and should move to
/// `FuturesUnordered`. This is the arm `closed_loop_saturating` can't be — its
/// echo pipe never suspends, so concurrency there is illusory.
fn bench_closed_loop_scaling(criterion: &mut Criterion) {
    const SCALING_REQUESTS: u64 = 2_048;

    let mut group = criterion.benchmark_group("closed_loop_concurrency");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(8));
    group.throughput(Throughput::Elements(SCALING_REQUESTS));

    for concurrency in [8usize, 64, 512] {
        group.bench_function(format!("yield_pipe_c{concurrency}"), |bencher| {
            let tokio_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio current_thread");
            let context = LoadContext::with_default_registry().expect("context");
            let workload =
                WorkloadSpec::new("yield", SCALING_REQUESTS as usize).with_concurrency(concurrency);
            bencher.iter(|| {
                tokio_rt.block_on(async {
                    let local = tokio::task::LocalSet::new();
                    local
                        .run_until(async {
                            let pipe = into_handle(YieldPipe);
                            proxima::scenarios::orchestrator::drive_workload(
                                pipe,
                                &workload,
                                context.telemetry.clone(),
                            )
                            .await
                            .expect("closed-loop drive")
                        })
                        .await
                });
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_load_driver_throughput,
    bench_closed_loop_scaling,
    bench_histogram_record_hot_path
);
criterion_main!(benches);
