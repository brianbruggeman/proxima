#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
// P12 re-bench: e2e composition with matched terminal sinks.
// Final integration pass: all proxima arms now use TelemetryConfig::default() +
// Recorder::from_config() so the bench measures proxima under the actual
// production-default config (sampler=AlwaysOn, record_sharing=Arc, SharedRing fanout).
//
// Four paired comparisons:
//   1. traces  : proxima (InMemoryPipe) vs opentelemetry_sdk (InMemorySpanExporter)
//   2. metrics : proxima (direct atomics) vs metrics-crate + prometheus (also atomics)
//   3. logs    : proxima (FormatterPipe<io::sink, Json>) vs tracing_subscriber::fmt (io::sink)
//   4. full    : proxima 5-signal workload vs composed 4-crate stack (traces+metrics+logs)
//
// P12 fix: prior runs used NullExporter (no-op) on the proxima side while incumbents
// stored (InMemorySpanExporter) or formatted (fmt::layer). Asymmetric — proxima's terminal
// did zero work. Now:
//   - traces: InMemoryPipe — lock + clone + Vec push per span (matches InMemorySpanExporter)
//   - metrics: direct Counter atomics — same shape as prometheus::IntCounter (no change needed)
//   - logs: FormatterPipe<io::sink, Json> — format + write per log (matches fmt::layer)
//   - full: InMemoryPipe for traces, FormatterPipe for logs (same as above)
//
// All four arms emit the SAME logical workload per iteration:
//   traces  : 1 span, 8 attrs, 2 events
//   metrics : 1 counter increment (2 labels), 1 histogram record (2 labels, 0.0123s)
//   logs    : 1 INFO log, 4 fields
//   full    : all of the above composed
//
// Setup is hoisted OUT of b.iter() loops per the C9 discipline.md lesson (16.2 µs artifact).
// Draining is the per-batch setup so drain cost does not contaminate emit measurement.

extern crate alloc;

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

// ── shared workload constants ────────────────────────────────────────────────

// The logical "HTTP request" workload — same semantic values across all arms.
// Metric labels
const ROUTE_LABEL: &str = "/v1/upload";
const METHOD_LABEL: &str = "POST";

// ── pair 1: traces ───────────────────────────────────────────────────────────

fn bench_traces_proxima(criterion: &mut Criterion) {
    use proxima_telemetry::config::TelemetryConfig;
    use proxima_telemetry::pipes::InMemoryPipe;
    use proxima_telemetry::recorder::Recorder;

    let recorder = Recorder::from_config(&TelemetryConfig::default())
        .with_pipe(InMemoryPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench");

    let mut group = criterion.benchmark_group("e2e_composition_traces");
    group.bench_function("proxima_traces", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                let mut guard = recorder
                    .span(black_box("process_request"))
                    .tag("http.method", black_box("POST"))
                    .tag("http.route", black_box("/v1/upload"))
                    .tag("http.scheme", black_box("https"))
                    .tag("net.host.name", black_box("api.example.com"))
                    .tag("net.host.port", black_box(443i64))
                    .tag("user_agent", black_box("bench/1.0"))
                    .tag("http.status_code", black_box(200i64))
                    .tag("response.size_bytes", black_box(4096i64))
                    .start();

                if let Some(evt) = guard.event(black_box("validated")) {
                    evt.emit();
                }
                if let Some(evt) = guard.event(black_box("dispatched")) {
                    evt.emit();
                }

                black_box(&guard);
                drop(guard);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_traces_opentelemetry_sdk(criterion: &mut Criterion) {
    use opentelemetry::KeyValue;
    use opentelemetry::trace::{Span, Tracer, TracerProvider};
    use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    let mut group = criterion.benchmark_group("e2e_composition_traces");
    group.bench_function("opentelemetry_sdk_traces", |bencher| {
        bencher.iter(|| {
            let mut span = tracer.start(black_box("process_request"));
            span.set_attribute(KeyValue::new("http.method", black_box("POST")));
            span.set_attribute(KeyValue::new("http.route", black_box("/v1/upload")));
            span.set_attribute(KeyValue::new("http.scheme", black_box("https")));
            span.set_attribute(KeyValue::new("net.host.name", black_box("api.example.com")));
            span.set_attribute(KeyValue::new("net.host.port", black_box(443i64)));
            span.set_attribute(KeyValue::new("user_agent", black_box("bench/1.0")));
            span.set_attribute(KeyValue::new("http.status_code", black_box(200i64)));
            span.set_attribute(KeyValue::new("response.size_bytes", black_box(4096i64)));
            span.add_event("validated", vec![]);
            span.add_event("dispatched", vec![]);
            span.end();
            black_box(span)
        });
    });
    group.finish();
}

// ── pair 2: metrics ──────────────────────────────────────────────────────────

fn bench_metrics_proxima(criterion: &mut Criterion) {
    use proxima_telemetry::config::TelemetryConfig;
    use proxima_telemetry::recorder::Recorder;

    // metrics are direct atomic increments — NullPipe (from the Noop default exporter)
    // is the correct sink: counters/histograms bypass the pipe on the hot path.
    let recorder = Recorder::from_config(&TelemetryConfig::default())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench");

    // direct-instrument handles — allocated once at startup, zero-alloc on hot path
    let counter = recorder.counter("http.requests");
    let histogram = recorder.histogram("http.latency");

    let mut group = criterion.benchmark_group("e2e_composition_metrics");
    group.bench_function("proxima_metrics", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                // counter: single AtomicU64::fetch_add, no registry lookup, no clock, no alloc
                counter.add(black_box(1u64), &[]);
                // histogram: branchless bucket pick + AtomicU64::fetch_add, no ring push
                black_box(&histogram).record(black_box(0.0123f64));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_metrics_prometheus(criterion: &mut Criterion) {
    let counter = prometheus::IntCounterVec::new(
        prometheus::Opts::new("http_requests_total", "total request count"),
        &["route", "method"],
    )
    .expect("valid opts");
    let histogram = prometheus::HistogramVec::new(
        prometheus::HistogramOpts::new("http_latency_seconds", "request latency"),
        &["route", "method"],
    )
    .expect("valid opts");

    let mut group = criterion.benchmark_group("e2e_composition_metrics");
    group.bench_function("prometheus_metrics", |bencher| {
        bencher.iter(|| {
            black_box(&counter)
                .with_label_values(&[black_box(ROUTE_LABEL), black_box(METHOD_LABEL)])
                .inc();
            black_box(&histogram)
                .with_label_values(&[black_box(ROUTE_LABEL), black_box(METHOD_LABEL)])
                .observe(black_box(0.0123));
        });
    });
    group.finish();
}

fn bench_metrics_crate(criterion: &mut Criterion) {
    // metrics crate: register-once pattern (counter + histogram pre-registered)
    let mut group = criterion.benchmark_group("e2e_composition_metrics");
    group.bench_function("metrics_crate", |bencher| {
        bencher.iter(|| {
            metrics::counter!(
                black_box("http.requests"),
                "route" => black_box(ROUTE_LABEL),
                "method" => black_box(METHOD_LABEL)
            )
            .increment(black_box(1));
            metrics::histogram!(
                black_box("http.latency"),
                "route" => black_box(ROUTE_LABEL),
                "method" => black_box(METHOD_LABEL)
            )
            .record(black_box(0.0123));
        });
    });
    group.finish();
}

// ── pair 3: logs ─────────────────────────────────────────────────────────────

fn bench_logs_proxima(criterion: &mut Criterion) {
    use proxima_telemetry::config::TelemetryConfig;
    use proxima_telemetry::pipes::{FormatterPipe, LogFormat};
    use proxima_telemetry::recorder::Recorder;

    let recorder = Recorder::from_config(&TelemetryConfig::default())
        .with_pipe(FormatterPipe::new(std::io::sink(), LogFormat::Json))
        .core_count(1)
        .start()
        .expect("recorder build failed in bench");

    let mut group = criterion.benchmark_group("e2e_composition_logs");
    group.bench_function("proxima_logs", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                recorder
                    .log()
                    .message(black_box("request processed"))
                    .tag("user_id", black_box(1001i64))
                    .tag("session_id", black_box("abc123"))
                    .tag("bytes_sent", black_box(4096i64))
                    .tag("duration_ms", black_box(12i64))
                    .emit();
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_logs_tracing(criterion: &mut Criterion) {
    // tracing-subscriber with fmt layer wired (not no-op) — same shape as C9 bench.
    // tracing-log bridge is implicit when tracing::info! is used directly.
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink));
    let _guard = subscriber.set_default();

    let mut group = criterion.benchmark_group("e2e_composition_logs");
    group.bench_function("tracing_logs", |bencher| {
        bencher.iter(|| {
            tracing::info!(
                user_id = black_box(1001i64),
                session_id = black_box("abc123"),
                bytes_sent = black_box(4096i64),
                duration_ms = black_box(12i64),
                "request processed"
            );
        });
    });
    group.finish();
}

fn bench_logs_log_crate(criterion: &mut Criterion) {
    // log crate wired with env_logger writing to sink — real subscriber, not no-op.
    // log::Log dispatches to the registered global logger.
    struct SinkLogger;

    impl log::Log for SinkLogger {
        fn enabled(&self, meta: &log::Metadata<'_>) -> bool {
            meta.level() <= log::Level::Info
        }
        fn log(&self, record: &log::Record<'_>) {
            black_box(record.args());
        }
        fn flush(&self) {}
    }

    static SINK_LOGGER: SinkLogger = SinkLogger;
    // set_logger returns Err if already set; fine for bench re-runs.
    let _ = log::set_logger(&SINK_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let mut group = criterion.benchmark_group("e2e_composition_logs");
    group.bench_function("log_crate", |bencher| {
        bencher.iter(|| {
            log::info!(
                "request processed; user_id={}; session_id={}; bytes_sent={}; duration_ms={}",
                black_box(1001i64),
                black_box("abc123"),
                black_box(4096i64),
                black_box(12i64),
            );
        });
    });
    group.finish();
}

// ── pair 4: full 5-signal composition ───────────────────────────────────────

fn bench_full_proxima(criterion: &mut Criterion) {
    use proxima_telemetry::config::TelemetryConfig;
    use proxima_telemetry::metric::Histogram;
    use proxima_telemetry::pipes::InMemoryPipe;
    use proxima_telemetry::recorder::Recorder;
    use proxima_telemetry::tag::{ScalarValue, Tag};

    static HIST_FULL: Histogram<f64> = Histogram::new("http.latency.full");

    // from_config applies production defaults: sampler=AlwaysOn, record_sharing=Arc.
    // with_pipe overrides the Noop default exporter with InMemoryPipe so traces are
    // stored (matched to OTel InMemorySpanExporter on the incumbent side).
    let recorder = Recorder::from_config(&TelemetryConfig::default())
        .with_pipe(InMemoryPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench");

    let metric_tags = [
        Tag::Scalar {
            key: "route",
            value: ScalarValue::Str(ROUTE_LABEL),
        },
        Tag::Scalar {
            key: "method",
            value: ScalarValue::Str(METHOD_LABEL),
        },
    ];

    static BOUNDS: &[f64] = &[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0];

    let mut group = criterion.benchmark_group("e2e_composition_full");
    group.bench_function("proxima_full", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                // 1 span + 8 attrs + 2 events
                let mut guard = recorder
                    .span(black_box("process_request"))
                    .tag("http.method", black_box("POST"))
                    .tag("http.route", black_box("/v1/upload"))
                    .tag("http.scheme", black_box("https"))
                    .tag("net.host.name", black_box("api.example.com"))
                    .tag("net.host.port", black_box(443i64))
                    .tag("user_agent", black_box("bench/1.0"))
                    .tag("http.status_code", black_box(200i64))
                    .tag("response.size_bytes", black_box(4096i64))
                    .start();

                if let Some(evt) = guard.event(black_box("validated")) {
                    evt.emit();
                }
                if let Some(evt) = guard.event(black_box("dispatched")) {
                    evt.emit();
                }

                // 1 counter + 1 histogram
                recorder.emit_counter_add(
                    black_box("http.requests"),
                    black_box(1u64),
                    black_box(&metric_tags),
                );
                black_box(&HIST_FULL).record(black_box(0.0123f64));
                let snap = HIST_FULL.bucket_snapshot();
                let count = HIST_FULL.count();
                let sum = HIST_FULL.sum();
                recorder.emit_histogram_record(
                    black_box("http.latency.full"),
                    black_box(count),
                    black_box(sum),
                    snap.to_vec(),
                    BOUNDS,
                    black_box(&metric_tags),
                );

                // 1 log + 4 fields
                recorder
                    .log()
                    .message(black_box("request processed"))
                    .tag("user_id", black_box(1001i64))
                    .tag("session_id", black_box("abc123"))
                    .tag("bytes_sent", black_box(4096i64))
                    .tag("duration_ms", black_box(12i64))
                    .emit();

                black_box(&guard);
                drop(guard);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_full_four_crate_stack(criterion: &mut Criterion) {
    // Composed 4-crate stack: opentelemetry_sdk (traces) + prometheus (metrics) + tracing (logs).
    // Same 5-signal workload as proxima_full.
    use opentelemetry::KeyValue;
    use opentelemetry::trace::{Span, Tracer, TracerProvider};
    use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    let counter = prometheus::IntCounterVec::new(
        prometheus::Opts::new("full_requests_total", "total request count"),
        &["route", "method"],
    )
    .expect("valid opts");
    let histogram = prometheus::HistogramVec::new(
        prometheus::HistogramOpts::new("full_latency_seconds", "request latency"),
        &["route", "method"],
    )
    .expect("valid opts");

    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink));
    let _log_guard = subscriber.set_default();

    let mut group = criterion.benchmark_group("e2e_composition_full");
    group.bench_function("four_crate_stack_full", |bencher| {
        bencher.iter(|| {
            // trace: 1 span + 8 attrs + 2 events
            let mut span = tracer.start(black_box("process_request"));
            span.set_attribute(KeyValue::new("http.method", black_box("POST")));
            span.set_attribute(KeyValue::new("http.route", black_box("/v1/upload")));
            span.set_attribute(KeyValue::new("http.scheme", black_box("https")));
            span.set_attribute(KeyValue::new("net.host.name", black_box("api.example.com")));
            span.set_attribute(KeyValue::new("net.host.port", black_box(443i64)));
            span.set_attribute(KeyValue::new("user_agent", black_box("bench/1.0")));
            span.set_attribute(KeyValue::new("http.status_code", black_box(200i64)));
            span.set_attribute(KeyValue::new("response.size_bytes", black_box(4096i64)));
            span.add_event("validated", vec![]);
            span.add_event("dispatched", vec![]);
            span.end();
            black_box(span);

            // metrics: 1 counter + 1 histogram
            black_box(&counter)
                .with_label_values(&[black_box(ROUTE_LABEL), black_box(METHOD_LABEL)])
                .inc();
            black_box(&histogram)
                .with_label_values(&[black_box(ROUTE_LABEL), black_box(METHOD_LABEL)])
                .observe(black_box(0.0123));

            // log: 1 INFO + 4 fields via tracing
            tracing::info!(
                user_id = black_box(1001i64),
                session_id = black_box("abc123"),
                bytes_sent = black_box(4096i64),
                duration_ms = black_box(12i64),
                "request processed"
            );
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_traces_proxima,
    bench_traces_opentelemetry_sdk,
    bench_metrics_proxima,
    bench_metrics_prometheus,
    bench_metrics_crate,
    bench_logs_proxima,
    bench_logs_tracing,
    bench_logs_log_crate,
    bench_full_proxima,
    bench_full_four_crate_stack,
);
criterion_main!(benches);
