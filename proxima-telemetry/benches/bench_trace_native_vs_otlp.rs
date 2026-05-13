#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Head-to-head: proxima native tracing vs OTLP for the same 8-attr span.
//!
//! Two fair comparisons on identical input:
//!   - encode-only: SpanRecord -> wire bytes (native postcard frame vs OTLP
//!     protobuf with an exact-sized buffer — OTLP's best case, no realloc).
//!   - emit->drain: a recorder span with 8 tags emitted and drained through the
//!     terminal pipe (native frame sink vs OTLP buffer). This is the realistic
//!     per-span cost of "tracing".
//!
//! The claim under test: native beats OTLP on both. Numbers land in
//! docs/tracing/discipline.md.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use prost::Message;

use proxima_telemetry::id::{SpanId, TraceFlags, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::LogRecord;
use proxima_telemetry::log::body::LogBody;
use proxima_telemetry::metric::{MetricSample, NumberDataPoint};
use proxima_telemetry::out::native::{
    FrameSink, NATIVE_FRAME_SIZE, NativeExporter, NativePayload, NativePayloadRef,
    log_to_native_ref, metric_to_native_ref, span_to_native, span_to_native_ref,
};
use proxima_telemetry::out::otlp_http::conv::{log_to_proto, metric_to_proto, span_to_proto};
use proxima_telemetry::out::otlp_http::proto::{
    ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest, ResourceLogs,
    ResourceMetrics, ResourceSpans, ScopeLogs, ScopeMetrics, ScopeSpans,
};
use proxima_telemetry::pipes::{NativePipe, OtlpHttpPipe};
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};
use proxima_telemetry::trace::status::Status;
use proxima_telemetry::trace::{SpanKind, SpanRecord, TraceState};

struct NullSink;
impl FrameSink for NullSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
    }
}

fn make_span_8attr() -> SpanRecord {
    let attrs = (0..8)
        .map(|index| Tag::Scalar {
            key: "http.status_code",
            value: ScalarValue::U64(index as u64),
        })
        .collect();
    SpanRecord {
        trace_id: TraceId::from_bytes([0xab; 16]),
        span_id: SpanId::from_bytes([0xcd; 8]),
        parent_span_id: Some(SpanId::from_bytes([0xef; 8])),
        name: "bench.span",
        kind: SpanKind::Server,
        start_ns: 1_700_000_000_000_000_000,
        duration_ns: 12_500_000,
        status: Status::Ok,
        attrs,
        events: Default::default(),
        links: Default::default(),
        tracestate: TraceState::empty(),
        module_path: "proxima::bench",
        file_line: (1, 1),
    }
}

fn emit_one_span(recorder: &Recorder) {
    let guard = recorder
        .span("bench.span")
        .kind(SpanKind::Server)
        .tag("a0", 0u64)
        .tag("a1", 1u64)
        .tag("a2", 2u64)
        .tag("a3", 3u64)
        .tag("a4", 4u64)
        .tag("a5", 5u64)
        .tag("a6", 6u64)
        .tag("a7", 7u64)
        .start();
    drop(guard);
}

fn bench_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("trace_native_vs_otlp");
    let record = make_span_8attr();

    let native = NativeExporter::new(NullSink);
    // the production drain path (dispatch_native): zero-alloc borrowed encode.
    group.bench_function("native_encode_span_8attr", |bencher| {
        bencher.iter(|| {
            let payload = span_to_native_ref(black_box(&record));
            native.encode_and_emit_payload_ref(NativePayloadRef::Span(payload));
        });
    });
    // the prior owned path, kept to show the zero-alloc optimization delta.
    group.bench_function("native_encode_span_8attr_owned", |bencher| {
        bencher.iter(|| {
            let payload = span_to_native(black_box(&record));
            native.encode_and_emit_payload(NativePayload::Span(payload));
        });
    });

    group.bench_function("otlp_encode_span_8attr_exact", |bencher| {
        bencher.iter(|| {
            let span = span_to_proto(black_box(&record));
            let request = ExportTraceServiceRequest {
                resource_spans: vec![ResourceSpans {
                    resource: None,
                    scope_spans: vec![ScopeSpans {
                        scope: None,
                        spans: vec![span],
                        schema_url: String::new(),
                    }],
                    schema_url: String::new(),
                }],
            };
            let mut buffer = Vec::with_capacity(request.encoded_len());
            request.encode(&mut buffer).unwrap_or(());
            black_box(buffer)
        });
    });

    group.finish();
}

fn bench_emit_drain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("trace_native_vs_otlp");

    let native_recorder = Recorder::builder()
        .pipe(NativePipe::new(NullSink))
        .core_count(1)
        .start()
        .expect("native recorder");
    group.bench_function("native_emit_drain_span_8attr", |bencher| {
        bencher.iter(|| {
            emit_one_span(&native_recorder);
            black_box(native_recorder.drain());
        });
    });

    let otlp_recorder = Recorder::builder()
        .pipe(OtlpHttpPipe::new("http://127.0.0.1:4318/v1/traces"))
        .core_count(1)
        .start()
        .expect("otlp recorder");
    group.bench_function("otlp_emit_drain_span_8attr", |bencher| {
        bencher.iter(|| {
            emit_one_span(&otlp_recorder);
            black_box(otlp_recorder.drain());
        });
    });

    // amortized: emit 64 spans, drain once — divides out the per-drain fixed cost
    // so the per-span throughput (and the native-vs-otlp delta) is visible.
    group.bench_function("native_emit64_drain1_span_8attr", |bencher| {
        bencher.iter(|| {
            for _ in 0..64 {
                emit_one_span(&native_recorder);
            }
            black_box(native_recorder.drain());
        });
    });
    group.bench_function("otlp_emit64_drain1_span_8attr", |bencher| {
        bencher.iter(|| {
            for _ in 0..64 {
                emit_one_span(&otlp_recorder);
            }
            black_box(otlp_recorder.drain());
        });
    });

    group.finish();
}

fn make_log_4attr() -> LogRecord {
    LogRecord {
        ts_ns: 1_700_000_000_000_000_001,
        observed_ts_ns: 1_700_000_000_000_000_002,
        level: Level::INFO,
        body: LogBody::Text("request handled"),
        attrs: smallvec::smallvec![
            Tag::Scalar {
                key: "service",
                value: ScalarValue::Str("api")
            },
            Tag::Scalar {
                key: "version",
                value: ScalarValue::Str("1.0.0")
            },
            Tag::Scalar {
                key: "request_id",
                value: ScalarValue::I64(42)
            },
            Tag::Scalar {
                key: "duration_ms",
                value: ScalarValue::F64(1.5)
            },
        ],
        trace_id: None,
        span_id: None,
        trace_flags: TraceFlags(0),
        module_path: "bench",
        file_line: (0, 0),
    }
}

fn make_counter() -> MetricSample {
    MetricSample::Counter(NumberDataPoint {
        value: ScalarValue::U64(1024),
        attrs: smallvec::smallvec![Tag::Scalar {
            key: "host",
            value: ScalarValue::Str("node-1")
        }],
        ts_ns: 1_700_000_000_000_000_003,
        start_ts_ns: 1_700_000_000_000_000_000,
    })
}

// the zero-alloc borrowed encode benefits logs and metrics too (shared
// tags_to_native_ref) — prove the win generalises past spans.
fn bench_encode_log_metric(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("trace_native_vs_otlp");
    let native = NativeExporter::new(NullSink);

    let log = make_log_4attr();
    group.bench_function("native_encode_log_4attr", |bencher| {
        bencher.iter(|| {
            native.encode_and_emit_payload_ref(NativePayloadRef::Log(log_to_native_ref(
                black_box(&log),
            )));
        });
    });
    group.bench_function("otlp_encode_log_4attr_exact", |bencher| {
        bencher.iter(|| {
            let proto = log_to_proto(black_box(&log));
            let request = ExportLogsServiceRequest {
                resource_logs: vec![ResourceLogs {
                    resource: None,
                    scope_logs: vec![ScopeLogs {
                        scope: None,
                        log_records: vec![proto],
                        schema_url: String::new(),
                    }],
                    schema_url: String::new(),
                }],
            };
            let mut buffer = Vec::with_capacity(request.encoded_len());
            request.encode(&mut buffer).unwrap_or(());
            black_box(buffer)
        });
    });

    let counter = make_counter();
    group.bench_function("native_encode_counter", |bencher| {
        bencher.iter(|| {
            native.encode_and_emit_payload_ref(NativePayloadRef::Metric(metric_to_native_ref(
                black_box(&counter),
            )));
        });
    });
    group.bench_function("otlp_encode_counter_exact", |bencher| {
        bencher.iter(|| {
            let proto = metric_to_proto(black_box(&counter));
            let request = ExportMetricsServiceRequest {
                resource_metrics: vec![ResourceMetrics {
                    resource: None,
                    scope_metrics: vec![ScopeMetrics {
                        scope: None,
                        metrics: vec![proto],
                        schema_url: String::new(),
                    }],
                    schema_url: String::new(),
                }],
            };
            let mut buffer = Vec::with_capacity(request.encoded_len());
            request.encode(&mut buffer).unwrap_or(());
            black_box(buffer)
        });
    });

    group.finish();
}

criterion_group!(
    trace_native_vs_otlp,
    bench_encode,
    bench_emit_drain,
    bench_encode_log_metric
);
criterion_main!(trace_native_vs_otlp);
