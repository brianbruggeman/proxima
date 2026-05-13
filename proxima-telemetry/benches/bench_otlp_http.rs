#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use prost::Message;
use std::hint::black_box;

use proxima_telemetry::id::{SpanId, TraceFlags, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::LogRecord;
use proxima_telemetry::log::body::LogBody;
#[cfg(feature = "histogram")]
use proxima_telemetry::metric::sample::HistogramDataPoint;
use proxima_telemetry::metric::sample::{MetricSample, NumberDataPoint};
use proxima_telemetry::out::otlp_http::conv::{log_to_proto, metric_to_proto, span_to_proto};
use proxima_telemetry::out::otlp_http::proto::{
    ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest, ResourceLogs,
    ResourceMetrics, ResourceSpans, ScopeLogs, ScopeMetrics, ScopeSpans,
};
use proxima_telemetry::tag::{ScalarValue, Tag};
use proxima_telemetry::trace::kind::SpanKind;
use proxima_telemetry::trace::span::SpanRecord;
use proxima_telemetry::trace::status::Status;
use proxima_telemetry::trace::tracestate::TraceState;

extern crate alloc;

fn make_span_8attr() -> SpanRecord {
    SpanRecord {
        trace_id: TraceId::from_bytes([0xab; 16]),
        span_id: SpanId::from_bytes([0xcd; 8]),
        parent_span_id: None,
        name: "bench.span",
        kind: SpanKind::Server,
        start_ns: 1_700_000_000_000_000_000,
        duration_ns: 1_234_567,
        status: Status::Ok,
        attrs: smallvec::smallvec![
            Tag::Scalar {
                key: "http.method",
                value: ScalarValue::Str("POST")
            },
            Tag::Scalar {
                key: "http.status_code",
                value: ScalarValue::I64(200)
            },
            Tag::Scalar {
                key: "http.url",
                value: ScalarValue::Str("https://example.com/api"),
            },
            Tag::Scalar {
                key: "net.peer.port",
                value: ScalarValue::I64(443)
            },
            Tag::Scalar {
                key: "sampled",
                value: ScalarValue::Bool(true)
            },
            Tag::Scalar {
                key: "retry.count",
                value: ScalarValue::I64(0)
            },
            Tag::Scalar {
                key: "latency.ms",
                value: ScalarValue::F64(1.234)
            },
            Tag::Scalar {
                key: "region",
                value: ScalarValue::Str("us-east-1")
            },
        ],
        events: Default::default(),
        links: Default::default(),
        tracestate: TraceState::empty(),
        module_path: "bench",
        file_line: (0, 0),
    }
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

// Original arm — Vec::with_capacity(256) under-sizes for our 264 B output, so prost::encode
// forces a reallocation inside the hot loop. Kept as audit-trail.
fn bench_encode_span_8attr(criterion: &mut Criterion) {
    let record = make_span_8attr();
    criterion.bench_function("proxima_otlp_encode_span_8attr", |bencher| {
        bencher.iter(|| {
            let span = span_to_proto(black_box(&record));
            let request = ExportTraceServiceRequest {
                resource_spans: alloc::vec![ResourceSpans {
                    resource: None,
                    scope_spans: alloc::vec![ScopeSpans {
                        scope: None,
                        spans: alloc::vec![span],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(256);
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

// Meet-or-beat arm: exact-sized buffer via encoded_len() — no reallocation during encode.
// Mirrors the v2 backpatch approach from C11 (which beat the incumbent home-turf arm by 10.3%).
fn bench_encode_span_8attr_exact_size(criterion: &mut Criterion) {
    use prost::Message;
    let record = make_span_8attr();
    criterion.bench_function("proxima_otlp_encode_span_8attr_exact_size", |bencher| {
        bencher.iter(|| {
            let span = span_to_proto(black_box(&record));
            let request = ExportTraceServiceRequest {
                resource_spans: alloc::vec![ResourceSpans {
                    resource: None,
                    scope_spans: alloc::vec![ScopeSpans {
                        scope: None,
                        spans: alloc::vec![span],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(request.encoded_len());
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

fn bench_encode_log_4attr(criterion: &mut Criterion) {
    let record = make_log_4attr();
    criterion.bench_function("proxima_otlp_encode_log_4attr", |bencher| {
        bencher.iter(|| {
            let log = log_to_proto(black_box(&record));
            let request = ExportLogsServiceRequest {
                resource_logs: alloc::vec![ResourceLogs {
                    resource: None,
                    scope_logs: alloc::vec![ScopeLogs {
                        scope: None,
                        log_records: alloc::vec![log],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(128);
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

fn bench_encode_counter(criterion: &mut Criterion) {
    let sample = MetricSample::Counter(NumberDataPoint {
        value: ScalarValue::U64(1024),
        attrs: smallvec::smallvec![Tag::Scalar {
            key: "host",
            value: ScalarValue::Str("node-1")
        }],
        ts_ns: 1_700_000_000_000_000_003,
        start_ts_ns: 1_700_000_000_000_000_000,
    });
    criterion.bench_function("proxima_otlp_encode_counter", |bencher| {
        bencher.iter(|| {
            let metric = metric_to_proto(black_box(&sample));
            let request = ExportMetricsServiceRequest {
                resource_metrics: alloc::vec![ResourceMetrics {
                    resource: None,
                    scope_metrics: alloc::vec![ScopeMetrics {
                        scope: None,
                        metrics: alloc::vec![metric],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(64);
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

#[cfg(feature = "histogram")]
fn bench_encode_histogram(criterion: &mut Criterion) {
    static BOUNDS: &[f64] = &[1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
    let sample = MetricSample::Histogram(HistogramDataPoint {
        count: 1000,
        sum: 123_456.0,
        bucket_counts: alloc::vec![10, 50, 200, 400, 200, 100, 30, 10],
        bounds: BOUNDS,
        attrs: smallvec::smallvec![Tag::Scalar {
            key: "host",
            value: ScalarValue::Str("node-1")
        }],
        ts_ns: 1_700_000_000_000_000_004,
        start_ts_ns: 1_700_000_000_000_000_000,
    });
    criterion.bench_function("proxima_otlp_encode_histogram", |bencher| {
        bencher.iter(|| {
            let metric = metric_to_proto(black_box(&sample));
            let request = ExportMetricsServiceRequest {
                resource_metrics: alloc::vec![ResourceMetrics {
                    resource: None,
                    scope_metrics: alloc::vec![ScopeMetrics {
                        scope: None,
                        metrics: alloc::vec![metric],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(128);
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

#[cfg(not(feature = "histogram"))]
fn bench_encode_histogram(_criterion: &mut Criterion) {}

fn build_otel_span_8attr() -> opentelemetry_proto::tonic::trace::v1::Span {
    use opentelemetry_proto::tonic::common::v1::any_value::Value as OtelValue;
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue};
    use opentelemetry_proto::tonic::trace::v1::Span as OtelSpan;

    OtelSpan {
        trace_id: [0xab; 16].to_vec(),
        span_id: [0xcd; 8].to_vec(),
        parent_span_id: alloc::vec![],
        trace_state: alloc::string::String::new(),
        flags: 0,
        name: alloc::string::String::from("bench.span"),
        kind: 2,
        start_time_unix_nano: 1_700_000_000_000_000_000,
        end_time_unix_nano: 1_700_000_001_234_567,
        attributes: alloc::vec![
            KeyValue {
                key: alloc::string::String::from("http.method"),
                value: Some(AnyValue {
                    value: Some(OtelValue::StringValue(alloc::string::String::from("POST"))),
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("http.status_code"),
                value: Some(AnyValue {
                    value: Some(OtelValue::IntValue(200))
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("http.url"),
                value: Some(AnyValue {
                    value: Some(OtelValue::StringValue(alloc::string::String::from(
                        "https://example.com/api",
                    ))),
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("net.peer.port"),
                value: Some(AnyValue {
                    value: Some(OtelValue::IntValue(443))
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("sampled"),
                value: Some(AnyValue {
                    value: Some(OtelValue::BoolValue(true))
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("retry.count"),
                value: Some(AnyValue {
                    value: Some(OtelValue::IntValue(0))
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("latency.ms"),
                value: Some(AnyValue {
                    value: Some(OtelValue::DoubleValue(1.234))
                }),
                ..Default::default()
            },
            KeyValue {
                key: alloc::string::String::from("region"),
                value: Some(AnyValue {
                    value: Some(OtelValue::StringValue(alloc::string::String::from(
                        "us-east-1",
                    ))),
                }),
                ..Default::default()
            },
        ],
        dropped_attributes_count: 0,
        events: alloc::vec![],
        dropped_events_count: 0,
        links: alloc::vec![],
        dropped_links_count: 0,
        status: None,
    }
}

fn bench_otel_encode_span_8attr(criterion: &mut Criterion) {
    use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest as OtelReq;
    use opentelemetry_proto::tonic::trace::v1::{ResourceSpans as OtelRS, ScopeSpans as OtelSS};

    criterion.bench_function("opentelemetry_otlp_encode_span_8attr", |bencher| {
        bencher.iter(|| {
            let span = black_box(build_otel_span_8attr());
            let request = OtelReq {
                resource_spans: alloc::vec![OtelRS {
                    resource: None,
                    scope_spans: alloc::vec![OtelSS {
                        scope: None,
                        spans: alloc::vec![span],
                        schema_url: alloc::string::String::new(),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
            };
            let mut buf = alloc::vec::Vec::with_capacity(256);
            request.encode(&mut buf).unwrap_or(());
            black_box(buf)
        });
    });
}

fn bench_output_size(criterion: &mut Criterion) {
    use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest as OtelReq;
    use opentelemetry_proto::tonic::trace::v1::{ResourceSpans as OtelRS, ScopeSpans as OtelSS};

    let record = make_span_8attr();
    let span = span_to_proto(&record);
    let proxima_request = ExportTraceServiceRequest {
        resource_spans: alloc::vec![ResourceSpans {
            resource: None,
            scope_spans: alloc::vec![ScopeSpans {
                scope: None,
                spans: alloc::vec![span],
                schema_url: alloc::string::String::new(),
            }],
            schema_url: alloc::string::String::new(),
        }],
    };
    let mut proxima_buf = alloc::vec::Vec::new();
    proxima_request.encode(&mut proxima_buf).unwrap_or(());
    let proxima_size = proxima_buf.len();

    let otel_request = OtelReq {
        resource_spans: alloc::vec![OtelRS {
            resource: None,
            scope_spans: alloc::vec![OtelSS {
                scope: None,
                spans: alloc::vec![build_otel_span_8attr()],
                schema_url: alloc::string::String::new(),
            }],
            schema_url: alloc::string::String::new(),
        }],
    };
    let mut otel_buf = alloc::vec::Vec::new();
    otel_request.encode(&mut otel_buf).unwrap_or(());
    let otel_size = otel_buf.len();

    let mut group = criterion.benchmark_group("output_bytes_per_span");
    group.bench_function(BenchmarkId::new("proxima", proxima_size), |bencher| {
        bencher.iter(|| black_box(proxima_size))
    });
    group.bench_function(BenchmarkId::new("opentelemetry", otel_size), |bencher| {
        bencher.iter(|| black_box(otel_size))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_span_8attr,
    bench_encode_span_8attr_exact_size,
    bench_encode_log_4attr,
    bench_encode_counter,
    bench_encode_histogram,
    bench_otel_encode_span_8attr,
    bench_output_size,
);
criterion_main!(benches);
