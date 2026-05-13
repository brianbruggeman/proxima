#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::id::{SpanId, TraceFlags, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::LogRecord;
use proxima_telemetry::log::body::LogBody;
use proxima_telemetry::metric::sample::{MetricSample, NumberDataPoint};
use proxima_telemetry::out::otlp_grpc::{encode_grpc_framed, frame_grpc};
use proxima_telemetry::out::otlp_http::conv::{log_to_proto, metric_to_proto, span_to_proto};
use proxima_telemetry::out::otlp_http::proto::{
    ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest, ResourceLogs,
    ResourceMetrics, ResourceSpans, ScopeLogs, ScopeMetrics, ScopeSpans,
};
use proxima_telemetry::tag::{ScalarValue, Tag};
use proxima_telemetry::trace::SpanRecord;
use proxima_telemetry::trace::kind::SpanKind;
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
        attrs: (0..8u64)
            .map(|index| Tag::Scalar {
                key: "http.status_code",
                value: ScalarValue::U64(index),
            })
            .collect(),
        events: Default::default(),
        links: Default::default(),
        tracestate: TraceState::empty(),
        module_path: "proxima::bench",
        file_line: (1, 1),
    }
}

fn make_log_4attr() -> LogRecord {
    LogRecord {
        ts_ns: 1_700_000_000_000_000_000,
        observed_ts_ns: 1_700_000_000_000_000_000,
        level: Level::INFO,
        body: LogBody::Text("bench log"),
        attrs: (0..4u64)
            .map(|index| Tag::Scalar {
                key: "request.id",
                value: ScalarValue::I64(index as i64),
            })
            .collect(),
        trace_id: None,
        span_id: None,
        trace_flags: TraceFlags::SAMPLED,
        module_path: "proxima::bench",
        file_line: (2, 1),
    }
}

fn bench_proxima_grpc_encode_span_8attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c11_otlp_grpc");
    let record = make_span_8attr();

    group.bench_function("proxima_grpc_encode_span_8attr", |bencher| {
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
            black_box(encode_grpc_framed(&request).unwrap())
        });
    });
    group.finish();
}

fn bench_proxima_grpc_encode_log_4attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c11_otlp_grpc");
    let record = make_log_4attr();

    group.bench_function("proxima_grpc_encode_log_4attr", |bencher| {
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
            black_box(encode_grpc_framed(&request).unwrap())
        });
    });
    group.finish();
}

fn bench_proxima_grpc_encode_counter(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c11_otlp_grpc");
    let sample = MetricSample::Counter(NumberDataPoint {
        value: ScalarValue::U64(1),
        attrs: Default::default(),
        ts_ns: 0,
        start_ts_ns: 0,
    });

    group.bench_function("proxima_grpc_encode_counter", |bencher| {
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
            black_box(encode_grpc_framed(&request).unwrap())
        });
    });
    group.finish();
}

// Fair-comparison arm: proxima encode + frame ONLY, with the request pre-built
// outside the iteration loop. Two variants:
//
// (a) `proxima_grpc_encode_prebuilt_v1` uses `frame_grpc(Bytes)` — the original
//     path, two allocations (prost body Vec + framed Vec) plus two `Bytes::from`
//     wraps. Apples-to-apples but slightly worse than home-turf due to the
//     Bytes wraps.
//
// (b) `proxima_grpc_encode_prebuilt_v2` uses `encode_grpc_framed(&request)` —
//     SINGLE-allocation backpatch: reserve 5-byte header, encode appends body,
//     backpatch the length. Same allocation pattern as the home-turf arm.
//     This is the meet-or-beat arm.
fn bench_proxima_grpc_encode_prebuilt_v1(criterion: &mut Criterion) {
    use bytes::Bytes;
    use prost::Message;

    let mut group = criterion.benchmark_group("c11_otlp_grpc");

    let record = make_span_8attr();
    let proto_span = span_to_proto(&record);
    let request = ExportTraceServiceRequest {
        resource_spans: alloc::vec![ResourceSpans {
            resource: None,
            scope_spans: alloc::vec![ScopeSpans {
                scope: None,
                spans: alloc::vec![proto_span],
                schema_url: alloc::string::String::new(),
            }],
            schema_url: alloc::string::String::new(),
        }],
    };

    group.bench_function("proxima_grpc_encode_prebuilt_v1_bytes_wrap", |bencher| {
        bencher.iter(|| {
            let mut buf = alloc::vec::Vec::with_capacity(256);
            request.encode(&mut buf).unwrap();
            let body = Bytes::from(buf);
            black_box(frame_grpc(black_box(body)))
        });
    });
    group.finish();
}

fn bench_proxima_grpc_encode_prebuilt_v2(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c11_otlp_grpc");

    let record = make_span_8attr();
    let proto_span = span_to_proto(&record);
    let request = ExportTraceServiceRequest {
        resource_spans: alloc::vec![ResourceSpans {
            resource: None,
            scope_spans: alloc::vec![ScopeSpans {
                scope: None,
                spans: alloc::vec![proto_span],
                schema_url: alloc::string::String::new(),
            }],
            schema_url: alloc::string::String::new(),
        }],
    };

    group.bench_function("proxima_grpc_encode_prebuilt_v2_backpatch", |bencher| {
        bencher.iter(|| black_box(encode_grpc_framed(black_box(&request)).unwrap()));
    });
    group.finish();
}

// Home-turf arm: opentelemetry-proto's tonic-derived types + prost::encode +
// manual gRPC framing. This is identical to what tonic produces on the wire in
// production — pulling the actual `tonic` crate would dwarf the dev-dep tree
// without changing the encode work measured here.
fn bench_opentelemetry_otlp_grpc_encode_span_8attr(criterion: &mut Criterion) {
    use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value};
    use opentelemetry_proto::tonic::trace::v1::{ResourceSpans, ScopeSpans, Span};
    use prost::Message as _;

    let mut group = criterion.benchmark_group("c11_otlp_grpc");

    let attrs: alloc::vec::Vec<KeyValue> = (0..8u64)
        .map(|index| KeyValue {
            key: "http.status_code".to_string(),
            value: Some(AnyValue {
                value: Some(any_value::Value::IntValue(index as i64)),
            }),
            ..Default::default()
        })
        .collect();

    let span = Span {
        trace_id: alloc::vec![0xab; 16],
        span_id: alloc::vec![0xcd; 8],
        name: "bench.span".to_string(),
        kind: opentelemetry_proto::tonic::trace::v1::span::SpanKind::Server as i32,
        start_time_unix_nano: 1_700_000_000_000_000_000,
        end_time_unix_nano: 1_700_000_001_234_567,
        attributes: attrs,
        ..Default::default()
    };

    let request = ExportTraceServiceRequest {
        resource_spans: alloc::vec![ResourceSpans {
            scope_spans: alloc::vec![ScopeSpans {
                spans: alloc::vec![span],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };

    group.bench_function("opentelemetry_otlp_grpc_encode_span_8attr", |bencher| {
        bencher.iter(|| {
            let mut buf = alloc::vec::Vec::with_capacity(256);
            request.encode(&mut buf).unwrap();
            // manual gRPC frame: same as tonic produces on the wire
            let body_len = buf.len() as u32;
            let mut framed = alloc::vec::Vec::with_capacity(5 + buf.len());
            framed.push(0u8);
            framed.extend_from_slice(&body_len.to_be_bytes());
            framed.extend_from_slice(&buf);
            black_box(framed)
        });
    });
    group.finish();
}

// Isolates the gRPC framing overhead from the protobuf encoding cost.
// `frame_grpc` over a pre-encoded body shows pure framing time.
fn bench_grpc_frame_overhead(criterion: &mut Criterion) {
    use bytes::Bytes;

    let mut group = criterion.benchmark_group("c11_otlp_grpc");
    // 260 B matches the typical otlp-http body for an 8-attr span (per C10 measurements)
    let body = Bytes::from(alloc::vec![0u8; 260]);

    group.bench_function("grpc_frame_overhead_260B", |bencher| {
        bencher.iter(|| black_box(frame_grpc(black_box(body.clone()))));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_grpc_encode_span_8attr,
    bench_proxima_grpc_encode_log_4attr,
    bench_proxima_grpc_encode_counter,
    bench_proxima_grpc_encode_prebuilt_v1,
    bench_proxima_grpc_encode_prebuilt_v2,
    bench_opentelemetry_otlp_grpc_encode_span_8attr,
    bench_grpc_frame_overhead,
);
criterion_main!(benches);
