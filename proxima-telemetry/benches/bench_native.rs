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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::id::{SpanId, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::LogRecord;
use proxima_telemetry::log::body::LogBody;
use proxima_telemetry::metric::{MetricSample, NumberDataPoint};
use proxima_telemetry::out::native::{
    FrameSink, NATIVE_FRAME_SIZE, NativeExporter, NativeFrame, NativePayload, log_to_native,
    metric_to_native, span_to_native,
};
use proxima_telemetry::tag::{ScalarValue, Tag};
use proxima_telemetry::trace::status::Status;
use proxima_telemetry::trace::{SpanKind, SpanRecord, TraceState};

struct NullSink;

impl FrameSink for NullSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
    }
}

#[allow(dead_code)]
struct CountSink(Arc<AtomicUsize>);

#[allow(dead_code)]
impl FrameSink for CountSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn make_span(attr_count: usize) -> SpanRecord {
    let attrs = (0..attr_count)
        .map(|index| Tag::Scalar {
            key: "http.status_code",
            value: ScalarValue::U64(index as u64),
        })
        .collect();
    SpanRecord {
        trace_id: TraceId::from_bytes([0xabu8; 16]),
        span_id: SpanId::from_bytes([0xcdu8; 8]),
        parent_span_id: Some(SpanId::from_bytes([0xefu8; 8])),
        name: "bench_operation",
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

fn make_log(attr_count: usize) -> LogRecord {
    let attrs = (0..attr_count)
        .map(|index| Tag::Scalar {
            key: "request.id",
            value: ScalarValue::I64(index as i64),
        })
        .collect();
    LogRecord {
        ts_ns: 1_700_000_000_000_000_000,
        observed_ts_ns: 1_700_000_000_000_001_000,
        level: Level::INFO,
        body: LogBody::Text("request processed"),
        attrs,
        trace_id: None,
        span_id: None,
        trace_flags: proxima_telemetry::id::TraceFlags::SAMPLED,
        module_path: "proxima::bench",
        file_line: (2, 1),
    }
}

fn bench_proxima_native_encode_span_8attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c12_native");
    let exporter = NativeExporter::new(NullSink);
    let record = make_span(8);

    group.bench_function("proxima_native_encode_span_8attr", |bencher| {
        bencher.iter(|| {
            let native = span_to_native(black_box(&record));
            exporter.encode_and_emit_payload(NativePayload::Span(native));
        });
    });
    group.finish();
}

fn bench_proxima_native_encode_log_4attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c12_native");
    let exporter = NativeExporter::new(NullSink);
    let record = make_log(4);

    group.bench_function("proxima_native_encode_log_4attr", |bencher| {
        bencher.iter(|| {
            let native = log_to_native(black_box(&record));
            exporter.encode_and_emit_payload(NativePayload::Log(native));
        });
    });
    group.finish();
}

fn bench_proxima_native_encode_metric_counter(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c12_native");
    let exporter = NativeExporter::new(NullSink);
    let sample = MetricSample::Counter(NumberDataPoint {
        value: ScalarValue::U64(1),
        attrs: Default::default(),
        ts_ns: 1_000_000,
        start_ts_ns: 0,
    });

    group.bench_function("proxima_native_encode_metric_counter", |bencher| {
        bencher.iter(|| {
            let native = metric_to_native(black_box(&sample));
            exporter.encode_and_emit_payload(NativePayload::Metric(native));
        });
    });
    group.finish();
}

#[cfg(feature = "histogram")]
fn bench_proxima_native_encode_metric_histogram(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c12_native");
    let exporter = NativeExporter::new(NullSink);

    static BOUNDS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
    let sample = MetricSample::Histogram(proxima_telemetry::metric::HistogramDataPoint {
        count: 1000,
        sum: 15432.0,
        bucket_counts: alloc::vec![10, 20, 50, 100, 200, 300, 200, 100, 20],
        bounds: BOUNDS,
        attrs: Default::default(),
        ts_ns: 1_000_000,
        start_ts_ns: 0,
    });

    group.bench_function("proxima_native_encode_metric_histogram", |bencher| {
        bencher.iter(|| {
            let native = metric_to_native(black_box(&sample));
            exporter.encode_and_emit_payload(NativePayload::Metric(native));
        });
    });
    group.finish();
}

#[cfg(not(feature = "histogram"))]
fn bench_proxima_native_encode_metric_histogram(_criterion: &mut Criterion) {}

fn bench_proxima_native_frame_size_span_8attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c12_native");
    let record = make_span(8);

    group.bench_function("proxima_native_frame_size_span_8attr", |bencher| {
        bencher.iter(|| {
            let native = span_to_native(black_box(&record));
            let frame_msg = NativeFrame {
                kind: 0,
                seq: 0,
                payload: NativePayload::Span(native),
            };
            let size = postcard::to_allocvec(&frame_msg)
                .map(|vec| vec.len())
                .unwrap_or(0);
            black_box(size)
        });
    });
    group.finish();
}

fn bench_postcard_manual_encode_span_equivalent(criterion: &mut Criterion) {
    use bytes::Bytes;
    use proxima_telemetry::out::native::NativeSpan;

    let mut group = criterion.benchmark_group("c12_native");

    let span = NativeSpan {
        trace_id: [0xabu8; 16],
        span_id: [0xcdu8; 8],
        parent_span_id: Some([0xefu8; 8]),
        name: Bytes::from_static(b"bench_operation").to_vec(),
        kind: 1,
        start_ns: 1_700_000_000_000_000_000,
        duration_ns: 12_500_000,
        status: 1,
        status_reason: Bytes::new().to_vec(),
        attrs: (0..8u64)
            .map(|index| proxima_telemetry::out::native::NativeAttr {
                key: Bytes::from_static(b"http.status_code").to_vec(),
                value: proxima_telemetry::out::native::NativeAttrValue::U64(index),
            })
            .collect(),
        module_path: Bytes::from_static(b"proxima::bench").to_vec(),
    };

    group.bench_function("postcard_manual_encode_span_equivalent", |bencher| {
        bencher.iter(|| {
            let encoded = postcard::to_allocvec(black_box(&span));
            black_box(encoded)
        });
    });
    group.finish();
}

fn bench_opentelemetry_proto_encode_span_8attr_size(criterion: &mut Criterion) {
    use opentelemetry_proto::tonic::trace::v1 as otlp_trace;
    use prost::Message as _;

    let mut group = criterion.benchmark_group("c12_native");

    let attrs: Vec<opentelemetry_proto::tonic::common::v1::KeyValue> = (0..8u64)
        .map(|index| opentelemetry_proto::tonic::common::v1::KeyValue {
            key: "http.status_code".to_string(),
            value: Some(opentelemetry_proto::tonic::common::v1::AnyValue {
                value: Some(
                    opentelemetry_proto::tonic::common::v1::any_value::Value::IntValue(
                        index as i64,
                    ),
                ),
            }),
            ..Default::default()
        })
        .collect();

    let span = otlp_trace::Span {
        trace_id: vec![0xabu8; 16],
        span_id: vec![0xcdu8; 8],
        parent_span_id: vec![0xefu8; 8],
        name: "bench_operation".to_string(),
        kind: otlp_trace::span::SpanKind::Server as i32,
        start_time_unix_nano: 1_700_000_000_000_000_000,
        end_time_unix_nano: 1_700_000_012_500_000_000,
        attributes: attrs,
        status: None,
        ..Default::default()
    };

    group.bench_function("opentelemetry_proto_encode_span_8attr_size", |bencher| {
        bencher.iter(|| {
            let encoded = span.encode_to_vec();
            black_box(encoded)
        });
    });
    group.finish();
}

extern crate alloc;

criterion_group!(
    benches,
    bench_proxima_native_encode_span_8attr,
    bench_proxima_native_encode_log_4attr,
    bench_proxima_native_encode_metric_counter,
    bench_proxima_native_encode_metric_histogram,
    bench_proxima_native_frame_size_span_8attr,
    bench_postcard_manual_encode_span_equivalent,
    bench_opentelemetry_proto_encode_span_8attr_size,
);
criterion_main!(benches);
