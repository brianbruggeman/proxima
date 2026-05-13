use alloc::string::String;
use alloc::vec::Vec;

use bytes::Bytes;
use parking_lot::Mutex;
use prost::Message;

use self::proto::{
    ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest, ResourceLogs,
    ResourceMetrics, ResourceSpans, ScopeLogs, ScopeMetrics, ScopeSpans,
};

pub mod conv;
pub mod proto;

pub struct OtlpHttpExporter {
    pub endpoint: String,
    pub(crate) pending_spans: Mutex<Vec<proto::Span>>,
    pub(crate) pending_logs: Mutex<Vec<proto::LogRecord>>,
    pub(crate) pending_metrics: Mutex<Vec<proto::Metric>>,
}

impl OtlpHttpExporter {
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            pending_spans: Mutex::new(Vec::new()),
            pending_logs: Mutex::new(Vec::new()),
            pending_metrics: Mutex::new(Vec::new()),
        }
    }

    #[must_use]
    pub fn encode_spans(&self) -> Bytes {
        let spans = {
            let mut locked = self.pending_spans.lock();
            core::mem::take(&mut *locked)
        };
        if spans.is_empty() {
            return Bytes::new();
        }
        let request = ExportTraceServiceRequest {
            resource_spans: alloc::vec![ResourceSpans {
                resource: None,
                scope_spans: alloc::vec![ScopeSpans {
                    scope: None,
                    spans,
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap_or(());
        Bytes::from(buf)
    }

    #[must_use]
    pub fn encode_logs(&self) -> Bytes {
        let records = {
            let mut locked = self.pending_logs.lock();
            core::mem::take(&mut *locked)
        };
        if records.is_empty() {
            return Bytes::new();
        }
        let request = ExportLogsServiceRequest {
            resource_logs: alloc::vec![ResourceLogs {
                resource: None,
                scope_logs: alloc::vec![ScopeLogs {
                    scope: None,
                    log_records: records,
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap_or(());
        Bytes::from(buf)
    }

    #[must_use]
    pub fn encode_metrics(&self) -> Bytes {
        let metrics = {
            let mut locked = self.pending_metrics.lock();
            core::mem::take(&mut *locked)
        };
        if metrics.is_empty() {
            return Bytes::new();
        }
        let request = ExportMetricsServiceRequest {
            resource_metrics: alloc::vec![ResourceMetrics {
                resource: None,
                scope_metrics: alloc::vec![ScopeMetrics {
                    scope: None,
                    metrics,
                    schema_url: String::new(),
                }],
                schema_url: String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap_or(());
        Bytes::from(buf)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs,
        clippy::approx_constant
    )]

    #[cfg(feature = "histogram")]
    use alloc::vec;
    use alloc::vec::Vec;

    use bytes::Bytes;
    use prost::Message;
    use rstest::rstest;

    use crate::id::TraceFlags;
    use crate::id::{SpanId, TraceId};
    use crate::level::Level;
    use crate::log::LogRecord;
    use crate::log::body::LogBody;
    #[cfg(feature = "histogram")]
    use crate::metric::sample::HistogramDataPoint;
    use crate::metric::sample::{MetricSample, NumberDataPoint};
    use crate::tag::{ScalarValue, Tag};
    use crate::trace::kind::SpanKind;
    use crate::trace::span::SpanRecord;
    use crate::trace::status::Status;

    use super::conv::log_to_proto;
    use super::conv::metric_to_proto;
    use super::conv::span_to_proto;
    use super::conv::{scalar_to_anyvalue, status_to_proto};
    use super::proto::any_value::Value as AnyValueVariant;
    use super::proto::{
        self, ExportLogsServiceRequest, ExportMetricsServiceRequest, ExportTraceServiceRequest,
    };

    fn make_span_record(attrs: smallvec::SmallVec<[Tag; 4]>) -> SpanRecord {
        SpanRecord {
            trace_id: TraceId::from_bytes([1u8; 16]),
            span_id: SpanId::from_bytes([2u8; 8]),
            parent_span_id: None,
            name: "test.span",
            kind: SpanKind::Internal,
            start_ns: 1_000_000,
            duration_ns: 500_000,
            status: Status::Unset,
            attrs,
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: crate::trace::tracestate::TraceState::empty(),
            module_path: "test",
            file_line: (0, 0),
        }
    }

    fn make_log_record(body: LogBody) -> LogRecord {
        LogRecord {
            ts_ns: 2_000_000,
            observed_ts_ns: 2_000_001,
            level: Level::INFO,
            body,
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags(0),
            module_path: "test",
            file_line: (0, 0),
        }
    }

    // 1. empty SpanRecord → encode → round-trip decode matches
    #[test]
    fn empty_span_encodes_and_decodes() {
        let record = make_span_record(smallvec::SmallVec::new());
        let span = span_to_proto(&record);
        let request = ExportTraceServiceRequest {
            resource_spans: alloc::vec![proto::ResourceSpans {
                resource: None,
                scope_spans: alloc::vec![proto::ScopeSpans {
                    scope: None,
                    spans: alloc::vec![span],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap();
        let decoded = ExportTraceServiceRequest::decode(buf.as_slice()).unwrap();
        let decoded_span = &decoded.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(decoded_span.name, "test.span");
        assert_eq!(decoded_span.trace_id, [1u8; 16]);
        assert_eq!(decoded_span.span_id, [2u8; 8]);
        assert!(decoded_span.attributes.is_empty());
    }

    // 2. SpanRecord with 8 attrs → decoded attrs match
    #[test]
    fn span_with_8_attrs_round_trips() {
        let attrs = smallvec::smallvec![
            Tag::Scalar {
                key: "k0",
                value: ScalarValue::I64(0)
            },
            Tag::Scalar {
                key: "k1",
                value: ScalarValue::I64(1)
            },
            Tag::Scalar {
                key: "k2",
                value: ScalarValue::Bool(true)
            },
            Tag::Scalar {
                key: "k3",
                value: ScalarValue::F64(3.14)
            },
            Tag::Scalar {
                key: "k4",
                value: ScalarValue::Str("hello")
            },
            Tag::Scalar {
                key: "k5",
                value: ScalarValue::U64(5)
            },
            Tag::Scalar {
                key: "k6",
                value: ScalarValue::I64(6)
            },
            Tag::Scalar {
                key: "k7",
                value: ScalarValue::Bool(false)
            },
        ];
        let record = make_span_record(attrs);
        let span = span_to_proto(&record);
        let mut buf = Vec::new();
        ExportTraceServiceRequest {
            resource_spans: alloc::vec![proto::ResourceSpans {
                resource: None,
                scope_spans: alloc::vec![proto::ScopeSpans {
                    scope: None,
                    spans: alloc::vec![span],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        }
        .encode(&mut buf)
        .unwrap();
        let decoded = ExportTraceServiceRequest::decode(buf.as_slice()).unwrap();
        let decoded_attrs = &decoded.resource_spans[0].scope_spans[0].spans[0].attributes;
        assert_eq!(decoded_attrs.len(), 8);
        assert_eq!(decoded_attrs[0].key, "k0");
        assert_eq!(decoded_attrs[4].key, "k4");
    }

    // 3. LogRecord with body Text → decoded body matches
    #[test]
    fn log_with_text_body_round_trips() {
        let record = make_log_record(LogBody::Text("hello world"));
        let log = log_to_proto(&record);
        let request = ExportLogsServiceRequest {
            resource_logs: alloc::vec![proto::ResourceLogs {
                resource: None,
                scope_logs: alloc::vec![proto::ScopeLogs {
                    scope: None,
                    log_records: alloc::vec![log],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap();
        let decoded = ExportLogsServiceRequest::decode(buf.as_slice()).unwrap();
        let decoded_log = &decoded.resource_logs[0].scope_logs[0].log_records[0];
        let body = decoded_log.body.as_ref().unwrap();
        assert_eq!(
            body.value,
            Some(AnyValueVariant::StringValue(alloc::string::String::from(
                "hello world"
            )))
        );
    }

    // 4. MetricSample::Counter → decoded Sum.data_points match
    #[test]
    fn counter_metric_round_trips() {
        let point = NumberDataPoint {
            value: ScalarValue::U64(42),
            attrs: smallvec::SmallVec::new(),
            ts_ns: 3_000_000,
            start_ts_ns: 1_000_000,
        };
        let sample = MetricSample::Counter(point);
        let metric = metric_to_proto(&sample);
        let request = ExportMetricsServiceRequest {
            resource_metrics: alloc::vec![proto::ResourceMetrics {
                resource: None,
                scope_metrics: alloc::vec![proto::ScopeMetrics {
                    scope: None,
                    metrics: alloc::vec![metric],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        };
        let mut buf = Vec::new();
        request.encode(&mut buf).unwrap();
        let decoded = ExportMetricsServiceRequest::decode(buf.as_slice()).unwrap();
        let decoded_metric = &decoded.resource_metrics[0].scope_metrics[0].metrics[0];
        let proto::metric::Data::Sum(sum) = decoded_metric.data.as_ref().unwrap() else {
            panic!("expected Sum");
        };
        assert_eq!(sum.data_points.len(), 1);
        assert!(sum.is_monotonic);
        assert_eq!(sum.data_points[0].time_unix_nano, 3_000_000);
    }

    // 5. MetricSample::Histogram → bucket_counts + explicit_bounds round-trip
    #[cfg(feature = "histogram")]
    #[test]
    fn histogram_metric_round_trips() {
        let point = HistogramDataPoint {
            count: 10,
            sum: 55.0,
            bucket_counts: vec![1, 2, 3, 4],
            bounds: &[1.0, 2.0, 4.0],
            attrs: smallvec::SmallVec::new(),
            ts_ns: 5_000_000,
            start_ts_ns: 1_000_000,
        };
        let sample = MetricSample::Histogram(point);
        let metric = metric_to_proto(&sample);
        let mut buf = Vec::new();
        ExportMetricsServiceRequest {
            resource_metrics: alloc::vec![proto::ResourceMetrics {
                resource: None,
                scope_metrics: alloc::vec![proto::ScopeMetrics {
                    scope: None,
                    metrics: alloc::vec![metric],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        }
        .encode(&mut buf)
        .unwrap();
        let decoded = ExportMetricsServiceRequest::decode(buf.as_slice()).unwrap();
        let decoded_metric = &decoded.resource_metrics[0].scope_metrics[0].metrics[0];
        let proto::metric::Data::Histogram(hist) = decoded_metric.data.as_ref().unwrap() else {
            panic!("expected Histogram");
        };
        let hdp = &hist.data_points[0];
        assert_eq!(hdp.count, 10);
        assert_eq!(hdp.sum, Some(55.0));
        assert_eq!(hdp.bucket_counts, vec![1, 2, 3, 4]);
        assert_eq!(hdp.explicit_bounds, vec![1.0, 2.0, 4.0]);
    }

    // 6. scalar_to_anyvalue: each ScalarValue variant → AnyValue → back
    #[rstest]
    #[case::i64(ScalarValue::I64(-7), AnyValueVariant::IntValue(-7))]
    #[case::u64(ScalarValue::U64(99), AnyValueVariant::IntValue(99))]
    #[case::f64(ScalarValue::F64(2.5), AnyValueVariant::DoubleValue(2.5))]
    #[case::bool_true(ScalarValue::Bool(true), AnyValueVariant::BoolValue(true))]
    #[case::bool_false(ScalarValue::Bool(false), AnyValueVariant::BoolValue(false))]
    #[case::str_val(
        ScalarValue::Str("abc"),
        AnyValueVariant::StringValue(alloc::string::String::from("abc"))
    )]
    #[case::bytes_val(
        ScalarValue::Bytes(Bytes::from_static(b"raw")),
        AnyValueVariant::BytesValue(alloc::vec![b'r', b'a', b'w'])
    )]
    fn scalar_value_to_anyvalue_variants(
        #[case] input: ScalarValue,
        #[case] expected: AnyValueVariant,
    ) {
        let got = scalar_to_anyvalue(&input);
        assert_eq!(got.value, Some(expected));
    }

    // 7. status conversion: Unset/Ok/Error → StatusCode + message
    #[test]
    fn status_conversions() {
        let unset = status_to_proto(&Status::Unset);
        assert_eq!(unset.code, 0);
        assert!(unset.message.is_empty());

        let ok = status_to_proto(&Status::Ok);
        assert_eq!(ok.code, 1);

        let err = status_to_proto(&Status::Error { reason: "timeout" });
        assert_eq!(err.code, 2);
        assert_eq!(err.message, "timeout");
    }

    // 8. cross-check: our encoded bytes are structurally valid OTLP protobuf.
    //
    // opentelemetry-proto 0.32 uses prost 0.14; we use prost 0.13 — calling
    // cross-crate decode is not possible across the prost version boundary.
    // We prove wire-format correctness two ways:
    //   (a) full round-trip: encode → our decoder → all fields intact
    //   (b) byte-for-byte match against OTel-struct-encoded bytes (same field tags →
    //       same wire output for identical values)
    #[test]
    fn otel_proto_wire_compatible_round_trip_and_size() {
        use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest as OtelRequest;
        use opentelemetry_proto::tonic::common::v1::any_value::Value as OtelAV;
        use opentelemetry_proto::tonic::common::v1::{
            AnyValue as OtelAnyValue, KeyValue as OtelKV,
        };
        use opentelemetry_proto::tonic::trace::v1::{
            ResourceSpans as OtelRS, ScopeSpans as OtelSS, Span as OtelSpan, Status as OtelStatus,
        };

        let attrs = smallvec::smallvec![
            Tag::Scalar {
                key: "http.method",
                value: ScalarValue::Str("GET")
            },
            Tag::Scalar {
                key: "http.status_code",
                value: ScalarValue::I64(200)
            },
        ];
        let record = SpanRecord {
            trace_id: TraceId::from_bytes([1u8; 16]),
            span_id: SpanId::from_bytes([2u8; 8]),
            parent_span_id: None,
            name: "test.span",
            kind: SpanKind::Internal,
            start_ns: 1_000_000,
            duration_ns: 500_000,
            status: Status::Ok,
            attrs,
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: crate::trace::tracestate::TraceState::empty(),
            module_path: "test",
            file_line: (0, 0),
        };

        let our_span = span_to_proto(&record);
        let our_request = ExportTraceServiceRequest {
            resource_spans: alloc::vec![proto::ResourceSpans {
                resource: None,
                scope_spans: alloc::vec![proto::ScopeSpans {
                    scope: None,
                    spans: alloc::vec![our_span],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        };
        let mut our_buf = Vec::new();
        our_request.encode(&mut our_buf).unwrap();

        // (a) round-trip: our decoder must reconstruct all fields
        let decoded = ExportTraceServiceRequest::decode(our_buf.as_slice()).unwrap();
        let decoded_span = &decoded.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(decoded_span.name, "test.span");
        assert_eq!(decoded_span.trace_id, [1u8; 16]);
        assert_eq!(decoded_span.span_id, [2u8; 8]);
        assert_eq!(decoded_span.attributes.len(), 2);
        assert_eq!(decoded_span.attributes[0].key, "http.method");
        assert_eq!(decoded_span.start_time_unix_nano, 1_000_000);

        // (b) build the identical logical content with OTel structs; the encoded
        //     bytes must be identical (protobuf encoding is deterministic for the same
        //     field values with the same field tags)
        let otel_request = OtelRequest {
            resource_spans: alloc::vec![OtelRS {
                resource: None,
                scope_spans: alloc::vec![OtelSS {
                    scope: None,
                    spans: alloc::vec![OtelSpan {
                        trace_id: [1u8; 16].to_vec(),
                        span_id: [2u8; 8].to_vec(),
                        parent_span_id: alloc::vec![],
                        trace_state: alloc::string::String::new(),
                        flags: 0,
                        name: alloc::string::String::from("test.span"),
                        kind: 1,
                        start_time_unix_nano: 1_000_000,
                        end_time_unix_nano: 1_500_000,
                        attributes: alloc::vec![
                            OtelKV {
                                key: alloc::string::String::from("http.method"),
                                value: Some(OtelAnyValue {
                                    value: Some(OtelAV::StringValue(alloc::string::String::from(
                                        "GET"
                                    ),)),
                                }),
                                ..Default::default()
                            },
                            OtelKV {
                                key: alloc::string::String::from("http.status_code"),
                                value: Some(OtelAnyValue {
                                    value: Some(OtelAV::IntValue(200)),
                                }),
                                ..Default::default()
                            },
                        ],
                        dropped_attributes_count: 0,
                        events: alloc::vec![],
                        dropped_events_count: 0,
                        links: alloc::vec![],
                        dropped_links_count: 0,
                        status: Some(OtelStatus {
                            message: alloc::string::String::new(),
                            code: 1,
                        }),
                    }],
                    schema_url: alloc::string::String::new(),
                }],
                schema_url: alloc::string::String::new(),
            }],
        };
        // use prost 0.14 (the version opentelemetry-proto 0.32 uses) via renamed dev-dep
        use prost::Message as _;
        let mut otel_buf = Vec::new();
        otel_request.encode(&mut otel_buf).unwrap();

        // size within 10%: our encoder must not bloat vs the reference implementation
        let ratio = our_buf.len() as f64 / otel_buf.len() as f64;
        assert!(
            (0.90..=1.10).contains(&ratio),
            "size ratio {ratio:.2} out of [0.90, 1.10]: our={} otel={}",
            our_buf.len(),
            otel_buf.len(),
        );
        // byte-for-byte match: same field tags + same values → identical wire bytes
        assert_eq!(
            our_buf,
            otel_buf,
            "wire mismatch: our {} bytes vs otel {} bytes",
            our_buf.len(),
            otel_buf.len()
        );
    }
}
