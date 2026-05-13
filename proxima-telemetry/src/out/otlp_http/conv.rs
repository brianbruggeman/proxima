use alloc::string::String;
use alloc::vec::Vec;

use crate::log::LogRecord;
use crate::log::body::LogBody;
#[cfg(feature = "histogram")]
use crate::metric::sample::HistogramDataPoint;
use crate::metric::sample::{MetricSample, NumberDataPoint};
use crate::tag::{NestedValue, ScalarValue, Tag};
use crate::trace::EventRecord;
use crate::trace::kind::SpanKind;
use crate::trace::link::SpanLink;
use crate::trace::span::SpanRecord;
use crate::trace::status::Status;

use super::proto;
use super::proto::any_value::Value as AnyValueVariant;
use super::proto::metric::Data as MetricData;
use super::proto::number_data_point::Value as NumberValue;

pub fn span_to_proto(record: &SpanRecord) -> proto::Span {
    let end_ns = record.start_ns.saturating_add(record.duration_ns);
    let parent_span_id = record
        .parent_span_id
        .map(|sid| sid.to_bytes().to_vec())
        .unwrap_or_default();

    proto::Span {
        trace_id: record.trace_id.to_bytes().to_vec(),
        span_id: record.span_id.to_bytes().to_vec(),
        trace_state: alloc::string::String::new(),
        parent_span_id,
        flags: 0,
        name: String::from(record.name),
        kind: span_kind_to_proto(record.kind) as i32,
        start_time_unix_nano: record.start_ns,
        end_time_unix_nano: end_ns,
        attributes: tags_to_keyvalues(&record.attrs),
        dropped_attributes_count: 0,
        events: record.events.iter().map(event_to_proto).collect(),
        dropped_events_count: 0,
        links: record.links.iter().map(link_to_proto).collect(),
        dropped_links_count: 0,
        status: Some(status_to_proto(&record.status)),
    }
}

pub fn event_to_proto(record: &EventRecord) -> proto::SpanEvent {
    proto::SpanEvent {
        time_unix_nano: record.ts_ns,
        name: String::from(record.name),
        attributes: tags_to_keyvalues(&record.attrs),
        dropped_attributes_count: 0,
    }
}

pub fn link_to_proto(link: &SpanLink) -> proto::SpanLink {
    proto::SpanLink {
        trace_id: link.trace_id.to_bytes().to_vec(),
        span_id: link.span_id.to_bytes().to_vec(),
        trace_state: alloc::string::String::new(),
        attributes: tags_to_keyvalues(&link.attrs),
        dropped_attributes_count: 0,
        flags: 0,
    }
}

pub fn log_to_proto(record: &LogRecord) -> proto::LogRecord {
    let body = Some(body_to_anyvalue(&record.body));
    let trace_id = record
        .trace_id
        .map(|tid| tid.to_bytes().to_vec())
        .unwrap_or_default();
    let span_id = record
        .span_id
        .map(|sid| sid.to_bytes().to_vec())
        .unwrap_or_default();

    proto::LogRecord {
        time_unix_nano: record.ts_ns,
        observed_time_unix_nano: record.observed_ts_ns,
        severity_number: record.level.severity() as i32,
        severity_text: String::from(record.level.name()),
        body,
        attributes: tags_to_keyvalues(&record.attrs),
        dropped_attributes_count: 0,
        flags: u32::from(record.trace_flags.0),
        trace_id,
        span_id,
    }
}

pub fn metric_to_proto(sample: &MetricSample) -> proto::Metric {
    match sample {
        MetricSample::Counter(point) => counter_to_proto(point),
        MetricSample::Gauge(point) => gauge_to_proto(point),
        MetricSample::UpDownCounter(point) => updown_to_proto(point),
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(point) => histogram_metric_to_proto(point),
    }
}

pub fn tag_to_keyvalue(tag: &Tag) -> proto::KeyValue {
    match tag {
        Tag::Scalar { key, value } => proto::KeyValue {
            key: String::from(*key),
            value: Some(scalar_to_anyvalue(value)),
        },
        Tag::Structured { key, value } => proto::KeyValue {
            key: String::from(*key),
            value: Some(nested_to_anyvalue(value)),
        },
    }
}

pub fn scalar_to_anyvalue(value: &ScalarValue) -> proto::AnyValue {
    let inner = match value {
        ScalarValue::I64(int_val) => AnyValueVariant::IntValue(*int_val),
        ScalarValue::U64(uint_val) => AnyValueVariant::IntValue(*uint_val as i64),
        ScalarValue::F64(float_val) => AnyValueVariant::DoubleValue(*float_val),
        ScalarValue::Bool(bool_val) => AnyValueVariant::BoolValue(*bool_val),
        ScalarValue::Str(str_val) => AnyValueVariant::StringValue(String::from(*str_val)),
        ScalarValue::Bytes(bytes_val) => AnyValueVariant::BytesValue(bytes_val.as_ref().to_vec()),
    };
    proto::AnyValue { value: Some(inner) }
}

pub fn nested_to_anyvalue(value: &NestedValue) -> proto::AnyValue {
    match value {
        NestedValue::Scalar(scalar) => scalar_to_anyvalue(scalar),
        NestedValue::Array(items) => {
            let values = items.iter().map(nested_to_anyvalue).collect::<Vec<_>>();
            proto::AnyValue {
                value: Some(AnyValueVariant::ArrayValue(proto::ArrayValue { values })),
            }
        }
        NestedValue::Kv(pairs) => {
            let values = pairs
                .iter()
                .map(|(key, val)| proto::KeyValue {
                    key: String::from(*key),
                    value: Some(nested_to_anyvalue(val)),
                })
                .collect::<Vec<_>>();
            proto::AnyValue {
                value: Some(AnyValueVariant::KvlistValue(proto::KeyValueList { values })),
            }
        }
    }
}

fn tags_to_keyvalues(tags: &[Tag]) -> Vec<proto::KeyValue> {
    tags.iter().map(tag_to_keyvalue).collect()
}

fn span_kind_to_proto(kind: SpanKind) -> proto::SpanKind {
    match kind {
        SpanKind::Internal => proto::SpanKind::Internal,
        SpanKind::Server => proto::SpanKind::Server,
        SpanKind::Client => proto::SpanKind::Client,
        SpanKind::Producer => proto::SpanKind::Producer,
        SpanKind::Consumer => proto::SpanKind::Consumer,
    }
}

pub fn status_to_proto(status: &Status) -> proto::Status {
    match status {
        Status::Unset => proto::Status {
            message: String::new(),
            code: proto::StatusCode::Unset as i32,
        },
        Status::Ok => proto::Status {
            message: String::new(),
            code: proto::StatusCode::Ok as i32,
        },
        Status::Error { reason } => proto::Status {
            message: String::from(*reason),
            code: proto::StatusCode::Error as i32,
        },
    }
}

fn body_to_anyvalue(body: &LogBody) -> proto::AnyValue {
    match body {
        LogBody::Empty => proto::AnyValue { value: None },
        LogBody::Text(text) => proto::AnyValue {
            value: Some(AnyValueVariant::StringValue(String::from(*text))),
        },
        LogBody::Owned(bytes_val) => proto::AnyValue {
            value: Some(AnyValueVariant::BytesValue(bytes_val.as_ref().to_vec())),
        },
        LogBody::Structured(nested) => nested_to_anyvalue(nested),
    }
}

fn number_point_to_proto(point: &NumberDataPoint) -> proto::NumberDataPoint {
    let value = match &point.value {
        ScalarValue::F64(float_val) => Some(NumberValue::AsDouble(*float_val)),
        ScalarValue::I64(int_val) => Some(NumberValue::AsInt(*int_val)),
        ScalarValue::U64(uint_val) => Some(NumberValue::AsInt(*uint_val as i64)),
        ScalarValue::Bool(bool_val) => Some(NumberValue::AsInt(i64::from(*bool_val))),
        ScalarValue::Str(_) | ScalarValue::Bytes(_) => None,
    };
    proto::NumberDataPoint {
        attributes: tags_to_keyvalues(&point.attrs),
        start_time_unix_nano: point.start_ts_ns,
        time_unix_nano: point.ts_ns,
        value,
    }
}

fn counter_to_proto(point: &NumberDataPoint) -> proto::Metric {
    proto::Metric {
        name: String::new(),
        description: String::new(),
        unit: String::new(),
        data: Some(MetricData::Sum(proto::Sum {
            data_points: alloc::vec![number_point_to_proto(point)],
            aggregation_temporality: 2,
            is_monotonic: true,
        })),
    }
}

fn gauge_to_proto(point: &NumberDataPoint) -> proto::Metric {
    proto::Metric {
        name: String::new(),
        description: String::new(),
        unit: String::new(),
        data: Some(MetricData::Gauge(proto::Gauge {
            data_points: alloc::vec![number_point_to_proto(point)],
        })),
    }
}

fn updown_to_proto(point: &NumberDataPoint) -> proto::Metric {
    proto::Metric {
        name: String::new(),
        description: String::new(),
        unit: String::new(),
        data: Some(MetricData::Sum(proto::Sum {
            data_points: alloc::vec![number_point_to_proto(point)],
            aggregation_temporality: 2,
            is_monotonic: false,
        })),
    }
}

#[cfg(feature = "histogram")]
fn histogram_metric_to_proto(point: &HistogramDataPoint) -> proto::Metric {
    let hdp = proto::HistogramDataPoint {
        attributes: tags_to_keyvalues(&point.attrs),
        start_time_unix_nano: point.start_ts_ns,
        time_unix_nano: point.ts_ns,
        count: point.count,
        sum: Some(point.sum),
        bucket_counts: point.bucket_counts.clone(),
        explicit_bounds: point.bounds.to_vec(),
    };
    proto::Metric {
        name: String::new(),
        description: String::new(),
        unit: String::new(),
        data: Some(MetricData::Histogram(proto::Histogram {
            data_points: alloc::vec![hdp],
            aggregation_temporality: 2,
        })),
    }
}
