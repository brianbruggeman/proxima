// C12 deviates from "no Arc<dyn> in user types" in exactly one place: NativeExporter<S>
// stores S directly (monomorphic), but when the caller needs type-erasure they wrap in
// Arc<dyn FrameSink + Send + Sync>.  This is the same documented exception as C9's Arc<dyn Exporter>.
//
// Wire structs use Vec<u8> for owned byte content — bytes::Bytes requires a serde feature
// flag that we don't want to force on the workspace dep. Vec<u8> serialises identically
// with postcard and avoids the external feature dependency.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::trace::{EventRecord, SpanLink, SpanRecord};

pub const NATIVE_FRAME_SIZE: usize = 1500;

pub trait FrameSink: Send + Sync {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]);
}

impl<S: FrameSink + ?Sized> FrameSink for Arc<S> {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        (**self).write_frame(frame);
    }
}

pub struct NativeExporter<S: FrameSink> {
    sink: S,
    schema_version: u8,
    seq: AtomicU32,
}

impl<S: FrameSink> NativeExporter<S> {
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            schema_version: 0,
            seq: AtomicU32::new(0),
        }
    }

    pub const fn schema_version(mut self, version: u8) -> Self {
        self.schema_version = version;
        self
    }

    fn next_seq(&self) -> u32 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    fn encode_and_emit(&self, kind_low: u8, payload: NativePayload) -> Result<(), Error> {
        let kind = (self.schema_version << 4) | (kind_low & 0x0f);
        let seq = self.next_seq();
        let frame_msg = NativeFrame { kind, seq, payload };
        let mut buf = [0u8; NATIVE_FRAME_SIZE];
        postcard::to_slice(&frame_msg, &mut buf).map_err(|_| Error::InvalidInput)?;
        self.sink.write_frame(&buf);
        Ok(())
    }

    /// Encode a pre-built NativePayload into a frame and write it to the sink.
    /// Called by NativePipe to avoid re-entering the kind-selection logic.
    pub fn encode_and_emit_payload(&self, payload: NativePayload) {
        let kind_low = match &payload {
            NativePayload::Span(_) => KIND_SPAN,
            NativePayload::Event(_) => KIND_EVENT,
            NativePayload::Log(_) => KIND_LOG,
            NativePayload::Metric(_) => KIND_METRIC,
            NativePayload::Link(_) => KIND_LINK,
            NativePayload::OverflowAttr(_) => KIND_OVERFLOW_ATTR,
        };
        let _ = self.encode_and_emit(kind_low, payload);
    }

    fn encode_and_emit_ref(
        &self,
        kind_low: u8,
        payload: NativePayloadRef<'_>,
    ) -> Result<(), Error> {
        let kind = (self.schema_version << 4) | (kind_low & 0x0f);
        let seq = self.next_seq();
        let frame_msg = NativeFrameRef { kind, seq, payload };
        let mut buf = [0u8; NATIVE_FRAME_SIZE];
        postcard::to_slice(&frame_msg, &mut buf).map_err(|_| Error::InvalidInput)?;
        self.sink.write_frame(&buf);
        Ok(())
    }

    /// Zero-alloc encode: the payload borrows directly from the source record,
    /// producing a byte-identical frame to [`encode_and_emit_payload`] without
    /// the per-record name/key/attr heap allocations. This is the drain hot path.
    pub fn encode_and_emit_payload_ref(&self, payload: NativePayloadRef<'_>) {
        let kind_low = match &payload {
            NativePayloadRef::Span(_) => KIND_SPAN,
            NativePayloadRef::Event(_) => KIND_EVENT,
            NativePayloadRef::Log(_) => KIND_LOG,
            NativePayloadRef::Metric(_) => KIND_METRIC,
            NativePayloadRef::Link(_) => KIND_LINK,
            NativePayloadRef::OverflowAttr(_) => KIND_OVERFLOW_ATTR,
        };
        let _ = self.encode_and_emit_ref(kind_low, payload);
    }
}

const KIND_SPAN: u8 = 0;
const KIND_EVENT: u8 = 1;
const KIND_LOG: u8 = 2;
const KIND_METRIC: u8 = 3;
const KIND_LINK: u8 = 4;
#[allow(dead_code)]
const KIND_OVERFLOW_ATTR: u8 = 5;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeFrame {
    pub kind: u8,
    pub seq: u32,
    pub payload: NativePayload,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum NativePayload {
    Span(NativeSpan),
    Event(NativeEvent),
    Log(NativeLog),
    Metric(NativeMetric),
    Link(NativeLink),
    OverflowAttr(NativeOverflowAttr),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeAttr {
    pub key: Vec<u8>,
    pub value: NativeAttrValue,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum NativeAttrValue {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Str(Vec<u8>),
    Bytes(Vec<u8>),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeSpan {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub name: Vec<u8>,
    pub kind: u8,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub status: u8,
    pub status_reason: Vec<u8>,
    pub attrs: Vec<NativeAttr>,
    pub module_path: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeEvent {
    pub parent_span_id: [u8; 8],
    pub name: Vec<u8>,
    pub ts_ns: u64,
    pub attrs: Vec<NativeAttr>,
    pub module_path: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeLog {
    pub ts_ns: u64,
    pub severity: u8,
    pub body: Vec<u8>,
    pub attrs: Vec<NativeAttr>,
    pub trace_id: Option<[u8; 16]>,
    pub span_id: Option<[u8; 8]>,
    pub module_path: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum NativeMetric {
    Counter(NativeDataPoint),
    Gauge(NativeDataPoint),
    UpDownCounter(NativeDataPoint),
    Histogram(NativeHistogram),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeDataPoint {
    pub value_i64: i64,
    pub value_u64: u64,
    pub value_f64: f64,
    pub kind: u8,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
    pub attrs: Vec<NativeAttr>,
}

const DP_KIND_I64: u8 = 0;
const DP_KIND_U64: u8 = 1;
const DP_KIND_F64: u8 = 2;
const DP_KIND_BOOL: u8 = 3;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeHistogram {
    pub count: u64,
    pub sum: f64,
    pub bucket_counts: Vec<u64>,
    pub bound_count: u16,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
    pub attrs: Vec<NativeAttr>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeLink {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub attrs: Vec<NativeAttr>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeOverflowAttr {
    pub parent_span_id: [u8; 8],
    pub attrs: Vec<NativeAttr>,
}

/// v1 stub — wire surface exists; transport implementation is out of scope.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum NativeControlMessage {
    SetLevelOverride {
        module_prefix: Vec<u8>,
        severity: u8,
    },
    SetSamplingRate {
        fraction_per_million: u32,
    },
    Flush,
}

pub fn span_to_native(record: &SpanRecord) -> NativeSpan {
    use crate::trace::Status;

    let (status_code, status_reason) = match &record.status {
        Status::Unset => (0u8, Vec::new()),
        Status::Ok => (1u8, Vec::new()),
        Status::Error { reason } => (2u8, reason.as_bytes().to_vec()),
    };

    NativeSpan {
        trace_id: record.trace_id.to_bytes(),
        span_id: record.span_id.to_bytes(),
        parent_span_id: record.parent_span_id.map(|sid| sid.to_bytes()),
        name: record.name.as_bytes().to_vec(),
        kind: span_kind_to_u8(record.kind),
        start_ns: record.start_ns,
        duration_ns: record.duration_ns,
        status: status_code,
        status_reason,
        attrs: tags_to_native(&record.attrs),
        module_path: record.module_path.as_bytes().to_vec(),
    }
}

pub fn event_to_native(record: &EventRecord) -> NativeEvent {
    NativeEvent {
        parent_span_id: record.parent_span_id.to_bytes(),
        name: record.name.as_bytes().to_vec(),
        ts_ns: record.ts_ns,
        attrs: tags_to_native(&record.attrs),
        module_path: record.module_path.as_bytes().to_vec(),
    }
}

pub fn log_to_native(record: &LogRecord) -> NativeLog {
    use crate::log::LogBody;

    let body = match &record.body {
        LogBody::Empty => Vec::new(),
        LogBody::Text(text) => text.as_bytes().to_vec(),
        LogBody::Owned(bytes) => bytes.to_vec(),
        LogBody::Structured(_) => b"<structured>".to_vec(),
    };

    NativeLog {
        ts_ns: record.ts_ns,
        severity: record.level.severity(),
        body,
        attrs: tags_to_native(&record.attrs),
        trace_id: record.trace_id.map(|tid| tid.to_bytes()),
        span_id: record.span_id.map(|sid| sid.to_bytes()),
        module_path: record.module_path.as_bytes().to_vec(),
    }
}

pub fn metric_to_native(sample: &MetricSample) -> NativeMetric {
    match sample {
        MetricSample::Counter(point) => NativeMetric::Counter(data_point_to_native(point)),
        MetricSample::Gauge(point) => NativeMetric::Gauge(data_point_to_native(point)),
        MetricSample::UpDownCounter(point) => {
            NativeMetric::UpDownCounter(data_point_to_native(point))
        }
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(point) => NativeMetric::Histogram(NativeHistogram {
            count: point.count,
            sum: point.sum,
            bucket_counts: point.bucket_counts.clone(),
            bound_count: point.bounds.len() as u16,
            ts_ns: point.ts_ns,
            start_ts_ns: point.start_ts_ns,
            attrs: tags_to_native(&point.attrs),
        }),
    }
}

pub fn link_to_native(link: &SpanLink) -> NativeLink {
    NativeLink {
        trace_id: link.trace_id.to_bytes(),
        span_id: link.span_id.to_bytes(),
        attrs: tags_to_native(&link.attrs),
    }
}

fn data_point_to_native(point: &crate::metric::NumberDataPoint) -> NativeDataPoint {
    use crate::tag::ScalarValue;

    let (value_i64, value_u64, value_f64, kind) = match &point.value {
        ScalarValue::I64(v) => (*v, 0u64, 0.0f64, DP_KIND_I64),
        ScalarValue::U64(v) => (0i64, *v, 0.0f64, DP_KIND_U64),
        ScalarValue::F64(v) => (0i64, 0u64, *v, DP_KIND_F64),
        ScalarValue::Bool(v) => (0i64, *v as u64, 0.0f64, DP_KIND_BOOL),
        ScalarValue::Str(_) | ScalarValue::Bytes(_) => (0i64, 0u64, 0.0f64, DP_KIND_U64),
    };

    NativeDataPoint {
        value_i64,
        value_u64,
        value_f64,
        kind,
        ts_ns: point.ts_ns,
        start_ts_ns: point.start_ts_ns,
        attrs: tags_to_native(&point.attrs),
    }
}

fn tags_to_native(tags: &[crate::tag::Tag]) -> Vec<NativeAttr> {
    use crate::tag::{ScalarValue, Tag};

    tags.iter()
        .filter_map(|tag| match tag {
            Tag::Scalar { key, value } => {
                let native_value = match value {
                    ScalarValue::I64(v) => NativeAttrValue::I64(*v),
                    ScalarValue::U64(v) => NativeAttrValue::U64(*v),
                    ScalarValue::F64(v) => NativeAttrValue::F64(*v),
                    ScalarValue::Bool(v) => NativeAttrValue::Bool(*v),
                    ScalarValue::Str(v) => NativeAttrValue::Str(v.as_bytes().to_vec()),
                    ScalarValue::Bytes(v) => NativeAttrValue::Bytes(v.to_vec()),
                };
                Some(NativeAttr {
                    key: key.as_bytes().to_vec(),
                    value: native_value,
                })
            }
            Tag::Structured { .. } => None,
        })
        .collect()
}

fn span_kind_to_u8(kind: crate::trace::SpanKind) -> u8 {
    use crate::trace::SpanKind;
    match kind {
        SpanKind::Internal => 0,
        SpanKind::Server => 1,
        SpanKind::Client => 2,
        SpanKind::Producer => 3,
        SpanKind::Consumer => 4,
    }
}

// ---- borrowed (zero-alloc) encode mirror -------------------------------
//
// The owned Native* types above are the DECODE surface (they own their bytes).
// On the ENCODE hot path we never need to own: the source record outlives the
// encode call, so we borrow its &'static str names, tag keys, and attr slices
// directly. postcard serialises `&[u8]` identically to `Vec<u8>` and a borrowed
// seq identically to an owned one, so these produce byte-for-byte the same
// frame as the owned path — proven by `native_ref_roundtrip_matches_owned`.
// This removes ~one heap allocation per name/key/attr-vec per record.

#[derive(Debug, Serialize)]
pub struct NativeAttrRef<'a> {
    pub key: &'a [u8],
    pub value: NativeAttrValueRef<'a>,
}

#[derive(Debug, Serialize)]
pub enum NativeAttrValueRef<'a> {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Str(&'a [u8]),
    Bytes(&'a [u8]),
}

// Zero-alloc attr seq: serialise directly from the borrowed `&[Tag]` slice
// (filtering Structured) instead of collecting into a Vec<NativeAttrRef>.
// postcard encodes a seq as varint(len) + elements, byte-identical to the owned
// Vec<NativeAttr> path — proven by the parity tests. No heap allocation per
// record on the drain encode loop.
#[derive(Debug)]
pub struct AttrsRef<'a>(&'a [crate::tag::Tag]);

impl serde::Serialize for AttrsRef<'_> {
    fn serialize<SerOut>(&self, serializer: SerOut) -> Result<SerOut::Ok, SerOut::Error>
    where
        SerOut: serde::Serializer,
    {
        use crate::tag::{ScalarValue, Tag};
        use serde::ser::SerializeSeq;

        let len = self
            .0
            .iter()
            .filter(|tag| matches!(tag, Tag::Scalar { .. }))
            .count();
        let mut seq = serializer.serialize_seq(Some(len))?;
        for tag in self.0 {
            if let Tag::Scalar { key, value } = tag {
                let native_value = match value {
                    ScalarValue::I64(v) => NativeAttrValueRef::I64(*v),
                    ScalarValue::U64(v) => NativeAttrValueRef::U64(*v),
                    ScalarValue::F64(v) => NativeAttrValueRef::F64(*v),
                    ScalarValue::Bool(v) => NativeAttrValueRef::Bool(*v),
                    ScalarValue::Str(v) => NativeAttrValueRef::Str(v.as_bytes()),
                    ScalarValue::Bytes(v) => NativeAttrValueRef::Bytes(v.as_ref()),
                };
                seq.serialize_element(&NativeAttrRef {
                    key: key.as_bytes(),
                    value: native_value,
                })?;
            }
        }
        seq.end()
    }
}

#[derive(Debug, Serialize)]
pub struct NativeSpanRef<'a> {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub name: &'a [u8],
    pub kind: u8,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub status: u8,
    pub status_reason: &'a [u8],
    pub attrs: AttrsRef<'a>,
    pub module_path: &'a [u8],
}

#[derive(Debug, Serialize)]
pub struct NativeEventRef<'a> {
    pub parent_span_id: [u8; 8],
    pub name: &'a [u8],
    pub ts_ns: u64,
    pub attrs: AttrsRef<'a>,
    pub module_path: &'a [u8],
}

#[derive(Debug, Serialize)]
pub struct NativeLogRef<'a> {
    pub ts_ns: u64,
    pub severity: u8,
    pub body: &'a [u8],
    pub attrs: AttrsRef<'a>,
    pub trace_id: Option<[u8; 16]>,
    pub span_id: Option<[u8; 8]>,
    pub module_path: &'a [u8],
}

#[derive(Debug, Serialize)]
pub struct NativeDataPointRef<'a> {
    pub value_i64: i64,
    pub value_u64: u64,
    pub value_f64: f64,
    pub kind: u8,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
    pub attrs: AttrsRef<'a>,
}

#[derive(Debug, Serialize)]
pub struct NativeHistogramRef<'a> {
    pub count: u64,
    pub sum: f64,
    pub bucket_counts: &'a [u64],
    pub bound_count: u16,
    pub ts_ns: u64,
    pub start_ts_ns: u64,
    pub attrs: AttrsRef<'a>,
}

#[derive(Debug, Serialize)]
pub enum NativeMetricRef<'a> {
    Counter(NativeDataPointRef<'a>),
    Gauge(NativeDataPointRef<'a>),
    UpDownCounter(NativeDataPointRef<'a>),
    Histogram(NativeHistogramRef<'a>),
}

#[derive(Debug, Serialize)]
pub struct NativeLinkRef<'a> {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub attrs: AttrsRef<'a>,
}

#[derive(Debug, Serialize)]
pub struct NativeOverflowAttrRef<'a> {
    pub parent_span_id: [u8; 8],
    pub attrs: AttrsRef<'a>,
}

// variant order MUST mirror NativePayload — postcard keys enums by index.
#[derive(Debug, Serialize)]
pub enum NativePayloadRef<'a> {
    Span(NativeSpanRef<'a>),
    Event(NativeEventRef<'a>),
    Log(NativeLogRef<'a>),
    Metric(NativeMetricRef<'a>),
    Link(NativeLinkRef<'a>),
    OverflowAttr(NativeOverflowAttrRef<'a>),
}

#[derive(Debug, Serialize)]
pub struct NativeFrameRef<'a> {
    pub kind: u8,
    pub seq: u32,
    pub payload: NativePayloadRef<'a>,
}

pub fn span_to_native_ref(record: &SpanRecord) -> NativeSpanRef<'_> {
    use crate::trace::Status;

    let (status_code, status_reason): (u8, &[u8]) = match &record.status {
        Status::Unset => (0, b""),
        Status::Ok => (1, b""),
        Status::Error { reason } => (2, reason.as_bytes()),
    };

    NativeSpanRef {
        trace_id: record.trace_id.to_bytes(),
        span_id: record.span_id.to_bytes(),
        parent_span_id: record.parent_span_id.map(|sid| sid.to_bytes()),
        name: record.name.as_bytes(),
        kind: span_kind_to_u8(record.kind),
        start_ns: record.start_ns,
        duration_ns: record.duration_ns,
        status: status_code,
        status_reason,
        attrs: AttrsRef(&record.attrs),
        module_path: record.module_path.as_bytes(),
    }
}

pub fn event_to_native_ref(record: &EventRecord) -> NativeEventRef<'_> {
    NativeEventRef {
        parent_span_id: record.parent_span_id.to_bytes(),
        name: record.name.as_bytes(),
        ts_ns: record.ts_ns,
        attrs: AttrsRef(&record.attrs),
        module_path: record.module_path.as_bytes(),
    }
}

pub fn log_to_native_ref(record: &LogRecord) -> NativeLogRef<'_> {
    use crate::log::LogBody;

    let body: &[u8] = match &record.body {
        LogBody::Empty => b"",
        LogBody::Text(text) => text.as_bytes(),
        LogBody::Owned(bytes) => bytes.as_ref(),
        LogBody::Structured(_) => b"<structured>",
    };

    NativeLogRef {
        ts_ns: record.ts_ns,
        severity: record.level.severity(),
        body,
        attrs: AttrsRef(&record.attrs),
        trace_id: record.trace_id.map(|tid| tid.to_bytes()),
        span_id: record.span_id.map(|sid| sid.to_bytes()),
        module_path: record.module_path.as_bytes(),
    }
}

fn data_point_to_native_ref(point: &crate::metric::NumberDataPoint) -> NativeDataPointRef<'_> {
    use crate::tag::ScalarValue;

    let (value_i64, value_u64, value_f64, kind) = match &point.value {
        ScalarValue::I64(v) => (*v, 0u64, 0.0f64, DP_KIND_I64),
        ScalarValue::U64(v) => (0i64, *v, 0.0f64, DP_KIND_U64),
        ScalarValue::F64(v) => (0i64, 0u64, *v, DP_KIND_F64),
        ScalarValue::Bool(v) => (0i64, *v as u64, 0.0f64, DP_KIND_BOOL),
        ScalarValue::Str(_) | ScalarValue::Bytes(_) => (0i64, 0u64, 0.0f64, DP_KIND_U64),
    };

    NativeDataPointRef {
        value_i64,
        value_u64,
        value_f64,
        kind,
        ts_ns: point.ts_ns,
        start_ts_ns: point.start_ts_ns,
        attrs: AttrsRef(&point.attrs),
    }
}

pub fn metric_to_native_ref(sample: &MetricSample) -> NativeMetricRef<'_> {
    match sample {
        MetricSample::Counter(point) => NativeMetricRef::Counter(data_point_to_native_ref(point)),
        MetricSample::Gauge(point) => NativeMetricRef::Gauge(data_point_to_native_ref(point)),
        MetricSample::UpDownCounter(point) => {
            NativeMetricRef::UpDownCounter(data_point_to_native_ref(point))
        }
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(point) => NativeMetricRef::Histogram(NativeHistogramRef {
            count: point.count,
            sum: point.sum,
            bucket_counts: point.bucket_counts.as_slice(),
            bound_count: point.bounds.len() as u16,
            ts_ns: point.ts_ns,
            start_ts_ns: point.start_ts_ns,
            attrs: AttrsRef(&point.attrs),
        }),
    }
}

pub fn link_to_native_ref(link: &SpanLink) -> NativeLinkRef<'_> {
    NativeLinkRef {
        trace_id: link.trace_id.to_bytes(),
        span_id: link.span_id.to_bytes(),
        attrs: AttrsRef(&link.attrs),
    }
}

pub fn encode_frame(frame_msg: &NativeFrame) -> Result<[u8; NATIVE_FRAME_SIZE], Error> {
    let mut buf = [0u8; NATIVE_FRAME_SIZE];
    postcard::to_slice(frame_msg, &mut buf).map_err(|_| Error::InvalidInput)?;
    Ok(buf)
}

pub fn decode_frame(buf: &[u8; NATIVE_FRAME_SIZE]) -> Result<NativeFrame, Error> {
    postcard::from_bytes(buf).map_err(|_| Error::InvalidInput)
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
        clippy::default_constructed_unit_structs
    )]

    extern crate std;

    use alloc::string::ToString;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        NATIVE_FRAME_SIZE, NativeControlMessage, NativeExporter, NativeFrame, NativeFrameRef,
        NativePayload, NativePayloadRef, decode_frame, encode_frame, event_to_native,
        event_to_native_ref, link_to_native, link_to_native_ref, log_to_native, log_to_native_ref,
        metric_to_native, metric_to_native_ref, span_to_native, span_to_native_ref,
    };
    use crate::error::Error;
    use crate::id::{SpanId, TraceId};
    use crate::level::Level;
    use crate::log::LogRecord;
    use crate::log::body::LogBody;
    use crate::metric::{MetricSample, NumberDataPoint};
    use crate::out::native::{FrameSink, NativeAttr, NativeAttrValue, NativeOverflowAttr};
    use crate::tag::{ScalarValue, Tag};
    use crate::trace::link::SpanLink;
    use crate::trace::status::Status;
    use crate::trace::{EventRecord, SpanKind, SpanRecord};

    fn make_span_record(attr_count: usize) -> SpanRecord {
        let trace_id = TraceId::from_bytes([1u8; 16]);
        let span_id = SpanId::from_bytes([2u8; 8]);
        let attrs = (0..attr_count)
            .map(|index| Tag::Scalar {
                key: "key",
                value: ScalarValue::U64(index as u64),
            })
            .collect();
        SpanRecord {
            trace_id,
            span_id,
            parent_span_id: None,
            name: "test_span",
            kind: SpanKind::Internal,
            start_ns: 1_000_000,
            duration_ns: 500_000,
            status: Status::Ok,
            attrs,
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: crate::trace::TraceState::empty(),
            module_path: "my::module",
            file_line: (42, 1),
        }
    }

    // parity (principle 14): the zero-alloc borrowed encode must produce a
    // byte-identical frame to the owned path, and those bytes must round-trip
    // back to the owned frame. Mixed attr types + Error status exercise every
    // borrow path (name, module_path, status_reason, Str/Bytes tag values).
    #[test]
    fn native_ref_span_encode_is_byte_identical_and_roundtrips() {
        let attrs = vec![
            Tag::Scalar {
                key: "http.method",
                value: ScalarValue::Str("POST"),
            },
            Tag::Scalar {
                key: "http.status",
                value: ScalarValue::U64(200),
            },
            Tag::Scalar {
                key: "retry",
                value: ScalarValue::I64(-1),
            },
            Tag::Scalar {
                key: "ok",
                value: ScalarValue::Bool(true),
            },
            Tag::Scalar {
                key: "body",
                value: ScalarValue::Bytes(bytes::Bytes::from_static(b"\x00\x01\x02")),
            },
        ]
        .into_iter()
        .collect();
        let record = SpanRecord {
            trace_id: TraceId::from_bytes([9u8; 16]),
            span_id: SpanId::from_bytes([8u8; 8]),
            parent_span_id: Some(SpanId::from_bytes([7u8; 8])),
            name: "mixed",
            kind: SpanKind::Client,
            start_ns: 5,
            duration_ns: 6,
            status: Status::Error { reason: "boom" },
            attrs,
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: crate::trace::TraceState::empty(),
            module_path: "m::p",
            file_line: (1, 2),
        };

        let owned = NativeFrame {
            kind: 0x07,
            seq: 42,
            payload: NativePayload::Span(span_to_native(&record)),
        };
        let borrowed = NativeFrameRef {
            kind: 0x07,
            seq: 42,
            payload: NativePayloadRef::Span(span_to_native_ref(&record)),
        };

        let mut owned_buf = [0u8; NATIVE_FRAME_SIZE];
        let mut ref_buf = [0u8; NATIVE_FRAME_SIZE];
        let owned_bytes = postcard::to_slice(&owned, &mut owned_buf).unwrap();
        let ref_bytes = postcard::to_slice(&borrowed, &mut ref_buf).unwrap();

        assert_eq!(
            owned_bytes, ref_bytes,
            "borrowed encode must be byte-identical to owned"
        );
        let decoded: NativeFrame = postcard::from_bytes(ref_bytes).unwrap();
        assert_eq!(
            decoded, owned,
            "borrowed bytes must round-trip to the owned frame"
        );
    }

    // assert the borrowed encode of a payload is byte-identical to the owned
    // encode and round-trips to the owned frame.
    fn assert_payload_identical(kind: u8, owned: NativePayload, borrowed: NativePayloadRef<'_>) {
        let owned_frame = NativeFrame {
            kind,
            seq: 9,
            payload: owned,
        };
        let ref_frame = NativeFrameRef {
            kind,
            seq: 9,
            payload: borrowed,
        };
        let mut owned_buf = [0u8; NATIVE_FRAME_SIZE];
        let mut ref_buf = [0u8; NATIVE_FRAME_SIZE];
        let owned_bytes = postcard::to_slice(&owned_frame, &mut owned_buf).unwrap();
        let ref_bytes = postcard::to_slice(&ref_frame, &mut ref_buf).unwrap();
        assert_eq!(
            owned_bytes, ref_bytes,
            "borrowed encode must be byte-identical to owned (kind {kind})"
        );
        let decoded: NativeFrame = postcard::from_bytes(ref_bytes).unwrap();
        assert_eq!(
            decoded, owned_frame,
            "borrowed bytes must round-trip (kind {kind})"
        );
    }

    // proof-keeps-pace: dispatch_native switched ALL record kinds to the borrowed
    // ref encode, so every kind needs the byte-identity proof, not just spans.
    #[test]
    fn native_ref_event_log_metric_link_byte_identical() {
        let span_id = SpanId::from_bytes([3u8; 8]);
        let trace_id = TraceId::from_bytes([4u8; 16]);

        let event = EventRecord {
            parent_span_id: span_id,
            name: "ev",
            ts_ns: 100,
            attrs: vec![
                Tag::Scalar {
                    key: "k.str",
                    value: ScalarValue::Str("v"),
                },
                Tag::Scalar {
                    key: "k.bytes",
                    value: ScalarValue::Bytes(bytes::Bytes::from_static(b"xy")),
                },
            ]
            .into_iter()
            .collect(),
            module_path: "m::p",
            file_line: (1, 1),
        };
        assert_payload_identical(
            1,
            NativePayload::Event(event_to_native(&event)),
            NativePayloadRef::Event(event_to_native_ref(&event)),
        );

        let log = make_log_record(4);
        assert_payload_identical(
            2,
            NativePayload::Log(log_to_native(&log)),
            NativePayloadRef::Log(log_to_native_ref(&log)),
        );

        let counter = MetricSample::Counter(NumberDataPoint {
            value: ScalarValue::I64(-9),
            attrs: vec![Tag::Scalar {
                key: "dim",
                value: ScalarValue::Str("opus"),
            }]
            .into_iter()
            .collect(),
            ts_ns: 1000,
            start_ts_ns: 5,
        });
        assert_payload_identical(
            3,
            NativePayload::Metric(metric_to_native(&counter)),
            NativePayloadRef::Metric(metric_to_native_ref(&counter)),
        );

        let mut link = SpanLink::new(trace_id, span_id);
        link.attrs = vec![Tag::Scalar {
            key: "rel",
            value: ScalarValue::U64(1),
        }]
        .into_iter()
        .collect();
        assert_payload_identical(
            4,
            NativePayload::Link(link_to_native(&link)),
            NativePayloadRef::Link(link_to_native_ref(&link)),
        );
    }

    fn make_log_record(attr_count: usize) -> LogRecord {
        let attrs = (0..attr_count)
            .map(|index| Tag::Scalar {
                key: "key",
                value: ScalarValue::I64(index as i64),
            })
            .collect();
        LogRecord {
            ts_ns: 2_000_000,
            observed_ts_ns: 2_001_000,
            level: Level::INFO,
            body: LogBody::Text("test log message"),
            attrs,
            trace_id: None,
            span_id: None,
            trace_flags: crate::id::TraceFlags::NOT_SAMPLED,
            module_path: "my::module",
            file_line: (10, 1),
        }
    }

    struct CaptureSink {
        count: Arc<AtomicUsize>,
        last: std::sync::Mutex<Option<[u8; NATIVE_FRAME_SIZE]>>,
    }

    impl CaptureSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                count: Arc::new(AtomicUsize::new(0)),
                last: std::sync::Mutex::new(None),
            })
        }

        fn frame_count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }

        fn last_frame(&self) -> Option<[u8; NATIVE_FRAME_SIZE]> {
            *self.last.lock().unwrap()
        }
    }

    impl FrameSink for Arc<CaptureSink> {
        fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
            self.count.fetch_add(1, Ordering::Relaxed);
            *self.last.lock().unwrap() = Some(*frame);
        }
    }

    // 1. happy: encode SpanRecord → 1500-byte frame → decode back → semantic equality
    #[test]
    fn happy_span_round_trips() {
        let record = make_span_record(8);
        let native = span_to_native(&record);

        let frame_msg = NativeFrame {
            kind: 0,
            seq: 0,
            payload: NativePayload::Span(native),
        };
        let buf = encode_frame(&frame_msg).expect("encode failed");
        assert_eq!(buf.len(), NATIVE_FRAME_SIZE);

        let decoded = decode_frame(&buf).expect("decode failed");
        let NativePayload::Span(span) = decoded.payload else {
            panic!("expected Span payload");
        };
        assert_eq!(span.trace_id, [1u8; 16]);
        assert_eq!(span.span_id, [2u8; 8]);
        assert_eq!(span.name, b"test_span");
        assert_eq!(span.attrs.len(), 8);
        assert_eq!(span.status, 1);
    }

    // 2. happy: each kind variant round-trips
    #[test]
    fn happy_all_variants_round_trip() {
        let span_id = SpanId::from_bytes([3u8; 8]);
        let trace_id = TraceId::from_bytes([4u8; 16]);

        let event = EventRecord {
            parent_span_id: span_id,
            name: "event_name",
            ts_ns: 100,
            attrs: smallvec::SmallVec::new(),
            module_path: "m",
            file_line: (1, 1),
        };
        let event_native = event_to_native(&event);
        let frame = NativeFrame {
            kind: 1,
            seq: 0,
            payload: NativePayload::Event(event_native),
        };
        let decoded = decode_frame(&encode_frame(&frame).unwrap()).unwrap();
        assert!(matches!(decoded.payload, NativePayload::Event(_)));

        let log = make_log_record(4);
        let log_native = log_to_native(&log);
        let frame = NativeFrame {
            kind: 2,
            seq: 0,
            payload: NativePayload::Log(log_native),
        };
        let decoded = decode_frame(&encode_frame(&frame).unwrap()).unwrap();
        assert!(matches!(decoded.payload, NativePayload::Log(_)));

        let counter = MetricSample::Counter(NumberDataPoint {
            value: ScalarValue::U64(42),
            attrs: smallvec::SmallVec::new(),
            ts_ns: 1000,
            start_ts_ns: 0,
        });
        let metric_native = metric_to_native(&counter);
        let frame = NativeFrame {
            kind: 3,
            seq: 0,
            payload: NativePayload::Metric(metric_native),
        };
        let decoded = decode_frame(&encode_frame(&frame).unwrap()).unwrap();
        assert!(matches!(decoded.payload, NativePayload::Metric(_)));

        let link = SpanLink::new(trace_id, span_id);
        let link_native = link_to_native(&link);
        let frame = NativeFrame {
            kind: 4,
            seq: 0,
            payload: NativePayload::Link(link_native),
        };
        let decoded = decode_frame(&encode_frame(&frame).unwrap()).unwrap();
        assert!(matches!(decoded.payload, NativePayload::Link(_)));

        let overflow = NativeOverflowAttr {
            parent_span_id: [5u8; 8],
            attrs: vec![NativeAttr {
                key: b"extra".to_vec(),
                value: NativeAttrValue::U64(1),
            }],
        };
        let frame = NativeFrame {
            kind: 5,
            seq: 0,
            payload: NativePayload::OverflowAttr(overflow),
        };
        let decoded = decode_frame(&encode_frame(&frame).unwrap()).unwrap();
        assert!(matches!(decoded.payload, NativePayload::OverflowAttr(_)));
    }

    // 3. edge: payload that fits in <1500 bytes is zero-padded after postcard end
    #[test]
    fn edge_small_payload_is_zero_padded() {
        let record = make_span_record(0);
        let native = span_to_native(&record);
        let frame_msg = NativeFrame {
            kind: 0,
            seq: 0,
            payload: NativePayload::Span(native),
        };
        let buf = encode_frame(&frame_msg).unwrap();

        let encoded_size = postcard::to_allocvec(&frame_msg).unwrap().len();
        assert!(
            encoded_size < NATIVE_FRAME_SIZE,
            "small payload should not fill the frame"
        );

        let trailing_zeros = buf[encoded_size..].iter().all(|&byte| byte == 0);
        assert!(trailing_zeros, "bytes after postcard data must be zero");
    }

    // 4. edge: payload that exceeds 1500 bytes returns Error::InvalidInput
    #[test]
    fn edge_oversized_payload_returns_error() {
        let record = make_span_record(128);
        let native = span_to_native(&record);
        let frame_msg = NativeFrame {
            kind: 0,
            seq: 0,
            payload: NativePayload::Span(native),
        };

        let raw_size = postcard::to_allocvec(&frame_msg).unwrap().len();
        if raw_size > NATIVE_FRAME_SIZE {
            let result = encode_frame(&frame_msg);
            assert_eq!(result, Err(Error::InvalidInput));
        }
    }

    // 5. size delta: 8-attr native SpanRecord is smaller than opentelemetry-proto encoding
    #[test]
    fn size_native_smaller_than_otlp_proto() {
        use opentelemetry_proto::tonic::trace::v1 as otlp_trace;
        use prost::Message as _;

        let record = make_span_record(8);
        let native = span_to_native(&record);
        let frame_msg = NativeFrame {
            kind: 0,
            seq: 0,
            payload: NativePayload::Span(native),
        };
        let native_bytes = postcard::to_allocvec(&frame_msg).unwrap();

        let otlp_attrs: Vec<opentelemetry_proto::tonic::common::v1::KeyValue> = (0..8u64)
            .map(|index| opentelemetry_proto::tonic::common::v1::KeyValue {
                key: "key".to_string(),
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

        let otlp_span = otlp_trace::Span {
            trace_id: vec![1u8; 16],
            span_id: vec![2u8; 8],
            parent_span_id: vec![],
            name: "test_span".to_string(),
            kind: otlp_trace::span::SpanKind::Internal as i32,
            start_time_unix_nano: 1_000_000,
            end_time_unix_nano: 1_500_000,
            attributes: otlp_attrs,
            status: None,
            ..Default::default()
        };

        let otlp_bytes = otlp_span.encode_to_vec();

        std::eprintln!(
            "size comparison — native: {}B  otlp-proto: {}B  delta: {}B",
            native_bytes.len(),
            otlp_bytes.len(),
            otlp_bytes.len() as i64 - native_bytes.len() as i64,
        );

        assert!(
            native_bytes.len() < otlp_bytes.len(),
            "native ({} B) should be smaller than otlp-proto ({} B)",
            native_bytes.len(),
            otlp_bytes.len()
        );
    }

    // 6. bidirectional: NativeControlMessage encodes and decodes
    #[test]
    fn bidirectional_control_message_round_trips() {
        let msg = NativeControlMessage::SetLevelOverride {
            module_prefix: b"my::module".to_vec(),
            severity: 9,
        };
        let encoded = postcard::to_allocvec(&msg).unwrap();
        let decoded: NativeControlMessage = postcard::from_bytes(&encoded).unwrap();
        let NativeControlMessage::SetLevelOverride {
            module_prefix,
            severity,
        } = decoded
        else {
            panic!("wrong variant");
        };
        assert_eq!(module_prefix, b"my::module");
        assert_eq!(severity, 9);

        let flush = NativeControlMessage::Flush;
        let enc2 = postcard::to_allocvec(&flush).unwrap();
        let dec2: NativeControlMessage = postcard::from_bytes(&enc2).unwrap();
        assert!(matches!(dec2, NativeControlMessage::Flush));

        let rate = NativeControlMessage::SetSamplingRate {
            fraction_per_million: 500_000,
        };
        let enc3 = postcard::to_allocvec(&rate).unwrap();
        let dec3: NativeControlMessage = postcard::from_bytes(&enc3).unwrap();
        let NativeControlMessage::SetSamplingRate {
            fraction_per_million,
        } = dec3
        else {
            panic!("wrong variant");
        };
        assert_eq!(fraction_per_million, 500_000);
    }

    // 7. schema version: high-nibble in kind byte round-trips
    #[test]
    fn schema_version_high_nibble_round_trips() {
        let record = make_span_record(1);
        let native = span_to_native(&record);
        let schema_v = 3u8;
        let low = 0u8;
        let kind = (schema_v << 4) | low;
        let frame_msg = NativeFrame {
            kind,
            seq: 0,
            payload: NativePayload::Span(native),
        };
        let buf = encode_frame(&frame_msg).unwrap();
        let decoded = decode_frame(&buf).unwrap();
        assert_eq!(decoded.kind >> 4, schema_v);
        assert_eq!(decoded.kind & 0x0f, low);
    }

    // 8. seq monotonicity: NativeExporter increments seq per emitted frame
    #[test]
    fn seq_monotonicity_increments_per_frame() {
        let sink = CaptureSink::new();
        let exporter = NativeExporter::new(Arc::clone(&sink));

        let record = make_span_record(2);
        let native = span_to_native(&record);
        exporter.encode_and_emit_payload(NativePayload::Span(native));
        let first_frame = sink.last_frame().unwrap();
        let first = decode_frame(&first_frame).unwrap();

        let native2 = span_to_native(&record);
        exporter.encode_and_emit_payload(NativePayload::Span(native2));
        let second_frame = sink.last_frame().unwrap();
        let second = decode_frame(&second_frame).unwrap();

        assert_eq!(first.seq, 0);
        assert_eq!(second.seq, 1);
        assert_eq!(sink.frame_count(), 2);
    }
}
