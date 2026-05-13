// C11 OTLP-gRPC encoder: reuses C10's prost encoder; wraps the protobuf body in
// the standard gRPC unary-RPC length-prefix frame (5 bytes: compression flag +
// u32 big-endian length). v1 is encode-only: tonic transport, retries, and h2
// flow control are out of scope — caller wires those on top.
//
// Frame layout per https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md
//   byte 0:      compression flag (0 = uncompressed)
//   bytes 1..5:  message length, u32 big-endian
//   bytes 5..N:  protobuf-encoded ExportTraceServiceRequest (or Logs/Metrics)
//
// Production gRPC clients (tonic, grpc-rust, h2 + manual framing) all build on
// top of this same wire shape. Our home-turf bench arm engages opentelemetry-otlp's
// tonic-based encode path so the comparison is against the incumbent's design
// point, not a strawman.

use alloc::string::String;
use alloc::vec::Vec;

use bytes::Bytes;
use parking_lot::Mutex;

use super::otlp_http::OtlpHttpExporter;

/// Length of the gRPC frame header (compression flag + u32 BE length).
pub const GRPC_FRAME_HEADER_LEN: usize = 5;

/// gRPC-framed OTLP exporter. Encodes the same OTLP protobuf body as
/// [`OtlpHttpExporter`] then prepends the 5-byte gRPC frame header.
pub struct OtlpGrpcExporter {
    /// Reuses the C10 encoder for the protobuf body; we only add framing.
    inner: OtlpHttpExporter,
    pending: Mutex<()>, // reserved for future cross-thread coordination
}

impl OtlpGrpcExporter {
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            inner: OtlpHttpExporter::new(endpoint),
            pending: Mutex::new(()),
        }
    }

    /// Endpoint URL.
    pub fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    /// Encode pending spans as a gRPC-framed `ExportTraceServiceRequest`.
    /// Returns empty `Bytes` if no spans are pending.
    #[must_use]
    pub fn encode_spans(&self) -> Bytes {
        let _guard = self.pending.lock();
        frame_grpc(self.inner.encode_spans())
    }

    /// Encode pending logs as a gRPC-framed `ExportLogsServiceRequest`.
    #[must_use]
    pub fn encode_logs(&self) -> Bytes {
        let _guard = self.pending.lock();
        frame_grpc(self.inner.encode_logs())
    }

    /// Encode pending metrics as a gRPC-framed `ExportMetricsServiceRequest`.
    #[must_use]
    pub fn encode_metrics(&self) -> Bytes {
        let _guard = self.pending.lock();
        frame_grpc(self.inner.encode_metrics())
    }

    /// Borrow the inner OTLP-HTTP encoder. Used by OtlpGrpcPipe to push
    /// records into the shared pending buffers.
    pub(crate) fn inner(&self) -> &OtlpHttpExporter {
        &self.inner
    }
}

/// Wrap `body` in a gRPC length-prefix frame. Returns empty `Bytes` for empty input.
///
/// Allocates a fresh Vec and copies `body` into it. For best perf when the
/// protobuf body is being produced fresh, prefer [`encode_grpc_framed`] which
/// reserves the 5-byte header in-place and lets the encoder append directly,
/// avoiding the copy.
pub fn frame_grpc(body: Bytes) -> Bytes {
    if body.is_empty() {
        return Bytes::new();
    }
    let body_len = body.len();
    let mut framed = Vec::with_capacity(GRPC_FRAME_HEADER_LEN + body_len);
    framed.push(0u8); // compression flag — uncompressed
    let length_bytes = (body_len as u32).to_be_bytes();
    framed.extend_from_slice(&length_bytes);
    framed.extend_from_slice(&body);
    Bytes::from(framed)
}

/// Encode `request` directly into a gRPC-framed buffer in a SINGLE allocation.
///
/// Layout: reserves the 5-byte header up front, lets prost append the body,
/// then writes the actual length into bytes 1..5. No intermediate Bytes wraps,
/// no body copy — matches the allocation pattern of `prost::encode_to_vec` +
/// manual `Vec::extend_from_slice` (the tonic/opentelemetry-otlp home-turf path).
pub fn encode_grpc_framed<M: prost::Message>(request: &M) -> Result<Vec<u8>, prost::EncodeError> {
    let body_len = request.encoded_len();
    let mut framed = Vec::with_capacity(GRPC_FRAME_HEADER_LEN + body_len);
    framed.push(0u8); // compression flag — uncompressed
    framed.extend_from_slice(&(body_len as u32).to_be_bytes());
    request.encode(&mut framed)?;
    Ok(framed)
}

/// Parse the gRPC frame header from `framed`. Returns `(compression_flag, body_len)`
/// or `None` if the buffer is too short or malformed.
#[must_use]
pub fn parse_grpc_header(framed: &[u8]) -> Option<(u8, u32)> {
    if framed.len() < GRPC_FRAME_HEADER_LEN {
        return None;
    }
    let compression = framed[0];
    let length = u32::from_be_bytes([framed[1], framed[2], framed[3], framed[4]]);
    Some((compression, length))
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

    use super::*;
    use crate::id::{SpanId, TraceId};
    use crate::metric::sample::MetricSample;
    use crate::out::otlp_http::conv::{log_to_proto, metric_to_proto, span_to_proto};
    use crate::tag::{ScalarValue, Tag};
    use crate::trace::kind::SpanKind;
    use crate::trace::span::SpanRecord;
    use crate::trace::status::Status;
    use crate::trace::tracestate::TraceState;

    fn sample_span() -> SpanRecord {
        SpanRecord {
            trace_id: TraceId::from_bytes([0x11; 16]),
            span_id: SpanId::from_bytes([0x22; 8]),
            parent_span_id: None,
            name: "test.span",
            kind: SpanKind::Server,
            start_ns: 1_000_000,
            duration_ns: 5_000_000,
            status: Status::Ok,
            attrs: smallvec::smallvec![Tag::Scalar {
                key: "k",
                value: ScalarValue::I64(42),
            }],
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: TraceState::empty(),
            module_path: "test",
            file_line: (1, 1),
        }
    }

    // 1. happy: empty encode returns empty Bytes
    #[test]
    fn empty_encode_returns_empty_bytes() {
        let exporter = OtlpGrpcExporter::new("http://localhost:4317");
        assert_eq!(exporter.encode_spans().len(), 0);
        assert_eq!(exporter.encode_logs().len(), 0);
        assert_eq!(exporter.encode_metrics().len(), 0);
    }

    // 2. happy: encoding one span produces a gRPC frame
    #[test]
    fn one_span_produces_grpc_frame() {
        let exporter = OtlpGrpcExporter::new("http://localhost:4317");
        exporter
            .inner()
            .pending_spans
            .lock()
            .push(span_to_proto(&sample_span()));
        let framed = exporter.encode_spans();
        assert!(framed.len() > GRPC_FRAME_HEADER_LEN);

        let (compression, body_len) = parse_grpc_header(&framed).unwrap();
        assert_eq!(compression, 0);
        assert_eq!(body_len as usize, framed.len() - GRPC_FRAME_HEADER_LEN);
    }

    // 3. round-trip: frame_grpc + parse_grpc_header preserve body length
    #[test]
    fn frame_round_trip_preserves_body() {
        let body = Bytes::from(alloc::vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let framed = frame_grpc(body.clone());
        assert_eq!(framed.len(), GRPC_FRAME_HEADER_LEN + body.len());

        let (compression, body_len) = parse_grpc_header(&framed).unwrap();
        assert_eq!(compression, 0);
        assert_eq!(body_len as usize, body.len());
        assert_eq!(&framed[GRPC_FRAME_HEADER_LEN..], &body[..]);
    }

    // 4. edge: parse_grpc_header rejects too-short input
    #[test]
    fn parse_header_rejects_short_input() {
        assert!(parse_grpc_header(&[]).is_none());
        assert!(parse_grpc_header(&[0, 0, 0, 0]).is_none());
    }

    // 5. body-len encoding: u32 big-endian (gRPC spec)
    #[test]
    fn body_length_is_u32_big_endian() {
        let body = Bytes::from(alloc::vec![0u8; 0x01020304]);
        let framed = frame_grpc(body);
        // bytes 1..5 should be 0x01 0x02 0x03 0x04 (big-endian 0x01020304)
        assert_eq!(&framed[1..5], &[0x01u8, 0x02, 0x03, 0x04]);
    }

    // 6. logs: encoding one log produces a gRPC frame
    #[test]
    fn one_log_produces_grpc_frame() {
        use crate::id::TraceFlags;
        use crate::level::Level;
        use crate::log::LogRecord;
        use crate::log::body::LogBody;

        let exporter = OtlpGrpcExporter::new("http://localhost:4317");
        let record = LogRecord {
            ts_ns: 1_000_000,
            observed_ts_ns: 1_000_000,
            level: Level::INFO,
            body: LogBody::Text("test"),
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags::SAMPLED,
            module_path: "test",
            file_line: (1, 1),
        };
        exporter
            .inner()
            .pending_logs
            .lock()
            .push(log_to_proto(&record));
        let framed = exporter.encode_logs();
        assert!(framed.len() > GRPC_FRAME_HEADER_LEN);
        let (_, body_len) = parse_grpc_header(&framed).unwrap();
        assert_eq!(body_len as usize, framed.len() - GRPC_FRAME_HEADER_LEN);
    }

    // 7. metrics: encoding one counter produces a gRPC frame
    #[test]
    fn one_metric_produces_grpc_frame() {
        let exporter = OtlpGrpcExporter::new("http://localhost:4317");
        let sample = MetricSample::Counter(crate::metric::NumberDataPoint {
            value: ScalarValue::U64(1),
            attrs: smallvec::SmallVec::new(),
            ts_ns: 0,
            start_ts_ns: 0,
        });
        exporter
            .inner()
            .pending_metrics
            .lock()
            .push(metric_to_proto(&sample));
        let framed = exporter.encode_metrics();
        assert!(framed.len() > GRPC_FRAME_HEADER_LEN);
    }
}
