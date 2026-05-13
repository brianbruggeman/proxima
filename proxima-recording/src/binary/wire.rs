use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::event::{
    CacheOutcome, FrameMetadata, HttpEvent, InteractionId, PipelineEvent, PipelineOutcome,
    ProcessCwd, ProcessEvent, ProtocolEvent, RECORDING_FORMAT_VERSION, RecordMeta, RecordingEvent,
    RequestHeader,
};
use proxima_core::ProximaError;

/// High bit of a frame's `u32` length prefix: set = payload is stored raw
/// (uncompressed), clear = payload is a zstd frame. Old recordings predate
/// stored frames and have it clear (everything was zstd), so they stay readable
/// — only new sub-threshold frames set the bit. Frames are therefore capped at
/// `FRAME_LEN_MASK` (2 GiB), far above any real event.
// consumed by binary::bin_format / binary::source, both std-gated (file I/O)
// so both are dead from the alloc-only tier's point of view; the wire codec
// itself never reads its own frame-length flag.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) const FRAME_STORED_FLAG: u32 = 0x8000_0000;
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) const FRAME_LEN_MASK: u32 = 0x7fff_ffff;

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct BinEnvelope {
    pub version: u32,
    pub id: InteractionId,
    pub ts_ms: u64,
    pub parent: Option<InteractionId>,
    pub event: BinProtocolEvent,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum BinProtocolEvent {
    Pipeline(BinPipelineEvent),
    Process(BinProcessEvent),
    Http(BinHttpEvent),
    Custom { kind: String, payload_json: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum BinPipelineEvent {
    Started {
        ts_unix_nanos_lo: u64,
        ts_unix_nanos_hi: u64,
        ts_negative: bool,
        spec_hash: [u8; 32],
        name: Option<String>,
    },
    Ended {
        outcome: BinPipelineOutcome,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum BinPipelineOutcome {
    Completed,
    Failed { reason: String },
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum BinProcessEvent {
    Started {
        ts_unix_nanos_lo: u64,
        ts_unix_nanos_hi: u64,
        ts_negative: bool,
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<ProcessCwd>,
    },
    Stdout {
        data: Vec<u8>,
    },
    Stderr {
        data: Vec<u8>,
    },
    Exited {
        exit_code: Option<i32>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum BinHttpEvent {
    Started {
        ts_unix_nanos_lo: u64,
        ts_unix_nanos_hi: u64,
        ts_negative: bool,
        pipe: String,
        request: BinRequestHeader,
        meta: Option<BinRecordMeta>,
    },
    RequestChunk {
        data: Vec<u8>,
        metadata: Vec<(String, Vec<u8>)>,
    },
    RequestEnded,
    ResponseStarted {
        status: u16,
        headers: Vec<(String, String)>,
    },
    ResponseChunk {
        data: Vec<u8>,
        metadata: Vec<(String, Vec<u8>)>,
    },
    Ended {
        latency_ms: u64,
        meta: BinRecordMeta,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct BinRequestHeader {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub query: Vec<(String, String)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct BinRecordMeta {
    pub cache: Option<CacheOutcome>,
    pub retries: u32,
    pub upstream: Option<String>,
    pub instance_id: Option<String>,
    #[serde(default)]
    pub source: Option<crate::event::EventSource>,
    /// W3C trace-id bytes (16 bytes). `None` when the request arrived
    /// without a traceparent header or before c15-prime-hooks typed the
    /// field. `#[serde(default)]` preserves backward compat with fixtures
    /// written before this field existed.
    #[serde(default)]
    pub trace_id: Option<[u8; 16]>,
    /// span-id of this request's top-level span (8 bytes). same backward-
    /// compat guarantee as `trace_id`.
    #[serde(default)]
    pub span_id: Option<[u8; 8]>,
    /// parent span-id carried from the upstream caller (8 bytes).
    #[serde(default)]
    pub parent_span_id: Option<[u8; 8]>,
    pub extra_json: Option<String>,
}

pub(super) fn event_to_bin(event: RecordingEvent) -> BinEnvelope {
    let RecordingEvent {
        id,
        ts_ms,
        parent,
        event,
    } = event;
    BinEnvelope {
        version: RECORDING_FORMAT_VERSION,
        id,
        ts_ms,
        parent,
        event: protocol_to_bin(event),
    }
}

pub(super) fn bin_to_event(envelope: BinEnvelope) -> Result<RecordingEvent, ProximaError> {
    if envelope.version != RECORDING_FORMAT_VERSION {
        return Err(ProximaError::Record(format!(
            "unsupported recording version: expected {}, got {}",
            RECORDING_FORMAT_VERSION, envelope.version,
        )));
    }
    Ok(RecordingEvent {
        id: envelope.id,
        ts_ms: envelope.ts_ms,
        parent: envelope.parent,
        event: bin_to_protocol(envelope.event)?,
    })
}

fn protocol_to_bin(event: ProtocolEvent) -> BinProtocolEvent {
    match event {
        ProtocolEvent::Pipeline(inner) => BinProtocolEvent::Pipeline(pipeline_to_bin(inner)),
        ProtocolEvent::Process(inner) => BinProtocolEvent::Process(process_to_bin(inner)),
        ProtocolEvent::Http(inner) => BinProtocolEvent::Http(http_to_bin(inner)),
        ProtocolEvent::Custom { kind, payload } => {
            let payload_json =
                serde_json::to_string(&payload).unwrap_or_else(|_| String::from("null"));
            BinProtocolEvent::Custom { kind, payload_json }
        }
    }
}

fn bin_to_protocol(event: BinProtocolEvent) -> Result<ProtocolEvent, ProximaError> {
    Ok(match event {
        BinProtocolEvent::Pipeline(inner) => ProtocolEvent::Pipeline(bin_to_pipeline(inner)?),
        BinProtocolEvent::Process(inner) => ProtocolEvent::Process(bin_to_process(inner)?),
        BinProtocolEvent::Http(inner) => ProtocolEvent::Http(bin_to_http(inner)?),
        BinProtocolEvent::Custom { kind, payload_json } => {
            let payload = serde_json::from_str(&payload_json)
                .map_err(|err| ProximaError::Record(format!("decode custom payload: {err}")))?;
            ProtocolEvent::Custom { kind, payload }
        }
    })
}

fn pipeline_to_bin(event: PipelineEvent) -> BinPipelineEvent {
    match event {
        PipelineEvent::Started {
            ts,
            spec_hash,
            name,
        } => {
            let (ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative) =
                split_unix_nanos(ts.unix_timestamp_nanos());
            BinPipelineEvent::Started {
                ts_unix_nanos_lo,
                ts_unix_nanos_hi,
                ts_negative,
                spec_hash,
                name,
            }
        }
        PipelineEvent::Ended { outcome } => BinPipelineEvent::Ended {
            outcome: pipeline_outcome_to_bin(outcome),
        },
    }
}

fn bin_to_pipeline(event: BinPipelineEvent) -> Result<PipelineEvent, ProximaError> {
    Ok(match event {
        BinPipelineEvent::Started {
            ts_unix_nanos_lo,
            ts_unix_nanos_hi,
            ts_negative,
            spec_hash,
            name,
        } => {
            let unix_nanos = combine_unix_nanos(ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative);
            let ts = OffsetDateTime::from_unix_timestamp_nanos(unix_nanos)
                .map_err(|err| ProximaError::Record(format!("invalid bin timestamp: {err}")))?;
            PipelineEvent::Started {
                ts,
                spec_hash,
                name,
            }
        }
        BinPipelineEvent::Ended { outcome } => PipelineEvent::Ended {
            outcome: bin_to_pipeline_outcome(outcome),
        },
    })
}

fn pipeline_outcome_to_bin(outcome: PipelineOutcome) -> BinPipelineOutcome {
    match outcome {
        PipelineOutcome::Completed => BinPipelineOutcome::Completed,
        PipelineOutcome::Failed { reason } => BinPipelineOutcome::Failed { reason },
        PipelineOutcome::Cancelled => BinPipelineOutcome::Cancelled,
    }
}

fn bin_to_pipeline_outcome(outcome: BinPipelineOutcome) -> PipelineOutcome {
    match outcome {
        BinPipelineOutcome::Completed => PipelineOutcome::Completed,
        BinPipelineOutcome::Failed { reason } => PipelineOutcome::Failed { reason },
        BinPipelineOutcome::Cancelled => PipelineOutcome::Cancelled,
    }
}

fn process_to_bin(event: ProcessEvent) -> BinProcessEvent {
    match event {
        ProcessEvent::Started {
            ts,
            command,
            args,
            env,
            cwd,
        } => {
            let (ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative) =
                split_unix_nanos(ts.unix_timestamp_nanos());
            BinProcessEvent::Started {
                ts_unix_nanos_lo,
                ts_unix_nanos_hi,
                ts_negative,
                command,
                args,
                env: env.into_iter().collect(),
                cwd,
            }
        }
        ProcessEvent::Stdout(data) => BinProcessEvent::Stdout {
            data: data.to_vec(),
        },
        ProcessEvent::Stderr(data) => BinProcessEvent::Stderr {
            data: data.to_vec(),
        },
        ProcessEvent::Exited { exit_code } => BinProcessEvent::Exited { exit_code },
    }
}

fn bin_to_process(event: BinProcessEvent) -> Result<ProcessEvent, ProximaError> {
    Ok(match event {
        BinProcessEvent::Started {
            ts_unix_nanos_lo,
            ts_unix_nanos_hi,
            ts_negative,
            command,
            args,
            env,
            cwd,
        } => {
            let unix_nanos = combine_unix_nanos(ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative);
            let ts = OffsetDateTime::from_unix_timestamp_nanos(unix_nanos)
                .map_err(|err| ProximaError::Record(format!("invalid bin timestamp: {err}")))?;
            ProcessEvent::Started {
                ts,
                command,
                args,
                env: env.into_iter().collect::<BTreeMap<_, _>>(),
                cwd,
            }
        }
        BinProcessEvent::Stdout { data } => ProcessEvent::Stdout(Bytes::from(data)),
        BinProcessEvent::Stderr { data } => ProcessEvent::Stderr(Bytes::from(data)),
        BinProcessEvent::Exited { exit_code } => ProcessEvent::Exited { exit_code },
    })
}

fn http_to_bin(event: HttpEvent) -> BinHttpEvent {
    match event {
        HttpEvent::Started {
            ts,
            pipe,
            request,
            meta,
        } => {
            let (ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative) =
                split_unix_nanos(ts.unix_timestamp_nanos());
            BinHttpEvent::Started {
                ts_unix_nanos_lo,
                ts_unix_nanos_hi,
                ts_negative,
                pipe,
                request: request_header_to_bin(request),
                meta: meta.map(record_meta_to_bin),
            }
        }
        HttpEvent::RequestChunk { data, metadata } => BinHttpEvent::RequestChunk {
            data: data.to_vec(),
            metadata: encode_metadata(metadata),
        },
        HttpEvent::RequestEnded => BinHttpEvent::RequestEnded,
        HttpEvent::ResponseStarted { status, headers } => {
            BinHttpEvent::ResponseStarted { status, headers }
        }
        HttpEvent::ResponseChunk { data, metadata } => BinHttpEvent::ResponseChunk {
            data: data.to_vec(),
            metadata: encode_metadata(metadata),
        },
        HttpEvent::Ended { latency_ms, meta } => BinHttpEvent::Ended {
            latency_ms,
            meta: record_meta_to_bin(meta),
        },
    }
}

fn bin_to_http(event: BinHttpEvent) -> Result<HttpEvent, ProximaError> {
    Ok(match event {
        BinHttpEvent::Started {
            ts_unix_nanos_lo,
            ts_unix_nanos_hi,
            ts_negative,
            pipe,
            request,
            meta,
        } => {
            let unix_nanos = combine_unix_nanos(ts_unix_nanos_lo, ts_unix_nanos_hi, ts_negative);
            let ts = OffsetDateTime::from_unix_timestamp_nanos(unix_nanos)
                .map_err(|err| ProximaError::Record(format!("invalid bin timestamp: {err}")))?;
            let meta = match meta {
                Some(bin) => Some(bin_to_record_meta(bin)?),
                None => None,
            };
            HttpEvent::Started {
                ts,
                pipe,
                request: bin_to_request_header(request),
                meta,
            }
        }
        BinHttpEvent::RequestChunk { data, metadata } => HttpEvent::RequestChunk {
            data: Bytes::from(data),
            metadata: decode_metadata(metadata),
        },
        BinHttpEvent::RequestEnded => HttpEvent::RequestEnded,
        BinHttpEvent::ResponseStarted { status, headers } => {
            HttpEvent::ResponseStarted { status, headers }
        }
        BinHttpEvent::ResponseChunk { data, metadata } => HttpEvent::ResponseChunk {
            data: Bytes::from(data),
            metadata: decode_metadata(metadata),
        },
        BinHttpEvent::Ended { latency_ms, meta } => {
            let meta = bin_to_record_meta(meta)?;
            HttpEvent::Ended { latency_ms, meta }
        }
    })
}

fn request_header_to_bin(header: RequestHeader) -> BinRequestHeader {
    BinRequestHeader {
        method: header.method,
        path: header.path,
        headers: header.headers.into_iter().collect(),
        query: header.query.into_iter().collect(),
    }
}

fn bin_to_request_header(bin: BinRequestHeader) -> RequestHeader {
    RequestHeader {
        method: bin.method,
        path: bin.path,
        headers: bin.headers.into_iter().collect(),
        query: bin.query.into_iter().collect(),
    }
}

fn record_meta_to_bin(meta: RecordMeta) -> BinRecordMeta {
    let extra_json = if meta.extra.is_empty() {
        None
    } else {
        let encoded = serde_json::to_string(&meta.extra).unwrap_or_else(|_| String::from("{}"));
        Some(encoded)
    };
    BinRecordMeta {
        cache: meta.cache,
        retries: meta.retries,
        upstream: meta.upstream,
        instance_id: meta.instance_id,
        source: meta.source,
        trace_id: None,
        span_id: None,
        parent_span_id: None,
        extra_json,
    }
}

fn bin_to_record_meta(bin: BinRecordMeta) -> Result<RecordMeta, ProximaError> {
    let extra = match bin.extra_json {
        Some(raw) => serde_json::from_str(&raw)
            .map_err(|err| ProximaError::Record(format!("decode bin meta extra: {err}")))?,
        None => Default::default(),
    };
    Ok(RecordMeta {
        cache: bin.cache,
        retries: bin.retries,
        upstream: bin.upstream,
        instance_id: bin.instance_id,
        source: bin.source,
        extra,
    })
}

fn encode_metadata(metadata: FrameMetadata) -> Vec<(String, Vec<u8>)> {
    metadata
        .into_iter()
        .map(|(key, value)| (key, value.to_vec()))
        .collect()
}

fn decode_metadata(wire: Vec<(String, Vec<u8>)>) -> FrameMetadata {
    wire.into_iter()
        .map(|(key, value)| (key, Bytes::from(value)))
        .collect()
}

fn split_unix_nanos(value: i128) -> (u64, u64, bool) {
    let negative = value < 0;
    let unsigned = if negative {
        value.unsigned_abs()
    } else {
        value as u128
    };
    let lo = unsigned as u64;
    let hi = (unsigned >> 64) as u64;
    (lo, hi, negative)
}

fn combine_unix_nanos(lo: u64, hi: u64, negative: bool) -> i128 {
    let unsigned = (u128::from(hi) << 64) | u128::from(lo);
    if negative {
        -(unsigned as i128)
    } else {
        unsigned as i128
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use rstest::rstest;
    // tests only run under the std tier (cargo nextest run); PathBuf isn't
    // available on the alloc-only no_std target, so it stays local to here
    // rather than becoming a crate-wide alloc-tier import.
    use std::path::PathBuf;

    fn make_id() -> InteractionId {
        InteractionId::from_bytes([3; 16])
    }

    fn envelope(id: InteractionId, ts_ms: u64, event: ProtocolEvent) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms,
            parent: None,
            event,
        }
    }

    fn child(
        id: InteractionId,
        parent: InteractionId,
        ts_ms: u64,
        event: ProtocolEvent,
    ) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms,
            parent: Some(parent),
            event,
        }
    }

    #[rstest]
    #[case::http_started(envelope(make_id(), 0, ProtocolEvent::Http(HttpEvent::Started {
        ts: OffsetDateTime::UNIX_EPOCH,
        pipe: "echo".into(),
        request: RequestHeader::default(),
        meta: None,
    })))]
    #[case::http_resp_chunk(envelope(make_id(), 12, ProtocolEvent::Http(HttpEvent::ResponseChunk {
        data: Bytes::from_static(b"chunk"),
        metadata: FrameMetadata::new(),
    })))]
    #[case::pipeline_started(envelope(InteractionId::from_bytes([13; 16]), 0, ProtocolEvent::Pipeline(PipelineEvent::Started {
        ts: OffsetDateTime::UNIX_EPOCH,
        spec_hash: [9; 32],
        name: Some("bench".into()),
    })))]
    #[case::pipeline_ended_failed(envelope(InteractionId::from_bytes([13; 16]), 500, ProtocolEvent::Pipeline(PipelineEvent::Ended {
        outcome: PipelineOutcome::Failed { reason: "stage bench exit 2".into() },
    })))]
    #[case::process_started(child(
        InteractionId::from_bytes([2; 16]),
        InteractionId::from_bytes([13; 16]),
        50,
        ProtocolEvent::Process(ProcessEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            command: "/bin/sh".into(),
            args: vec!["-c".into(), "echo hi".into()],
            env: [("RUST_LOG".to_string(), "debug".to_string())].into_iter().collect(),
            cwd: Some(PathBuf::from("/tmp/work")),
        }),
    ))]
    #[case::process_stdout(child(
        InteractionId::from_bytes([2; 16]),
        InteractionId::from_bytes([13; 16]),
        51,
        ProtocolEvent::Process(ProcessEvent::Stdout(Bytes::from_static(b"line of output"))),
    ))]
    #[case::process_exited_signaled(child(
        InteractionId::from_bytes([2; 16]),
        InteractionId::from_bytes([13; 16]),
        100,
        ProtocolEvent::Process(ProcessEvent::Exited { exit_code: None }),
    ))]
    #[case::custom_protocol(envelope(make_id(), 1, ProtocolEvent::Custom {
        kind: "redis".into(),
        payload: serde_json::json!({"cmd": "GET", "key": "x"}),
    }))]
    fn round_trips_through_postcard(#[case] event: RecordingEvent) {
        let envelope = event_to_bin(event.clone());
        let bytes = postcard::to_allocvec(&envelope).expect("encode postcard");
        let parsed: BinEnvelope = postcard::from_bytes(&bytes).expect("parse postcard");
        let restored = bin_to_event(parsed).expect("convert to event");
        assert_eq!(event, restored);
    }
}
