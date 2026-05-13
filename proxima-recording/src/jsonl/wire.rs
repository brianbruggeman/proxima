use std::collections::BTreeMap;
use std::path::PathBuf;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::event::{
    FrameMetadata, HttpEvent, InteractionId, PipelineEvent, PipelineOutcome, ProcessEvent,
    ProtocolEvent, RECORDING_FORMAT_VERSION, RecordMeta, RecordingEvent, RequestHeader,
};
use proxima_core::ProximaError;

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WireEnvelope {
    pub v: u32,
    pub id: InteractionId,
    pub ts_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<InteractionId>,
    #[serde(flatten)]
    pub event: WireProtocolEvent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "proto", rename_all = "snake_case")]
pub(super) enum WireProtocolEvent {
    Pipeline(WirePipelineEvent),
    Process(WireProcessEvent),
    Http(WireHttpEvent),
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub(super) enum WirePipelineEvent {
    Started {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        spec_hash_hex: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Ended {
        outcome: PipelineOutcome,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub(super) enum WireProcessEvent {
    Started {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        command: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
    },
    Stdout {
        b64: String,
    },
    Stderr {
        b64: String,
    },
    Exited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub(super) enum WireHttpEvent {
    Started {
        #[serde(with = "time::serde::rfc3339")]
        ts: OffsetDateTime,
        pipe: String,
        request: RequestHeader,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        meta: Option<RecordMeta>,
    },
    RequestChunk {
        b64: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: BTreeMap<String, String>,
    },
    RequestEnded,
    ResponseStarted {
        status: u16,
        headers: Vec<(String, String)>,
    },
    ResponseChunk {
        b64: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: BTreeMap<String, String>,
    },
    Ended {
        latency_ms: u64,
        meta: RecordMeta,
    },
}

pub(super) fn event_to_envelope(event: RecordingEvent) -> WireEnvelope {
    let RecordingEvent {
        id,
        ts_ms,
        parent,
        event,
    } = event;
    WireEnvelope {
        v: RECORDING_FORMAT_VERSION,
        id,
        ts_ms,
        parent,
        event: protocol_to_wire(event),
    }
}

pub(super) fn envelope_to_event(envelope: WireEnvelope) -> Result<RecordingEvent, ProximaError> {
    if envelope.v != RECORDING_FORMAT_VERSION {
        return Err(ProximaError::Record(format!(
            "unsupported recording version: expected {}, got {}",
            RECORDING_FORMAT_VERSION, envelope.v,
        )));
    }
    Ok(RecordingEvent {
        id: envelope.id,
        ts_ms: envelope.ts_ms,
        parent: envelope.parent,
        event: wire_to_protocol(envelope.event)?,
    })
}

fn protocol_to_wire(event: ProtocolEvent) -> WireProtocolEvent {
    match event {
        ProtocolEvent::Pipeline(inner) => WireProtocolEvent::Pipeline(pipeline_to_wire(inner)),
        ProtocolEvent::Process(inner) => WireProtocolEvent::Process(process_to_wire(inner)),
        ProtocolEvent::Http(inner) => WireProtocolEvent::Http(http_to_wire(inner)),
        ProtocolEvent::Custom { kind, payload } => WireProtocolEvent::Custom { kind, payload },
    }
}

fn wire_to_protocol(event: WireProtocolEvent) -> Result<ProtocolEvent, ProximaError> {
    Ok(match event {
        WireProtocolEvent::Pipeline(inner) => ProtocolEvent::Pipeline(wire_to_pipeline(inner)?),
        WireProtocolEvent::Process(inner) => ProtocolEvent::Process(wire_to_process(inner)?),
        WireProtocolEvent::Http(inner) => ProtocolEvent::Http(wire_to_http(inner)?),
        WireProtocolEvent::Custom { kind, payload } => ProtocolEvent::Custom { kind, payload },
    })
}

fn pipeline_to_wire(event: PipelineEvent) -> WirePipelineEvent {
    match event {
        PipelineEvent::Started {
            ts,
            spec_hash,
            name,
        } => WirePipelineEvent::Started {
            ts,
            spec_hash_hex: hex_encode(&spec_hash),
            name,
        },
        PipelineEvent::Ended { outcome } => WirePipelineEvent::Ended { outcome },
    }
}

fn wire_to_pipeline(event: WirePipelineEvent) -> Result<PipelineEvent, ProximaError> {
    Ok(match event {
        WirePipelineEvent::Started {
            ts,
            spec_hash_hex,
            name,
        } => {
            let spec_hash = hex_decode_32(&spec_hash_hex)?;
            PipelineEvent::Started {
                ts,
                spec_hash,
                name,
            }
        }
        WirePipelineEvent::Ended { outcome } => PipelineEvent::Ended { outcome },
    })
}

fn process_to_wire(event: ProcessEvent) -> WireProcessEvent {
    match event {
        ProcessEvent::Started {
            ts,
            command,
            args,
            env,
            cwd,
        } => WireProcessEvent::Started {
            ts,
            command,
            args,
            env,
            cwd,
        },
        ProcessEvent::Stdout(data) => WireProcessEvent::Stdout {
            b64: BASE64.encode(&data),
        },
        ProcessEvent::Stderr(data) => WireProcessEvent::Stderr {
            b64: BASE64.encode(&data),
        },
        ProcessEvent::Exited { exit_code } => WireProcessEvent::Exited { exit_code },
    }
}

fn wire_to_process(event: WireProcessEvent) -> Result<ProcessEvent, ProximaError> {
    Ok(match event {
        WireProcessEvent::Started {
            ts,
            command,
            args,
            env,
            cwd,
        } => ProcessEvent::Started {
            ts,
            command,
            args,
            env,
            cwd,
        },
        WireProcessEvent::Stdout { b64 } => ProcessEvent::Stdout(decode_base64(&b64)?),
        WireProcessEvent::Stderr { b64 } => ProcessEvent::Stderr(decode_base64(&b64)?),
        WireProcessEvent::Exited { exit_code } => ProcessEvent::Exited { exit_code },
    })
}

fn http_to_wire(event: HttpEvent) -> WireHttpEvent {
    match event {
        HttpEvent::Started {
            ts,
            pipe,
            request,
            meta,
        } => WireHttpEvent::Started {
            ts,
            pipe,
            request,
            meta,
        },
        HttpEvent::RequestChunk { data, metadata } => WireHttpEvent::RequestChunk {
            b64: BASE64.encode(&data),
            metadata: encode_metadata(&metadata),
        },
        HttpEvent::RequestEnded => WireHttpEvent::RequestEnded,
        HttpEvent::ResponseStarted { status, headers } => {
            WireHttpEvent::ResponseStarted { status, headers }
        }
        HttpEvent::ResponseChunk { data, metadata } => WireHttpEvent::ResponseChunk {
            b64: BASE64.encode(&data),
            metadata: encode_metadata(&metadata),
        },
        HttpEvent::Ended { latency_ms, meta } => WireHttpEvent::Ended { latency_ms, meta },
    }
}

fn wire_to_http(event: WireHttpEvent) -> Result<HttpEvent, ProximaError> {
    Ok(match event {
        WireHttpEvent::Started {
            ts,
            pipe,
            request,
            meta,
        } => HttpEvent::Started {
            ts,
            pipe,
            request,
            meta,
        },
        WireHttpEvent::RequestChunk { b64, metadata } => HttpEvent::RequestChunk {
            data: decode_base64(&b64)?,
            metadata: decode_metadata(metadata)?,
        },
        WireHttpEvent::RequestEnded => HttpEvent::RequestEnded,
        WireHttpEvent::ResponseStarted { status, headers } => {
            HttpEvent::ResponseStarted { status, headers }
        }
        WireHttpEvent::ResponseChunk { b64, metadata } => HttpEvent::ResponseChunk {
            data: decode_base64(&b64)?,
            metadata: decode_metadata(metadata)?,
        },
        WireHttpEvent::Ended { latency_ms, meta } => HttpEvent::Ended { latency_ms, meta },
    })
}

fn decode_base64(input: &str) -> Result<Bytes, ProximaError> {
    BASE64
        .decode(input)
        .map(Bytes::from)
        .map_err(|err| ProximaError::Record(format!("base64 decode: {err}")))
}

fn encode_metadata(metadata: &FrameMetadata) -> BTreeMap<String, String> {
    metadata
        .iter()
        .map(|(key, value)| (key.clone(), BASE64.encode(value)))
        .collect()
}

fn decode_metadata(wire: BTreeMap<String, String>) -> Result<FrameMetadata, ProximaError> {
    let mut decoded = FrameMetadata::new();
    for (key, encoded) in wire {
        let bytes = decode_base64(&encoded)?;
        decoded.insert(key, bytes);
    }
    Ok(decoded)
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn hex_decode_32(text: &str) -> Result<[u8; 32], ProximaError> {
    if text.len() != 64 {
        return Err(ProximaError::Record(format!(
            "spec_hash_hex must be 64 chars, got {}",
            text.len()
        )));
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in text.as_bytes().chunks_exact(2).enumerate() {
        let raw = std::str::from_utf8(chunk)
            .map_err(|err| ProximaError::Record(format!("spec_hash_hex non-utf8: {err}")))?;
        output[index] = u8::from_str_radix(raw, 16)
            .map_err(|err| ProximaError::Record(format!("spec_hash_hex parse: {err}")))?;
    }
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::event::HttpEvent;
    use rstest::rstest;

    fn make_id() -> InteractionId {
        InteractionId::from_bytes([7; 16])
    }

    fn round_trip(event: RecordingEvent) -> RecordingEvent {
        let envelope = event_to_envelope(event);
        let line = serde_json::to_string(&envelope).expect("serialize envelope");
        let parsed: WireEnvelope = serde_json::from_str(&line).expect("parse envelope");
        envelope_to_event(parsed).expect("convert envelope to event")
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
        request: RequestHeader {
            method: "POST".into(),
            path: "/v1/chat".into(),
            headers: [("accept".to_string(), "application/json".to_string())].into_iter().collect(),
            query: Default::default(),
        },
        meta: None,
    })))]
    #[case::http_req_chunk(envelope(make_id(), 5, ProtocolEvent::Http(HttpEvent::RequestChunk {
        data: Bytes::from_static(b"\x00\x01\x02chunk"),
        metadata: FrameMetadata::new(),
    })))]
    #[case::http_req_chunk_metadata(envelope(make_id(), 6, ProtocolEvent::Http(HttpEvent::RequestChunk {
        data: Bytes::from_static(b"payload"),
        metadata: {
            let mut metadata = FrameMetadata::new();
            metadata.insert("clock_at_call".into(), Bytes::from_static(&[0, 0, 0, 0, 0, 0, 0, 42]));
            metadata.insert("trace_id".into(), Bytes::from_static(b"abc"));
            metadata
        },
    })))]
    #[case::http_req_end(envelope(make_id(), 10, ProtocolEvent::Http(HttpEvent::RequestEnded)))]
    #[case::http_resp_start(envelope(make_id(), 12, ProtocolEvent::Http(HttpEvent::ResponseStarted {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
    })))]
    #[case::http_resp_chunk(envelope(make_id(), 15, ProtocolEvent::Http(HttpEvent::ResponseChunk {
        data: Bytes::from_static(b"hello world"),
        metadata: FrameMetadata::new(),
    })))]
    #[case::http_end(envelope(make_id(), 100, ProtocolEvent::Http(HttpEvent::Ended {
        latency_ms: 88,
        meta: RecordMeta::default(),
    })))]
    #[case::pipeline_started(envelope(InteractionId::from_bytes([9; 16]), 0, ProtocolEvent::Pipeline(PipelineEvent::Started {
        ts: OffsetDateTime::UNIX_EPOCH,
        spec_hash: {
            let mut bytes = [0_u8; 32];
            for (index, slot) in bytes.iter_mut().enumerate() {
                *slot = index as u8;
            }
            bytes
        },
        name: Some("bench-example-search".into()),
    })))]
    #[case::pipeline_ended_ok(envelope(InteractionId::from_bytes([9; 16]), 500, ProtocolEvent::Pipeline(PipelineEvent::Ended {
        outcome: PipelineOutcome::Completed,
    })))]
    #[case::pipeline_ended_failed(envelope(InteractionId::from_bytes([9; 16]), 500, ProtocolEvent::Pipeline(PipelineEvent::Ended {
        outcome: PipelineOutcome::Failed { reason: "exit code 1".into() },
    })))]
    #[case::process_started(child(
        InteractionId::from_bytes([5; 16]),
        InteractionId::from_bytes([9; 16]),
        10,
        ProtocolEvent::Process(ProcessEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            command: "cargo".into(),
            args: vec!["bench".into(), "--bench".into(), "example-search".into()],
            env: [("RUSTFLAGS".to_string(), "-C target-cpu=native".to_string())].into_iter().collect(),
            cwd: Some(PathBuf::from("/tmp/workspace")),
        }),
    ))]
    #[case::process_stdout(child(
        InteractionId::from_bytes([5; 16]),
        InteractionId::from_bytes([9; 16]),
        11,
        ProtocolEvent::Process(ProcessEvent::Stdout(Bytes::from_static(b"compiling example-search"))),
    ))]
    #[case::process_stderr(child(
        InteractionId::from_bytes([5; 16]),
        InteractionId::from_bytes([9; 16]),
        12,
        ProtocolEvent::Process(ProcessEvent::Stderr(Bytes::from_static(b"warning"))),
    ))]
    #[case::process_exited_zero(child(
        InteractionId::from_bytes([5; 16]),
        InteractionId::from_bytes([9; 16]),
        100,
        ProtocolEvent::Process(ProcessEvent::Exited { exit_code: Some(0) }),
    ))]
    #[case::process_exited_signaled(child(
        InteractionId::from_bytes([5; 16]),
        InteractionId::from_bytes([9; 16]),
        100,
        ProtocolEvent::Process(ProcessEvent::Exited { exit_code: None }),
    ))]
    #[case::custom_protocol(envelope(make_id(), 1, ProtocolEvent::Custom {
        kind: "redis".into(),
        payload: serde_json::json!({"cmd": "GET", "key": "x"}),
    }))]
    fn round_trips_through_jsonl_envelope(#[case] event: RecordingEvent) {
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn version_mismatch_returns_typed_error() {
        let envelope = WireEnvelope {
            v: 999,
            id: make_id(),
            ts_ms: 1,
            parent: None,
            event: WireProtocolEvent::Http(WireHttpEvent::RequestEnded),
        };
        let err = envelope_to_event(envelope).expect_err("must reject unsupported version");
        assert!(matches!(err, ProximaError::Record(_)));
    }

    #[test]
    fn empty_metadata_is_omitted_from_serialized_form() {
        let env = event_to_envelope(envelope(
            make_id(),
            1,
            ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(b"x"),
                metadata: FrameMetadata::new(),
            }),
        ));
        let json = serde_json::to_string(&env).expect("serialize");
        assert!(
            !json.contains("\"metadata\""),
            "empty metadata must skip serialization: {json}"
        );
    }

    #[test]
    fn malformed_base64_returns_typed_error() {
        let envelope = WireEnvelope {
            v: RECORDING_FORMAT_VERSION,
            id: make_id(),
            ts_ms: 1,
            parent: None,
            event: WireProtocolEvent::Http(WireHttpEvent::RequestChunk {
                b64: "!!!not base64!!!".into(),
                metadata: BTreeMap::new(),
            }),
        };
        let err = envelope_to_event(envelope).expect_err("must reject malformed base64");
        assert!(matches!(err, ProximaError::Record(_)));
    }

    #[test]
    fn parent_edge_round_trips() {
        let event = child(
            InteractionId::from_bytes([5; 16]),
            InteractionId::from_bytes([9; 16]),
            42,
            ProtocolEvent::Process(ProcessEvent::Stdout(Bytes::from_static(b"x"))),
        );
        assert_eq!(round_trip(event.clone()), event);
    }
}
