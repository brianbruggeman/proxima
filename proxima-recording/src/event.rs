use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use ulid::Ulid;

pub const RECORDING_FORMAT_VERSION: u32 = 3;

pub type FrameMetadata = BTreeMap<String, Bytes>;

/// Working directory of a recorded process. `PathBuf` under `std`; a UTF-8
/// `String` under the alloc-only no_std tier — there is no OS path type
/// without std, and a captured cwd is always UTF-8 in practice. Transparent
/// alias: the `std` tier (the crate default) sees exactly `PathBuf`, so this
/// changes nothing for existing callers.
#[cfg(feature = "std")]
pub type ProcessCwd = std::path::PathBuf;
#[cfg(not(feature = "std"))]
pub type ProcessCwd = alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InteractionId(Ulid);

impl InteractionId {
    // Ulid::new() reads the current time (std::time::SystemTime), so ulid
    // only compiles it in behind its own `std` feature — no OS clock exists
    // on the alloc-only no_std tier. Minting fresh IDs is a std-tier
    // capability; the alloc tier only ever moves IDs that already exist.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn new() -> Self {
        Self(Ulid::new())
    }

    #[must_use]
    pub fn from_ulid(ulid: Ulid) -> Self {
        Self(ulid)
    }

    #[must_use]
    pub fn as_ulid(&self) -> Ulid {
        self.0
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; 16] {
        self.0.to_bytes()
    }

    #[must_use]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Ulid::from_bytes(bytes))
    }
}

#[cfg(feature = "std")]
impl Default for InteractionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InteractionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl Serialize for InteractionId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for InteractionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let parsed: Ulid = raw.parse().map_err(serde::de::Error::custom)?;
        Ok(Self(parsed))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheOutcome {
    Hit,
    Miss,
    Bypass,
    Revalidated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RequestHeader {
    pub method: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub query: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RecordMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheOutcome>,
    #[serde(default)]
    pub retries: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    /// Provenance of this event: came from a real recorded
    /// interaction, or synthesized by an inference path during
    /// derivation. Optional for backward compatibility — events
    /// emitted before the discriminator existed deserialize with
    /// `source = None`, which the verify replay walker treats as
    /// `Recorded` unless explicitly flagged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<EventSource>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Provenance discriminator for events. `Recorded` events come from
/// a real interaction captured at the source; `Inferred` events are
/// synthesized by a derivation path (pipeline status inference, etc.)
/// when the source did not produce a direct event. The
/// `inferred_not_recorded` verify-policy rule asserts that pipes
/// flagged `must_derive_from_record` only produce `Recorded` events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventSource {
    Recorded,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineOutcome {
    Completed,
    Failed { reason: String },
    Cancelled,
}

/// Universal recording envelope. Every recorded event carries this shell,
/// regardless of what protocol it represents. The protocol-specific shape
/// lives in `event`. `parent` is the only first-class relationship — it
/// expresses "this interaction is a child of that one" (e.g., pipeline →
/// stage), which is universal across protocols. Other relationships
/// (causality, depends-on) stay in protocol-specific payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordingEvent {
    pub id: InteractionId,
    pub ts_ms: u64,
    pub parent: Option<InteractionId>,
    pub event: ProtocolEvent,
}

/// What protocol this interaction speaks. Each variant owns its full
/// lifecycle (Started → middle events → Ended). New built-in protocols
/// add a variant here. Plugin protocols ride `Custom` and register a
/// `ProtocolRenderer` for display/parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolEvent {
    Pipeline(PipelineEvent),
    Process(ProcessEvent),
    Http(HttpEvent),
    Custom {
        kind: String,
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    Started {
        ts: OffsetDateTime,
        spec_hash: [u8; 32],
        name: Option<String>,
    },
    Ended {
        outcome: PipelineOutcome,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessEvent {
    Started {
        ts: OffsetDateTime,
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<ProcessCwd>,
    },
    Stdout(Bytes),
    Stderr(Bytes),
    Exited {
        exit_code: Option<i32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum HttpEvent {
    Started {
        ts: OffsetDateTime,
        pipe: String,
        request: RequestHeader,
        meta: Option<RecordMeta>,
    },
    RequestChunk {
        data: Bytes,
        metadata: FrameMetadata,
    },
    RequestEnded,
    ResponseStarted {
        status: u16,
        headers: Vec<(String, String)>,
    },
    ResponseChunk {
        data: Bytes,
        metadata: FrameMetadata,
    },
    Ended {
        latency_ms: u64,
        meta: RecordMeta,
    },
}

impl RecordingEvent {
    #[must_use]
    pub fn id(&self) -> InteractionId {
        self.id
    }

    #[must_use]
    pub fn ts_ms(&self) -> u64 {
        self.ts_ms
    }

    #[must_use]
    pub fn parent(&self) -> Option<InteractionId> {
        self.parent
    }

    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        match &self.event {
            ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. })
            | ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. })
            | ProtocolEvent::Process(ProcessEvent::Stdout(data))
            | ProtocolEvent::Process(ProcessEvent::Stderr(data)) => data.len(),
            _ => 0,
        }
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            &self.event,
            ProtocolEvent::Http(HttpEvent::Ended { .. })
                | ProtocolEvent::Pipeline(PipelineEvent::Ended { .. })
                | ProtocolEvent::Process(ProcessEvent::Exited { .. })
        )
    }

    #[must_use]
    pub fn kind(&self) -> &str {
        match &self.event {
            ProtocolEvent::Pipeline(_) => "pipeline",
            ProtocolEvent::Process(_) => "process",
            ProtocolEvent::Http(_) => "http",
            ProtocolEvent::Custom { kind, .. } => kind,
        }
    }
}

/// Pluggable renderer for `ProtocolEvent::Custom` payloads (and any
/// built-in protocol whose display can be customized). Registered in a
/// `ProtocolRendererRegistry` parallel to the format-registry pattern.
/// Built-in renderers ship for Pipeline / Process / Http; plugins
/// register their own for the kinds they own.
pub trait ProtocolRenderer: Send + Sync {
    fn kind(&self) -> &str;
    fn summary(&self, event: &RecordingEvent) -> String;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn make_id() -> InteractionId {
        InteractionId::from_bytes([42; 16])
    }

    #[test]
    fn interaction_id_round_trips_through_bytes() {
        let original = make_id();
        let restored = InteractionId::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }

    #[test]
    fn interaction_id_serializes_as_string() {
        let id = make_id();
        let json = serde_json::to_string(&id).expect("serialize id");
        assert!(json.starts_with('"'), "ulid serializes as string: {json}");
        let back: InteractionId = serde_json::from_str(&json).expect("parse id");
        assert_eq!(back, id);
    }

    fn http_started() -> RecordingEvent {
        RecordingEvent {
            id: make_id(),
            ts_ms: 0,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::Started {
                ts: OffsetDateTime::UNIX_EPOCH,
                pipe: "echo".into(),
                request: RequestHeader::default(),
                meta: None,
            }),
        }
    }

    fn http_response_chunk(bytes: &'static [u8]) -> RecordingEvent {
        RecordingEvent {
            id: make_id(),
            ts_ms: 5,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(bytes),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    fn http_ended() -> RecordingEvent {
        RecordingEvent {
            id: make_id(),
            ts_ms: 100,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::Ended {
                latency_ms: 95,
                meta: RecordMeta::default(),
            }),
        }
    }

    #[rstest]
    #[case::http_start(http_started())]
    #[case::http_resp_chunk(http_response_chunk(b"hello"))]
    #[case::http_end(http_ended())]
    fn id_accessor_returns_envelope_id(#[case] event: RecordingEvent) {
        assert_eq!(event.id(), make_id());
    }

    #[test]
    fn pipeline_event_has_no_payload_bytes() {
        let event = RecordingEvent {
            id: make_id(),
            ts_ms: 1,
            parent: None,
            event: ProtocolEvent::Pipeline(PipelineEvent::Started {
                ts: OffsetDateTime::UNIX_EPOCH,
                spec_hash: [0; 32],
                name: Some("bench".into()),
            }),
        };
        assert_eq!(event.payload_bytes(), 0);
        assert!(!event.is_terminal());
        assert_eq!(event.kind(), "pipeline");
    }

    #[test]
    fn http_response_chunk_counts_payload_bytes() {
        let event = http_response_chunk(b"hello");
        assert_eq!(event.payload_bytes(), 5);
        assert!(!event.is_terminal());
        assert_eq!(event.kind(), "http");
    }

    #[test]
    fn process_stdout_counts_payload_bytes_and_carries_process_kind() {
        let event = RecordingEvent {
            id: make_id(),
            ts_ms: 1,
            parent: Some(InteractionId::from_bytes([7; 16])),
            event: ProtocolEvent::Process(ProcessEvent::Stdout(Bytes::from_static(b"line"))),
        };
        assert_eq!(event.payload_bytes(), 4);
        assert_eq!(event.kind(), "process");
        assert_eq!(event.parent(), Some(InteractionId::from_bytes([7; 16])));
    }

    #[test]
    fn pipeline_ended_and_process_exited_and_http_ended_are_terminal() {
        let pipeline_end = RecordingEvent {
            id: make_id(),
            ts_ms: 100,
            parent: None,
            event: ProtocolEvent::Pipeline(PipelineEvent::Ended {
                outcome: PipelineOutcome::Completed,
            }),
        };
        let process_exit = RecordingEvent {
            id: make_id(),
            ts_ms: 50,
            parent: Some(InteractionId::from_bytes([7; 16])),
            event: ProtocolEvent::Process(ProcessEvent::Exited { exit_code: Some(0) }),
        };
        assert!(pipeline_end.is_terminal());
        assert!(process_exit.is_terminal());
        assert!(http_ended().is_terminal());
    }

    #[test]
    fn custom_kind_passes_through() {
        let event = RecordingEvent {
            id: make_id(),
            ts_ms: 1,
            parent: None,
            event: ProtocolEvent::Custom {
                kind: "redis".into(),
                payload: serde_json::json!({"cmd": "GET", "key": "x"}),
            },
        };
        assert_eq!(event.kind(), "redis");
        assert_eq!(event.payload_bytes(), 0);
        assert!(!event.is_terminal());
    }
}
