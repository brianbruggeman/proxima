//! C1 of the proxima-notify initiative: tier-3 sans-IO data types + postcard
//! codec for alert events and guidance request/response.
//!
//! # Tier
//!
//! - tier-1 (default, `--features std,alloc`): full surface with optional
//!   `as_json_shape()` helper for parity tests.
//! - tier-3 (`--no-default-features`): `#![no_std]`, no per-call allocation.
//!   `heapless::String<N>` / `heapless::FnvIndexMap<K, V, N>` /
//!   `heapless::Vec<u8, N>` with const-generic caps from
//!   `proxima-notify-proto.toml` per principle 12.
//!
//! Wire format is postcard (sans-IO binary) — the proxima native codec, also
//! used for binary recording (`proxima-recording/src/binary/wire.rs`).
//! Postcard composes with `heapless` containers without ever touching the
//! global allocator.
//!
//! # Composed primitives
//!
//! - `proxima_core::markers::*` — every public type explicitly impls the
//!   marker subset it qualifies for. AlertEvent and the Guidance types are
//!   `IsPure + Deterministic + Reproducible + all Without*` (pure data).
//! - `heapless` (no_alloc-capable bounded containers).
//! - `postcard` (sans-IO binary codec).
//! - `ulid` (`Ulid` newtype — 128-bit lexicographic IDs).
//!
//! # Why this crate exists vs. extending an existing primitive (principle 1)
//!
//! `ProtocolEvent::Custom { kind: String, payload: serde_json::Value }`
//! (in `proxima-recording/src/event.rs`) already exists for ad-hoc
//! event payloads. But it requires `alloc` (`String` + `serde_json::Value`)
//! and supports arbitrary nesting — incompatible with the tier-3 invariant
//! and the const-generic-caps discipline. proxima-notify-proto provides a
//! typed, bounded shape for the alert + guidance domain that compiles at
//! tier-3 AND round-trips through postcard.

/// Build-time per-crate sizing constants emitted by build.rs from
/// `proxima-notify.toml` per principle 12. Every const-generic cap below
/// traces back here.
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_notify_sized.rs"));
}

use heapless::index_map::FnvIndexMap;
use heapless::{String as HeaplessString, Vec as HeaplessVec};
use proxima_core::markers::{
    AllocFree, Deterministic, IsPure, NoStd, Reproducible, WithoutFilesystem, WithoutNetwork,
    WithoutRandom, WithoutSpawn, WithoutTime,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

/// Globally-unique identifier for an [`AlertEvent`]. Crockford-base32 ULID
/// — lexicographically sortable, embeds a millisecond-precision timestamp
/// plus 80 bits of entropy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[must_use]
pub struct AlertId(pub Ulid);

impl AlertId {
    /// Build from raw bytes (16 bytes, big-endian per ULID spec).
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Ulid::from_bytes(bytes))
    }

    /// Extract the underlying 128-bit value.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.to_bytes()
    }
}

impl NoStd for AlertId {}
impl AllocFree for AlertId {}
impl IsPure for AlertId {}
impl WithoutFilesystem for AlertId {}
impl WithoutNetwork for AlertId {}
impl WithoutSpawn for AlertId {}
impl WithoutTime for AlertId {}
impl WithoutRandom for AlertId {}
impl Deterministic for AlertId {}
impl Reproducible for AlertId {}

/// Globally-unique identifier for a [`GuidanceQuestion`] + matching
/// [`GuidanceAnswer`]. Crockford-base32 ULID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[must_use]
pub struct GuidanceRequestId(pub Ulid);

impl GuidanceRequestId {
    /// Build from raw bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Ulid::from_bytes(bytes))
    }

    /// Extract the underlying 128-bit value.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.to_bytes()
    }
}

impl NoStd for GuidanceRequestId {}
impl AllocFree for GuidanceRequestId {}
impl IsPure for GuidanceRequestId {}
impl WithoutFilesystem for GuidanceRequestId {}
impl WithoutNetwork for GuidanceRequestId {}
impl WithoutSpawn for GuidanceRequestId {}
impl WithoutTime for GuidanceRequestId {}
impl WithoutRandom for GuidanceRequestId {}
impl Deterministic for GuidanceRequestId {}
impl Reproducible for GuidanceRequestId {}

/// Identifier for a logical agent in the hierarchical-guidance tree. Used
/// by [`GuidanceQuestion::agent_id`] and [`GuidanceQuestion::parent_id`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[must_use]
pub struct AgentId(pub Ulid);

impl AgentId {
    /// Build from raw bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Ulid::from_bytes(bytes))
    }

    /// Extract the underlying 128-bit value.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.to_bytes()
    }
}

impl NoStd for AgentId {}
impl AllocFree for AgentId {}
impl IsPure for AgentId {}
impl WithoutFilesystem for AgentId {}
impl WithoutNetwork for AgentId {}
impl WithoutSpawn for AgentId {}
impl WithoutTime for AgentId {}
impl WithoutRandom for AgentId {}
impl Deterministic for AgentId {}
impl Reproducible for AgentId {}

/// Alert severity levels. Numeric values match `proxima-telemetry::Level`
/// for consistency across the workspace:
///
/// - `Trace = 1`
/// - `Debug = 5`
/// - `Info  = 9`
/// - `Warn  = 13`
/// - `Error = 17`
/// - `Fatal = 21`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Severity {
    /// Diagnostic, hot-path-frequency (least severe).
    Trace = 1,
    /// Developer narrative — state transitions, decision points.
    Debug = 5,
    /// Major workflow transitions (business-meaningful, sparse).
    Info = 9,
    /// Degraded but self-healing; succeeded but deserves later attention.
    Warn = 13,
    /// Contract violations or user-visible failures.
    Error = 17,
    /// Unrecoverable; immediate intervention required.
    Fatal = 21,
}

impl Severity {
    /// Numeric severity value, suitable for logging or comparison.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Lowercase name, matching the JSON shape in
    /// `docs/proxima-notify/ALERT_EVENT_SCHEMA.md`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Fatal => "fatal",
        }
    }
}

impl NoStd for Severity {}
impl AllocFree for Severity {}
impl IsPure for Severity {}
impl WithoutFilesystem for Severity {}
impl WithoutNetwork for Severity {}
impl WithoutSpawn for Severity {}
impl WithoutTime for Severity {}
impl WithoutRandom for Severity {}
impl Deterministic for Severity {}
impl Reproducible for Severity {}

/// A bounded label key (UTF-8, ≤ [`sized::ALERT_LABEL_KEY_MAX`] bytes).
pub type LabelKey = HeaplessString<{ sized::ALERT_LABEL_KEY_MAX }>;
/// A bounded label value (UTF-8, ≤ [`sized::ALERT_LABEL_VAL_MAX`] bytes).
pub type LabelValue = HeaplessString<{ sized::ALERT_LABEL_VAL_MAX }>;
/// A bounded label map (≤ [`sized::ALERT_LABELS_MAX`] entries; cap is a power of 2 per heapless requirement).
pub type LabelMap = FnvIndexMap<LabelKey, LabelValue, { sized::ALERT_LABELS_MAX }>;
/// A bounded kind string (e.g. "heartbeat", "threshold_breach").
pub type KindString = HeaplessString<{ sized::ALERT_KIND_MAX }>;
/// A bounded opaque payload (postcard-encoded domain bytes).
pub type Payload = HeaplessVec<u8, { sized::ALERT_PAYLOAD_MAX }>;

/// One-way alert event. Pure data: no clock read, no filesystem, no network.
/// Carrying `fired_at_micros` as a caller-provided `u64` preserves
/// `WithoutTime` — the producer reads the clock and pours the value in;
/// the proto layer never touches it.
///
/// # Example
///
/// See `examples/alert_walkthrough.rs` for a hand-built `AlertEvent` whose
/// postcard encoding is asserted bit-exact against a fixture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Globally-unique event id (ULID).
    pub id: AlertId,
    /// Severity level (Trace .. Fatal).
    pub severity: Severity,
    /// Semantic alert kind, e.g. "heartbeat", "threshold_breach".
    pub kind: KindString,
    /// Routing label map (e.g. `{"host": "node-1", "service": "api"}`).
    pub labels: LabelMap,
    /// Opaque payload bytes — typically postcard-encoded domain data.
    pub payload: Payload,
    /// Caller-provided microseconds since UNIX epoch (preserves `WithoutTime`).
    pub fired_at_micros: u64,
}

impl NoStd for AlertEvent {}
impl AllocFree for AlertEvent {}
impl IsPure for AlertEvent {}
impl WithoutFilesystem for AlertEvent {}
impl WithoutNetwork for AlertEvent {}
impl WithoutSpawn for AlertEvent {}
impl WithoutTime for AlertEvent {}
impl WithoutRandom for AlertEvent {}
impl Deterministic for AlertEvent {}
impl Reproducible for AlertEvent {}

/// Bounded question text.
pub type QuestionString = HeaplessString<{ sized::GUIDANCE_QUESTION_MAX }>;
/// Bounded answer text.
pub type AnswerString = HeaplessString<{ sized::GUIDANCE_ANSWER_MAX }>;
/// Bounded context bytes (opaque, postcard-encoded by caller).
pub type ContextBytes = HeaplessVec<u8, { sized::GUIDANCE_CONTEXT_MAX }>;
/// Bounded responder identifier (e.g. "stdin", "telegram:chat_id=12345").
pub type ResponderString = HeaplessString<{ sized::GUIDANCE_RESPONDER_MAX }>;

/// Request from an agent to its orchestrator for direction. Pure data.
/// `parent_id` names the orchestrator-agent if guidance is being
/// hierarchically routed; `None` means the request goes to the configured
/// root sink (a human via stdin / Telegram / etc.).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidanceQuestion {
    /// Globally-unique request id (ULID). Used by [`GuidanceAnswer::request_id`]
    /// to correlate replies.
    pub id: GuidanceRequestId,
    /// The asking agent's id.
    pub agent_id: AgentId,
    /// Optional orchestrator-agent id (hierarchical guidance). `None` →
    /// root sink (human).
    pub parent_id: Option<AgentId>,
    /// The question text.
    pub question: QuestionString,
    /// Opaque context bytes (postcard-encoded by caller).
    pub context: ContextBytes,
    /// Caller-provided microseconds since UNIX epoch.
    pub asked_at_micros: u64,
    /// Caller-provided timeout (microseconds).
    pub timeout_micros: u64,
}

impl NoStd for GuidanceQuestion {}
impl AllocFree for GuidanceQuestion {}
impl IsPure for GuidanceQuestion {}
impl WithoutFilesystem for GuidanceQuestion {}
impl WithoutNetwork for GuidanceQuestion {}
impl WithoutSpawn for GuidanceQuestion {}
impl WithoutTime for GuidanceQuestion {}
impl WithoutRandom for GuidanceQuestion {}
impl Deterministic for GuidanceQuestion {}
impl Reproducible for GuidanceQuestion {}

/// Answer from an orchestrator (human or parent agent) to a
/// [`GuidanceQuestion`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidanceAnswer {
    /// Matches [`GuidanceQuestion::id`].
    pub request_id: GuidanceRequestId,
    /// The answer text.
    pub content: AnswerString,
    /// Identifies who answered (e.g. "stdin", "telegram:chat_id=12345").
    pub responder: ResponderString,
    /// Caller-provided microseconds since UNIX epoch when the answer was
    /// received by the proto layer.
    pub responded_at_micros: u64,
}

impl NoStd for GuidanceAnswer {}
impl AllocFree for GuidanceAnswer {}
impl IsPure for GuidanceAnswer {}
impl WithoutFilesystem for GuidanceAnswer {}
impl WithoutNetwork for GuidanceAnswer {}
impl WithoutSpawn for GuidanceAnswer {}
impl WithoutTime for GuidanceAnswer {}
impl WithoutRandom for GuidanceAnswer {}
impl Deterministic for GuidanceAnswer {}
impl Reproducible for GuidanceAnswer {}

/// Errors from postcard encode / decode.
#[derive(Debug)]
#[non_exhaustive]
pub enum CodecError {
    /// postcard encode failure (typically: output buffer too small).
    Encode(postcard::Error),
    /// postcard decode failure (corrupted bytes, version mismatch, etc.).
    Decode(postcard::Error),
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Encode(err) => write!(formatter, "postcard encode: {err}"),
            Self::Decode(err) => write!(formatter, "postcard decode: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}

/// Encode an [`AlertEvent`] into `out`. Returns the number of bytes written
/// or [`CodecError::Encode`] if the buffer is too small.
///
/// Hot-path zero-alloc: postcard streams directly into the caller's buffer.
pub fn encode_alert(event: &AlertEvent, out: &mut [u8]) -> Result<usize, CodecError> {
    let slice = postcard::to_slice(event, out).map_err(CodecError::Encode)?;
    Ok(slice.len())
}

/// Decode an [`AlertEvent`] from `bytes`.
///
/// Hot-path zero-alloc: postcard reads bytes in place; heapless containers
/// inside the returned struct copy data from the input slice but do not
/// touch the global allocator.
pub fn decode_alert(bytes: &[u8]) -> Result<AlertEvent, CodecError> {
    postcard::from_bytes(bytes).map_err(CodecError::Decode)
}

/// Encode a [`GuidanceQuestion`] into `out`.
pub fn encode_guidance_question(
    question: &GuidanceQuestion,
    out: &mut [u8],
) -> Result<usize, CodecError> {
    let slice = postcard::to_slice(question, out).map_err(CodecError::Encode)?;
    Ok(slice.len())
}

/// Decode a [`GuidanceQuestion`] from `bytes`.
pub fn decode_guidance_question(bytes: &[u8]) -> Result<GuidanceQuestion, CodecError> {
    postcard::from_bytes(bytes).map_err(CodecError::Decode)
}

/// Encode a [`GuidanceAnswer`] into `out`.
pub fn encode_guidance_answer(
    answer: &GuidanceAnswer,
    out: &mut [u8],
) -> Result<usize, CodecError> {
    let slice = postcard::to_slice(answer, out).map_err(CodecError::Encode)?;
    Ok(slice.len())
}

/// Decode a [`GuidanceAnswer`] from `bytes`.
pub fn decode_guidance_answer(bytes: &[u8]) -> Result<GuidanceAnswer, CodecError> {
    postcard::from_bytes(bytes).map_err(CodecError::Decode)
}

/// Conversion to the documented JSON shape (per
/// `docs/proxima-notify/ALERT_EVENT_SCHEMA.md`). Used by C1's parity test.
///
/// Requires `alloc` (via `serde_json`). NOT available in tier-3 builds.
#[cfg(feature = "json-shape")]
pub mod json_shape {
    use super::{AlertEvent, GuidanceAnswer, GuidanceQuestion};
    use serde_json::{Value, json};

    /// Render an [`AlertEvent`] into the documented JSON shape. Labels are
    /// sorted by key for deterministic output (matches the schema).
    #[must_use]
    pub fn alert_event_to_json(event: &AlertEvent) -> Value {
        let mut labels_sorted: alloc::vec::Vec<(&str, &str)> = event
            .labels
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        labels_sorted.sort_by(|left, right| left.0.cmp(right.0));
        let labels_obj: serde_json::Map<String, Value> = labels_sorted
            .into_iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect();
        json!({
            "id": event.id.0.to_string(),
            "severity": event.severity.as_str(),
            "kind": event.kind.as_str(),
            "labels": labels_obj,
            "payload_bytes_base64": base64_url_safe_no_pad(event.payload.as_slice()),
            "fired_at_micros": event.fired_at_micros,
        })
    }

    /// Render a [`GuidanceQuestion`] into a JSON shape (informal — only
    /// used for diagnostic display, not a stable schema).
    #[must_use]
    pub fn guidance_question_to_json(question: &GuidanceQuestion) -> Value {
        json!({
            "id": question.id.0.to_string(),
            "agent_id": question.agent_id.0.to_string(),
            "parent_id": question.parent_id.map(|id| id.0.to_string()),
            "question": question.question.as_str(),
            "asked_at_micros": question.asked_at_micros,
            "timeout_micros": question.timeout_micros,
        })
    }

    /// Render a [`GuidanceAnswer`] into a JSON shape.
    #[must_use]
    pub fn guidance_answer_to_json(answer: &GuidanceAnswer) -> Value {
        json!({
            "request_id": answer.request_id.0.to_string(),
            "content": answer.content.as_str(),
            "responder": answer.responder.as_str(),
            "responded_at_micros": answer.responded_at_micros,
        })
    }

    /// URL-safe base64 with no padding — RFC 4648 §5 alphabet, used by the
    /// `payload_bytes_base64` field in the documented schema. Hand-rolled
    /// to avoid pulling in a separate base64 crate.
    fn base64_url_safe_no_pad(bytes: &[u8]) -> alloc::string::String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = alloc::string::String::with_capacity((bytes.len() + 2) / 3 * 4);
        for chunk in bytes.chunks(3) {
            let buf = match chunk.len() {
                3 => [chunk[0], chunk[1], chunk[2]],
                2 => [chunk[0], chunk[1], 0],
                1 => [chunk[0], 0, 0],
                _ => unreachable!(),
            };
            let combined = (u32::from(buf[0]) << 16) | (u32::from(buf[1]) << 8) | u32::from(buf[2]);
            let chars = [
                ALPHABET[((combined >> 18) & 0x3F) as usize],
                ALPHABET[((combined >> 12) & 0x3F) as usize],
                ALPHABET[((combined >> 6) & 0x3F) as usize],
                ALPHABET[(combined & 0x3F) as usize],
            ];
            let take = chunk.len() + 1;
            for &symbol in chars.iter().take(take) {
                out.push(symbol as char);
            }
        }
        out
    }
}
