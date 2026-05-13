pub mod json_format;
pub mod source;
mod wire;

pub use json_format::JsonFormat;
pub use source::JsonlSource;

use crate::event::RecordingEvent;
use proxima_core::ProximaError;

/// Encode a single `RecordingEvent` as one JSONL line (no trailing
/// newline). Reuses the same wire envelope as `JsonlSink` so encoded
/// events round-trip through `JsonlSource` and the universal-envelope
/// readers (e.g. the daemon's `tail` / `events` chunked streams).
pub fn encode_jsonl_line(event: RecordingEvent) -> Result<Vec<u8>, ProximaError> {
    let envelope = wire::event_to_envelope(event);
    serde_json::to_vec(&envelope)
        .map_err(|err| ProximaError::Encode(format!("encode jsonl line: {err}")))
}
