//! `JsonFormat` — the jsonl recording codec: one `{envelope}\n` line per event
//! (the same wire envelope as the legacy `JsonlSink`). Pure codec, no file I/O.
//! Line-framed: a "block" encodes to N lines; decode reads ONE line (a
//! one-event unit), so the cursor still paginates to EOF. No index — the byte
//! offset of a line IS its address.

use std::io::BufRead;

use crate::event::RecordingEvent;
use crate::format::Format;
use crate::jsonl::wire::{WireEnvelope, envelope_to_event, event_to_envelope};
use proxima_core::ProximaError;

/// The jsonl line codec.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonFormat;

impl JsonFormat {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Format for JsonFormat {
    fn name(&self) -> &'static str {
        "json"
    }

    fn encode_block(&mut self, events: Vec<RecordingEvent>) -> Result<Vec<u8>, ProximaError> {
        let mut bytes = Vec::new();
        for event in events {
            let envelope = event_to_envelope(event);
            // `to_vec` needs only serde_json's `alloc` feature; `to_writer`
            // needs `std` (Vec's io::Write impl), which the no_std+alloc tier
            // does not carry.
            let encoded = serde_json::to_vec(&envelope)
                .map_err(|err| ProximaError::Record(format!("encode jsonl event: {err}")))?;
            bytes.extend_from_slice(&encoded);
            bytes.push(b'\n');
        }
        Ok(bytes)
    }

    fn decode_block(
        &self,
        reader: &mut dyn BufRead,
    ) -> Result<Option<(Vec<RecordingEvent>, u64)>, ProximaError> {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|err| ProximaError::Record(format!("read jsonl line: {err}")))?;
        if read == 0 {
            return Ok(None);
        }
        let consumed = read as u64;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(Some((Vec::new(), consumed)));
        }
        let envelope: WireEnvelope = serde_json::from_str(trimmed)
            .map_err(|err| ProximaError::Record(format!("parse jsonl line: {err}")))?;
        let event = envelope_to_event(envelope)?;
        Ok(Some((vec![event], consumed)))
    }
}
