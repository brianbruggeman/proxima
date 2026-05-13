//! Sans-IO block encoder. Serializes a batch of events into ONE inner buffer —
//! each event framed as `[u32 evt_len][postcard]` — which the IO sink then
//! compresses as a single block and writes with one syscall.
//!
//! Compression is aggregated at the **block** (per-append/per-interaction)
//! level, not per event: a streaming turn's 32 tiny chunks compress together
//! (small frames in isolation can't — zstd setup exceeds the payload) while
//! per-event boundaries are preserved inside the block, and zstd setup is paid
//! once per block instead of once per event. This obsoletes the old
//! coalesce-into-one-event hack (which compressed by destroying boundaries).
//!
//! `#![no_std]`-portable: `alloc`/`core`/`postcard` only. Compression is the
//! sink's job (std), so this stays sans-IO and drops into a `#![no_std]` gate
//! unchanged.

use alloc::format;
use alloc::vec::Vec;
use core::mem;

use crate::binary::wire::{BinEnvelope, bin_to_event, event_to_bin};
use crate::event::{InteractionId, RecordingEvent};
use proxima_core::ProximaError;

/// Width of each inner event's length prefix.
const EVENT_LEN_PREFIX: usize = 4;

/// Accumulates a batch of events into one contiguous inner buffer. Reset +
/// reuse across blocks to keep the buffer warm (no per-block reallocation).
pub struct FrameEncoder {
    inner: Vec<u8>,
    first: Option<(u64, InteractionId)>,
    count: usize,
}

impl Default for FrameEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            first: None,
            count: 0,
        }
    }

    /// Clear for a new block, keeping the buffer's capacity.
    pub fn reset(&mut self) {
        self.inner.clear();
        self.first = None;
        self.count = 0;
    }

    /// Append one event as `[u32 evt_len][postcard]`. No compression here — the
    /// sink compresses the whole inner buffer once.
    pub fn push(&mut self, event: RecordingEvent) -> Result<(), ProximaError> {
        let ts_ms = event.ts_ms();
        let id = event.id();
        if self.first.is_none() {
            self.first = Some((ts_ms, id));
        }
        let envelope = event_to_bin(event);

        let len_pos = self.inner.len();
        self.inner.extend_from_slice(&[0_u8; EVENT_LEN_PREFIX]);
        let payload_start = self.inner.len();
        let buffer = mem::take(&mut self.inner);
        self.inner = postcard::to_extend(&envelope, buffer)
            .map_err(|err| ProximaError::Record(format!("encode bin event: {err}")))?;
        let evt_len = u32::try_from(self.inner.len() - payload_start)
            .map_err(|_overflow| ProximaError::Record("bin event exceeds u32::MAX".into()))?;
        self.inner[len_pos..payload_start].copy_from_slice(&evt_len.to_le_bytes());
        self.count += 1;
        Ok(())
    }

    /// The concatenated event-frames — what the sink compresses into a block.
    #[must_use]
    pub fn inner(&self) -> &[u8] {
        &self.inner
    }

    /// `(ts_ms, id)` of the first event — what the block's index record carries.
    #[must_use]
    pub fn first(&self) -> Option<(u64, InteractionId)> {
        self.first
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Decode a (already-decompressed) inner block buffer back into its events.
/// Inverse of [`FrameEncoder::push`]'s framing.
pub fn decode_block(inner: &[u8]) -> Result<Vec<RecordingEvent>, ProximaError> {
    let mut events = Vec::new();
    let mut pos = 0_usize;
    while pos + EVENT_LEN_PREFIX <= inner.len() {
        let mut len_bytes = [0_u8; EVENT_LEN_PREFIX];
        len_bytes.copy_from_slice(&inner[pos..pos + EVENT_LEN_PREFIX]);
        let evt_len = u32::from_le_bytes(len_bytes) as usize;
        pos += EVENT_LEN_PREFIX;
        if pos + evt_len > inner.len() {
            return Err(ProximaError::Record("bin block event truncated".into()));
        }
        let envelope: BinEnvelope = postcard::from_bytes(&inner[pos..pos + evt_len])
            .map_err(|err| ProximaError::Record(format!("decode bin event: {err}")))?;
        events.push(bin_to_event(envelope)?);
        pos += evt_len;
    }
    Ok(events)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent};
    use bytes::Bytes;

    fn chunk(id: InteractionId, data: &'static [u8]) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms: 7,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(data),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // a block round-trips: encode N events into the inner buffer, decode back.
    #[test]
    fn block_round_trips_events_in_order() {
        let id = InteractionId::new();
        let mut encoder = FrameEncoder::new();
        encoder.push(chunk(id, b"aa")).unwrap();
        encoder.push(chunk(id, b"bbbb")).unwrap();

        assert_eq!(encoder.count(), 2);
        assert_eq!(encoder.first().map(|(ts, _)| ts), Some(7));

        let decoded = decode_block(encoder.inner()).unwrap();
        assert_eq!(decoded.len(), 2);
        let datas: Vec<Bytes> = decoded
            .into_iter()
            .filter_map(|event| match event.event {
                ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => Some(data),
                _ => None,
            })
            .collect();
        assert_eq!(
            datas,
            vec![Bytes::from_static(b"aa"), Bytes::from_static(b"bbbb")]
        );
    }

    #[test]
    fn reset_clears_block_state() {
        let id = InteractionId::new();
        let mut encoder = FrameEncoder::new();
        encoder.push(chunk(id, b"x")).unwrap();
        encoder.reset();
        assert!(encoder.is_empty());
        assert_eq!(encoder.first(), None);
        assert!(decode_block(encoder.inner()).unwrap().is_empty());
    }
}
