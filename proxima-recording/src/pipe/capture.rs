use bytes::Bytes;
use crossbeam_queue::SegQueue;

use crate::event::FrameMetadata;
use proxima_primitives::pipe::capture_surface::CaptureContext;

/// Concrete `CaptureContext` impl. Lock-free `SegQueue` of
/// `(key, value)` pairs, drained to a `FrameMetadata` on emit.
/// Multiple writers across cores are safe; reads happen at
/// frame-emission boundaries where the queue is drained in
/// insertion order. Last-write-wins per key is preserved by the
/// final drain folding pairs into the map.
///
/// Landed in `proxima-recording`'s `pipe` module in Phase 5 (folded from
/// the former `proxima-recording-pipe` crate). The `CaptureContext` trait
/// it implements lives in `proxima-pipe::capture_surface`.
pub struct LiveCaptureContext {
    pending: SegQueue<(String, Bytes)>,
}

impl LiveCaptureContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: SegQueue::new(),
        }
    }

    /// Drain the pending pairs into a fresh `FrameMetadata`.
    /// RecordUpstream's ChunkRecorder calls this when emitting a
    /// frame so the metadata travels with that frame and the queue is
    /// empty for the next one.
    #[must_use]
    pub fn drain(&self) -> FrameMetadata {
        let mut metadata = FrameMetadata::new();
        while let Some((key, value)) = self.pending.pop() {
            metadata.insert(key, value);
        }
        metadata
    }
}

impl CaptureContext for LiveCaptureContext {
    fn attach(&self, key: &str, value: Bytes) {
        self.pending.push((key.to_string(), value));
    }
}

impl Default for LiveCaptureContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LiveCaptureContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveCaptureContext")
            .field("pending_size", &self.pending.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn attach_then_drain_returns_pairs() {
        let capture = LiveCaptureContext::new();
        capture.attach("clock_at_call", Bytes::from_static(&[0, 0, 0, 42]));
        capture.attach("trace", Bytes::from_static(b"abc"));
        let drained = capture.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained["clock_at_call"], Bytes::from_static(&[0, 0, 0, 42]));
        assert_eq!(drained["trace"], Bytes::from_static(b"abc"));
    }

    #[test]
    fn drain_resets_pending() {
        let capture = LiveCaptureContext::new();
        capture.attach("key", Bytes::from_static(b"value"));
        let first = capture.drain();
        assert_eq!(first.len(), 1);
        let second = capture.drain();
        assert!(second.is_empty(), "second drain must be empty: {second:?}");
    }

    #[test]
    fn last_write_wins_per_key() {
        let capture = LiveCaptureContext::new();
        capture.attach("clock", Bytes::from_static(&[1]));
        capture.attach("clock", Bytes::from_static(&[2]));
        let drained = capture.drain();
        assert_eq!(drained["clock"], Bytes::from_static(&[2]));
    }
}
