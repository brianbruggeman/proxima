//! Capture-context trait surface destined for `proxima-pipe`.
//!
//! Lifted from `recording/capture.rs` during Phase 2 of the decomposition
//! (see `docs/decomposition/discipline.md`). The `CaptureContext` trait
//! lives here so `RequestContext` in `proxima-pipe` can hold an
//! `Option<Arc<dyn CaptureContext>>` without depending on
//! `proxima-recording-core` (which depends on `proxima-pipe`).
//!
//! The concrete `LiveCaptureContext` implementation lives in
//! `proxima-recording`'s `pipe` module (folded from the former
//! `proxima-recording-pipe` in Phase 5).

use bytes::Bytes;

/// Per-call sidecar a Pipe writes to when it wants to round-trip
/// reproducibility-relevant bytes (clock reads, RNG seeds, request
/// hashes) alongside recorded frames. Proxima never reads or interprets
/// the attached values — it just round-trips them opaquely through
/// record→replay.
///
/// Access via `request.context.capture`; if `None`, no recording is
/// active and attachments are dropped.
pub trait CaptureContext: Send + Sync + 'static {
    /// Attach a key-value pair. Applies to the next frame emitted by
    /// the surrounding recording sink; subsequent attaches before that
    /// emission accumulate under their respective keys (last write
    /// wins per key in the concrete impl).
    fn attach(&self, key: &str, value: Bytes);
}
