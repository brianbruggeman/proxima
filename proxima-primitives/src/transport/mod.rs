//! Transport stream primitives: `Replay`, the `tap_complete` taps, and the
//! `GenericStream` seam.
//!
//! This module sits BELOW [`crate::pipe`] — it depends only on
//! `proxima-core` (for `ProximaError`), so `crate::pipe` can depend on it
//! without a cycle. The principle: anything in core must be generic;
//! `Request`/`Response`/`Bytes` is ONE degenerate case, never baked into a
//! core primitive.
//!
//! ## Tiers
//! - **std** (default): full surface — `Replay` (record-and-replay stream with
//!   fan-out sinks), the `tap_complete` / `tap_complete_with_size` taps, and
//!   the `GenericStream` boxed-stream seam. Runtime-agnostic; needs only std
//!   for `Mutex`/`Vec`.
//! - **alloc**: thin marker (only the type path; no alloc-only subset
//!   implemented yet — the stream types need std's Mutex/Vec freely).

#[cfg(feature = "std")]
pub mod replay;
#[cfg(feature = "std")]
pub mod stream;

#[cfg(feature = "std")]
pub use replay::{
    BytesReplay, DEFAULT_REPLAY_CAP_BYTES, DEFAULT_SINK_QUEUE, Replay, ReplayEvent, tap_complete,
    tap_complete_with_size,
};
#[cfg(feature = "std")]
pub use stream::GenericStream;
