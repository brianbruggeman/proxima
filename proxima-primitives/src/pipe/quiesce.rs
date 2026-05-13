//! Graceful-shutdown response type shared by HTTP/1/2/3 listeners.
//! When a listener begins draining, it returns this shape to in-flight
//! requests so the dispatcher can emit a `503 Retry-After`-style
//! response instead of accepting new work.
//!
//! Lives in proxima-pipe (rather than a specific HTTP-version crate)
//! because every HTTP listener needs it and we don't want HTTP-version
//! crates depending on each other for this small value type.

#![cfg(feature = "alloc")]

use alloc::string::String;

#[derive(Debug)]
pub struct QuiesceResponse {
    pub status: u16,
    pub retry_after: String,
}
