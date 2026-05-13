//! Sans-IO HTTP/2 codec — frame parser + per-stream state machine +
//! per-connection state. Extracted from `proxima-h2` for the no_std +
//! alloc cliff.
//!
//! Layers:
//!
//! - [`frame`]: wire framing — 9-byte header + per-type payloads.
//!   Parse / encode. State-machine-free; pure bytes.
//! - [`stream`]: per-stream state machine + per-stream flow-control
//!   windows (RFC 7540 §5).
//! - [`stream_table`]: per-connection stream registry + ID accounting.
//! - [`connection`]: connection lifecycle (preface, SETTINGS exchange,
//!   GOAWAY, drive loop) as a sans-IO state machine.
//!
//! Header compression lives in [`crate::hpack`]; the codec consumes
//! it directly.
//!
//! # Tier
//!
//! Compiles under `#![no_std]` with `alloc`. `--no-default-features
//! --features http2_codec-no-alloc` builds tier-3 (`core::*` only) —
//! exposes just [`stream`], the per-stream RFC 7540 §5 state machine +
//! flow-control windows (already heap-free). [`frame`] (owned `Bytes`
//! payload views, `SmallVec` heap fallback), [`stream_table`]
//! (`BTreeMap` registry), and [`connection`] (event queue, buffers,
//! HPACK dynamic table) require `alloc`.
//!
//! The std IO adapter over this sans-IO core is `proxima-http::http2`;
//! it owns the tokio/transport edge and stays `std` by design — the
//! absence of `no_std` there is intentional, not an oversight.

#[cfg(all(feature = "http2_codec-codec-trait", not(feature = "http2_codec-no-alloc")))]
pub mod codec_trait;
#[cfg(not(feature = "http2_codec-no-alloc"))]
pub mod connection;
#[cfg(not(feature = "http2_codec-no-alloc"))]
pub mod frame;
pub mod stream;
#[cfg(not(feature = "http2_codec-no-alloc"))]
pub mod stream_table;

#[cfg(all(feature = "http2_codec-codec-trait", not(feature = "http2_codec-no-alloc")))]
pub use codec_trait::{FrameError as H2FrameError, H2Frame, H2FrameCodec};
