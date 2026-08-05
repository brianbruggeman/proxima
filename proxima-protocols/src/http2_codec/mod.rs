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
//! `--no-default-features --features http2_codec` builds tier-3
//! (`core::*` only) — exposes just [`stream`], the per-stream RFC 7540
//! §5 state machine + flow-control windows (already heap-free).
//! `--features http2_codec-alloc` adds `frame` (owned `Bytes` payload
//! views, `SmallVec` heap fallback), `stream_table` (`BTreeMap`
//! registry), and `connection` (event queue, buffers, HPACK dynamic
//! table), which need a heap.
//!
//! The floor is the BASE feature and `-alloc` widens it, matching
//! [`crate::quic`] / `quic-alloc`. Enabling a feature never removes a
//! module: Cargo unifies features across the whole graph, so a
//! subtractive gate lets one consumer delete API out from under every
//! other one.
//!
//! The std IO adapter over this sans-IO core is `proxima-http::http2`;
//! it owns the tokio/transport edge and stays `std` by design — the
//! absence of `no_std` there is intentional, not an oversight.

#[cfg(feature = "http2_codec-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "http2_codec-alloc")]
pub mod connection;
#[cfg(feature = "http2_codec-alloc")]
pub mod frame;
pub mod stream;
#[cfg(feature = "http2_codec-alloc")]
pub mod stream_table;

#[cfg(feature = "http2_codec-codec-trait")]
pub use codec_trait::{FrameError as H2FrameError, H2Frame, H2FrameCodec};
