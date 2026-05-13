//! Sans-IO HTTP/3 + QPACK protocol implementation per RFC 9114 / 9204 /
//! 9220 / 9297. Publicly consumable outside proxima — designed to be
//! driven from any I/O loop (an in-process QUIC sans-IO loopback for
//! testing; a real transport via `proxima-quic`; a third-party QUIC
//! implementation that exposes a sans-IO stream API).
//!
//! No I/O traits in the signature. No `async`. No `tokio`. No transport
//! coupling — stream bytes in (`&[u8]`), frames + state-machine events out.
//!
//! # Layout
//!
//! Codec leaves (tier-3):
//!
//! - `frame` — RFC 9114 §7 frame codec (DATA, HEADERS, SETTINGS, CANCEL_PUSH,
//!   PUSH_PROMISE, GOAWAY, MAX_PUSH_ID).
//! - `qpack::static_table` — RFC 9204 Appendix A static table; binary search.
//! - `qpack::integer` — QPACK integer encoding (re-uses QUIC varint shape).
//!
//! QPACK (tier-1 — dynamic table is variable-size):
//!
//! - `qpack::encoder` — RFC 9204 encoder with static + dynamic table.
//! - `qpack::decoder` — RFC 9204 decoder under blocked-streams limit.
//!
//! State machines (tier-1 — request fan-out scales unbounded with peer):
//!
//! - `server` / `client` — typestate per request.
//! - `settings` — SETTINGS frame exchange + value validation.
//! - `ext::datagram` — RFC 9297 H3-Datagrams over RFC 9221 QUIC DATAGRAM.
//! - `ext::extended_connect` — RFC 9220 extended CONNECT for WebSocket /
//!   future MASQUE.
//!
//! # Tier
//!
//! `--no-default-features --features http3_codec-alloc` builds tier-1 with
//! `core::*` + `alloc::*` only. `--no-default-features --features
//! http3_codec-no-alloc` builds tier-3 with `core::*` only — exposes the
//! leaf module subset above.
//!
//! The std IO adapter over this sans-IO core is `proxima-http::http3`
//! (driven over `proxima-quic`); it owns the IO edge and stays `std` by
//! design — the absence of `no_std` there is intentional, not an
//! oversight.
//!
//! Discipline per the workspace guiding principles (sans-IO must be
//! enum-shaped state machine, low/no alloc, extreme benching, extreme
//! performance) is binding for every component here.

#[cfg(feature = "http3_codec-codec-trait")]
pub mod codec_trait;
pub mod frame;
pub mod qpack;

#[cfg(feature = "http3_codec-codec-trait")]
pub use codec_trait::{H3CodecError, H3FrameCodec};

#[cfg(feature = "http3_codec-alloc")]
pub mod client;
#[cfg(feature = "http3_codec-alloc")]
pub mod ext;
#[cfg(feature = "http3_codec-alloc")]
pub mod request;
#[cfg(feature = "http3_codec-alloc")]
pub mod server;
#[cfg(feature = "http3_codec-alloc")]
pub mod settings;
