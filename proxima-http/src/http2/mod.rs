//! Native HTTP/2 implementation. No `h2` crate dependency.
//!
//! Targets `futures::io::AsyncRead + AsyncWrite` so DPDK / glommio /
//! any non-tokio transport plugs in directly via the substrate's
//! `StreamConnection` trait.
//!
//! Built in layers, each with its own module:
//!
//! - [`frame`]: wire framing — 9-byte header + per-type payloads.
//!   Parse / encode. State-machine-free; pure bytes.
//! - [`hpack`]: header compression (RFC 7541) — integer / string
//!   literal codec, Huffman, static + dynamic tables, encoder /
//!   decoder.
//! - [`stream`]: per-stream state machine + per-stream flow-control
//!   windows (RFC 7540 §5).
//! - `conn`: connection lifecycle (preface, SETTINGS exchange,
//!   GOAWAY, drive loop) (planned).
//!
//! Today: framing + HPACK + stream state machine implemented.
//! Connection driver lands next. The `h2`-crate-based listener at
//! `proxima::listeners::h2` ships alongside until the native impl
//! reaches feature parity, then replaces it.

// Sans-IO codec lives in proxima-protocols::http2_codec; re-exported
// here so existing `proxima_http::http2::{frame, stream, ...}` call
// sites keep working.
pub use proxima_protocols::http2_codec::{connection, frame, stream, stream_table};

pub use proxima_protocols::hpack;
pub mod client;
pub mod listener;
pub mod server;

pub use client::H2ClientUpstream;
pub use listener::H2ListenProtocol;
pub use server::serve_h2_connection;
