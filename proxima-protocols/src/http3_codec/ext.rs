//! HTTP/3 extension modules.
//!
//! - [`datagram`] — H3-Datagrams per RFC 9297 (composes with RFC 9221
//!   QUIC DATAGRAM frames at the QUIC layer).
//! - [`extended_connect`] — RFC 9220 extended CONNECT method (used by
//!   WebSocket-over-HTTP/3 + future MASQUE proxies).

#[cfg(feature = "http3_codec-alloc")]
pub mod datagram;
#[cfg(feature = "http3_codec-alloc")]
pub mod extended_connect;
