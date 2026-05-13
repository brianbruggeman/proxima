//! WebSocket (RFC 6455) handshake helpers — sans-IO.
//!
//! Complements [`crate::websocket_frame`], whose charter is the framing layer
//! *alone* (it deliberately excludes the handshake). This crate is the matching
//! handshake half: the `Sec-WebSocket-Accept` key derivation (RFC 6455 §1.3)
//! and its verification. Pure string/crypto computation, no I/O — callers wire
//! it into whichever transport performs the upgrade.
//!
//! It exists because the computation was duplicated (downstream clients and
//! `proxima-recording`'s replay module each hand-rolled it) and the framing crate refuses
//! handshake responsibility by design, while `proxima-websocket` is a
//! server-side, async-tungstenite-backed connector — neither is a home.


use alloc::string::String;
use base64::Engine as _;
use sha1::{Digest, Sha1};

/// RFC 6455 §1.3 GUID appended to the client key before hashing.
pub const WS_ACCEPT_MAGIC: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// `Sec-WebSocket-Accept = base64(sha1(client_key + magic))`. The server
/// computes this from the client's `Sec-WebSocket-Key`; a client recomputes it
/// to confirm a genuine websocket peer.
#[must_use]
pub fn compute_accept_key(client_key: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(client_key.as_bytes());
    hasher.update(WS_ACCEPT_MAGIC.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(hasher.finalize())
}

/// True when `server_accept` matches the accept key derived from `client_key`.
/// Whitespace around the header value is ignored.
#[must_use]
pub fn accept_matches(client_key: &str, server_accept: &str) -> bool {
    server_accept.trim() == compute_accept_key(client_key)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn rfc6455_vector() {
        // RFC 6455 §1.3 worked example.
        assert_eq!(
            compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn accept_matches_ignores_surrounding_whitespace() {
        let key = "x3JJHMbDL1EzLkh9GBhXDw==";
        let accept = compute_accept_key(key);
        assert!(accept_matches(key, &format!("  {accept}  ")));
        assert!(!accept_matches(key, "not-the-accept"));
    }
}
