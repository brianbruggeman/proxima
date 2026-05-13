//! Sans-IO HTTP connection-preface classifier — decides h1 vs h2
//! prior-knowledge dispatch from the leading bytes of a fresh
//! connection, before any I/O trait, socket, or async runtime is
//! involved.
//!
//! Folded from the former `proxima-preface-codec` satellite crate
//! (single consumer: this crate's `http` module) into `proxima-listen`
//! as the `preface` module. Carved out of the `http` module's inline
//! byte-sniff (the `dispatch_h1_or_h2` accept-loop helper) the same
//! way `proxima-listen-core` carved connection admission out of the
//! same accept loop: a pure `&[u8] -> decision` function, no sockets,
//! no futures, no spawn. The listener owns reading bytes off the wire
//! and feeding them here; this module only classifies what arrived.
//!
//! Used on transports without ALPN (UDS, plain TCP without TLS). The
//! TLS path doesn't need this classifier — ALPN negotiates h1 vs h2
//! during the handshake instead.
//!
//! # Why a new crate, not a module in `proxima-h1-codec` / `proxima-h2-codec`
//!
//! The decision straddles both protocols without belonging to either:
//! it runs *before* either codec's parser sees a byte, and its only
//! job is choosing which one gets the connection. Housing it inside
//! `proxima-h1-codec` would make h1 the arbiter of h2 dispatch (and
//! vice versa) and force one codec crate to depend on the other's
//! wire constant. `proxima-listen-core` already sits as a dependency-
//! free peer of both, doing the analogous accept-layer admission
//! decision — this module is that same shape, one layer up: a small,
//! self-contained boundary primitive that the `http` module (the
//! shared consumer of both codecs) composes at the accept loop.
//!
//! # Tier
//!
//! Tier-3: no_std, no alloc by construction. The decision is a
//! fixed-size byte compare against a 24-byte constant — no allocation
//! at any point.

/// Length in bytes of the HTTP/2 client connection preface.
pub const H2_CLIENT_PREFACE_LEN: usize = 24;

/// The 24-byte HTTP/2 client connection preface (RFC 9113 §3.4). Sent
/// as the first bytes of any prior-knowledge h2 connection, in place
/// of an HTTP/1.1 request line. The `PRI` pseudo-method is reserved by
/// the spec precisely so it never collides with a real HTTP/1.1
/// method (`GET`, `POST`, ...), which is what makes a short leading-
/// byte sniff enough to route the connection before the full preface
/// arrives.
pub const H2_CLIENT_PREFACE: &[u8; H2_CLIENT_PREFACE_LEN] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Number of leading bytes that disambiguate h1 from h2 prior
/// knowledge. h1's shortest verb ("GET ") and h2's preface ("PRI ")
/// both fill 4 bytes, and no real HTTP/1.1 method starts with `PRI `.
const ROUTE_SNIFF_LEN: usize = 4;

/// Outcome of classifying the leading bytes of a fresh connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrefaceClass {
    /// Not the h2 preface — dispatch as HTTP/1.1. Decided from the
    /// first [`ROUTE_SNIFF_LEN`] bytes; the caller does not need to
    /// wait for more bytes before routing.
    Http1,
    /// The full 24-byte h2 client connection preface matched exactly.
    Http2PriorKnowledge,
    /// Fewer bytes are available than the classifier needs to reach a
    /// decision. The caller reads more and calls again with the
    /// larger buffer — bytes already seen are a prefix of the next
    /// call's buffer, nothing is discarded.
    NeedMoreBytes,
    /// The first [`ROUTE_SNIFF_LEN`] bytes matched `PRI `, committing
    /// the connection to the h2 path, but the remaining bytes did not
    /// match the rest of the canonical preface. `PRI ` is reserved and
    /// never a valid HTTP/1.1 method, so this is neither valid h1 nor
    /// valid h2 — a malformed connection.
    Invalid,
}

/// Classify the leading bytes of a fresh connection as HTTP/1.1,
/// HTTP/2 prior-knowledge, or "need more bytes to decide."
///
/// `bytes` is the buffer accumulated so far, from byte zero of the
/// connection — callers grow it across reads and re-call; they do
/// not slice a fixed window. No I/O, no allocation: a pure function
/// over borrowed input.
#[must_use]
pub fn classify_preface(bytes: &[u8]) -> PrefaceClass {
    if bytes.len() < ROUTE_SNIFF_LEN {
        return PrefaceClass::NeedMoreBytes;
    }
    if bytes[..ROUTE_SNIFF_LEN] != H2_CLIENT_PREFACE[..ROUTE_SNIFF_LEN] {
        return PrefaceClass::Http1;
    }
    if bytes.len() < H2_CLIENT_PREFACE_LEN {
        return PrefaceClass::NeedMoreBytes;
    }
    if bytes[..H2_CLIENT_PREFACE_LEN] == H2_CLIENT_PREFACE[..] {
        PrefaceClass::Http2PriorKnowledge
    } else {
        PrefaceClass::Invalid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the h2 preface bytes, as an h2 client would send them in a single
    // write — the production shape on a plain-TCP prior-knowledge dial.
    #[test]
    fn h2_preface_classifies_as_prior_knowledge() {
        let decision = classify_preface(H2_CLIENT_PREFACE);
        assert_eq!(decision, PrefaceClass::Http2PriorKnowledge);
    }

    // the production shape for the overwhelming majority of h1 traffic.
    #[test]
    fn h1_get_request_line_classifies_as_http1() {
        let decision = classify_preface(b"GET / HTTP/1.1\r\n");
        assert_eq!(decision, PrefaceClass::Http1);
    }

    // CONNECT (proxy tunneling) is a real h1 verb whose first 4 bytes
    // still diverge from "PRI " at byte 0 — same code path as GET.
    #[test]
    fn h1_connect_request_line_classifies_as_http1() {
        let decision = classify_preface(b"CONNECT example.com:443 HTTP/1.1\r\n");
        assert_eq!(decision, PrefaceClass::Http1);
    }

    #[test]
    fn empty_buffer_needs_more_bytes() {
        let decision = classify_preface(b"");
        assert_eq!(decision, PrefaceClass::NeedMoreBytes);
    }

    // fewer than the route-sniff length: too early to tell "GET " from "PRI ".
    #[test]
    fn partial_buffer_under_sniff_length_needs_more_bytes() {
        let decision = classify_preface(b"GE");
        assert_eq!(decision, PrefaceClass::NeedMoreBytes);
    }

    // first 4 bytes committed to h2, but the rest of the 24-byte preface
    // has not arrived yet — must not decide before it does.
    #[test]
    fn partial_h2_preface_needs_more_bytes_before_verifying() {
        let decision = classify_preface(&H2_CLIENT_PREFACE[..10]);
        assert_eq!(decision, PrefaceClass::NeedMoreBytes);
    }

    // "PRI " is reserved and never a valid h1 method, so a mismatch here
    // is protocol garbage, not a legitimate h1 request.
    #[test]
    fn pri_prefixed_garbage_is_invalid_not_http1() {
        let mut malformed = *H2_CLIENT_PREFACE;
        malformed[H2_CLIENT_PREFACE_LEN - 1] = b'X';
        let decision = classify_preface(&malformed);
        assert_eq!(decision, PrefaceClass::Invalid);
    }
}
