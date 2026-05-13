//! Extended CONNECT per [RFC 9220].
//!
//! Extends the HTTP/3 CONNECT method to negotiate alternate protocols
//! (WebSocket-over-HTTP/3 per RFC 9220; future MASQUE per
//! draft-ietf-masque-*). The wire-level addition is a single
//! pseudo-header field `:protocol` that names the desired sub-protocol.
//!
//! Negotiation: server advertises `SETTINGS_ENABLE_CONNECT_PROTOCOL = 1`
//! (see [`crate::http3_codec::settings::SETTINGS_ENABLE_CONNECT_PROTOCOL`]). Client
//! then sends:
//!
//! ```text
//! :method = CONNECT
//! :scheme = https
//! :authority = ...
//! :path = ...
//! :protocol = websocket    # the RFC 9220 addition
//! ```
//!
//! [RFC 9220]: https://www.rfc-editor.org/rfc/rfc9220
//!
//! # Tier
//!
//! Tier-1 (alloc) — convenience helpers + validation.

use alloc::vec::Vec;

use crate::http3_codec::qpack::decoder::DecodedField;

/// Pseudo-header name for the extended CONNECT `:protocol` field.
pub const PSEUDO_HEADER_PROTOCOL: &[u8] = b":protocol";

/// Extended CONNECT validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExtendedConnectError {
    /// `:method` was not "CONNECT".
    NotConnectMethod,
    /// `:protocol` pseudo-header was missing.
    MissingProtocol,
    /// `:protocol` was present but had no value.
    EmptyProtocol,
}

/// Build a CONNECT request header set for `protocol` over the given
/// authority and path. Returns the pseudo-header tuples ready for
/// the QPACK encoder.
#[must_use]
pub fn build_request_headers<'a>(
    authority: &'a [u8],
    path: &'a [u8],
    scheme: &'a [u8],
    protocol: &'a [u8],
) -> [(&'a [u8], &'a [u8]); 5] {
    [
        (b":method", b"CONNECT"),
        (b":scheme", scheme),
        (b":authority", authority),
        (b":path", path),
        (PSEUDO_HEADER_PROTOCOL, protocol),
    ]
}

/// Validate a decoded request header set against the extended CONNECT
/// shape per RFC 9220 §3.
///
/// Returns the `:protocol` value on success.
///
/// # Errors
///
/// See [`ExtendedConnectError`].
pub fn validate_request(headers: &[DecodedField]) -> Result<&[u8], ExtendedConnectError> {
    let mut method: Option<&[u8]> = None;
    let mut protocol: Option<&[u8]> = None;
    for field in headers {
        if field.name == b":method" {
            method = Some(&field.value);
        } else if field.name == PSEUDO_HEADER_PROTOCOL {
            protocol = Some(&field.value);
        }
    }
    let method = method.ok_or(ExtendedConnectError::NotConnectMethod)?;
    if method != b"CONNECT" {
        return Err(ExtendedConnectError::NotConnectMethod);
    }
    let protocol = protocol.ok_or(ExtendedConnectError::MissingProtocol)?;
    if protocol.is_empty() {
        return Err(ExtendedConnectError::EmptyProtocol);
    }
    Ok(protocol)
}

/// Owned header set ready to hand to the encoder. Convenience wrapper
/// over [`build_request_headers`] when the caller's lifetimes don't
/// permit the borrowed return.
#[must_use]
pub fn build_request_headers_owned(
    authority: &[u8],
    path: &[u8],
    scheme: &[u8],
    protocol: &[u8],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    alloc::vec![
        (b":method".to_vec(), b"CONNECT".to_vec()),
        (b":scheme".to_vec(), scheme.to_vec()),
        (b":authority".to_vec(), authority.to_vec()),
        (b":path".to_vec(), path.to_vec()),
        (PSEUDO_HEADER_PROTOCOL.to_vec(), protocol.to_vec()),
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn field(name: &[u8], value: &[u8]) -> DecodedField {
        DecodedField {
            name: name.to_vec(),
            value: value.to_vec(),
        }
    }

    #[test]
    fn build_request_headers_yields_five_pseudo_headers() {
        let headers = build_request_headers(b"example.com", b"/chat", b"https", b"websocket");
        assert_eq!(headers.len(), 5);
        assert_eq!(headers[0], (b":method".as_slice(), b"CONNECT".as_slice()));
        assert_eq!(
            headers[4],
            (PSEUDO_HEADER_PROTOCOL, b"websocket".as_slice())
        );
    }

    #[test]
    fn validate_request_recognizes_websocket() {
        let headers = alloc::vec![
            field(b":method", b"CONNECT"),
            field(b":scheme", b"https"),
            field(b":authority", b"example.com"),
            field(b":path", b"/chat"),
            field(PSEUDO_HEADER_PROTOCOL, b"websocket"),
        ];
        let protocol = validate_request(&headers).unwrap();
        assert_eq!(protocol, b"websocket");
    }

    #[test]
    fn validate_request_rejects_get_method() {
        let headers = alloc::vec![
            field(b":method", b"GET"),
            field(PSEUDO_HEADER_PROTOCOL, b"websocket"),
        ];
        assert_eq!(
            validate_request(&headers),
            Err(ExtendedConnectError::NotConnectMethod)
        );
    }

    #[test]
    fn validate_request_rejects_missing_protocol() {
        let headers = alloc::vec![field(b":method", b"CONNECT"), field(b":scheme", b"https"),];
        assert_eq!(
            validate_request(&headers),
            Err(ExtendedConnectError::MissingProtocol)
        );
    }

    #[test]
    fn validate_request_rejects_empty_protocol() {
        let headers = alloc::vec![
            field(b":method", b"CONNECT"),
            field(PSEUDO_HEADER_PROTOCOL, b""),
        ];
        assert_eq!(
            validate_request(&headers),
            Err(ExtendedConnectError::EmptyProtocol)
        );
    }
}
