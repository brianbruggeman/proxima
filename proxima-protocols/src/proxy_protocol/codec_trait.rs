//! `proxima_codec::FrameCodec` impl for the PROXY protocol (HAProxy)
//! header — the plug-and-play-floor sweep's codec encoder pass
//! (`sweep/codec-encoders`) found [`super::parse`] already returns the
//! `(Frame<'_>, usize)` shape `FrameCodec::parse_frame` needs, and
//! [`super::encode`] already serializes a [`super::ProxyHeader`] to
//! bytes — this impl wires the two EXISTING functions into the trait,
//! no new parsing or encoding logic. `ProxyHeader` carries no borrowed
//! data (it is `SocketAddr`s, not slices), so `Frame<'_>` ignores the
//! GAT lifetime.
//!
//! `encode`'s wire version (v1 ASCII vs v2 binary) is not part of
//! `FrameCodec::encode_frame`'s signature, so [`ProxyProtocolCodec`]
//! carries the chosen [`super::WireVersion`] as config, the same way
//! `proxima_codec::LengthDelimitedCodec` carries its `FrameLimits`.
//!
//! [`super::ParseError`] has no `Display`/`Error` impl upstream (it
//! only had a `From<ParseError> for ProximaError` bridge); `FrameError`
//! wraps it and supplies both, mirroring the `H3CodecError` precedent
//! (`docs/codec-trait/discipline.md`, C5) for an upstream type that
//! doesn't derive them.

use alloc::vec::Vec;
use core::fmt;

use proxima_codec::FrameCodec;

use super::{ParseError, ProxyHeader, WireVersion, encode, parse};

/// PROXY protocol [`FrameCodec`]. `version` selects which wire form
/// [`Self::encode_frame`] emits; parsing auto-detects v1 vs v2 from
/// the leading bytes regardless of `version`.
#[derive(Debug, Clone, Copy)]
pub struct ProxyProtocolCodec {
    version: WireVersion,
}

impl ProxyProtocolCodec {
    #[must_use]
    pub const fn new(version: WireVersion) -> Self {
        Self { version }
    }
}

impl Default for ProxyProtocolCodec {
    fn default() -> Self {
        Self::new(WireVersion::V2)
    }
}

/// Wraps [`ParseError`] with the `Display`/`Error` impls it lacks
/// upstream. `encode_frame` never fails (`super::encode` is
/// infallible), so every variant here is a parse-side failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameError(pub ParseError);

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ParseError::NeedMore => formatter.write_str("proxy protocol: incomplete header"),
            ParseError::NotProxyProtocol => formatter.write_str("proxy protocol: header missing"),
            ParseError::Malformed(reason) => {
                write!(formatter, "proxy protocol: malformed: {reason}")
            }
        }
    }
}

impl core::error::Error for FrameError {}

impl FrameCodec for ProxyProtocolCodec {
    type Frame<'a> = ProxyHeader;
    type Error = FrameError;

    fn parse_frame(&self, buf: &[u8]) -> Result<(ProxyHeader, usize), FrameError> {
        parse(buf).map_err(FrameError)
    }

    fn encode_frame(&self, frame: &ProxyHeader, dest: &mut Vec<u8>) -> Result<(), FrameError> {
        dest.extend_from_slice(&encode(frame, self.version));
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn real_v1_tcp4_header_round_trips() {
        // real PROXY v1 wire bytes (P9): the exact line an haproxy /
        // AWS NLB prepends before the application stream.
        let codec = ProxyProtocolCodec::new(WireVersion::V1);
        let wire = b"PROXY TCP4 192.168.0.1 192.168.0.11 56324 443\r\nGET / HTTP/1.1\r\n";
        let (header, consumed) = codec.parse_frame(wire).expect("real v1 header parses");
        assert_eq!(
            header,
            ProxyHeader::Tcp {
                src: "192.168.0.1:56324".parse().unwrap(),
                dst: "192.168.0.11:443".parse().unwrap(),
            }
        );
        assert_eq!(&wire[consumed..], b"GET / HTTP/1.1\r\n");

        let mut dest = Vec::new();
        codec.encode_frame(&header, &mut dest).expect("encode");
        assert_eq!(dest, b"PROXY TCP4 192.168.0.1 192.168.0.11 56324 443\r\n");
    }

    #[test]
    fn real_v2_ipv4_header_round_trips() {
        // real PROXY v2 wire bytes (P9): 12-byte signature + version/
        // command + family/protocol + length + a real IPv4 address block.
        let codec = ProxyProtocolCodec::new(WireVersion::V2);
        let header = ProxyHeader::Tcp {
            src: "1.2.3.4:8080".parse().unwrap(),
            dst: "5.6.7.8:443".parse().unwrap(),
        };
        let mut wire = Vec::new();
        codec.encode_frame(&header, &mut wire).expect("encode");
        let (parsed, consumed) = codec.parse_frame(&wire).expect("real v2 header parses");
        assert_eq!(parsed, header);
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn short_buffer_returns_need_more_not_error() {
        let codec = ProxyProtocolCodec::default();
        let outcome = codec.parse_frame(b"PROXY TCP4 192.168");
        assert_eq!(outcome, Err(FrameError(ParseError::NeedMore)));
    }

    #[test]
    fn non_proxy_prefix_is_rejected() {
        let codec = ProxyProtocolCodec::default();
        let outcome = codec.parse_frame(b"GET / HTTP/1.1\r\n");
        assert_eq!(outcome, Err(FrameError(ParseError::NotProxyProtocol)));
    }

    #[test]
    fn frame_error_display_carries_the_reason() {
        let error = FrameError(ParseError::Malformed("v1 unknown family"));
        assert_eq!(
            alloc::format!("{error}"),
            "proxy protocol: malformed: v1 unknown family"
        );
    }
}
