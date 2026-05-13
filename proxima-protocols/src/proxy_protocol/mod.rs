//! PROXY protocol (HAProxy) v1 + v2 header parsing.
//!
//! When proxima sits behind a TCP load balancer (AWS NLB, GCP TCP LB,
//! haproxy, nginx stream, envoy in L4 mode), the upstream socket
//! address proxima sees is the LB's, not the original client's. The
//! PROXY protocol prepends a small header to the TCP stream the LB
//! and the upstream agree on, carrying the real client address.
//!
//! Both versions are supported:
//!
//! - **v1** — ASCII line `PROXY <FAMILY> <SRC> <DST> <SRC_PORT> <DST_PORT>\r\n`
//!   terminated by CRLF; max 107 bytes per spec. Family is one of
//!   `TCP4`, `TCP6`, `UNKNOWN`. We accept `UNKNOWN` (no addresses)
//!   and return [`ProxyHeader::Unknown`] — the upstream chooses
//!   whether to keep or reject.
//! - **v2** — 16-byte fixed prefix (12-byte signature + 4-byte
//!   header) followed by an address block whose length is in the
//!   header. We parse TCP/UDP over IPv4/IPv6 and accept LOCAL (the
//!   LB's own health-check probes).
//!
//! Spec: <https://www.haproxy.org/download/2.4/doc/proxy-protocol.txt>

#[cfg(feature = "proxy_protocol-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "proxy_protocol-codec-trait")]
pub use codec_trait::{FrameError as ProxyProtocolFrameError, ProxyProtocolCodec};

use alloc::format;
use alloc::vec::Vec;
use core::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

#[cfg(feature = "proxy_protocol-std")]
use futures::io::{AsyncRead, AsyncReadExt};

use proxima_core::ProximaError;

/// v1's hard cap (107 bytes) plus a safety margin for v2 short
/// headers; large enough for any v2 address block we accept.
#[cfg(feature = "proxy_protocol-std")]
const HEADER_READ_BUDGET: usize = 256;

/// Decoded PROXY header. `Tcp` carries the real client + dest
/// addresses; `Unknown` / `Local` are the "no address info" variants
/// the upstream usually treats as either trust-the-socket or reject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyHeader {
    /// Genuine proxied connection. `src` = original client address,
    /// `dst` = address the client originally hit on the LB.
    Tcp { src: SocketAddr, dst: SocketAddr },
    /// v1 `UNKNOWN` family. Upstream should fall back to the raw
    /// socket's `peer_addr()`.
    Unknown,
    /// v2 `LOCAL` command. Used by the LB's own health-check probes;
    /// no address translation applies.
    Local,
}

/// v2 signature: `\r\n\r\n\x00\r\nQUIT\n`. 12 bytes.
const V2_SIGNATURE: &[u8; 12] = b"\r\n\r\n\x00\r\nQUIT\n";
const V1_PREFIX: &[u8; 6] = b"PROXY ";

/// Read PROXY off a tokio socket. Mirrors [`read_header`] but uses
/// tokio's AsyncRead so the http listener can call it directly on
/// `tokio::net::TcpStream` before the TLS handshake.
#[cfg(feature = "proxy_protocol-tokio-runtime")]
pub async fn read_header_tokio<R>(stream: &mut R) -> Result<(ProxyHeader, Vec<u8>), ProximaError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut buf = Vec::with_capacity(64);
    loop {
        if buf.len() >= HEADER_READ_BUDGET {
            return Err(ProximaError::Upstream(
                "proxy protocol: header exceeded budget".into(),
            ));
        }
        let mut chunk = [0u8; 64];
        let read = tokio::io::AsyncReadExt::read(stream, &mut chunk)
            .await
            .map_err(|err| ProximaError::Upstream(format!("proxy protocol: read: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "proxy protocol: connection closed during header".into(),
            ));
        }
        buf.extend_from_slice(&chunk[..read]);
        match parse(&buf) {
            Ok((header, consumed)) => {
                let leftover = buf.split_off(consumed);
                return Ok((header, leftover));
            }
            Err(ParseError::NeedMore) => continue,
            Err(other) => return Err(other.into()),
        }
    }
}

/// Read a PROXY header off the front of `stream`, returning the
/// decoded header. Any bytes read past the header (start of the
/// application protocol — h1 request line, TLS ClientHello, etc.)
/// are returned in `leftover` so the caller can prepend them to the
/// downstream reader. Useful pattern: chain `Cursor::new(leftover)`
/// with the original socket via a small "prepended-reader" wrapper.
#[cfg(feature = "proxy_protocol-std")]
pub async fn read_header<R>(stream: &mut R) -> Result<(ProxyHeader, Vec<u8>), ProximaError>
where
    R: AsyncRead + Unpin,
{
    let mut buf = Vec::with_capacity(64);
    loop {
        if buf.len() >= HEADER_READ_BUDGET {
            return Err(ProximaError::Upstream(
                "proxy protocol: header exceeded budget".into(),
            ));
        }
        let mut chunk = [0u8; 64];
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|err| ProximaError::Upstream(format!("proxy protocol: read: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "proxy protocol: connection closed during header".into(),
            ));
        }
        buf.extend_from_slice(&chunk[..read]);
        match parse(&buf) {
            Ok((header, consumed)) => {
                let leftover = buf.split_off(consumed);
                return Ok((header, leftover));
            }
            Err(ParseError::NeedMore) => continue,
            Err(other) => return Err(other.into()),
        }
    }
}

/// Wire version selector for [`encode`]. v2 (binary) is the modern
/// default — smaller, faster, fixed-size address blocks. v1 (ASCII)
/// is kept for older receivers (haproxy < 1.5.5, some hardware LBs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVersion {
    V1,
    V2,
}

/// Encode a PROXY header for prepending to an outgoing connection.
/// Closes the receive-but-can't-send asymmetry: proxima can now
/// originate PROXY-protected connections to upstreams that expect
/// the header, mirroring how the http listener consumes inbound
/// PROXY headers.
pub fn encode(header: &ProxyHeader, version: WireVersion) -> Vec<u8> {
    match version {
        WireVersion::V1 => encode_v1(header),
        WireVersion::V2 => encode_v2(header),
    }
}

fn encode_v1(header: &ProxyHeader) -> Vec<u8> {
    match header {
        ProxyHeader::Tcp { src, dst } => {
            let family = match (src.ip(), dst.ip()) {
                (IpAddr::V4(_), IpAddr::V4(_)) => "TCP4",
                (IpAddr::V6(_), IpAddr::V6(_)) => "TCP6",
                _ => {
                    // RFC: mixed-family addresses aren't representable
                    // in v1. Fall back to UNKNOWN; the upstream sees
                    // no address info, which is the documented escape
                    // hatch.
                    return b"PROXY UNKNOWN\r\n".to_vec();
                }
            };
            format!(
                "PROXY {family} {} {} {} {}\r\n",
                src.ip(),
                dst.ip(),
                src.port(),
                dst.port(),
            )
            .into_bytes()
        }
        ProxyHeader::Unknown | ProxyHeader::Local => b"PROXY UNKNOWN\r\n".to_vec(),
    }
}

fn encode_v2(header: &ProxyHeader) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + 36);
    buf.extend_from_slice(V2_SIGNATURE);
    let (version_command, family_protocol, payload) = match header {
        ProxyHeader::Tcp { src, dst } => match (src.ip(), dst.ip()) {
            (IpAddr::V4(src_ip), IpAddr::V4(dst_ip)) => {
                let mut payload = Vec::with_capacity(12);
                payload.extend_from_slice(&src_ip.octets());
                payload.extend_from_slice(&dst_ip.octets());
                payload.extend_from_slice(&src.port().to_be_bytes());
                payload.extend_from_slice(&dst.port().to_be_bytes());
                // 0x21 = v2 (high nibble) | PROXY command (low nibble).
                // 0x11 = AF_INET (high nibble) | TCP/SOCK_STREAM (low nibble).
                (0x21u8, 0x11u8, payload)
            }
            (IpAddr::V6(src_ip), IpAddr::V6(dst_ip)) => {
                let mut payload = Vec::with_capacity(36);
                payload.extend_from_slice(&src_ip.octets());
                payload.extend_from_slice(&dst_ip.octets());
                payload.extend_from_slice(&src.port().to_be_bytes());
                payload.extend_from_slice(&dst.port().to_be_bytes());
                // 0x21 | AF_INET6 (0x21) | TCP (0x1).
                (0x21u8, 0x21u8, payload)
            }
            _ => {
                // Mixed-family — emit UNSPEC payload.
                (0x21u8, 0x00u8, Vec::new())
            }
        },
        ProxyHeader::Unknown => (0x21u8, 0x00u8, Vec::new()),
        // LOCAL command: 0x20 = v2 | LOCAL (low nibble 0).
        ProxyHeader::Local => (0x20u8, 0x00u8, Vec::new()),
    };
    buf.push(version_command);
    buf.push(family_protocol);
    let length = payload.len() as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Async helper: emit a PROXY header on `stream` before any
/// application bytes go out. Use this at the start of an outbound
/// connection that targets an upstream expecting PROXY protocol.
#[cfg(feature = "proxy_protocol-tokio-runtime")]
pub async fn write_header_tokio<W>(
    stream: &mut W,
    header: &ProxyHeader,
    version: WireVersion,
) -> Result<(), ProximaError>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let bytes = encode(header, version);
    tokio::io::AsyncWriteExt::write_all(stream, &bytes)
        .await
        .map_err(|err| ProximaError::Upstream(format!("proxy protocol: write: {err}")))?;
    tokio::io::AsyncWriteExt::flush(stream)
        .await
        .map_err(|err| ProximaError::Upstream(format!("proxy protocol: flush: {err}")))?;
    Ok(())
}

/// Parse a PROXY header from `buf`. On success returns the decoded
/// header and the number of bytes consumed. On `NeedMore` the caller
/// must read more bytes and retry.
pub fn parse(buf: &[u8]) -> Result<(ProxyHeader, usize), ParseError> {
    if buf.is_empty() {
        return Err(ParseError::NeedMore);
    }
    if buf.starts_with(V2_SIGNATURE) {
        return parse_v2(buf);
    }
    if buf.starts_with(V1_PREFIX) {
        return parse_v1(buf);
    }
    // Could be either: we have <6 bytes of an ambiguous prefix.
    // If buf prefix could still grow into either v1 or v2, ask for
    // more bytes; otherwise reject.
    if buf.len() < V2_SIGNATURE.len() && V2_SIGNATURE.starts_with(buf) {
        return Err(ParseError::NeedMore);
    }
    if buf.len() < V1_PREFIX.len() && V1_PREFIX.starts_with(buf) {
        return Err(ParseError::NeedMore);
    }
    Err(ParseError::NotProxyProtocol)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Not enough bytes to decide; read more and retry.
    NeedMore,
    /// Prefix is neither v1 nor v2 — caller should fall back to
    /// treating bytes as application data.
    NotProxyProtocol,
    /// Prefix matched but the rest of the header is malformed.
    Malformed(&'static str),
}

impl From<ParseError> for ProximaError {
    fn from(value: ParseError) -> Self {
        match value {
            ParseError::NeedMore => ProximaError::Upstream("proxy protocol: short read".into()),
            ParseError::NotProxyProtocol => {
                ProximaError::Upstream("proxy protocol: header missing".into())
            }
            ParseError::Malformed(reason) => {
                ProximaError::Upstream(format!("proxy protocol: malformed: {reason}"))
            }
        }
    }
}

fn parse_v1(buf: &[u8]) -> Result<(ProxyHeader, usize), ParseError> {
    let crlf = match find_crlf(buf, 107) {
        FindCrlf::Found(pos) => pos,
        FindCrlf::NeedMore => return Err(ParseError::NeedMore),
        FindCrlf::TooLong => return Err(ParseError::Malformed("v1 header > 107 bytes")),
    };
    let line = &buf[V1_PREFIX.len()..crlf];
    let text = core::str::from_utf8(line).map_err(|_| ParseError::Malformed("v1 not ASCII"))?;
    let mut parts = text.split(' ');
    let family = parts
        .next()
        .ok_or(ParseError::Malformed("v1 family missing"))?;
    match family {
        "UNKNOWN" => {
            // RFC says ignore everything after UNKNOWN. Consume.
            Ok((ProxyHeader::Unknown, crlf + 2))
        }
        "TCP4" | "TCP6" => {
            let src_ip = parts
                .next()
                .ok_or(ParseError::Malformed("v1 src ip missing"))?;
            let dst_ip = parts
                .next()
                .ok_or(ParseError::Malformed("v1 dst ip missing"))?;
            let src_port = parts
                .next()
                .ok_or(ParseError::Malformed("v1 src port missing"))?
                .parse::<u16>()
                .map_err(|_| ParseError::Malformed("v1 src port"))?;
            let dst_port = parts
                .next()
                .ok_or(ParseError::Malformed("v1 dst port missing"))?
                .parse::<u16>()
                .map_err(|_| ParseError::Malformed("v1 dst port"))?;
            if parts.next().is_some() {
                return Err(ParseError::Malformed("v1 trailing tokens"));
            }
            let src = parse_ip(family, src_ip)?;
            let dst = parse_ip(family, dst_ip)?;
            Ok((
                ProxyHeader::Tcp {
                    src: SocketAddr::new(src, src_port),
                    dst: SocketAddr::new(dst, dst_port),
                },
                crlf + 2,
            ))
        }
        _ => Err(ParseError::Malformed("v1 unknown family")),
    }
}

fn parse_ip(family: &str, raw: &str) -> Result<IpAddr, ParseError> {
    match family {
        "TCP4" => raw
            .parse::<Ipv4Addr>()
            .map(IpAddr::V4)
            .map_err(|_| ParseError::Malformed("v1 invalid ipv4")),
        "TCP6" => raw
            .parse::<Ipv6Addr>()
            .map(IpAddr::V6)
            .map_err(|_| ParseError::Malformed("v1 invalid ipv6")),
        _ => Err(ParseError::Malformed("v1 unknown family")),
    }
}

enum FindCrlf {
    Found(usize),
    NeedMore,
    TooLong,
}

fn find_crlf(buf: &[u8], max_pos: usize) -> FindCrlf {
    let limit = buf.len().min(max_pos);
    for index in 0..limit.saturating_sub(1) {
        if buf[index] == b'\r' && buf[index + 1] == b'\n' {
            return FindCrlf::Found(index);
        }
    }
    if buf.len() > max_pos {
        FindCrlf::TooLong
    } else {
        FindCrlf::NeedMore
    }
}

fn parse_v2(buf: &[u8]) -> Result<(ProxyHeader, usize), ParseError> {
    if buf.len() < 16 {
        return Err(ParseError::NeedMore);
    }
    let version_and_command = buf[12];
    let family_and_protocol = buf[13];
    let length = u16::from_be_bytes([buf[14], buf[15]]) as usize;
    if (version_and_command >> 4) != 2 {
        return Err(ParseError::Malformed("v2 version != 2"));
    }
    let command = version_and_command & 0x0F;
    if buf.len() < 16 + length {
        return Err(ParseError::NeedMore);
    }
    let total = 16 + length;
    let payload = &buf[16..total];
    let header = match command {
        // LOCAL — health-check probe; ignore addresses.
        0x00 => ProxyHeader::Local,
        // PROXY — real proxied connection.
        0x01 => decode_v2_addresses(family_and_protocol, payload)?,
        _ => return Err(ParseError::Malformed("v2 unknown command")),
    };
    Ok((header, total))
}

fn decode_v2_addresses(family_and_protocol: u8, payload: &[u8]) -> Result<ProxyHeader, ParseError> {
    let family = family_and_protocol >> 4;
    let protocol = family_and_protocol & 0x0F;
    match (family, protocol) {
        // AF_UNSPEC or UNSPEC protocol — treat as Unknown.
        (0, _) | (_, 0) => Ok(ProxyHeader::Unknown),
        // AF_INET / TCP or UDP
        (1, _) => {
            if payload.len() < 12 {
                return Err(ParseError::Malformed("v2 ipv4 payload short"));
            }
            let src = Ipv4Addr::new(payload[0], payload[1], payload[2], payload[3]);
            let dst = Ipv4Addr::new(payload[4], payload[5], payload[6], payload[7]);
            let src_port = u16::from_be_bytes([payload[8], payload[9]]);
            let dst_port = u16::from_be_bytes([payload[10], payload[11]]);
            Ok(ProxyHeader::Tcp {
                src: SocketAddr::new(IpAddr::V4(src), src_port),
                dst: SocketAddr::new(IpAddr::V4(dst), dst_port),
            })
        }
        // AF_INET6
        (2, _) => {
            if payload.len() < 36 {
                return Err(ParseError::Malformed("v2 ipv6 payload short"));
            }
            let mut src_bytes = [0u8; 16];
            let mut dst_bytes = [0u8; 16];
            src_bytes.copy_from_slice(&payload[0..16]);
            dst_bytes.copy_from_slice(&payload[16..32]);
            let src = Ipv6Addr::from(src_bytes);
            let dst = Ipv6Addr::from(dst_bytes);
            let src_port = u16::from_be_bytes([payload[32], payload[33]]);
            let dst_port = u16::from_be_bytes([payload[34], payload[35]]);
            Ok(ProxyHeader::Tcp {
                src: SocketAddr::new(IpAddr::V6(src), src_port),
                dst: SocketAddr::new(IpAddr::V6(dst), dst_port),
            })
        }
        // AF_UNIX (3) — proxima doesn't surface unix peer addrs from
        // upstream PROXY headers today. Treat as Unknown rather than
        // reject; the operator can layer policy if they care.
        (3, _) => Ok(ProxyHeader::Unknown),
        _ => Err(ParseError::Malformed("v2 unknown family")),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn v1_tcp4_parses() {
        let line = b"PROXY TCP4 192.168.0.1 192.168.0.11 56324 443\r\nGET / HTTP/1.1\r\n";
        let (header, consumed) = parse(line).unwrap();
        let ProxyHeader::Tcp { src, dst } = header else {
            panic!("expected TCP variant");
        };
        assert_eq!(src, "192.168.0.1:56324".parse().unwrap());
        assert_eq!(dst, "192.168.0.11:443".parse().unwrap());
        assert_eq!(&line[consumed..consumed + 4], b"GET ");
    }

    #[test]
    fn v1_tcp6_parses() {
        let line = b"PROXY TCP6 fe80::1 fe80::2 12345 443\r\n";
        let (header, consumed) = parse(line).unwrap();
        let ProxyHeader::Tcp { src, dst } = header else {
            panic!();
        };
        assert_eq!(src.port(), 12345);
        assert_eq!(dst.port(), 443);
        assert_eq!(consumed, line.len());
    }

    #[test]
    fn v1_unknown_consumed_and_reported() {
        let line = b"PROXY UNKNOWN\r\nGET ";
        let (header, consumed) = parse(line).unwrap();
        assert_eq!(header, ProxyHeader::Unknown);
        assert_eq!(consumed, b"PROXY UNKNOWN\r\n".len());
    }

    #[test]
    fn v1_short_returns_need_more() {
        let line = b"PROXY TCP4 192.168";
        assert_eq!(parse(line), Err(ParseError::NeedMore));
    }

    #[test]
    fn v1_over_107_bytes_rejected() {
        let mut buf = b"PROXY TCP4 ".to_vec();
        buf.extend(std::iter::repeat_n(b'1', 110));
        let outcome = parse(&buf);
        assert!(matches!(outcome, Err(ParseError::Malformed(_))));
    }

    #[test]
    fn v2_ipv4_parses() {
        let mut buf = V2_SIGNATURE.to_vec();
        buf.push(0x21); // version 2 | PROXY command
        buf.push(0x11); // AF_INET | TCP
        buf.extend_from_slice(&12u16.to_be_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4]); // src ip
        buf.extend_from_slice(&[5, 6, 7, 8]); // dst ip
        buf.extend_from_slice(&8080u16.to_be_bytes()); // src port
        buf.extend_from_slice(&443u16.to_be_bytes()); // dst port
        let (header, consumed) = parse(&buf).unwrap();
        let ProxyHeader::Tcp { src, dst } = header else {
            panic!();
        };
        assert_eq!(src, "1.2.3.4:8080".parse().unwrap());
        assert_eq!(dst, "5.6.7.8:443".parse().unwrap());
        assert_eq!(consumed, 16 + 12);
    }

    #[test]
    fn v2_local_command_is_local_variant() {
        let mut buf = V2_SIGNATURE.to_vec();
        buf.push(0x20); // version 2 | LOCAL command
        buf.push(0x00);
        buf.extend_from_slice(&0u16.to_be_bytes());
        let (header, consumed) = parse(&buf).unwrap();
        assert_eq!(header, ProxyHeader::Local);
        assert_eq!(consumed, 16);
    }

    #[test]
    fn v2_short_returns_need_more() {
        let mut buf = V2_SIGNATURE.to_vec();
        buf.push(0x21);
        buf.push(0x11);
        buf.extend_from_slice(&12u16.to_be_bytes());
        // payload missing entirely
        assert_eq!(parse(&buf), Err(ParseError::NeedMore));
    }

    #[test]
    fn v2_bad_version_rejected() {
        let mut buf = V2_SIGNATURE.to_vec();
        buf.push(0x31); // version 3 — not v2
        buf.push(0x11);
        buf.extend_from_slice(&0u16.to_be_bytes());
        assert!(matches!(parse(&buf), Err(ParseError::Malformed(_))));
    }

    #[test]
    fn non_proxy_prefix_rejected() {
        let line = b"GET / HTTP/1.1\r\n";
        assert_eq!(parse(line), Err(ParseError::NotProxyProtocol));
    }

    #[test]
    fn ambiguous_short_v1_prefix_asks_for_more() {
        // "PROX" — could still grow into "PROXY ".
        assert_eq!(parse(b"PROX"), Err(ParseError::NeedMore));
    }

    #[test]
    fn ambiguous_short_v2_prefix_asks_for_more() {
        // \r\n\r\n — first four bytes of v2 signature.
        assert_eq!(parse(b"\r\n\r\n"), Err(ParseError::NeedMore));
    }

    #[test]
    fn encode_v1_tcp4_then_parses_back_to_same_header() {
        let header = ProxyHeader::Tcp {
            src: "10.0.0.1:4000".parse().unwrap(),
            dst: "10.0.0.2:443".parse().unwrap(),
        };
        let bytes = encode(&header, WireVersion::V1);
        assert!(bytes.starts_with(b"PROXY TCP4 "));
        assert!(bytes.ends_with(b"\r\n"));
        let (parsed, consumed) = parse(&bytes).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn encode_v1_tcp6_round_trips() {
        let header = ProxyHeader::Tcp {
            src: "[2001:db8::1]:1234".parse().unwrap(),
            dst: "[2001:db8::2]:443".parse().unwrap(),
        };
        let bytes = encode(&header, WireVersion::V1);
        assert!(bytes.starts_with(b"PROXY TCP6 "));
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn encode_v2_ipv4_round_trips() {
        let header = ProxyHeader::Tcp {
            src: "192.168.1.1:8080".parse().unwrap(),
            dst: "192.168.1.2:443".parse().unwrap(),
        };
        let bytes = encode(&header, WireVersion::V2);
        assert!(bytes.starts_with(V2_SIGNATURE));
        let (parsed, consumed) = parse(&bytes).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn encode_v2_ipv6_round_trips() {
        let header = ProxyHeader::Tcp {
            src: "[fe80::1]:1234".parse().unwrap(),
            dst: "[fe80::2]:443".parse().unwrap(),
        };
        let bytes = encode(&header, WireVersion::V2);
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn encode_v2_local_round_trips() {
        let bytes = encode(&ProxyHeader::Local, WireVersion::V2);
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, ProxyHeader::Local);
    }

    #[test]
    fn encode_v2_unknown_round_trips() {
        let bytes = encode(&ProxyHeader::Unknown, WireVersion::V2);
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, ProxyHeader::Unknown);
    }

    #[test]
    fn encode_v1_mixed_family_falls_back_to_unknown() {
        // v1 doesn't carry mixed-family pairs; encoder emits UNKNOWN
        // per the RFC's documented escape hatch.
        let header = ProxyHeader::Tcp {
            src: "10.0.0.1:4000".parse().unwrap(),
            dst: "[::1]:443".parse().unwrap(),
        };
        let bytes = encode(&header, WireVersion::V1);
        assert_eq!(bytes, b"PROXY UNKNOWN\r\n");
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, ProxyHeader::Unknown);
    }

    #[cfg(feature = "proxy_protocol-tokio-runtime")]
    #[proxima::test]
    async fn write_header_tokio_emits_bytes_that_parse_back() {
        let header = ProxyHeader::Tcp {
            src: "1.2.3.4:5678".parse().unwrap(),
            dst: "5.6.7.8:443".parse().unwrap(),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_header_tokio(&mut buf, &header, WireVersion::V2)
            .await
            .unwrap();
        let (parsed, _) = parse(&buf).unwrap();
        assert_eq!(parsed, header);
    }

    #[proxima::test]
    async fn read_header_v1_then_returns_leftover_application_bytes() {
        let payload = b"PROXY TCP4 10.0.0.1 10.0.0.2 4000 443\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut cursor = futures::io::Cursor::new(payload.to_vec());
        let (header, leftover) = read_header(&mut cursor).await.expect("ok");
        let ProxyHeader::Tcp { src, .. } = header else {
            panic!();
        };
        assert_eq!(src.port(), 4000);
        assert!(leftover.starts_with(b"GET / HTTP/1.1\r\n"));
    }

    #[cfg(feature = "proxy_protocol-tokio-runtime")]
    #[proxima::test]
    async fn read_header_tokio_v1_round_trips() {
        let payload = b"PROXY TCP4 10.0.0.1 10.0.0.2 4000 443\r\nGET / HTTP/1.1\r\n".to_vec();
        let mut cursor = std::io::Cursor::new(payload);
        let (header, leftover) = read_header_tokio(&mut cursor).await.unwrap();
        let ProxyHeader::Tcp { src, .. } = header else {
            panic!();
        };
        assert_eq!(src.port(), 4000);
        assert!(leftover.starts_with(b"GET / HTTP/1.1\r\n"));
    }

    #[proxima::test]
    async fn read_header_v2_local_returns_no_leftover_when_aligned() {
        let mut buf = V2_SIGNATURE.to_vec();
        buf.push(0x20);
        buf.push(0x00);
        buf.extend_from_slice(&0u16.to_be_bytes());
        let mut cursor = futures::io::Cursor::new(buf);
        let (header, leftover) = read_header(&mut cursor).await.expect("ok");
        assert_eq!(header, ProxyHeader::Local);
        assert!(leftover.is_empty());
    }
}
