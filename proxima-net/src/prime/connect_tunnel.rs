//! Egress-proxy upstream: tunnel a `StreamUpstream` through an HTTP
//! `CONNECT` forward proxy. Wraps any inner upstream that dials the proxy
//! (e.g. [`PrimeTcpUpstream`](super::PrimeTcpUpstream)); on `connect` it
//! opens the proxy socket, issues `CONNECT host:port`, reads the `2xx`,
//! and hands back the now-transparent tunnel as its own connection.
//!
//! The same tunnel carries both schemes: layer
//! [`TlsStreamUpstream`](../../proxima-tls) over it for `https`, or speak
//! plain HTTP/1.1 in origin-form over it for `http`. Because the wrapper
//! sits below the protocol, it needs no change to the h1 client encoder.

use std::io;
use std::net::Ipv6Addr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, Chain, Cursor};
use proxima_primitives::stream::{
    ConnectFuture, PeerInfo, StreamConnection, StreamUpstream, StreamUpstreamExt,
};

/// Scratch read size for the `CONNECT` response head — one read almost
/// always covers `HTTP/1.1 200 Connection established\r\n\r\n`.
const CONNECT_READ_CHUNK: usize = 1024;
const DEFAULT_RESPONSE_HEADER_BYTES: usize = 16 * 1024;

/// A `StreamUpstream` that reaches its target through an HTTP `CONNECT`
/// forward proxy. `proxy` dials the proxy itself; `target_host`/
/// `target_port` are the origin the proxy is asked to tunnel to.
pub struct ConnectTunneledUpstream<U: StreamUpstream> {
    proxy: Arc<U>,
    target_host: String,
    target_port: u16,
    /// maximum response-head bytes, including status line and final CRLFCRLF.
    max_response_header_bytes: usize,
}

impl<U: StreamUpstream> ConnectTunneledUpstream<U> {
    /// `proxy` is the upstream that connects to the forward proxy;
    /// `target_host:target_port` is the origin to tunnel to.
    pub fn new(proxy: U, target_host: impl Into<String>, target_port: u16) -> Self {
        Self {
            proxy: Arc::new(proxy),
            target_host: target_host.into(),
            target_port,
            max_response_header_bytes: DEFAULT_RESPONSE_HEADER_BYTES,
        }
    }

    /// set the response-head bound (16 KiB by default); zero fails before dialing.
    pub fn with_max_response_header_bytes(mut self, maximum: usize) -> Self {
        self.max_response_header_bytes = maximum;
        self
    }
}

/// a tunnel retaining server bytes read alongside the successful CONNECT head.
pub struct ConnectTunnelConnection<C: StreamConnection> {
    inner: Chain<Cursor<Vec<u8>>, C>,
}

impl<C: StreamConnection> AsyncRead for ConnectTunnelConnection<C> {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(context, buffer)
    }
}

impl<C: StreamConnection> AsyncWrite for ConnectTunnelConnection<C> {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(self.get_mut().inner.get_mut().1).poll_write(context, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.get_mut().inner.get_mut().1).poll_flush(context)
    }

    fn poll_close(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(self.get_mut().inner.get_mut().1).poll_close(context)
    }
}

impl<C: StreamConnection> StreamConnection for ConnectTunnelConnection<C> {
    fn peer(&self) -> Option<PeerInfo> {
        self.inner.get_ref().1.peer()
    }
}

fn target_authority(host: &str, port: u16) -> io::Result<String> {
    let invalid = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid CONNECT target authority",
        )
    };
    if host.is_empty() || port == 0 {
        return Err(invalid());
    }
    if host.starts_with('[') {
        let address = host
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .and_then(|value| value.parse::<Ipv6Addr>().ok())
            .ok_or_else(invalid)?;
        return Ok(format!("[{address}]:{port}"));
    }
    if host.contains(':') {
        let address = host.parse::<Ipv6Addr>().map_err(|_| invalid())?;
        return Ok(format!("[{address}]:{port}"));
    }
    let mut bytes = host.as_bytes().iter().copied();
    while let Some(byte) = bytes.next() {
        if byte == b'%' {
            if !bytes.next().is_some_and(|value| value.is_ascii_hexdigit())
                || !bytes.next().is_some_and(|value| value.is_ascii_hexdigit())
            {
                return Err(invalid());
            }
        } else if !byte.is_ascii_alphanumeric() && !b"-._~!$&'()*+,;=".contains(&byte) {
            return Err(invalid());
        }
    }
    Ok(format!("{host}:{port}"))
}

/// Parse the status code out of a `CONNECT` response head
/// (`HTTP/1.1 200 ...`). Returns the code and the byte offset just past
/// the terminating `\r\n\r\n`, or `None` if the head is not yet complete.
fn parse_connect_response(buffer: &[u8]) -> io::Result<Option<(u16, usize)>> {
    let Some(headers_end) = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4)
    else {
        return Ok(None);
    };
    let line_end = buffer
        .windows(2)
        .position(|window| window == b"\r\n")
        .unwrap_or(headers_end);
    let status_line = &buffer[..line_end];
    if status_line.len() < 13
        || !matches!(&status_line[..8], b"HTTP/1.0" | b"HTTP/1.1")
        || status_line[8] != b' '
        || !status_line[9..12].iter().all(u8::is_ascii_digit)
        || status_line[12] != b' '
        || !status_line[13..]
            .iter()
            .all(|byte| *byte == b'\t' || (*byte >= b' ' && *byte != 0x7f))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid proxy CONNECT status line",
        ));
    }
    let code = u16::from(status_line[9] - b'0') * 100
        + u16::from(status_line[10] - b'0') * 10
        + u16::from(status_line[11] - b'0');
    Ok(Some((code, headers_end)))
}

impl<U: StreamUpstream> StreamUpstream for ConnectTunneledUpstream<U> {
    type Conn = ConnectTunnelConnection<U::Conn>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let proxy = Arc::clone(&self.proxy);
        let host = self.target_host.clone();
        let port = self.target_port;
        let maximum = self.max_response_header_bytes;
        Box::pin(async move {
            let authority = target_authority(&host, port)?;
            if maximum == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "CONNECT response-head bound must be positive",
                ));
            }
            let mut conn = proxy.connect().await?;
            let request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n");
            conn.write_all(request.as_bytes()).await?;
            conn.flush().await?;

            let mut buffer = Vec::with_capacity(maximum);
            let mut scratch = [0_u8; CONNECT_READ_CHUNK];
            loop {
                let remaining = maximum - buffer.len();
                if remaining == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "proxy CONNECT response head exceeds configured bound",
                    ));
                }
                let capacity = remaining.min(scratch.len());
                let read = conn.read(&mut scratch[..capacity]).await?;
                if read == 0 {
                    return Err(io::Error::other("proxy closed before CONNECT response"));
                }
                let search_start = buffer.len().saturating_sub(3);
                buffer.extend_from_slice(&scratch[..read]);
                if !buffer[search_start..]
                    .windows(4)
                    .any(|window| window == b"\r\n\r\n")
                {
                    continue;
                }
                if let Some((status, headers_end)) = parse_connect_response(&buffer)? {
                    if !(200..300).contains(&status) {
                        return Err(io::Error::other(format!(
                            "proxy CONNECT to {host}:{port} returned {status}"
                        )));
                    }
                    let prefix = buffer.split_off(headers_end);
                    return Ok(ConnectTunnelConnection {
                        inner: Cursor::new(prefix).chain(conn),
                    });
                }
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures::io::{AsyncRead, AsyncWrite};
    use proxima_primitives::stream::{PeerInfo, StreamConnection};
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    struct ScriptedProxy {
        fragments: Vec<Vec<u8>>,
        dials: Arc<AtomicUsize>,
        bytes_read: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl ScriptedProxy {
        fn new(fragments: Vec<Vec<u8>>) -> Self {
            Self {
                fragments,
                dials: Arc::new(AtomicUsize::new(0)),
                bytes_read: Arc::new(AtomicUsize::new(0)),
                drops: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl StreamUpstream for ScriptedProxy {
        type Conn = ScriptedConnection;

        fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
            Box::pin(async move {
                self.dials.fetch_add(1, Ordering::SeqCst);
                Ok(ScriptedConnection {
                    fragments: self.fragments.clone().into(),
                    offset: 0,
                    written: Vec::new(),
                    bytes_read: Arc::clone(&self.bytes_read),
                    drops: Arc::clone(&self.drops),
                    flushed: false,
                    closed: false,
                })
            })
        }
    }

    struct ScriptedConnection {
        fragments: VecDeque<Vec<u8>>,
        offset: usize,
        written: Vec<u8>,
        bytes_read: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        flushed: bool,
        closed: bool,
    }

    impl StreamConnection for ScriptedConnection {
        fn peer(&self) -> Option<PeerInfo> {
            Some(PeerInfo::Tcp(
                "127.0.0.1:3128".parse().expect("scripted proxy address"),
            ))
        }
    }

    impl Drop for ScriptedConnection {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl AsyncRead for ScriptedConnection {
        fn poll_read(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buffer: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let this = self.get_mut();
            let Some(fragment) = this.fragments.front() else {
                return Poll::Ready(Ok(0));
            };
            let count = buffer.len().min(fragment.len() - this.offset);
            buffer[..count].copy_from_slice(&fragment[this.offset..this.offset + count]);
            this.offset += count;
            if this.offset == fragment.len() {
                this.fragments.pop_front();
                this.offset = 0;
            }
            this.bytes_read.fetch_add(count, Ordering::SeqCst);
            Poll::Ready(Ok(count))
        }
    }

    impl AsyncWrite for ScriptedConnection {
        fn poll_write(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.get_mut().written.extend_from_slice(buffer);
            Poll::Ready(Ok(buffer.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.get_mut().flushed = true;
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.get_mut().closed = true;
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn connect_tunnel_rejects_invalid_authority_before_dial() {
        for host in [
            "",
            "example.com\r\nX-Injected: yes",
            "a\0b",
            "bad host",
            "bad\thost",
            "user@example.com",
            "example.com/path",
            "example.com?query",
            "example.com#fragment",
            "example.com:443",
            "[::1]:443",
            "[broken]",
            "broken]",
            "bad%",
            "bad%2",
            "bad%gg",
            "nonascii-é",
            "example.com\\path",
            "bad\u{7f}host",
        ] {
            let proxy = ScriptedProxy::new(Vec::new());
            let dials = Arc::clone(&proxy.dials);
            let upstream = ConnectTunneledUpstream::new(proxy, host, 443);
            let error = block_on(upstream.connect())
                .err()
                .expect("reject invalid target");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{host:?}");
            assert_eq!(dials.load(Ordering::SeqCst), 0, "{host:?}");
        }
        for (port, maximum) in [(0, 1024), (443, 0)] {
            let proxy = ScriptedProxy::new(Vec::new());
            let dials = Arc::clone(&proxy.dials);
            let upstream = ConnectTunneledUpstream::new(proxy, "example.com", port)
                .with_max_response_header_bytes(maximum);
            assert_eq!(
                block_on(upstream.connect())
                    .err()
                    .expect("reject invalid configuration")
                    .kind(),
                io::ErrorKind::InvalidInput
            );
            assert_eq!(dials.load(Ordering::SeqCst), 0);
        }
        for host in [
            "service_name",
            "example.com.",
            "a!$&'()*+,;=~b",
            "percent%41name",
        ] {
            assert_eq!(
                target_authority(host, 443).expect("valid reg-name"),
                format!("{host}:443")
            );
        }
        for host in ["::1", "[::1]"] {
            assert_eq!(
                target_authority(host, 443).expect("valid IPv6"),
                "[::1]:443"
            );
        }
    }

    #[test]
    fn connect_tunnel_bounds_fragmented_proxy_headers() {
        for maximum in [31, DEFAULT_RESPONSE_HEADER_BYTES] {
            let mut response = b"HTTP/1.1 200 OK\r\nX-Long: ".to_vec();
            response.resize(maximum + 20, b'x');
            response.extend_from_slice(b"\r\n\r\n");
            for fragment_size in [1, 7, response.len()] {
                let fragments = response.chunks(fragment_size).map(<[u8]>::to_vec).collect();
                let proxy = ScriptedProxy::new(fragments);
                let bytes_read = Arc::clone(&proxy.bytes_read);
                let upstream = ConnectTunneledUpstream::new(proxy, "example.com", 443)
                    .with_max_response_header_bytes(maximum);
                let error = block_on(upstream.connect())
                    .err()
                    .expect("reject over-limit header");
                assert_eq!(error.kind(), io::ErrorKind::InvalidData);
                assert_eq!(bytes_read.load(Ordering::SeqCst), maximum);
            }
        }
        let head = b"HTTP/1.1 200 OK\r\n\r\n";
        for fragment_size in [1, 3] {
            let proxy =
                ScriptedProxy::new(head.chunks(fragment_size).map(<[u8]>::to_vec).collect());
            let bytes_read = Arc::clone(&proxy.bytes_read);
            let upstream = ConnectTunneledUpstream::new(proxy, "example.com", 443)
                .with_max_response_header_bytes(head.len());
            let _connection = block_on(upstream.connect()).expect("complete header at exact limit");
            assert_eq!(bytes_read.load(Ordering::SeqCst), head.len());
        }
    }

    #[test]
    fn connect_tunnel_preserves_fragmented_response_and_payload() {
        for (host, expected_request) in [
            (
                "example.com",
                b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n".as_slice(),
            ),
            (
                "::1",
                b"CONNECT [::1]:443 HTTP/1.1\r\nHost: [::1]:443\r\n\r\n".as_slice(),
            ),
        ] {
            let proxy = ScriptedProxy::new(vec![
                b"HTTP/1.1 200 Connection ".to_vec(),
                b"established\r\nX-Proxy: scripted\r\n\r".to_vec(),
                b"\n".to_vec(),
                b"tunnel-response".to_vec(),
            ]);
            let upstream = ConnectTunneledUpstream::new(proxy, host, 443);
            let mut connection = block_on(upstream.connect()).expect("establish fragmented tunnel");
            assert_eq!(connection.inner.get_ref().1.written, expected_request);
            block_on(connection.write_all(b"tunnel-request")).expect("write tunnel payload");
            assert_eq!(
                &connection.inner.get_ref().1.written[expected_request.len()..],
                b"tunnel-request"
            );
            let mut response = [0; 15];
            block_on(connection.read_exact(&mut response)).expect("read untouched tunnel payload");
            assert_eq!(&response, b"tunnel-response");
        }
        let proxy = ScriptedProxy::new(vec![
            b"HTTP/1.1 200 OK\r\n\r\nbanner-".to_vec(),
            b"tail".to_vec(),
        ]);
        let drops = Arc::clone(&proxy.drops);
        let upstream = ConnectTunneledUpstream::new(proxy, "example.com", 443);
        let mut connection =
            block_on(upstream.connect()).expect("preserve coalesced server banner");
        assert!(
            matches!(connection.peer(), Some(PeerInfo::Tcp(address)) if address == "127.0.0.1:3128".parse().expect("expected proxy peer"))
        );
        let mut response = [0; 11];
        block_on(connection.read_exact(&mut response)).expect("read banner and later tail");
        assert_eq!(&response, b"banner-tail");
        let request_length = connection.inner.get_ref().1.written.len();
        block_on(connection.write_all(b"client-payload")).expect("write after banner");
        assert_eq!(
            &connection.inner.get_ref().1.written[request_length..],
            b"client-payload"
        );
        connection.inner.get_mut().1.flushed = false;
        block_on(connection.flush()).expect("flush underlying stream");
        assert!(connection.inner.get_ref().1.flushed);
        block_on(connection.close()).expect("close underlying write direction");
        assert!(connection.inner.get_ref().1.closed);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(connection);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_connect_response_reads_200() {
        let head = b"HTTP/1.1 200 Connection established\r\n\r\n";
        let parsed = parse_connect_response(head).expect("parse");
        assert_eq!(parsed, Some((200, head.len())));
        let head = b"HTTP/1.0 201 Created\r\n\r\n";
        assert_eq!(
            parse_connect_response(head).expect("HTTP/1.0 success"),
            Some((201, head.len()))
        );
        let head = b"HTTP/1.1 200 \r\n\r\n";
        assert_eq!(
            parse_connect_response(head).expect("empty reason retains separator"),
            Some((200, head.len()))
        );
        for invalid in [
            b"garbage 200 OK\r\n\r\n".as_slice(),
            b"HTTP/2 200 OK\r\n\r\n".as_slice(),
            b"HTTP/1.1 20 OK\r\n\r\n".as_slice(),
            b"HTTP/1.1 2000 OK\r\n\r\n".as_slice(),
            b"HTTP/1.1 +200 OK\r\n\r\n".as_slice(),
            b"HTTP/1.1 2x0 OK\r\n\r\n".as_slice(),
            b"HTTP/1.1 200\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\0bad\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\nbad\r\n\r\n".as_slice(),
            b"HTTP/1.1 200 OK\x7fbad\r\n\r\n".as_slice(),
        ] {
            assert_eq!(
                parse_connect_response(invalid)
                    .expect_err("reject malformed status line")
                    .kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn parse_connect_response_surfaces_403() {
        let head = b"HTTP/1.1 403 Forbidden\r\n\r\n";
        let parsed = parse_connect_response(head).expect("parse");
        assert_eq!(parsed, Some((403, head.len())));
    }

    #[test]
    fn parse_connect_response_partial_head_is_incomplete() {
        let head = b"HTTP/1.1 200 Connection established\r\n";
        assert_eq!(parse_connect_response(head).expect("parse"), None);
    }
}
