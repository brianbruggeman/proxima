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
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::io::{AsyncReadExt, AsyncWriteExt};
use proxima_primitives::stream::{StreamUpstream, StreamUpstreamExt};

type TunnelFuture<C> = Pin<Box<dyn std::future::Future<Output = io::Result<C>> + Send>>;

/// Scratch read size for the `CONNECT` response head — one read almost
/// always covers `HTTP/1.1 200 Connection established\r\n\r\n`.
const CONNECT_READ_CHUNK: usize = 1024;

/// A `StreamUpstream` that reaches its target through an HTTP `CONNECT`
/// forward proxy. `proxy` dials the proxy itself; `target_host`/
/// `target_port` are the origin the proxy is asked to tunnel to.
pub struct ConnectTunneledUpstream<U: StreamUpstream> {
    proxy: Arc<U>,
    target_host: String,
    target_port: u16,
    in_flight: Mutex<Option<TunnelFuture<U::Conn>>>,
}

impl<U: StreamUpstream> ConnectTunneledUpstream<U> {
    /// `proxy` is the upstream that connects to the forward proxy;
    /// `target_host:target_port` is the origin to tunnel to.
    pub fn new(proxy: U, target_host: impl Into<String>, target_port: u16) -> Self {
        Self {
            proxy: Arc::new(proxy),
            target_host: target_host.into(),
            target_port,
            in_flight: Mutex::new(None),
        }
    }
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
    let code = status_line
        .split(|byte| *byte == b' ')
        .nth(1)
        .and_then(|field| std::str::from_utf8(field).ok())
        .and_then(|text| text.parse::<u16>().ok())
        .ok_or_else(|| io::Error::other("proxy CONNECT response missing status code"))?;
    Ok(Some((code, headers_end)))
}

impl<U: StreamUpstream> StreamUpstream for ConnectTunneledUpstream<U> {
    type Conn = U::Conn;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other(
                "ConnectTunneledUpstream: lock poisoned",
            )));
        };
        let proxy = Arc::clone(&self.proxy);
        let host = self.target_host.clone();
        let port = self.target_port;
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let mut conn = proxy.connect().await?;
                let request =
                    format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n");
                conn.write_all(request.as_bytes()).await?;
                conn.flush().await?;

                let mut buffer = Vec::with_capacity(CONNECT_READ_CHUNK);
                let mut scratch = [0_u8; CONNECT_READ_CHUNK];
                loop {
                    let read = conn.read(&mut scratch).await?;
                    if read == 0 {
                        return Err(io::Error::other("proxy closed before CONNECT response"));
                    }
                    buffer.extend_from_slice(&scratch[..read]);
                    if let Some((status, headers_end)) = parse_connect_response(&buffer)? {
                        if !(200..300).contains(&status) {
                            return Err(io::Error::other(format!(
                                "proxy CONNECT to {host}:{port} returned {status}"
                            )));
                        }
                        // per RFC 9110 §9.3.6 the tunnel is transparent only
                        // after a 2xx, and we have not written tunnel data yet,
                        // so a well-behaved proxy sends nothing past the head.
                        if headers_end != buffer.len() {
                            return Err(io::Error::other(
                                "proxy sent data before tunnel was established",
                            ));
                        }
                        return Ok(conn);
                    }
                }
            })
        });
        match future.as_mut().poll(cx) {
            Poll::Ready(result) => {
                *slot = None;
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_connect_response_reads_200() {
        let head = b"HTTP/1.1 200 Connection established\r\n\r\n";
        let parsed = parse_connect_response(head).expect("parse");
        assert_eq!(parsed, Some((200, head.len())));
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
