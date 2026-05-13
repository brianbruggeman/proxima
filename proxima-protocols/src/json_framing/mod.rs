//! Length-prefixed JSON framing over `StreamUpstream` with reconnect +
//! retry, for sidecars that speak the common
//! `[u32 BE len][payload]` shape. The sans-IO codec (encode/decode of
//! the 4-byte length prefix) lives in [`codec`] — the reconnect +
//! StreamUpstream-binding + tokio Mutex client stays here.
//!
//! Tier: std + tokio (client, gated by `json_framing-std`). The
//! [`codec`] module is no_std + alloc, gated by `json_framing`.

pub mod codec;

#[cfg(feature = "json_framing-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "json_framing-codec-trait")]
pub use codec_trait::{FrameError as JsonFrameError, JsonFrameCodec};

#[cfg(feature = "json_framing-std")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "json_framing-std")]
use std::time::Duration;

#[cfg(feature = "json_framing-std")]
use codec::{HEADER_BYTES, decode_header, encode_header};
#[cfg(feature = "json_framing-std")]
use serde_json::Value;
#[cfg(feature = "json_framing-std")]
use thiserror::Error;
#[cfg(feature = "json_framing-std")]
use tokio::sync::Mutex;
#[cfg(feature = "json_framing-std")]
use tracing::{debug, warn};

#[cfg(feature = "json_framing-std")]
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};

#[cfg(feature = "json_framing-std")]
#[derive(Debug, Error)]
pub enum TransportError {
    #[error("backend: {0}")]
    Backend(String),

    #[error("timeout after {0:?}")]
    Timeout(Duration),

    #[error("retries exhausted after {attempts} attempts: {last_error}")]
    RetriesExhausted { attempts: usize, last_error: String },
}

#[cfg(feature = "json_framing-std")]
pub struct TransportConfig {
    /// max retry attempts before surfacing `RetriesExhausted`. default: 3
    pub max_retries: usize,
    /// initial backoff between retries; doubles each attempt. default: 500ms
    pub retry_backoff: Duration,
    /// per-attempt connect timeout. default: 5s
    pub connect_timeout: Duration,
}

#[cfg(feature = "json_framing-std")]
impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_backoff: Duration::from_millis(500),
            connect_timeout: Duration::from_secs(5),
        }
    }
}

// MAX_FRAME_BYTES + HEADER_BYTES now live in proxima-framing-json-codec.

/// Length-prefixed-JSON client over any `StreamUpstream` backend.
#[cfg(feature = "json_framing-std")]
pub struct LengthPrefixedJsonClient<U: StreamUpstream> {
    upstream: U,
    // WHY Mutex here:
    //   Holds the long-lived `StreamConnection` between requests so
    //   the next call can reuse it without re-connecting (per-process
    //   connection pool of size 1). `&self` API on the framing client
    //   forces interior mutability.
    //
    // WHY NOT removable:
    //   - RefCell: !Send, breaks the framing client's Send bound
    //     (chain dispatch may carry it cross-thread).
    //   - Lock-free queue of connections: would change the pool
    //     semantics from "reuse single connection" to "first available
    //     of N" — different design, not equivalent.
    //   - AtomicPtr<Conn>: connections aren't movable through atomics
    //     soundly.
    //
    // WHY this is right:
    //   One framing client per upstream binding. Lock acquire is
    //   single-poll-per-request, uncontested in steady state. The
    //   lock is held for the entire request lifetime (write request,
    //   read response, return connection to slot) but no other caller
    //   is competing — the slot represents the connection's owner.
    //   No bench needed: contention is bounded by call rate per
    //   upstream binding, and the syscall cost of the connection's
    //   actual I/O dwarfs the lock cost by orders of magnitude.
    conn: Mutex<Option<U::Conn>>,
    config: TransportConfig,
    connect_count: AtomicUsize,
    label: String,
}

#[cfg(feature = "json_framing-std")]
impl<U: StreamUpstream> LengthPrefixedJsonClient<U> {
    pub fn new(upstream: U, config: TransportConfig, label: impl Into<String>) -> Self {
        Self {
            upstream,
            conn: Mutex::new(None),
            config,
            connect_count: AtomicUsize::new(0),
            label: label.into(),
        }
    }

    pub fn connect_count(&self) -> usize {
        self.connect_count.load(Ordering::Relaxed)
    }

    pub async fn ensure_connected(&self) -> Result<(), TransportError> {
        let mut guard = self.conn.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        let conn = proxima_core::time::timeout(self.config.connect_timeout, self.upstream.connect())
            .await
            .map_err(|_| TransportError::Timeout(self.config.connect_timeout))?
            .map_err(|err| TransportError::Backend(format!("connect: {err}")))?;
        self.connect_count.fetch_add(1, Ordering::Relaxed);
        debug!(label = %self.label, "framing client connected");
        *guard = Some(conn);
        Ok(())
    }

    pub async fn disconnect(&self) {
        self.conn.lock().await.take();
    }

    pub async fn send_frame(&self, payload: &[u8]) -> Result<(), TransportError> {
        use futures::io::AsyncWriteExt;
        let mut guard = self.conn.lock().await;
        let conn = guard
            .as_mut()
            .ok_or_else(|| TransportError::Backend("not connected".into()))?;
        let header = encode_header(payload.len())
            .map_err(|err| TransportError::Backend(format!("encode header: {err}")))?;
        conn.write_all(&header)
            .await
            .map_err(|err| TransportError::Backend(format!("write len: {err}")))?;
        conn.write_all(payload)
            .await
            .map_err(|err| TransportError::Backend(format!("write payload: {err}")))?;
        conn.flush()
            .await
            .map_err(|err| TransportError::Backend(format!("flush: {err}")))?;
        Ok(())
    }

    pub async fn recv_frame(&self) -> Result<Vec<u8>, TransportError> {
        use futures::io::AsyncReadExt;
        let mut guard = self.conn.lock().await;
        let conn = guard
            .as_mut()
            .ok_or_else(|| TransportError::Backend("not connected".into()))?;
        let mut len_buf = [0_u8; HEADER_BYTES];
        conn.read_exact(&mut len_buf)
            .await
            .map_err(|err| TransportError::Backend(format!("read len: {err}")))?;
        let len = decode_header(len_buf)
            .map_err(|err| TransportError::Backend(format!("decode header: {err}")))?;
        let mut payload = vec![0_u8; len];
        conn.read_exact(&mut payload)
            .await
            .map_err(|err| TransportError::Backend(format!("read payload: {err}")))?;
        Ok(payload)
    }

    pub async fn send_json(&self, value: &Value) -> Result<(), TransportError> {
        let bytes = serde_json::to_vec(value)
            .map_err(|err| TransportError::Backend(format!("encode json: {err}")))?;
        self.send_frame(&bytes).await
    }

    pub async fn recv_json(&self) -> Result<Value, TransportError> {
        let buf = self.recv_frame().await?;
        serde_json::from_slice(&buf)
            .map_err(|err| TransportError::Backend(format!("invalid json: {err}")))
    }

    pub async fn request_json(&self, request: &Value) -> Result<Value, TransportError> {
        self.send_json(request).await?;
        let response = self.recv_json().await?;
        if response.get("type").and_then(|val| val.as_str()) == Some("Error") {
            let msg = response["message"].as_str().unwrap_or("unknown error");
            return Err(TransportError::Backend(msg.to_string()));
        }
        Ok(response)
    }

    pub async fn request_binary(&self, request: &Value) -> Result<Vec<u8>, TransportError> {
        self.send_json(request).await?;
        let buf = self.recv_frame().await?;
        // daemons may emit a JSON error envelope on a binary endpoint
        // when something goes wrong; sniff the first byte for `{` to
        // detect the error case.
        if !buf.is_empty()
            && buf[0] == b'{'
            && let Ok(value) = serde_json::from_slice::<Value>(&buf)
            && value.get("type").and_then(|val| val.as_str()) == Some("Error")
        {
            let msg = value["message"].as_str().unwrap_or("unknown error");
            return Err(TransportError::Backend(msg.to_string()));
        }
        Ok(buf)
    }

    pub fn is_retryable(err: &TransportError) -> bool {
        matches!(err, TransportError::Backend(_) | TransportError::Timeout(_))
    }

    pub async fn with_retry<F, Fut, T>(&self, mut operation: F) -> Result<T, TransportError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, TransportError>>,
    {
        let max = self.config.max_retries;
        let mut last_error = None;
        for attempt in 0..max {
            if let Err(err) = self.ensure_connected().await {
                if attempt + 1 < max {
                    let backoff = self.config.retry_backoff * (1_u32 << attempt.min(4));
                    proxima_core::time::sleep(backoff).await;
                }
                last_error = Some(err);
                continue;
            }
            match operation().await {
                Ok(result) => return Ok(result),
                Err(err) if Self::is_retryable(&err) => {
                    warn!(label = %self.label, attempt, error = %err, "request failed, reconnecting");
                    self.disconnect().await;
                    if attempt + 1 < max {
                        let backoff = self.config.retry_backoff * (1_u32 << attempt.min(4));
                        proxima_core::time::sleep(backoff).await;
                    }
                    last_error = Some(err);
                }
                Err(err) => return Err(err),
            }
        }
        Err(
            last_error.unwrap_or_else(|| TransportError::RetriesExhausted {
                attempts: max,
                last_error: "no attempts made".into(),
            }),
        )
    }
}

/// Server-side framing decoder for a single `StreamConnection`.
/// Symmetric to `LengthPrefixedJsonClient::recv_frame`/`send_frame`
/// for use inside a `Pipe::call(Request)` body that wants to
/// drive a request/reply loop over the inbound conn.
#[cfg(feature = "json_framing-std")]
pub mod server {
    use super::*;
    use std::io;

    // io::Result preserves ErrorKind so callers can detect EOF /
    // ConnectionReset / BrokenPipe via kind() rather than matching
    // on a stringified message.
    pub async fn read_frame<C: StreamConnection>(conn: &mut C) -> io::Result<Vec<u8>> {
        use futures::io::AsyncReadExt;
        let mut len_buf = [0_u8; HEADER_BYTES];
        conn.read_exact(&mut len_buf).await?;
        let len = decode_header(len_buf)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("{err}")))?;
        let mut payload = vec![0_u8; len];
        conn.read_exact(&mut payload).await?;
        Ok(payload)
    }

    pub async fn write_frame<C: StreamConnection>(conn: &mut C, payload: &[u8]) -> io::Result<()> {
        use futures::io::AsyncWriteExt;
        let header = encode_header(payload.len())
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("{err}")))?;
        conn.write_all(&header).await?;
        conn.write_all(payload).await?;
        conn.flush().await?;
        Ok(())
    }
}

#[cfg(all(test, feature = "json_framing-std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
    use proxima_primitives::stream::{StreamListener, StreamListenerExt};
    use serde_json::json;
    use std::net::{Ipv4Addr, SocketAddr};

    #[proxima::test]
    async fn round_trip_request_json_against_echo_server() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };

        // server: read one frame, echo it back
        tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let frame = server::read_frame(&mut conn).await.expect("read");
            server::write_frame(&mut conn, &frame).await.expect("write");
        });

        let upstream = TokioTcpUpstream::new(local);
        let client = LengthPrefixedJsonClient::new(upstream, TransportConfig::default(), "test");
        client.ensure_connected().await.expect("connect");
        let response = client
            .request_json(&json!({"type": "Echo", "value": 42}))
            .await
            .expect("round trip");
        assert_eq!(response["type"], "Echo");
        assert_eq!(response["value"], 42);
    }

    #[proxima::test]
    async fn request_json_propagates_error_envelope() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };

        tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let _ = server::read_frame(&mut conn).await.expect("read");
            let err_envelope = serde_json::to_vec(&json!({
                "type": "Error",
                "message": "no such slot",
            }))
            .expect("encode");
            server::write_frame(&mut conn, &err_envelope)
                .await
                .expect("write");
        });

        let upstream = TokioTcpUpstream::new(local);
        let client = LengthPrefixedJsonClient::new(upstream, TransportConfig::default(), "test");
        client.ensure_connected().await.expect("connect");
        let outcome = client.request_json(&json!({"type": "Lookup"})).await;
        match outcome {
            Err(TransportError::Backend(msg)) => {
                assert!(msg.contains("no such slot"), "got: {msg}")
            }
            other => panic!("expected Backend error, got {other:?}"),
        }
    }
}
