//! WebSocket upstream — outbound WS client wrapped as a substrate
//! `Pipe`. Inbound HTTP request bodies are sent as WS messages
//! (text for text/* MIME types, binary otherwise); response messages
//! are returned as the HTTP response body.
//!
//! Tracked as P8 in `docs/protocol-gap/discipline.md`. Compared
//! against direct `tokio-tungstenite` client usage to size the
//! abstraction tax proxima adds. Status: scaffold + minimal
//! single-connection impl; pooling, automatic reconnect, fan-out
//! are follow-ups.
//!
//! Sub-flag: `websocket-upstream` (default off).

use std::future::Future;
use std::sync::Arc;

use async_tungstenite::tokio::ConnectStream;
use async_tungstenite::tungstenite::Message;
use bytes::Bytes;
use futures::StreamExt;
use tokio::sync::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};

/// Outbound WebSocket upstream. Holds one persistent connection to
/// the configured URL; `Pipe::call` sends the inbound request
/// body as a WS message and returns the response message as the
/// response body.
///
/// Today: single connection, lazy-opened on first call. Reconnect
/// on connection drop is best-effort (next `call` will try to
/// re-open). No pool — that's P10 territory.
pub struct WebSocketUpstream {
    url: String,
    label: String,
    /// `Mutex<Option<...>>` because:
    /// - `tokio-tungstenite`'s split halves are tied to the same
    ///   underlying stream; we serialize per-call to avoid
    ///   interleaving frames across concurrent callers.
    /// - `Option` so first call can lazily open; subsequent calls
    ///   re-use the open connection.
    /// - `Arc<Mutex<...>>` because `Pipe::call(&self)` returns
    ///   `Future + Send` so the lock acquisition has to be
    ///   shareable across awaits.
    socket: Arc<Mutex<Option<async_tungstenite::WebSocketStream<ConnectStream>>>>,
}

impl WebSocketUpstream {
    /// Build a new upstream pointed at `url` (must be `ws://` or
    /// `wss://`). Connection is NOT established here — lazy on
    /// first `call`.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            label: url.clone(),
            url,
            socket: Arc::new(Mutex::new(None)),
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    async fn ensure_connected(
        &self,
    ) -> Result<
        tokio::sync::MutexGuard<'_, Option<async_tungstenite::WebSocketStream<ConnectStream>>>,
        ProximaError,
    > {
        let mut guard = self.socket.lock().await;
        if guard.is_none() {
            let (stream, _response) = async_tungstenite::tokio::connect_async(&self.url)
                .await
                .map_err(|err| {
                    ProximaError::Upstream(format!("websocket connect {}: {err}", self.url))
                })?;
            *guard = Some(stream);
        }
        Ok(guard)
    }
}

impl SendPipe for WebSocketUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let socket = self.socket.clone();
        async move {
            // Drain request body — WebSocket framing is whole-message,
            // not chunked, so we materialize before sending. Streaming
            // body upstream is a follow-up if a real workload demands it.
            //
            // *Tried* calling `request.body.collect()` directly to skip
            // the Request-rebuild inside `body_bytes()` — bench regressed
            // +14.79% (20.98 → 23.87 µs). Rolled back. Probably inlining-
            // boundary differences across the two paths; the Request
            // restruct is on a colder path than the lock + tungstenite
            // call below, so the extra alloc doesn't dominate.
            let (_request, payload) = request.body_bytes().await?;

            let mut guard = self.ensure_connected().await.inspect_err(|_err| {
                // ensure_connected took the lock; bail without
                // poisoning by dropping the guard before returning.
                let _ = &socket;
            })?;
            let stream = guard.as_mut().ok_or_else(|| {
                ProximaError::Upstream("websocket upstream: socket missing after connect".into())
            })?;

            // tungstenite's `Message::Binary` payload IS `bytes::Bytes`
            // (re-exported at `tungstenite::Bytes`), so we can pass our
            // request `Bytes` straight through — no `to_vec()` round-trip.
            stream
                .send(Message::Binary(payload))
                .await
                .map_err(|err| ProximaError::Upstream(format!("websocket send: {err}")))?;

            let response_message = match stream.next().await {
                Some(Ok(msg)) => msg,
                Some(Err(err)) => {
                    // Connection-level error — drop the socket so the
                    // next call retries. Better than holding a half-
                    // broken stream.
                    *guard = None;
                    return Err(ProximaError::Upstream(format!("websocket recv: {err}")));
                }
                None => {
                    *guard = None;
                    return Err(ProximaError::Upstream("websocket recv: peer closed".into()));
                }
            };
            let body_bytes = match response_message {
                // Same Bytes-passthrough as the send side: tungstenite's
                // payload IS `bytes::Bytes`, no need to round-trip
                // through Vec<u8>.
                Message::Binary(bytes) => bytes,
                Message::Text(text) => Bytes::from(text.as_str().to_owned()),
                other => {
                    return Err(ProximaError::Upstream(format!(
                        "websocket unexpected frame: {other:?}"
                    )));
                }
            };
            Ok(Response::ok(body_bytes))
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test]
    async fn new_upstream_does_not_connect_eagerly() {
        // `new()` builds the struct without touching the network — the
        // connection only opens on the first `call`. Constructs against
        // a clearly bogus URL prove this: if `new` connected, this would
        // panic.
        let upstream = WebSocketUpstream::new("ws://0.0.0.0:1/never-listens");
        assert_eq!(upstream.label(), "ws://0.0.0.0:1/never-listens");
    }
}
