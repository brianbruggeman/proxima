//! `.websocket(handler)` — wires a WebSocket (RFC 6455) handler into h1's
//! EXISTING connection-upgrade seam
//! (`proxima_primitives::pipe::upgrade::UpgradeHandler`, the same primitive
//! `tests/e2e/listener_upgrade.rs`'s 101-Switching-Protocols test exercises),
//! rather than minting a new peer `AnyProtocol` candidate. A WebSocket
//! handshake is an ordinary HTTP/1.1 request (`GET` plus `Connection:
//! Upgrade`, `Upgrade: websocket`, and `Sec-WebSocket-Key` headers) that
//! only diverges from a normal request AFTER the 101 response — exactly the
//! shape the upgrade seam already exists for; a bespoke `AnyProtocol`
//! candidate would have to reimplement H1 request parsing just to notice
//! the Upgrade header.
//!
//! [`WebSocketUpgradePipe`] wraps the listener's ordinary dispatch pipe:
//! every non-upgrade request passes through unchanged; a request carrying a
//! valid WebSocket handshake gets a 101 response whose `UpgradeHandler`
//! hands the raw post-handshake socket to the caller-supplied
//! [`WebSocketHandler`] — the same "byte-stream, same as any other
//! `StreamListener`" contract `proxima_http::websocket`'s full-connection
//! listener already documents. Frame parsing is the CALLER's business
//! (`proxima_protocols::websocket_frame`, behind the `websocket-frame`
//! feature) — this module only performs the HTTP-level handshake.

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;

use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeFuture, UpgradeHandler};

use crate::error::ProximaError;

/// A reusable (multi-connection) post-handshake handler — unlike
/// [`UpgradeHandler`] itself (a one-shot `FnOnce` per connection), a
/// `WebSocketHandler` is called once PER accepted connection, so it must be
/// callable more than once over the listener's lifetime.
pub type WebSocketHandler =
    Arc<dyn Fn(HijackedSocket) -> UpgradeFuture + Send + Sync>;

/// RFC 6455 §1.3 handshake check: `Connection: Upgrade` + `Upgrade:
/// websocket` (case-insensitive per RFC 7230 §6.7) + a present
/// `Sec-WebSocket-Key`. Returns the key so the caller doesn't re-fetch it.
fn websocket_handshake_key(request: &Request<Bytes>) -> Option<&str> {
    let connection = request.metadata.get_str("connection")?;
    let upgrade = request.metadata.get_str("upgrade")?;
    if !connection.to_ascii_lowercase().contains("upgrade") {
        return None;
    }
    if !upgrade.eq_ignore_ascii_case("websocket") {
        return None;
    }
    request.metadata.get_str("sec-websocket-key")
}

/// Wraps `inner` (the listener's ordinary dispatch pipe): a request that
/// passes [`websocket_handshake_key`] gets a 101 response carrying an
/// [`UpgradeHandler`] that calls `handler` with the hijacked socket; every
/// other request dispatches through `inner` unchanged.
struct WebSocketUpgradePipe {
    inner: PipeHandle,
    handler: WebSocketHandler,
}

impl SendPipe for WebSocketUpgradePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let handler = self.handler.clone();
        let inner = self.inner.clone();
        async move {
            let Some(client_key) = websocket_handshake_key(&request) else {
                return inner.call_dyn(request).await;
            };
            let accept = proxima_protocols::websocket_handshake::compute_accept_key(client_key);
            let upgrade = UpgradeHandler::new(move |socket: HijackedSocket| (handler)(socket));
            Ok(Response::new(101)
                .with_header("connection", "Upgrade")
                .with_header("upgrade", "websocket")
                .with_header("sec-websocket-accept", accept)
                .with_upgrade(upgrade))
        }
    }
}

/// Wrap `dispatch` so an incoming WebSocket handshake routes to `handler`
/// instead of the ordinary dispatch pipe — the one call site
/// `ListenerBuilder::serve` makes when `.websocket(handler)` was called.
pub(crate) fn wrap_dispatch(dispatch: PipeHandle, handler: WebSocketHandler) -> PipeHandle {
    proxima_primitives::pipe::handler::into_handle(WebSocketUpgradePipe {
        inner: dispatch,
        handler,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::method::Method;
    use proxima_primitives::pipe::request::RequestContext;

    fn upgrade_request(connection: &str, upgrade: &str, key: Option<&str>) -> Request<Bytes> {
        let mut metadata = HeaderList::new();
        metadata.insert("connection", connection);
        metadata.insert("upgrade", upgrade);
        if let Some(key) = key {
            metadata.insert("sec-websocket-key", key);
        }
        Request {
            method: Method::from_bytes(b"GET"),
            path: Bytes::from_static(b"/ws"),
            query: HeaderList::new(),
            metadata,
            payload: Bytes::new(),
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[test]
    fn recognizes_a_well_formed_handshake() {
        let request = upgrade_request("Upgrade", "websocket", Some("dGhlIHNhbXBsZSBub25jZQ=="));
        assert_eq!(
            websocket_handshake_key(&request),
            Some("dGhlIHNhbXBsZSBub25jZQ==")
        );
    }

    #[test]
    fn rejects_missing_upgrade_key() {
        let request = upgrade_request("Upgrade", "websocket", None);
        assert_eq!(websocket_handshake_key(&request), None);
    }

    #[test]
    fn rejects_non_websocket_upgrade_target() {
        let request = upgrade_request("Upgrade", "h2c", Some("key"));
        assert_eq!(websocket_handshake_key(&request), None);
    }

    #[test]
    fn rejects_missing_connection_upgrade() {
        let request = upgrade_request("keep-alive", "websocket", Some("key"));
        assert_eq!(websocket_handshake_key(&request), None);
    }

    #[proxima::test]
    async fn non_upgrade_request_passes_through_to_inner() {
        struct FixedOk;
        impl SendPipe for FixedOk {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
                Ok(Response::ok("plain"))
            }
        }
        let dispatch = proxima_primitives::pipe::handler::into_handle(FixedOk);
        let handler: WebSocketHandler = Arc::new(|_socket| Box::pin(async { Ok(()) }));
        let wrapped = wrap_dispatch(dispatch, handler);

        let plain_request = Request {
            method: Method::from_bytes(b"GET"),
            path: Bytes::from_static(b"/"),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: Bytes::new(),
            stream: None,
            context: RequestContext::default(),
        };
        let response = wrapped.call_dyn(plain_request).await.expect("dispatch");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn handshake_request_gets_a_101_with_upgrade_attached() {
        struct NeverCalled;
        impl SendPipe for NeverCalled {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
                unreachable!("a genuine handshake must never reach the inner dispatch pipe")
            }
        }
        let dispatch = proxima_primitives::pipe::handler::into_handle(NeverCalled);
        let handler: WebSocketHandler = Arc::new(|_socket| Box::pin(async { Ok(()) }));
        let wrapped = wrap_dispatch(dispatch, handler);

        let request = upgrade_request("Upgrade", "websocket", Some("dGhlIHNhbXBsZSBub25jZQ=="));
        let response = wrapped.call_dyn(request).await.expect("dispatch");
        assert_eq!(response.status, 101);
        assert_eq!(
            response.metadata.get_str("sec-websocket-accept"),
            Some("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")
        );
        assert!(response.upgrade.is_some());
    }
}
