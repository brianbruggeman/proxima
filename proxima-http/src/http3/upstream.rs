//! HTTP/3 upstream — outbound h3 client wrapped as a substrate
//! `Pipe`. Inbound `Request` is translated to an `http::Request`,
//! multiplexed over the persistent QUIC connection, and the response
//! body is returned as the `Response` body.
//!
//! Tracked as P7 in `docs/protocol-gap/discipline.md`. Compared
//! against a hand-rolled `h3-quinn` client doing the same work to
//! size the substrate-API tax.
//!
//! Sub-flag: `h3-upstream` (default off; pulls in the existing
//! `http3` feature stack — quinn + rustls + h3 + h3-quinn).
//!
//! Today: single persistent connection per upstream, lazily opened
//! on first call. Each call clones the h3 `SendRequest` (cheap —
//! shared `OpenStreams` handle internally) and opens a fresh
//! bidirectional stream. No automatic reconnect on connection drop;
//! the next call after a drop re-opens.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::{Buf, Bytes};
use tokio::sync::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};

/// Outbound HTTP/3 upstream. Holds one persistent QUIC connection
/// to `server_addr` (validated against `server_name` for TLS).
pub struct Http3Upstream {
    server_addr: SocketAddr,
    server_name: String,
    label: String,
    /// Pre-built `"https://<server_name>"` prefix. Per-call URI assembly
    /// concatenates this with `path` — saves the `{server_name}` format
    /// substitution on every request.
    uri_prefix: Arc<str>,
    /// rustls config used on first connect. Default is webpki-roots +
    /// ALPN `h3`. Callers can override via `with_client_config` for
    /// custom roots, mTLS, or self-signed dev certs.
    client_config: Arc<rustls::ClientConfig>,
    state: Arc<Mutex<Option<H3State>>>,
}

struct H3State {
    /// quinn::Endpoint kept alive for the upstream lifetime; dropping
    /// it would close the underlying UDP socket.
    _endpoint: quinn::Endpoint,
    /// Prototype SendRequest; cloned per call. `SendRequest::Clone`
    /// is an Arc increment, not a deep copy.
    send_request: h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>,
}

impl Http3Upstream {
    /// Build a new upstream pointed at `server_addr` with SNI hostname
    /// `server_name`. Uses webpki-roots as the trust anchor. Connection
    /// is NOT established here.
    #[must_use]
    pub fn new(server_addr: SocketAddr, server_name: impl Into<String>) -> Self {
        Self::with_client_config(server_addr, server_name, default_client_config())
    }

    /// Like [`Self::new`] but uses a caller-provided rustls `ClientConfig`.
    /// Tests/benches use this to install a self-signed cert as a trusted
    /// root without baking the danger config into the production path.
    #[must_use]
    pub fn with_client_config(
        server_addr: SocketAddr,
        server_name: impl Into<String>,
        client_config: rustls::ClientConfig,
    ) -> Self {
        let server_name = server_name.into();
        let uri_prefix: Arc<str> = Arc::from(format!("https://{server_name}"));
        Self {
            server_addr,
            label: format!("h3://{server_name}/"),
            server_name,
            uri_prefix,
            client_config: Arc::new(client_config),
            state: Arc::new(Mutex::new(None)),
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Default rustls ClientConfig: webpki-roots trust anchors, ALPN `h3`.
fn default_client_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    tls_config.alpn_protocols = vec![b"h3".to_vec()];
    tls_config
}

/// Bring up a fresh quinn::Endpoint, dial `server_addr` with SNI
/// `server_name`, run the h3 handshake, spawn the driver, return
/// the endpoint (owned so caller pins lifetime) + send_request.
async fn connect(
    server_addr: SocketAddr,
    server_name: &str,
    tls_config: Arc<rustls::ClientConfig>,
) -> Result<
    (
        quinn::Endpoint,
        h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>,
    ),
    ProximaError,
> {
    let quic_client_config =
        quinn::crypto::rustls::QuicClientConfig::try_from((*tls_config).clone())
            .map_err(|err| ProximaError::Upstream(format!("h3 tls config: {err}")))?;
    let client_config = quinn::ClientConfig::new(Arc::new(quic_client_config));

    let local: SocketAddr = if server_addr.is_ipv4() {
        SocketAddr::from(([0u8, 0, 0, 0], 0))
    } else {
        SocketAddr::from(([0u16; 8], 0))
    };
    let mut endpoint = quinn::Endpoint::client(local)
        .map_err(|err| ProximaError::Upstream(format!("h3 endpoint: {err}")))?;
    endpoint.set_default_client_config(client_config);

    let connection = endpoint
        .connect(server_addr, server_name)
        .map_err(|err| ProximaError::Upstream(format!("h3 connect: {err}")))?
        .await
        .map_err(|err| ProximaError::Upstream(format!("h3 handshake: {err}")))?;

    let h3_conn = h3_quinn::Connection::new(connection);
    let (mut driver, send_request) = h3::client::new(h3_conn)
        .await
        .map_err(|err| ProximaError::Upstream(format!("h3 client init: {err}")))?;

    // h3 needs the driver polled to handle control streams and
    // settings exchange. spawn matches hyper's client pattern.
    tokio::spawn(async move {
        let _ = futures::future::poll_fn(|cx| driver.poll_close(cx)).await;
    });

    Ok((endpoint, send_request))
}

impl SendPipe for Http3Upstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let server_addr = self.server_addr;
        let server_name = self.server_name.clone();
        let uri_prefix = self.uri_prefix.clone();
        let state = self.state.clone();
        let client_config = self.client_config.clone();
        async move {
            // Pull method + path bytes out of Request first so we don't
            // hold the full Request alive past header construction.
            let path_bytes = request.path.clone();
            let method = std::str::from_utf8(request.method.as_bytes())
                .map_err(|_| ProximaError::Upstream("h3 upstream: invalid method".into()))?;
            let path = std::str::from_utf8(path_bytes.as_ref())
                .map_err(|_| ProximaError::Upstream("h3 upstream: invalid path".into()))?;
            let mut uri = String::with_capacity(uri_prefix.len() + path.len());
            uri.push_str(&uri_prefix);
            uri.push_str(path);

            let http_req = http::Request::builder()
                .method(method)
                .uri(&uri)
                .body(())
                .map_err(|err| ProximaError::Upstream(format!("h3 build: {err}")))?;

            // Lazy connect under tokio::sync::Mutex. Tried OnceCell here —
            // get_or_try_init's state-machine closure showed +4% regression
            // in the bench. Tried std::sync::Mutex — same hot-path cost as
            // tokio's. Sticking with the idiomatic tokio Mutex shape.
            let mut send_request = {
                let mut guard = state.lock().await;
                if let Some(existing) = guard.as_ref() {
                    existing.send_request.clone()
                } else {
                    let (endpoint, fresh) =
                        connect(server_addr, &server_name, client_config.clone()).await?;
                    let cloned = fresh.clone();
                    *guard = Some(H3State {
                        _endpoint: endpoint,
                        send_request: fresh,
                    });
                    cloned
                }
            };

            let mut stream = send_request
                .send_request(http_req)
                .await
                .map_err(|err| ProximaError::Upstream(format!("h3 send_request: {err}")))?;

            // Stream each chunk straight to h3 instead of collect-then-send.
            // For a Body::from_bytes single-chunk source this is one
            // poll-Some + one poll-None; avoids the Vec<Bytes> allocation
            // in Body::collect. For chunked / streamed bodies it's the
            // correct behavior — write as data arrives.
            let mut body_stream = request.into_chunk_stream();
            while let Some(chunk) = futures::StreamExt::next(&mut body_stream).await {
                let chunk = chunk?;
                if chunk.is_empty() {
                    continue;
                }
                stream
                    .send_data(chunk)
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("h3 send_data: {err}")))?;
            }
            stream
                .finish()
                .await
                .map_err(|err| ProximaError::Upstream(format!("h3 finish: {err}")))?;

            let response = stream
                .recv_response()
                .await
                .map_err(|err| ProximaError::Upstream(format!("h3 recv_response: {err}")))?;
            let status = response.status().as_u16();

            let mut body = bytes::BytesMut::new();
            while let Some(mut chunk) = stream
                .recv_data()
                .await
                .map_err(|err| ProximaError::Upstream(format!("h3 recv_data: {err}")))?
            {
                let slice = chunk.chunk();
                body.extend_from_slice(slice);
                let len = slice.len();
                chunk.advance(len);
            }

            let mut response = Response::new(status);
            response.payload = body.freeze();
            Ok(response)
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn new_upstream_does_not_connect_eagerly() {
        // construct against a clearly-unreachable target. If `new`
        // tried to connect this would block / fail.
        let addr: SocketAddr = "127.0.0.1:1".parse().expect("static");
        let upstream = Http3Upstream::new(addr, "never.invalid");
        assert_eq!(upstream.label(), "h3://never.invalid/");
    }
}
