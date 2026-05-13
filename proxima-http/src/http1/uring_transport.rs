//! io_uring HTTP transport (proxima-h1, `--features io-uring` on Linux).
//!
//! Two halves, both io_uring-specific and shared by the io_uring listener
//! (umbrella `listeners::http_uring`, which reuses [`UringAsyncStream`])
//! and the io_uring outbound client ([`request_via_uring`], routed to by
//! `shared_http` when the per-core worker runs on the `tokio_uring`
//! runtime — `tokio::net::TcpStream` has no reactor there and would stall):
//!
//! - [`UringAsyncStream`] — an `AsyncRead`/`AsyncWrite` adapter over
//!   `Rc<tokio_uring::net::TcpStream>`, owned-buffer io_uring submissions
//!   underneath. `!Send` (holds an `Rc`).
//! - [`request_via_uring`] — per-request connect (no pool yet),
//!   DNS via `tokio::net::lookup_host`, raw hyper http1 handshake (the
//!   per-connection driver is `!Send`), TLS via `tokio_rustls` for https.
//!   Public surface returns a Send future; the `!Send` work runs in a
//!   `spawn_local`'d task and the result returns over a oneshot channel.
#![cfg(all(target_os = "linux", feature = "http1-io-uring"))]

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use crate::http1::hyper_body::StreamingHyperBody;
use proxima_core::ProximaError;

type UringReadFuture = Pin<Box<dyn Future<Output = (io::Result<usize>, Vec<u8>)>>>;
type UringWriteFuture = Pin<Box<dyn Future<Output = (io::Result<()>, Vec<u8>)>>>;

pub struct UringAsyncStream {
    inner: Rc<tokio_uring::net::TcpStream>,
    read_future: Option<UringReadFuture>,
    write_future: Option<UringWriteFuture>,
    write_in_flight: usize,
}

impl UringAsyncStream {
    #[must_use]
    pub fn new(stream: tokio_uring::net::TcpStream) -> Self {
        Self::from_rc(Rc::new(stream))
    }

    /// Build from a pre-existing `Rc<TcpStream>` so the listener can
    /// share the same stream with the io_uring read/write path before
    /// hijacking it (the listener's `serve_connection_uring` holds an
    /// `Rc<TcpStream>` already; on upgrade it converts that into a
    /// `UringAsyncStream` for the upgrade handler).
    #[must_use]
    pub fn from_rc(inner: Rc<tokio_uring::net::TcpStream>) -> Self {
        Self {
            inner,
            read_future: None,
            write_future: None,
            write_in_flight: 0,
        }
    }
}

impl AsyncRead for UringAsyncStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.read_future.is_none() {
            let capacity = buf.remaining();
            if capacity == 0 {
                return Poll::Ready(Ok(()));
            }
            let owned = vec![0_u8; capacity];
            let inner = this.inner.clone();
            let future: UringReadFuture = Box::pin(async move {
                let (result, returned) = inner.read(owned).await;
                (result, returned)
            });
            this.read_future = Some(future);
        }
        // structural: read_future was set above when None.
        #[allow(clippy::expect_used)]
        let future = this
            .read_future
            .as_mut()
            .expect("read_future set when None above");
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready((result, owned)) => {
                this.read_future = None;
                match result {
                    Ok(n) => {
                        buf.put_slice(&owned[..n]);
                        Poll::Ready(Ok(()))
                    }
                    Err(error) => Poll::Ready(Err(error)),
                }
            }
        }
    }
}

impl AsyncWrite for UringAsyncStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if this.write_future.is_none() {
            if data.is_empty() {
                return Poll::Ready(Ok(0));
            }
            let owned = data.to_vec();
            this.write_in_flight = owned.len();
            let inner = this.inner.clone();
            let future: UringWriteFuture = Box::pin(async move {
                let (result, returned) = inner.write_all(owned).await;
                (result, returned)
            });
            this.write_future = Some(future);
        }
        // structural: write_future was set above when None.
        #[allow(clippy::expect_used)]
        let future = this
            .write_future
            .as_mut()
            .expect("write_future set when None above");
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready((result, _owned)) => {
                this.write_future = None;
                match result {
                    Ok(()) => Poll::Ready(Ok(this.write_in_flight)),
                    Err(error) => Poll::Ready(Err(error)),
                }
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // tokio_uring writes submit straight to the kernel; no
        // user-space buffer between submission and the wire.
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.inner.shutdown(std::net::Shutdown::Write))
    }
}

pub async fn request_via_uring(
    request: hyper::Request<StreamingHyperBody>,
) -> Result<hyper::Response<Incoming>, ProximaError> {
    let (tx, rx) = oneshot::channel::<Result<hyper::Response<Incoming>, ProximaError>>();
    tokio::task::spawn_local(async move {
        let outcome = do_request(request).await;
        let _ = tx.send(outcome);
    });
    rx.await.unwrap_or_else(|_| {
        Err(ProximaError::Upstream(
            "uring upstream task dropped before sending response".into(),
        ))
    })
}

async fn do_request(
    request: hyper::Request<StreamingHyperBody>,
) -> Result<hyper::Response<Incoming>, ProximaError> {
    let uri = request.uri().clone();
    let scheme = uri.scheme_str().unwrap_or("http");
    let host = uri
        .host()
        .ok_or_else(|| ProximaError::Upstream(format!("uri has no host: {uri}")))?
        .to_string();
    let port = uri
        .port_u16()
        .unwrap_or_else(|| if scheme == "https" { 443 } else { 80 });

    let socket = connect_uring(&host, port).await?;

    match scheme {
        "http" => send_http(socket, request).await,
        #[cfg(feature = "http1-tls")]
        "https" => send_https(socket, &host, request).await,
        #[cfg(not(feature = "http1-tls"))]
        "https" => Err(ProximaError::Upstream(
            "https upstream requires the `tls` feature".into(),
        )),
        other => Err(ProximaError::Upstream(format!(
            "unsupported scheme on uring path: {other}"
        ))),
    }
}

async fn connect_uring(host: &str, port: u16) -> Result<UringAsyncStream, ProximaError> {
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|error| ProximaError::Upstream(format!("dns {host}:{port}: {error}")))?
        .collect();
    if addrs.is_empty() {
        return Err(ProximaError::Upstream(format!(
            "dns {host}:{port}: no addresses"
        )));
    }
    let mut last_error: Option<std::io::Error> = None;
    for addr in addrs {
        match tokio_uring::net::TcpStream::connect(addr).await {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                return Ok(UringAsyncStream::new(stream));
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(ProximaError::Upstream(format!(
        "tcp connect {host}:{port}: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no addresses tried".into())
    )))
}

async fn send_http(
    stream: UringAsyncStream,
    request: hyper::Request<StreamingHyperBody>,
) -> Result<hyper::Response<Incoming>, ProximaError> {
    let (mut send_request, conn) =
        hyper::client::conn::http1::handshake::<_, StreamingHyperBody>(TokioIo::new(stream))
            .await
            .map_err(|error| ProximaError::Upstream(format!("handshake: {error}")))?;
    tokio::task::spawn_local(async move {
        if let Err(error) = conn.await {
            tracing::debug!(?error, "uring upstream connection ended with error");
        }
    });
    send_request
        .send_request(request)
        .await
        .map_err(|error| ProximaError::Upstream(format!("send: {error}")))
}

#[cfg(feature = "http1-tls")]
async fn send_https(
    stream: UringAsyncStream,
    host: &str,
    request: hyper::Request<StreamingHyperBody>,
) -> Result<hyper::Response<Incoming>, ProximaError> {
    use std::sync::Arc;
    use tokio_rustls::TlsConnector;
    use tokio_rustls::rustls::ClientConfig;
    use tokio_rustls::rustls::RootCertStore;
    use tokio_rustls::rustls::pki_types::ServerName;

    let server_name: ServerName<'static> = ServerName::try_from(host.to_string())
        .map_err(|error| ProximaError::Upstream(format!("invalid sni {host}: {error}")))?;

    // shared once-cell ClientConfig: webpki-roots + no client auth.
    // Built lazily on first https-uring request and reused. Matches
    // the default hyper-rustls config in shared_http.rs.
    static CLIENT_CONFIG: std::sync::OnceLock<Arc<ClientConfig>> = std::sync::OnceLock::new();
    let config = CLIENT_CONFIG
        .get_or_init(|| {
            let mut roots = RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            Arc::new(
                ClientConfig::builder()
                    .with_root_certificates(roots)
                    .with_no_client_auth(),
            )
        })
        .clone();
    let connector = TlsConnector::from(config);
    let tls_stream = connector
        .connect(server_name, stream)
        .await
        .map_err(|error| ProximaError::Upstream(format!("tls handshake: {error}")))?;
    let (mut send_request, conn) =
        hyper::client::conn::http1::handshake::<_, StreamingHyperBody>(TokioIo::new(tls_stream))
            .await
            .map_err(|error| ProximaError::Upstream(format!("https handshake: {error}")))?;
    tokio::task::spawn_local(async move {
        if let Err(error) = conn.await {
            tracing::debug!(?error, "uring upstream tls connection ended with error");
        }
    });
    send_request
        .send_request(request)
        .await
        .map_err(|error| ProximaError::Upstream(format!("https send: {error}")))
}
