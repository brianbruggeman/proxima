//! Async HTTP/2 CLIENT driver over the sans-IO client
//! [`Connection`](proxima_protocols::http2_codec::connection::Connection) — the client-role
//! mirror of [`serve_h2_connection`](crate::http2::serve_h2_connection).
//!
//! # What form is this?
//!
//! The same one the HTTP/1.1 client is — see
//! [`crate::http1::client`] for the long version. Briefly: everything in
//! proxima that does work is a **pipe**, one async `call` from `In` to
//! `Result<Out, Err>` ([`Pipe`](proxima_primitives::pipe::Pipe)), and the
//! form is decided by the types. [`H2ClientUpstream`] picks
//! `In = `[`Request<Bytes>`](proxima_primitives::pipe::Request),
//! `Out = `[`Response<Bytes>`](proxima_primitives::pipe::Response),
//! `Err = `[`ProximaError`], which makes it a **transform**:
//!
//! ```text
//! Request<Bytes>  ──►  H2ClientUpstream  ──►  Response<Bytes>
//! ```
//!
//! Byte-identical in form to the h1 client. That is the point: HTTP/2 is a
//! different wire protocol, not a different shape. Anything that accepts one
//! accepts the other, and neither has to know which it got.
//!
//! Each `call` opens a connection over a [`StreamUpstream`], does the h2
//! handshake (client preface + SETTINGS), opens one stream, sends the request
//! HEADERS (+ DATA body), and reads the response HEADERS (+ DATA body) — one
//! unary request/response. Built for the gRPC / OTLP unary export path;
//! connection-reuse + stream multiplexing are a later layer.
//!
//! # Creating one
//!
//! [`H2ClientUpstream::new`] is the only constructor. Unlike
//! [`H1ClientConfig`](crate::http1::H1ClientConfig) there is no
//! `H2ClientConfig`, no builder, and no `from_config`: the h2 client is
//! configured positionally, in Rust, and its settings cannot be sourced from
//! a file or the environment. See [`H2ClientUpstream`] for the worked
//! example of what does exist.
//!
//! Runtime-neutral: drives any `futures::io` `StreamConnection`, so it runs on
//! the prime reactor (`PrimeTcpUpstream`, optionally TLS-wrapped), DPDK, smol, or
//! a test loopback — no tokio in the request path.

use std::future::Future;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::io::{AsyncReadExt, AsyncWriteExt};

use proxima_core::ProximaError;
use proxima_protocols::http2_codec::connection::{Connection, ConnectionEvent, SendOutcome};
use proxima_protocols::http2_codec::frame::StandardSettings;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::stream::{StreamUpstream, StreamUpstreamExt};

/// Socket read chunk. 16 KiB matches the default SETTINGS_MAX_FRAME_SIZE so a
/// whole frame is typically read in one syscall.
const READ_CHUNK: usize = 16_384;

/// Local SETTINGS the client announces (mirrors the native server's defaults).
fn default_local_settings() -> StandardSettings {
    StandardSettings {
        header_table_size: Some(4096),
        enable_push: Some(false),
        max_concurrent_streams: Some(100),
        initial_window_size: Some(65_535),
        max_frame_size: Some(16_384),
        max_header_list_size: None,
        extensions: Default::default(),
    }
}

/// Names a request header must NOT carry over h2 — h1 connection-specific
/// headers (RFC 7540 §8.1.2.2) plus `host` (h2 uses `:authority`).
fn is_forbidden_h2_request_header(name: &[u8]) -> bool {
    matches!(
        name,
        b"connection"
            | b"keep-alive"
            | b"proxy-connection"
            | b"transfer-encoding"
            | b"upgrade"
            | b"host"
    )
}

/// HTTP/2 client. The **transform** form of
/// [`Pipe`](proxima_primitives::pipe::Pipe):
/// [`Request<Bytes>`](proxima_primitives::pipe::Request) in,
/// [`Response<Bytes>`](proxima_primitives::pipe::Response) out, failing with
/// [`ProximaError`]. See the [module docs](self) for what that means.
///
/// One client = one upstream authority. `secure` selects the `:scheme`
/// (`https` vs `http`). `U` is the transport — any [`StreamUpstream`] — so
/// the client never names a socket type.
///
/// # Creating one
///
/// [`H2ClientUpstream::new`] is the whole construction surface: transport,
/// authority, `secure`, label. Compiled and run by `cargo test`, so the
/// signature cannot drift.
///
/// A [`StreamUpstream`] does not connect when you build it — it connects when
/// the client asks it to — so this opens no socket:
///
/// ```
/// use proxima_http::http2::client::H2ClientUpstream;
/// use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
///
/// let upstream = TokioTcpUpstream::new("127.0.0.1:4317".parse().unwrap());
/// let _client = H2ClientUpstream::new(upstream, "collector:4317", false, "otlp");
/// ```
///
/// There is no `H2ClientConfig`, no builder, and no `from_config`: unlike
/// [`H1ClientConfig`](crate::http1::H1ClientConfig), an h2 client's settings
/// cannot be sourced from a file or the environment. Everything above is
/// positional Rust.
pub struct H2ClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    authority: String,
    secure: bool,
    label: String,
}

impl<U: StreamUpstream> H2ClientUpstream<U> {
    /// `authority` is the `:authority` pseudo-header (e.g. `"collector:4317"`);
    /// `secure` picks the `:scheme`; `label` names the upstream for
    /// diagnostics/tracing (TARGET 3 — served-Pipe naming now lives at the
    /// mount-site label, not the handle).
    pub fn new(
        upstream: U,
        authority: impl Into<String>,
        secure: bool,
        label: impl Into<String>,
    ) -> Self {
        Self {
            upstream: Arc::new(upstream),
            authority: authority.into(),
            secure,
            label: label.into(),
        }
    }

    /// This client's label, set at construction.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl<U: StreamUpstream> SendPipe for H2ClientUpstream<U> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let upstream = Arc::clone(&self.upstream);
        let authority = self.authority.clone();
        let secure = self.secure;
        async move { drive_unary(upstream, authority, secure, request).await }
    }
}


/// Build the request HEADERS (pseudo-headers first, then forwardable regular
/// headers, lowercased per RFC §8.1.2).
fn request_headers(authority: &str, secure: bool, request: &Request<Bytes>) -> Vec<(Bytes, Bytes)> {
    let scheme = if secure {
        Bytes::from_static(b"https")
    } else {
        Bytes::from_static(b"http")
    };
    let mut headers: Vec<(Bytes, Bytes)> = Vec::with_capacity(8);
    headers.push((Bytes::from_static(b":method"), request.method.to_bytes()));
    headers.push((Bytes::from_static(b":scheme"), scheme));
    headers.push((Bytes::from_static(b":path"), request.path.clone()));
    headers.push((
        Bytes::from_static(b":authority"),
        Bytes::from(authority.to_string()),
    ));
    for (name, value) in request.metadata.iter() {
        if name.first() == Some(&b':') || is_forbidden_h2_request_header(name.as_ref()) {
            continue;
        }
        headers.push((
            Bytes::from(name.as_ref().to_ascii_lowercase()),
            value.clone(),
        ));
    }
    headers
}

async fn drive_unary<U: StreamUpstream>(
    upstream: Arc<U>,
    authority: String,
    secure: bool,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let mut socket = upstream
        .connect()
        .await
        .map_err(|err| ProximaError::Upstream(format!("h2 connect: {err}")))?;
    let mut connection = Connection::new_client(default_local_settings());

    let headers = request_headers(&authority, secure, &request);
    let body = request.payload.clone();
    let has_body = !body.is_empty();
    let stream_id = connection.next_local_stream_id();
    connection
        .send_request_head(stream_id, headers, !has_body)
        .map_err(|err| ProximaError::Upstream(format!("h2 send request head: {err}")))?;
    // Body the send window couldn't fit yet; resumed on WindowGranted.
    let mut pending_body: Option<Bytes> = None;
    if has_body {
        match connection
            .send_body(stream_id, body, true)
            .map_err(|err| ProximaError::Upstream(format!("h2 send body: {err}")))?
        {
            SendOutcome::Done => {}
            SendOutcome::WindowExhausted { remainder, .. } => pending_body = Some(remainder),
        }
    }

    let mut status: Option<u16> = None;
    let mut response_headers = HeaderList::new();
    let mut response_body = BytesMut::new();
    let mut done = false;
    let mut read_buf = vec![0u8; READ_CHUNK];

    loop {
        // Flush queued output (preface + SETTINGS + request frames the first
        // pass; SETTINGS-ACK / WINDOW_UPDATE / resumed DATA thereafter).
        let outbound = connection.take_output();
        if !outbound.is_empty() {
            socket
                .write_all(&outbound)
                .await
                .map_err(|err| ProximaError::Upstream(format!("h2 write: {err}")))?;
        }
        if done {
            break;
        }
        let read = socket
            .read(&mut read_buf)
            .await
            .map_err(|err| ProximaError::Upstream(format!("h2 read: {err}")))?;
        if read == 0 {
            return Err(ProximaError::Upstream(
                "h2 connection closed before response complete".into(),
            ));
        }
        connection
            .feed(&read_buf[..read])
            .map_err(|err| ProximaError::Upstream(format!("h2 parse: {err}")))?;

        while let Some(event) = connection.next_event() {
            match event {
                ConnectionEvent::ResponseHead {
                    stream_id: sid,
                    headers,
                    end_stream,
                } if sid == stream_id => {
                    for (name, value) in headers {
                        if name.as_ref() == b":status" {
                            status = core::str::from_utf8(value.as_ref())
                                .ok()
                                .and_then(|text| text.parse().ok());
                        } else if name.first() != Some(&b':') {
                            let _ = response_headers.insert(name, value);
                        }
                    }
                    done = end_stream;
                }
                ConnectionEvent::BodyData {
                    stream_id: sid,
                    data,
                    end_stream,
                } if sid == stream_id => {
                    response_body.extend_from_slice(&data);
                    done = end_stream;
                }
                ConnectionEvent::WindowGranted { .. } => {
                    if let Some(remainder) = pending_body.take() {
                        match connection
                            .send_body(stream_id, remainder, true)
                            .map_err(|err| {
                                ProximaError::Upstream(format!("h2 resume body: {err}"))
                            })? {
                            SendOutcome::Done => {}
                            SendOutcome::WindowExhausted { remainder, .. } => {
                                pending_body = Some(remainder);
                            }
                        }
                    }
                }
                ConnectionEvent::StreamReset {
                    stream_id: sid,
                    error_code,
                } if sid == stream_id => {
                    return Err(ProximaError::Upstream(format!(
                        "h2 stream reset by peer: error_code={error_code}"
                    )));
                }
                ConnectionEvent::PeerGoaway { error_code, .. } if !done => {
                    return Err(ProximaError::Upstream(format!(
                        "h2 connection GOAWAY before response: error_code={error_code}"
                    )));
                }
                _ => {}
            }
        }
    }

    let status =
        status.ok_or_else(|| ProximaError::Upstream("h2 response missing :status".into()))?;
    let mut response = Response::new(status);
    response.metadata = response_headers;
    response.payload = response_body.freeze();
    Ok(response)
}
