//! HTTP/3 server driver. Maps each accepted bidi stream to one
//! `Pipe::call`. Mirrors [`crate::h2::serve_h2_connection`]'s shape
//! but rides on QUIC streams instead of an h2 multiplexer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::{Buf, Bytes, BytesMut};
use futures::StreamExt;
use futures::channel::mpsc;
use futures::stream::FuturesUnordered;
use http::HeaderName;

use proxima_core::ProximaError;
use proxima_primitives::pipe::body::RequestStream;
use proxima_primitives::pipe::endpoint::PeerInfo;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};
use proxima_quic::Connection as QuicConnection;

type H3Connection = h3::server::Connection<h3_quinn::Connection, Bytes>;
type H3RequestStream = h3::server::RequestStream<
    <h3_quinn::Connection as h3::quic::OpenStreams<Bytes>>::BidiStream,
    Bytes,
>;
type HandlerOutcome = Result<(), ProximaError>;
type HandlerFuture = std::pin::Pin<Box<dyn Future<Output = HandlerOutcome> + Send>>;
type BuiltRequest = Result<
    (
        Request<Bytes>,
        futures::channel::mpsc::UnboundedSender<Result<Bytes, ProximaError>>,
    ),
    ProximaError,
>;

/// Drive a single QUIC connection through HTTP/3 to completion.
/// Each accepted request runs concurrently inside a FuturesUnordered
/// — no `tokio::spawn`, so the driver works on per-core runtimes.
pub async fn serve_h3_connection(
    quic: QuicConnection,
    dispatch: PipeHandle,
    in_flight: Arc<AtomicU64>,
) -> Result<(), ProximaError> {
    let peer = Some(PeerInfo::Tcp(quic.remote_address()));
    let quinn_conn = quic.quinn();
    let h3_quinn_conn = h3_quinn::Connection::new(quinn_conn);

    let mut h3_conn: H3Connection = h3::server::builder()
        .build(h3_quinn_conn)
        .await
        .map_err(|err| ProximaError::Upstream(format!("h3 build: {err}")))?;

    let mut handlers: FuturesUnordered<HandlerFuture> = FuturesUnordered::new();
    let mut peer_done = false;

    loop {
        if peer_done && handlers.is_empty() {
            break;
        }

        tokio::select! {
            biased;
            accept = h3_conn.accept(), if !peer_done => {
                match accept {
                    Ok(Some(resolver)) => {
                        let dispatch = dispatch.clone();
                        let in_flight = in_flight.clone();
                        let peer = peer.clone();
                        handlers.push(Box::pin(async move {
                            let (request, stream) = match resolver.resolve_request().await {
                                Ok(pair) => pair,
                                Err(err) => {
                                    return Err(ProximaError::Upstream(format!(
                                        "h3 resolve_request: {err}"
                                    )));
                                }
                            };
                            serve_h3_request(request, stream, dispatch, in_flight, peer).await
                        }));
                    }
                    Ok(None) => {
                        peer_done = true;
                    }
                    Err(err) => {
                        return Err(ProximaError::Upstream(format!(
                            "h3 accept: {err}"
                        )));
                    }
                }
            }
            // A single request's failure (malformed headers, a filter
            // rejection, an internal pipe error, ...) must not tear
            // down the whole multiplexed connection — every other
            // concurrent stream would die with it. `drive_h3_request`
            // already renders/aborts per-stream; this is the backstop
            // for anything that fails before a stream exists to
            // render on (e.g. `resolve_request`).
            Some(result) = handlers.next(), if !handlers.is_empty() => {
                if let Err(error) = result {
                    tracing::warn!(?error, "h3 native request task failed");
                }
            }
        }
    }

    Ok(())
}

async fn serve_h3_request(
    request: http::Request<()>,
    mut stream: H3RequestStream,
    dispatch: PipeHandle,
    in_flight: Arc<AtomicU64>,
    peer: Option<PeerInfo>,
) -> Result<(), ProximaError> {
    in_flight.fetch_add(1, Ordering::Relaxed);
    let result = drive_h3_request(request, &mut stream, &dispatch, peer).await;
    in_flight.fetch_sub(1, Ordering::Relaxed);
    result
}

async fn drive_h3_request(
    request: http::Request<()>,
    stream: &mut H3RequestStream,
    dispatch: &PipeHandle,
    peer: Option<PeerInfo>,
) -> Result<(), ProximaError> {
    let (proxima_request, request_body_tx) = build_request(&request, peer)?;

    // Body bytes arrive after the headers — kick a pump that
    // forwards `recv_data` chunks into the body channel. The
    // pipe can start consuming before the body finishes.
    let body_pump = pump_request_body(stream, request_body_tx);
    let dispatch_future = dispatch_request(dispatch, proxima_request);

    let response = match futures::try_join!(dispatch_future, body_pump) {
        Ok((response, ())) => response,
        // A rejection (or any other per-request failure) must stay
        // scoped to THIS stream — returning `Err` here would
        // propagate out of the pushed handler future and, left
        // unguarded, tear down the whole h3 connection over one
        // request. Render the deliberate-refusal case as a real
        // response (mirrors h1's status/body mapping); abort
        // everything else with a stream-level reset, the h3
        // equivalent of h2's RST_STREAM.
        Err(error) => return finish_stream_with_error(stream, error).await,
    };

    write_response(stream, response).await
}

/// Turn a per-request error into a terminal, connection-preserving
/// outcome for its stream. `Forbidden` renders as a real response
/// (reuses the shared h1 status/body mapping); anything else resets
/// the stream and leaves the connection to keep serving other
/// requests. Always returns `Ok(())` — the caller must not propagate
/// this error further up into the connection driver.
async fn finish_stream_with_error(
    stream: &mut H3RequestStream,
    error: ProximaError,
) -> Result<(), ProximaError> {
    if !matches!(error, ProximaError::Forbidden(_)) {
        tracing::warn!(?error, "h3 native handler error");
        stream.stop_stream(h3::error::Code::H3_INTERNAL_ERROR);
        return Ok(());
    }
    tracing::debug!("h3 native handler rejected request");
    let status = crate::error_render::http_status_for(&error);
    let body = crate::error_render::error_response_body(&error);
    let response = Response::new(status)
        .with_body(body)
        .with_header("content-type", "text/plain");
    write_response(stream, response).await
}

/// Dispatch one request through the Pipe chain, opening a span that
/// continues the inbound W3C trace when the request carried a `traceparent`
/// header — `proxima_telemetry::propagation::establish_trace_context` +
/// `RequestContext::adopt_trace_context` already restamped it onto
/// `request.context` at ingress (`build_request` below) — or a fresh root
/// otherwise. Mirrors the h1/h2 boundary seam in `proxima-h1/src/serve.rs`
/// and `proxima-h2/src/server.rs`.
#[proxima_telemetry::instrument(name = "h3_request", parent = request.context.traceparent(), err)]
async fn dispatch_request(
    dispatch: &PipeHandle,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    SendPipe::call(dispatch, request).await
}

fn build_request(request: &http::Request<()>, peer: Option<PeerInfo>) -> BuiltRequest {
    let mut headers = HeaderList::new();
    for (name, value) in request.headers() {
        headers.insert(name.as_str().as_bytes(), value.as_bytes());
    }

    let (path_bytes, query) = split_path_query(request.uri().path_and_query());
    let method_bytes: Bytes = Bytes::copy_from_slice(request.method().as_str().as_bytes());

    let (tx, rx) = mpsc::unbounded::<Result<Bytes, ProximaError>>();
    let body = RequestStream::new(rx);

    let mut context = RequestContext {
        peer,
        ..RequestContext::default()
    };
    let (trace_id, baggage) = proxima_telemetry::propagation::establish_trace_context(&headers);
    context.adopt_trace_context(trace_id, baggage);
    let mut built = Request::builder()
        .method(method_bytes)
        .path(path_bytes)
        .stream(body)
        .context(context)
        .build()?;
    built.metadata = headers;
    built.query = query;
    Ok((built, tx))
}

fn split_path_query(pq: Option<&http::uri::PathAndQuery>) -> (Bytes, HeaderList) {
    let Some(pq) = pq else {
        return (Bytes::from_static(b"/"), HeaderList::new());
    };
    let path = Bytes::copy_from_slice(pq.path().as_bytes());
    let mut query = HeaderList::new();
    if let Some(raw) = pq.query() {
        for pair in raw.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
            query.insert(name.as_bytes(), value.as_bytes());
        }
    }
    (path, query)
}

async fn pump_request_body(
    stream: &mut H3RequestStream,
    tx: mpsc::UnboundedSender<Result<Bytes, ProximaError>>,
) -> Result<(), ProximaError> {
    loop {
        match stream.recv_data().await {
            Ok(Some(mut buf)) => {
                let len = buf.remaining();
                if len == 0 {
                    continue;
                }
                let mut bytes = BytesMut::with_capacity(len);
                while buf.has_remaining() {
                    let chunk = buf.chunk();
                    bytes.extend_from_slice(chunk);
                    let advance = chunk.len();
                    buf.advance(advance);
                }
                if tx.unbounded_send(Ok(bytes.freeze())).is_err() {
                    return Ok(());
                }
            }
            Ok(None) => return Ok(()),
            Err(err) => {
                let _ =
                    tx.unbounded_send(Err(ProximaError::Upstream(format!("h3 recv_data: {err}"))));
                return Err(ProximaError::Upstream(format!("h3 recv_data: {err}")));
            }
        }
    }
}

async fn write_response(
    stream: &mut H3RequestStream,
    response: Response<Bytes>,
) -> Result<(), ProximaError> {
    let status = response.status;
    let headers = response.metadata.clone();

    let mut http_response = http::Response::builder()
        .status(status)
        .body(())
        .map_err(|err| ProximaError::Upstream(format!("h3 response build: {err}")))?;
    for (name, value) in headers.iter() {
        let header_name = HeaderName::from_bytes(name)
            .map_err(|err| ProximaError::Upstream(format!("h3 response header name: {err}")))?;
        let header_value = http::HeaderValue::from_bytes(value)
            .map_err(|err| ProximaError::Upstream(format!("h3 response header value: {err}")))?;
        http_response
            .headers_mut()
            .append(header_name, header_value);
    }

    stream
        .send_response(http_response)
        .await
        .map_err(|err| ProximaError::Upstream(format!("h3 send_response: {err}")))?;

    let mut body_stream = response.into_chunk_stream();
    while let Some(chunk) = body_stream.next().await {
        let chunk = chunk.map_err(|err| ProximaError::Upstream(format!("h3 body chunk: {err}")))?;
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
    Ok(())
}
