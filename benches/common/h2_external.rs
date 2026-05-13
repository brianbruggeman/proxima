//! Benches-only HTTP/2 listener built on the external `h2` crate.
//!
//! Lives under `benches/common/` so the bench files can compare
//! proxima's native `h2` impl against the public `h2` crate without
//! proxima itself shipping a wrapper module. Each bench that needs
//! to drive the public h2 crate includes this file via
//! `#[path = "common/h2_external.rs"] mod h2_external;`.
//!
//! Production (TLS+ALPN+h2) uses `proxima::h2::serve_h2_connection`
//! directly. This file is bench-comparison code, not production code.
//!
//! ## Runtime coupling (honest accounting)
//!
//! **Executor:** the listener does NOT require a tokio executor to
//! drive. Multi-stream concurrency uses `FuturesUnordered +
//! tokio::select!`, both of which are runtime-agnostic (`select!` is a
//! macro that expands to `poll_fn` polling; no runtime calls). No
//! `tokio::spawn` inside. Same task polls `Connection::accept()` and
//! all in-flight handlers; SendStream sends queued by handlers flush
//! on the next `select!` iteration.
//!
//! **IO traits:** the listener requires `tokio::io::AsyncRead +
//! AsyncWrite` on the socket type because the `h2` crate ties to
//! those trait shapes — same coupling as the h1 listener. The
//! substrate-level `crate::stream::{StreamConnection, StreamListener}`
//! traits use `futures::io::*`; bridging substrate streams into h2 is
//! a planned `tokio_util::compat`-style adapter at this listener's
//! boundary. Until that lands, DPDK and other non-tokio transports
//! can't feed this listener directly.
//!
//! ## Dispatch shape
//!
//! Per h2 stream, not per connection: h2 multiplexes many concurrent
//! request/response pairs onto one socket. The handler future for each
//! stream is added to a `FuturesUnordered` so the connection task can
//! drive all of them cooperatively with the accept loop.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt};
use h2::server::SendResponse;
use h2::{RecvStream, SendStream};
use http::{HeaderName, HeaderValue, Method, Request as HttpRequest, Response as HttpResponse};
use tokio::io::{AsyncRead, AsyncWrite};

use proxima::RequestStream;
use proxima::error::ProximaError;
use proxima::header_list::HeaderList;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::PipeHandle;
use proxima::request::{Request, RequestContext, Response};
use proxima_primitives::pipe::Method as ProximaMethod;
use proxima_primitives::pipe::SendPipe;

type HandlerFuture = Pin<Box<dyn std::future::Future<Output = Result<(), ProximaError>> + Send>>;

/// Drive an h2 connection to completion. Each accepted stream maps to
/// one substrate `Pipe::call`; bodies stream both directions through
/// the h2 flow-control machinery.
///
/// Returns when the peer closes the connection or sends GOAWAY.
pub async fn serve_h2_connection<Stream>(
    socket: Stream,
    dispatch: PipeHandle,
    in_flight: Arc<AtomicU64>,
    _quiesce_response: Arc<QuiesceResponse>,
) -> Result<(), ProximaError>
where
    Stream: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut connection = match h2::server::handshake(socket).await {
        Ok(conn) => conn,
        Err(error) => {
            return Err(ProximaError::Upstream(format!("h2 handshake: {error}")));
        }
    };

    // Multi-stream concurrency without runtime-specific spawn: the
    // same task polls `accept()` and a `FuturesUnordered` of in-flight
    // handlers. Same-task wakers wake locally — no tokio::spawn, no
    // dependency on a particular Runtime impl. h2 SendStream sends
    // queued by handlers get flushed on the next accept() poll inside
    // the select! loop.
    let mut in_flight_handlers: FuturesUnordered<HandlerFuture> = FuturesUnordered::new();
    loop {
        tokio::select! {
            biased;
            accept_outcome = connection.accept() => {
                match accept_outcome {
                    Some(Ok((request, respond))) => {
                        in_flight.fetch_add(1, Ordering::Relaxed);
                        in_flight_handlers.push(Box::pin(handle_h2_stream(
                            request,
                            respond,
                            dispatch.clone(),
                        )));
                    }
                    Some(Err(error)) => {
                        return Err(ProximaError::Upstream(format!("h2 accept: {error}")));
                    }
                    None => break,
                }
            }
            Some(handler_outcome) = in_flight_handlers.next(), if !in_flight_handlers.is_empty() => {
                if let Err(error) = handler_outcome {
                    tracing::warn!(?error, "h2 stream error");
                }
                in_flight.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
    // Drain remaining handlers before returning so end-of-stream
    // frames they enqueued get a chance to flush.
    while let Some(handler_outcome) = in_flight_handlers.next().await {
        if let Err(error) = handler_outcome {
            tracing::warn!(?error, "h2 stream drain error");
        }
        in_flight.fetch_sub(1, Ordering::Relaxed);
    }
    Ok(())
}

async fn handle_h2_stream(
    http_request: HttpRequest<RecvStream>,
    mut respond: SendResponse<Bytes>,
    dispatch: PipeHandle,
) -> Result<(), ProximaError> {
    let (parts, recv) = http_request.into_parts();
    let request = build_substrate_request(parts, recv)?;
    let response = match SendPipe::call(&dispatch, request).await {
        Ok(response) => response,
        Err(error) => {
            send_error_response(&mut respond, 500, format!("dispatch: {error}"))?;
            return Err(error);
        }
    };
    write_h2_response(respond, response).await
}

fn build_substrate_request(
    parts: http::request::Parts,
    recv: RecvStream,
) -> Result<Request<Bytes>, ProximaError> {
    let mut headers = HeaderList::new();
    for (name, value) in &parts.headers {
        headers.insert(name.as_str().as_bytes(), value.as_bytes());
    }
    let method = method_bytes(&parts.method);
    let (path, query) = split_path_and_query(&parts.uri);
    Ok(Request {
        method,
        path,
        query,
        metadata: headers,
        payload: Bytes::new(),
        stream: Some(RequestStream::new(H2BodyStream::new(recv))),
        context: RequestContext::default(),
    })
}

fn method_bytes(method: &Method) -> ProximaMethod {
    ProximaMethod::from_bytes(method.as_str().as_bytes())
}

fn split_path_and_query(uri: &http::Uri) -> (Bytes, HeaderList) {
    let path = uri
        .path_and_query()
        .map(|pq| pq.path())
        .unwrap_or("/")
        .to_string();
    let mut query = HeaderList::new();
    if let Some(raw_query) = uri.path_and_query().and_then(|pq| pq.query()) {
        for pair in raw_query.split('&') {
            if let Some((name, value)) = pair.split_once('=') {
                query.insert(name.as_bytes(), value.as_bytes());
            } else if !pair.is_empty() {
                query.insert(pair.as_bytes(), b"".as_slice());
            }
        }
    }
    (Bytes::from(path), query)
}

async fn write_h2_response(
    mut respond: SendResponse<Bytes>,
    response: Response<Bytes>,
) -> Result<(), ProximaError> {
    let status = response.status;
    let mut head = HttpResponse::builder().status(status);
    {
        let head_headers = head
            .headers_mut()
            .ok_or_else(|| ProximaError::Upstream("h2: response builder headers".into()))?;
        for (name, value) in response.metadata.iter() {
            let header_name = HeaderName::from_bytes(name.as_ref())
                .map_err(|err| ProximaError::Upstream(format!("h2 header name: {err}")))?;
            let header_value = HeaderValue::from_bytes(value.as_ref())
                .map_err(|err| ProximaError::Upstream(format!("h2 header value: {err}")))?;
            head_headers.append(header_name, header_value);
        }
    }
    let head = head
        .body(())
        .map_err(|err| ProximaError::Upstream(format!("h2 head build: {err}")))?;
    // Buffer the body first so we can decide whether to send the head with
    // end_of_stream=true (no body) or end_of_stream=false (followed by data).
    let mut buffered: Vec<Bytes> = Vec::new();
    {
        let mut stream = response.into_chunk_stream();
        while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
            let bytes = chunk?;
            if !bytes.is_empty() {
                buffered.push(bytes);
            }
        }
    }
    let body_is_empty = buffered.is_empty();
    let send_stream = respond
        .send_response(head, body_is_empty)
        .map_err(|err| ProximaError::Upstream(format!("h2 send_response: {err}")))?;
    if body_is_empty {
        return Ok(());
    }
    stream_response_buffered(send_stream, buffered).await
}

async fn stream_response_buffered(
    mut send: SendStream<Bytes>,
    chunks: Vec<Bytes>,
) -> Result<(), ProximaError> {
    let total: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    send.reserve_capacity(total);
    let last_index = chunks.len() - 1;
    for (index, bytes) in chunks.into_iter().enumerate() {
        let end_of_stream = index == last_index;
        send.send_data(bytes, end_of_stream)
            .map_err(|err| ProximaError::Upstream(format!("h2 send_data: {err}")))?;
    }
    Ok(())
}

fn send_error_response(
    respond: &mut SendResponse<Bytes>,
    status: u16,
    message: String,
) -> Result<(), ProximaError> {
    let head = HttpResponse::builder()
        .status(status)
        .body(())
        .map_err(|err| ProximaError::Upstream(format!("h2 error head: {err}")))?;
    let mut send = respond
        .send_response(head, false)
        .map_err(|err| ProximaError::Upstream(format!("h2 error send_response: {err}")))?;
    let bytes = Bytes::from(message);
    send.send_data(bytes, true)
        .map_err(|err| ProximaError::Upstream(format!("h2 error send_data: {err}")))?;
    Ok(())
}

/// Stream adapter pulling Bytes chunks from `h2::RecvStream`. Releases
/// h2 flow-control credit per consumed chunk so the peer can keep
/// sending without window starvation.
struct H2BodyStream {
    recv: RecvStream,
}

impl H2BodyStream {
    fn new(recv: RecvStream) -> Self {
        Self { recv }
    }
}

impl Stream for H2BodyStream {
    type Item = Result<Bytes, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.recv).poll_data(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(err))) => {
                Poll::Ready(Some(Err(ProximaError::Upstream(format!("h2 recv: {err}")))))
            }
            Poll::Ready(Some(Ok(chunk))) => {
                let len = chunk.len();
                if len > 0 {
                    let _ = this.recv.flow_control().release_capacity(len);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
        }
    }
}
