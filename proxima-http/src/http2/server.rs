//! Async I/O wrapper around the sans-IO [`super::connection::Connection`].
//!
//! `serve_h2_connection` matches the existing
//! [`crate::listeners::h2_crate::serve_h2_connection`] signature but drives
//! the native HPACK + framing + state-machine stack instead of the
//! `h2` crate.
//!
//! ## Runtime neutrality
//!
//! Takes any `futures::io::AsyncRead + AsyncWrite + Unpin + Send`.
//! No `tokio::spawn`. No `tokio::sync`. No `tokio::io`. No `tokio`
//! dependency at all under `http2-native` — DPDK / io-uring / smol /
//! hand-rolled runtimes plug in by implementing the futures-io traits
//! over their own transport. Tokio callers still work: `Unpin +
//! AsyncRead + AsyncWrite` is satisfied by `tokio_util::compat`-wrapped
//! tokio streams the same as before.
//!
//! ## Loop shape
//!
//! Single task per connection. Each iteration:
//!
//! 1. Drain any pending bytes from `Connection::take_output()` and
//!    write them to the socket.
//! 2. Poll, in fixed priority order (read, then handler completions,
//!    then chunk pulls — mirrors a `biased` select), a hand-rolled
//!    `poll_fn` that returns the first branch to become ready. This
//!    is the tokio-free equivalent of `tokio::select! { biased; ... }`
//!    with per-branch guards: same "poll in order, first Ready wins,
//!    disabled branches skipped" semantics, no tokio dependency.
//! 3. After the select fires, drain ANY other handlers / chunk-pulls
//!    that are already ready via non-blocking `now_or_never` poll.
//!    Batches DATA frames into one syscall instead of one syscall per
//!    frame — critical at high stream concurrency.
//! 4. Drain `Connection::next_event()` and route:
//!    - `RequestHead` with `end_stream`: push handler with empty body.
//!    - `RequestHead` without `end_stream`: stash; wait for body data.
//!    - `BodyData`: pipe into the per-stream body sender.
//!    - `StreamReset`: drop any pending body for that stream.
//!    - `PeerGoaway`: stop accepting new streams (existing keep going).
//!
//! Termination: socket EOF, `PeerGoaway`, or fatal `ConnectionError`.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use futures::stream::FuturesUnordered;
use rustc_hash::FxHashMap;

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, RequestAdmit};
use proxima_protocols::http2_codec::connection::{Connection, ConnectionEvent, SendOutcome};
use proxima_protocols::http2_codec::frame::StandardSettings;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::body::{ChunkStream, RequestStream};
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};
use proxima_primitives::stream::PeerInfo;

/// INTERNAL_ERROR code per RFC 7540 §7.
const INTERNAL_ERROR: u32 = 0x2;

type HandlerFuture =
    Pin<Box<dyn std::future::Future<Output = (u32, Result<Response<Bytes>, ProximaError>)> + Send>>;

/// Default local SETTINGS the native listener announces. Mirrors
/// h2 crate's defaults plus a sensible HEADER_TABLE_SIZE.
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

/// Body-stream adapter backed by a futures mpsc channel. The connection
/// driver pushes `BodyData` chunks via `UnboundedSender`; the Pipe
/// consumes via the futures `Stream` interface (through
/// `Body::from_stream`). Channel is unbounded — peer-side backpressure
/// is enforced by our receive-window auto-replenishment in
/// [`super::connection`]: the peer can have at most `recv_window`
/// bytes in flight per stream. Runtime-neutral: futures::channel has
/// no scheduler hooks; works under tokio, smol, dpdk, or hand-rolled.
struct NativeBodyStream {
    rx: mpsc::UnboundedReceiver<Result<Bytes, ProximaError>>,
}

impl Stream for NativeBodyStream {
    type Item = Result<Bytes, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().rx).poll_next(cx)
    }
}

/// Outcome of the hand-rolled biased poll in `serve_h2_connection`'s main
/// loop — the tokio-free replacement for `tokio::select!`'s three arms.
enum SelectOutcome {
    Read(std::io::Result<usize>),
    Handler((u32, Result<Response<Bytes>, ProximaError>)),
    ChunkPull(ChunkPullOutput),
}

/// Drive a native HTTP/2 connection to completion. Each accepted
/// stream maps to one `Pipe::call`. Request bodies stream into the
/// pipe via `Body::from_stream`; the handler can start consuming
/// chunks before `END_STREAM` arrives.
pub async fn serve_h2_connection<S>(
    socket: S,
    dispatch: PipeHandle,
    admission: ConnAdmission,
    peer: Option<PeerInfo>,
) -> Result<(), ProximaError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut socket = socket;
    let mut connection = Connection::new(default_local_settings());
    // Per-stream body-data sender. When a request has a body, this
    // pipes incoming DATA frames into the streaming `Body` we handed
    // to the pipe. Dropping the sender (on END_STREAM or stream
    // reset) signals end-of-stream to the body consumer.
    // use fxhashmap: per-connection state owned by one task (no
    // cross-thread access, dashmap would only add lock overhead);
    // FxHash on u32 stream-id is one multiply vs SipHash's full
    // rounds, and the table is bounded by max_concurrent_streams
    // so there's no DoS-resistance benefit to crypto hashing.
    let mut body_senders: FxHashMap<u32, mpsc::UnboundedSender<Result<Bytes, ProximaError>>> =
        FxHashMap::default();
    // Per-stream FIFO queue. When a DATA emit hits the send window
    // we queue the remainder; subsequent chunks for that stream
    // also queue (window's still exhausted). A single-slot map
    // would clobber chunk N with chunk N+1, dropping data — that
    // bug shipped in earlier rev. WindowGranted drains the front
    // of the queue per stream and re-queues anything that still
    // can't fit.
    // use vecdeque: push_back / pop_front / push_front are all O(1)
    // on the ring; smallvec inline would force O(n) shifts per drain.
    let mut pending_sends: FxHashMap<u32, VecDeque<PendingSend>> = FxHashMap::default();
    let mut handlers: FuturesUnordered<HandlerFuture> = FuturesUnordered::new();
    // In-flight response body chunk-pulls. Each future yields the
    // next chunk from some stream's body; on resolution we emit
    // a DATA frame and re-push the pull for the next chunk (or
    // emit the terminator if the body ended). Runs in the same
    // task as reads and handler completions — no `tokio::spawn`.
    let mut chunk_pulls: FuturesUnordered<ChunkPullFuture> = FuturesUnordered::new();
    let mut read_buf = vec![0u8; 16_384];
    let mut peer_closed = false;

    loop {
        // Step 1: flush any queued output.
        let outbound = connection.take_output();
        if !outbound.is_empty() {
            socket
                .write_all(&outbound)
                .await
                .map_err(|err| ProximaError::Upstream(format!("h2 native write: {err}")))?;
        }

        // Step 2: termination check.
        if peer_closed
            && handlers.is_empty()
            && body_senders.is_empty()
            && chunk_pulls.is_empty()
            && pending_sends.is_empty()
        {
            break;
        }

        // Step 3: select on (read | handler completion | chunk pull).
        // Only blocks if NO event is currently ready. The "drain ready"
        // loop below picks up everything else that's ready before we
        // pay another syscall to flush — at high stream concurrency
        // this is the difference between 1 frame/syscall and many.
        let outcome = std::future::poll_fn(|cx| {
            if !peer_closed
                && let Poll::Ready(result) = Pin::new(&mut socket).poll_read(cx, &mut read_buf)
            {
                return Poll::Ready(SelectOutcome::Read(result));
            }
            if !handlers.is_empty()
                && let Poll::Ready(Some(item)) = handlers.poll_next_unpin(cx)
            {
                return Poll::Ready(SelectOutcome::Handler(item));
            }
            if !chunk_pulls.is_empty()
                && let Poll::Ready(Some(item)) = chunk_pulls.poll_next_unpin(cx)
            {
                return Poll::Ready(SelectOutcome::ChunkPull(item));
            }
            Poll::Pending
        })
        .await;

        match outcome {
            SelectOutcome::Read(read_result) => {
                let count = read_result
                    .map_err(|err| ProximaError::Upstream(format!("h2 native read: {err}")))?;
                if count == 0 {
                    peer_closed = true;
                } else if let Err(error) = connection.feed(&read_buf[..count]) {
                    // RFC 7540 §6.8: send GOAWAY with the appropriate
                    // error code so the peer knows why we're closing.
                    // Best-effort flush before dropping the socket.
                    let code = error.goaway_code();
                    let debug = Bytes::from(format!("{error}"));
                    connection.send_goaway(code, debug);
                    let outbound = connection.take_output();
                    if !outbound.is_empty() {
                        let _ = socket.write_all(&outbound).await;
                    }
                    let _ = socket.flush().await;
                    return Err(ProximaError::Upstream(format!("h2 native parse: {error}")));
                }
            }
            SelectOutcome::Handler(item) => {
                process_handler_completion(item, &admission, &mut connection, &mut chunk_pulls)?;
            }
            SelectOutcome::ChunkPull(item) => {
                process_chunk_pull_completion(
                    item,
                    &mut connection,
                    &mut chunk_pulls,
                    &mut pending_sends,
                )?;
            }
        }

        // Drain everything else that's ready RIGHT NOW. Batches frames
        // into connection.output so the single write_all at the top of
        // the next iteration carries many DATA frames in one syscall.
        // No await — `now_or_never` polls each next() once; Pending
        // returns None and we move on.
        while let Some(item) = handlers.next().now_or_never().flatten() {
            process_handler_completion(item, &admission, &mut connection, &mut chunk_pulls)?;
        }
        while let Some(item) = chunk_pulls.next().now_or_never().flatten() {
            process_chunk_pull_completion(
                item,
                &mut connection,
                &mut chunk_pulls,
                &mut pending_sends,
            )?;
        }

        // Step 4: drain protocol events.
        while let Some(event) = connection.next_event() {
            match event {
                ConnectionEvent::RequestHead {
                    stream_id,
                    headers,
                    end_stream,
                } => {
                    // RFC 9113 §8.5 + RFC 9110 §10.1.1 — Expect:
                    // 100-continue support. If the request advertises
                    // it and carries a body, emit an interim 100
                    // HEADERS frame eagerly so the client can start
                    // sending DATA. Unlike h1, h2 doesn't have a
                    // server-side gate the Pipe can hold open;
                    // sending 100 here matches envoy / nginx /
                    // pingora h2 behavior. Pipe can still reject
                    // with a 4xx final status without consuming the
                    // body.
                    if !end_stream && expects_100_continue(&headers) {
                        let _ = connection.send_response_head(
                            stream_id,
                            std::iter::once((
                                Bytes::from_static(b":status"),
                                Bytes::from_static(b"100"),
                            )),
                            false,
                        );
                    }
                    let body = if end_stream {
                        // No body — pipe gets a buffered empty request.
                        None
                    } else {
                        // Streaming body — open a channel; subsequent
                        // `BodyData` events push chunks into it. The
                        // pipe starts running on the next poll —
                        // it doesn't wait for END_STREAM.
                        let (tx, rx) = mpsc::unbounded();
                        body_senders.insert(stream_id, tx);
                        Some(RequestStream::new(NativeBodyStream { rx }))
                    };
                    let (request, headers_buffer) = build_request(headers, body, peer.clone());
                    connection.return_headers_buffer(headers_buffer);
                    // Request-level admission: h2 has its own natural
                    // per-stream boundary (RequestHead), so it calls
                    // `request_admit`/`request_release` directly instead of
                    // the legacy-atomics bridge h1 uses (see
                    // `ConnAdmission::in_flight_counter`'s doc). On `Shed`
                    // this stream alone gets an in-band 503 — the
                    // connection and every other live stream keep running.
                    match admission.request_admit() {
                        RequestAdmit::Admit => {
                            spawn_handler(stream_id, request, &dispatch, &mut handlers);
                        }
                        RequestAdmit::Shed { reason } => {
                            tracing::debug!(
                                stream_id,
                                ?reason,
                                "h2 native request shed by listener admission"
                            );
                            let response = Response::new(503)
                                .with_body(Bytes::from_static(b"service unavailable"))
                                .with_header("content-type", "text/plain")
                                .with_header("retry-after", "1");
                            match emit_response_head_and_first_pull(
                                &mut connection,
                                stream_id,
                                response,
                            ) {
                                Ok(pull) => chunk_pulls.push(pull),
                                Err(render_error) => {
                                    tracing::warn!(
                                        ?render_error,
                                        stream_id,
                                        "h2 native shed-response render failed"
                                    );
                                    let _ = connection.send_rst(stream_id, INTERNAL_ERROR);
                                }
                            }
                        }
                    }
                }
                ConnectionEvent::BodyData {
                    stream_id,
                    data,
                    end_stream,
                } => {
                    // Push the chunk to the body consumer. If the
                    // sender errored (receiver dropped, e.g. pipe
                    // canceled the body), reset the stream and clean
                    // up — the peer must stop sending DATA.
                    let send_result = body_senders
                        .get(&stream_id)
                        .map(|tx| tx.unbounded_send(Ok(data)));
                    match send_result {
                        None => {} // stream already cleaned up
                        Some(Ok(())) if end_stream => {
                            body_senders.remove(&stream_id);
                        }
                        Some(Ok(())) => {}
                        Some(Err(_)) => {
                            body_senders.remove(&stream_id);
                            let _ = connection.send_rst(stream_id, INTERNAL_ERROR);
                        }
                    }
                }
                ConnectionEvent::StreamReset { stream_id, .. } => {
                    body_senders.remove(&stream_id);
                    pending_sends.remove(&stream_id);
                }
                ConnectionEvent::PeerGoaway { .. } => {
                    peer_closed = true;
                }
                ConnectionEvent::WindowGranted { stream_id, .. } => {
                    resume_pending_sends(&mut connection, &mut pending_sends, stream_id)?;
                }
                // ResponseHead is a CLIENT-role event; a server connection never
                // emits it (the codec only produces it for `Connection::new_client`).
                ConnectionEvent::ResponseHead { .. }
                | ConnectionEvent::PingAcked { .. }
                | ConnectionEvent::SettingsApplied => {}
            }
        }

        // Step 5: reap closed streams so the per-connection table
        // doesn't grow unbounded across long-lived connections.
        connection.gc_closed_streams();
    }

    // Final flush before close (covers the GOAWAY we may have queued).
    let final_bytes = connection.take_output();
    if !final_bytes.is_empty() {
        let _ = socket.write_all(&final_bytes).await;
    }
    let _ = socket.flush().await;
    Ok(())
}

/// Body bytes the wrapper is holding because the connection's send
/// window stalled mid-response. Resumed on `WindowGranted`.
struct PendingSend {
    remainder: Bytes,
    end_stream: bool,
}

fn resume_pending_sends(
    connection: &mut Connection,
    pending_sends: &mut FxHashMap<u32, VecDeque<PendingSend>>,
    granted_stream_id: u32,
) -> Result<(), ProximaError> {
    // Stream 0 grants connection-level credit -> try every stalled
    // stream. Stream-level grants only help that one stream. Inline
    // 4 stalled streams covers the common case (typical concurrency
    // is low and most streams aren't stalled).
    let stream_ids: smallvec::SmallVec<[u32; 4]> = if granted_stream_id == 0 {
        pending_sends.keys().copied().collect()
    } else if pending_sends.contains_key(&granted_stream_id) {
        smallvec::smallvec![granted_stream_id]
    } else {
        return Ok(());
    };
    for stream_id in stream_ids {
        let Some(queue) = pending_sends.get_mut(&stream_id) else {
            continue;
        };
        while let Some(pending) = queue.pop_front() {
            match connection
                .send_body(stream_id, pending.remainder, pending.end_stream)
                .map_err(|err| ProximaError::Upstream(format!("h2 native resume send: {err}")))?
            {
                SendOutcome::Done => {}
                SendOutcome::WindowExhausted {
                    remainder,
                    end_stream,
                } => {
                    // window's still tight — push the unsent slice
                    // back to the FRONT so chunk order is preserved
                    // against any later queued chunks.
                    queue.push_front(PendingSend {
                        remainder,
                        end_stream,
                    });
                    break;
                }
            }
        }
        if queue.is_empty() {
            pending_sends.remove(&stream_id);
        }
    }
    Ok(())
}

fn process_handler_completion(
    item: (u32, Result<Response<Bytes>, ProximaError>),
    admission: &ConnAdmission,
    connection: &mut Connection,
    chunk_pulls: &mut FuturesUnordered<ChunkPullFuture>,
) -> Result<(), ProximaError> {
    let (stream_id, response_result) = item;
    admission.request_release();
    match response_result {
        Ok(response) => {
            let pull = emit_response_head_and_first_pull(connection, stream_id, response)?;
            chunk_pulls.push(pull);
        }
        // A deliberate refusal (filter `RejectMode::Drop`) is not a
        // failure — render it as a real response (mirrors h1's
        // `http_status_for`/`error_response_body`) instead of
        // RST_STREAM, or the rejection never reaches the client.
        // Genuinely internal errors keep today's RST_STREAM behaviour.
        Err(error @ ProximaError::Forbidden(_)) => {
            tracing::debug!(stream_id, "h2 native handler rejected request");
            let status = crate::error_render::http_status_for(&error);
            let body = crate::error_render::error_response_body(&error);
            let response = Response::new(status)
                .with_body(body)
                .with_header("content-type", "text/plain");
            match emit_response_head_and_first_pull(connection, stream_id, response) {
                Ok(pull) => chunk_pulls.push(pull),
                Err(render_error) => {
                    tracing::warn!(?render_error, stream_id, "h2 native rejection render failed");
                    let _ = connection.send_rst(stream_id, INTERNAL_ERROR);
                }
            }
        }
        Err(error) => {
            tracing::warn!(?error, stream_id, "h2 native handler error");
            let _ = connection.send_rst(stream_id, INTERNAL_ERROR);
        }
    }
    Ok(())
}

fn process_chunk_pull_completion(
    item: ChunkPullOutput,
    connection: &mut Connection,
    chunk_pulls: &mut FuturesUnordered<ChunkPullFuture>,
    pending_sends: &mut FxHashMap<u32, VecDeque<PendingSend>>,
) -> Result<(), ProximaError> {
    let (stream_id, chunk, body_stream) = item;
    match chunk {
        Some(Ok(data)) => {
            emit_response_chunk(connection, stream_id, data, false, pending_sends)?;
            chunk_pulls.push(next_chunk_future(stream_id, body_stream));
        }
        Some(Err(error)) => {
            tracing::warn!(?error, stream_id, "h2 native response body error");
            let _ = connection.send_rst(stream_id, INTERNAL_ERROR);
        }
        None => {
            // Body completed. If there are stashed remainder chunks
            // for this stream, mark END_STREAM on the LAST queued
            // entry — that's the chunk whose DATA frame will be
            // emitted last. Don't emit a fresh empty terminator: it
            // would close the stream ahead of the unsent queued bytes.
            let marked = pending_sends
                .get_mut(&stream_id)
                .and_then(|queue| queue.back_mut())
                .map(|last| {
                    last.end_stream = true;
                })
                .is_some();
            if !marked {
                emit_response_chunk(connection, stream_id, Bytes::new(), true, pending_sends)?;
            }
        }
    }
    Ok(())
}

/// Detect `Expect: 100-continue` (case-insensitive name + value)
/// in the decoded HEADERS for an inbound request. RFC 9110 §10.1.1
/// pins "100-continue" as the only defined expectation token.
fn expects_100_continue(headers: &[(Bytes, Bytes)]) -> bool {
    headers.iter().any(|(name, value)| {
        name.as_ref().eq_ignore_ascii_case(b"expect")
            && value.as_ref().eq_ignore_ascii_case(b"100-continue")
    })
}

/// Builds the substrate `Request` from a decoded HEADERS event AND
/// hands the (now-empty, capacity-retained) `headers` buffer back to
/// the caller — which passes it to
/// [`proxima_protocols::http2_codec::connection::Connection::return_headers_buffer`]
/// so the connection's NEXT `complete_headers` call reuses the
/// allocation instead of paying a fresh `Vec::with_capacity` (see that
/// method's doc). `headers.drain(..)` (not `headers.into_iter()`) is
/// the load-bearing choice here: `into_iter()` would consume the
/// Vec's backing allocation along with its elements, leaving nothing
/// to return.
fn build_request(
    mut headers: Vec<(Bytes, Bytes)>,
    body: Option<RequestStream>,
    peer: Option<PeerInfo>,
) -> (Request<Bytes>, Vec<(Bytes, Bytes)>) {
    let mut method_bytes = Bytes::from_static(b"GET");
    let mut path = Bytes::from_static(b"/");
    let mut header_list = HeaderList::new();
    for (name, value) in headers.drain(..) {
        match name.as_ref() {
            b":method" => method_bytes = value,
            b":path" => path = value,
            b":scheme" | b":authority" | b":status" | b":protocol" => {
                // pseudo-headers: status is response-only; scheme +
                // authority feed the URI which we approximate via
                // headers below.
            }
            _ => {
                let _ = header_list.insert(name, value);
            }
        }
    }
    let mut context = RequestContext {
        peer,
        ..RequestContext::default()
    };
    let (trace_id, baggage) =
        proxima_telemetry::propagation::establish_trace_context(&header_list);
    context.adopt_trace_context(trace_id, baggage);
    let request = Request {
        method: Method::from_wire(method_bytes),
        path,
        query: HeaderList::new(),
        metadata: header_list,
        payload: Bytes::new(),
        stream: body,
        context,
    };
    (request, headers)
}

fn spawn_handler(
    stream_id: u32,
    request: Request<Bytes>,
    dispatch: &PipeHandle,
    handlers: &mut FuturesUnordered<HandlerFuture>,
) {
    // no in-flight increment here — `request_admit()` already incremented
    // the shared counter at the call site before deciding to admit.
    let dispatch = Arc::clone(dispatch);
    handlers.push(Box::pin(async move {
        let result = dispatch_request(&dispatch, request).await;
        (stream_id, result)
    }));
}

/// Dispatch one request through the Pipe chain, opening a span that
/// continues the inbound W3C trace when the request carried a `traceparent`
/// header — `proxima_telemetry::propagation::establish_trace_context` +
/// `RequestContext::adopt_trace_context` already restamped it onto
/// `request.context` at ingress (`build_request` above) — or a fresh root
/// otherwise. Mirrors the h1 boundary seam in `proxima-h1/src/serve.rs`.
#[proxima_telemetry::instrument(name = "h2_request", parent = request.context.traceparent(), err)]
async fn dispatch_request(
    dispatch: &PipeHandle,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    dispatch.call_dyn(request).await
}

/// Pre-built `Bytes` for HTTP status code values we hand to the
/// HPACK encoder as the `:status` value. Saves a `to_string()` +
/// `Bytes::from(String)` alloc per response on the common cases.
/// Falls back to runtime stringification for everything else.
fn status_bytes(status: u16) -> Bytes {
    match status {
        200 => Bytes::from_static(b"200"),
        201 => Bytes::from_static(b"201"),
        202 => Bytes::from_static(b"202"),
        204 => Bytes::from_static(b"204"),
        206 => Bytes::from_static(b"206"),
        301 => Bytes::from_static(b"301"),
        302 => Bytes::from_static(b"302"),
        304 => Bytes::from_static(b"304"),
        307 => Bytes::from_static(b"307"),
        308 => Bytes::from_static(b"308"),
        400 => Bytes::from_static(b"400"),
        401 => Bytes::from_static(b"401"),
        403 => Bytes::from_static(b"403"),
        404 => Bytes::from_static(b"404"),
        405 => Bytes::from_static(b"405"),
        409 => Bytes::from_static(b"409"),
        410 => Bytes::from_static(b"410"),
        429 => Bytes::from_static(b"429"),
        500 => Bytes::from_static(b"500"),
        502 => Bytes::from_static(b"502"),
        503 => Bytes::from_static(b"503"),
        504 => Bytes::from_static(b"504"),
        _ => Bytes::from(status.to_string()),
    }
}

/// Send the response head (HEADERS frame with `END_STREAM=false`),
/// then return a future that resolves to the first chunk of the
/// body. The caller pushes the future onto a `FuturesUnordered` so
/// chunk-pulling happens alongside reads + handler completions in
/// the same connection task. No `tokio::spawn` — runtime-agnostic.
fn emit_response_head_and_first_pull(
    connection: &mut Connection,
    stream_id: u32,
    response: Response<Bytes>,
) -> Result<ChunkPullFuture, ProximaError> {
    let mut headers: smallvec::SmallVec<[(Bytes, Bytes); 12]> = smallvec::SmallVec::new();
    headers.push((
        Bytes::from_static(b":status"),
        status_bytes(response.status),
    ));
    for (name, value) in response.metadata.iter() {
        headers.push((name.clone(), value.clone()));
    }
    // HEADERS with END_STREAM=false always; the chunk pump emits
    // a final DATA with END_STREAM=true even for empty bodies
    // (one trailing empty DATA frame — 9 bytes — vs branching on
    // body emptiness which we can't determine sync without
    // consuming the stream).
    connection
        .send_response_head(stream_id, headers, false)
        .map_err(|err| ProximaError::Upstream(format!("h2 native response head: {err}")))?;
    Ok(next_chunk_future(stream_id, response.into_chunk_stream()))
}

/// Output of one chunk pull: `(stream_id, next_chunk_or_end,
/// remaining_stream)`. `None` for the second tuple element means the
/// body stream completed (no more chunks); `Some(Err(_))` means the
/// body errored mid-stream.
type ChunkPullOutput = (u32, Option<Result<Bytes, ProximaError>>, ChunkStream);
type ChunkPullFuture = Pin<Box<dyn std::future::Future<Output = ChunkPullOutput> + Send>>;

fn next_chunk_future(stream_id: u32, mut body_stream: ChunkStream) -> ChunkPullFuture {
    Box::pin(async move {
        let next = body_stream.next().await;
        (stream_id, next, body_stream)
    })
}

/// Emit one chunk as a DATA frame respecting flow control. Returns
/// the optional next-chunk-pull future to re-push onto the
/// `FuturesUnordered` (None if this was the terminating chunk).
fn emit_response_chunk(
    connection: &mut Connection,
    stream_id: u32,
    data: Bytes,
    end_stream: bool,
    pending_sends: &mut FxHashMap<u32, VecDeque<PendingSend>>,
) -> Result<(), ProximaError> {
    // already-queued chunks mean the window is exhausted for this
    // stream; calling send_body now would emit nothing new and just
    // hand us back another remainder. Preserve order by queueing.
    if let Some(queue) = pending_sends.get_mut(&stream_id)
        && !queue.is_empty()
    {
        queue.push_back(PendingSend {
            remainder: data,
            end_stream,
        });
        return Ok(());
    }
    match connection
        .send_body(stream_id, data, end_stream)
        .map_err(|err| ProximaError::Upstream(format!("h2 native response chunk: {err}")))?
    {
        SendOutcome::Done => {}
        SendOutcome::WindowExhausted {
            remainder,
            end_stream,
        } => {
            pending_sends
                .entry(stream_id)
                .or_default()
                .push_back(PendingSend {
                    remainder,
                    end_stream,
                });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn pair(name: &'static [u8], value: &'static [u8]) -> (Bytes, Bytes) {
        (Bytes::from_static(name), Bytes::from_static(value))
    }

    #[test]
    fn expects_100_continue_matches_exact_lowercase() {
        let headers = vec![pair(b"expect", b"100-continue")];
        assert!(expects_100_continue(&headers));
    }

    #[test]
    fn expects_100_continue_matches_mixed_case() {
        let headers = vec![pair(b"Expect", b"100-Continue")];
        assert!(expects_100_continue(&headers));
    }

    #[test]
    fn expects_100_continue_rejects_other_expectation() {
        let headers = vec![pair(b"expect", b"some-future-expectation")];
        assert!(!expects_100_continue(&headers));
    }

    #[test]
    fn expects_100_continue_returns_false_when_header_absent() {
        let headers = vec![pair(b":method", b"POST"), pair(b"host", b"example.com")];
        assert!(!expects_100_continue(&headers));
    }
}
