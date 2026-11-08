//! HTTP/1.1 connection driver — the per-socket serve loop.
//!
//! `serve_h1_connection` is the public, caller-friendly entry point for
//! one-off use (control-plane UDS, short-lived listeners). The umbrella's
//! ALPN-multiplex combiner (`HttpListenProtocol`) drives `serve_connection`
//! directly so it can pass the listener-scoped in_flight / quiesce
//! state through. Both go through the same `Connection` state machine,
//! same body framing, same response writer.
//!
//! Moved out of the umbrella's `listeners/http.rs` so the proxima-h1
//! crate carries its own serve impl (the plan's "h1 parser + listener
//! + upstream" file mapping). The umbrella combiner keeps the ALPN
//!   multiplex, TLS termination, UDS, and SO_REUSEPORT orchestration.

use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::task::Poll;

use bytes::Bytes;
use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use proxima_telemetry::{debug, error};

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, RequestAdmit};
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::body::RequestStream;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::quiesce::QuiesceResponse;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};
use proxima_runtime::Runtime;

use crate::error_render::{error_response_body, http_status_for};
use crate::http1::h1_body::BodyFraming;
use crate::http1::h1_connection::{Advanced, AutoStreamPolicy, Connection, Poll as ConnPoll};

/// Body-channel depth between the listener (producer) and the
/// Pipe (consumer). Bounded — when the consumer is slow the
/// listener parks on `send().await`, which back-pressures the
/// socket: read pauses → kernel recv buffer fills → TCP window
/// closes → peer slows.
const BODY_CHANNEL_DEPTH: usize = 8;

#[derive(Debug, Clone, Default)]
pub struct HttpListenerSpec {
    pub max_body_bytes: Option<usize>,
}

/// The in-band response for a request `ConnAdmission::request_admit` sheds
/// — identical body/headers to h2 native's own shed arm
/// (`proxima-http/src/http2/server.rs`'s `RequestAdmit::Shed` branch), so a
/// client sees the same wire contract regardless of which listener
/// candidate it landed on. Never a connection reset: the connection (and,
/// for h1, its keep-alive state) is untouched by a shed decision.
fn shed_response() -> Response<Bytes> {
    Response::new(503)
        .with_body(Bytes::from_static(b"service unavailable"))
        .with_header("content-type", "text/plain")
        .with_header("retry-after", "1")
}

struct RequestAdmissionGuard<'a> {
    admission: &'a ConnAdmission,
    outcome: RequestAdmit,
    route: &'static str,
    trace_id: Option<Arc<[u8]>>,
    released: bool,
}

impl<'a> RequestAdmissionGuard<'a> {
    fn new(admission: &'a ConnAdmission, outcome: RequestAdmit, route: &'static str) -> Self {
        Self {
            admission,
            outcome,
            route,
            trace_id: None,
            released: false,
        }
    }

    fn release(&mut self, release_owner: &'static str) {
        if self.released {
            return;
        }
        let admitted = matches!(self.outcome, RequestAdmit::Admit);
        if admitted {
            self.admission.request_release();
        }
        debug!(
            route = self.route,
            trace_id = ?self.trace_id,
            lifecycle_phase = "response_ready_or_terminal",
            admission_state = if admitted { "admitted" } else { "shed" },
            admission_release = admitted,
            release_owner = release_owner,
            "h1 request admission ownership ended"
        );
        self.released = true;
    }
}

impl Drop for RequestAdmissionGuard<'_> {
    fn drop(&mut self) {
        self.release("owner_drop");
    }
}

/// The streaming-path counterpart of h2 native's body-drain fix
/// (`proxima-http/src/http2/server.rs`'s `drain_request_body`): a shed
/// streaming request still has its body arriving on the wire (the pump
/// keeps feeding `body_rx` regardless of admission's decision), so this
/// reads-and-discards it instead of dispatching to the real Pipe —
/// keeping the connection's byte stream in sync so the 503 rendered
/// alongside this lands cleanly instead of desyncing h1 framing.
async fn drain_shed_body(mut request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    if let Some(stream) = request.stream.take() {
        let _ = stream.collect().await;
    }
    Ok(shed_response())
}

/// Dispatch one request through the Pipe chain, opening a span that
/// continues the inbound W3C trace when the request carried a `traceparent`
/// header — `proxima_telemetry::propagation::establish_trace_context` +
/// `RequestContext::adopt_trace_context` already restamped it onto
/// `request.context` at ingress (`build_proxima_request` / `build_streaming_request`
/// below) — or a fresh root otherwise. This is the seam
/// `examples/distributed_trace.rs` connects to h1's boundary.
#[proxima_telemetry::instrument(name = "h1_request", parent = request.context.traceparent(), err)]
pub(crate) async fn dispatch_request(
    dispatch: &PipeHandle,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    SendPipe::call(dispatch, request).await
}

pub async fn serve_h1_connection<Stream>(
    socket: Stream,
    dispatch: PipeHandle,
    max_body_bytes: Option<usize>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Result<(), ProximaError>
where
    Stream: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let spec = Arc::new(HttpListenerSpec { max_body_bytes });
    // One-off convenience entry point (no listener-wide capacity policy to
    // share) — unbounded admission, matching this function's pre-existing
    // behavior of never shedding on capacity, while still gaining a real
    // `request_admit`/`request_release` handshake instead of a bespoke
    // counter.
    let admission = ConnAdmission::unbounded();
    let quiesce_response = Arc::new(QuiesceResponse {
        status: 503,
        retry_after: "1".into(),
    });
    serve_connection(
        socket,
        dispatch,
        spec,
        admission,
        quiesce_response,
        None,
        runtime,
    )
    .await
}

/// Outcome of the hand-rolled biased poll racing the buffered-path
/// dispatch future against socket-EOF detection in
/// [`serve_connection`] — the tokio-free equivalent of
/// `tokio::select!`.
enum DispatchRace {
    Response(Result<Response<Bytes>, ProximaError>),
    Read(std::io::Result<usize>),
}

/// Outcome of the hand-rolled biased poll racing a streaming-path
/// pending response against socket-EOF detection in
/// [`dispatch_streaming_request`]. `Response(None)` only ever comes
/// from the runtime-spawned arm (the spawned task dropped its
/// `oneshot` sender without replying); the in-task-joined arm always
/// resolves to `Some`.
enum StreamDispatchRace {
    Response(Option<Result<Response<Bytes>, ProximaError>>),
    Read(std::io::Result<usize>),
}

/// `http1::Builder::serve_connection`. Uses our `Connection` state
/// machine: zero-copy parse into the connection buffer, alloc-free
/// hot path, per-connection bump arena for the Pipe-boundary
/// conversion.
// per-connection serve loop threads listener-scoped state plus the
// runtime handle used to dispatch the streaming Pipe::call
#[allow(clippy::too_many_arguments)]
pub async fn serve_connection<Stream>(
    socket: Stream,
    dispatch: PipeHandle,
    spec: Arc<HttpListenerSpec>,
    admission: ConnAdmission,
    quiesce_response: Arc<QuiesceResponse>,
    peer: Option<proxima_primitives::stream::PeerInfo>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Result<(), ProximaError>
where
    Stream: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (mut reader, mut writer) = socket.split();
    let mut connection = Connection::new();
    // Auto-stream: chunked or Content-Length > 1 MiB jumps to the
    // streaming dispatch path; smaller bodies stay on the buffered
    // single-`RequestReady` path. Keeps small-body latency low while
    // making large uploads and chunked ingest memory-safe.
    connection.set_auto_stream_policy(Some(AutoStreamPolicy::default()));
    // Stack-buffered read slot; 16 KB is enough for a typical
    // request head + small body in one syscall.
    let mut read_buf = [0_u8; 16 * 1024];
    // Single response output buffer reused across all responses on
    // this connection — cleared between writes; never reallocated
    // once big enough for the largest response head + chunk.
    let mut out = Vec::with_capacity(8 * 1024);

    loop {
        // Drain everything we can from the current buffered bytes
        // before pulling more from the socket. `advance` returns
        // typed handles per state so gated accessors (body, head,
        // trailers) can only be reached when valid.
        let outcome = match connection.advance() {
            Ok(outcome) => outcome,
            Err(read_error) => {
                error!(?read_error, "request parse error; closing connection");
                let status = 400_u16;
                write_minimal_error(&mut writer, &mut out, status, "Bad Request").await?;
                return Ok(());
            }
        };

        match outcome {
            Advanced::Close => return Ok(()),
            Advanced::NeedInput => {
                let read = reader.read(&mut read_buf).await.map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!("http read: {err}")))
                })?;
                if read == 0 {
                    // peer closed cleanly; we're done.
                    return Ok(());
                }
                connection.feed_bytes(&read_buf[..read]);
                continue;
            }
            Advanced::RequestReady(_request) => {
                // Drop the handle and let the post-match dispatch flow
                // operate on the connection directly. Handle-mediated
                // access stays available for early gating (413, quiesce);
                // the post-match flow needs raw mutability to dispatch
                // and write the response.
            }
            Advanced::HeadReady(_head) => {
                // _head drops at end of arm scope, releasing the
                // connection borrow before dispatch_streaming_request
                // takes &mut connection.
                let outcome = dispatch_streaming_request(
                    &mut connection,
                    &mut reader,
                    &mut writer,
                    &mut read_buf,
                    &mut out,
                    &dispatch,
                    &spec,
                    &admission,
                    &quiesce_response,
                    &peer,
                    runtime.as_ref(),
                )
                .await;
                match outcome {
                    Ok(StreamingOutcome::KeepAlive) => {
                        connection.reset_for_next_request();
                        continue;
                    }
                    Ok(StreamingOutcome::Close) => return Ok(()),
                    Ok(StreamingOutcome::Upgrade(handler)) => {
                        return invoke_upgrade(handler, reader, writer, &mut connection).await;
                    }
                    Err(error) => return Err(error),
                }
            }
            Advanced::BodyChunk(_) | Advanced::BodyEnd(_) => {
                error!("streaming poll variant leaked into outer loop");
                return Ok(());
            }
            Advanced::Expect100Continue(gate) => {
                // RFC 7231 §5.1.1: pre-screen via Content-Length before
                // signaling the client to send the body. Typed gate gives
                // us inspect-then-accept-or-reject in one consume-on-decision
                // shape.
                let content_length = gate.content_length();
                if let Some(limit) = spec.max_body_bytes
                    && let Some(length) = content_length
                    && length > limit as u64
                {
                    error!(
                        content_length = length,
                        limit, "Expect: 100-continue rejected (limit)"
                    );
                    let message =
                        Bytes::from(format!("request body exceeds limit ({length} > {limit})"));
                    let response_headers =
                        vec![("content-type".to_string(), "text/plain".to_string())];
                    out.clear();
                    let response_writer = gate.reject(
                        413,
                        "Payload Too Large",
                        &response_headers,
                        BodyFraming::ContentLength(message.len() as u64),
                        &mut out,
                    );
                    response_writer.write_chunk(&message, &mut out);
                    response_writer.end_response(&mut out);
                    writer.write_all(&out).await.map_err(io_err)?;
                    return Ok(());
                }
                if admission.is_quiescing() {
                    let quiesce_headers = vec![
                        (
                            "Retry-After".to_string(),
                            quiesce_response.retry_after.clone(),
                        ),
                        ("X-Proxima-Quiescing".to_string(), "1".to_string()),
                    ];
                    out.clear();
                    let response_writer = gate.reject(
                        quiesce_response.status,
                        "",
                        &quiesce_headers,
                        BodyFraming::ContentLength(9),
                        &mut out,
                    );
                    response_writer.write_chunk(b"quiescing", &mut out);
                    response_writer.end_response(&mut out);
                    writer.write_all(&out).await.map_err(io_err)?;
                    return Ok(());
                }
                // Accept: emit 100 Continue (consumes gate), loop back
                // to advance — next iteration delivers HeadReady or
                // RequestReady so dispatch proceeds.
                out.clear();
                gate.accept(&mut out);
                writer.write_all(&out).await.map_err(io_err)?;
                continue;
            }
        }

        // Quiescing — refuse the request with the configured status
        // and a Retry-After header. Don't dispatch; close connection
        // after writing the response.
        if admission.is_quiescing() {
            let quiesce_headers = vec![
                (
                    "Retry-After".to_string(),
                    quiesce_response.retry_after.clone(),
                ),
                ("X-Proxima-Quiescing".to_string(), "1".to_string()),
            ];
            out.clear();
            let response_writer = connection.begin_response(
                quiesce_response.status,
                "",
                &quiesce_headers,
                BodyFraming::ContentLength(9),
                &mut out,
            );
            response_writer.write_chunk(b"quiescing", &mut out);
            response_writer.end_response(&mut out);
            writer.write_all(&out).await.map_err(io_err)?;
            return Ok(());
        }

        // max_body_bytes enforcement: reject 413 if the buffered body
        // exceeds the configured limit. Connection has already read the
        // bytes (it streams to End before returning RequestReady) — the
        // budget here protects downstream Pipes from over-large
        // payloads. Tighter at-arrival enforcement would require
        // capping Connection's read budget before parse completes; a
        // follow-up when streaming-body dispatch lands.
        if let Some(limit) = spec.max_body_bytes
            && connection.body().len() > limit
        {
            let body_len = connection.body().len();
            error!(body_len, limit, "request body exceeds limit");
            let message = Bytes::from(format!("request body exceeds limit ({body_len} > {limit})"));
            let response_headers = vec![("content-type".to_string(), "text/plain".to_string())];
            out.clear();
            let response_writer = connection.begin_response(
                413,
                "Payload Too Large",
                &response_headers,
                BodyFraming::ContentLength(message.len() as u64),
                &mut out,
            );
            response_writer.write_chunk(&message, &mut out);
            response_writer.end_response(&mut out);
            writer.write_all(&out).await.map_err(io_err)?;
            return Ok(());
        }

        // Request-level admission: the request-count twin of the
        // connection-level `ListenerCore` gate, same handshake h2 native
        // uses at its own per-stream boundary
        // (`proxima-http/src/http2/server.rs`'s `RequestAdmit` match).
        // `Shed` never dispatches to the Pipe and never touches
        // `cancel`/`in_flight` bookkeeping (there was nothing to admit);
        // the shed response flows through the SAME tail below as any
        // other outcome, so write framing, keep-alive, and upgrade
        // handling stay identical for both.
        let mut trace_id: Option<Arc<[u8]>> = None;
        let outcome: Result<Response<Bytes>, ProximaError> = match admission.request_admit() {
            RequestAdmit::Shed { reason } => {
                debug!(?reason, "h1 request shed by listener admission");
                Ok(shed_response())
            }
            RequestAdmit::Admit => {
                let mut admission_guard =
                    RequestAdmissionGuard::new(&admission, RequestAdmit::Admit, "buffered");
                // Build the proxima::Request from Connection accessors and
                // dispatch through the Pipe chain. One Bytes::copy_from_slice
                // for each of: method, path, body. Headers are copied into a
                // HeaderList — Tier 3 (the bump arena) would absorb this if we
                // moved Request fields off owned Bytes, but the existing
                // proxima::Request shape uses Bytes today so we pay it for now.
                let cancel = proxima_core::signal::Signal::new();
                let cancel_guard = cancel.clone().guard();
                let mut request = build_proxima_request(&mut connection, &spec);
                request.context.cancel = cancel.clone();
                request.context.peer = peer.clone();
                trace_id = request.context.trace_id.clone();
                admission_guard.trace_id = trace_id.clone();
                let telemetry = request.context.telemetry.clone();
                telemetry.gauge_set(
                    "proxima.requests.in_flight",
                    &proxima_primitives::pipe::telemetry_surface::Labels::empty(),
                    admission.in_flight() as i64,
                );
                // Race the dispatch against socket EOF detection. If the
                // client disconnects mid-dispatch, the read returns 0; we
                // fire `cancel.fire()` and then poll the dispatch future
                // to completion so the Pipe has a chance to observe the
                // cancellation and clean up. Hand-rolled biased poll —
                // tokio-free equivalent of `tokio::select!` (see
                // `http2::server::SelectOutcome`'s doc for the pattern).
                let mut dispatch_future = pin!(dispatch_request(&dispatch, request));
                let mut watch_buf = [0_u8; 1];
                let race = poll_fn(|cx| {
                    if let Poll::Ready(response) = dispatch_future.as_mut().poll(cx) {
                        return Poll::Ready(DispatchRace::Response(response));
                    }
                    if let Poll::Ready(read) = Pin::new(&mut reader).poll_read(cx, &mut watch_buf) {
                        return Poll::Ready(DispatchRace::Read(read));
                    }
                    Poll::Pending
                })
                .await;
                let race_read_error = match &race {
                    DispatchRace::Read(Err(error)) => Some(error),
                    _ => None,
                };
                debug!(
                    route = "buffered",
                    trace_id = ?trace_id,
                    lifecycle_phase = "body_complete_response_wait",
                    race_arm = match &race {
                        DispatchRace::Response(_) => "dispatch_response",
                        DispatchRace::Read(Ok(0)) => "read_eof",
                        DispatchRace::Read(Ok(_)) => "read_bytes",
                        DispatchRace::Read(Err(_)) => "read_error",
                    },
                    read_result = match &race {
                        DispatchRace::Response(_) => "not_selected",
                        DispatchRace::Read(Ok(0)) => "eof",
                        DispatchRace::Read(Ok(_)) => "bytes",
                        DispatchRace::Read(Err(_)) => "error",
                    },
                    read_bytes = match &race {
                        DispatchRace::Read(Ok(count)) => *count as u64,
                        _ => 0,
                    },
                    read_error = ?race_read_error,
                    cancel_fired = false,
                    dispatch_state = match &race {
                        DispatchRace::Response(_) => "ready",
                        DispatchRace::Read(_) => "pending",
                    },
                    admission_state = "admitted",
                    response_writer_entered = false,
                    "h1 request race selected an arm"
                );
                let result = match race {
                    DispatchRace::Response(response) => response,
                    DispatchRace::Read(Ok(0)) => {
                        debug!(
                            route = "buffered",
                            trace_id = ?trace_id,
                            lifecycle_phase = "body_complete_response_wait",
                            race_arm = "read_eof",
                            read_result = "eof",
                            cancel_fired = false,
                            dispatch_state = "awaiting_response_after_peer_write_eof",
                            admission_state = "admitted",
                            response_writer_entered = false,
                            "h1 request read eof retained the response task"
                        );
                        dispatch_future.await
                    }
                    DispatchRace::Read(Ok(_n)) => {
                        // Pipelined bytes from the client. Buffer
                        // the byte by feeding it back into
                        // Connection; the next iteration's poll
                        // picks up the trailing request after this
                        // one's response is written.
                        connection.feed_bytes(&watch_buf[..1]);
                        dispatch_future.await
                    }
                    DispatchRace::Read(Err(error)) => {
                        debug!(?error, "read error during dispatch");
                        cancel.fire();
                        debug!(
                            route = "buffered",
                            trace_id = ?trace_id,
                            lifecycle_phase = "body_complete_response_wait",
                            race_arm = "read_error",
                            read_result = "error",
                            read_error = ?error,
                            cancel_fired = true,
                            dispatch_state = "pending",
                            admission_state = "admitted",
                            response_writer_entered = false,
                            "h1 dispatch cancellation fired after read error"
                        );
                        let dispatch_result = dispatch_future.await;
                        debug!(
                            route = "buffered",
                            trace_id = ?trace_id,
                            lifecycle_phase = "body_complete_response_wait",
                            read_result = "error",
                            cancel_fired = true,
                            dispatch_state = match dispatch_result {
                                Ok(_) => "completed_after_cancel",
                                Err(_) => "failed_after_cancel",
                            },
                            admission_state = "admitted",
                            response_writer_entered = false,
                            "h1 buffered dispatch observed read-error cancellation"
                        );
                        admission_guard.release("read_error_terminal");
                        return Ok(());
                    }
                };
                admission_guard.release("response_resolved_before_writer");
                // Successful response — disarm the cancel guard so a normal
                // return doesn't fire it.
                cancel_guard.disarm();
                result
            }
        };

        let upgrade_after_write = match outcome {
            Ok(response) => {
                write_response(
                    &mut writer,
                    &mut out,
                    &mut connection,
                    response,
                    trace_id.as_deref(),
                    "buffered",
                )
                .await?
            }
            Err(error) => {
                error!(?error, "request handling failed");
                let status = http_status_for(&error);
                let body_bytes = error_response_body(&error);
                let body_len = body_bytes.len();
                let error_headers = vec![("content-type".to_string(), "text/plain".to_string())];
                out.clear();
                let response_writer = connection.begin_response(
                    status,
                    "",
                    &error_headers,
                    BodyFraming::ContentLength(body_len as u64),
                    &mut out,
                );
                response_writer.write_chunk(&body_bytes, &mut out);
                response_writer.end_response(&mut out);
                writer.write_all(&out).await.map_err(io_err)?;
                None
            }
        };

        if let Some(handler) = upgrade_after_write {
            // Listener cedes the socket to the upgrade handler. Any
            // bytes the client sent past the request head go with
            // it; the connection is at end-of-life.
            return invoke_upgrade(handler, reader, writer, &mut connection).await;
        }

        if connection.keep_alive() {
            connection.reset_for_next_request();
        } else {
            return Ok(());
        }
    }
}

/// Hand the socket halves to an upgrade handler. Reunites reader +
/// writer back into the original stream, drains any pipelined bytes
/// the listener buffered past the request head, and awaits the
/// handler to completion.
///
/// The framing layer holds the stream as `futures::io::AsyncRead +
/// AsyncWrite`; `HijackStream` is now keyed on the same `futures::io`
/// traits, so the reunited stream boxes directly with no compat wrap.
async fn invoke_upgrade<S>(
    handler: proxima_primitives::pipe::upgrade::UpgradeHandler,
    reader: futures::io::ReadHalf<S>,
    writer: futures::io::WriteHalf<S>,
    connection: &mut Connection,
) -> Result<(), ProximaError>
where
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let leftover = connection.drain_pipelined_bytes();
    let socket = reader
        .reunite(writer)
        .map_err(|_| ProximaError::Upstream("upgrade reunite mismatch".into()))?;
    let stream: Box<dyn proxima_primitives::pipe::upgrade::HijackStream> = Box::new(socket);
    let hijacked = proxima_primitives::pipe::upgrade::HijackedSocket::new(stream, leftover);
    handler.invoke(hijacked).await
}

enum StreamingOutcome {
    KeepAlive,
    Close,
    /// Streaming Pipe returned a Response with an `upgrade`
    /// handler attached. The streaming helper wrote the response
    /// head and surfaced the handler back to `serve_connection`,
    /// which owns the socket halves and can unsplit + hijack.
    Upgrade(proxima_primitives::pipe::upgrade::UpgradeHandler),
}

/// Outcome of the in-task join between the dispatch future and the
/// body pump — reached only when no `Runtime` is available to spawn
/// onto. The tokio-free replacement for `tokio::task::spawn_local`:
/// h1 carries exactly one request per connection, so a single named
/// dispatch future polled alongside the pump is enough — no
/// `FuturesUnordered` fan-out needed (contrast with h2's
/// per-stream `handlers` set in `http2::server`).
enum StreamJoinOutcome {
    DispatchDone(Result<Response<Bytes>, ProximaError>),
    PumpDone(Result<(), PumpError>),
}

/// Publish chunked request trailers into the stream's slot. Called
/// after the body pump finishes, before the channel closes, so the
/// Pipe's drain completes only after the trailers are visible
/// (deterministic — no read-before-publish race).
fn publish_pending_trailers(
    connection: &mut Connection,
    trailers_slot: &proxima_primitives::pipe::body::TrailersSlot,
) {
    let captured_trailers = connection.take_trailers();
    if captured_trailers.is_empty() {
        return;
    }
    let mut trailers = proxima_primitives::pipe::header_list::HeaderList::new();
    for (name, value) in captured_trailers {
        trailers.insert(name, value);
    }
    *trailers_slot.lock() = Some(trailers);
}

/// Write the streaming path's final response (or rendered error) and
/// decide keep-alive vs. close. Shared tail for both the
/// runtime-spawned and in-task-joined dispatch paths in
/// [`dispatch_streaming_request`] — everything upstream of this
/// differs only in HOW the response was obtained, never in how it's
/// written.
async fn finish_streaming_response<W>(
    writer: &mut W,
    out: &mut Vec<u8>,
    connection: &mut Connection,
    trace_id: Option<&[u8]>,
    route: &'static str,
    response_result: Result<Response<Bytes>, ProximaError>,
) -> Result<StreamingOutcome, ProximaError>
where
    W: AsyncWrite + Unpin,
{
    let upgrade_after_write = match response_result {
        Ok(response) => {
            write_response(writer, out, connection, response, trace_id, route).await?
        }
        Err(error) => {
            error!(?error, "streaming request handling failed");
            let status = http_status_for(&error);
            let body_bytes = error_response_body(&error);
            let body_len = body_bytes.len();
            let error_headers = vec![("content-type".to_string(), "text/plain".to_string())];
            out.clear();
            let response_writer = connection.begin_response(
                status,
                "",
                &error_headers,
                BodyFraming::ContentLength(body_len as u64),
                out,
            );
            response_writer.write_chunk(&body_bytes, out);
            response_writer.end_response(out);
            writer.write_all(out).await.map_err(io_err)?;
            None
        }
    };

    if let Some(handler) = upgrade_after_write {
        return Ok(StreamingOutcome::Upgrade(handler));
    }
    if connection.keep_alive() {
        Ok(StreamingOutcome::KeepAlive)
    } else {
        Ok(StreamingOutcome::Close)
    }
}

/// Streaming-mode dispatch path. Reached when the connection's
/// auto-stream policy fires (chunked transfer or large
/// Content-Length). Builds an mpsc-backed `Body`; with a `Runtime`,
/// spawns `Pipe::call` on the current core so it runs concurrently
/// with body ingestion (mirrors h2/h3, no tokio involved); with no
/// runtime, drives the dispatch future and the body pump together
/// in this task via a hand-rolled biased poll (tokio-free — no
/// `LocalSet` required). Pumps body chunks into the channel
/// (backpressure: `send().await` parks when full), and on `BodyEnd`
/// drops the sender to signal end-of-stream. Then awaits the
/// response, racing it against a single-byte reader.read for
/// client-disconnect detection (mirrors the buffered path's
/// cancel-token race).
// the streaming dispatch needs the connection, both socket halves, the
// request, and the listener-scoped admission/quiesce state in one frame.
#[allow(clippy::too_many_arguments)]
async fn dispatch_streaming_request<R, W>(
    connection: &mut Connection,
    reader: &mut R,
    writer: &mut W,
    read_buf: &mut [u8],
    out: &mut Vec<u8>,
    dispatch: &PipeHandle,
    spec: &Arc<HttpListenerSpec>,
    admission: &ConnAdmission,
    quiesce_response: &Arc<QuiesceResponse>,
    peer: &Option<proxima_primitives::stream::PeerInfo>,
    runtime: Option<&Arc<dyn Runtime>>,
) -> Result<StreamingOutcome, ProximaError>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    // Quiescing — refuse with the configured status WITHOUT consuming
    // the body. Closes the connection after writing; the half-uploaded
    // body bytes drain via TCP RST when we close.
    if admission.is_quiescing() {
        let quiesce_headers = vec![
            (
                "Retry-After".to_string(),
                quiesce_response.retry_after.clone(),
            ),
            ("X-Proxima-Quiescing".to_string(), "1".to_string()),
        ];
        out.clear();
        let response_writer = connection.begin_response(
            quiesce_response.status,
            "",
            &quiesce_headers,
            BodyFraming::ContentLength(9),
            out,
        );
        response_writer.write_chunk(b"quiescing", out);
        response_writer.end_response(out);
        writer.write_all(out).await.map_err(io_err)?;
        return Ok(StreamingOutcome::Close);
    }

    // Pre-check Content-Length against max_body_bytes when the
    // header is present. Avoids letting a hostile client tie up a
    // mpsc + spawned task just to overshoot the limit.
    if let Some(limit) = spec.max_body_bytes
        && let Some(content_length) = parse_content_length(connection)
        && content_length > limit as u64
    {
        error!(content_length, limit, "request body exceeds limit");
        let message = Bytes::from(format!(
            "request body exceeds limit ({content_length} > {limit})"
        ));
        let response_headers = vec![("content-type".to_string(), "text/plain".to_string())];
        out.clear();
        let response_writer = connection.begin_response(
            413,
            "Payload Too Large",
            &response_headers,
            BodyFraming::ContentLength(message.len() as u64),
            out,
        );
        response_writer.write_chunk(&message, out);
        response_writer.end_response(out);
        writer.write_all(out).await.map_err(io_err)?;
        return Ok(StreamingOutcome::Close);
    }

    // Request-level admission: same handshake as the buffered path
    // above and h2 native's per-stream boundary. The body pump below
    // must run regardless of the outcome — the client is already
    // sending (or about to send) body bytes on the wire, and those
    // bytes have to be drained to keep h1 framing in sync even when
    // shed. `RequestAdmit` is `Copy`; `admission_outcome` is read again
    // inside `pipe_task`/`dispatch_future` below to pick real dispatch
    // vs. drain-and-shed.
    let admission_outcome = admission.request_admit();
    let cancel = proxima_core::signal::Signal::new();

    let (mut body_tx, body_rx) = mpsc::channel::<Result<Bytes, ProximaError>>(BODY_CHANNEL_DEPTH);

    let (mut request, request_trailers_slot) = build_streaming_request(connection, spec, body_rx);
    request.context.cancel = cancel.clone();
    request.context.peer = peer.clone();
    let trace_id = request.context.trace_id.clone();
    let route = if runtime.is_some() {
        "streaming_some_runtime"
    } else {
        "streaming_no_runtime"
    };
    let mut admission_guard = RequestAdmissionGuard::new(admission, admission_outcome, route);
    admission_guard.trace_id = trace_id.clone();
    let cancel_guard = cancel.clone().guard();
    match admission_outcome {
        RequestAdmit::Admit => {
            let telemetry = request.context.telemetry.clone();
            telemetry.gauge_set(
                "proxima.requests.in_flight",
                &proxima_primitives::pipe::telemetry_surface::Labels::empty(),
                admission.in_flight() as i64,
            );
        }
        RequestAdmit::Shed { reason } => {
            debug!(?reason, "h1 streaming request shed by listener admission");
        }
    }

    let dispatch_clone = dispatch.clone();

    match runtime {
        Some(rt) => {
            // Spawns the Pipe onto the runtime's current-core queue,
            // so it runs concurrently with the body pump below (the
            // Pipe drains `body_rx` as fast as the pump fills it,
            // bounded backpressure both ways). Mirrors h2/h3 — no
            // tokio involved, `Runtime` is the core's own
            // executor handle.
            let (resp_tx, mut resp_rx) =
                oneshot::channel::<Result<Response<Bytes>, ProximaError>>();
            let pipe_task = async move {
                let response = match admission_outcome {
                    RequestAdmit::Admit => dispatch_request(&dispatch_clone, request).await,
                    RequestAdmit::Shed { .. } => drain_shed_body(request).await,
                };
                let _ = resp_tx.send(response);
            };
            rt.spawn_on_current_core(Box::pin(pipe_task));

            let pump_outcome = pump_body_stream(
                connection,
                reader,
                read_buf,
                &mut body_tx,
                &cancel,
                "streaming_some_runtime",
                trace_id.as_deref(),
                admission_outcome,
            )
            .await;
            publish_pending_trailers(connection, &request_trailers_slot);
            drop(body_tx);

            match pump_outcome {
                Ok(()) | Err(PumpError::PipeDroppedBody) => {}
                Err(PumpError::ClientEof | PumpError::ConnectionClosed | PumpError::Cancelled) => {
                    cancel.fire();
                    // Give the Pipe a poll cycle to observe cancel
                    // before dropping its task — mirrors the
                    // buffered path's dispatch_future-then-poll
                    // pattern.
                    let response_wait = resp_rx.await;
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_incomplete",
                        read_result = match &pump_outcome {
                            Err(PumpError::ClientEof) => "eof",
                            Err(PumpError::ConnectionClosed) => "connection_closed",
                            Err(PumpError::Cancelled) => "cancelled",
                            _ => "other",
                        },
                        cancel_fired = true,
                        dispatch_state = match response_wait {
                            Ok(_) => "completed_after_cancel",
                            Err(_) => "sender_dropped_after_cancel",
                        },
                        admission_state = "admitted",
                        response_writer_entered = false,
                        "h1 streaming request cancellation completed before body end"
                    );
                    admission_guard.release("body_incomplete_terminal");
                    return Ok(StreamingOutcome::Close);
                }
                Err(PumpError::Decode) => {
                    cancel.fire();
                    let _ = resp_rx.await;
                    admission_guard.release("body_decode_error");
                    write_minimal_error(writer, out, 400, "Bad Request").await?;
                    return Ok(StreamingOutcome::Close);
                }
                Err(PumpError::Io(error)) => {
                    cancel.fire();
                    let response_wait = resp_rx.await;
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_incomplete",
                        read_result = "error",
                        cancel_fired = true,
                        dispatch_state = match &response_wait {
                            Ok(Ok(_)) => "completed_after_cancel",
                            Ok(Err(_)) => "failed_after_cancel",
                            Err(_) => "sender_dropped_after_cancel",
                        },
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming dispatch observed body-read-error cancellation"
                    );
                    admission_guard.release("body_read_error");
                    return Err(ProximaError::Io(error));
                }
            }

            // Response phase — race resp_rx against socket EOF
            // detection so a disconnect during a slow Pipe still
            // triggers cancel and tears down cleanly. Hand-rolled
            // biased poll, the tokio-free equivalent of
            // `tokio::select!`.
            let mut watch_buf = [0_u8; 1];
            let race = poll_fn(|cx| {
                if let Poll::Ready(response) = Pin::new(&mut resp_rx).poll(cx) {
                    return Poll::Ready(StreamDispatchRace::Response(response.ok()));
                }
                if let Poll::Ready(read) = Pin::new(&mut *reader).poll_read(cx, &mut watch_buf) {
                    return Poll::Ready(StreamDispatchRace::Read(read));
                }
                Poll::Pending
            })
            .await;
            let race_read_error = match &race {
                StreamDispatchRace::Read(Err(error)) => Some(error),
                _ => None,
            };
            debug!(
                route = "streaming_some_runtime",
                trace_id = ?trace_id,
                lifecycle_phase = "body_complete_response_wait",
                race_arm = match &race {
                    StreamDispatchRace::Response(None) => "response_sender_dropped",
                    StreamDispatchRace::Response(Some(_)) => "response_ready",
                    StreamDispatchRace::Read(Ok(0)) => "read_eof",
                    StreamDispatchRace::Read(Ok(_)) => "read_bytes",
                    StreamDispatchRace::Read(Err(_)) => "read_error",
                },
                read_result = match &race {
                    StreamDispatchRace::Response(_) => "not_selected",
                    StreamDispatchRace::Read(Ok(0)) => "eof",
                    StreamDispatchRace::Read(Ok(_)) => "bytes",
                    StreamDispatchRace::Read(Err(_)) => "error",
                },
                read_bytes = match &race {
                    StreamDispatchRace::Read(Ok(count)) => *count as u64,
                    _ => 0,
                },
                read_error = ?race_read_error,
                cancel_fired = false,
                dispatch_state = match &race {
                    StreamDispatchRace::Response(Some(_)) => "ready",
                    StreamDispatchRace::Response(None) => "sender_dropped",
                    StreamDispatchRace::Read(_) => "pending",
                },
                admission_state = match admission_outcome {
                    RequestAdmit::Admit => "admitted",
                    RequestAdmit::Shed { .. } => "shed",
                },
                response_writer_entered = false,
                "h1 streaming response race selected an arm"
            );
            let response_result = match race {
                StreamDispatchRace::Response(None) => {
                    error!("streaming dispatch task dropped response sender");
                    admission_guard.release("response_sender_dropped");
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_complete_response_wait",
                        race_arm = "response_sender_dropped",
                        read_result = "not_selected",
                        cancel_fired = false,
                        dispatch_state = "sender_dropped",
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming response sender dropped before response"
                    );
                    return Ok(StreamingOutcome::Close);
                }
                StreamDispatchRace::Response(Some(response)) => response,
                StreamDispatchRace::Read(Ok(0)) => {
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_complete_response_wait",
                        race_arm = "read_eof",
                        read_result = "eof",
                        cancel_fired = false,
                        dispatch_state = "awaiting_response_after_peer_write_eof",
                        admission_state = "admitted",
                        response_writer_entered = false,
                        "h1 streaming request read eof retained the response task"
                    );
                    match resp_rx.await {
                        Ok(response) => response,
                        Err(_canceled) => {
                            admission_guard.release("response_sender_dropped_after_eof");
                            error!("streaming dispatch task dropped response sender after read eof");
                            return Ok(StreamingOutcome::Close);
                        }
                    }
                }
                StreamDispatchRace::Read(Ok(_n)) => {
                    // First byte of a pipelined request — queue it.
                    connection.feed_bytes(&watch_buf[..1]);
                    match resp_rx.await {
                        Ok(response) => response,
                        Err(_canceled) => {
                            admission_guard.release("response_sender_dropped_after_pipeline_byte");
                            error!("streaming dispatch task dropped response sender");
                            return Ok(StreamingOutcome::Close);
                        }
                    }
                }
                StreamDispatchRace::Read(Err(error)) => {
                    debug!(?error, "read error during streaming dispatch");
                    cancel.fire();
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_complete_response_wait",
                        race_arm = "read_error",
                        read_result = "error",
                        read_error = ?error,
                        cancel_fired = true,
                        dispatch_state = "pending",
                        admission_state = "admitted",
                        response_writer_entered = false,
                        "h1 streaming cancellation fired after read error"
                    );
                    let response_wait = resp_rx.await;
                    debug!(
                        route = "streaming_some_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_complete_response_wait",
                        read_result = "error",
                        cancel_fired = true,
                        dispatch_state = match &response_wait {
                            Ok(Ok(_)) => "completed_after_cancel",
                            Ok(Err(_)) => "failed_after_cancel",
                            Err(_) => "sender_dropped_after_cancel",
                        },
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming dispatch observed response-read-error cancellation"
                    );
                    admission_guard.release("response_read_error_terminal");
                    return Ok(StreamingOutcome::Close);
                }
            };
            admission_guard.release("response_resolved_before_writer");
            cancel_guard.disarm();
            finish_streaming_response(
                writer,
                out,
                connection,
                trace_id.as_deref(),
                "streaming_some_runtime",
                response_result,
            )
            .await
        }
        None => {
            // No executor to spawn onto — drive the dispatch future
            // and the body pump together in this task via a
            // hand-rolled biased poll (the tokio-free replacement
            // for `tokio::task::spawn_local`). Same concurrency as
            // the spawned-task arm above: the Pipe drains `body_rx`
            // as the pump fills it.
            let mut dispatch_future = pin!(async move {
                match admission_outcome {
                    RequestAdmit::Admit => dispatch_request(&dispatch_clone, request).await,
                    RequestAdmit::Shed { .. } => drain_shed_body(request).await,
                }
            });
            let mut dispatch_done: Option<Result<Response<Bytes>, ProximaError>> = None;
            let pump_outcome = {
                let mut pump_future = pin!(pump_body_stream(
                    connection,
                    reader,
                    read_buf,
                    &mut body_tx,
                        &cancel,
                        "streaming_no_runtime",
                        trace_id.as_deref(),
                        admission_outcome,
                ));
                // `dispatch_pending` (a plain bool, checked OUTSIDE the
                // poll_fn closure) gates whether the closure below even
                // touches `dispatch_future` — reading `dispatch_done`
                // from inside the closure would force `Response<Bytes>`
                // (it carries a `Box<dyn Stream + Send>`, not `Sync`) to
                // be `Sync` for the closure to stay `Send`. A bool has
                // no such bound.
                let mut dispatch_pending = true;
                loop {
                    let outcome = if dispatch_pending {
                        poll_fn(|cx| {
                            if let Poll::Ready(response) = dispatch_future.as_mut().poll(cx) {
                                return Poll::Ready(StreamJoinOutcome::DispatchDone(response));
                            }
                            if let Poll::Ready(result) = pump_future.as_mut().poll(cx) {
                                return Poll::Ready(StreamJoinOutcome::PumpDone(result));
                            }
                            Poll::Pending
                        })
                        .await
                    } else {
                        StreamJoinOutcome::PumpDone(
                            poll_fn(|cx| pump_future.as_mut().poll(cx)).await,
                        )
                    };
                    match outcome {
                        StreamJoinOutcome::DispatchDone(response) => {
                            dispatch_done = Some(response);
                            dispatch_pending = false;
                        }
                        StreamJoinOutcome::PumpDone(result) => break result,
                    }
                }
            };
            publish_pending_trailers(connection, &request_trailers_slot);
            drop(body_tx);

            match pump_outcome {
                Ok(()) | Err(PumpError::PipeDroppedBody) => {}
                Err(PumpError::ClientEof | PumpError::ConnectionClosed | PumpError::Cancelled) => {
                    cancel.fire();
                    let dispatch_was_ready = dispatch_done.is_some();
                    let dispatch_state = if dispatch_was_ready {
                        "ready_before_cancel"
                    } else {
                        match dispatch_future.await {
                            Ok(_) => "completed_after_cancel",
                            Err(_) => "failed_after_cancel",
                        }
                    };
                    debug!(
                        route = "streaming_no_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_incomplete",
                        read_result = match &pump_outcome {
                            Err(PumpError::ClientEof) => "eof",
                            Err(PumpError::ConnectionClosed) => "connection_closed",
                            Err(PumpError::Cancelled) => "cancelled",
                            _ => "other",
                        },
                        cancel_fired = true,
                        dispatch_state = dispatch_state,
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming request cancellation completed before body end"
                    );
                    admission_guard.release("body_incomplete_terminal");
                    return Ok(StreamingOutcome::Close);
                }
                Err(PumpError::Decode) => {
                    cancel.fire();
                    if dispatch_done.is_none() {
                        let _ = dispatch_future.await;
                    }
                    admission_guard.release("body_decode_error");
                    write_minimal_error(writer, out, 400, "Bad Request").await?;
                    return Ok(StreamingOutcome::Close);
                }
                Err(PumpError::Io(error)) => {
                    cancel.fire();
                    let dispatch_state = match dispatch_done {
                        Some(Ok(_)) => "completed_before_cancel",
                        Some(Err(_)) => "failed_before_cancel",
                        None => match dispatch_future.await {
                            Ok(_) => "completed_after_cancel",
                            Err(_) => "failed_after_cancel",
                        },
                    };
                    debug!(
                        route = "streaming_no_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_incomplete",
                        read_result = "error",
                        cancel_fired = true,
                        dispatch_state = dispatch_state,
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming dispatch observed body-read-error cancellation"
                    );
                    admission_guard.release("body_read_error");
                    return Err(ProximaError::Io(error));
                }
            }

            let response_result = match dispatch_done {
                Some(response) => response,
                None => {
                    // Response phase — race the dispatch future
                    // against socket EOF detection, same shape as
                    // the spawned-task arm's resp_rx race.
                    let mut watch_buf = [0_u8; 1];
                    let race = poll_fn(|cx| {
                        if let Poll::Ready(response) = dispatch_future.as_mut().poll(cx) {
                            return Poll::Ready(DispatchRace::Response(response));
                        }
                        if let Poll::Ready(read) =
                            Pin::new(&mut *reader).poll_read(cx, &mut watch_buf)
                        {
                            return Poll::Ready(DispatchRace::Read(read));
                        }
                        Poll::Pending
                    })
                    .await;
                    let race_read_error = match &race {
                        DispatchRace::Read(Err(error)) => Some(error),
                        _ => None,
                    };
                    debug!(
                        route = "streaming_no_runtime",
                        trace_id = ?trace_id,
                        lifecycle_phase = "body_complete_response_wait",
                        race_arm = match &race {
                            DispatchRace::Response(_) => "dispatch_response",
                            DispatchRace::Read(Ok(0)) => "read_eof",
                            DispatchRace::Read(Ok(_)) => "read_bytes",
                            DispatchRace::Read(Err(_)) => "read_error",
                        },
                        read_result = match &race {
                            DispatchRace::Response(_) => "not_selected",
                            DispatchRace::Read(Ok(0)) => "eof",
                            DispatchRace::Read(Ok(_)) => "bytes",
                            DispatchRace::Read(Err(_)) => "error",
                        },
                        read_bytes = match &race {
                            DispatchRace::Read(Ok(count)) => *count as u64,
                            _ => 0,
                        },
                        read_error = ?race_read_error,
                        cancel_fired = false,
                        dispatch_state = match &race {
                            DispatchRace::Response(_) => "ready",
                            DispatchRace::Read(_) => "pending",
                        },
                        admission_state = match admission_outcome {
                            RequestAdmit::Admit => "admitted",
                            RequestAdmit::Shed { .. } => "shed",
                        },
                        response_writer_entered = false,
                        "h1 streaming response race selected an arm"
                    );
                    match race {
                        DispatchRace::Response(response) => response,
                        DispatchRace::Read(Ok(0)) => {
                            debug!(
                                route = "streaming_no_runtime",
                                trace_id = ?trace_id,
                                lifecycle_phase = "body_complete_response_wait",
                                race_arm = "read_eof",
                                read_result = "eof",
                                cancel_fired = false,
                                dispatch_state = "awaiting_response_after_peer_write_eof",
                                admission_state = match admission_outcome {
                                    RequestAdmit::Admit => "admitted",
                                    RequestAdmit::Shed { .. } => "shed",
                                },
                                response_writer_entered = false,
                                "h1 streaming request read eof retained the response task"
                            );
                            dispatch_future.await
                        }
                        DispatchRace::Read(Ok(_n)) => {
                            connection.feed_bytes(&watch_buf[..1]);
                            dispatch_future.await
                        }
                        DispatchRace::Read(Err(error)) => {
                            debug!(?error, "read error during streaming dispatch");
                            cancel.fire();
                            debug!(
                                route = "streaming_no_runtime",
                                trace_id = ?trace_id,
                                lifecycle_phase = "body_complete_response_wait",
                                race_arm = "read_error",
                                read_result = "error",
                                read_error = ?error,
                                cancel_fired = true,
                                dispatch_state = "pending",
                                admission_state = match admission_outcome {
                                    RequestAdmit::Admit => "admitted",
                                    RequestAdmit::Shed { .. } => "shed",
                                },
                                response_writer_entered = false,
                                "h1 streaming cancellation fired after read error"
                            );
                            let dispatch_result = dispatch_future.await;
                            debug!(
                                route = "streaming_no_runtime",
                                trace_id = ?trace_id,
                                lifecycle_phase = "body_complete_response_wait",
                                read_result = "error",
                                cancel_fired = true,
                                dispatch_state = match dispatch_result {
                                    Ok(_) => "completed_after_cancel",
                                    Err(_) => "failed_after_cancel",
                                },
                                admission_state = match admission_outcome {
                                    RequestAdmit::Admit => "admitted",
                                    RequestAdmit::Shed { .. } => "shed",
                                },
                                response_writer_entered = false,
                                "h1 streaming dispatch observed response-read-error cancellation"
                            );
                            admission_guard.release("response_read_error_terminal");
                            return Ok(StreamingOutcome::Close);
                        }
                    }
                }
            };
            admission_guard.release("response_resolved_before_writer");
            cancel_guard.disarm();
            finish_streaming_response(
                writer,
                out,
                connection,
                trace_id.as_deref(),
                "streaming_no_runtime",
                response_result,
            )
            .await
        }
    }
}

#[derive(Debug)]
enum PumpError {
    /// Peer closed the connection before sending the full body.
    ClientEof,
    /// Pipe dropped its body receiver — not fatal, the Pipe
    /// may still return a valid response.
    PipeDroppedBody,
    /// Connection state machine reported terminal Close.
    ConnectionClosed,
    /// Cancellation token fired while sending.
    Cancelled,
    /// Body decoder rejected wire bytes (chunk size, framing, …).
    Decode,
    /// Socket read returned an IO error.
    Io(std::io::Error),
}

/// Outcome of the hand-rolled biased poll racing a body-chunk send
/// against cancellation in [`pump_body_stream`] — the tokio-free
/// equivalent of `tokio::select!`.
enum BodySendOutcome {
    Sent(Result<(), mpsc::SendError>),
    Cancelled,
}

/// Outcome of the hand-rolled biased poll racing a socket read
/// against cancellation in [`pump_body_stream`].
enum BodyReadOutcome {
    Read(std::io::Result<usize>),
    Cancelled,
}

/// Drive the connection's body decoder, copying each emitted chunk
/// into the bounded `body_tx`. Returns on `BodyEnd` (success) or on
/// any of the `PumpError` conditions. The caller is responsible for
/// dropping `body_tx` and reacting to the outcome.
async fn pump_body_stream<R>(
    connection: &mut Connection,
    reader: &mut R,
    read_buf: &mut [u8],
    body_tx: &mut mpsc::Sender<Result<Bytes, ProximaError>>,
    cancel: &proxima_core::signal::Signal,
    route: &'static str,
    trace_id: Option<&[u8]>,
    admission_outcome: RequestAdmit,
) -> Result<(), PumpError>
where
    R: AsyncRead + Unpin,
{
    loop {
        let outcome = match connection.poll() {
            Ok(value) => value,
            Err(read_error) => {
                let detail = format!("{read_error:?}");
                let _ = body_tx
                    .send(Err(ProximaError::Body(format!("decode: {detail}"))))
                    .await;
                return Err(PumpError::Decode);
            }
        };
        match outcome {
            ConnPoll::BodyChunk => {
                let chunk = match connection.take_body_chunk() {
                    Some(chunk) => chunk,
                    None => continue,
                };
                let mut send_future = pin!(body_tx.send(Ok(chunk)));
                let mut fired = cancel.fired();
                let send_outcome = poll_fn(|cx| {
                    if let Poll::Ready(result) = send_future.as_mut().poll(cx) {
                        return Poll::Ready(BodySendOutcome::Sent(result));
                    }
                    if let Poll::Ready(()) = Pin::new(&mut fired).poll(cx) {
                        return Poll::Ready(BodySendOutcome::Cancelled);
                    }
                    Poll::Pending
                })
                .await;
                match send_outcome {
                    BodySendOutcome::Sent(Ok(())) => {}
                    BodySendOutcome::Sent(Err(_send_error)) => {
                        return Err(PumpError::PipeDroppedBody);
                    }
                    BodySendOutcome::Cancelled => return Err(PumpError::Cancelled),
                }
            }
            ConnPoll::BodyEnd => {
                debug!(
                    route = route,
                    trace_id = ?trace_id,
                    lifecycle_phase = "body_complete_response_wait",
                    body_state = "body_end",
                    cancel_fired = false,
                    dispatch_state = "not_observed_by_body_pump",
                    admission_state = match admission_outcome {
                        RequestAdmit::Admit => "admitted",
                        RequestAdmit::Shed { .. } => "shed",
                    },
                    "h1 streaming request body reached end"
                );
                return Ok(());
            }
            ConnPoll::NeedInput => {
                let mut fired = cancel.fired();
                let read_outcome = poll_fn(|cx| {
                    if let Poll::Ready(read) = Pin::new(&mut *reader).poll_read(cx, read_buf) {
                        return Poll::Ready(BodyReadOutcome::Read(read));
                    }
                    if let Poll::Ready(()) = Pin::new(&mut fired).poll(cx) {
                        return Poll::Ready(BodyReadOutcome::Cancelled);
                    }
                    Poll::Pending
                })
                .await;
                match read_outcome {
                    BodyReadOutcome::Read(Ok(0)) => {
                        debug!(
                            route = route,
                            trace_id = ?trace_id,
                            lifecycle_phase = "body_incomplete",
                            read_result = "eof",
                            body_state = "need_input",
                            cancel_fired = false,
                            dispatch_state = "not_observed_by_body_pump",
                            admission_state = match admission_outcome {
                                RequestAdmit::Admit => "admitted",
                                RequestAdmit::Shed { .. } => "shed",
                            },
                            "h1 streaming request read eof before body end"
                        );
                        return Err(PumpError::ClientEof);
                    }
                    BodyReadOutcome::Read(Ok(n)) => connection.feed_bytes(&read_buf[..n]),
                    BodyReadOutcome::Read(Err(error)) => {
                        debug!(
                            route = route,
                            trace_id = ?trace_id,
                            lifecycle_phase = "body_incomplete",
                            read_result = "error",
                            read_error = ?error,
                            body_state = "need_input",
                            cancel_fired = false,
                            admission_state = match admission_outcome {
                                RequestAdmit::Admit => "admitted",
                                RequestAdmit::Shed { .. } => "shed",
                            },
                            "h1 streaming request read failed before body end"
                        );
                        return Err(PumpError::Io(error));
                    }
                    BodyReadOutcome::Cancelled => return Err(PumpError::Cancelled),
                }
            }
            ConnPoll::Close => return Err(PumpError::ConnectionClosed),
            ConnPoll::HeadReady | ConnPoll::RequestReady | ConnPoll::Expect100Continue => {
                // HeadReady / Expect100Continue are resolved by the
                // outer loop before pump starts; RequestReady never
                // appears in streaming mode. Anything reaching here
                // is a state-machine bug.
                return Err(PumpError::Decode);
            }
        }
    }
}

/// Build a `proxima::Request` for the streaming dispatch path. The
/// body is the bounded mpsc receiver itself — `futures::channel::mpsc`
/// receivers already implement `Stream`, so chunks flow from the
/// listener's pump loop to the Pipe as they arrive with no adapter.
fn build_streaming_request(
    connection: &Connection,
    _spec: &HttpListenerSpec,
    body_rx: mpsc::Receiver<Result<Bytes, ProximaError>>,
) -> (Request<Bytes>, proxima_primitives::pipe::body::TrailersSlot) {
    let path_bytes = connection.path();
    let (path, query) = split_path_and_query(path_bytes);
    let method = Method::from_bytes(connection.method());
    let mut headers = proxima_primitives::pipe::header_list::HeaderList::new();
    for header in connection.headers() {
        headers.insert(
            Bytes::copy_from_slice(header.name()),
            Bytes::copy_from_slice(header.value()),
        );
    }
    let mut context = RequestContext::default();
    let (trace_id, baggage) = proxima_telemetry::propagation::establish_trace_context(&headers);
    context.adopt_trace_context(trace_id, baggage);
    // Trailers slot the chunked decoder populates at body-end; the Pipe's
    // `body_bytes()` folds it into `headers` after draining (RFC 7230
    // §4.1.2 request trailers on the streaming path).
    let trailers_slot: proxima_primitives::pipe::body::TrailersSlot =
        Arc::new(proxima_primitives::sync::blocking::Mutex::new(None));
    let request = Request {
        method,
        path,
        query,
        metadata: headers,
        payload: Bytes::new(),
        stream: Some(RequestStream::new(body_rx).with_trailers_slot(trailers_slot.clone())),
        context,
    };
    (request, trailers_slot)
}

fn parse_content_length(connection: &Connection) -> Option<u64> {
    let value = connection.header_value(b"content-length")?;
    let text = std::str::from_utf8(value).ok()?;
    text.trim().parse::<u64>().ok()
}

fn io_err(error: std::io::Error) -> ProximaError {
    ProximaError::Io(std::io::Error::other(format!("http write: {error}")))
}

async fn write_minimal_error<W>(
    writer: &mut W,
    out: &mut Vec<u8>,
    status: u16,
    body: &str,
) -> Result<(), ProximaError>
where
    W: AsyncWrite + Unpin,
{
    out.clear();
    out.extend_from_slice(b"HTTP/1.1 ");
    let _ = std::io::Write::write_fmt(out, format_args!("{status} "));
    out.extend_from_slice(body.as_bytes());
    out.extend_from_slice(b"\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
    writer.write_all(out).await.map_err(io_err)
}

/// Convert a `Connection`'s parsed head + buffered body into a
/// `proxima::Request`. Each field allocates exactly one owned `Bytes`
/// (method, path, body) plus the HeaderList copy. Pre-Stage 3
/// integration: the bump arena could absorb these if Request stops
/// using owned `Bytes` for these fields; out of scope today.
fn build_proxima_request(connection: &mut Connection, spec: &HttpListenerSpec) -> Request<Bytes> {
    let _ = spec;
    let path_bytes = connection.path();
    let (path, query) = split_path_and_query(path_bytes);
    let method = Method::from_bytes(connection.method());
    let mut headers = proxima_primitives::pipe::header_list::HeaderList::new();
    for header in connection.headers() {
        headers.insert(
            Bytes::copy_from_slice(header.name()),
            Bytes::copy_from_slice(header.value()),
        );
    }
    let body_bytes = Bytes::copy_from_slice(connection.body());
    let mut context = RequestContext::default();
    let (trace_id, baggage) = proxima_telemetry::propagation::establish_trace_context(&headers);
    context.adopt_trace_context(trace_id, baggage);
    // Pull any trailers the chunked decoder captured (zero-cost for
    // non-chunked / no-trailer requests — returns an empty Vec). Request
    // trailers fold into `headers` at chunked-decode end (yank-body).
    let captured_trailers = connection.take_trailers();
    for (name, value) in captured_trailers {
        headers.insert(name, value);
    }
    Request {
        method,
        path,
        query,
        metadata: headers,
        payload: body_bytes,
        stream: None,
        context,
    }
}

/// Split the raw request-target into `path` (`Bytes`) and a query
/// `HeaderList`. Percent-decodes query values to match the existing
/// hyper-listener behavior.
pub(crate) fn split_path_and_query(
    raw: &[u8],
) -> (Bytes, proxima_primitives::pipe::header_list::HeaderList) {
    let mut query = proxima_primitives::pipe::header_list::HeaderList::new();
    let (path_bytes, query_bytes) = match raw.iter().position(|&byte| byte == b'?') {
        Some(index) => (&raw[..index], &raw[index + 1..]),
        None => (raw, &b""[..]),
    };
    let path = if path_bytes.is_empty() {
        Bytes::from_static(b"/")
    } else {
        Bytes::copy_from_slice(path_bytes)
    };
    if !query_bytes.is_empty()
        && let Ok(query_text) = std::str::from_utf8(query_bytes)
    {
        for pair in query_text.split('&') {
            let mut split = pair.splitn(2, '=');
            if let Some(name) = split.next() {
                let value = split.next().unwrap_or("");
                query.insert(percent_decode(name), percent_decode(value));
            }
        }
    }
    (path, query)
}

/// Write a `proxima::Response` to the wire via the Connection's
/// response API. Streams the response body chunks as they arrive
/// from the Pipe so large / SSE-style bodies don't buffer.
///
/// Returns the response's optional `UpgradeHandler` (taken out of
/// the Response) so the caller can hijack the socket after the
/// head has been written. When upgrade is set, body framing is
/// suppressed: no Content-Length / Transfer-Encoding header is
/// emitted, no body bytes are written. The next bytes on the wire
/// belong to the handler's protocol (CONNECT tunnel, WebSocket
/// frames, h2c SETTINGS, …).
async fn write_response<W>(
    writer: &mut W,
    out: &mut Vec<u8>,
    connection: &mut Connection,
    response: Response<Bytes>,
    trace_id: Option<&[u8]>,
    route: &'static str,
) -> Result<Option<proxima_primitives::pipe::upgrade::UpgradeHandler>, ProximaError>
where
    W: AsyncWrite + Unpin,
{
    debug!(
        route = route,
        trace_id = ?trace_id,
        lifecycle_phase = "response_ready_or_terminal",
        response_writer_entered = true,
        "h1 response writer entered"
    );
    let mut response = response;
    let status = response.status;
    let headers = core::mem::take(&mut response.metadata);
    let upgrade = response.upgrade.take();
    let is_upgrade = upgrade.is_some();
    // Trailers ride on the ResponseStream — pull the slot now and read
    // it after the stream completes, passing trailers to
    // encode_end_with_trailers for chunked responses.
    let trailers_slot = response
        .stream
        .as_ref()
        .and_then(|stream| stream.trailers_slot().cloned());

    // Pick framing from the response headers. If content-length is
    // declared, use ContentLength; a body that ends inside the gather
    // phase below frames itself from its known length; anything still
    // pending is chunked. For upgrade responses, force BodyFraming::None
    // — the listener cedes the socket immediately after the head, no H1
    // body framing applies.
    let mut framing = if is_upgrade {
        BodyFraming::None
    } else {
        BodyFraming::Chunked
    };
    let mut header_pairs: Vec<(String, String)> = Vec::with_capacity(headers.len() + 1);
    let mut has_traceparent = false;
    let mut has_transfer_encoding = false;
    for (name, value) in &headers {
        let name_str = std::str::from_utf8(name.as_ref()).unwrap_or("");
        let value_str = std::str::from_utf8(value.as_ref()).unwrap_or("");
        if name_str.eq_ignore_ascii_case("content-length")
            && let Ok(parsed) = value_str.trim().parse::<u64>()
            && !is_upgrade
        {
            framing = BodyFraming::ContentLength(parsed);
        }
        if name_str.eq_ignore_ascii_case("transfer-encoding") {
            has_transfer_encoding = true;
        }
        if name_str.eq_ignore_ascii_case("traceparent") {
            has_traceparent = true;
        }
        // Skip framing-implying headers for upgrade responses so the
        // wire ends cleanly at the blank line.
        if is_upgrade
            && (name_str.eq_ignore_ascii_case("content-length")
                || name_str.eq_ignore_ascii_case("transfer-encoding"))
        {
            continue;
        }
        header_pairs.push((name_str.to_string(), value_str.to_string()));
    }
    if !has_traceparent
        && let Some(trace_id) = trace_id
        && let Ok(trace_id_str) = std::str::from_utf8(trace_id)
    {
        header_pairs.push(("traceparent".to_string(), trace_id_str.to_string()));
    }

    if is_upgrade {
        // Head flushes eagerly — the caller hijacks the socket next, so
        // the body machinery below never runs. end_response with
        // BodyFraming::None is a no-op on the wire and advances
        // Connection to AfterResponse.
        out.clear();
        let response_writer = connection.begin_response(status, "", &header_pairs, framing, out);
        writer.write_all(out).await.map_err(io_err)?;
        out.clear();
        response_writer.end_response(out);
        if !out.is_empty() {
            writer.write_all(out).await.map_err(io_err)?;
        }
        // The Pipe may still have populated body with leftover
        // data (e.g., a Sec-WebSocket-Protocol echo) but for the
        // hijack handshake we drop it deliberately — the post-101
        // protocol owns the wire from here.
        drop(response);
        return Ok(upgrade);
    }

    // Every body is the pipe's chunk stream — a buffered payload is the
    // degenerate stream that ends before its first pending. Gather the
    // chunks that are ALREADY ready (never waiting), so a body that
    // completes here seals with an implicit content-length and the whole
    // response leaves in ONE write. Three small sends per response was
    // the h1 throughput killer once TCP_NODELAY stopped Nagle from
    // coalescing them (6e876a24). A body that outlives the gather seals
    // chunked and flushes on pending edges, so streaming latency is
    // untouched; the budget bounds memory when a stream is ready-forever.
    const GATHER_BUDGET_BYTES: usize = 64 * 1024;
    let mut body_stream = response.into_chunk_stream();
    let mut gathered: Vec<Bytes> = Vec::new();
    let mut gathered_len: usize = 0;
    let mut body_ended = false;
    loop {
        match body_stream.next().now_or_never() {
            Some(Some(chunk)) => {
                let chunk = chunk?;
                gathered_len += chunk.len();
                if !chunk.is_empty() {
                    gathered.push(chunk);
                }
                if gathered_len > GATHER_BUDGET_BYTES {
                    break;
                }
            }
            Some(None) => {
                body_ended = true;
                break;
            }
            None => break,
        }
    }

    // Seal: the one place undeclared framing is decided. Declared
    // headers win; trailers require chunked; a body that ended while
    // gathering frames itself from its known length. text/event-stream
    // is exempt — SSE is a live incremental-delivery contract, so it
    // stays chunked even when a finite burst gathered whole; collapsing
    // it to content-length would defeat the protocol.
    let is_event_stream = header_pairs.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type")
            && value
                .trim()
                .to_ascii_lowercase()
                .starts_with("text/event-stream")
    });
    if matches!(framing, BodyFraming::Chunked) && !has_transfer_encoding {
        if body_ended && trailers_slot.is_none() && !is_event_stream {
            framing = BodyFraming::ContentLength(gathered_len as u64);
            header_pairs.push(("content-length".to_string(), gathered_len.to_string()));
        } else {
            header_pairs.push(("transfer-encoding".to_string(), "chunked".to_string()));
        }
    }

    out.clear();
    let response_writer = connection.begin_response(status, "", &header_pairs, framing, out);
    for chunk in &gathered {
        response_writer.write_chunk(chunk, out);
    }

    // Streaming tail — runs only when the body outlived the gather.
    // Each pass flushes on the pending edge (first pass: head + gathered
    // ride one write), awaits one chunk, then re-gathers whatever else
    // is ready so writes batch between pendings instead of per chunk.
    while !body_ended {
        writer.write_all(out).await.map_err(io_err)?;
        out.clear();
        match body_stream.next().await {
            Some(chunk) => {
                let chunk = chunk?;
                if !chunk.is_empty() {
                    response_writer.write_chunk(&chunk, out);
                }
            }
            None => break,
        }
        loop {
            match body_stream.next().now_or_never() {
                Some(Some(chunk)) => {
                    let chunk = chunk?;
                    if !chunk.is_empty() {
                        response_writer.write_chunk(&chunk, out);
                    }
                }
                Some(None) => {
                    body_ended = true;
                    break;
                }
                None => break,
            }
        }
    }

    // Response trailers (RFC 7230 §4.1.2) only flow on chunked
    // transfer-encoding; for Content-Length / None framings the
    // encoder ignores them. Body publishes its trailers via the
    // slot once the producing stream completes; we read after the
    // drain above so any stream-end-emitted trailers are visible.
    // The terminator rides the final write — in the degenerate case
    // that is THE one write for the whole response.
    let trailers_now = trailers_slot.as_ref().and_then(|slot| slot.lock().clone());
    if let Some(trailers) = trailers_now
        && matches!(framing, BodyFraming::Chunked)
        && !trailers.is_empty()
    {
        let trailer_refs: Vec<(&[u8], &[u8])> = trailers
            .iter()
            .map(|(name, value)| (name.as_ref(), value.as_ref()))
            .collect();
        response_writer.end_response_with_trailers(&trailer_refs, out);
    } else {
        response_writer.end_response(out);
    }
    writer.write_all(out).await.map_err(io_err)?;
    Ok(None)
}

fn percent_decode(input: &str) -> Bytes {
    // Returns Bytes so the caller can drop straight into a HeaderList
    // without an extra String→Bytes conversion. The percent-decoded
    // result is byte-oriented (matches RFC 3986 / 6750 semantics where
    // query values may carry non-UTF-8 bytes via %XX escapes).
    let bytes = input.as_bytes();
    let mut output: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'%' && index + 2 < bytes.len() {
            let hi = hex_digit(bytes[index + 1]);
            let lo = hex_digit(bytes[index + 2]);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                output.push((hi << 4) | lo);
                index += 3;
                continue;
            }
        }
        if byte == b'+' {
            output.push(b' ');
            index += 1;
            continue;
        }
        output.push(byte);
        index += 1;
    }
    Bytes::from(output)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;
    use std::net::Shutdown;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Condvar, Mutex};
    use std::sync::atomic::Ordering;
    use std::task::{Context, Poll};

    use futures::executor::block_on;
    use proxima_primitives::pipe::body::ResponseStream;
    use proxima_primitives::pipe::handler::into_handle;
    use tokio::io::{AsyncReadExt as TokioAsyncReadExt, AsyncWriteExt as TokioAsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::{Notify, oneshot};
    use tokio::task::LocalSet;
    use tokio_util::compat::{Compat, TokioAsyncReadCompatExt};

    use super::*;

    const HALF_CLOSE_EVIDENCE_DIR: &str = "/tmp/proxima-http8-target/http1-peer-half-close-evidence";

    struct TraceCollector {
        bytes: Mutex<Vec<u8>>,
        changed: Condvar,
    }

    struct TraceWriter(Arc<TraceCollector>);

    impl Write for TraceWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut captured = self.0.bytes.lock().expect("trace bytes lock");
            captured.extend_from_slice(bytes);
            self.0.changed.notify_all();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl TraceCollector {
        fn wait_for_marker(&self, marker: &[u8]) -> Vec<u8> {
            let mut bytes = self.bytes.lock().expect("trace bytes lock");
            while !bytes
                .windows(marker.len())
                .any(|window| window == marker)
            {
                bytes = self.changed.wait(bytes).expect("trace bytes wait");
            }
            bytes.clone()
        }
    }

    struct EofObservedSocket {
        inner: Compat<TcpStream>,
        eof_observed: Arc<Notify>,
        eof_seen: Arc<std::sync::atomic::AtomicBool>,
    }

    struct ReadByteObservedSocket {
        inner: Compat<TcpStream>,
        observation_enabled: Arc<std::sync::atomic::AtomicBool>,
        byte_observed: Arc<std::sync::atomic::AtomicBool>,
        byte_signal: Arc<Notify>,
    }

    impl AsyncRead for EofObservedSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let result = Pin::new(&mut self.inner).poll_read(context, buffer);
            if matches!(result, Poll::Ready(Ok(0))) {
                self.eof_seen.store(true, Ordering::SeqCst);
                self.eof_observed.notify_one();
            }
            result
        }
    }

    impl AsyncWrite for EofObservedSocket {
        fn poll_write(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.inner).poll_write(context, buffer)
        }

        fn poll_flush(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(context)
        }

        fn poll_close(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_close(context)
        }
    }

    impl AsyncRead for ReadByteObservedSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let result = Pin::new(&mut self.inner).poll_read(context, buffer);
            if self.observation_enabled.load(Ordering::SeqCst)
                && matches!(&result, Poll::Ready(Ok(count)) if *count > 0)
            {
                self.byte_observed.store(true, Ordering::SeqCst);
                self.byte_signal.notify_one();
            }
            result
        }
    }

    impl AsyncWrite for ReadByteObservedSocket {
        fn poll_write(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.inner).poll_write(context, buffer)
        }

        fn poll_flush(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(context)
        }

        fn poll_close(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_close(context)
        }
    }

    fn half_close_trace_collector() -> (Arc<TraceCollector>, Arc<proxima_telemetry::recorder::Recorder>) {
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        std::fs::create_dir_all(evidence_directory).expect("create telemetry evidence directory");
        let collector = Arc::new(TraceCollector {
            bytes: Mutex::new(Vec::new()),
            changed: Condvar::new(),
        });
        proxima_telemetry::emit::global::install(
            proxima_telemetry::emit::EnvFilter::parse("debug"),
        );
        let exporter = proxima_telemetry::export::Exporter::fan(vec![
            proxima_telemetry::export::Exporter::file(
                evidence_directory.join("structured-trace.jsonl"),
            )
            .format(proxima_telemetry::export::Formatter::Json),
            proxima_telemetry::export::Exporter::writer(TraceWriter(collector.clone()))
                .format(proxima_telemetry::export::Formatter::Json),
        ])
        .expect("compose structured trace file and test signal sinks");
        let recorder = proxima_telemetry::recorder::Recorder::builder()
            .core_count(1)
            .export(exporter)
            .expect("build structured trace exporter")
            .install()
            .expect("install structured trace recorder");
        (collector, recorder)
    }

    fn half_close_request(route: &str, trace_id: &str) -> Vec<u8> {
        let (method, framing) = if route == "buffered" {
            ("GET", "")
        } else {
            ("POST", "Transfer-Encoding: chunked\r\n")
        };
        let body = if route == "buffered" {
            ""
        } else {
            "5\r\nhello\r\n0\r\n\r\n"
        };
        format!(
            "{method} /half-close HTTP/1.1\r\nHost: localhost\r\n{framing}traceparent: 00-{trace_id}-0123456789abcdef-01\r\nConnection: close\r\n\r\n{body}"
        )
        .into_bytes()
    }

    fn expected_trace_tag(trace_id: &str) -> String {
        let traceparent = format!("00-{trace_id}-0123456789abcdef-01");
        format!("{:?}", Some(traceparent.as_bytes()))
    }

    struct HalfCloseObservation {
        evidence_name: String,
        request: Vec<u8>,
        response: Vec<u8>,
        eof_observed: bool,
        dispatch_calls: usize,
        server_result: Result<(), String>,
        admission_in_flight: u64,
    }

    struct PipelineByteObservation {
        route: &'static str,
        response: Vec<u8>,
        byte_observed: bool,
        dispatch_calls: usize,
        admission_in_flight: u64,
    }

    struct IncompleteBodyObservation {
        evidence_name: String,
        response: Vec<u8>,
        eof_observed: bool,
        cancel_seen: bool,
        admission_in_flight: u64,
    }

    async fn run_peer_half_close_case(
        route: &'static str,
        trace_id: &'static str,
        half_close: bool,
        runtime_route: bool,
    ) -> HalfCloseObservation {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind half-close listener");
        let address = listener.local_addr().expect("half-close listener address");
        let (client, server) = tokio::join!(TcpStream::connect(address), listener.accept());
        let mut client = client.expect("connect half-close client");
        let (server, _) = server.expect("accept half-close server");
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let dispatch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipe = HalfCloseBarrierPipe {
            started: Arc::new(Mutex::new(Some(started_tx))),
            release: Arc::new(Mutex::new(Some(release_rx))),
            calls: dispatch_calls.clone(),
        };
        let eof_observed = Arc::new(Notify::new());
        let eof_signal = eof_observed.clone();
        let eof_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let eof_seen_by_server = eof_seen.clone();
        let socket = EofObservedSocket {
            inner: server.compat(),
            eof_observed: eof_signal,
            eof_seen: eof_seen_by_server,
        };
        let runtime: Option<Arc<dyn Runtime>> =
            runtime_route.then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
        let admission = ConnAdmission::new(1);
        let server_admission = admission.clone();
        let spec = Arc::new(HttpListenerSpec::default());
        let quiesce_response = Arc::new(QuiesceResponse {
            status: 503,
            retry_after: "1".into(),
        });
        let server_task = tokio::task::spawn_local(async move {
            serve_connection(
                socket,
                into_handle(pipe),
                spec,
                server_admission,
                quiesce_response,
                None,
                runtime,
            )
            .await
        });
        let request = half_close_request(route, trace_id);
        client
            .write_all(&request)
            .await
            .expect("write complete half-close request");
        started_rx
            .await
            .expect("dispatch reached response barrier after body completion");
        if half_close {
            let standard_client = client
                .into_std()
                .expect("convert client to standard TCP stream");
            standard_client
                .shutdown(Shutdown::Write)
                .expect("shutdown client write direction");
            client = TcpStream::from_std(standard_client)
                .expect("restore client to async TCP stream");
            eof_observed.notified().await;
        }
        let _ = release_tx.send(());
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .expect("read half-close response bytes");
        let server_result = server_task
            .await
            .expect("serve task joins")
            .map_err(|error| format!("{error:?}"));
        let evidence_name = format!(
            "{route}-{}",
            if half_close { "shutdown-write" } else { "write-open" }
        );
        let scratch_directory = tempfile::tempdir().expect("create per-case evidence directory");
        let scratch_request = scratch_directory.path().join("request.raw");
        let scratch_response = scratch_directory.path().join("response.raw");
        std::fs::write(&scratch_request, &request).expect("write exact request payload");
        std::fs::write(&scratch_response, &response).expect("write exact response payload");
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        std::fs::create_dir_all(evidence_directory).expect("create HTTP/1 evidence directory");
        std::fs::copy(
            scratch_request,
            evidence_directory.join(format!("{evidence_name}.request.raw")),
        )
        .expect("retain exact request payload");
        std::fs::copy(
            scratch_response,
            evidence_directory.join(format!("{evidence_name}.response.raw")),
        )
        .expect("retain exact response payload");
        HalfCloseObservation {
            evidence_name,
            request,
            response,
            eof_observed: eof_seen.load(Ordering::SeqCst),
            dispatch_calls: dispatch_calls.load(Ordering::SeqCst),
            server_result,
            admission_in_flight: admission.in_flight(),
        }
    }

    async fn run_positive_pipeline_byte_case(
        route: &'static str,
        runtime_route: bool,
        trace_id: &'static str,
    ) -> PipelineByteObservation {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind pipeline-byte listener");
        let address = listener.local_addr().expect("pipeline-byte address");
        let (client, server) = tokio::join!(TcpStream::connect(address), listener.accept());
        let mut client = client.expect("connect pipeline-byte client");
        let (server, _) = server.expect("accept pipeline-byte client");
        let first_request = String::from_utf8(half_close_request(route, trace_id))
            .expect("pipeline request is UTF-8")
            .replace("Connection: close", "Connection: keep-alive")
            .into_bytes();
        let second_request = half_close_request(route, "99999999999999999999999999999999");
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let dispatch_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pipe = HalfCloseBarrierPipe {
            started: Arc::new(Mutex::new(Some(started_tx))),
            release: Arc::new(Mutex::new(Some(release_rx))),
            calls: dispatch_calls.clone(),
        };
        let observation_enabled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let byte_observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let byte_signal = Arc::new(Notify::new());
        let socket = ReadByteObservedSocket {
            inner: server.compat(),
            observation_enabled: observation_enabled.clone(),
            byte_observed: byte_observed.clone(),
            byte_signal: byte_signal.clone(),
        };
        let runtime: Option<Arc<dyn Runtime>> =
            runtime_route.then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
        let admission = ConnAdmission::new(1);
        let server_admission = admission.clone();
        let server_task = tokio::task::spawn_local(async move {
            serve_connection(
                socket,
                into_handle(pipe),
                Arc::new(HttpListenerSpec::default()),
                server_admission,
                Arc::new(QuiesceResponse {
                    status: 503,
                    retry_after: "1".into(),
                }),
                None,
                runtime,
            )
            .await
        });
        client
            .write_all(&first_request)
            .await
            .expect("write first pipelined request");
        started_rx
            .await
            .expect("first dispatch reaches response barrier");
        observation_enabled.store(true, Ordering::SeqCst);
        client
            .write_all(&second_request)
            .await
            .expect("write next request while response is pending");
        byte_signal.notified().await;
        let _ = release_tx.send(());
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .expect("read both pipelined responses");
        server_task
            .await
            .expect("pipeline-byte server joins")
            .expect("pipeline-byte route completes");
        let scratch_directory = tempfile::tempdir().expect("create pipeline-byte evidence tempdir");
        let scratch_request = scratch_directory.path().join("request.raw");
        let scratch_response = scratch_directory.path().join("response.raw");
        std::fs::write(
            &scratch_request,
            [first_request.as_slice(), second_request.as_slice()].concat(),
        )
        .expect("write pipeline-byte request payload");
        std::fs::write(&scratch_response, &response).expect("write pipeline-byte response payload");
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        std::fs::copy(
            scratch_request,
            evidence_directory.join(format!("{route}-pipeline-byte.request.raw")),
        )
        .expect("retain pipeline-byte request payload");
        std::fs::copy(
            scratch_response,
            evidence_directory.join(format!("{route}-pipeline-byte.response.raw")),
        )
        .expect("retain pipeline-byte response payload");
        PipelineByteObservation {
            route,
            response,
            byte_observed: byte_observed.load(Ordering::SeqCst),
            dispatch_calls: dispatch_calls.load(Ordering::SeqCst),
            admission_in_flight: admission.in_flight(),
        }
    }

    async fn run_response_sender_dropped_case(admitted: bool) -> (Vec<u8>, Vec<u8>, u64) {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind response-sender-drop listener");
        let address = listener
            .local_addr()
            .expect("response-sender-drop listener address");
        let (client, server) = tokio::join!(TcpStream::connect(address), listener.accept());
        let mut client = client.expect("connect response-sender-drop client");
        let (server, _) = server.expect("accept response-sender-drop server");
        let trace_id = if admitted {
            "44444444444444444444444444444444"
        } else {
            "dddddddddddddddddddddddddddddddd"
        };
        let evidence_name = if admitted {
            "streaming-response-sender-dropped-admitted"
        } else {
            "streaming-response-sender-dropped-shed"
        };
        let request = half_close_request("streaming_some_runtime", trace_id);
        let admission = ConnAdmission::new(if admitted { 1 } else { 0 });
        let server_admission = admission.clone();
        let socket = server.compat();
        let spec = Arc::new(HttpListenerSpec::default());
        let quiesce_response = Arc::new(QuiesceResponse {
            status: 503,
            retry_after: "1".into(),
        });
        let runtime: Arc<dyn Runtime> = Arc::new(DroppingRuntime);
        let server_task = tokio::task::spawn_local(async move {
            serve_connection(
                socket,
                into_handle(FixedResponse {
                    response: || Response::new(200).with_body(Bytes::from_static(b"unused")),
                }),
                spec,
                server_admission,
                quiesce_response,
                None,
                Some(runtime),
            )
            .await
        });
        client
            .write_all(&request)
            .await
            .expect("write complete response-sender-drop request");
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .expect("read response-sender-drop bytes");
        server_task
            .await
            .expect("response-sender-drop server joins")
            .expect("response-sender-drop server closes cleanly");
        let in_flight = admission.in_flight();
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        let scratch_directory = tempfile::tempdir().expect("create sender-drop evidence tempdir");
        let scratch_request = scratch_directory.path().join("request.raw");
        let scratch_response = scratch_directory.path().join("response.raw");
        let scratch_state = scratch_directory.path().join("admission-state.txt");
        std::fs::write(&scratch_request, &request).expect("write sender-drop request payload");
        std::fs::write(&scratch_response, &response).expect("write sender-drop response payload");
        std::fs::write(&scratch_state, format!("admission.in_flight={in_flight}\n"))
            .expect("write sender-drop admission state");
        std::fs::copy(
            scratch_request,
            evidence_directory.join(format!("{evidence_name}.request.raw")),
        )
        .expect("retain sender-drop request payload");
        std::fs::copy(
            scratch_response,
            evidence_directory.join(format!("{evidence_name}.response.raw")),
        )
        .expect("retain sender-drop response payload");
        std::fs::copy(
            scratch_state,
            evidence_directory.join(format!("{evidence_name}-state.txt")),
        )
        .expect("retain sender-drop admission state");
        (request, response, in_flight)
    }

    async fn run_dispatch_error_case(
        route: &'static str,
        trace_id: &'static str,
        runtime_route: bool,
    ) -> (Vec<u8>, u64) {
        let request = half_close_request(route, trace_id);
        let (socket, writes) = TestSocket::new(&request);
        let admission = ConnAdmission::new(1);
        let server_admission = admission.clone();
        let runtime: Option<Arc<dyn Runtime>> =
            runtime_route.then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
        serve_connection(
            socket,
            into_handle(DispatchErrorPipe),
            Arc::new(HttpListenerSpec::default()),
            server_admission,
            Arc::new(QuiesceResponse {
                status: 503,
                retry_after: "1".into(),
            }),
            None,
            runtime,
        )
        .await
        .expect("dispatch error renders an error response and closes cleanly");
        let response = writes.lock().unwrap().concat();
        let scratch_directory = tempfile::tempdir().expect("create dispatch-error evidence tempdir");
        let scratch_request = scratch_directory.path().join("request.raw");
        let scratch_response = scratch_directory.path().join("response.raw");
        std::fs::write(&scratch_request, &request).expect("write dispatch-error request");
        std::fs::write(&scratch_response, &response).expect("write dispatch-error response");
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        std::fs::copy(
            scratch_request,
            evidence_directory.join(format!("{route}-dispatch-error.request.raw")),
        )
        .expect("retain dispatch-error request");
        std::fs::copy(
            scratch_response,
            evidence_directory.join(format!("{route}-dispatch-error.response.raw")),
        )
        .expect("retain dispatch-error response");
        (response, admission.in_flight())
    }

    async fn run_incomplete_body_eof_case(runtime_route: bool) -> IncompleteBodyObservation {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind incomplete-body listener");
        let address = listener
            .local_addr()
            .expect("incomplete-body listener address");
        let (client, server) = tokio::join!(TcpStream::connect(address), listener.accept());
        let mut client = client.expect("connect incomplete-body client");
        let (server, _) = server.expect("accept incomplete-body server");
        let request = b"POST /incomplete HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\ntraceparent: 00-55555555555555555555555555555555-0123456789abcdef-01\r\nConnection: close\r\n\r\n5\r\nhe".to_vec();
        let admission = ConnAdmission::new(1);
        let server_admission = admission.clone();
        let cancel_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipe = CancelAwaitingPipe {
            cancel_seen: cancel_seen.clone(),
        };
        let eof_observed = Arc::new(Notify::new());
        let eof_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let socket = EofObservedSocket {
            inner: server.compat(),
            eof_observed: eof_observed.clone(),
            eof_seen: eof_seen.clone(),
        };
        let runtime: Option<Arc<dyn Runtime>> =
            runtime_route.then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
        let spec = Arc::new(HttpListenerSpec::default());
        let quiesce_response = Arc::new(QuiesceResponse {
            status: 503,
            retry_after: "1".into(),
        });
        let server_task = tokio::task::spawn_local(async move {
            serve_connection(
                socket,
                into_handle(pipe),
                spec,
                server_admission,
                quiesce_response,
                None,
                runtime,
            )
            .await
        });
        client
            .write_all(&request)
            .await
            .expect("write incomplete chunked request bytes");
        let standard_client = client
            .into_std()
            .expect("convert incomplete-body client to standard TCP stream");
        standard_client
            .shutdown(Shutdown::Write)
            .expect("shutdown incomplete-body client write direction");
        client = TcpStream::from_std(standard_client)
            .expect("restore incomplete-body client to async TCP stream");
        eof_observed.notified().await;
        let mut response = Vec::new();
        client
            .read_to_end(&mut response)
            .await
            .expect("read incomplete-body connection bytes");
        server_task
            .await
            .expect("incomplete-body server joins")
            .expect("incomplete-body connection closes cleanly");
        let evidence_name = if runtime_route {
            "incomplete-body-some-runtime"
        } else {
            "incomplete-body-no-runtime"
        };
        let scratch_directory = tempfile::tempdir().expect("create incomplete-body tempdir");
        let scratch_request = scratch_directory.path().join("request.raw");
        let scratch_response = scratch_directory.path().join("response.raw");
        std::fs::write(&scratch_request, &request).expect("write incomplete-body request");
        std::fs::write(&scratch_response, &response).expect("write incomplete-body response");
        let evidence_directory = std::path::Path::new(HALF_CLOSE_EVIDENCE_DIR);
        std::fs::copy(
            scratch_request,
            evidence_directory.join(format!("{evidence_name}.request.raw")),
        )
        .expect("retain incomplete-body request");
        std::fs::copy(
            scratch_response,
            evidence_directory.join(format!("{evidence_name}.response.raw")),
        )
        .expect("retain incomplete-body response");
        IncompleteBodyObservation {
            evidence_name: evidence_name.to_string(),
            response,
            eof_observed: eof_seen.load(Ordering::SeqCst),
            cancel_seen: cancel_seen.load(Ordering::SeqCst),
            admission_in_flight: admission.in_flight(),
        }
    }

    async fn run_streaming_body_reset_case(runtime_route: bool) -> (bool, Vec<Vec<u8>>, u64) {
        let request = b"POST /reset-body HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\ntraceparent: 00-12121212121212121212121212121212-0123456789abcdef-01\r\nConnection: close\r\n\r\n5\r\nhe";
        let (socket, writes) = TestSocket::new_with_reset_after_input(request);
        let cancel_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipe = CancelAwaitingPipe {
            cancel_seen: cancel_seen.clone(),
        };
        let admission = ConnAdmission::new(1);
        let server_admission = admission.clone();
        let runtime: Option<Arc<dyn Runtime>> =
            runtime_route.then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
        let result = serve_connection(
            socket,
            into_handle(pipe),
            Arc::new(HttpListenerSpec::default()),
            server_admission,
            Arc::new(QuiesceResponse {
                status: 503,
                retry_after: "1".into(),
            }),
            None,
            runtime,
        )
        .await;
        assert!(matches!(result, Err(ProximaError::Io(_))));
        (
            cancel_seen.load(Ordering::SeqCst),
            writes.lock().unwrap().clone(),
            admission.in_flight(),
        )
    }

    struct HalfCloseBarrierPipe {
        started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        release: Arc<Mutex<Option<oneshot::Receiver<()>>>>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    struct CancelAwaitingPipe {
        cancel_seen: Arc<std::sync::atomic::AtomicBool>,
    }

    struct LocalSetRuntime;

    struct DroppingRuntime;

    impl Runtime for LocalSetRuntime {
        fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            tokio::task::spawn_local(future);
        }

        fn spawn_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test dispatch uses the current LocalSet only")
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test dispatch uses the current LocalSet only")
        }

        fn spawn_background_blocking(
            &self,
            _work: Box<
                dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send,
            >,
        ) -> proxima_runtime::BackgroundHandle<Box<dyn std::any::Any + Send>> {
            unreachable!("test dispatch performs no background work")
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("test dispatch installs no timers")
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> proxima_runtime::CoreId {
            proxima_runtime::CoreId(0)
        }
    }

    impl Runtime for DroppingRuntime {
        fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            drop(future);
        }

        fn spawn_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test dispatch never crosses cores")
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test dispatch never crosses cores")
        }

        fn spawn_background_blocking(
            &self,
            _work: Box<
                dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send,
            >,
        ) -> proxima_runtime::BackgroundHandle<Box<dyn std::any::Any + Send>> {
            unreachable!("test dispatch never starts background work")
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("test dispatch never installs timers")
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> proxima_runtime::CoreId {
            proxima_runtime::CoreId(0)
        }
    }

    impl SendPipe for HalfCloseBarrierPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let started = self
                .started
                .lock()
                .expect("dispatch-start sender lock")
                .take();
            let release = self
                .release
                .lock()
                .expect("dispatch release receiver lock")
                .take();
            self.calls.fetch_add(1, Ordering::SeqCst);
            async move {
                let (_request, _body) = request.body_bytes().await?;
                if let Some(started) = started {
                    let _ = started.send(());
                }
                if let Some(release) = release {
                    let _ = release.await;
                }
                Ok(Response::new(200).with_body(Bytes::from_static(b"half-close-ready")))
            }
        }
    }

    impl SendPipe for CancelAwaitingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let cancel_seen = self.cancel_seen.clone();
            async move {
                let (request, _body) = request.body_bytes().await?;
                let cancel_signal = request.context.cancel.clone();
                cancel_signal.fired().await;
                cancel_seen.store(true, Ordering::SeqCst);
                Ok(Response::new(200).with_body(Bytes::new()))
            }
        }
    }

    #[test]
    fn percent_decode_handles_basic_escapes() {
        assert_eq!(percent_decode("hello%20world").as_ref(), b"hello world");
        assert_eq!(percent_decode("a%2Bb").as_ref(), b"a+b");
        assert_eq!(percent_decode("plus+sign").as_ref(), b"plus sign");
    }

    #[test]
    fn percent_decode_passes_through_unencoded_bytes() {
        assert_eq!(percent_decode("simple-word_42").as_ref(), b"simple-word_42");
    }

    /// In-memory half-open socket: serves a canned request, then reads
    /// return Pending (client waiting on the response, never EOF). Every
    /// poll_write is logged separately so tests can assert how many
    /// socket writes a response cost — the coalescing contract itself.
    struct TestSocket {
        input: Vec<u8>,
        read_pos: usize,
        writes: Arc<Mutex<Vec<Vec<u8>>>>,
        read_after_input: TestReadAfterInput,
        fail_writes: bool,
    }

    enum TestReadAfterInput {
        Pending,
        Eof,
        Reset,
    }

    impl TestSocket {
        fn new(request: &[u8]) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            Self::with_config(request, TestReadAfterInput::Pending, false)
        }

        fn new_with_eof_after_input(request: &[u8]) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            Self::with_config(request, TestReadAfterInput::Eof, false)
        }

        fn new_with_reset_after_input(request: &[u8]) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            Self::with_config(request, TestReadAfterInput::Reset, false)
        }

        fn new_with_write_failure(request: &[u8]) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            Self::with_config(request, TestReadAfterInput::Pending, true)
        }

        fn with_config(
            request: &[u8],
            read_after_input: TestReadAfterInput,
            fail_writes: bool,
        ) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            let writes = Arc::new(Mutex::new(Vec::new()));
            let socket = Self {
                input: request.to_vec(),
                read_pos: 0,
                writes: writes.clone(),
                read_after_input,
                fail_writes,
            };
            (socket, writes)
        }
    }

    impl AsyncRead for TestSocket {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let this = self.get_mut();
            if this.read_pos >= this.input.len() {
                return match this.read_after_input {
                    TestReadAfterInput::Pending => Poll::Pending,
                    TestReadAfterInput::Eof => Poll::Ready(Ok(0)),
                    TestReadAfterInput::Reset => Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionReset,
                        "test peer reset",
                    ))),
                };
            }
            let take = (this.input.len() - this.read_pos).min(buf.len());
            buf[..take].copy_from_slice(&this.input[this.read_pos..this.read_pos + take]);
            this.read_pos += take;
            Poll::Ready(Ok(take))
        }
    }

    impl AsyncWrite for TestSocket {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            let this = self.get_mut();
            if this.fail_writes {
                return Poll::Ready(Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "test response writer failure",
                )));
            }
            this.writes.lock().unwrap().push(buf.to_vec());
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    struct FixedResponse {
        response: fn() -> Response<Bytes>,
    }

    impl SendPipe for FixedResponse {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        // `SendPipe::call` returns `impl Future + Send`; an `async fn` here
        // would not carry the Send bound the trait requires.
        #[allow(clippy::manual_async_fn)]
        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let response = (self.response)();
            async move { Ok(response) }
        }
    }

    /// Drains the request body (streamed or buffered — `body_bytes`
    /// handles both) and echoes it back as the response body. Used to
    /// prove chunked/streamed request bodies actually reach the Pipe.
    struct EchoBodyPipe;

    impl SendPipe for EchoBodyPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        // `SendPipe::call` returns `impl Future + Send`; an `async fn` here
        // would not carry the Send bound the trait requires.
        #[allow(clippy::manual_async_fn)]
        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                let (_request, body) = request.body_bytes().await?;
                Ok(Response::new(200).with_body(body))
            }
        }
    }

    /// Resolves `Pending` exactly once (waking itself immediately), then
    /// `Ready`. Gives the disconnect-detection race below a genuine
    /// window to observe a peer close while a Pipe is mid-flight,
    /// instead of the Pipe resolving synchronously before the race
    /// ever gets a turn.
    struct Yield {
        polled: bool,
    }

    impl Future for Yield {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.polled {
                Poll::Ready(())
            } else {
                self.polled = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    /// Yields once, then records whether `request.context.cancel` had
    /// fired by that point — proving the disconnect race fired cancel
    /// AND gave the Pipe a poll cycle to observe it before dropping.
    struct CancelObservingPipe {
        cancel_seen: Arc<std::sync::atomic::AtomicBool>,
    }

    struct PendingSignalCapturePipe {
        signal: Arc<Mutex<Option<proxima_core::signal::Signal>>>,
        started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        cancel_observed: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    }

    struct DispatchErrorPipe;

    impl SendPipe for CancelObservingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let cancel_seen = self.cancel_seen.clone();
            async move {
                Yield { polled: false }.await;
                if request.context.cancel.is_fired() {
                    cancel_seen.store(true, Ordering::SeqCst);
                }
                Ok(Response::new(200).with_body(Bytes::new()))
            }
        }
    }

    impl SendPipe for PendingSignalCapturePipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let signal = self.signal.clone();
            let started = self.started.clone();
            let cancel_observed = self.cancel_observed.clone();
            async move {
                let (request, _body) = request.body_bytes().await?;
                let cancel_signal = request.context.cancel.clone();
                *signal.lock().expect("captured cancellation signal lock") =
                    Some(cancel_signal.clone());
                if let Some(started) = started
                    .lock()
                    .expect("dispatch-start sender lock")
                    .take()
                {
                    let _ = started.send(());
                }
                cancel_signal.fired().await;
                if let Some(cancel_observed) = cancel_observed
                    .lock()
                    .expect("cancel-observed sender lock")
                    .take()
                {
                    let _ = cancel_observed.send(());
                }
                Ok(Response::new(200).with_body(Bytes::new()))
            }
        }
    }

    impl SendPipe for DispatchErrorPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                let (_request, _body) = request.body_bytes().await?;
                Err(ProximaError::Upstream(
                    "controlled dispatch failure".into(),
                ))
            }
        }
    }

    const CLOSE_GET: &[u8] = b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";

    fn serve_once(response: fn() -> Response<Bytes>) -> Vec<Vec<u8>> {
        let (socket, writes) = TestSocket::new(CLOSE_GET);
        block_on(serve_h1_connection(
            socket,
            into_handle(FixedResponse { response }),
            None,
            None,
        ))
        .expect("serve should complete cleanly on connection: close");
        writes.lock().unwrap().clone()
    }

    fn joined(writes: &[Vec<u8>]) -> String {
        String::from_utf8(writes.concat()).expect("response should be utf8")
    }

    #[test]
    fn buffered_body_seals_implicit_content_length_in_one_write() {
        let writes = serve_once(|| Response::new(200).with_body(Bytes::from_static(b"ok")));

        assert_eq!(
            writes.len(),
            1,
            "buffered response must leave in ONE socket write, got {writes:?}"
        );
        let wire = joined(&writes);
        assert!(
            wire.contains("content-length: 2"),
            "implicit length: {wire}"
        );
        assert!(
            !wire.contains("transfer-encoding"),
            "no chunked framing: {wire}"
        );
        assert!(wire.ends_with("ok"), "body rides the same write: {wire}");
    }

    #[test]
    fn ready_stream_seals_implicit_content_length_in_one_write() {
        let writes = serve_once(|| {
            let chunks = futures::stream::iter([
                Ok(Bytes::from_static(b"he")),
                Ok(Bytes::from_static(b"llo")),
            ]);
            Response::new(200).with_stream(ResponseStream::new(chunks))
        });

        assert_eq!(
            writes.len(),
            1,
            "a stream that is ready end-to-end degenerates to ONE write, got {writes:?}"
        );
        let wire = joined(&writes);
        assert!(
            wire.contains("content-length: 5"),
            "length inferred: {wire}"
        );
        assert!(
            !wire.contains("transfer-encoding"),
            "no chunked framing: {wire}"
        );
        assert!(wire.ends_with("hello"), "body rides the same write: {wire}");
    }

    #[test]
    fn pending_stream_seals_chunked_and_batches_on_pending_edges() {
        let writes = serve_once(|| {
            let mut step = 0_u32;
            let chunks = futures::stream::poll_fn(move |poll_cx| {
                step += 1;
                match step {
                    1 => Poll::Ready(Some(Ok(Bytes::from_static(b"he")))),
                    2 => {
                        poll_cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                    3 => Poll::Ready(Some(Ok(Bytes::from_static(b"llo")))),
                    _ => Poll::Ready(None),
                }
            });
            Response::new(200).with_stream(ResponseStream::new(chunks))
        });

        assert_eq!(
            writes.len(),
            2,
            "one write per pending edge: head+first chunk, then tail+terminator, got {writes:?}"
        );
        let head_write = String::from_utf8(writes[0].clone()).expect("head utf8");
        assert!(
            head_write.starts_with("HTTP/1.1 200"),
            "head first: {head_write}"
        );
        assert!(
            head_write.contains("transfer-encoding: chunked"),
            "pending stream stays chunked: {head_write}"
        );
        assert!(
            head_write.ends_with("2\r\nhe\r\n"),
            "first ready chunk coalesces with the head: {head_write:?}"
        );
        let tail_write = String::from_utf8(writes[1].clone()).expect("tail utf8");
        assert_eq!(
            tail_write, "3\r\nllo\r\n0\r\n\r\n",
            "final chunk and terminator share the last write"
        );
    }

    /// Byte queue shared by one direction of [`test_duplex`]. Dropping
    /// the writing half marks `closed` and wakes the reader — the
    /// same "peer closed" signal a real socket gives on drop.
    #[derive(Default)]
    struct DuplexBuf {
        bytes: std::collections::VecDeque<u8>,
        closed: bool,
        waker: Option<std::task::Waker>,
    }

    /// One half of a hand-rolled, tokio-free in-memory duplex stream:
    /// two independent `DuplexBuf` queues, one per direction. Exists
    /// because `futures` (unlike `tokio`) ships no `io::duplex` —
    /// this is the minimal `AsyncRead + AsyncWrite` pair needed to
    /// drive `serve_connection`'s "client" and "server" sides as two
    /// genuinely interleaved futures under one `block_on`.
    struct DuplexHalf {
        read_buf: Arc<Mutex<DuplexBuf>>,
        write_buf: Arc<Mutex<DuplexBuf>>,
    }

    fn test_duplex() -> (DuplexHalf, DuplexHalf) {
        let a_to_b = Arc::new(Mutex::new(DuplexBuf::default()));
        let b_to_a = Arc::new(Mutex::new(DuplexBuf::default()));
        (
            DuplexHalf {
                read_buf: b_to_a.clone(),
                write_buf: a_to_b.clone(),
            },
            DuplexHalf {
                read_buf: a_to_b,
                write_buf: b_to_a,
            },
        )
    }

    impl AsyncRead for DuplexHalf {
        fn poll_read(
            self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let mut guard = self.read_buf.lock().unwrap();
            if guard.bytes.is_empty() {
                if guard.closed {
                    return Poll::Ready(Ok(0));
                }
                guard.waker = Some(cx.waker().clone());
                return Poll::Pending;
            }
            let take = guard.bytes.len().min(buf.len());
            for slot in buf.iter_mut().take(take) {
                *slot = guard.bytes.pop_front().expect("checked len above");
            }
            Poll::Ready(Ok(take))
        }
    }

    impl AsyncWrite for DuplexHalf {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            let mut guard = self.write_buf.lock().unwrap();
            guard.bytes.extend(buf.iter().copied());
            if let Some(waker) = guard.waker.take() {
                waker.wake();
            }
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            let mut guard = self.write_buf.lock().unwrap();
            guard.closed = true;
            if let Some(waker) = guard.waker.take() {
                waker.wake();
            }
            Poll::Ready(Ok(()))
        }
    }

    impl Drop for DuplexHalf {
        fn drop(&mut self) {
            let mut guard = self.write_buf.lock().unwrap();
            guard.closed = true;
            if let Some(waker) = guard.waker.take() {
                waker.wake();
            }
        }
    }

    /// The tokio-freedom proof: `serve_connection` driven entirely by
    /// `futures::executor::block_on` over an in-memory duplex — no
    /// tokio runtime, no `LocalSet`, anywhere in this test. The
    /// "client" and "server" halves run as two futures joined under
    /// one `block_on` call, so the exchange is a genuine two-sided
    /// round trip, not a pre-buffered canned input.
    #[test]
    fn serve_connection_completes_over_an_in_memory_duplex_with_no_tokio_runtime() {
        let (server_half, mut client_half) = test_duplex();
        let spec = Arc::new(HttpListenerSpec::default());
        let admission = ConnAdmission::unbounded();
        let quiesce_response = Arc::new(QuiesceResponse {
            status: 503,
            retry_after: "1".into(),
        });

        let server = serve_connection(
            server_half,
            into_handle(FixedResponse {
                response: || Response::new(200).with_body(Bytes::from_static(b"pong")),
            }),
            spec,
            admission,
            quiesce_response,
            None,
            None,
        );

        let client = async move {
            client_half
                .write_all(b"GET /ping HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .await
                .expect("client write");
            let mut response = Vec::new();
            client_half
                .read_to_end(&mut response)
                .await
                .expect("client read");
            response
        };

        let (server_result, response) = block_on(futures::future::join(server, client));
        server_result.expect("serve_connection should complete with no tokio runtime present");
        let wire = String::from_utf8(response).expect("response should be utf8");
        assert!(wire.starts_with("HTTP/1.1 200"), "response head: {wire}");
        assert!(wire.ends_with("pong"), "response body: {wire}");
    }

    /// Read one full HTTP/1.1 response off a keep-alive connection: parse
    /// headers up to `\r\n\r\n`, then read exactly `Content-Length` more
    /// body bytes. Unlike `read_to_end` (used elsewhere in this file for
    /// `Connection: close` responses), this never blocks past the end of
    /// ONE response, so the caller can keep reusing the same connection.
    async fn read_http_response<R: AsyncRead + Unpin>(reader: &mut R) -> (String, Vec<u8>) {
        let mut buf = Vec::new();
        let mut chunk = [0_u8; 512];
        let header_end = loop {
            if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
                break pos + 4;
            }
            let count = reader.read(&mut chunk).await.expect("read response head");
            assert!(count > 0, "unexpected eof while reading response headers");
            buf.extend_from_slice(&chunk[..count]);
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        let content_length: usize = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().ok())
                    .flatten()
            })
            .unwrap_or(0);
        let mut body = buf[header_end..].to_vec();
        while body.len() < content_length {
            let count = reader.read(&mut chunk).await.expect("read response body");
            assert!(count > 0, "unexpected eof while reading response body");
            body.extend_from_slice(&chunk[..count]);
        }
        body.truncate(content_length);
        (head, body)
    }

    /// The handler behind [`h1_in_flight_shed_renders_503_then_recovers_after_release`]:
    /// the FIRST call (whichever connection reaches the shared, capacity-1
    /// `ConnAdmission` first) signals `entered_tx` — proving to the test
    /// that admission genuinely admitted it before a second connection's
    /// request can be shed against a real held slot — then blocks on
    /// `release_rx` until the test says to proceed. Every later call
    /// (there is exactly one, in this test: the follow-up request that
    /// proves `request_release` restored capacity) returns immediately.
    struct GatedOnceHandler {
        first_call_done: std::sync::atomic::AtomicBool,
        entered_tx: Mutex<Option<oneshot::Sender<()>>>,
        release_rx: Mutex<Option<oneshot::Receiver<()>>>,
    }

    impl SendPipe for GatedOnceHandler {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let is_first_call = !self
                .first_call_done
                .swap(true, std::sync::atomic::Ordering::SeqCst);
            let entered_tx = if is_first_call {
                self.entered_tx.lock().unwrap().take()
            } else {
                None
            };
            let release_rx = if is_first_call {
                self.release_rx.lock().unwrap().take()
            } else {
                None
            };
            async move {
                if let Some(entered_tx) = entered_tx {
                    let _ = entered_tx.send(());
                }
                if let Some(release_rx) = release_rx {
                    let _ = release_rx.await;
                }
                Ok(Response::new(200).with_body(Bytes::from_static(b"ok")))
            }
        }
    }

    /// The h1 admission defect this proves fixed: `max_in_flight_requests`
    /// used to be a silent no-op on h1 because the request loop never
    /// called `request_admit`/`request_release` — it only incremented a
    /// bare counter for telemetry. With a capacity-1 `ConnAdmission` shared
    /// across two connections: connection A's request is admitted and held
    /// open (proving admission happened before B ever dials); connection
    /// B's concurrent request is shed and gets a real in-band 503 (parsed
    /// status line, not a reset); B's connection stays usable — its NEXT
    /// request, sent after A releases, succeeds — proving `request_release`
    /// actually restored capacity instead of leaving the cap permanently
    /// exhausted.
    #[test]
    fn h1_in_flight_shed_renders_503_then_recovers_after_release() {
        block_on(async {
            let admission = ConnAdmission::new(1);
            let (entered_tx, entered_rx) = oneshot::channel::<()>();
            let (release_tx, release_rx) = oneshot::channel::<()>();
            let dispatch = into_handle(GatedOnceHandler {
                first_call_done: std::sync::atomic::AtomicBool::new(false),
                entered_tx: Mutex::new(Some(entered_tx)),
                release_rx: Mutex::new(Some(release_rx)),
            });
            let spec = Arc::new(HttpListenerSpec::default());
            let quiesce_response = Arc::new(QuiesceResponse {
                status: 503,
                retry_after: "1".into(),
            });

            let (server_a, mut client_a) = test_duplex();
            let (server_b, mut client_b) = test_duplex();

            let server_a_fut = serve_connection(
                server_a,
                dispatch.clone(),
                spec.clone(),
                admission.clone(),
                quiesce_response.clone(),
                None,
                None,
            );
            let server_b_fut = serve_connection(
                server_b,
                dispatch,
                spec,
                admission,
                quiesce_response,
                None,
                None,
            );

            let client_fut = async {
                client_a
                    .write_all(b"GET /a HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                    .await
                    .expect("client a write");

                // A's request is now admitted and blocked in the handler —
                // the ONLY way `entered_rx` resolves is past
                // `request_admit()` returning `Admit`.
                entered_rx.await.expect("handler signals entered");

                client_b
                    .write_all(b"GET /b1 HTTP/1.1\r\nHost: x\r\n\r\n")
                    .await
                    .expect("client b write (first, over capacity)");
                let (head_b1, body_b1) = read_http_response(&mut client_b).await;
                assert!(
                    head_b1.starts_with("HTTP/1.1 503"),
                    "the (N+1)th concurrent request must be shed in-band: {head_b1}"
                );
                assert_eq!(
                    body_b1, b"service unavailable",
                    "shed body matches h1's rendering of the h2-parity 503"
                );

                release_tx.send(()).expect("release send");
                let (head_a, body_a) = read_http_response(&mut client_a).await;
                assert!(
                    head_a.starts_with("HTTP/1.1 200"),
                    "the admitted request completes normally once released: {head_a}"
                );
                assert_eq!(body_a, b"ok");

                // Connection B stayed usable after its 503 (never reset) —
                // and capacity is now free, so this follow-up on the SAME
                // connection must succeed, proving `request_release`
                // genuinely gave the slot back.
                client_b
                    .write_all(b"GET /b2 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                    .await
                    .expect("client b write (second, after release)");
                let (head_b2, body_b2) = read_http_response(&mut client_b).await;
                assert!(
                    head_b2.starts_with("HTTP/1.1 200"),
                    "capacity freed by request_release must admit the follow-up: {head_b2}"
                );
                assert_eq!(body_b2, b"ok");
            };

            let (server_a_result, server_b_result, ()) =
                futures::join!(server_a_fut, server_b_fut, client_fut);
            server_a_result.expect("connection A completes cleanly");
            server_b_result.expect("connection B completes cleanly");
        });
    }

    #[test]
    fn expect_100_continue_is_accepted_before_the_body_is_read() {
        const EXPECT_CONTINUE_POST: &[u8] =
            b"POST /submit HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nExpect: 100-continue\r\nConnection: close\r\n\r\nhello";
        let (socket, writes) = TestSocket::new(EXPECT_CONTINUE_POST);
        block_on(serve_h1_connection(
            socket,
            into_handle(EchoBodyPipe),
            None,
            None,
        ))
        .expect("serve should complete cleanly on connection: close");
        let writes = writes.lock().unwrap();
        let wire = joined(&writes);
        assert!(
            writes.first().is_some_and(|first| {
                String::from_utf8_lossy(first).starts_with("HTTP/1.1 100")
            }),
            "interim 100 Continue must be written before the final response: {wire}"
        );
        assert!(
            wire.contains("HTTP/1.1 200"),
            "final response follows the interim: {wire}"
        );
        assert!(wire.ends_with("hello"), "body read after continue: {wire}");
    }

    #[test]
    fn keep_alive_serves_two_pipelined_requests_on_one_connection() {
        const TWO_REQUESTS: &[u8] =
            b"GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
        let (socket, writes) = TestSocket::new(TWO_REQUESTS);
        block_on(serve_h1_connection(
            socket,
            into_handle(FixedResponse {
                response: || Response::new(200).with_body(Bytes::from_static(b"ok")),
            }),
            None,
            None,
        ))
        .expect("serve should complete cleanly once the second request closes");
        let writes = writes.lock().unwrap();
        let wire = joined(&writes);
        assert_eq!(
            wire.matches("HTTP/1.1 200").count(),
            2,
            "both pipelined requests get a response on the same connection: {wire}"
        );
    }

    /// Proves the streaming dispatch path (`dispatch_streaming_request`'s
    /// no-runtime, in-task join between the body pump and the dispatch
    /// future) delivers a chunked request body to the Pipe with no
    /// tokio runtime — `serve_h1_connection` is called with
    /// `runtime = None`, so the old code's `tokio::task::spawn_local`
    /// would have panicked ("not currently running on a Tokio
    /// LocalSet") the moment the chunked body triggered auto-stream.
    #[test]
    fn chunked_request_body_streams_through_pipe_without_tokio_runtime() {
        const CHUNKED_POST: &[u8] = b"POST /upload HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let (socket, writes) = TestSocket::new(CHUNKED_POST);
        block_on(serve_h1_connection(
            socket,
            into_handle(EchoBodyPipe),
            None,
            None,
        ))
        .expect("serve should complete cleanly on connection: close");
        let wire = joined(&writes.lock().unwrap());
        assert!(wire.starts_with("HTTP/1.1 200"), "response head: {wire}");
        assert!(
            wire.ends_with("hello world"),
            "chunked body reassembled and echoed: {wire}"
        );
    }

    /// Read EOF after complete request framing must not cancel dispatch;
    /// the same read result can occur after a peer half-closes its write side.
    #[test]
    fn mid_dispatch_read_eof_preserves_the_response_and_does_not_cancel() {
        const SINGLE_GET: &[u8] = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let (socket, writes) = TestSocket::new_with_eof_after_input(SINGLE_GET);
        let cancel_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipe = CancelObservingPipe {
            cancel_seen: cancel_seen.clone(),
        };
        block_on(serve_h1_connection(socket, into_handle(pipe), None, None))
            .expect("serve should finish the response after request read eof");
        assert!(
            !cancel_seen.load(Ordering::SeqCst),
            "complete-request read eof must not fire request cancellation"
        );
        let response = joined(&writes.lock().unwrap());
        assert!(response.starts_with("HTTP/1.1 200"), "response head: {response}");
    }

    #[proxima::test(runtime = "tokio")]
    async fn mid_dispatch_read_reset_fires_cancel_and_is_observed_by_the_pipe() {
        let local_set = LocalSet::new();
        local_set
            .run_until(async {
                for (route, runtime_route) in [
                    ("buffered", false),
                    ("streaming_some_runtime", true),
                    ("streaming_no_runtime", false),
                ] {
                    let request = if route == "buffered" {
                        CLOSE_GET.to_vec()
                    } else {
                        half_close_request(route, "88888888888888888888888888888888")
                    };
                    let (socket, writes) = TestSocket::new_with_reset_after_input(&request);
                    let cancel_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let pipe = CancelAwaitingPipe {
                        cancel_seen: cancel_seen.clone(),
                    };
                    let admission = ConnAdmission::new(1);
                    let server_admission = admission.clone();
                    let runtime: Option<Arc<dyn Runtime>> = runtime_route
                        .then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
                    let spec = Arc::new(HttpListenerSpec::default());
                    let quiesce_response = Arc::new(QuiesceResponse {
                        status: 503,
                        retry_after: "1".into(),
                    });
                    tokio::task::spawn_local(async move {
                        serve_connection(
                            socket,
                            into_handle(pipe),
                            spec,
                            server_admission,
                            quiesce_response,
                            None,
                            runtime,
                        )
                        .await
                    })
                    .await
                    .expect("read-reset server task joins")
                    .expect("read reset closes each server route cleanly");
                    assert!(
                        cancel_seen.load(Ordering::SeqCst),
                        "read reset fires cancellation for {route}"
                    );
                    assert!(writes.lock().unwrap().is_empty(), "read reset writes no response for {route}");
                    assert_eq!(admission.in_flight(), 0, "read reset releases {route} admission");
                }
            })
            .await;
    }

    #[test]
    fn buffered_response_write_failure_propagates_io_error() {
        let (socket, writes) = TestSocket::new_with_write_failure(CLOSE_GET);
        let admission = ConnAdmission::new(1);
        let result = block_on(serve_connection(
            socket,
            into_handle(FixedResponse {
                response: || Response::new(200).with_body(Bytes::from_static(b"ok")),
            }),
            Arc::new(HttpListenerSpec::default()),
            admission.clone(),
            Arc::new(QuiesceResponse {
                status: 503,
                retry_after: "1".into(),
            }),
            None,
            None,
        ));
        assert!(matches!(result, Err(ProximaError::Io(_))));
        assert!(writes.lock().unwrap().is_empty(), "failed writer retains no bytes");
        assert_eq!(admission.in_flight(), 0, "writer failure releases admission");
    }

    #[proxima::test(runtime = "tokio")]
    async fn streaming_response_write_failure_propagates_io_error() {
        let local_set = LocalSet::new();
        local_set
            .run_until(async {
                for (route, runtime_route) in [
                    ("streaming_some_runtime", true),
                    ("streaming_no_runtime", false),
                ] {
                    let request = half_close_request(route, "66666666666666666666666666666666");
                    let (socket, writes) = TestSocket::new_with_write_failure(&request);
                    let admission = ConnAdmission::new(1);
                    let server_admission = admission.clone();
                    let runtime: Option<Arc<dyn Runtime>> = runtime_route
                        .then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
                    let spec = Arc::new(HttpListenerSpec::default());
                    let quiesce_response = Arc::new(QuiesceResponse {
                        status: 503,
                        retry_after: "1".into(),
                    });
                    let result = tokio::task::spawn_local(async move {
                        serve_connection(
                            socket,
                            into_handle(FixedResponse {
                                response: || {
                                    Response::new(200).with_body(Bytes::from_static(b"ok"))
                                },
                            }),
                            spec,
                            server_admission,
                            quiesce_response,
                            None,
                            runtime,
                        )
                        .await
                    })
                    .await
                    .expect("streaming writer-failure task joins");
                    assert!(matches!(result, Err(ProximaError::Io(_))));
                    assert!(writes.lock().unwrap().is_empty(), "failed writer retains no bytes for {route}");
                    assert_eq!(admission.in_flight(), 0, "writer failure releases {route} admission");
                }
            })
            .await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn dropping_driver_fires_cancel_and_releases_admission_once() {
        let local_set = LocalSet::new();
        local_set
            .run_until(async {
                for (route, runtime_route) in [
                    ("buffered", false),
                    ("streaming_some_runtime", true),
                    ("streaming_no_runtime", false),
                ] {
                    let request = if route == "buffered" {
                        CLOSE_GET.to_vec()
                    } else {
                        half_close_request(route, "77777777777777777777777777777777")
                    };
                    let (socket, _writes) = TestSocket::new(&request);
                    let signal = Arc::new(Mutex::new(None));
                    let (started_tx, started_rx) = oneshot::channel();
                    let pipe = PendingSignalCapturePipe {
                        signal: signal.clone(),
                        started: Arc::new(Mutex::new(Some(started_tx))),
                        cancel_observed: Arc::new(Mutex::new(None)),
                    };
                    let admission = ConnAdmission::new(1);
                    let server_admission = admission.clone();
                    let runtime: Option<Arc<dyn Runtime>> = runtime_route
                        .then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
                    let spec = Arc::new(HttpListenerSpec::default());
                    let quiesce_response = Arc::new(QuiesceResponse {
                        status: 503,
                        retry_after: "1".into(),
                    });
                    let server_task = tokio::task::spawn_local(async move {
                        serve_connection(
                            socket,
                            into_handle(pipe),
                            spec,
                            server_admission,
                            quiesce_response,
                            None,
                            runtime,
                        )
                        .await
                    });
                    started_rx
                        .await
                        .expect("dispatch reaches pending response before driver drop");
                    let captured_signal = signal
                        .lock()
                        .expect("captured cancellation signal lock")
                        .clone()
                        .expect("dispatch stores its request signal");
                    server_task.abort();
                    assert!(
                        server_task.await.expect_err("aborted driver task").is_cancelled(),
                        "test abort drops the driver future"
                    );
                    assert!(captured_signal.is_fired(), "dropping driver fires request cancellation");
                    assert_eq!(
                        admission.in_flight(),
                        0,
                        "driver drop releases {route} admission once"
                    );

                    let (socket, writes) = TestSocket::new(&request);
                    let signal = Arc::new(Mutex::new(None));
                    let (started_tx, started_rx) = oneshot::channel();
                    let (cancel_observed_tx, cancel_observed_rx) = oneshot::channel();
                    let pipe = PendingSignalCapturePipe {
                        signal: signal.clone(),
                        started: Arc::new(Mutex::new(Some(started_tx))),
                        cancel_observed: Arc::new(Mutex::new(Some(cancel_observed_tx))),
                    };
                    let admission = ConnAdmission::new(1);
                    let server_admission = admission.clone();
                    let runtime: Option<Arc<dyn Runtime>> = runtime_route
                        .then(|| Arc::new(LocalSetRuntime) as Arc<dyn Runtime>);
                    let spec = Arc::new(HttpListenerSpec::default());
                    let quiesce_response = Arc::new(QuiesceResponse {
                        status: 503,
                        retry_after: "1".into(),
                    });
                    let server_task = tokio::task::spawn_local(async move {
                        serve_connection(
                            socket,
                            into_handle(pipe),
                            spec,
                            server_admission,
                            quiesce_response,
                            None,
                            runtime,
                        )
                        .await
                    });
                    started_rx
                        .await
                        .expect("dispatch reaches pending response before explicit cancel");
                    let captured_signal = signal
                        .lock()
                        .expect("captured cancellation signal lock")
                        .clone()
                        .expect("dispatch stores its request signal");
                    captured_signal.fire();
                    cancel_observed_rx
                        .await
                        .expect("dispatch observes explicit cancellation");
                    server_task
                        .await
                        .expect("explicit-cancel server task joins")
                        .expect("explicit-cancel route completes");
                    assert!(
                        !writes.lock().unwrap().is_empty(),
                        "resolved explicit-cancel dispatch writes its response for {route}"
                    );
                    assert_eq!(
                        admission.in_flight(),
                        0,
                        "explicit cancellation releases {route} admission once"
                    );
                }
            })
            .await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn peer_half_close_real_tcp_preserves_response_across_all_h1_routes() {
        let (trace_collector, trace_recorder) = half_close_trace_collector();
        let local_set = LocalSet::new();
        let (
            observations,
            sender_drop_cases,
            incomplete_body_cases,
            pipeline_cases,
            dispatch_error_cases,
            body_reset_cases,
        ) = local_set
            .run_until(async {
                let mut observations = Vec::new();
                for (route, trace_id, runtime_route) in [
                    ("buffered", "11111111111111111111111111111111", false),
                    ("streaming_some_runtime", "22222222222222222222222222222222", true),
                    ("streaming_no_runtime", "33333333333333333333333333333333", false),
                ] {
                    observations.push(
                        run_peer_half_close_case(route, trace_id, false, runtime_route).await,
                    );
                    observations.push(
                        run_peer_half_close_case(route, trace_id, true, runtime_route).await,
                    );
                }
            let sender_drop_cases = vec![
                (true, run_response_sender_dropped_case(true).await),
                (false, run_response_sender_dropped_case(false).await),
            ];
                let incomplete_body_cases = vec![
                    run_incomplete_body_eof_case(true).await,
                    run_incomplete_body_eof_case(false).await,
                ];
                let pipeline_cases = vec![
                    run_positive_pipeline_byte_case("buffered", false, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").await,
                    run_positive_pipeline_byte_case("streaming_some_runtime", true, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").await,
                    run_positive_pipeline_byte_case("streaming_no_runtime", false, "cccccccccccccccccccccccccccccccc").await,
                ];
                let dispatch_error_cases = vec![
                    run_dispatch_error_case("buffered", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", false).await,
                    run_dispatch_error_case("streaming_some_runtime", "ffffffffffffffffffffffffffffffff", true).await,
                    run_dispatch_error_case("streaming_no_runtime", "abababababababababababababababab", false).await,
                ];
                let body_reset_cases = vec![
                    ("streaming_some_runtime", run_streaming_body_reset_case(true).await),
                    ("streaming_no_runtime", run_streaming_body_reset_case(false).await),
                ];
                (
                    observations,
                    sender_drop_cases,
                    incomplete_body_cases,
                    pipeline_cases,
                    dispatch_error_cases,
                    body_reset_cases,
                )
            })
            .await;

        proxima_telemetry::debug!(
            trace_marker = "complete",
            "peer_half_close_trace_flush"
        );
        while trace_recorder.drain() > 0 {}
        let trace_records = trace_collector.wait_for_marker(b"peer_half_close_trace_flush");

        let mut contract_failures = Vec::new();
        let trace_ids = [
            "11111111111111111111111111111111",
            "22222222222222222222222222222222",
            "33333333333333333333333333333333",
        ];
        for route_index in 0..3 {
            let control = &observations[route_index * 2];
            let treatment = &observations[route_index * 2 + 1];
            if control.request != treatment.request {
                contract_failures.push(format!("{} request differs from matched control", treatment.evidence_name));
            }
            if control.dispatch_calls != 1 || treatment.dispatch_calls != 1 {
                contract_failures.push(format!(
                    "{} dispatch counts are control={} treatment={}",
                    treatment.evidence_name, control.dispatch_calls, treatment.dispatch_calls
                ));
            }
            if control.eof_observed || !treatment.eof_observed {
                contract_failures.push(format!(
                    "{} EOF observation is control={} treatment={}",
                    treatment.evidence_name, control.eof_observed, treatment.eof_observed
                ));
            }
            if control.server_result.is_err() || treatment.server_result.is_err() {
                contract_failures.push(format!(
                    "{} server results are control={:?} treatment={:?}",
                    treatment.evidence_name, control.server_result, treatment.server_result
                ));
            }
            if control.admission_in_flight != 0 || treatment.admission_in_flight != 0 {
                contract_failures.push(format!(
                    "{} admission remains control={} treatment={}",
                    treatment.evidence_name,
                    control.admission_in_flight,
                    treatment.admission_in_flight
                ));
            }
            let expected_response = format!(
                "HTTP/1.1 200 OK\r\ntraceparent: 00-{}-0123456789abcdef-01\r\ncontent-length: 16\r\n\r\nhalf-close-ready",
                trace_ids[route_index]
            );
            if control.response != expected_response.as_bytes() {
                contract_failures.push(format!(
                    "{} control response differs from exact expected HTTP wire bytes",
                    control.evidence_name
                ));
            }
            if treatment.response != control.response {
                contract_failures.push(format!(
                    "{} treatment is {} bytes and control is {} bytes",
                    treatment.evidence_name,
                    treatment.response.len(),
                    control.response.len()
                ));
            }
            if treatment.response != expected_response.as_bytes() {
                contract_failures.push(format!(
                    "{} treatment response differs from exact expected HTTP wire bytes",
                    treatment.evidence_name
                ));
            }
        }
        for (admitted, sender_drop) in &sender_drop_cases {
            if !sender_drop.1.is_empty() {
                contract_failures.push(format!(
                    "{} response sender wrote {} response bytes",
                    if *admitted { "admitted" } else { "shed" },
                    sender_drop.1.len()
                ));
            }
            if sender_drop.2 != 0 {
                contract_failures.push(format!(
                    "{} response sender retained admission in_flight={}",
                    if *admitted { "admitted" } else { "shed" },
                    sender_drop.2
                ));
            }
        }
        for incomplete in &incomplete_body_cases {
            if !incomplete.cancel_seen {
                contract_failures.push(format!(
                    "{} dispatch did not observe request cancellation",
                    incomplete.evidence_name
                ));
            }
            if !incomplete.response.is_empty() {
                contract_failures.push(format!(
                    "{} wrote {} response bytes after body-incomplete EOF",
                    incomplete.evidence_name,
                    incomplete.response.len()
                ));
            }
            if !incomplete.eof_observed {
                contract_failures.push(format!(
                    "{} did not observe accepted-socket read EOF",
                    incomplete.evidence_name
                ));
            }
            if incomplete.admission_in_flight != 0 {
                contract_failures.push(format!(
                    "{} retained admission in_flight={}",
                    incomplete.evidence_name, incomplete.admission_in_flight
                ));
            }
        }
        for pipeline in &pipeline_cases {
            if !pipeline.byte_observed {
                contract_failures.push(format!(
                    "{} did not observe a positive read in response wait",
                    pipeline.route
                ));
            }
            if pipeline.dispatch_calls != 2 {
                contract_failures.push(format!(
                    "{} pipeline dispatch count was {}",
                    pipeline.route, pipeline.dispatch_calls
                ));
            }
            if pipeline.response
                .windows(b"HTTP/1.1 200 OK".len())
                .filter(|window| *window == b"HTTP/1.1 200 OK")
                .count()
                != 2
                || pipeline
                    .response
                    .windows(b"half-close-ready".len())
                    .filter(|window| *window == b"half-close-ready")
                    .count()
                    != 2
            {
                contract_failures.push(format!(
                    "{} did not return both pipelined responses: {:?}",
                    pipeline.route, pipeline.response
                ));
            }
            if pipeline.admission_in_flight != 0 {
                contract_failures.push(format!(
                    "{} pipeline retained admission={}",
                    pipeline.route, pipeline.admission_in_flight
                ));
            }
        }
        let dispatch_error_body = error_response_body(&ProximaError::Upstream(
            "controlled dispatch failure".into(),
        ));
        for (route, (response, admission_in_flight)) in [
            ("buffered", &dispatch_error_cases[0]),
            ("streaming_some_runtime", &dispatch_error_cases[1]),
            ("streaming_no_runtime", &dispatch_error_cases[2]),
        ] {
            let expected_response = format!(
                "HTTP/1.1 502 Bad Gateway\r\ncontent-type: text/plain\r\ncontent-length: {}\r\n\r\n{}",
                dispatch_error_body.len(),
                String::from_utf8_lossy(&dispatch_error_body)
            );
            if response != expected_response.as_bytes() {
                contract_failures.push(format!(
                    "{route} dispatch-error response differs from exact expected bytes: {response:?}"
                ));
            }
            if *admission_in_flight != 0 {
                contract_failures.push(format!(
                    "{route} dispatch error retained admission={admission_in_flight}"
                ));
            }
        }
        for (route, (cancel_seen, writes, admission_in_flight)) in &body_reset_cases {
            assert!(*cancel_seen, "streaming body reset cancellation reaches dispatch for {route}");
            assert!(writes.is_empty(), "streaming body reset writes no response for {route}");
            assert_eq!(*admission_in_flight, 0, "streaming body reset releases {route} admission");
        }
        assert!(
            trace_records
                .windows(b"h1 request race selected an arm".len())
                .any(|window| window == b"h1 request race selected an arm"),
            "buffered branch trace is retained"
        );
        assert!(
            trace_records
                .windows(b"h1 streaming response race selected an arm".len())
                .any(|window| window == b"h1 streaming response race selected an arm"),
            "streaming response branch trace is retained"
        );
        let structured_trace = std::str::from_utf8(&trace_records)
            .expect("structured telemetry is UTF-8 JSONL");
        let trace_events = structured_trace
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid trace JSON"))
            .collect::<Vec<_>>();
        for (route_index, route) in [
            "buffered",
            "streaming_some_runtime",
            "streaming_no_runtime",
        ]
        .into_iter()
        .enumerate()
        {
            let eof_arms = trace_events
                .iter()
                .filter(|event| {
                    event["body"] == "h1 request race selected an arm"
                        || event["body"] == "h1 streaming response race selected an arm"
                })
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["race_arm"] == "read_eof")
                .count();
            assert_eq!(eof_arms, 1, "one observed read EOF arm for {route}");
            let retained_response_events = trace_events
                .iter()
                .filter(|event| {
                    event["body"] == "h1 request read eof retained the response task"
                        || event["body"] == "h1 streaming request read eof retained the response task"
                })
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["cancel_fired"] == false)
                .filter(|event| event["tags"]["response_writer_entered"] == false)
                .count();
            assert_eq!(
                retained_response_events, 1,
                "one recorded response retained after read EOF for {route}"
            );
            let writer_entries = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 response writer entered")
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["response_writer_entered"] == true)
                .count();
            assert_eq!(writer_entries, 4, "treatment, control, and pipeline requests enter writer for {route}");

            let trace_tag = expected_trace_tag(trace_ids[route_index]);
            let release_positions = trace_events
                .iter()
                .enumerate()
                .filter(|(_, event)| event["body"] == "h1 request admission ownership ended")
                .filter(|(_, event)| event["tags"]["route"] == route)
                .filter(|(_, event)| event["tags"]["trace_id"] == trace_tag)
                .filter(|(_, event)| event["tags"]["admission_state"] == "admitted")
                .filter(|(_, event)| event["tags"]["admission_release"] == true)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            let writer_positions = trace_events
                .iter()
                .enumerate()
                .filter(|(_, event)| event["body"] == "h1 response writer entered")
                .filter(|(_, event)| event["tags"]["route"] == route)
                .filter(|(_, event)| event["tags"]["trace_id"] == trace_tag)
                .filter(|(_, event)| event["tags"]["response_writer_entered"] == true)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            assert_eq!(release_positions.len(), 2, "one owner release per {route} request");
            assert_eq!(writer_positions.len(), 2, "both {route} requests enter the writer");
            assert!(
                release_positions
                    .iter()
                    .zip(&writer_positions)
                    .all(|(release, writer)| release < writer),
                "each correlated {route} admission release precedes its writer entry"
            );
        }
        for (route, trace_id) in [
            ("buffered", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            ("streaming_some_runtime", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
            ("streaming_no_runtime", "cccccccccccccccccccccccccccccccc"),
        ] {
            let trace_tag = expected_trace_tag(trace_id);
            let pipelined_byte_events = trace_events
                .iter()
                .filter(|event| {
                    event["body"] == "h1 request race selected an arm"
                        || event["body"] == "h1 streaming response race selected an arm"
                })
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| event["tags"]["race_arm"] == "read_bytes")
                .filter(|event| event["tags"]["read_result"] == "bytes")
                .count();
            assert_eq!(pipelined_byte_events, 1, "one correlated positive pipeline byte for {route}");
        }
        for (admitted, trace_id) in [
            (true, "44444444444444444444444444444444"),
            (false, "dddddddddddddddddddddddddddddddd"),
        ] {
            let trace_tag = expected_trace_tag(trace_id);
            let release_events = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 request admission ownership ended")
                .filter(|event| event["tags"]["route"] == "streaming_some_runtime")
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| {
                    event["tags"]["admission_state"]
                        == if admitted { "admitted" } else { "shed" }
                })
                .filter(|event| event["tags"]["admission_release"] == admitted)
                .filter(|event| event["tags"]["release_owner"] == "response_sender_dropped")
                .count();
            assert_eq!(
                release_events, 1,
                "one correct admission ownership event for sender-drop admitted={admitted}"
            );
            let sender_drop_events = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 streaming response sender dropped before response")
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| {
                    event["tags"]["admission_state"]
                        == if admitted { "admitted" } else { "shed" }
                })
                .count();
            assert_eq!(sender_drop_events, 1, "sender-drop terminal record exists");
        }
        for (route, trace_id) in [
            ("buffered", "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"),
            ("streaming_some_runtime", "ffffffffffffffffffffffffffffffff"),
            ("streaming_no_runtime", "abababababababababababababababab"),
        ] {
            let trace_tag = expected_trace_tag(trace_id);
            let release_events = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 request admission ownership ended")
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| event["tags"]["admission_state"] == "admitted")
                .filter(|event| event["tags"]["admission_release"] == true)
                .filter(|event| event["tags"]["release_owner"] == "response_resolved_before_writer")
                .count();
            assert_eq!(release_events, 1, "dispatch error releases {route} admission once");
        }
        for route in ["streaming_some_runtime", "streaming_no_runtime"] {
            let trace_tag = expected_trace_tag("12121212121212121212121212121212");
            let body_read_error_events = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 streaming request read failed before body end")
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| event["tags"]["lifecycle_phase"] == "body_incomplete")
                .filter(|event| event["tags"]["read_result"] == "error")
                .filter(|event| {
                    event["tags"]["read_error"]
                        .as_str()
                        .is_some_and(|read_error| read_error.contains("ConnectionReset"))
                })
                .count();
            assert_eq!(body_read_error_events, 1, "one observed streaming body reset for {route}");
            let release_events = trace_events
                .iter()
                .filter(|event| event["body"] == "h1 request admission ownership ended")
                .filter(|event| event["tags"]["route"] == route)
                .filter(|event| event["tags"]["trace_id"] == trace_tag)
                .filter(|event| event["tags"]["release_owner"] == "body_read_error")
                .filter(|event| event["tags"]["admission_release"] == true)
                .count();
            assert_eq!(release_events, 1, "one admission release after body reset for {route}");
        }
        assert!(
            contract_failures.is_empty(),
            "observed HTTP/1 contract failures after all evidence capture: {contract_failures:#?}"
        );
    }
}
