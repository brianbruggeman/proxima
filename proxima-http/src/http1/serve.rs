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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use futures::FutureExt;
use futures::StreamExt;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tracing::{debug, error};

use proxima_core::ProximaError;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::body::RequestStream;
use proxima_primitives::pipe::SendPipe;
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
    let in_flight = Arc::new(AtomicU64::new(0));
    let quiescing = Arc::new(AtomicBool::new(false));
    let quiesce_response = Arc::new(QuiesceResponse {
        status: 503,
        retry_after: "1".into(),
    });
    serve_connection(
        socket,
        dispatch,
        spec,
        in_flight,
        quiescing,
        quiesce_response,
        None,
        runtime,
    )
    .await
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
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
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
                    &in_flight,
                    &quiescing,
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
                if quiescing.load(Ordering::Relaxed) {
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
        if quiescing.load(Ordering::Relaxed) {
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

        // Build the proxima::Request from Connection accessors and
        // dispatch through the Pipe chain. One Bytes::copy_from_slice
        // for each of: method, path, body. Headers are copied into a
        // HeaderList — Tier 3 (the bump arena) would absorb this if we
        // moved Request fields off owned Bytes, but the existing
        // proxima::Request shape uses Bytes today so we pay it for now.
        let in_flight_now = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel = proxima_core::signal::Signal::new();
        let cancel_guard = cancel.clone().guard();
        let mut request = build_proxima_request(&mut connection, &spec);
        request.context.cancel = cancel.clone();
        request.context.peer = peer.clone();
        let trace_id = request.context.trace_id.clone();
        let telemetry = request.context.telemetry.clone();
        telemetry.gauge_set(
            "proxima.requests.in_flight",
            &proxima_primitives::pipe::telemetry_surface::Labels::empty(),
            in_flight_now as i64,
        );
        // Race the dispatch against socket EOF detection. If the
        // client disconnects mid-dispatch, the read returns 0; we
        // fire `cancel.fire()` and then poll the dispatch future
        // to completion so the Pipe has a chance to observe the
        // cancellation and clean up.
        let dispatch_future = dispatch_request(&dispatch, request);
        tokio::pin!(dispatch_future);
        let mut watch_buf = [0_u8; 1];
        let outcome = tokio::select! {
            response = &mut dispatch_future => response,
            read = reader.read(&mut watch_buf) => {
                match read {
                    Ok(0) => {
                        debug!("client disconnected during dispatch");
                        cancel.fire();
                        // Give the Pipe a poll cycle to observe
                        // the cancel signal before we drop its
                        // future. Without this, drop happens before
                        // the cancelled arm executes and the
                        // Pipe's cleanup code never runs.
                        let _ = (&mut dispatch_future).await;
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    Ok(_n) => {
                        // Pipelined bytes from the client. Buffer
                        // the byte by feeding it back into
                        // Connection; the next iteration's poll
                        // picks up the trailing request after this
                        // one's response is written.
                        connection.feed_bytes(&watch_buf[..1]);
                        (&mut dispatch_future).await
                    }
                    Err(error) => {
                        debug!(?error, "read error during dispatch");
                        cancel.fire();
                        let _ = (&mut dispatch_future).await;
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                        return Ok(());
                    }
                }
            }
        };
        in_flight.fetch_sub(1, Ordering::Relaxed);
        // Successful response — disarm the cancel guard so a normal
        // return doesn't fire it.
        cancel_guard.disarm();

        let upgrade_after_write = match outcome {
            Ok(response) => {
                write_response(
                    &mut writer,
                    &mut out,
                    &mut connection,
                    response,
                    trace_id.as_deref(),
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

/// Streaming-mode dispatch path. Reached when the connection's
/// auto-stream policy fires (chunked transfer or large
/// Content-Length). Builds a mpsc-backed `Body`, spawns
/// `Pipe::call` on the current core's local set so it runs
/// concurrently with body ingestion, pumps body chunks into the
/// channel (backpressure: `send().await` parks when full), and on
/// `BodyEnd` drops the sender to signal end-of-stream. Then awaits
/// the response from a oneshot, racing it against a single-byte
/// reader.read for client-disconnect detection (mirrors the
/// buffered path's cancel-token race).
#[allow(clippy::too_many_arguments)]
async fn dispatch_streaming_request<R, W>(
    connection: &mut Connection,
    reader: &mut R,
    writer: &mut W,
    read_buf: &mut [u8],
    out: &mut Vec<u8>,
    dispatch: &PipeHandle,
    spec: &Arc<HttpListenerSpec>,
    in_flight: &Arc<AtomicU64>,
    quiescing: &Arc<AtomicBool>,
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
    if quiescing.load(Ordering::Relaxed) {
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

    let in_flight_now = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
    let cancel = proxima_core::signal::Signal::new();
    let cancel_guard = cancel.clone().guard();

    let (body_tx, body_rx) =
        tokio::sync::mpsc::channel::<Result<Bytes, ProximaError>>(BODY_CHANNEL_DEPTH);

    let (mut request, request_trailers_slot) = build_streaming_request(connection, spec, body_rx);
    request.context.cancel = cancel.clone();
    request.context.peer = peer.clone();
    let trace_id = request.context.trace_id.clone();
    let telemetry = request.context.telemetry.clone();
    telemetry.gauge_set(
        "proxima.requests.in_flight",
        &proxima_primitives::pipe::telemetry_surface::Labels::empty(),
        in_flight_now as i64,
    );

    let (resp_tx, mut resp_rx) = oneshot::channel::<Result<Response<Bytes>, ProximaError>>();
    let dispatch_clone = dispatch.clone();
    let pipe_task = async move {
        let response = dispatch_request(&dispatch_clone, request).await;
        let _ = resp_tx.send(response);
    };
    match runtime {
        Some(rt) => rt.spawn_on_current_core(Box::pin(pipe_task)),
        None => {
            tokio::task::spawn_local(pipe_task);
        }
    }

    // Pump body chunks → bounded mpsc. Backpressure is implicit:
    // body_tx.send().await parks when the channel is full, which
    // pauses reader.read and lets the kernel TCP window close on the
    // peer.
    let pump_outcome = pump_body_stream(connection, reader, read_buf, &body_tx, &cancel).await;
    // Publish chunked request trailers into the stream's slot BEFORE
    // closing the channel, so the Pipe's drain completes only after the
    // trailers are visible (deterministic — no read-before-publish race).
    let captured_trailers = connection.take_trailers();
    if !captured_trailers.is_empty() {
        let mut trailers = proxima_primitives::pipe::header_list::HeaderList::new();
        for (name, value) in captured_trailers {
            trailers.insert(name, value);
        }
        if let Ok(mut guard) = request_trailers_slot.lock() {
            *guard = Some(trailers);
        }
    }
    drop(body_tx);

    match pump_outcome {
        Ok(()) | Err(PumpError::PipeDroppedBody) => {}
        Err(PumpError::ClientEof | PumpError::ConnectionClosed | PumpError::Cancelled) => {
            cancel.fire();
            // Give the Pipe a poll cycle to observe cancel before
            // dropping its task — mirrors the buffered path's
            // dispatch_future-then-poll pattern.
            let _ = resp_rx.await;
            in_flight.fetch_sub(1, Ordering::Relaxed);
            return Ok(StreamingOutcome::Close);
        }
        Err(PumpError::Decode) => {
            // Body decoder rejected the wire bytes — fire cancel and
            // surface a 400 to the client.
            cancel.fire();
            let _ = resp_rx.await;
            in_flight.fetch_sub(1, Ordering::Relaxed);
            write_minimal_error(writer, out, 400, "Bad Request").await?;
            return Ok(StreamingOutcome::Close);
        }
        Err(PumpError::Io(error)) => {
            cancel.fire();
            let _ = resp_rx.await;
            in_flight.fetch_sub(1, Ordering::Relaxed);
            return Err(ProximaError::Io(error));
        }
    }

    // Response phase — race resp_rx against socket EOF detection so a
    // disconnect during a slow Pipe still triggers cancel and
    // tears down cleanly.
    let mut watch_buf = [0_u8; 1];
    let outcome = tokio::select! {
        response = &mut resp_rx => response,
        read = reader.read(&mut watch_buf) => {
            match read {
                Ok(0) => {
                    debug!("client disconnected during streaming dispatch");
                    cancel.fire();
                    let _ = (&mut resp_rx).await;
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    return Ok(StreamingOutcome::Close);
                }
                Ok(_n) => {
                    // First byte of a pipelined request — queue it.
                    connection.feed_bytes(&watch_buf[..1]);
                    (&mut resp_rx).await
                }
                Err(error) => {
                    debug!(?error, "read error during streaming dispatch");
                    cancel.fire();
                    let _ = (&mut resp_rx).await;
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                    return Ok(StreamingOutcome::Close);
                }
            }
        }
    };
    in_flight.fetch_sub(1, Ordering::Relaxed);
    cancel_guard.disarm();

    let response_result = match outcome {
        Ok(response) => response,
        Err(_recv) => {
            error!("streaming dispatch task dropped response sender");
            return Ok(StreamingOutcome::Close);
        }
    };

    let upgrade_after_write = match response_result {
        Ok(response) => {
            write_response(writer, out, connection, response, trace_id.as_deref()).await?
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

/// Drive the connection's body decoder, copying each emitted chunk
/// into the bounded `body_tx`. Returns on `BodyEnd` (success) or on
/// any of the `PumpError` conditions. The caller is responsible for
/// dropping `body_tx` and reacting to the outcome.
async fn pump_body_stream<R>(
    connection: &mut Connection,
    reader: &mut R,
    read_buf: &mut [u8],
    body_tx: &tokio::sync::mpsc::Sender<Result<Bytes, ProximaError>>,
    cancel: &proxima_core::signal::Signal,
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
                tokio::select! {
                    send = body_tx.send(Ok(chunk)) => {
                        if send.is_err() {
                            return Err(PumpError::PipeDroppedBody);
                        }
                    }
                    () = cancel.fired() => return Err(PumpError::Cancelled),
                }
            }
            ConnPoll::BodyEnd => return Ok(()),
            ConnPoll::NeedInput => {
                let read = tokio::select! {
                    read = reader.read(read_buf) => read,
                    () = cancel.fired() => return Err(PumpError::Cancelled),
                };
                match read {
                    Ok(0) => return Err(PumpError::ClientEof),
                    Ok(n) => connection.feed_bytes(&read_buf[..n]),
                    Err(error) => return Err(PumpError::Io(error)),
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
/// body is a stream wrapping the bounded mpsc receiver — chunks flow
/// from the listener's pump task to the Pipe as they arrive.
fn build_streaming_request(
    connection: &Connection,
    _spec: &HttpListenerSpec,
    body_rx: tokio::sync::mpsc::Receiver<Result<Bytes, ProximaError>>,
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
    let body_stream = futures::stream::unfold(body_rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    // Trailers slot the chunked decoder populates at body-end; the Pipe's
    // `body_bytes()` folds it into `headers` after draining (RFC 7230
    // §4.1.2 request trailers on the streaming path).
    let trailers_slot: proxima_primitives::pipe::body::TrailersSlot = Arc::new(std::sync::Mutex::new(None));
    let request = Request {
        method,
        path,
        query,
        metadata: headers,
        payload: Bytes::new(),
        stream: Some(RequestStream::new(body_stream).with_trailers_slot(trailers_slot.clone())),
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
) -> Result<Option<proxima_primitives::pipe::upgrade::UpgradeHandler>, ProximaError>
where
    W: AsyncWrite + Unpin,
{
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
    let trailers_now = trailers_slot
        .as_ref()
        .and_then(|slot| slot.lock().ok().and_then(|guard| guard.clone()));
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
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll};

    use futures::executor::block_on;
    use proxima_primitives::pipe::body::ResponseStream;
    use proxima_primitives::pipe::handler::into_handle;

    use super::*;

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
    }

    impl TestSocket {
        fn new(request: &[u8]) -> (Self, Arc<Mutex<Vec<Vec<u8>>>>) {
            let writes = Arc::new(Mutex::new(Vec::new()));
            let socket = Self {
                input: request.to_vec(),
                read_pos: 0,
                writes: writes.clone(),
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
                return Poll::Pending;
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
            self.get_mut().writes.lock().unwrap().push(buf.to_vec());
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

        #[allow(clippy::manual_async_fn)]
        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let response = (self.response)();
            async move { Ok(response) }
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
}
