//! io_uring-backed HTTP/1 listener path. Cfg-gated to
//! `target_os = "linux"` + `feature = "io-uring"`; everywhere else
//! this module compiles to an empty stub so the default listener
//! drives the I/O via tokio's epoll reactor.
//!
//! Why a parallel listener: tokio-uring's I/O is owned-buffer —
//! every `read` / `write_all` consumes a `Vec` (or any `BoundedBuf`)
//! and returns it on completion. The substrate's main
//! `serve_connection` is generic over `AsyncRead + AsyncWrite +
//! Send + Unpin + 'static`. Bridging owned-buffer reads to the
//! AsyncRead poll-based contract requires either self-referential
//! pinning (the read future borrows the stream) or wrapping the
//! stream in `Rc`, which kills `Send`. The Send bound cascades
//! through TLS, upgrade, and the streaming dispatch helpers, so
//! relaxing it on the main path would be invasive.
//!
//! Instead, this module is a focused listener that:
//!
//! - Binds via `tokio_uring::net::TcpListener` so accept itself is
//!   io_uring-driven.
//! - Per connection: drives the same `Connection` state machine but
//!   reads via owned `Vec` buffers handed to `stream.read(buf).await`
//!   and writes via `stream.write_all(out).await`. Buffer cycling
//!   reuses one read Vec + one write Vec per connection.
//! - **TLS termination supported** via `UringAsyncStream` — an
//!   `AsyncRead`/`AsyncWrite` adapter over `Rc<tokio_uring::TcpStream>`
//!   that lets `tokio_rustls::TlsAcceptor` wrap the io_uring stream.
//!   The TLS path uses `serve_connection_uring_async` (AsyncRead-based)
//!   while the plaintext path keeps the owned-buffer fast path.
//! - Buffered request path only — no streaming dispatch (the mpsc
//!   body channel + dispatched Pipe::call assumes a Send future)
//!   and no CONNECT/Upgrade hijack (HijackStream requires Send).
//!   These remain documented follow-ups.
//!
//! What this delivers: io_uring drives accept + every read + every
//! write on a per-connection conversation. The kernel never sees an
//! `epoll_wait`; submissions and completions flow through the
//! shared submission/completion queues. On a workload that's
//! limited by syscall throughput (many small requests / responses)
//! this is the expected win.

#![cfg(all(target_os = "linux", feature = "io-uring"))]

use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, select};
use tokio::io::{AsyncRead, AsyncWrite};
#[cfg(feature = "tls")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::compat::TokioAsyncReadCompatExt;
use tracing::{debug, error, warn};

use crate::body::RequestStream;
use crate::error::ProximaError;
use crate::h1_body::BodyFraming;
use crate::h1_connection::{Advanced, AutoStreamPolicy, Connection};
use crate::header_list::HeaderList;
use crate::pipe::PipeHandle;
use crate::request::{Request, RequestContext, Response};
use proxima_http::http1::{HttpListenerSpec, UringAsyncStream};
use proxima_primitives::pipe::SendPipe;

/// Drive accept on the io_uring TCP listener and spawn each
/// connection onto the local executor (the tokio_uring runtime is
/// current-thread per worker, so `spawn_local` is the right shape).
///
/// Mirrors the surface of the default tokio-backed listener's
/// `serve` so the per-core runtime can route to this function
/// behind the io-uring feature flag without changing higher-level
/// orchestration.
pub async fn serve_uring(
    bind: SocketAddr,
    dispatch: PipeHandle,
    spec: Arc<HttpListenerSpec>,
    raw_spec: &serde_json::Value,
    telemetry: crate::telemetry::TelemetryHandle,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<(), ProximaError> {
    let listener = tokio_uring::net::TcpListener::bind(bind).map_err(ProximaError::Io)?;
    // TLS termination wraps the io_uring stream via the
    // AsyncRead/AsyncWrite adapter below, then through
    // tokio_rustls's acceptor. Reads/writes inside the TLS session
    // route through the adapter and end up as owned-buffer io_uring
    // submissions one layer down.
    #[cfg(feature = "tls")]
    let tls_acceptor: Option<tokio_rustls::TlsAcceptor> =
        match crate::tls::config_from_spec_value(raw_spec.get(crate::tls::SPEC_KEY)) {
            Ok(Some(config)) => Some(crate::tls::build_acceptor(&config)?),
            Ok(None) => None,
            Err(error) => return Err(error),
        };
    let drain_timeout_ms = raw_spec
        .get("drain_timeout_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(5_000);
    let quiesce_duration_ms = raw_spec
        .get("quiesce_duration_ms")
        .and_then(serde_json::Value::as_u64);
    let quiesce_status = raw_spec
        .get("quiesce_status")
        .and_then(serde_json::Value::as_u64)
        .map(|raw| raw as u16)
        .unwrap_or(503);
    let quiesce_retry_after = raw_spec
        .get("quiesce_retry_after")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let quiesce_response = Arc::new(proxima_primitives::pipe::quiesce::QuiesceResponse {
        status: quiesce_status,
        retry_after: quiesce_retry_after,
    });
    let listener_label: Arc<[u8]> = Arc::from(
        raw_spec
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("http")
            .as_bytes(),
    );
    debug!(?bind, "io_uring http listener bound");
    let in_flight = Arc::new(AtomicU64::new(0));
    let quiescing = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "tls")]
    let tls_acceptor_outer = tls_acceptor;
    loop {
        select! {
            _ = (&mut shutdown).fuse() => break,
            accepted = listener.accept().fuse() => match accepted {
                Ok((stream, peer)) => {
                    handle_accept_uring(
                        stream,
                        peer,
                        &dispatch,
                        &spec,
                        &in_flight,
                        &quiescing,
                        &quiesce_response,
                        &telemetry,
                        &listener_label,
                        #[cfg(feature = "tls")]
                        tls_acceptor_outer.as_ref(),
                    );
                }
                Err(error) => warn!(?error, "io_uring accept failed"),
            }
        }
    }
    if let Some(quiesce_ms) = quiesce_duration_ms
        && quiesce_ms > 0
    {
        quiescing.store(true, Ordering::Relaxed);
        debug!(quiesce_ms, "io_uring listener entering quiesce window");
        #[cfg(feature = "sync-wrappers")]
        let deadline = crate::time::Instant::now() + std::time::Duration::from_millis(quiesce_ms);
        #[cfg(not(feature = "sync-wrappers"))]
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(quiesce_ms);
        loop {
            // futures::select! does not accept #[cfg] on its arms; build the
            // cfg-selected sleep future before the macro and select on it.
            #[cfg(feature = "sync-wrappers")]
            let sleep = crate::time::sleep_until(deadline).fuse();
            #[cfg(not(feature = "sync-wrappers"))]
            let sleep = tokio::time::sleep_until(deadline).fuse();
            futures::pin_mut!(sleep);
            select! {
                _ = sleep => break,
                accepted = listener.accept().fuse() => match accepted {
                    Ok((stream, peer)) => {
                        handle_accept_uring(
                            stream,
                            peer,
                            &dispatch,
                            &spec,
                            &in_flight,
                            &quiescing,
                            &quiesce_response,
                            &telemetry,
                            &listener_label,
                            #[cfg(feature = "tls")]
                            tls_acceptor_outer.as_ref(),
                        );
                    }
                    Err(error) => warn!(?error, "io_uring accept during quiesce failed"),
                },
            }
        }
    }
    drain_in_flight_uring(
        &in_flight,
        std::time::Duration::from_millis(drain_timeout_ms),
    )
    .await;
    // explicit drop so port is released before the caller can rebind.
    drop(listener);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_accept_uring(
    stream: tokio_uring::net::TcpStream,
    peer: SocketAddr,
    dispatch: &PipeHandle,
    spec: &Arc<HttpListenerSpec>,
    in_flight: &Arc<AtomicU64>,
    quiescing: &Arc<AtomicBool>,
    quiesce_response: &Arc<proxima_primitives::pipe::quiesce::QuiesceResponse>,
    telemetry: &crate::telemetry::TelemetryHandle,
    listener_label: &Arc<[u8]>,
    #[cfg(feature = "tls")] tls_acceptor: Option<&tokio_rustls::TlsAcceptor>,
) {
    let _ = stream.set_nodelay(true);
    let listener_labels = crate::telemetry::Labels::from_pairs(&[(
        "listener",
        std::str::from_utf8(listener_label).unwrap_or(""),
    )]);
    telemetry.counter_inc("proxima.connections_accepted_total", &listener_labels, 1);
    let dispatch = dispatch.clone();
    let spec = spec.clone();
    let in_flight = in_flight.clone();
    let quiescing = quiescing.clone();
    let quiesce_response = quiesce_response.clone();
    #[cfg(feature = "tls")]
    let acceptor_for_conn = tls_acceptor.cloned();
    tokio::task::spawn_local(async move {
        let outcome = {
            #[cfg(feature = "tls")]
            {
                match acceptor_for_conn {
                    Some(acceptor) => {
                        let adapter = UringAsyncStream::new(stream);
                        match acceptor.accept(adapter).await {
                            Ok(tls_stream) => {
                                serve_connection_uring_async(
                                    tls_stream,
                                    dispatch,
                                    spec,
                                    in_flight,
                                    quiescing,
                                    quiesce_response,
                                )
                                .await
                            }
                            Err(error) => {
                                Err(ProximaError::Upstream(format!("tls handshake: {error}")))
                            }
                        }
                    }
                    None => {
                        serve_connection_uring(
                            stream,
                            dispatch,
                            spec,
                            in_flight,
                            quiescing,
                            quiesce_response,
                        )
                        .await
                    }
                }
            }
            #[cfg(not(feature = "tls"))]
            {
                serve_connection_uring(
                    stream,
                    dispatch,
                    spec,
                    in_flight,
                    quiescing,
                    quiesce_response,
                )
                .await
            }
        };
        if let Err(error) = outcome {
            warn!(?error, ?peer, "io_uring connection error");
        }
    });
}

async fn drain_in_flight_uring(in_flight: &Arc<AtomicU64>, timeout: std::time::Duration) {
    let started = std::time::Instant::now();
    let poll_interval = std::time::Duration::from_millis(20);
    while in_flight.load(Ordering::Relaxed) > 0 {
        if started.elapsed() >= timeout {
            warn!(
                remaining = in_flight.load(Ordering::Relaxed),
                "io_uring drain timeout exceeded; aborting in-flight"
            );
            return;
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn serve_connection_uring(
    raw_stream: tokio_uring::net::TcpStream,
    dispatch: PipeHandle,
    spec: Arc<HttpListenerSpec>,
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
    quiesce_response: Arc<proxima_primitives::pipe::quiesce::QuiesceResponse>,
) -> Result<(), ProximaError> {
    // Rc-wrap so the streaming-body pump and response-write path
    // can share the stream by Rc clone — both paths use `&self`
    // tokio_uring methods.
    let stream = Rc::new(raw_stream);
    let mut connection = Connection::new();
    // Auto-stream: chunked or Content-Length > 1 MiB takes the
    // streaming dispatch path (mpsc body channel + spawn_local'd
    // Pipe::call). Send-friendly because the mpsc carries Bytes
    // (Send) and the Pipe::call future itself is Send; only
    // this listener task holds the !Send Rc<TcpStream>.
    connection.set_auto_stream_policy(Some(AutoStreamPolicy::default()));
    let mut read_buf: Vec<u8> = vec![0_u8; 16 * 1024];
    let mut out: Vec<u8> = Vec::with_capacity(8 * 1024);

    loop {
        let outcome = match connection.advance() {
            Ok(outcome) => outcome,
            Err(read_error) => {
                error!(?read_error, "io_uring parse error; closing connection");
                let _ = write_all_owned(
                    &stream,
                    b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        .to_vec(),
                )
                .await;
                return Ok(());
            }
        };
        match outcome {
            Advanced::Close => return Ok(()),
            Advanced::NeedInput => {
                let buf = std::mem::take(&mut read_buf);
                let (result, returned) = stream.read(buf).await;
                let n = match result {
                    Ok(0) => return Ok(()), // peer closed
                    Ok(n) => n,
                    Err(error) => {
                        return Err(ProximaError::Io(std::io::Error::other(format!(
                            "io_uring read: {error}"
                        ))));
                    }
                };
                connection.feed_bytes(&returned[..n]);
                read_buf = returned;
                read_buf.resize(16 * 1024, 0);
            }
            Advanced::Expect100Continue(gate) => {
                // Without streaming we can't preview Content-Length
                // and respond meaningfully here — auto-accept (write
                // 100 Continue) and let the buffered path drain the
                // body. The default listener has richer policy.
                out.clear();
                gate.accept(&mut out);
                write_and_reset(&stream, &mut out).await?;
            }
            Advanced::HeadReady(_head) => {
                let _ = _head;
                match dispatch_streaming_uring(
                    &stream,
                    &mut connection,
                    &dispatch,
                    &spec,
                    &in_flight,
                    &quiescing,
                    &quiesce_response,
                    &mut read_buf,
                    &mut out,
                )
                .await?
                {
                    StreamingOutcome::KeepAlive => {
                        connection.reset_for_next_request();
                    }
                    StreamingOutcome::Close => return Ok(()),
                }
            }
            Advanced::BodyChunk(_) | Advanced::BodyEnd(_) => {
                // The outer loop never sees these — `dispatch_streaming_uring`
                // drains chunks via its own advance() loop before
                // returning.
                error!("io_uring listener leaked streaming poll variant");
                return Ok(());
            }
            Advanced::RequestReady(_request) => {
                let _ = _request;
                if quiescing.load(Ordering::Relaxed) {
                    write_quiesce_response_uring(
                        &stream,
                        &mut connection,
                        &quiesce_response,
                        &mut out,
                    )
                    .await?;
                    return Ok(());
                }
                if let Some(limit) = spec.max_body_bytes
                    && connection.body().len() > limit
                {
                    let body_len = connection.body().len();
                    let message = format!("request body exceeds limit ({body_len} > {limit})");
                    let len = message.len();
                    out.clear();
                    let writer = connection.begin_response(
                        413,
                        "Payload Too Large",
                        &[
                            ("content-type".to_string(), "text/plain".to_string()),
                            ("content-length".to_string(), len.to_string()),
                        ],
                        BodyFraming::ContentLength(len as u64),
                        &mut out,
                    );
                    writer.write_chunk(message.as_bytes(), &mut out);
                    writer.end_response(&mut out);
                    write_and_reset(&stream, &mut out).await?;
                    return Ok(());
                }
                in_flight.fetch_add(1, Ordering::Relaxed);
                let upgrade_ticket = crate::upgrade::local_slots::next_ticket();
                let mut request = build_request_uring(&connection);
                request.context.local_upgrade_ticket = Some(upgrade_ticket);
                let outcome = SendPipe::call(&dispatch, request).await;
                in_flight.fetch_sub(1, Ordering::Relaxed);
                // Cancel-on-disconnect on the buffered uring path is
                // not wired. Two approaches were tried and reverted:
                //
                // 1. `select! { read(1), dispatch }` — when dispatch
                //    won, the in-flight io_uring read submission was
                //    abandoned via tokio_uring drop. The kernel may
                //    deliver bytes to the abandoned buffer; the next
                //    iteration's NeedInput issues a fresh read; two
                //    concurrent reads on the same fd compete for the
                //    next CQE in undefined order → drain test flake.
                //
                // 2. spawn_local'd watchdog with `JoinHandle::abort`
                //    — same orphan-read problem. abort() drops the
                //    read future, kernel still has the submission,
                //    next read races against it.
                //
                // The correct fix needs either (a) `MSG_PEEK` which
                // tokio_uring 0.5 doesn't expose, or (b) a
                // continuous-read model where the main loop issues
                // reads and feeds the parser incrementally (no
                // separate watchdog needed). (b) is a serve-loop
                // restructure tracked as Stage 7. The streaming
                // dispatch path already has cancel-on-disconnect via
                // `pump_body_stream_uring`'s read loop — which is
                // exactly the continuous-read model.
                //
                // `request.context.cancel` defaults to a fresh
                // cancel Signal (never fires here), so Pipes
                // see "no cancel" on the buffered uring path.
                let local_upgrade = crate::upgrade::local_slots::take(upgrade_ticket);
                match outcome {
                    Ok(response) => {
                        let status = response.status;
                        let headers = response.headers.clone();
                        let has_send_upgrade = response.upgrade.is_some();
                        if has_send_upgrade {
                            warn!(
                                "io_uring listener received a Send UpgradeHandler; \
                                 ignored. Use LocalUpgradeHandler via \
                                 request.context.local_upgrade_ticket on this listener path."
                            );
                        }
                        let mut framing = BodyFraming::Chunked;
                        let mut header_pairs: Vec<(String, String)> =
                            Vec::with_capacity(headers.len() + 1);
                        let mut has_te = false;
                        for (name, value) in &headers {
                            let name_str = std::str::from_utf8(name.as_ref()).unwrap_or("");
                            let value_str = std::str::from_utf8(value.as_ref()).unwrap_or("");
                            if name_str.eq_ignore_ascii_case("content-length")
                                && let Ok(parsed) = value_str.trim().parse::<u64>()
                            {
                                framing = BodyFraming::ContentLength(parsed);
                            }
                            if name_str.eq_ignore_ascii_case("transfer-encoding") {
                                has_te = true;
                            }
                            header_pairs.push((name_str.to_string(), value_str.to_string()));
                        }
                        if matches!(framing, BodyFraming::Chunked) && !has_te {
                            header_pairs
                                .push(("transfer-encoding".to_string(), "chunked".to_string()));
                        }
                        out.clear();
                        let writer =
                            connection.begin_response(status, "", &header_pairs, framing, &mut out);
                        // Send head first so chunked + simple
                        // responses both work; for chunked, body
                        // streaming follows.
                        write_and_reset(&stream, &mut out).await?;

                        let body_bytes = match response.collect_body().await {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                error!(?error, "body collect failed in io_uring path");
                                return Ok(());
                            }
                        };
                        if !body_bytes.is_empty() {
                            writer.write_chunk(&body_bytes, &mut out);
                            if !out.is_empty() {
                                write_and_reset(&stream, &mut out).await?;
                            }
                        }
                        writer.end_response(&mut out);
                        if !out.is_empty() {
                            write_and_reset(&stream, &mut out).await?;
                        }
                        if let Some(handler) = local_upgrade {
                            // listener cedes the socket. drain any
                            // pipelined bytes the connection buffered
                            // past the request head, wrap the stream
                            // in the io_uring AsyncRead/AsyncWrite
                            // adapter, and invoke the handler. the
                            // adapter owns a Rc clone of the stream;
                            // when the handler returns, both this
                            // function and the handler future drop,
                            // Rc reaches zero, TcpStream closes.
                            let leftover = connection.drain_pipelined_bytes();
                            let adapter = UringAsyncStream::from_rc(Rc::clone(&stream));
                            let socket = crate::upgrade::LocalHijackedSocket::new(
                                Box::new(adapter.compat()),
                                leftover,
                            );
                            return handler.invoke(socket).await;
                        }
                    }
                    Err(error) => {
                        crate::upgrade::local_slots::discard(upgrade_ticket);
                        let message = format!("proxima error: {error}");
                        let len = message.len();
                        out.clear();
                        let writer = connection.begin_response(
                            500,
                            "",
                            &[
                                ("content-type".to_string(), "text/plain".to_string()),
                                ("content-length".to_string(), len.to_string()),
                            ],
                            BodyFraming::ContentLength(len as u64),
                            &mut out,
                        );
                        writer.write_chunk(message.as_bytes(), &mut out);
                        writer.end_response(&mut out);
                        write_and_reset(&stream, &mut out).await?;
                    }
                }
                if connection.keep_alive() {
                    connection.reset_for_next_request();
                } else {
                    return Ok(());
                }
            }
        }
    }
}

/// Streaming dispatch outcome after the io_uring path drains the
/// request body and writes the response.
enum StreamingOutcome {
    KeepAlive,
    Close,
}

#[derive(Debug)]
enum PumpUringError {
    /// Peer closed mid-body. Surface cancel + abort the dispatch.
    ClientEof,
    /// Pipe dropped its body receiver — not fatal, the Pipe
    /// may still produce a valid response.
    PipeDroppedBody,
    /// Body decoder rejected wire bytes (chunk framing, etc).
    Decode,
    /// Socket read returned an IO error.
    Io(std::io::Error),
    /// Connection state machine reported terminal Close.
    ConnectionClosed,
}

/// Streaming dispatch for the io_uring listener. Builds a streaming
/// `Request` whose body is a `tokio::mpsc` receiver, spawns
/// `Pipe::call` via `spawn_local`, then pumps body chunks into
/// the channel using owned-buffer io_uring reads. The Pipe
/// future itself is Send-friendly (mpsc + Bytes are Send); only
/// the pump task on this listener thread holds the `!Send`
/// `Rc<TcpStream>`.
#[allow(clippy::too_many_arguments)]
async fn dispatch_streaming_uring(
    stream: &Rc<tokio_uring::net::TcpStream>,
    connection: &mut Connection,
    dispatch: &PipeHandle,
    spec: &Arc<HttpListenerSpec>,
    in_flight: &Arc<AtomicU64>,
    quiescing: &Arc<AtomicBool>,
    quiesce_response: &Arc<proxima_primitives::pipe::quiesce::QuiesceResponse>,
    read_buf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> Result<StreamingOutcome, ProximaError> {
    // Quiesce gate before opening the body channel — no point
    // inviting the upload if we're refusing it.
    if quiescing.load(Ordering::Relaxed) {
        write_quiesce_response_uring(stream, connection, quiesce_response, out).await?;
        return Ok(StreamingOutcome::Close);
    }
    // Content-Length pre-check (chunked bodies are streamed
    // regardless — listener can still 413 by closing after the
    // pump if needed).
    if let Some(limit) = spec.max_body_bytes
        && let Some(length) = parse_content_length_from_connection(connection)
        && length > limit as u64
    {
        let message = format!("request body exceeds limit ({length} > {limit})");
        let len = message.len();
        out.clear();
        let writer_handle = connection.begin_response(
            413,
            "Payload Too Large",
            &[
                ("content-type".to_string(), "text/plain".to_string()),
                ("content-length".to_string(), len.to_string()),
            ],
            BodyFraming::ContentLength(len as u64),
            out,
        );
        writer_handle.write_chunk(message.as_bytes(), out);
        writer_handle.end_response(out);
        write_and_reset(stream, out).await?;
        return Ok(StreamingOutcome::Close);
    }

    let in_flight_now = in_flight.fetch_add(1, Ordering::Relaxed) + 1;
    let _ = in_flight_now;
    let cancel = proxima_core::signal::Signal::new();
    let cancel_guard = cancel.clone().guard();
    // Bounded mpsc depth between this pump and the Pipe's body
    // stream — matches the default listener's BODY_CHANNEL_DEPTH.
    const BODY_CHANNEL_DEPTH: usize = 8;
    let (body_tx, body_rx) =
        tokio::sync::mpsc::channel::<Result<Bytes, ProximaError>>(BODY_CHANNEL_DEPTH);
    let trailers_slot: crate::body::TrailersSlot = Arc::new(std::sync::Mutex::new(None));

    let upgrade_ticket = crate::upgrade::local_slots::next_ticket();
    let mut request = build_streaming_request_uring(connection, body_rx);
    request.context.cancel = cancel.clone();
    request.context.local_upgrade_ticket = Some(upgrade_ticket);

    let (resp_tx, mut resp_rx) =
        futures::channel::oneshot::channel::<Result<Response<Bytes>, ProximaError>>();
    let dispatch_clone = dispatch.clone();
    tokio::task::spawn_local(async move {
        let response = SendPipe::call(&dispatch_clone, request).await;
        let _ = resp_tx.send(response);
    });

    // Pump body chunks → bounded mpsc. Backpressure: body_tx.send()
    // parks when full → pump pauses → kernel recv buffer fills →
    // TCP window closes.
    let pump_outcome =
        pump_body_stream_uring(stream, connection, read_buf, &body_tx, &cancel).await;
    // Publish trailers, if any, before dropping the sender.
    if matches!(pump_outcome, Ok(()) | Err(PumpUringError::PipeDroppedBody)) {
        let captured = connection.take_trailers();
        if !captured.is_empty() {
            let mut trailers = HeaderList::new();
            for (name, value) in captured {
                trailers.insert(name, value);
            }
            if let Ok(mut guard) = trailers_slot.lock() {
                *guard = Some(trailers);
            }
        }
    }
    drop(body_tx);

    match pump_outcome {
        Ok(()) | Err(PumpUringError::PipeDroppedBody) => {}
        Err(PumpUringError::ClientEof | PumpUringError::ConnectionClosed) => {
            cancel.cancel();
            let _ = resp_rx.await;
            crate::upgrade::local_slots::discard(upgrade_ticket);
            in_flight.fetch_sub(1, Ordering::Relaxed);
            return Ok(StreamingOutcome::Close);
        }
        Err(PumpUringError::Decode) => {
            cancel.cancel();
            let _ = resp_rx.await;
            crate::upgrade::local_slots::discard(upgrade_ticket);
            in_flight.fetch_sub(1, Ordering::Relaxed);
            let _ = write_all_owned(
                stream,
                b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    .to_vec(),
            )
            .await;
            return Ok(StreamingOutcome::Close);
        }
        Err(PumpUringError::Io(error)) => {
            cancel.cancel();
            let _ = resp_rx.await;
            crate::upgrade::local_slots::discard(upgrade_ticket);
            in_flight.fetch_sub(1, Ordering::Relaxed);
            return Err(ProximaError::Io(error));
        }
    }

    let response_result = match (&mut resp_rx).await {
        Ok(response) => response,
        Err(_) => {
            crate::upgrade::local_slots::discard(upgrade_ticket);
            in_flight.fetch_sub(1, Ordering::Relaxed);
            error!("io_uring streaming dispatch task dropped sender");
            return Ok(StreamingOutcome::Close);
        }
    };
    let local_upgrade = crate::upgrade::local_slots::take(upgrade_ticket);
    in_flight.fetch_sub(1, Ordering::Relaxed);
    cancel_guard.disarm();

    match response_result {
        Ok(response) => {
            if response.upgrade.is_some() {
                warn!(
                    "io_uring streaming listener received a Send UpgradeHandler; \
                     ignored. Use LocalUpgradeHandler via \
                     request.context.local_upgrade_ticket on this listener path."
                );
            }
            write_response_uring(stream, connection, response, out).await?;
            if let Some(handler) = local_upgrade {
                let leftover = connection.drain_pipelined_bytes();
                let adapter = UringAsyncStream::from_rc(Rc::clone(stream));
                let socket =
                    crate::upgrade::LocalHijackedSocket::new(Box::new(adapter.compat()), leftover);
                handler.invoke(socket).await?;
                return Ok(StreamingOutcome::Close);
            }
        }
        Err(error) => {
            let message = format!("proxima error: {error}");
            let len = message.len();
            out.clear();
            let writer_handle = connection.begin_response(
                500,
                "",
                &[
                    ("content-type".to_string(), "text/plain".to_string()),
                    ("content-length".to_string(), len.to_string()),
                ],
                BodyFraming::ContentLength(len as u64),
                out,
            );
            writer_handle.write_chunk(message.as_bytes(), out);
            writer_handle.end_response(out);
            write_and_reset(stream, out).await?;
        }
    }

    if connection.keep_alive() {
        Ok(StreamingOutcome::KeepAlive)
    } else {
        Ok(StreamingOutcome::Close)
    }
}

/// Pump request body chunks from the io_uring stream into the
/// bounded mpsc backing the Pipe's request `RequestStream`. Returns
/// `Ok(())` on `BodyEnd`; otherwise see `PumpUringError`. The pump
/// uses owned-buffer reads so the io_uring submission queue gets a
/// real read SQE per call.
async fn pump_body_stream_uring(
    stream: &Rc<tokio_uring::net::TcpStream>,
    connection: &mut Connection,
    read_buf: &mut Vec<u8>,
    body_tx: &tokio::sync::mpsc::Sender<Result<Bytes, ProximaError>>,
    cancel: &proxima_core::signal::Signal,
) -> Result<(), PumpUringError> {
    loop {
        let outcome = match connection.advance() {
            Ok(o) => o,
            Err(read_error) => {
                let detail = format!("{read_error:?}");
                let _ = body_tx
                    .send(Err(ProximaError::Body(format!("decode: {detail}"))))
                    .await;
                return Err(PumpUringError::Decode);
            }
        };
        match outcome {
            Advanced::BodyChunk(handle) => {
                let chunk = handle.take_chunk();
                select! {
                    send = body_tx.send(Ok(chunk)).fuse() => {
                        if send.is_err() {
                            return Err(PumpUringError::PipeDroppedBody);
                        }
                    }
                    _ = cancel.fired().fuse() => return Err(PumpUringError::ClientEof),
                }
            }
            Advanced::BodyEnd(_handle) => {
                let _ = _handle;
                return Ok(());
            }
            Advanced::NeedInput => {
                let buf = std::mem::take(read_buf);
                let (result, returned) = stream.read(buf).await;
                match result {
                    Ok(0) => return Err(PumpUringError::ClientEof),
                    Ok(n) => connection.feed_bytes(&returned[..n]),
                    Err(error) => {
                        *read_buf = returned;
                        read_buf.resize(16 * 1024, 0);
                        return Err(PumpUringError::Io(error));
                    }
                }
                *read_buf = returned;
                read_buf.resize(16 * 1024, 0);
            }
            Advanced::Close => return Err(PumpUringError::ConnectionClosed),
            Advanced::HeadReady(_) | Advanced::RequestReady(_) | Advanced::Expect100Continue(_) => {
                // HeadReady consumed by outer loop; RequestReady
                // never fires in streaming mode; Expect handled
                // upstream. Anything reaching here is a state-machine
                // bug.
                return Err(PumpUringError::Decode);
            }
        }
    }
}

/// Build the streaming `Request` for the io_uring path. The body is
/// a stream wrapping the bounded mpsc receiver. Request-body trailers
/// are not carried on the body (the `RequestStream` has no trailers
/// slot); the pump publishes any captured request trailers into the
/// externally-owned slot for callers that consult it directly.
fn build_streaming_request_uring(
    connection: &Connection,
    body_rx: tokio::sync::mpsc::Receiver<Result<Bytes, ProximaError>>,
) -> Request<Bytes> {
    let path_bytes = connection.path();
    let (path, query) = split_path_and_query(path_bytes);
    let method = Bytes::copy_from_slice(connection.method());
    let mut headers = HeaderList::new();
    for header in connection.headers() {
        headers.insert(
            Bytes::copy_from_slice(header.name()),
            Bytes::copy_from_slice(header.value()),
        );
    }
    let body_stream = futures::stream::unfold(body_rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let request_stream = RequestStream::new(body_stream);
    let mut context = RequestContext::default();
    let (trace_id, baggage) = proxima_telemetry::propagation::establish_trace_context(&headers);
    context.adopt_trace_context(trace_id, baggage);
    Request {
        method,
        path,
        query,
        headers,
        payload: Bytes::new(),
        stream: Some(request_stream),
        context,
    }
}

/// Write a response back through the io_uring stream — head, body
/// chunks (streamed from `Response.body`), then terminator /
/// trailers. Mirrors the default path's `write_response` shape but
/// uses owned-buffer writes.
async fn write_response_uring(
    stream: &Rc<tokio_uring::net::TcpStream>,
    connection: &mut Connection,
    response: Response,
    out: &mut Vec<u8>,
) -> Result<(), ProximaError> {
    let status = response.status;
    let headers = response.headers.clone();
    let mut framing = BodyFraming::Chunked;
    let mut header_pairs: Vec<(String, String)> = Vec::with_capacity(headers.len() + 1);
    let mut has_te = false;
    for (name, value) in &headers {
        let name_str = std::str::from_utf8(name.as_ref()).unwrap_or("");
        let value_str = std::str::from_utf8(value.as_ref()).unwrap_or("");
        if name_str.eq_ignore_ascii_case("content-length")
            && let Ok(parsed) = value_str.trim().parse::<u64>()
        {
            framing = BodyFraming::ContentLength(parsed);
        }
        if name_str.eq_ignore_ascii_case("transfer-encoding") {
            has_te = true;
        }
        header_pairs.push((name_str.to_string(), value_str.to_string()));
    }
    if matches!(framing, BodyFraming::Chunked) && !has_te {
        header_pairs.push(("transfer-encoding".to_string(), "chunked".to_string()));
    }
    let trailers_slot = response
        .stream
        .as_ref()
        .and_then(|stream| stream.trailers_slot().cloned());
    out.clear();
    let writer_handle = connection.begin_response(status, "", &header_pairs, framing, out);
    write_and_reset(stream, out).await?;

    // Drain response body chunks.
    use futures::stream::StreamExt;
    let mut body_stream = response.into_chunk_stream();
    while let Some(chunk) = body_stream.next().await {
        let chunk = chunk?;
        if chunk.is_empty() {
            continue;
        }
        writer_handle.write_chunk(&chunk, out);
        if !out.is_empty() {
            write_and_reset(stream, out).await?;
        }
    }

    // Trailers — only emit for chunked framing; encoder ignores for
    // content-length / none.
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
        writer_handle.end_response_with_trailers(&trailer_refs, out);
    } else {
        writer_handle.end_response(out);
    }
    if !out.is_empty() {
        write_and_reset(stream, out).await?;
    }
    Ok(())
}

fn parse_content_length_from_connection(connection: &Connection) -> Option<u64> {
    let value = connection.header_value(b"content-length")?;
    let text = std::str::from_utf8(value).ok()?;
    text.trim().parse::<u64>().ok()
}

async fn write_all_owned(
    stream: &tokio_uring::net::TcpStream,
    buf: Vec<u8>,
) -> Result<(), std::io::Error> {
    let (result, _) = stream.write_all(buf).await;
    result
}

async fn write_quiesce_response_uring(
    stream: &Rc<tokio_uring::net::TcpStream>,
    connection: &mut Connection,
    quiesce: &proxima_primitives::pipe::quiesce::QuiesceResponse,
    out: &mut Vec<u8>,
) -> Result<(), ProximaError> {
    out.clear();
    let mut headers: Vec<(String, String)> = vec![("content-length".to_string(), "0".to_string())];
    if !quiesce.retry_after.is_empty() {
        headers.push(("retry-after".to_string(), quiesce.retry_after.clone()));
    }
    let writer_handle = connection.begin_response(
        quiesce.status,
        "Pipe Unavailable",
        &headers,
        BodyFraming::ContentLength(0),
        out,
    );
    writer_handle.end_response(out);
    write_and_reset(stream, out).await
}

/// Take `out`, hand it to `write_all`, put the returned buffer back
/// into `out` cleared so the caller can reuse it on the next write
/// without re-allocating. Centralizes the `mem::take` / cycle / clear
/// dance so call sites don't trip the `unused-assignments` lint.
async fn write_and_reset(
    stream: &tokio_uring::net::TcpStream,
    out: &mut Vec<u8>,
) -> Result<(), ProximaError> {
    let taken = std::mem::take(out);
    let (result, returned) = stream.write_all(taken).await;
    *out = returned;
    out.clear();
    result.map_err(ProximaError::Io)
}

fn build_request_uring(connection: &Connection) -> Request<Bytes> {
    let mut headers = HeaderList::new();
    for header in connection.headers() {
        headers.insert(
            Bytes::copy_from_slice(header.name()),
            Bytes::copy_from_slice(header.value()),
        );
    }
    let body_bytes = Bytes::copy_from_slice(connection.body());
    let method = Bytes::copy_from_slice(connection.method());
    let path_bytes = connection.path();
    let (path, query) = split_path_and_query(path_bytes);
    let mut context = RequestContext::default();
    let (trace_id, baggage) = proxima_telemetry::propagation::establish_trace_context(&headers);
    context.adopt_trace_context(trace_id, baggage);
    Request {
        method,
        path,
        query,
        headers,
        body: body_bytes,
        stream: None,
        context,
    }
}

fn split_path_and_query(raw: &[u8]) -> (Bytes, HeaderList) {
    let mut query = HeaderList::new();
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
                query.insert(name.as_bytes().to_vec(), value.as_bytes().to_vec());
            }
        }
    }
    (path, query)
}

/// AsyncRead-based serve loop. Used for the TLS path on io_uring
/// (TlsStream wraps `UringAsyncStream`). Same Connection-driven
/// shape as `serve_connection_uring` but reads/writes go through
#[cfg(feature = "tls")]
/// `AsyncReadExt::read` / `AsyncWriteExt::write_all`. Buffered
/// requests only — streaming dispatch isn't wired here for the
/// same Send-bound reason called out in the module doc.
async fn serve_connection_uring_async<S>(
    socket: S,
    dispatch: PipeHandle,
    spec: Arc<HttpListenerSpec>,
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
    quiesce_response: Arc<proxima_primitives::pipe::quiesce::QuiesceResponse>,
) -> Result<(), ProximaError>
where
    S: AsyncRead + AsyncWrite + Unpin + 'static,
{
    let (mut reader, mut writer) = tokio::io::split(socket);
    let mut connection = Connection::new();
    connection.set_auto_stream_policy(Some(AutoStreamPolicy {
        content_length_threshold: u64::MAX,
        stream_chunked: false,
    }));
    let mut read_buf = [0_u8; 16 * 1024];
    let mut out: Vec<u8> = Vec::with_capacity(8 * 1024);
    loop {
        let outcome = match connection.advance() {
            Ok(outcome) => outcome,
            Err(read_error) => {
                error!(?read_error, "io_uring async serve: parse error");
                let _ = writer
                    .write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await;
                return Ok(());
            }
        };
        match outcome {
            Advanced::Close => return Ok(()),
            Advanced::NeedInput => {
                let read = match reader.read(&mut read_buf).await {
                    Ok(0) => return Ok(()),
                    Ok(n) => n,
                    Err(error) => {
                        return Err(ProximaError::Io(std::io::Error::other(format!(
                            "tls/uring read: {error}"
                        ))));
                    }
                };
                connection.feed_bytes(&read_buf[..read]);
            }
            Advanced::Expect100Continue(gate) => {
                out.clear();
                gate.accept(&mut out);
                writer.write_all(&out).await.map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!("tls/uring write 100: {err}")))
                })?;
            }
            Advanced::HeadReady(_) | Advanced::BodyChunk(_) | Advanced::BodyEnd(_) => {
                error!("io_uring tls listener saw streaming variant; closing");
                return Ok(());
            }
            Advanced::RequestReady(_request) => {
                let _ = _request;
                if quiescing.load(Ordering::Relaxed) {
                    out.clear();
                    let mut headers: Vec<(String, String)> =
                        vec![("content-length".to_string(), "0".to_string())];
                    if !quiesce_response.retry_after.is_empty() {
                        headers.push((
                            "retry-after".to_string(),
                            quiesce_response.retry_after.clone(),
                        ));
                    }
                    let writer_handle = connection.begin_response(
                        quiesce_response.status,
                        "Pipe Unavailable",
                        &headers,
                        BodyFraming::ContentLength(0),
                        &mut out,
                    );
                    writer_handle.end_response(&mut out);
                    let _ = writer.write_all(&out).await;
                    let _ = writer.flush().await;
                    return Ok(());
                }
                if let Some(limit) = spec.max_body_bytes
                    && connection.body().len() > limit
                {
                    let body_len = connection.body().len();
                    let message = format!("request body exceeds limit ({body_len} > {limit})");
                    let len = message.len();
                    out.clear();
                    let writer_handle = connection.begin_response(
                        413,
                        "Payload Too Large",
                        &[
                            ("content-type".to_string(), "text/plain".to_string()),
                            ("content-length".to_string(), len.to_string()),
                        ],
                        BodyFraming::ContentLength(len as u64),
                        &mut out,
                    );
                    writer_handle.write_chunk(message.as_bytes(), &mut out);
                    writer_handle.end_response(&mut out);
                    let _ = writer.write_all(&out).await;
                    return Ok(());
                }
                in_flight.fetch_add(1, Ordering::Relaxed);
                let upgrade_ticket = crate::upgrade::local_slots::next_ticket();
                let mut request = build_request_uring(&connection);
                request.context.local_upgrade_ticket = Some(upgrade_ticket);
                let outcome = SendPipe::call(&dispatch, request).await;
                in_flight.fetch_sub(1, Ordering::Relaxed);
                let local_upgrade = crate::upgrade::local_slots::take(upgrade_ticket);
                // see comment in `serve_connection_uring`: cancel-on-disconnect
                // not yet wired on the buffered uring path.
                match outcome {
                    Ok(response) => {
                        let status = response.status;
                        let headers = response.headers.clone();
                        let has_send_upgrade = response.upgrade.is_some();
                        if has_send_upgrade {
                            warn!(
                                "io_uring/tls listener received a Send UpgradeHandler; \
                                 ignored. Use LocalUpgradeHandler via \
                                 request.context.local_upgrade_ticket on this listener path."
                            );
                        }
                        let mut framing = BodyFraming::Chunked;
                        let mut header_pairs: Vec<(String, String)> =
                            Vec::with_capacity(headers.len() + 1);
                        let mut has_te = false;
                        for (name, value) in &headers {
                            let name_str = std::str::from_utf8(name.as_ref()).unwrap_or("");
                            let value_str = std::str::from_utf8(value.as_ref()).unwrap_or("");
                            if name_str.eq_ignore_ascii_case("content-length")
                                && let Ok(parsed) = value_str.trim().parse::<u64>()
                            {
                                framing = BodyFraming::ContentLength(parsed);
                            }
                            if name_str.eq_ignore_ascii_case("transfer-encoding") {
                                has_te = true;
                            }
                            header_pairs.push((name_str.to_string(), value_str.to_string()));
                        }
                        if matches!(framing, BodyFraming::Chunked) && !has_te {
                            header_pairs
                                .push(("transfer-encoding".to_string(), "chunked".to_string()));
                        }
                        out.clear();
                        let writer_handle =
                            connection.begin_response(status, "", &header_pairs, framing, &mut out);
                        writer.write_all(&out).await.map_err(|err| {
                            ProximaError::Io(std::io::Error::other(format!(
                                "tls/uring write head: {err}"
                            )))
                        })?;
                        out.clear();
                        let body_bytes = response.collect_body().await.unwrap_or_default();
                        if !body_bytes.is_empty() {
                            writer_handle.write_chunk(&body_bytes, &mut out);
                            if !out.is_empty() {
                                writer.write_all(&out).await.map_err(|err| {
                                    ProximaError::Io(std::io::Error::other(format!(
                                        "tls/uring write chunk: {err}"
                                    )))
                                })?;
                                out.clear();
                            }
                        }
                        writer_handle.end_response(&mut out);
                        if !out.is_empty() {
                            writer.write_all(&out).await.map_err(|err| {
                                ProximaError::Io(std::io::Error::other(format!(
                                    "tls/uring write tail: {err}"
                                )))
                            })?;
                        }
                        // tokio-rustls buffers ciphertext writes; force a
                        // flush so the response body reaches the client
                        // before keep-alive idle or connection close.
                        writer.flush().await.map_err(|err| {
                            ProximaError::Io(std::io::Error::other(format!(
                                "tls/uring flush: {err}"
                            )))
                        })?;
                        if let Some(handler) = local_upgrade {
                            // unsplit halves into the original TlsStream
                            // and hand to the upgrade handler. the inner
                            // io_uring socket is wrapped via UringAsyncStream
                            // and lives inside the TLS session — handler
                            // reads/writes plaintext, ciphertext rides on
                            // io_uring under the hood.
                            let leftover = connection.drain_pipelined_bytes();
                            let socket_stream = reader.unsplit(writer);
                            let socket = crate::upgrade::LocalHijackedSocket::new(
                                Box::new(socket_stream.compat()),
                                leftover,
                            );
                            return handler.invoke(socket).await;
                        }
                    }
                    Err(error) => {
                        crate::upgrade::local_slots::discard(upgrade_ticket);
                        let message = format!("proxima error: {error}");
                        let len = message.len();
                        out.clear();
                        let writer_handle = connection.begin_response(
                            500,
                            "",
                            &[
                                ("content-type".to_string(), "text/plain".to_string()),
                                ("content-length".to_string(), len.to_string()),
                            ],
                            BodyFraming::ContentLength(len as u64),
                            &mut out,
                        );
                        writer_handle.write_chunk(message.as_bytes(), &mut out);
                        writer_handle.end_response(&mut out);
                        let _ = writer.write_all(&out).await;
                    }
                }
                if connection.keep_alive() {
                    connection.reset_for_next_request();
                } else {
                    return Ok(());
                }
            }
        }
    }
}
