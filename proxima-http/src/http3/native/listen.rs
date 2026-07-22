//! Native HTTP/3 [`ListenProtocol`] over the sans-IO proxima-quic-proto stack +
//! the driver in [`crate::http3::native::driver`] — no quinn, no h3-quinn, **no tokio**.
//!
//! Runtime-agnostic IO: the UDP socket comes from the [`ServeContext`]'s
//! `DatagramFactory` (the UDP sibling of the TCP `AcceptorFactory`) and the 1 ms
//! tick from `proxima_core::time::sleep` (whose driver is the build-selected one — the
//! prime per-core wheel under `timer = "prime-wheel"`). The listener names
//! neither prime nor tokio; either runtime supplies the factory + timer driver.
//!
//! # Composition over `proxima_quic::native::Listener<RustlsServerProvider>`
//!
//! The QUIC transport (DCID-keyed demux, per-connection accept/timeout/
//! transmit, Version Negotiation replies, ODCID routing) is
//! [`proxima_quic::native::listener::Listener`] — this module no longer
//! inlines its own `EndpointDemux` + `BTreeMap<u32, ConnEntry>` + timer +
//! transmit block. `H3NativeListenProtocol` KEEPS its own `serve()` loop
//! and its own H3 bookkeeping (`h3_state: BTreeMap<u32, PerConnection>`,
//! keyed by the SAME `ConnectionHandle` the `Listener` hands out) — the
//! composition is BY SHARED KEY, not inheritance: `Listener<P>` stays
//! QUIC-only, H3 stays H3-only, and a `ConnectionHandle` is the join key
//! between the two tables.
//!
//! Scope: single-task accept + per-connection driver fan-in. Each tick:
//!
//! 1. Recv a burst of datagrams; `listener.ingest_datagram(now, peer,
//!    bytes)` per datagram (demux/accept/ODCID/VN all live inside the
//!    listener now) — its `DatagramIngest` return NAMES the touched
//!    handle directly (fed straight into `dirty`, no side-channel drain)
//!    and carries any per-connection error `Listener` surfaced; drain
//!    `accept_rx` for freshly accepted handles and seed their
//!    `PerConnection` H3 state.
//! 2. Drive H3 for ONLY the connections `dirty` names — populated by step
//!    1's `ingest_datagram` returns, further unioned with any handle
//!    whose dispatched response was just applied (needs its H3 layer
//!    driven again to push the response into the QUIC send buffers) —
//!    targeted, not a full-table scan.
//! 3. Once a request's HEADERS + FIN both arrive, spawn the
//!    `PipeHandle::call_dyn` future cooperatively (no `tokio::spawn`; see
//!    `in_flight`). The result comes back via an mpsc channel.
//! 4. The next tick (or the SAME tick, via the fixpoint gate) consumes
//!    ready responses + ships HEADERS+DATA+FIN back through the H3 state
//!    machine, then drains `listener.transmit()` to the socket.
//!
//! # Errors never vanish
//!
//! A per-connection error surfaced by `Listener::ingest_datagram` is
//! NEVER silently dropped: `Listener<P>` itself already acted (closed
//! with a transport code) on RFC 9000's own unambiguous violations and
//! emitted a `proxima_telemetry::error!` event; for the remaining
//! ambiguous / application-flavored errors it left the connection open
//! and this module (`close_with_h3_code`) closes it with the H3-specific
//! code its own semantics call for — `Connection::close` is idempotent
//! (first call wins), so attempting this unconditionally on every
//! surfaced error is always safe: a no-op when `Listener<P>` already
//! acted, and the actual escalation when it didn't.
//!
//! Future work (out of scope for v1):
//! - request-body streaming (we currently buffer to FIN before
//!   dispatching; fine for GET, would need a tx end of an mpsc for
//!   bidirectional POST).
//! - proper per-connection task fan-out (today everything runs in the
//!   listener task).

use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Either;
use futures::stream::{FuturesUnordered, StreamExt};
use proxima_telemetry::{debug, warn};
use serde_json::Value;
use std::future::poll_fn;

use proxima_core::ProximaError;
use proxima_core::datagram_batch::DefaultDatagramBatch;
use proxima_listen::stream::DatagramProtocol;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_primitives::stream::DatagramSocketBatchExt;
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, StreamId as H3StreamId};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::quic::connection::{Connection, ConnectionState, HandshakeLimits};
use proxima_protocols::quic::endpoint::ConnectionHandle;
use proxima_protocols::quic::time::Instant as ProtoInstant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsConfig, RustlsServerProvider};
use proxima_quic::native::{AcceptFn, DatagramIngest, Listener};

use super::driver::{DriverState, drive_server_step};

const ALPN_H3: &[u8] = b"h3";
/// RFC 9114 §8.1 — HTTP/3 application error codes carried in the
/// QUIC CONNECTION_CLOSE frame's error_code field. We map every
/// driver-level error to H3_GENERAL_PROTOCOL_ERROR today; future
/// refinement can plumb the specific RFC-mandated code through the
/// driver's ConnectionError variants (H3_FRAME_UNEXPECTED for the
/// missing-SETTINGS case, H3_SETTINGS_ERROR for duplicate ids,
/// H3_STREAM_CREATION_ERROR for a second control stream, etc.).
const H3_GENERAL_PROTOCOL_ERROR: u64 = 0x0101;
const H3_CLOSED_CRITICAL_STREAM: u64 = 0x0104;
/// RFC 9204 §8.3 — decoder failed to interpret an encoded field section.
#[cfg(feature = "http3-part-source")]
const QPACK_DECOMPRESSION_FAILED: u64 = 0x0200;

/// Short, structured close reason for the wire — keeps the close
/// frame small (RFC 9000 §19.19 caps reason at ~1200 bytes; we
/// stay well under).
fn err_reason_for_close(err: &proxima_protocols::quic::connection::ConnectionError) -> &'static [u8] {
    match err {
        proxima_protocols::quic::connection::ConnectionError::ProtocolViolation { reason } => {
            reason.as_bytes()
        }
        _ => b"h3 driver step failed",
    }
}

/// Close `connection` with the most specific H3 error code the reason
/// string identifies (until the driver/connection layer carries typed H3
/// error variants) — the escalation THIS layer owns (H3 semantics),
/// applied on top of whatever `Listener<P>` already did for the SAME
/// error. `Connection::close` is idempotent (first call wins): if
/// `Listener<P>` already closed this connection with a transport code
/// (one of RFC 9000's own unambiguous violations —
/// `FlowControlError`/`ProtocolViolation`/`Frame`), this call is a
/// harmless no-op and the transport code is what reaches the wire; if
/// `Listener<P>` left the connection open (an ambiguous, application-
/// flavored error it has no transport code for), THIS call is what
/// actually reaches the wire — restoring the H3-specific escalation the
/// pre-`Listener<P>` code did, now explicitly at the layer that owns H3
/// semantics.
fn close_with_h3_code(
    connection: &mut Connection<RustlsServerProvider>,
    handle_id: u32,
    err: &proxima_protocols::quic::connection::ConnectionError,
) {
    let reason = err_reason_for_close(err);
    let h3_code = if reason
        .windows(b"CLOSED_CRITICAL_STREAM".len())
        .any(|window| window == b"CLOSED_CRITICAL_STREAM")
    {
        H3_CLOSED_CRITICAL_STREAM
    } else {
        H3_GENERAL_PROTOCOL_ERROR
    };
    warn!(?err, handle = handle_id, "h3-native connection-level error; closing with H3 code");
    let _ = connection.close(h3_code, reason);
}

/// Native HTTP/3 listener, generic over the [`Clock`] the serve loop reads
/// `now` from AND drives its timer sleep from — one clock, no split. Production
/// defaults `Clk` to [`TimeClock`] (the monotonic prime/tokio-bound clock);
/// deterministic tests inject a mock clock via [`with_clock`](Self::with_clock).
pub struct H3NativeListenProtocol<Clk = TimeClock> {
    label: String,
    clock: Clk,
}

impl Default for H3NativeListenProtocol<TimeClock> {
    fn default() -> Self {
        Self {
            label: "h3-native".into(),
            clock: TimeClock,
        }
    }
}

impl H3NativeListenProtocol<TimeClock> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Clk> H3NativeListenProtocol<Clk> {
    /// Materialise with an explicit [`Clock`] — the seam a deterministic test
    /// or example injects a fake clock through. The serve loop derives BOTH the
    /// `ProtoInstant` `now` it feeds connection creation / reap / transmit AND
    /// its timer sleep from this single clock, so virtual time and the parked
    /// sleep can never diverge. Production goes via [`new`](Self::new), which
    /// defaults `Clk` to [`TimeClock`].
    #[must_use]
    pub fn with_clock(clock: Clk) -> Self {
        Self {
            label: "h3-native".into(),
            clock,
        }
    }
}

impl<Clk> ListenProtocol for H3NativeListenProtocol<Clk>
where
    Clk: Clock + Clone + Send + Sync + 'static,
    Clk::Delay: Send,
{
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let server_config = match build_rustls_server_config(spec) {
            Ok(cfg) => cfg,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let handshake_limits = parse_handshake_limits_from_spec(spec);
        // opt-in Source-mode request headers (see PerConnection::set_part_source_mode);
        // inert unless built with the part-source feature.
        let part_source_mode = spec
            .get("part_source")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Runtime-agnostic IO: the UDP socket comes from the runtime's datagram
        // factory (the UDP sibling of the TCP AcceptorFactory path) and the 1 ms
        // tick from `proxima_core::time::sleep` — the listener names neither prime nor
        // tokio.
        let datagram_factory = context.datagram_factory.clone();
        let ready_signal = context.ready_signal.clone();
        // The single injected clock: both `now` and the timer sleep below read
        // from it, so a mock clock drives the loop deterministically.
        let clock = self.clock.clone();

        Box::pin(async move {
            let datagram_factory = datagram_factory.ok_or_else(|| {
                ProximaError::Config("h3-native listener requires a datagram factory".into())
            })?;
            let mut socket = datagram_factory
                .bind(bind)
                .map_err(|err| ProximaError::Upstream(format!("h3-native bind: {err}")))?;
            let local = socket
                .local_addr()
                .map_err(|err| ProximaError::Upstream(format!("h3-native local_addr: {err}")))?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(?bind, %local, "h3-native listener bound");

            // The QUIC transport: demux, accept, ODCID routing, Version
            // Negotiation, and per-connection timeout/transmit all live
            // here now — see the module doc for the composition shape.
            let accept_fn: AcceptFn<RustlsServerProvider> = {
                let server_config = server_config.clone();
                Arc::new(move |dcid: &[u8], scid: &[u8], local_scid: &[u8], now: ProtoInstant| {
                    // RFC 9000 §18.2 — server transport parameters MUST include
                    // the client's original DCID and our chosen SCID; clients
                    // reject a server flight that omits them.
                    let server_tp = encode_server_transport_parameters(dcid, local_scid);
                    Connection::<RustlsServerProvider>::new_server_with_limits(
                        RustlsConfig::Server {
                            config: server_config.clone(),
                        },
                        &server_tp,
                        dcid,
                        scid,
                        local_scid,
                        now,
                        handshake_limits,
                    )
                })
            };
            let (accept_tx, mut accept_rx) = futures::channel::mpsc::unbounded::<ConnectionHandle>();
            let mut listener = Listener::<RustlsServerProvider>::new(accept_fn, accept_tx);
            // H3-level bookkeeping, keyed by the SAME `ConnectionHandle.0`
            // the listener hands out — the shared-key join between the
            // QUIC-only transport and this module's H3-only state.
            let mut h3_state: BTreeMap<u32, PerConnection> = BTreeMap::new();

            let (response_tx, mut response_rx) =
                futures::channel::mpsc::unbounded::<DispatchResult>();
            // handler futures driven cooperatively in the serve loop — no
            // tokio::spawn, no runtime coupling. Send so the serve future
            // (which must be Send) stays Send; dispatch.call_dyn is Send.
            let mut in_flight: FuturesUnordered<Pin<Box<dyn Future<Output = ()> + Send>>> =
                FuturesUnordered::new();

            let mut shutdown = shutdown;

            // Canonical batched UDP I/O buffers (the tiered-io-buffer primitive),
            // hoisted once and reused every iteration (off the per-iteration alloc
            // path). `batch.recv` is the RecvSlab the drive layer fills + drains to
            // empty; `batch.send` is the SendBatch that stages every outbound packet
            // contiguously so one `sendmmsg` ships the whole burst. Replaces the
            // crate-local hand-rolled recv_storage/recv_meta/send_arena/send_spans —
            // same growable shape on the std/alloc tier, now shared + bench-gated.
            let mut batch = DefaultDatagramBatch::new();
            let mut transmit_scratch = [0u8; 2048];

            // Drive-to-fixpoint gate: after an event, re-run drive+transmit while a
            // pass stages output (a state transition can unblock more to send — e.g.
            // the Established transition emits HANDSHAKE_DONE then SETTINGS across
            // passes). It parks only when a pass stages nothing — NEVER with output
            // pending. The event-driven replacement for the old 1ms re-flush tick:
            // zero tick latency, and under load it stays busy instead of stranding
            // work behind a cap.
            let mut skip_wait = false;
            // Handles to drive this pass — declared OUTSIDE the loop and
            // deliberately NOT reset to empty on every iteration: a
            // `skip_wait` re-drive pass has NO new inbound datagram (so
            // this tick's `ingest_datagram` calls alone would populate
            // nothing), but the very reason it's re-driving at all is that
            // the PREVIOUS
            // pass staged output, and a state transition inside
            // `drive_server_step` can require a SECOND call before it has
            // more to say (the HANDSHAKE_DONE-then-SETTINGS shape the gate's
            // own doc above names). So `dirty` stays populated across
            // consecutive `skip_wait` passes and is cleared ONLY once a
            // pass genuinely settles (see the gate at the loop's tail) —
            // still O(dirty), never a full-table scan, but never dropped
            // mid-fixpoint either.
            let mut dirty: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

            loop {
                let tick_start = core_instant_now(&clock);
                let next_delay = listener
                    .next_deadline()
                    .map(|deadline| deadline.saturating_duration_since(tick_start));

                let mut shutting_down = false;
                let recv_outcome = if skip_wait {
                    // fixpoint re-drive (see the gate at the loop's tail): a prior
                    // pass staged output, so run drive+transmit again immediately
                    // without parking. Still drain recv NON-BLOCKING here — the burst
                    // keeps sending, so it must keep receiving too, or a busy socket
                    // (many connections) starves inbound and every handshake stalls.
                    // Only the park below ever blocks.
                    skip_wait = false;
                    let drained = socket.drain_recv_to_empty(&mut batch.recv);
                    if drained > 0 { Some(Ok(drained)) } else { None }
                } else {
                    let recv = poll_fn(|cx| socket.poll_fill_recv_batch(cx, &mut batch.recv));
                    let timer = async {
                        match next_delay {
                            Some(delay) => clock.delay(delay).await,
                            None => futures::future::pending::<()>().await,
                        }
                    };
                    // drive in-flight handlers alongside recv/timer; when none
                    // are queued this never resolves so the race is unchanged.
                    let handlers = async {
                        if in_flight.is_empty() {
                            futures::future::pending::<()>().await
                        } else {
                            let _ = in_flight.next().await;
                        }
                    };
                    futures::pin_mut!(recv, timer, handlers);
                    match futures::future::select(
                        futures::future::select(futures::future::select(recv, timer), handlers),
                        &mut shutdown,
                    )
                    .await
                    {
                        Either::Left((Either::Left((Either::Left((res, _)), _)), _)) => Some(res),
                        // timer fired — pump QUIC timeouts below (unconditional
                        // reap, matching the original tick shape).
                        Either::Left((Either::Left((Either::Right(((), _)), _)), _)) => None,
                        // a handler completed (it pushed its result to
                        // response_tx); fall through to drain it.
                        Either::Left((Either::Right(((), _)), _)) => None,
                        // shutdown signalled (sent, or sender dropped).
                        Either::Right(_) => {
                            shutting_down = true;
                            None
                        }
                    }
                };
                if shutting_down {
                    debug!("h3-native listener shutting down");
                    break;
                }
                // Re-sample now that the await (if any) has resolved: this is
                // the timestamp every connection created below must be
                // anchored to, not `tick_start`.
                let now = core_instant_now(&clock);
                if let Some(Err(ref err)) = recv_outcome {
                    debug!(
                        ?err,
                        "h3-native poll_recv_batch error; datagrams dropped before on_datagram"
                    );
                }
                if let Some(Ok(_first_count)) = recv_outcome {
                    // Drain the kernel socket to EMPTY into the growable slab
                    // before the O(N) process+send cycle below. A handshake
                    // burst (~2 CRYPTO packets per connecting peer) can exceed
                    // the kernel datagram buffer faster than one process cycle
                    // drains it; reading until WouldBlock decouples drain-rate
                    // from process-rate, so the default SO_RCVBUF suffices and
                    // we don't depend on a raised net.core.rmem_max. The slab
                    // grows to the largest burst seen and is reused thereafter.
                    socket.drain_recv_to_empty(&mut batch.recv);
                    for view in batch.recv.filled_datagrams() {
                        debug!(len = view.bytes.len(), peer = %view.peer, "h3-native recv datagram");
                        // The concrete `ingest_datagram` (not the generic
                        // `DatagramProtocol::on_datagram` trait method) —
                        // its return NAMES which handle this datagram
                        // touched (feeding `dirty` directly, no side-channel
                        // drain) and carries any per-connection error
                        // `Listener` surfaced (telemetry + transport close
                        // already happened there for the unambiguous RFC
                        // 9000 cases; an H3-level violation is still OPEN
                        // for THIS layer to close with its own code).
                        match listener.ingest_datagram(to_proto_instant(now), view.peer, view.bytes) {
                            Ok(DatagramIngest::Existing { handle, error } | DatagramIngest::Accepted { handle, error }) => {
                                dirty.insert(handle.0);
                                if let Some(err) = error
                                    && let Some(connection) = listener.connection_mut(handle)
                                {
                                    close_with_h3_code(connection, handle.0, &err);
                                }
                            }
                            Ok(DatagramIngest::VersionNegotiated | DatagramIngest::Dropped) => {}
                            Ok(_) => {
                                // Future non_exhaustive `DatagramIngest`
                                // variants — nothing to drive.
                            }
                            Err(error) => {
                                warn!(?error, peer = %view.peer, "h3-native ingest_datagram failed; dropping");
                            }
                        }
                    }
                    batch.recv.clear();
                    // Seed H3 state for every freshly accepted connection —
                    // the QUIC transport (demux/accept/ODCID/VN) already ran
                    // above, inside `listener.ingest_datagram`.
                    while let Ok(handle) = accept_rx.try_recv() {
                        h3_state.insert(handle.0, PerConnection::new(part_source_mode));
                        debug!(handle = handle.0, "h3-native accepted");
                    }
                }

                // 2) Drive H3 for the connections this tick actually
                // touched — the `dirty` set the recv arm just populated
                // straight off `ingest_datagram`'s return, further unioned
                // below with any handle whose dispatched response was just
                // applied. Targeted, not a full-table scan of every live
                // connection every tick.
                drive_dirty_connections(&mut listener, &mut h3_state, &dirty, &dispatch, &response_tx, &mut in_flight);

                // 2b) Drive ready response handlers to completion NOW so their
                // responses emit in THIS iteration instead of trickling across
                // subsequent ticks. `now_or_never` polls each ready handler
                // once; a handler that genuinely yields stays queued and is
                // woken via the `handlers` arm of the select above.
                while in_flight.next().now_or_never().flatten().is_some() {}

                // 3) Apply any ready responses. A freshly applied response is
                // queued into the H3 layer by step 4 below, but step 2's drive
                // (which moves H3 -> QUIC) already ran this pass — so without a
                // re-drive the response sits in H3 until the next drive and the
                // loop parks (timer-woken) before sending it. Marking the
                // handle dirty forces step 4 (and, via `applied_response`, the
                // fixpoint re-drive at the gate) to push it.
                let mut applied_response = false;
                while let Ok(result) = response_rx.try_recv() {
                    let connection_handle = result.connection_handle;
                    if let Some(driver) = h3_state.get_mut(&connection_handle) {
                        apply_dispatch_response(driver, result);
                        dirty.insert(connection_handle);
                        applied_response = true;
                    }
                }

                // 4) Emit pending responses through H3 for the dirty set.
                for handle_id in &dirty {
                    if let Some(driver) = h3_state.get_mut(handle_id)
                        && let Err(err) = flush_pending_responses(driver)
                    {
                        warn!(?err, handle = handle_id, "h3-native response flush error");
                    }
                }

                // Per-connection timer tick so PTOs / idle deadlines
                // advance, unconditional every tick (matches the original
                // shape) — the listener's INHERENT `handle_timeout` /
                // `remove_connection` (NOT the `DatagramProtocol` trait's
                // `on_timeout`, which discards the reaped handles) so this
                // loop can also drop the matching H3 state. MUST derive
                // from the SAME `now` sample `on_datagram` fed the listener
                // this tick (via `to_proto_instant`, the exact conversion
                // `Listener` uses internally to anchor a freshly-accepted
                // connection's `handshake_completion_deadline`) — an
                // independently-sampled, differently-epoched `now` here
                // reaped connections microseconds after accepting them
                // (see `to_proto_instant`'s doc for the two-epoch bug this
                // caused and the regression test that caught it).
                for handle in listener.handle_timeout(to_proto_instant(now)) {
                    listener.remove_connection(handle);
                    h3_state.remove(&handle.0);
                    dirty.remove(&handle.0);
                    debug!(handle = handle.0, "h3-native reaped terminal connection");
                }

                // 5) Stage every outbound datagram — Version Negotiation
                // replies AND every connection's QUIC egress — into one
                // contiguous arena via the listener's `transmit`, then ship
                // the whole burst with a single `sendmmsg`.
                batch.send.reset();
                loop {
                    match listener.transmit(now, &mut transmit_scratch).await {
                        Ok(Some((len, peer))) => {
                            if let Err(err) = batch.send.try_append(&transmit_scratch[..len], peer) {
                                // alloc tier grows; this fires only on u32 arena overflow.
                                // The dropped datagram is recovered by QUIC loss recovery.
                                warn!(?err, %peer, "h3-native send-batch append dropped datagram");
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(err) => {
                            warn!(?err, "h3-native transmit error");
                            break;
                        }
                    }
                }
                if !batch.send.is_empty() {
                    let staged = batch.send.len();
                    // Fully flush the staged burst, parking on backpressure exactly
                    // like the hand-rolled loop did: poll_drive ships a chunk then
                    // returns Ready when the kernel buffer fills; we re-arm Pending
                    // on a no-progress partial so the await parks on the waker the
                    // inner send already registered, instead of spinning.
                    let mut span_offset = 0;
                    let flush = poll_fn(|cx| {
                        match socket.poll_drive_send_batch(cx, &batch.send, &mut span_offset) {
                            Poll::Ready(Ok(())) if span_offset >= staged => Poll::Ready(Ok(())),
                            Poll::Ready(Ok(())) => Poll::Pending,
                            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
                            Poll::Pending => Poll::Pending,
                        }
                    })
                    .await;
                    if let Err(err) = flush {
                        warn!(?err, "h3-native send_batch error");
                    }
                }

                // Drive-to-fixpoint gate: this pass staged output, so a state
                // transition may have unblocked more to send — re-drive immediately
                // (skip the park, drain recv non-blocking) rather than parking with
                // work pending. NEVER park while the send buffer is non-empty: that
                // strands the output (the invariant a capped park violated and the
                // 8x100 errors exposed). Under sustained load this stays busy
                // (CPU-bound, correct); it parks on the deadline timer only when a
                // pass drains recv AND stages nothing — i.e. genuinely idle.
                //
                // `dirty` is the set the NEXT pass drives: keep re-driving the
                // SAME connections while the fixpoint gate keeps firing (a
                // state transition inside `drive_server_step` can need a
                // second call with no new inbound — see `dirty`'s doc at its
                // declaration above); only once a pass produces NOTHING new
                // for it (this branch's `else`) has it genuinely settled, so
                // it's safe to drop until the next real touch.
                if !batch.send.is_empty() || applied_response {
                    skip_wait = true;
                } else {
                    dirty.clear();
                }
            }
            Ok(())
        })
    }
}

/// Drive `drive_server_step` + `process_h3_events` for every handle in
/// `dirty` that has both live QUIC connection state AND live H3 state and
/// has reached `Established` — the driver opens streams, which is only
/// legal once 1-RTT keys are installed. On a driver-step error the
/// connection is closed with the most specific H3 error code the reason
/// string identifies (until the driver carries typed H3 error variants).
fn drive_dirty_connections(
    listener: &mut Listener<RustlsServerProvider>,
    h3_state: &mut BTreeMap<u32, PerConnection>,
    dirty: &std::collections::BTreeSet<u32>,
    dispatch: &PipeHandle,
    response_tx: &futures::channel::mpsc::UnboundedSender<DispatchResult>,
    in_flight: &mut FuturesUnordered<Pin<Box<dyn Future<Output = ()> + Send>>>,
) {
    for &handle_id in dirty {
        let Some(connection) = listener.connection_mut(ConnectionHandle(handle_id)) else {
            continue;
        };
        if !matches!(connection.state(), ConnectionState::Established(_)) {
            continue;
        }
        let Some(driver) = h3_state.get_mut(&handle_id) else {
            continue;
        };
        if let Err(err) = drive_server_step(connection, &mut driver.h3, &mut driver.driver_state) {
            // RFC 9114 requires connection-level errors to surface as
            // CONNECTION_CLOSE on the wire. close() transitions to
            // Closing; the next `transmit` pass ships the close frame
            // before the reap loop removes the entry.
            close_with_h3_code(connection, handle_id, &err);
            continue;
        }
        if let Err(err) = process_h3_events(ConnectionHandle(handle_id), driver, dispatch, response_tx, in_flight) {
            #[cfg(feature = "http3-part-source")]
            {
                warn!(?err, handle = handle_id, "h3-native request header decode failed; closing connection");
                let _ = connection.close(QPACK_DECOMPRESSION_FAILED, b"qpack decode failed");
            }
            #[cfg(not(feature = "http3-part-source"))]
            let _ = err; // owned path validates in feed_request; unreachable
        }
    }
}

struct PerConnection {
    h3: ServerConnection,
    driver_state: DriverState,
    pending: BTreeMap<u64, PendingRequest>,
}

impl PerConnection {
    fn new(part_source_mode: bool) -> Self {
        let mut connection = Self {
            h3: ServerConnection::new(Settings::default()),
            driver_state: DriverState::new(),
            pending: BTreeMap::new(),
        };
        connection.set_part_source_mode(part_source_mode);
        connection
    }

    /// Opt the H3 FSM into `Source`-mode request headers (spec key
    /// `part_source`, default false). See `request_head_from_source`.
    #[cfg(feature = "http3-part-source")]
    fn set_part_source_mode(&mut self, enabled: bool) {
        if enabled {
            self.h3.enable_header_source_mode();
        }
    }

    /// Without the `part-source` feature the spec key is inert.
    #[cfg(not(feature = "http3-part-source"))]
    fn set_part_source_mode(&mut self, _enabled: bool) {}
}

#[derive(Default)]
struct PendingRequest {
    headers: Option<Vec<(Vec<u8>, Vec<u8>)>>,
    /// `Source`-mode sibling of `headers`: the dispatch `Request` built
    /// DIRECTLY from stepping the lazy header source (no
    /// `Vec<DecodedField>` intermediate, no pairs re-clone) — the body
    /// attaches at dispatch time. Exactly one of `headers` /
    /// `request_head` is populated per request, by connection mode.
    #[cfg(feature = "http3-part-source")]
    request_head: Option<Request<Bytes>>,
    body: Vec<u8>,
    finished: bool,
    dispatched: bool,
    response: Option<CollectedResponse>,
    response_emitted: bool,
}

struct DispatchResult {
    connection_handle: u32,
    stream_id: u64,
    response: Result<CollectedResponse, ProximaError>,
}

struct CollectedResponse {
    status: u16,
    response_headers: Vec<(Vec<u8>, Vec<u8>)>,
    chunks: Vec<Bytes>,
}

/// Build the dispatch `Request` head straight from a stepped lazy header
/// source — the `Source`-mode replacement for `decode_bounded` → owned
/// event → pairs clone → [`build_request_from_h3`]. Pseudo-headers other
/// than `:method`/`:path` are skipped, matching `build_request_from_h3`.
///
/// # Errors
///
/// A deferred decode failure ([`FieldSectionSource::error`]) — the
/// caller MUST treat it as QPACK_DECOMPRESSION_FAILED (connection-fatal).
#[cfg(feature = "http3-part-source")]
fn request_head_from_source(
    source: &mut proxima_protocols::http3_codec::qpack::part_source::FieldSectionSource<'_>,
) -> Result<Request<Bytes>, ProximaError> {
    use proxima_primitives::pipe::part::{Part, PartSource as _};

    let mut method = Bytes::from_static(b"GET");
    let mut path = Bytes::from_static(b"/");
    let mut header_list = HeaderList::new();
    while let Some(part) = source.next() {
        match part {
            Part::Method(bytes) => method = Bytes::copy_from_slice(bytes),
            Part::Path(bytes) => path = Bytes::copy_from_slice(bytes),
            Part::Header(name, value) if !name.starts_with(b":") => {
                header_list.insert(name, value);
            }
            _ => {}
        }
    }
    if let Some(err) = source.error() {
        return Err(ProximaError::Upstream(format!(
            "h3 request header decode: {err:?}"
        )));
    }
    let mut built = Request::builder()
        .method(method)
        .path(path)
        .body(Bytes::new())
        .context(RequestContext::default())
        .build()?;
    built.metadata = header_list;
    Ok(built)
}

/// `Err` means a request header section failed QPACK decode — the caller
/// MUST close the connection with [`QPACK_DECOMPRESSION_FAILED`]
/// (RFC 9204 §8.3). Only the `part-source` `Source` mode can produce it;
/// the owned event path validates during `feed_request` instead.
fn process_h3_events(
    handle: ConnectionHandle,
    driver: &mut PerConnection,
    dispatch: &PipeHandle,
    response_tx: &futures::channel::mpsc::UnboundedSender<DispatchResult>,
    in_flight: &mut FuturesUnordered<Pin<Box<dyn Future<Output = ()> + Send>>>,
) -> Result<(), ProximaError> {
    #[cfg(feature = "http3-part-source")]
    while let Some((stream_id, mut source)) = driver.h3.poll_request_header_source() {
        let head = request_head_from_source(&mut source)?;
        let entry = driver.pending.entry(stream_id.0).or_default();
        entry.request_head = Some(head);
    }
    while let Some(event) = driver.h3.poll_event() {
        match event {
            H3ServerEvent::SettingsEstablished { .. } => {
                debug!(handle = handle.0, "h3-native SETTINGS established");
            }
            H3ServerEvent::RequestHeaders { stream_id, headers } => {
                let entry = driver.pending.entry(stream_id.0).or_default();
                let pairs: Vec<(Vec<u8>, Vec<u8>)> = headers
                    .into_iter()
                    .map(|field| (field.name.to_vec(), field.value.to_vec()))
                    .collect();
                entry.headers = Some(pairs);
            }
            H3ServerEvent::RequestData { stream_id, bytes } => {
                let entry = driver.pending.entry(stream_id.0).or_default();
                entry.body.extend(bytes);
            }
            H3ServerEvent::RequestFinished { stream_id } => {
                let entry = driver.pending.entry(stream_id.0).or_default();
                entry.finished = true;
            }
            H3ServerEvent::RequestTrailers { .. } => {}
            H3ServerEvent::GoAway { .. } => {
                debug!(handle = handle.0, "h3-native peer sent GOAWAY");
            }
            _ => {}
        }
    }

    let dispatch_ids: Vec<u64> = driver
        .pending
        .iter()
        .filter_map(|(id, pending)| {
            let has_head = pending.headers.is_some();
            #[cfg(feature = "http3-part-source")]
            let has_head = has_head || pending.request_head.is_some();
            if pending.finished && has_head && !pending.dispatched && pending.response.is_none() {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    for stream_id in dispatch_ids {
        let Some(pending) = driver.pending.get_mut(&stream_id) else {
            continue;
        };
        pending.dispatched = true;
        let headers = pending.headers.take().unwrap_or_default();
        let body = std::mem::take(&mut pending.body);
        #[cfg(feature = "http3-part-source")]
        let prebuilt_head = pending.request_head.take();
        #[cfg(not(feature = "http3-part-source"))]
        let prebuilt_head: Option<Request<Bytes>> = None;
        let request = if let Some(mut head) = prebuilt_head {
            head.payload = Bytes::from(body);
            head
        } else {
            match build_request_from_h3(&headers, body) {
                Ok(req) => req,
                Err(err) => {
                    let _ = response_tx.unbounded_send(DispatchResult {
                        connection_handle: handle.0,
                        stream_id,
                        response: Err(err),
                    });
                    continue;
                }
            }
        };
        let dispatch = dispatch.clone();
        let response_tx = response_tx.clone();
        let connection_handle = handle.0;
        in_flight.push(Box::pin(async move {
            let response = match dispatch.call_dyn(request).await {
                Ok(resp) => {
                    let status = resp.status;
                    let response_headers: Vec<(Vec<u8>, Vec<u8>)> = resp
                        .metadata
                        .iter()
                        .map(|(n, v)| (n.to_vec(), v.to_vec()))
                        .collect();
                    let mut body_stream = resp.into_chunk_stream();
                    let mut chunks: Vec<Bytes> = Vec::new();
                    let mut chunk_error = None;
                    while let Some(chunk) = futures::StreamExt::next(&mut body_stream).await {
                        match chunk {
                            Ok(bytes) => chunks.push(bytes),
                            Err(err) => {
                                chunk_error =
                                    Some(ProximaError::Upstream(format!("h3 body collect: {err}")));
                                break;
                            }
                        }
                    }
                    match chunk_error {
                        Some(err) => Err(err),
                        None => Ok(CollectedResponse {
                            status,
                            response_headers,
                            chunks,
                        }),
                    }
                }
                Err(err) => Err(err),
            };
            let _ = response_tx.unbounded_send(DispatchResult {
                connection_handle,
                stream_id,
                response,
            });
        }));
    }
    Ok(())
}

fn apply_dispatch_response(driver: &mut PerConnection, result: DispatchResult) {
    let entry = driver.pending.entry(result.stream_id).or_default();
    match result.response {
        Ok(collected) => {
            entry.response = Some(collected);
        }
        Err(err) => {
            warn!(?err, stream = result.stream_id, "h3-native dispatch error");
            entry.response = Some(CollectedResponse {
                status: 500,
                response_headers: Vec::new(),
                chunks: Vec::new(),
            });
        }
    }
}

fn flush_pending_responses(driver: &mut PerConnection) -> Result<(), ProximaError> {
    let ready_ids: Vec<u64> = driver
        .pending
        .iter()
        .filter_map(|(id, pending)| {
            if pending.response.is_some() && !pending.response_emitted {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    for stream_id in ready_ids {
        let Some(pending) = driver.pending.get_mut(&stream_id) else {
            continue;
        };
        let Some(response) = pending.response.take() else {
            continue;
        };
        pending.response_emitted = true;
        let status_bytes = format!("{}", response.status);
        let mut header_pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        header_pairs.push((b":status".to_vec(), status_bytes.into_bytes()));
        header_pairs.extend(response.response_headers);
        let header_refs: Vec<(&[u8], &[u8])> = header_pairs
            .iter()
            .map(|(n, v)| (n.as_slice(), v.as_slice()))
            .collect();
        driver
            .h3
            .send_response_headers(H3StreamId(stream_id), &header_refs)
            .map_err(|err| ProximaError::Upstream(format!("h3 send_response_headers: {err:?}")))?;
        for chunk in response.chunks {
            if chunk.is_empty() {
                continue;
            }
            driver
                .h3
                .send_response_data(H3StreamId(stream_id), &chunk)
                .map_err(|err| ProximaError::Upstream(format!("h3 send_response_data: {err:?}")))?;
        }
        driver
            .h3
            .finish_response(H3StreamId(stream_id))
            .map_err(|err| ProximaError::Upstream(format!("h3 finish_response: {err:?}")))?;
        // Response fully queued into the H3 layer — the request is done.
        // Drop its bookkeeping so the per-burst scans of `pending` and the
        // driver maps stay O(in-flight), not O(lifetime requests).
        driver.pending.remove(&stream_id);
        driver.driver_state.forget_stream(stream_id);
    }
    Ok(())
}

fn build_request_from_h3(
    headers: &[(Vec<u8>, Vec<u8>)],
    body: Vec<u8>,
) -> Result<Request<Bytes>, ProximaError> {
    let mut method = Bytes::from_static(b"GET");
    let mut path = Bytes::from_static(b"/");
    let mut header_list = HeaderList::new();
    for (name, value) in headers {
        if name.as_slice() == b":method" {
            method = Bytes::copy_from_slice(value);
        } else if name.as_slice() == b":path" {
            path = Bytes::copy_from_slice(value);
        } else if name.starts_with(b":") {
            continue;
        } else {
            header_list.insert(name.as_slice(), value.as_slice());
        }
    }
    let mut built = Request::builder()
        .method(method)
        .path(path)
        .body(Bytes::from(body))
        .context(RequestContext::default())
        .build()?;
    built.metadata = header_list;
    Ok(built)
}

/// RFC 9000 §18.2 — server transport parameters MUST include the
/// client's original DCID and the server's chosen SCID. Clients
/// reject server flights that omit these.
fn encode_server_transport_parameters(original_dcid: &[u8], local_scid: &[u8]) -> Vec<u8> {
    use proxima_protocols::quic::transport_parameters::TransportParameters;
    let mut buf = vec![0u8; 384];
    let written = TransportParameters {
        original_destination_connection_id: Some(original_dcid),
        initial_source_connection_id: Some(local_scid),
        initial_max_data: Some(1_048_576),
        max_idle_timeout_ms: Some(30_000),
        initial_max_stream_data_bidi_local: Some(65_536),
        initial_max_stream_data_bidi_remote: Some(65_536),
        initial_max_stream_data_uni: Some(65_536),
        // advertise the ACTUAL per-connection stream-table capacity, not a
        // fantasy ceiling: the table is a fixed-cap heapless map sized by
        // `sized::STREAMS_MAX_CONCURRENT_*`. promising more lets a conformant
        // peer open streams we cannot hold, which we then (wrongly) blamed on
        // the peer with a ProtocolViolation. MAX_STREAMS reissue on close
        // keeps cumulative credit flowing for sequential reuse.
        initial_max_streams_bidi: Some(
            proxima_protocols::quic::sized::STREAMS_MAX_CONCURRENT_BIDI as u64,
        ),
        initial_max_streams_uni: Some(proxima_protocols::quic::sized::STREAMS_MAX_CONCURRENT_UNI as u64),
        ..Default::default()
    }
    .encode(&mut buf)
    .unwrap_or(0);
    buf.truncate(written);
    buf
}

/// The loop's monotonic `now` (driver-agnostic `proxima_core::time::Instant`,
/// what `Listener::on_datagram`/`transmit` and `next_deadline` all speak),
/// derived from the single injected [`Clock`] so it and the timer sleep can
/// never diverge. The absolute clock reading is used directly — no origin
/// anchoring — because [`to_proto_instant`] below MUST convert it the exact
/// same way `Listener` converts internally; see that function's doc for why
/// this matters. Mirrors `proxima_listen::stream::datagram_protocol_listener`'s
/// private `instant_now` (not exported; this is the one-line equivalent).
fn core_instant_now<Clk: Clock>(clock: &Clk) -> proxima_core::time::Instant {
    proxima_core::time::Instant::from_monotonic(Duration::from_nanos(clock.now_nanos()))
}

/// Convert the tick's already-sampled driver-agnostic `now` into the QUIC
/// proto layer's microsecond `Instant` — needed ONLY for
/// `Listener::handle_timeout`, the one remaining call that takes the proto
/// layer's own `Instant` rather than the `proxima_core::time::Instant` every
/// `DatagramProtocol` trait method (`on_datagram`/`transmit`) takes.
///
/// MUST be byte-for-byte the same conversion `Listener::on_datagram` applies
/// internally (nanos → micros, absolute reading, no origin subtraction) —
/// `Connection::new_server_with_limits` anchors a freshly-accepted
/// connection's `handshake_completion_deadline` to THAT internal
/// conversion's output. An earlier version of this function instead
/// computed micros ELAPSED SINCE `serve()`'s start (mirroring the
/// pre-`Listener<P>` code, which owned its OWN connection construction and
/// so could pick any consistent epoch it liked) — with `Listener<P>` now
/// owning construction, that put accept-time anchoring and this reap call
/// on TWO DIFFERENT EPOCHS: accept anchored to the huge absolute-clock
/// value, reap compared against the tiny elapsed-since-origin value, so
/// `now >= handshake_completion_deadline` evaluated true immediately and
/// every freshly-accepted connection was reaped within microseconds of
/// being accepted. Caught by
/// `proxima-http/tests/native_listener_stale_now_reap.rs` (same failure
/// signature as the bug that test already guards, different root cause —
/// a clock-epoch mismatch introduced by composing onto `Listener<P>`,
/// not the pre-await/post-await `now` staleness the test was originally
/// written for).
fn to_proto_instant(now: proxima_core::time::Instant) -> ProtoInstant {
    ProtoInstant::from_micros(u64::try_from(now.into_monotonic().as_micros()).unwrap_or(u64::MAX))
}

/// Parse runtime [`HandshakeLimits`] from the listener spec `Value`.
///
/// Fields map 1:1 from JSON spec keys to limit fields; absent or
/// non-integer values fall back to the build-time floor via
/// [`HandshakeLimits::default()`]. This is the conflaguration override
/// surface: set `handshake_completion_micros` LOW for strict half-open
/// defence or HIGH for high-RTT environments — no rebuild required.
fn parse_handshake_limits_from_spec(spec: &Value) -> HandshakeLimits {
    let defaults = HandshakeLimits::default();
    HandshakeLimits {
        early_data_max_bytes: spec
            .get("handshake_early_data_max_bytes")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(defaults.early_data_max_bytes),
        early_data_max_datagrams: spec
            .get("handshake_early_data_max_datagrams")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(defaults.early_data_max_datagrams),
        early_data_hold_micros: spec
            .get("handshake_early_data_hold_micros")
            .and_then(Value::as_u64)
            .unwrap_or(defaults.early_data_hold_micros),
        handshake_completion_micros: spec
            .get("handshake_completion_micros")
            .and_then(Value::as_u64)
            .unwrap_or(defaults.handshake_completion_micros),
    }
}

fn build_rustls_server_config(spec: &Value) -> Result<Arc<rustls::ServerConfig>, ProximaError> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    if spec
        .get("dev_self_signed")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let sans = spec
            .get("dev_sans")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .filter(|sans| !sans.is_empty())
            .unwrap_or_else(|| vec!["localhost".to_string()]);
        let cert = rcgen::generate_simple_self_signed(sans)
            .map_err(|err| ProximaError::Upstream(format!("h3-native rcgen: {err}")))?;
        let cert_der = cert.cert.der().clone();
        let key_pkcs8 = cert.signing_key.serialize_der();
        let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.to_vec())];
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(key_pkcs8),
        );
        let mut server_config =
            rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
                .with_no_client_auth()
                .with_single_cert(chain, key)
                .map_err(|err| ProximaError::Upstream(format!("h3-native server config: {err}")))?;
        server_config.alpn_protocols = vec![ALPN_H3.to_vec()];
        return Ok(Arc::new(server_config));
    }
    let cert_path = spec
        .get("cert_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| ProximaError::Config("h3-native listener missing `cert_path`".into()))?;
    let key_path = spec
        .get("key_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| ProximaError::Config("h3-native listener missing `key_path`".into()))?;
    let cert_bytes = std::fs::read(&cert_path).map_err(|err| {
        ProximaError::Upstream(format!("h3-native cert read {cert_path:?}: {err}"))
    })?;
    let key_bytes = std::fs::read(&key_path)
        .map_err(|err| ProximaError::Upstream(format!("h3-native key read {key_path:?}: {err}")))?;
    let certs = <rustls::pki_types::CertificateDer<'_> as rustls::pki_types::pem::PemObject>::pem_slice_iter(
        &cert_bytes,
    )
    .collect::<Result<Vec<_>, _>>()
    .map_err(|err| ProximaError::Upstream(format!("h3-native cert parse: {err}")))?;
    let key = <rustls::pki_types::PrivateKeyDer<'_> as rustls::pki_types::pem::PemObject>::from_pem_slice(
        &key_bytes,
    )
    .map_err(|err| ProximaError::Upstream(format!("h3-native key parse: {err}")))?;
    let mut tls = rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|err| ProximaError::Upstream(format!("h3-native server config: {err}")))?;
    tls.alpn_protocols = vec![ALPN_H3.to_vec()];
    Ok(Arc::new(tls))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    // The CID-length + Version-Negotiation regression guards that used to
    // live here now target `proxima_quic::native::listener::Listener<P>`
    // directly, at the source — that logic moved INTO the listener as part
    // of the composition fold (see the module doc), so testing it here
    // would just be testing the same code through an extra layer of
    // indirection. See `proxima-quic/src/native/listener/tests.rs`:
    // `client_dcid_preserves_the_full_20_byte_quinn_length`,
    // `client_dcid_keeps_the_common_8_byte_length`,
    // `client_dcid_keeps_a_zero_length_cid`,
    // `client_dcid_rejects_over_max_length`,
    // `version_negotiation_echoes_swapped_cids_and_offers_v1`,
    // `unsupported_version_datagram_queues_a_vn_reply_drained_first_by_transmit`,
    // `second_initial_still_addressed_to_the_clients_own_dcid_routes_to_the_existing_connection`.

    // The regression guard for the clock-epoch bug `to_proto_instant`'s doc
    // comment describes: `core_instant_now` (fed to `Listener::on_datagram`,
    // which anchors a freshly-accepted connection's deadlines) and
    // `to_proto_instant` (fed to `Listener::handle_timeout` for reap) MUST
    // read the SAME absolute epoch — an ELAPSED-since-origin `now` for one
    // and an ABSOLUTE `now` for the other reaped every connection within
    // microseconds of accepting it. A large, non-zero clock reading (as
    // production's `TimeClock` returns; a `MockDriver` starting near zero
    // would not have caught this) is the case that actually exposes the
    // two-epoch mismatch.
    #[test]
    fn to_proto_instant_matches_the_absolute_epoch_core_instant_now_reads() {
        struct FixedClock {
            nanos: u64,
        }
        impl Clock for FixedClock {
            type Delay = core::future::Ready<()>;
            fn now_nanos(&self) -> u64 {
                self.nanos
            }
            fn delay(&self, _dur: Duration) -> Self::Delay {
                core::future::ready(())
            }
        }

        // A large absolute reading, exactly the shape `TimeClock` returns in
        // production (nowhere near zero) — the case that actually exercises
        // the epoch mismatch this test guards against.
        let clock = FixedClock { nanos: 1_700_000_000_123_456_000 };
        let now = core_instant_now(&clock);
        let proto = to_proto_instant(now);

        assert_eq!(
            proto.as_micros(),
            u64::try_from(now.into_monotonic().as_micros()).expect("fits u64"),
            "to_proto_instant must read the SAME absolute epoch core_instant_now does — \
             not elapsed-since-some-other-origin, or accept-time anchoring and the reap \
             comparison land on different timelines"
        );
        assert_eq!(
            proto.as_micros(),
            clock.now_nanos() / 1_000,
            "the conversion is a direct absolute nanos-to-micros truncation, no origin subtraction"
        );
    }
}
