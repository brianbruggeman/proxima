//! Native HTTP/3 [`ListenProtocol`] over the sans-IO proxima-quic-proto stack +
//! the driver in [`crate::http3::native::driver`] — no quinn, no h3-quinn, **no tokio**.
//!
//! Runtime-agnostic IO: the UDP socket comes from the [`ServeContext`]'s
//! `DatagramFactory` (the UDP sibling of the TCP `AcceptorFactory`) and the 1 ms
//! tick from `proxima_core::time::sleep` (whose driver is the build-selected one — the
//! prime per-core wheel under `timer = "prime-wheel"`). The listener names
//! neither prime nor tokio; either runtime supplies the factory + timer driver.
//!
//! The listener inlines the EndpointDemux + per-connection book-keeping rather
//! than going through [`proxima_quic::native::Listener`]; sharing them is future
//! work.
//!
//! Scope: single-task accept + per-connection driver fan-in. Each tick:
//!
//! 1. Recv any pending UDP datagram, classify via EndpointDemux, route
//!    to an existing connection or create a new one.
//! 2. For each connection, drive the H3 state machine + pump H3 events
//!    into a per-stream pending-request map.
//! 3. Once a request's HEADERS + FIN both arrive, spawn the
//!    `PipeHandle::call_dyn` future on tokio. The result comes back
//!    via an mpsc channel.
//! 4. The next tick consumes ready responses + ships HEADERS+DATA+FIN
//!    back through the H3 state machine, then drains the QUIC layer's
//!    outbound queue to the socket.
//!
//! Future work (out of scope for v1):
//! - request-body streaming (we currently buffer to FIN before
//!   dispatching; fine for GET, would need a tx end of an mpsc for
//!   bidirectional POST).
//! - proper per-connection task fan-out (today everything runs in the
//!   listener task).
//! - sharing the demux + driver with `proxima_quic::native::Listener`
//!   once that grows a tokio-backed transport.

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
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, StreamId as H3StreamId};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_protocols::quic::connection::{Connection, DatagramWrite, HandshakeLimits};
use proxima_protocols::quic::endpoint::{
    ConnectionHandle, ConnectionIdBytes, DatagramClassification, DropReason, EndpointDemux,
};
use proxima_protocols::quic::time::Instant as ProtoInstant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsConfig, RustlsServerProvider};
use proxima_primitives::stream::DatagramSocketBatchExt;

use super::driver::{DriverState, drive_server_step};

const ALPN_H3: &[u8] = b"h3";
/// QUIC version 1 (RFC 9000), big-endian, for the supported-versions list
/// of an outbound Version Negotiation packet.
const QUIC_V1_VERSION: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
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

/// Map a `handle_datagram` failure to the appropriate CONNECTION_CLOSE
/// frame on the wire. Returns `true` if the connection should be
/// torn down; `false` when the error is the kind RFC 9000 §10.3
/// classifies as "silently discard the packet" (header parse, decrypt
/// failure, anti-amplification reject, etc.). The caller logs either
/// way but only the `true` return path issues a CONNECTION_CLOSE.
fn close_for_datagram_error<P: proxima_protocols::quic::tls::TlsProvider>(
    connection: &mut proxima_protocols::quic::connection::Connection<P>,
    err: &proxima_protocols::quic::connection::ConnectionError,
) -> bool {
    use proxima_protocols::quic::connection::ConnectionError;
    match err {
        ConnectionError::FlowControlError { reason } => {
            // RFC 9000 §20.1 FLOW_CONTROL_ERROR = 0x03; the
            // triggering frame is STREAM (0x08..0x0f). We pick 0x08
            // as the canonical STREAM type.
            let _ = connection.close_transport(0x03, 0x08, reason.as_bytes());
            true
        }
        ConnectionError::ProtocolViolation { reason } => {
            // RFC 9000 §20.1 PROTOCOL_VIOLATION = 0x0a.
            let _ = connection.close_transport(0x0a, 0, reason.as_bytes());
            true
        }
        // RFC 9000 §10.3 — packets that fail header parsing,
        // decryption, or AEAD authentication "MUST be silently
        // discarded"; they do NOT terminate the connection. A peer
        // can ship a short-header 1-RTT packet at any moment after
        // the handshake; if our state machine isn't there yet,
        // returning Err is normal and dropping the connection would
        // break the very interop case the test
        // `listener_h3_native::h3_native_listener_round_trip`
        // exercises (quinn client → native server: ships short-header
        // packets while we're still mid-Handshake).
        ConnectionError::Frame(_) => {
            // RFC 9000 §12.4 — malformed frame after successful
            // decryption is FRAME_ENCODING_ERROR (0x07).
            let _ = connection.close_transport(0x07, 0, b"frame encoding error");
            true
        }
        ConnectionError::Header(_)
        | ConnectionError::PacketProtection(_)
        | ConnectionError::Aead(_)
        | ConnectionError::PacketNumber(_)
        // TransientRecvBufferFull: data was within our advertised
        // credit but exceeded our reassembly buffer cap. NOT a peer
        // violation; we deliberately skipped ACK so the peer
        // retransmits once we've drained. No CLOSE warranted.
        | ConnectionError::TransientRecvBufferFull { .. } => false,
        _ => {
            // For everything else (crypto error, version negotiation,
            // etc.) take the application close path — Connection::close
            // already maps INTERNAL_ERROR-ish failures to 0x1d.
            let _ = connection.close(H3_GENERAL_PROTOCOL_ERROR, b"handle_datagram failed");
            true
        }
    }
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
        let local_tp = encode_default_transport_parameters();
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

            // Fixed 8-byte local CIDs (see generate_local_scid below) →
            // EndpointDemux short-header dispatch goes through the
            // O(1) hash path, not the O(N) linear scan.
            let mut demux = EndpointDemux::with_local_cid_len(
                proxima_protocols::quic::connection::SUPPORTED_VERSIONS,
                8,
            );
            let mut connections: BTreeMap<u32, ConnEntry> = BTreeMap::new();
            let mut next_handle: u32 = 0;
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

            // Reused per-iteration scratch for the connection-handle
            // snapshot each pass needs (drive / flush / timeout / send).
            // Snapshotting decouples iteration from the &mut connections
            // borrow inside each body; hoisting the buffer keeps it off
            // the per-iteration alloc path (was 4 Vec allocs/iteration).
            let mut handle_scratch: Vec<u32> = Vec::new();
            // Drive-to-fixpoint gate: after an event, re-run drive+transmit while a
            // pass stages output (a state transition can unblock more to send — e.g.
            // the Established transition emits HANDSHAKE_DONE then SETTINGS across
            // passes). It parks only when a pass stages nothing — NEVER with output
            // pending. The event-driven replacement for the old 1ms re-flush tick:
            // zero tick latency, and under load it stays busy instead of stranding
            // work behind a cap.
            let mut skip_wait = false;
            // Origin reading of the injected clock, captured ONCE. `proto_now`
            // reports micros ELAPSED since this — the same serve-start-relative
            // ProtoInstant the loop always fed the QUIC deadline math (the old
            // `std::time::Instant` origin did exactly this via `elapsed()`).
            // Anchoring keeps the values small and identical in production
            // (`TimeClock`, a large absolute `now_nanos`) and under a mock clock
            // (starts at 0) — the clock is only the source, never the base.
            let origin_nanos = clock.now_nanos();
            loop {
                // Sampled BEFORE the recv/timer/handlers race below, which can
                // park for an arbitrarily long time on an idle socket — used
                // ONLY to size `next_delay` (a duration relative to tick
                // start). Any connection created or timed-out AFTER the
                // await resolves must use a freshly-sampled `now` instead:
                // reusing this pre-park value anchored a fresh connection's
                // handshake_completion_deadline to the moment the tick
                // started WAITING, not the moment its first datagram
                // actually arrived. On a socket idle longer than
                // HANDSHAKE_COMPLETION_MICROS (10s — routine under c1m32,
                // where one of several SO_REUSEPORT-sharded listener
                // instances can sit idle between sequential connections),
                // the very next reap pass saw its own fresh `now` already
                // past that deadline and reaped the connection within
                // microseconds of accepting it — orphaning the client's
                // in-flight Initial/Handshake retransmits, which then
                // misrouted onto a phantom replacement connection and
                // tripped "non-Initial packet received in Initial state".
                let tick_start = proto_now(&clock, origin_nanos);

                // 1) Receive a burst of datagrams in ONE `recvmmsg` syscall,
                //    raced against the connections' earliest real QUIC deadline
                //    and the shutdown signal — every arm is an event source, all
                //    async. recv wakes instantly on data. The timer is a source
                //    too: `sleep` schedules a wake on the bound time driver (the
                //    prime per-core wheel today; a DPDK/hardware timer under
                //    kernel-bypass — same code, link-bound), which the reactor
                //    arms as its kernel epoll/kqueue timeout. NO fixed tick: we
                //    arm the EXACT next protocol deadline, or NOTHING when none is
                //    pending — a quiescent socket costs zero wakeups, a live one
                //    pays zero tick latency.
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
                    // Earliest deadline across all connections, as a delay from
                    // `now`; `None` => nothing due => no timer armed.
                    let next_delay = connections
                        .values()
                        .filter_map(|entry| entry.connection.next_timeout())
                        .map(ProtoInstant::as_micros)
                        .min()
                        .map(|deadline| {
                            Duration::from_micros(deadline.saturating_sub(tick_start.as_micros()))
                        });
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
                        // timer fired — pump QUIC timeouts in step 6 below.
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
                // the timestamp every connection created or timed-out below
                // must be anchored to, not `tick_start`.
                let now = proto_now(&clock, origin_nanos);
                if let Some(Err(ref err)) = recv_outcome {
                    debug!(
                        ?err,
                        "h3-native poll_recv_batch error; datagrams dropped before handle_inbound"
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
                    // Version Negotiation replies (RFC 9000 §6) accumulate
                    // here across the recv batch; the sync handler can't await
                    // the socket, so the async loop flushes them below.
                    let mut pending_vn: Vec<(Vec<u8>, SocketAddr)> = Vec::new();
                    for view in batch.recv.filled_datagrams() {
                        debug!(len = view.bytes.len(), peer = %view.peer, "h3-native recv datagram");
                        handle_inbound(
                            view.bytes,
                            view.peer,
                            &mut demux,
                            &mut connections,
                            &mut next_handle,
                            &server_config,
                            &local_tp,
                            &mut pending_vn,
                            now,
                            handshake_limits,
                            part_source_mode,
                        );
                    }
                    batch.recv.clear();
                    // Flush any Version Negotiation replies the batch produced.
                    for (vn_bytes, vn_peer) in pending_vn.drain(..) {
                        if let Err(err) =
                            poll_fn(|cx| socket.poll_send_to(cx, &vn_bytes, vn_peer)).await
                        {
                            warn!(?err, %vn_peer, "h3-native version-negotiation send_to error");
                        }
                    }
                }

                // 2) Drive each connection that has reached Established.
                // The driver opens streams; that's only legal once 1-RTT
                // keys are installed.
                handle_scratch.clear();
                handle_scratch.extend(connections.keys().copied());
                for handle_id in handle_scratch.iter().copied() {
                    if let Some(entry) = connections.get_mut(&handle_id) {
                        if !matches!(
                            entry.connection.state(),
                            proxima_protocols::quic::connection::ConnectionState::Established(_)
                        ) {
                            continue;
                        }
                        // Established: the client now addresses us by our SCID,
                        // so the handshake-only ODCID route is dead weight in
                        // the bounded demux. Free it once to keep long-lived
                        // connections at one table entry each.
                        if let Some(odcid) = entry.original_dcid.take() {
                            let _ = demux.unregister(&odcid);
                        }
                        if let Err(err) = drive_server_step(
                            &mut entry.connection,
                            &mut entry.driver.h3,
                            &mut entry.driver.driver_state,
                        ) {
                            // RFC 9114 requires connection-level errors
                            // to surface as CONNECTION_CLOSE on the
                            // wire. close() transitions to Closing; the
                            // next poll_transmit pass ships the close
                            // frame before the reap loop removes the
                            // entry. H3_GENERAL_PROTOCOL_ERROR is the
                            // catch-all for protocol violations that
                            // don't map to a more specific code; future
                            // refinement can plumb (driver-error →
                            // H3_FRAME_UNEXPECTED / H3_SETTINGS_ERROR /
                            // H3_MISSING_SETTINGS / etc.) through the
                            // driver's ConnectionError variants.
                            // map the driver error to the most specific
                            // H3 error code we can determine from the
                            // reason string (until the driver carries
                            // typed H3 error variants)
                            let reason = err_reason_for_close(&err);
                            let h3_code = if reason
                                .windows(b"CLOSED_CRITICAL_STREAM".len())
                                .any(|w| w == b"CLOSED_CRITICAL_STREAM")
                            {
                                H3_CLOSED_CRITICAL_STREAM
                            } else {
                                H3_GENERAL_PROTOCOL_ERROR
                            };
                            warn!(
                                ?err,
                                handle = handle_id,
                                "h3-native driver step error; closing connection"
                            );
                            let _ = entry.connection.close(h3_code, reason);
                            continue;
                        }
                        if let Err(err) = process_h3_events(
                            ConnectionHandle(handle_id),
                            &mut entry.driver,
                            &dispatch,
                            &response_tx,
                            &mut in_flight,
                        ) {
                            #[cfg(feature = "http3-part-source")]
                            {
                                warn!(
                                    ?err,
                                    handle = handle_id,
                                    "h3-native request header decode failed; closing connection"
                                );
                                let _ = entry
                                    .connection
                                    .close(QPACK_DECOMPRESSION_FAILED, b"qpack decode failed");
                            }
                            #[cfg(not(feature = "http3-part-source"))]
                            let _ = err; // owned path validates in feed_request; unreachable
                            continue;
                        }
                    }
                }

                // 2b) Drive ready response handlers to completion NOW so their
                // responses emit in THIS iteration instead of trickling across
                // subsequent ticks. `now_or_never` polls each ready handler
                // once; a handler that genuinely yields stays queued and is
                // woken via the `handlers` arm of the select below.
                while in_flight.next().now_or_never().flatten().is_some() {}

                // 3) Apply any ready responses. A freshly applied response is
                // queued into the H3 layer by step 4 below, but step 2's drive
                // (which moves H3 -> QUIC) already ran this pass — so without a
                // re-drive the response sits in H3 until the next drive and the
                // loop parks (timer-woken ~400us/request) before sending it.
                // `applied_response` forces the fixpoint re-drive at the gate.
                let mut applied_response = false;
                while let Ok(result) = response_rx.try_recv() {
                    if let Some(entry) = connections.get_mut(&result.connection_handle) {
                        apply_dispatch_response(&mut entry.driver, result);
                        applied_response = true;
                    }
                }

                // 4) Emit pending responses through H3.
                handle_scratch.clear();
                handle_scratch.extend(connections.keys().copied());
                for handle_id in handle_scratch.iter().copied() {
                    if let Some(entry) = connections.get_mut(&handle_id)
                        && let Err(err) = flush_pending_responses(&mut entry.driver)
                    {
                        warn!(?err, handle = handle_id, "h3-native response flush error");
                    }
                }

                // Per-connection timer tick so PTOs / idle deadlines
                // advance. Reap any connection that transitioned to a
                // terminal state (idle-timed-out, drained, closed) so
                // the BTreeMap + EndpointDemux don't grow unbounded
                // with stale entries — that's a DoS surface on its own.
                let mut reap: Vec<ReapTarget> = Vec::new();
                handle_scratch.clear();
                handle_scratch.extend(connections.keys().copied());
                for handle_id in handle_scratch.iter().copied() {
                    let Some(entry) = connections.get_mut(&handle_id) else {
                        continue;
                    };
                    let _ = entry.connection.handle_timeout(now);
                    if matches!(
                        entry.connection.state(),
                        proxima_protocols::quic::connection::ConnectionState::Closed
                            | proxima_protocols::quic::connection::ConnectionState::Draining(_)
                    ) {
                        reap.push((handle_id, entry.local_scid, entry.original_dcid.take()));
                    }
                }
                for (handle_id, scid, odcid) in reap {
                    connections.remove(&handle_id);
                    if let Some(scid) = scid {
                        let _ = demux.unregister(&scid);
                    }
                    if let Some(odcid) = odcid {
                        let _ = demux.unregister(&odcid);
                    }
                    debug!(handle = handle_id, "h3-native reaped terminal connection");
                }

                // 5) Stage every outbound datagram across all connections into
                // one contiguous arena, then ship the whole burst with a single
                // `sendmmsg`. Staging costs one small memcpy per packet (the
                // response is a handful of bytes); batching the syscall is the
                // win — per-packet `sendto` was the send-side floor.
                batch.send.reset();
                let mut transmit_scratch = [0u8; 2048];
                handle_scratch.clear();
                handle_scratch.extend(connections.keys().copied());
                for handle_id in handle_scratch.iter().copied() {
                    let Some(entry) = connections.get_mut(&handle_id) else {
                        continue;
                    };
                    let peer = entry.peer;
                    loop {
                        let len = match entry.connection.poll_transmit(now, &mut transmit_scratch) {
                            Ok(Some(DatagramWrite { len, .. })) => len,
                            Ok(None) => break,
                            Err(err) => {
                                warn!(?err, handle = handle_id, "h3-native poll_transmit error");
                                break;
                            }
                        };
                        if let Err(err) = batch.send.try_append(&transmit_scratch[..len], peer) {
                            // alloc tier grows; this fires only on u32 arena overflow.
                            // The dropped datagram is recovered by QUIC loss recovery.
                            warn!(
                                ?err,
                                handle = handle_id,
                                "h3-native send-batch append dropped datagram"
                            );
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
                if !batch.send.is_empty() || applied_response {
                    skip_wait = true;
                }
            }
            Ok(())
        })
    }
}

/// A connection to reap: its handle plus the two demux keys (local SCID,
/// client ODCID) to unregister so the bounded table stays clean.
type ReapTarget = (u32, Option<[u8; 8]>, Option<ConnectionIdBytes>);

struct ConnEntry {
    /// `Connection<RustlsServerProvider>` carries inline send/recv
    /// buffers + the rustls quic-connection trait object; on tokio
    /// worker stacks it overflows. Box-allocate so each connection
    /// owns one heap allocation rather than living on the listener's
    /// stack frame.
    connection: Box<Connection<RustlsServerProvider>>,
    peer: SocketAddr,
    /// The 8-byte SCID we registered in `EndpointDemux` when this
    /// connection was accepted, retained so we can `demux.unregister`
    /// on terminal-state reap and keep the table bounded.
    local_scid: Option<[u8; 8]>,
    /// The client's original DCID, also registered in the demux so
    /// CRYPTO-fragmented / retransmitted Initials route here. Retained so
    /// reap unregisters BOTH entries — otherwise it leaks and the bounded
    /// demux table fills under load. Length is the CLIENT's choice (0..=20
    /// per RFC 9000 §17.2) — quinn uses the full 20 — so this MUST hold a
    /// variable-length CID, not a fixed [u8; 8].
    original_dcid: Option<ConnectionIdBytes>,
    driver: PerConnection,
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

#[allow(clippy::too_many_arguments)]
fn handle_inbound(
    datagram: &[u8],
    peer: SocketAddr,
    demux: &mut EndpointDemux,
    connections: &mut BTreeMap<u32, ConnEntry>,
    next_handle: &mut u32,
    server_config: &Arc<rustls::ServerConfig>,
    local_tp: &[u8],
    pending_vn: &mut Vec<(Vec<u8>, SocketAddr)>,
    now: ProtoInstant,
    handshake_limits: HandshakeLimits,
    part_source_mode: bool,
) {
    let class = demux.classify_datagram(datagram);
    debug!(
        ?class,
        datagram_len = datagram.len(),
        first_byte = datagram.first().copied().unwrap_or(0),
        %peer,
        "h3-native classified inbound datagram"
    );
    match class {
        DatagramClassification::Existing { handle, .. } => {
            if let Some(entry) = connections.get_mut(&handle.0)
                && let Err(err) = entry.connection.handle_datagram(now, datagram)
            {
                if close_for_datagram_error(&mut entry.connection, &err) {
                    warn!(
                        ?err,
                        handle = handle.0,
                        "h3-native handle_datagram (existing) failed; closing connection"
                    );
                } else {
                    debug!(
                        ?err,
                        handle = handle.0,
                        "h3-native handle_datagram (existing) failed; silently dropping packet per RFC 9000 §10.3"
                    );
                }
            }
        }
        DatagramClassification::NewInitial { dcid, scid, .. } => {
            let local_scid = generate_local_scid(dcid);
            // Server transport parameters MUST include the
            // original_destination_connection_id (client's first DCID)
            // and initial_source_connection_id (server's local SCID)
            // per RFC 9000 §18.2 — clients reject otherwise.
            let server_tp = encode_server_transport_parameters(dcid, &local_scid);
            let _ = local_tp;
            let connection = match Connection::<RustlsServerProvider>::new_server_with_limits(
                RustlsConfig::Server {
                    config: server_config.clone(),
                },
                &server_tp,
                dcid,
                scid,
                &local_scid,
                now,
                handshake_limits,
            ) {
                Ok(c) => c,
                Err(err) => {
                    warn!(?err, "h3-native new_server failed");
                    return;
                }
            };
            let handle = ConnectionHandle(*next_handle);
            *next_handle = next_handle.saturating_add(1);
            if demux.register(&local_scid, handle).is_err() {
                warn!("h3-native demux full");
                return;
            }
            // Route the client's ORIGINAL DCID to this same connection. Until
            // the client receives our SCID it addresses every Initial — the
            // CRYPTO-fragmented ClientHello continuations AND retransmits — to
            // its own chosen DCID. Without this, fragment 2 classifies as a
            // fresh Initial and spawns a phantom half-ClientHello connection
            // that never completes the handshake. Stored so reap frees it too.
            // the client's DCID is 0..=20 bytes (RFC 9000 §17.2); quinn uses
            // the full 20. store the ACTUAL length — narrowing to [u8; 8] here
            // silently dropped every non-8-byte client's ODCID, so its Initials
            // never routed and the handshake stalled forever.
            let original_dcid = client_dcid_for_demux(dcid);
            if let Some(ref odcid) = original_dcid {
                let _ = demux.register(odcid, handle);
            }
            let mut entry = ConnEntry {
                connection: Box::new(connection),
                peer,
                local_scid: Some(local_scid),
                original_dcid,
                driver: PerConnection::new(part_source_mode),
            };
            if let Err(err) = entry.connection.handle_datagram(now, datagram) {
                if close_for_datagram_error(&mut entry.connection, &err) {
                    warn!(
                        ?err,
                        handle = handle.0,
                        "h3-native first handle_datagram failed; closing connection"
                    );
                } else {
                    debug!(
                        ?err,
                        handle = handle.0,
                        "h3-native first handle_datagram failed; silently dropping packet per RFC 9000 §10.3"
                    );
                }
            }
            connections.insert(handle.0, entry);
            debug!(handle = handle.0, %peer, "h3-native accepted");
        }
        DatagramClassification::UnsupportedVersion {
            dcid,
            scid,
            peer_version,
        } => {
            // RFC 9000 §6 / §17.2.1 — the peer offered a version we don't
            // speak (commonly a GREASE probe, e.g. 0x?a?a?a?a). Reply with a
            // Version Negotiation packet listing v1 so it retries. The CIDs
            // are echoed SWAPPED (VN.dcid = peer's scid, VN.scid = peer's
            // dcid). Without this, version-probing clients (cloudflare quiche)
            // never learn we speak v1 and stall. Queued; the recv loop sends.
            debug!(
                peer_version,
                "h3-native unsupported version; replying with Version Negotiation"
            );
            let mut buf = [0u8; 64];
            match build_version_negotiation(dcid, scid, &mut buf) {
                Some(written) => pending_vn.push((buf[..written].to_vec(), peer)),
                None => warn!("h3-native version-negotiation encode failed"),
            }
        }
        DatagramClassification::Drop {
            reason: DropReason::MalformedHeader,
        } => {
            debug!(%peer, "h3-native malformed header; dropping per RFC 9000 §10.3");
        }
        DatagramClassification::Drop { reason } => {
            debug!(?reason, %peer, "h3-native classified drop");
        }
        _ => {}
    }
}

/// Build a Version Negotiation packet (RFC 9000 §17.2.1) replying to a
/// peer that offered an unsupported version. The CIDs are echoed
/// SWAPPED — the VN's DCID is the peer's SCID and the VN's SCID is the
/// peer's DCID — and the supported-versions list offers QUIC v1. Returns
/// the written length, or `None` if encoding failed.
fn build_version_negotiation(peer_dcid: &[u8], peer_scid: &[u8], out: &mut [u8]) -> Option<usize> {
    proxima_protocols::quic::packet::header::Header::VersionNegotiation {
        dcid: peer_scid,
        scid: peer_dcid,
        supported_versions_raw: &QUIC_V1_VERSION,
    }
    .encode(out)
    .ok()
}

/// Generate a per-connection 8-byte server SCID. RFC 9000 §5.3
/// requires this be **unpredictable to anyone other than the
/// generating endpoint** — a deterministic derivation from the peer's
/// DCID (which travels in the clear) lets any on-path observer
/// pre-compute it, enabling targeted spoofing and blocking future
/// hardening (CID rotation, retry-token integrity). `SysRng` reads
/// straight from the OS entropy source; on an OS-RNG failure (which
/// would also break TLS entirely) we fall back to a thread RNG so
/// the listener doesn't have a unique panic surface, but log loud.
fn generate_local_scid(_dcid: &[u8]) -> [u8; 8] {
    use rand::{RngExt, TryRng};
    let mut out = [0u8; 8];
    if let Err(err) = rand::rngs::SysRng.try_fill_bytes(&mut out) {
        // Fallback: ThreadRng (chacha-seeded from OS entropy at thread
        // init) keeps the listener up if SysRng momentarily fails; it
        // still satisfies "unpredictable to anyone other than this
        // endpoint" per RFC 9000 §5.3.
        warn!(?err, "h3-native SysRng failed; falling back to thread RNG");
        rand::rng().fill(&mut out[..]);
    }
    out
}

/// Store the CLIENT's chosen original DCID for demux routing. The client
/// picks its own length (0..=20 bytes per RFC 9000 §17.2 — quinn uses the
/// full 20), so this preserves the ACTUAL length: an earlier
/// `Option<[u8; 8]>` narrowing via `try_into` silently returned `None` for
/// any non-8-byte DCID, so the demux never learned it and the client's
/// fragmented/retransmitted Initials couldn't route home. `None` for an
/// over-length CID matches the demux's own `register` reject.
fn client_dcid_for_demux(dcid: &[u8]) -> Option<ConnectionIdBytes> {
    let mut cid = ConnectionIdBytes::new();
    cid.try_extend_from_slice(dcid).ok().map(|()| cid)
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

fn encode_default_transport_parameters() -> Vec<u8> {
    encode_server_transport_parameters(&[], &[])
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

/// The loop's monotonic `now` as micros ELAPSED since `origin_nanos`, derived
/// from the single injected [`Clock`] so it and the timer sleep can never
/// diverge (the two-clock defect the generic-clock threading closes).
/// Elapsed-since-origin (not the clock's absolute reading) keeps the
/// [`ProtoInstant`] serve-start-relative, exactly as the old `std::time::Instant`
/// origin did — small, identical values whether the clock is production
/// (`TimeClock`, large absolute) or a mock (starts at 0). `now_nanos` is
/// truncated to microseconds, [`ProtoInstant`]'s resolution and the unit every
/// QUIC deadline is in.
fn proto_now<Clk: Clock>(clock: &Clk, origin_nanos: u64) -> ProtoInstant {
    ProtoInstant::from_micros(clock.now_nanos().saturating_sub(origin_nanos) / 1_000)
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

    // The regression guard for the CID-truncation bug: quinn (and RFC 9000
    // §17.2's max) uses a 20-byte client DCID. The old `Option<[u8; 8]>`
    // narrowing dropped it to `None`, so its Initials never registered and the
    // handshake stalled for 30s. This MUST round-trip the full 20 bytes;
    // reverting to `[u8; 8]` fails here.
    #[test]
    fn client_dcid_preserves_the_full_20_byte_quinn_length() {
        let dcid20 = [0xABu8; 20];
        let stored = client_dcid_for_demux(&dcid20).expect("20-byte client DCID must be kept");
        assert_eq!(&stored[..], &dcid20[..]);
    }

    #[test]
    fn client_dcid_keeps_the_common_8_byte_length() {
        let dcid8 = [0x11u8; 8];
        let stored = client_dcid_for_demux(&dcid8).expect("8-byte client DCID kept");
        assert_eq!(&stored[..], &dcid8[..]);
    }

    #[test]
    fn client_dcid_keeps_a_zero_length_cid() {
        // A client MAY use a zero-length CID (RFC 9000 §5.1).
        assert_eq!(client_dcid_for_demux(&[]).map(|cid| cid.len()), Some(0));
    }

    #[test]
    fn client_dcid_rejects_over_max_length() {
        // > 20 bytes is illegal (RFC 9000 §17.2) — match the demux's own reject.
        let too_long = [0u8; 21];
        assert!(client_dcid_for_demux(&too_long).is_none());
    }

    #[test]
    fn version_negotiation_echoes_swapped_cids_and_offers_v1() {
        let peer_dcid = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let peer_scid = [10u8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        let mut buf = [0u8; 64];
        let written = build_version_negotiation(&peer_dcid, &peer_scid, &mut buf)
            .expect("version negotiation encodes");
        let parsed = proxima_protocols::quic::packet::header::parse_long(&buf[..written])
            .expect("parses as a long header");
        match parsed {
            proxima_protocols::quic::packet::header::Header::VersionNegotiation {
                dcid,
                scid,
                supported_versions_raw,
            } => {
                assert_eq!(dcid, &peer_scid, "VN DCID echoes the peer's SCID (swapped)");
                assert_eq!(scid, &peer_dcid, "VN SCID echoes the peer's DCID (swapped)");
                assert_eq!(supported_versions_raw, &[0, 0, 0, 1], "offers QUIC v1");
            }
            _ => panic!("expected a VersionNegotiation packet"),
        }
    }
}
