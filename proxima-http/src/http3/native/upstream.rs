//! Native HTTP/3 request/response upstream — the client-side mirror of
//! [`H3NativeListenProtocol`](super::listen::H3NativeListenProtocol),
//! packaged as a [`proxima_primitives::pipe`] [`SendPipe`] so it drops into the same
//! `proxima::Client` slot as [`H1ClientUpstream`] / [`H2ClientUpstream`].
//!
//! It composes the substrate the way the docs point: a native QUIC
//! [`Endpoint`](proxima_quic::native::Endpoint) (prime UDP socket +
//! sans-IO [`Connection`](proxima_protocols::quic::connection::Connection))
//! under the sans-IO H3 [`Client`](super::client::Client) state machine,
//! driven by [`drive_client_step`](super::driver::drive_client_step).
//! No quinn, no h3-quinn — the dual-surface native peer of the legacy
//! `Http3Upstream` (P7), selectable side-by-side per the C41 ruling.
//!
//! Sub-flag: `native-upstream` (default off; disciplined-component gate
//! 1 firewall until the compare-bench vs `Http3Upstream` seals C43).
//!
//! # Scope (v1)
//!
//! One QUIC connection per `call` — the same v1 shape
//! [`H2ClientUpstream`] documents ("connection-reuse + stream
//! multiplexing are a later layer"). Connection reuse + request
//! multiplexing land as a follow-on component so the bench against the
//! incumbent stays apples-to-apples (both per-call-connect).

use std::future::poll_fn;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant as StdInstant};

use bytes::{Bytes, BytesMut};
use prime::os::core_shard;
use rustls::pki_types::ServerName;
use tokio::sync::Mutex as AsyncMutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_quic::native::{ClientConfig as QuicClientConfig, Endpoint, EndpointConfig};
use proxima_protocols::quic::connection::{Connection, TimerOutcome};
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsClientProvider, RustlsConfig};
use proxima_telemetry::{debug, trace};

use super::client::{Client, ClientError};
use super::config::{ClientConfig as H3ClientConfig, DEFAULT_HANDSHAKE_TIMEOUT_MICROS};

/// Backstop on driver iterations for one request — each iteration parks
/// on a real inbound datagram (no busy spin), so this only trips if a
/// logic bug turns the loop into a no-progress Ready spin. A localhost
/// handshake + request/response settles in well under 100 iterations;
/// the executor / caller deadline is the real liveness guard.
const MAX_DRIVE_STEPS: usize = 100_000;

/// Connection-ID length QUIC endpoints mint locally (8 bytes is the
/// quinn/quiche default and within RFC 9000 §17.2's 0..=20 range).
const LOCAL_CID_LEN: usize = 8;

/// Outbound HTTP/3 upstream over the native sans-IO QUIC + H3 stack.
///
/// Holds the dial target + the rustls client config; each `call` dials a
/// fresh QUIC connection, runs the H3 request/response cycle, and tears
/// it down. Construct via [`Self::new`] (webpki roots) or
/// [`Self::with_client_config`] (custom roots / mTLS / dev self-signed).
pub struct H3NativeUpstream {
    server_addr: SocketAddr,
    /// SNI hostname + `:authority` value.
    server_name: Arc<str>,
    /// Local UDP bind — ephemeral port on loopback's family by default.
    bind: SocketAddr,
    h3_config: H3ClientConfig,
    quic_config: QuicClientConfig,
    rustls_config: Arc<rustls::ClientConfig>,
    /// Wall-clock cap on a single request (dial + handshake + exchange).
    /// Default 30 s, matching the QUIC idle-timeout default; without it a
    /// dead/silent peer would only fail on the QUIC idle timer.
    timeout: Duration,
    /// Persistent QUIC connection, lazily established on the first call and
    /// reused across calls (amortizing the handshake). `None` until the
    /// first call or after a connection error drops it. Serialized: each
    /// call holds the lock for its request, so reuse is request-at-a-time
    /// (concurrent stream multiplexing over the one connection is a
    /// further follow-on).
    conn: Arc<AsyncMutex<Option<Client<RustlsClientProvider>>>>,
    /// Opt-in: route response HEADERS through the lazy `PartSource`
    /// instead of the owned event — see [`Self::with_part_source`].
    #[cfg(feature = "http3-part-source")]
    part_source: bool,
}

impl H3NativeUpstream {
    /// Dial `server_addr` with SNI `server_name`, trusting the webpki
    /// root store with ALPN `h3`. No connection is opened here.
    ///
    /// A process-default rustls [`CryptoProvider`](rustls::crypto::CryptoProvider)
    /// must be installed (same contract as the legacy `Http3Upstream`).
    #[must_use]
    pub fn new(server_addr: SocketAddr, server_name: impl Into<String>) -> Self {
        Self::with_client_config(server_addr, server_name, default_rustls_client_config())
    }

    /// Like [`Self::new`] but with a caller-supplied rustls
    /// [`ClientConfig`](rustls::ClientConfig) — tests/benches install a
    /// self-signed cert as a trusted root here without baking a danger
    /// config into the production path.
    #[must_use]
    pub fn with_client_config(
        server_addr: SocketAddr,
        server_name: impl Into<String>,
        rustls_config: rustls::ClientConfig,
    ) -> Self {
        let server_name: Arc<str> = Arc::from(server_name.into());
        let bind = match server_addr {
            SocketAddr::V4(_) => SocketAddr::from(([0u8, 0, 0, 0], 0)),
            SocketAddr::V6(_) => SocketAddr::from(([0u16; 8], 0)),
        };
        Self {
            server_addr,
            server_name,
            bind,
            h3_config: H3ClientConfig::default(),
            quic_config: QuicClientConfig::default(),
            rustls_config: Arc::new(rustls_config),
            timeout: Duration::from_micros(DEFAULT_HANDSHAKE_TIMEOUT_MICROS),
            conn: Arc::new(AsyncMutex::new(None)),
            #[cfg(feature = "http3-part-source")]
            part_source: false,
        }
    }

    /// Opt every connection this upstream establishes into
    /// `ResponseHeaderMode::Source`: response HEADERS are stepped
    /// lazily off the wire block (0 heap allocations at the proto
    /// layer) instead of riding the owned event + a facade re-decode.
    /// Default off — flipping the production default is gated on the
    /// incumbent-matrix bench (`docs/proxima-pipe/discipline.md`).
    #[cfg(feature = "http3-part-source")]
    #[must_use]
    pub fn with_part_source(mut self) -> Self {
        self.part_source = true;
        self
    }

    /// Override the local UDP bind address (default: ephemeral on the
    /// server's address family).
    #[must_use]
    pub fn with_bind(mut self, bind: SocketAddr) -> Self {
        self.bind = bind;
        self
    }

    /// Override the per-request wall-clock timeout (default 30 s).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// webpki-roots trust anchors, TLS 1.3 only (QUIC mandates 1.3), ALPN
/// `h3`. Mirrors the legacy `Http3Upstream::default_client_config`.
fn default_rustls_client_config() -> rustls::ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut tls = rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .with_root_certificates(roots)
        .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    tls
}

/// Headers an h3 request MUST NOT carry — the HTTP/1 connection-specific
/// set (RFC 9114 §4.2) plus `host` (h3 uses `:authority`).
fn is_forbidden_h3_request_header(name: &[u8]) -> bool {
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

impl SendPipe for H3NativeUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl core::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let server_addr = self.server_addr;
        let server_name = self.server_name.clone();
        let bind = self.bind;
        let h3_config = self.h3_config.clone();
        let quic_config = self.quic_config.clone();
        let rustls_config = self.rustls_config.clone();
        let timeout = self.timeout;
        let conn = self.conn.clone();
        #[cfg(feature = "http3-part-source")]
        let part_source = self.part_source;
        async move {
            let header_bufs = request_header_bufs(&server_name, &request);
            let body = request.payload.clone();
            // Reuse the persistent connection; if a REUSED one fails
            // (likely idle-closed), drop it and re-establish once. A freshly
            // established connection that fails is a real error, not a
            // staleness retry.
            let mut last_err: Option<ProximaError> = None;
            for _ in 0..2 {
                let mut guard = conn.lock().await;
                let was_reused = guard.is_some();
                if guard.is_none() {
                    let mut client = connect(
                        server_addr,
                        &server_name,
                        bind,
                        h3_config.clone(),
                        &quic_config,
                        rustls_config.clone(),
                    )
                    .map_err(client_err)?;
                    establish_connection(&mut client, timeout).await?;
                    #[cfg(feature = "http3-part-source")]
                    if part_source {
                        client.h3_mut().enable_header_source_mode();
                    }
                    *guard = Some(client);
                }
                let Some(client) = guard.as_mut() else {
                    return Err(ProximaError::Upstream(
                        "h3-native: connection not established".into(),
                    ));
                };
                match drive_one_request(client, &header_bufs, &body, timeout).await {
                    Ok(response) => return Ok(response),
                    Err(err) => {
                        *guard = None;
                        if !was_reused {
                            return Err(err);
                        }
                        last_err = Some(err);
                    }
                }
            }
            Err(last_err
                .unwrap_or_else(|| ProximaError::Upstream("h3-native: request failed".into())))
        }
    }
}


fn client_err(err: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("h3-native upstream: {err}"))
}

/// Build a native H3 [`Client`] dialled at `server_addr`: derive the
/// local TPs from `quic_config`, mint random initial CIDs, construct the
/// rustls-backed client [`Connection`], bind the prime UDP
/// [`Endpoint`], and point it at the peer.
fn connect(
    server_addr: SocketAddr,
    server_name: &str,
    bind: SocketAddr,
    h3_config: H3ClientConfig,
    quic_config: &QuicClientConfig,
    rustls_config: Arc<rustls::ClientConfig>,
) -> Result<Client<RustlsClientProvider>, ClientError> {
    let server_name_owned =
        ServerName::try_from(server_name.to_string()).map_err(|_| ClientError::IllegalInState {
            state: "invalid-sni",
            method: "connect",
        })?;
    let dcid: [u8; LOCAL_CID_LEN] = rand::random();
    let scid: [u8; LOCAL_CID_LEN] = rand::random();
    // ISCID MUST equal the SCID we put in our Initial (RFC 9000 §7.3),
    // so the transport params can only be encoded once `scid` exists.
    let transport_params = quic_config
        .encode_transport_parameters_with_source_cid(&scid)
        .map_err(|_| ClientError::IllegalInState {
            state: "tp-encode",
            method: "connect",
        })?;
    let provider_config = RustlsConfig::Client {
        config: rustls_config,
        server_name: server_name_owned,
    };
    let connection = Connection::<RustlsClientProvider>::new_client(
        provider_config,
        &transport_params,
        &dcid,
        &scid,
        clock_now(),
    )
    .map_err(ClientError::Driver)?;
    let mut endpoint = Endpoint::new(
        EndpointConfig::client_only(bind, quic_config.clone()),
        connection,
    )?;
    endpoint.set_peer(server_addr);
    Ok(Client::new(h3_config, endpoint))
}

/// Pump the connection until the QUIC handshake + H3 SETTINGS exchange
/// complete (`peer_settings` lands). Tick loop mirroring the native
/// listener: drain outbound, then wait for the next inbound datagram OR a
/// 1 ms tick and pump the QUIC timers (`handle_timeout`) — without the
/// timer pump the client never flushes delayed ACKs / retransmits a lost
/// flight and the handshake stalls under server PTO.
async fn establish_connection(
    client: &mut Client<RustlsClientProvider>,
    timeout: Duration,
) -> Result<(), ProximaError> {
    let start = StdInstant::now();
    for iter in 0..MAX_DRIVE_STEPS {
        if start.elapsed() > timeout {
            debug!(?timeout, iter, "h3-native handshake timed out");
            return Err(ProximaError::Upstream(format!(
                "h3-native handshake timed out after {timeout:?}"
            )));
        }
        let mut sent = 0u64;
        loop {
            match poll_fn(|cx| client.poll_send(cx, clock_now())).await {
                Ok(true) => sent += 1,
                Ok(false) => break,
                Err(err) => {
                    debug!(?err, iter, "h3-native handshake poll_send error");
                    return Err(client_err(err));
                }
            }
        }
        let have_settings = client.peer_settings().is_some();
        trace!(
            iter,
            sent,
            have_settings,
            quic_state = client.quic_state_name(),
            elapsed_us = start.elapsed().as_micros() as u64,
            "h3-native handshake step"
        );
        if have_settings {
            debug!(iter, "h3-native handshake established");
            return Ok(());
        }
        if pump_tick(client, None).await? == TickOutcome::Closed {
            debug!(
                iter,
                elapsed_us = start.elapsed().as_micros() as u64,
                "h3-native connection closed during handshake"
            );
            return Err(ProximaError::Upstream(
                "h3-native: connection closed during handshake".into(),
            ));
        }
    }
    Err(ProximaError::Upstream(
        "h3-native: handshake did not complete".into(),
    ))
}

/// Open one request stream on an already-established connection and drive
/// it to its response. Reusable across calls on the persistent connection;
/// response events are filtered by this request's stream id, so a prior
/// completed request leaves nothing behind.
async fn drive_one_request(
    client: &mut Client<RustlsClientProvider>,
    header_bufs: &[(Vec<u8>, Vec<u8>)],
    body: &Bytes,
    timeout: Duration,
) -> Result<Response<Bytes>, ProximaError> {
    let header_refs: Vec<(&[u8], &[u8])> = header_bufs
        .iter()
        .map(|(name, value)| (name.as_slice(), value.as_slice()))
        .collect();
    let stream_id = client.open_request(&header_refs).map_err(client_err)?;
    if !body.is_empty() {
        client
            .send_request_data(stream_id, body)
            .map_err(client_err)?;
    }
    client.finish_request(stream_id).map_err(client_err)?;

    let mut status = None;
    let mut headers = HeaderList::new();
    let mut body_out = BytesMut::new();
    let mut finished = false;
    let start = StdInstant::now();
    for _ in 0..MAX_DRIVE_STEPS {
        if start.elapsed() > timeout {
            return Err(ProximaError::Upstream(format!(
                "h3-native request timed out after {timeout:?}"
            )));
        }
        while poll_fn(|cx| client.poll_send(cx, clock_now()))
            .await
            .map_err(client_err)?
        {}
        drain_events(
            client,
            Some(stream_id),
            &mut status,
            &mut headers,
            &mut body_out,
            &mut finished,
        )?;
        if finished {
            break;
        }
        if pump_tick(client, None).await? == TickOutcome::Closed {
            break;
        }
    }

    let status = status.ok_or_else(|| {
        ProximaError::Upstream("h3-native: connection ended before a response arrived".into())
    })?;
    let mut response = Response::new(status);
    response.metadata = headers;
    response.payload = body_out.freeze();
    Ok(response)
}

#[derive(PartialEq, Eq)]
enum TickOutcome {
    Continue,
    Closed,
}

/// One driver tick: wait for the next inbound datagram OR a 1 ms timer,
/// then pump the QUIC timers. The native listener's exact pattern.
async fn pump_tick(
    client: &mut Client<RustlsClientProvider>,
    park_until: Option<StdInstant>,
) -> Result<TickOutcome, ProximaError> {
    {
        // Park until the connection's REAL next deadline (PTO / idle / ack-
        // delay), not a blind 1 ms poll. recv wakes us immediately on any
        // datagram, so the timer only needs to fire for genuine QUIC timers.
        // A fixed 1 ms tick per connection is a wakeup storm — with N
        // request-at-a-time connections on one core it serializes them and
        // throughput *drops* as N rises. next_timeout collapses that to one
        // real deadline per connection (usually far out, or None when idle).
        // Computed before the recv closure borrows `client` mutably.
        let now = clock_now();
        let next_to = client.next_timeout();
        let mut deadline_tick = match next_to {
            Some(deadline) => {
                let delta_us = deadline.as_micros().saturating_sub(now.as_micros());
                core_shard::current_tick().saturating_add((delta_us / 1000).max(1))
            }
            None => core_shard::current_tick().saturating_add(100),
        };
        // never park past the caller's hard deadline (the bench run end): a
        // drained connection waiting on stream credit must wake at the deadline,
        // not on the idle/PTO timer.
        if let Some(until) = park_until {
            let rem_ms = until
                .saturating_duration_since(StdInstant::now())
                .as_millis() as u64;
            deadline_tick =
                deadline_tick.min(core_shard::current_tick().saturating_add(rem_ms.max(1)));
        }
        // Batch-drain on wake (recvmmsg), NOT a single poll_recv: a single recv
        // steals one datagram per park, starving the loop's poll_recv_batch and
        // forcing one park + one recvfrom per response datagram. Draining the
        // whole batch here is one park → many datagrams, the difference that
        // closes the gap to a batched C client.
        let recv = poll_fn(|cx| client.poll_recv_batch(cx, clock_now()));
        let tick = core_shard::timer_at(deadline_tick);
        futures::pin_mut!(recv, tick);
        let woke_recv = match futures::future::select(recv, tick).await {
            futures::future::Either::Left((result, _)) => {
                if let Err(err) = &result {
                    debug!(?err, "h3-native pump_tick poll_recv_batch error");
                }
                result.map_err(client_err)?;
                true
            }
            futures::future::Either::Right(_) => false,
        };
        trace!(woke_recv, ?next_to, "h3-native pump_tick woke");
    }
    let outcome = match client.handle_timeout(clock_now()) {
        Ok(outcome) => outcome,
        Err(err) => {
            debug!(?err, "h3-native pump_tick handle_timeout error");
            return Err(client_err(err));
        }
    };
    if matches!(outcome, TimerOutcome::Continue) {
        Ok(TickOutcome::Continue)
    } else {
        debug!(
            ?outcome,
            "h3-native pump_tick: handle_timeout closed the connection"
        );
        Ok(TickOutcome::Closed)
    }
}

/// Multiplexed h3 load driver: open ONE QUIC connection, keep `streams`
/// concurrent request streams in flight, and refill a stream the instant one
/// finishes — until `deadline`. Returns `(completed, errors)`.
///
/// This is the h3 analog of the multiplexed h2 client and the whole point of
/// h3/QUIC: stream concurrency over a single connection. The unary
/// request-at-a-time path (`drive_one_request`) measures round-trip LATENCY,
/// not server throughput — a single in-flight request idles the connection for
/// a full RTT between requests.
pub async fn bench_multiplexed(
    server_addr: SocketAddr,
    server_name: &str,
    rustls_config: Arc<rustls::ClientConfig>,
    streams: usize,
    deadline: StdInstant,
) -> (u64, u64) {
    use proxima_protocols::http3_codec::client::H3ClientEvent;

    let bind = match server_addr {
        SocketAddr::V4(_) => SocketAddr::from(([0u8, 0, 0, 0], 0)),
        SocketAddr::V6(_) => SocketAddr::from(([0u16; 8], 0)),
    };
    let h3_config = H3ClientConfig::default();
    let handshake_timeout = h3_config.handshake_timeout();
    let mut client = match connect(
        server_addr,
        server_name,
        bind,
        h3_config,
        &QuicClientConfig::default(),
        rustls_config,
    ) {
        Ok(client) => client,
        Err(err) => {
            debug!(?err, %server_addr, "h3-native bench connect failed");
            return (0, 1);
        }
    };
    if let Err(err) = establish_connection(&mut client, handshake_timeout).await {
        debug!(?err, %server_addr, "h3-native bench establish failed");
        return (0, 1);
    }
    debug!(%server_addr, "h3-native bench handshake established");

    let header_refs: [(&[u8], &[u8]); 4] = [
        (b":method", b"GET"),
        (b":scheme", b"https"),
        (b":authority", server_name.as_bytes()),
        (b":path", b"/"),
    ];
    let streams = streams.max(1);
    let mut inflight = 0usize;
    for _ in 0..streams {
        match client.open_request(&header_refs) {
            Ok(stream_id) => {
                let _ = client.finish_request(stream_id);
                inflight += 1;
            }
            Err(_) => break,
        }
    }

    let (mut completed, mut errors) = (0u64, 0u64);
    // prime the wire: flush the initial HEADERS batch once before the loop so
    // the loop body can keep its recv-before-flush ordering from pass one.
    if poll_fn(|cx| client.poll_send_batch(cx, clock_now()))
        .await
        .is_err()
    {
        return (completed, 1);
    }
    let pump_probe = std::env::var_os("REKT_PUMP").is_some();
    let (mut pit, mut psdg, mut pprk) = (0u64, 0u64, 0u64);
    while StdInstant::now() < deadline {
        pit += 1;
        // 1) drain response events produced by the prior recv pass; each
        //    finished stream is one completed GET.
        while let Some(event) = client.poll_event() {
            if matches!(event, H3ClientEvent::ResponseFinished { .. }) {
                completed += 1;
                inflight = inflight.saturating_sub(1);
            }
        }
        // 2) refill back up to `streams` in flight — queues fresh HEADERS for
        //    every stream the recv pass just completed.
        while inflight < streams && StdInstant::now() < deadline {
            match client.open_request(&header_refs) {
                Ok(stream_id) => {
                    let _ = client.finish_request(stream_id);
                    inflight += 1;
                }
                // stream credit exhausted this pass; retry after more close +
                // the peer's MAX_STREAMS replenishment lands.
                Err(_) => break,
            }
        }
        // 3) flush HEADERS + the ACK scheduled by the prior recv in one
        //    sendmmsg pass (the ACK coalesces with this pass's HEADERS).
        match poll_fn(|cx| client.poll_send_batch(cx, clock_now())).await {
            Ok(sent) => psdg += sent as u64,
            Err(_) => {
                errors += 1;
                return (completed, errors);
            }
        }
        // 4) ONE recv per pass: block on the fd read-waker until a response
        //    datagram lands (or the bench deadline). This replaces a
        //    speculative now_or_never recv that ran BEFORE the RTT elapsed and
        //    so EAGAIN'd every iteration (~0.25 wasted recvmmsg/req, a 368%-CPU
        //    spin that never actually slept). Parking here lets the other
        //    connections on this core run while this one waits — the event-loop
        //    shape a batched C client uses.
        pprk += 1;
        match pump_tick(&mut client, Some(deadline)).await {
            Ok(TickOutcome::Continue) => {}
            Ok(TickOutcome::Closed) | Err(_) => break,
        }
    }
    if pump_probe {
        let per = completed.max(1) as f64;
        eprintln!(
            "[PUMP] completed={completed} iters={pit} sent_dgram={psdg} parks={pprk} | iters/req={:.3} sent/req={:.3} park/req={:.3}",
            pit as f64 / per,
            psdg as f64 / per,
            pprk as f64 / per,
        );
    }
    (completed, errors)
}

/// `part-source`-gated sibling of [`bench_multiplexed`]: identical
/// connection bootstrap + stream-refill loop, but opts the client into
/// `ResponseHeaderMode::Source` and drains each response's HEADERS via
/// [`proxima_protocols::http3_codec::client::ClientConnection::poll_response_header_source`]
/// (a `PartSource`, 0 heap allocations to read `:status` + headers) instead
/// of the owned `H3ClientEvent::ResponseHeaders` event
/// [`bench_multiplexed`] consumes. Proves the design doc's step 3
/// (`docs/proxima-pipe/part-source-sink-design.md`), h3 client response
/// path, end-to-end against a real server — see
/// `docs/proxima-pipe/discipline.md` step 3.
///
/// A response whose `:status` is missing, unparsable, or outside 2xx counts
/// as an error — the same correctness bar a caller reading `status` off the
/// owned event would apply, now exercised through the 0-alloc source
/// instead of skipped.
#[cfg(feature = "http3-part-source")]
pub async fn bench_multiplexed_part_source(
    server_addr: SocketAddr,
    server_name: &str,
    rustls_config: Arc<rustls::ClientConfig>,
    streams: usize,
    deadline: StdInstant,
) -> (u64, u64) {
    use proxima_protocols::http3_codec::client::H3ClientEvent;
    use proxima_primitives::pipe::part::{Part, PartSource as _};

    let bind = match server_addr {
        SocketAddr::V4(_) => SocketAddr::from(([0u8, 0, 0, 0], 0)),
        SocketAddr::V6(_) => SocketAddr::from(([0u16; 8], 0)),
    };
    let h3_config = H3ClientConfig::default();
    let handshake_timeout = h3_config.handshake_timeout();
    let mut client = match connect(
        server_addr,
        server_name,
        bind,
        h3_config,
        &QuicClientConfig::default(),
        rustls_config,
    ) {
        Ok(client) => client,
        Err(err) => {
            debug!(?err, %server_addr, "h3-native bench(part-source) connect failed");
            return (0, 1);
        }
    };
    if let Err(err) = establish_connection(&mut client, handshake_timeout).await {
        debug!(?err, %server_addr, "h3-native bench(part-source) establish failed");
        return (0, 1);
    }
    client.h3_mut().enable_header_source_mode();
    debug!(%server_addr, "h3-native bench(part-source) handshake established");

    let header_refs: [(&[u8], &[u8]); 4] = [
        (b":method", b"GET"),
        (b":scheme", b"https"),
        (b":authority", server_name.as_bytes()),
        (b":path", b"/"),
    ];
    let streams = streams.max(1);
    let mut inflight = 0usize;
    for _ in 0..streams {
        match client.open_request(&header_refs) {
            Ok(stream_id) => {
                let _ = client.finish_request(stream_id);
                inflight += 1;
            }
            Err(_) => break,
        }
    }

    let (mut completed, mut errors) = (0u64, 0u64);
    if poll_fn(|cx| client.poll_send_batch(cx, clock_now()))
        .await
        .is_err()
    {
        return (completed, 1);
    }
    while StdInstant::now() < deadline {
        // Step every buffered response-header PartSource to exhaustion — 0
        // heap allocations. Reading `:status` here is the correctness proof
        // that the source path decodes the same field section the owned
        // event would have; a bad/missing status is counted as an error.
        while let Some((_stream_id, mut source)) = client.h3_mut().poll_response_header_source() {
            let mut status_ok = false;
            while let Some(part) = source.next() {
                if let Part::Header(name, value) = part
                    && name == b":status"
                {
                    status_ok = core::str::from_utf8(value)
                        .ok()
                        .and_then(|text| text.parse::<u16>().ok())
                        .is_some_and(|status| (200..300).contains(&status));
                }
            }
            // decode failures are deferred to stepping (lazy source) — a
            // reported error is a QPACK decompression failure and the
            // response cannot count as good regardless of what was seen.
            if source.error().is_some() {
                status_ok = false;
            }
            if !status_ok {
                errors += 1;
            }
        }
        while let Some(event) = client.poll_event() {
            if matches!(event, H3ClientEvent::ResponseFinished { .. }) {
                completed += 1;
                inflight = inflight.saturating_sub(1);
            }
        }
        while inflight < streams && StdInstant::now() < deadline {
            match client.open_request(&header_refs) {
                Ok(stream_id) => {
                    let _ = client.finish_request(stream_id);
                    inflight += 1;
                }
                Err(_) => break,
            }
        }
        if poll_fn(|cx| client.poll_send_batch(cx, clock_now()))
            .await
            .is_err()
        {
            errors += 1;
            return (completed, errors);
        }
        match pump_tick(&mut client, Some(deadline)).await {
            Ok(TickOutcome::Continue) => {}
            Ok(TickOutcome::Closed) | Err(_) => break,
        }
    }
    (completed, errors)
}

/// Drain pending H3 client events into the response accumulators.
///
/// `ResponseHeaders` carries `status` pre-extracted (Copy, 0 extra
/// allocation — see `proxima_protocols::http3_codec::client::H3ClientEvent`'s docs) and
/// `header_block` (the still-QPACK-encoded bytes, 1 allocation to have
/// crossed the event queue). This upstream forwards regular response
/// headers to the caller (a reverse-proxy correctness requirement), so it
/// decodes `header_block` itself via the 0-alloc borrowing
/// `qpack::decoder::decode_into` engine and writes straight into
/// `HeaderList`'s `Bytes` — one copy per forwarded header, not the
/// `DecodedField` intermediate the pre-redesign `decode_bounded` path
/// forced on every response regardless of whether headers were read past
/// `:status`.
/// # Errors
///
/// `Err` only from the `part-source` `Source` mode: a response header
/// section failed QPACK decode while being stepped (deferred
/// validation) — connection-fatal, the caller drops the connection. The
/// owned event path validates during `feed_response` and never errors
/// here.
fn drain_events(
    client: &mut Client<RustlsClientProvider>,
    request_id: Option<proxima_protocols::http3_codec::server::StreamId>,
    status: &mut Option<u16>,
    headers: &mut HeaderList,
    body_out: &mut BytesMut,
    finished: &mut bool,
) -> Result<(), ProximaError> {
    use proxima_protocols::http3_codec::client::H3ClientEvent;
    use proxima_protocols::http3_codec::qpack::decoder::{DecodeError, decode_into};
    use proxima_protocols::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN;
    #[cfg(feature = "http3-part-source")]
    drain_header_sources(client.h3_mut(), request_id, status, headers)?;
    while let Some(event) = client.poll_event() {
        match event {
            H3ClientEvent::ResponseHeaders {
                stream_id,
                status: response_status,
                header_block,
            } if Some(stream_id) == request_id => {
                *status = response_status;
                let mut scratch = [0u8; PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
                let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
                    if name.first() != Some(&b':') {
                        let _ = headers
                            .insert(Bytes::copy_from_slice(name), Bytes::copy_from_slice(value));
                    }
                    Ok(())
                };
                // the field section was already validated (cap-enforced)
                // when this event was produced; re-decoding here for
                // enumeration can't fail on a well-formed peer, so a
                // decode error just means "no extra headers surfaced" —
                // status is already captured above.
                let _ = decode_into(&header_block, u64::MAX, &mut scratch, &mut sink);
            }
            H3ClientEvent::ResponseData { stream_id, bytes } if Some(stream_id) == request_id => {
                body_out.extend_from_slice(&bytes);
            }
            H3ClientEvent::ResponseFinished { stream_id } if Some(stream_id) == request_id => {
                *finished = true;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Source-mode drain — the full forward path (status + every non-pseudo
/// header) stepped straight off the queued wire block into the caller's
/// `HeaderList`: no owned event, no re-decode. Only populated when the
/// connection opted in via [`H3NativeUpstream::with_part_source`].
/// Sources for other streams are drained and discarded (same shape as
/// the owned event loop's `_ => {}` arm — the upstream is
/// request-at-a-time per connection).
///
/// # Errors
///
/// A deferred QPACK decode failure — connection-fatal, the caller drops
/// the connection.
#[cfg(feature = "http3-part-source")]
fn drain_header_sources(
    h3: &mut proxima_protocols::http3_codec::client::ClientConnection,
    request_id: Option<proxima_protocols::http3_codec::server::StreamId>,
    status: &mut Option<u16>,
    headers: &mut HeaderList,
) -> Result<(), ProximaError> {
    use proxima_primitives::pipe::part::{Part, PartSource as _};
    while let Some((stream_id, mut source)) = h3.poll_response_header_source() {
        let for_this_request = Some(stream_id) == request_id;
        while let Some(part) = source.next() {
            if !for_this_request {
                continue;
            }
            if let Part::Header(name, value) = part {
                if name == b":status" {
                    *status = core::str::from_utf8(value)
                        .ok()
                        .and_then(|text| text.parse::<u16>().ok());
                } else if name.first() != Some(&b':') {
                    let _ =
                        headers.insert(Bytes::copy_from_slice(name), Bytes::copy_from_slice(value));
                }
            }
        }
        if let Some(err) = source.error() {
            return Err(ProximaError::Upstream(format!(
                "h3-native response header decode: {err:?}"
            )));
        }
    }
    Ok(())
}

/// Build the request pseudo-headers + forwardable regular headers as
/// owned byte pairs (the borrowed `&[u8]` view the proto wants is taken
/// at the call site, after these outlive it).
fn request_header_bufs(server_name: &str, request: &Request<Bytes>) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut bufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(8);
    bufs.push((b":method".to_vec(), request.method.as_bytes().to_vec()));
    bufs.push((b":scheme".to_vec(), b"https".to_vec()));
    bufs.push((b":authority".to_vec(), server_name.as_bytes().to_vec()));
    bufs.push((b":path".to_vec(), request.path.as_ref().to_vec()));
    for (name, value) in request.metadata.iter() {
        let lowered = name.as_ref().to_ascii_lowercase();
        if lowered.first() == Some(&b':') || is_forbidden_h3_request_header(&lowered) {
            continue;
        }
        bufs.push((lowered, value.as_ref().to_vec()));
    }
    bufs
}

/// Current proto [`Instant`] — origin + wall-clock micros since process
/// start. Monotonic enough for QUIC timers across one short request.
fn clock_now() -> Instant {
    use std::sync::OnceLock;
    use std::time::Instant as StdInstant;
    static START: OnceLock<StdInstant> = OnceLock::new();
    let start = START.get_or_init(StdInstant::now);
    let micros = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
    Instant::from_micros(1_000_000 + micros)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn new_upstream_does_not_dial_eagerly() {
        let addr: SocketAddr = "127.0.0.1:1".parse().expect("static addr");
        let upstream = H3NativeUpstream::new(addr, "never.invalid");
        assert_eq!(upstream.name(), "h3-native://never.invalid/");
        assert!(upstream.bind.ip().is_unspecified());
    }

    #[test]
    fn forbidden_h3_headers_are_filtered() {
        assert!(is_forbidden_h3_request_header(b"connection"));
        assert!(is_forbidden_h3_request_header(b"host"));
        assert!(!is_forbidden_h3_request_header(b"content-type"));
    }

    #[test]
    fn clock_now_is_monotonic_nondecreasing() {
        let first = clock_now();
        let second = clock_now();
        assert!(second.as_micros() >= first.as_micros());
    }

    /// Encode one HEADERS frame carrying `pairs` as a QPACK field section
    /// via the proto crate's canonical encoders — the same wire a real
    /// server produces.
    #[cfg(feature = "http3-part-source")]
    fn response_headers_frame(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut block = Vec::new();
        proxima_protocols::http3_codec::qpack::encoder::encode_refs(pairs.iter().copied(), &mut block)
            .expect("encode response header set");
        let mut out = vec![0u8; block.len() + 8];
        let written = proxima_protocols::http3_codec::frame::encode(
            &proxima_protocols::http3_codec::frame::H3Frame::Headers {
                header_block: &block,
            },
            &mut out,
        )
        .expect("encode HEADERS frame");
        out.truncate(written);
        out
    }

    /// The production forward path: `drain_header_sources` steps the lazy
    /// source into `:status` + EVERY non-pseudo header (proxying needs the
    /// full set, not just status) for the matching stream, and skips
    /// pseudo-headers.
    #[cfg(feature = "http3-part-source")]
    #[test]
    fn drain_header_sources_forwards_status_and_all_regular_headers() {
        use proxima_protocols::http3_codec::client::ClientConnection;
        use proxima_protocols::http3_codec::server::StreamId;
        use proxima_protocols::http3_codec::settings::Settings;

        let wire = response_headers_frame(&[
            (b":status", b"200"),
            (b"server", b"nginx/1.27.0"),
            (b"content-type", b"text/html"),
            (b"x-request-id", b"abc123"),
        ]);
        let mut h3 = ClientConnection::new(Settings::default());
        h3.enable_header_source_mode();
        h3.feed_response(StreamId(0), &wire, false)
            .expect("feed response headers");

        let mut status = None;
        let mut headers = HeaderList::new();
        drain_header_sources(&mut h3, Some(StreamId(0)), &mut status, &mut headers)
            .expect("well-formed section drains cleanly");

        assert_eq!(status, Some(200));
        assert_eq!(headers.get_str("server"), Some("nginx/1.27.0"));
        assert_eq!(headers.get_str("content-type"), Some("text/html"));
        assert_eq!(headers.get_str("x-request-id"), Some("abc123"));
        assert!(
            headers.get_str(":status").is_none(),
            "pseudo-headers must not leak into the forwarded set"
        );
    }

    /// Deferred-validation contract: a malformed section (non-zero
    /// Required Insert Count with no dynamic table wired) surfaces as an
    /// `Err` from the drain — connection-fatal — not a silent empty set.
    #[cfg(feature = "http3-part-source")]
    #[test]
    fn drain_header_sources_surfaces_decode_failure_as_error() {
        use proxima_protocols::http3_codec::client::ClientConnection;
        use proxima_protocols::http3_codec::server::StreamId;
        use proxima_protocols::http3_codec::settings::Settings;

        // RIC=1 prefix (one byte) — DynamicTableRequired at stepping.
        let mut out = vec![0u8; 16];
        let written = proxima_protocols::http3_codec::frame::encode(
            &proxima_protocols::http3_codec::frame::H3Frame::Headers {
                header_block: &[0x01, 0x00],
            },
            &mut out,
        )
        .expect("encode HEADERS frame");
        out.truncate(written);

        let mut h3 = ClientConnection::new(Settings::default());
        h3.enable_header_source_mode();
        h3.feed_response(StreamId(0), &out, false)
            .expect("feed defers validation — the malformed block queues");

        let mut status = None;
        let mut headers = HeaderList::new();
        let err = drain_header_sources(&mut h3, Some(StreamId(0)), &mut status, &mut headers)
            .expect_err("malformed section must fail the drain");
        assert!(matches!(err, ProximaError::Upstream(_)));
        assert_eq!(status, None);
    }
}
