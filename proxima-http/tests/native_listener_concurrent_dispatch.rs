//! Regression guard for the `Listener<RustlsServerProvider>` composition
//! fold: proves the serve loop drives MULTIPLE connections' H3 layers
//! independently (the "targeted, not a full-table scan" driving the fold
//! introduced) AND that a slow/pending dispatch on one connection never
//! blocks `recv` for another — the concurrency the module's `in_flight`
//! `FuturesUnordered` exists to preserve.
//!
//! Two independent native QUIC clients complete their handshakes and open
//! GET requests against the SAME server loop. The dispatch pipe blocks
//! BOTH requests on a shared gate the test controls directly (no real
//! executor, no wall-clock wait) — this proves the driver staged BOTH
//! dispatches (recv for client B was never starved by client A's
//! still-pending in-flight future) before either is allowed to complete.
//! Releasing the gate then lets both finish; both must see 200.

#![cfg(feature = "http3-native")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::type_complexity)]

use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::task::noop_waker;

use proxima_core::time::drivers::mock::MockDriver;
use proxima_core::time::{Driver, Instant};
use proxima_http::http3::native::{DriverState, drive_client_step};
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
use proxima_primitives::stream::{DatagramFactory, DatagramSocket};
use proxima_protocols::http3_codec::client::{ClientConnection, H3ClientEvent};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::quic::connection::{Connection, ConnectionState};
use proxima_protocols::quic::time::Instant as ProtoInstant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsClientProvider, RustlsConfig};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig as RustlsClientConfig, DigitallySignedStruct, SignatureScheme};

// ── injected mock clock over MockDriver (mirrors native_listener_stale_now_reap.rs) ──

#[derive(Clone)]
struct MockClock {
    driver: Arc<MockDriver>,
}

impl MockClock {
    fn new() -> Self {
        Self {
            driver: Arc::new(MockDriver::new()),
        }
    }

    fn advance(&self, delta: Duration) {
        self.driver.advance(delta);
    }
}

struct MockSleep {
    driver: Arc<MockDriver>,
    deadline: Instant,
}

impl Future for MockSleep {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.driver.now() >= self.deadline {
            Poll::Ready(())
        } else {
            self.driver
                .schedule_wake(self.deadline, context.waker().clone());
            Poll::Pending
        }
    }
}

impl Clock for MockClock {
    type Delay = MockSleep;

    fn now_nanos(&self) -> u64 {
        u64::try_from(self.driver.now().into_monotonic().as_nanos()).unwrap_or(u64::MAX)
    }

    fn delay(&self, duration: Duration) -> MockSleep {
        let deadline = self.driver.now() + duration;
        MockSleep {
            driver: self.driver.clone(),
            deadline,
        }
    }
}

// ── in-memory datagram socket, shared by both clients (distinct peer addrs) ──

#[derive(Default)]
struct SocketState {
    inbound: VecDeque<(Vec<u8>, SocketAddr)>,
    sent: Vec<(Vec<u8>, SocketAddr)>,
    waker: Option<Waker>,
}

#[derive(Clone)]
struct SharedSocket {
    state: Arc<Mutex<SocketState>>,
    local: SocketAddr,
}

impl SharedSocket {
    fn new(local: SocketAddr) -> Self {
        Self {
            state: Arc::new(Mutex::new(SocketState::default())),
            local,
        }
    }

    fn inject(&self, datagrams: impl IntoIterator<Item = Vec<u8>>, from: SocketAddr) {
        let mut state = self.state.lock().expect("socket lock");
        for bytes in datagrams {
            state.inbound.push_back((bytes, from));
        }
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }

    /// Drain every datagram sent since the last call, keyed by destination
    /// peer so each client only reads what's addressed to it.
    fn take_sent_for(&self, peer: SocketAddr) -> Vec<Vec<u8>> {
        let mut state = self.state.lock().expect("socket lock");
        let (mine, other): (Vec<_>, Vec<_>) = state.sent.drain(..).partition(|(_, to)| *to == peer);
        state.sent = other;
        mine.into_iter().map(|(bytes, _)| bytes).collect()
    }
}

impl DatagramSocket for SharedSocket {
    fn poll_recv_from(
        &mut self,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>> {
        let mut state = self.state.lock().expect("socket lock");
        match state.inbound.pop_front() {
            Some((bytes, from)) => {
                let len = bytes.len().min(buf.len());
                buf[..len].copy_from_slice(&bytes[..len]);
                Poll::Ready(Ok((len, from)))
            }
            None => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    fn poll_send_to(
        &mut self,
        _cx: &mut Context<'_>,
        buf: &[u8],
        peer: SocketAddr,
    ) -> Poll<io::Result<usize>> {
        let mut state = self.state.lock().expect("socket lock");
        state.sent.push((buf.to_vec(), peer));
        Poll::Ready(Ok(buf.len()))
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }
}

struct SharedFactory {
    socket: SharedSocket,
}

impl DatagramFactory for SharedFactory {
    fn bind(&self, _addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
        Ok(Box::new(self.socket.clone()))
    }
}

// ── dispatch: blocks on a shared gate the test releases explicitly ───────────

#[derive(Clone, Default)]
struct SlowGate {
    ready: Arc<AtomicBool>,
    // TWO concurrent waiters (client A's and client B's dispatch futures)
    // register here — a single `Option<Waker>` slot would let the second
    // registration silently discard the first's waker, permanently
    // starving it (`FuturesUnordered` only re-polls a sub-future whose OWN
    // waker fired, so an overwritten waker means that future is never
    // polled again even once `ready` flips true).
    wakers: Arc<Mutex<Vec<Waker>>>,
}

impl SlowGate {
    /// Marks the gate ready and wakes EVERY registered waiter — event-
    /// driven, no busy-poll, mirrors the `DatagramProtocol::ready`
    /// wake-source seam's own test technique, generalized to N waiters.
    fn release(&self) {
        self.ready.store(true, Ordering::SeqCst);
        for waker in self.wakers.lock().expect("gate lock").drain(..) {
            waker.wake();
        }
    }

    fn wait(&self) -> impl Future<Output = ()> + Send + 'static {
        let gate = self.clone();
        std::future::poll_fn(move |cx| {
            if gate.ready.load(Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                gate.wakers.lock().expect("gate lock").push(cx.waker().clone());
                Poll::Pending
            }
        })
    }
}

struct GatedOk {
    gate: SlowGate,
    calls_started: Arc<AtomicUsize>,
}

impl SendPipe for GatedOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = proxima_core::ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, Self::Err>> + Send {
        let gate = self.gate.clone();
        let calls_started = self.calls_started.clone();
        async move {
            // Records that dispatch STARTED before blocking — the test
            // reads this while both requests are still parked on the gate
            // to prove the driver staged BOTH before either was released.
            calls_started.fetch_add(1, Ordering::SeqCst);
            gate.wait().await;
            Ok(Response::ok(Bytes::from_static(b"ok")))
        }
    }
}

// ── native client harness (mirrors native_listener_stale_now_reap.rs) ────────

#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn client_rustls_config() -> Arc<RustlsClientConfig> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let mut config = RustlsClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    Arc::new(config)
}

fn encode_client_tp(scid: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 512];
    let written = TransportParameters {
        initial_max_data: Some(1_048_576),
        max_idle_timeout_ms: Some(30_000),
        initial_max_stream_data_bidi_local: Some(65_536),
        initial_max_stream_data_bidi_remote: Some(65_536),
        initial_max_stream_data_uni: Some(65_536),
        initial_max_streams_bidi: Some(100),
        initial_max_streams_uni: Some(100),
        initial_source_connection_id: Some(scid),
        ..Default::default()
    }
    .encode(&mut buf)
    .expect("encode client tp");
    buf.truncate(written);
    buf
}

fn drain_datagrams(conn: &mut Connection<RustlsClientProvider>, now: ProtoInstant) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let mut buf = [0u8; 2048];
        match conn.poll_transmit(now, &mut buf) {
            Ok(Some(write)) => out.push(buf[..write.len].to_vec()),
            Ok(None) => break,
            Err(err) => panic!("client poll_transmit: {err:?}"),
        }
    }
    out
}

fn poll_serve(
    serve: &mut Pin<Box<dyn Future<Output = Result<(), proxima_core::ProximaError>> + Send + '_>>,
    cx: &mut Context<'_>,
) {
    match serve.as_mut().poll(cx) {
        Poll::Pending => {}
        Poll::Ready(other) => panic!("serve loop exited early: {other:?}"),
    }
}

/// One client's driving state — two independent instances race against
/// the SAME server loop.
struct ClientHarness {
    conn: Connection<RustlsClientProvider>,
    h3: ClientConnection,
    state: DriverState,
    peer: SocketAddr,
    now: ProtoInstant,
    saw_settings: bool,
    request_opened: bool,
    response_status: Option<u16>,
    response_body: Vec<u8>,
    saw_response_finished: bool,
}

impl ClientHarness {
    fn new(dcid: [u8; 8], scid: [u8; 8], peer: SocketAddr) -> Self {
        let now = ProtoInstant::from_micros(1_000_000);
        let client_tp = encode_client_tp(&scid);
        let conn = Connection::<RustlsClientProvider>::new_client(
            RustlsConfig::Client {
                config: client_rustls_config(),
                server_name: ServerName::try_from("localhost").expect("server name"),
            },
            &client_tp,
            &dcid,
            &scid,
            now,
        )
        .expect("client conn");
        Self {
            conn,
            h3: ClientConnection::new(Settings::default()),
            state: DriverState::new(),
            peer,
            now,
            saw_settings: false,
            request_opened: false,
            response_status: None,
            response_body: Vec::new(),
            saw_response_finished: false,
        }
    }

    /// One round: consume what the server sent this client, drive H3 if
    /// established, open the request once SETTINGS arrive, then produce
    /// this client's outbound burst.
    fn round(&mut self, socket: &SharedSocket) -> Vec<Vec<u8>> {
        for datagram in socket.take_sent_for(self.peer) {
            let _ = self.conn.handle_datagram(self.now, &datagram);
        }
        let _ = self.conn.handle_timeout(self.now);

        if matches!(self.conn.state(), ConnectionState::Established(_)) {
            let _ = drive_client_step(&mut self.conn, &mut self.h3, &mut self.state);
            while let Some(event) = self.h3.poll_event() {
                match event {
                    H3ClientEvent::SettingsEstablished { .. } => self.saw_settings = true,
                    H3ClientEvent::ResponseHeaders { status, .. } => self.response_status = status,
                    H3ClientEvent::ResponseData { bytes, .. } => self.response_body.extend(bytes),
                    H3ClientEvent::ResponseFinished { .. } => self.saw_response_finished = true,
                    _ => {}
                }
            }
            if self.saw_settings && !self.request_opened {
                let request_id = self
                    .h3
                    .open_request(&[
                        (b":method", b"GET"),
                        (b":scheme", b"https"),
                        (b":authority", b"localhost"),
                        (b":path", b"/"),
                    ])
                    .expect("open_request");
                self.h3.finish_request(request_id).expect("finish_request");
                self.request_opened = true;
            }
            let _ = drive_client_step(&mut self.conn, &mut self.h3, &mut self.state);
        }

        let out = drain_datagrams(&mut self.conn, self.now);
        self.now = ProtoInstant::from_micros(self.now.as_micros() + 25_000);
        out
    }
}

#[test]
fn two_connections_dispatch_concurrently_and_recv_is_not_blocked_by_a_pending_response() {
    // `Connection<RustlsClientProvider>`/`Connection<RustlsServerProvider>`
    // inline their crypto/loss/congestion state (const-generic caps, no
    // heap, per the sans-IO stack-over-heap discipline) — tens of KB each.
    // This test holds TWO client connections plus the server's own two
    // live connections at once; run on an explicitly larger stack rather
    // than depend on the harness default (mirrors
    // `proxima-quic/src/native/listener/tests.rs`'s `run_on_big_stack`).
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_test)
        .expect("spawn big-stack test thread")
        .join()
        .expect("test body panicked");
}

fn run_test() {
    let round_step = Duration::from_millis(5);
    let clock = MockClock::new();
    let bind: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4434);
    let peer_a: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 55_601);
    let peer_b: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 55_602);
    let socket = SharedSocket::new(bind);
    let factory = Arc::new(SharedFactory {
        socket: socket.clone(),
    });

    let gate = SlowGate::default();
    let calls_started = Arc::new(AtomicUsize::new(0));
    let dispatch = into_handle(GatedOk {
        gate: gate.clone(),
        calls_started: calls_started.clone(),
    });

    let protocol = proxima_http::http3::native::H3NativeListenProtocol::with_clock(clock.clone());
    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    });
    let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let mut serve = protocol.serve(bind, dispatch, &spec, context, shutdown_rx);
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    poll_serve(&mut serve, &mut cx);

    let mut client_a = ClientHarness::new([0xA0; 8], [0xA1; 8], peer_a);
    let mut client_b = ClientHarness::new([0xB0; 8], [0xB1; 8], peer_b);

    socket.inject(drain_datagrams(&mut client_a.conn, client_a.now), peer_a);
    socket.inject(drain_datagrams(&mut client_b.conn, client_b.now), peer_b);
    poll_serve(&mut serve, &mut cx);

    // Drive BOTH clients until each has opened its request — targeted
    // driving (`Listener::ingest_datagram`'s returned handle) must
    // independently progress two DIFFERENT connections' H3 layers, not
    // just one.
    for _ in 0..100 {
        if client_a.request_opened && client_b.request_opened {
            break;
        }
        let out_a = client_a.round(&socket);
        if !out_a.is_empty() {
            socket.inject(out_a, peer_a);
        }
        let out_b = client_b.round(&socket);
        if !out_b.is_empty() {
            socket.inject(out_b, peer_b);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut cx);
    }
    assert!(client_a.request_opened, "client A must reach a request open");
    assert!(client_b.request_opened, "client B must reach a request open");

    // Drive a few more rounds so both dispatches are pushed into
    // `in_flight` and started (blocked on the gate) — WITHOUT releasing
    // it yet.
    for _ in 0..20 {
        if calls_started.load(Ordering::SeqCst) >= 2 {
            break;
        }
        let out_a = client_a.round(&socket);
        if !out_a.is_empty() {
            socket.inject(out_a, peer_a);
        }
        let out_b = client_b.round(&socket);
        if !out_b.is_empty() {
            socket.inject(out_b, peer_b);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut cx);
    }
    assert_eq!(
        calls_started.load(Ordering::SeqCst),
        2,
        "both dispatches must have STARTED (and be parked on the gate) before either \
         is released — proves client B's request was not starved behind client A's \
         still-pending in-flight future"
    );
    assert!(!client_a.saw_response_finished, "A's response must still be gated");
    assert!(!client_b.saw_response_finished, "B's response must still be gated");

    // Release both dispatches — event-driven: `gate.release()` wakes the
    // stored waker directly, no polling loop needed to notice.
    gate.release();

    for _ in 0..100 {
        if client_a.saw_response_finished && client_b.saw_response_finished {
            break;
        }
        let out_a = client_a.round(&socket);
        if !out_a.is_empty() {
            socket.inject(out_a, peer_a);
        }
        let out_b = client_b.round(&socket);
        if !out_b.is_empty() {
            socket.inject(out_b, peer_b);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut cx);
    }

    assert!(client_a.saw_response_finished, "client A never saw its H3 response");
    assert!(client_b.saw_response_finished, "client B never saw its H3 response");
    assert_eq!(client_a.response_status, Some(200));
    assert_eq!(client_b.response_status, Some(200));
    assert_eq!(client_a.response_body, b"ok");
    assert_eq!(client_b.response_body, b"ok");
}

// ── park-not-spin: H3's OWN `handlers` race arm ───────────────────────────
//
// Relocated from `proxima-listen`'s now-deleted `DatagramProtocol::ready`
// generic-trait wake-source seam: the design was rejected because a
// per-connection wake source belongs in the layer that owns concurrent
// dispatch (H3), not on the QUIC-generic trait every connectionless
// protocol implements. `H3NativeListenProtocol::serve` already races its
// own `in_flight: FuturesUnordered` completion (the `handlers` arm)
// alongside recv/timer/shutdown; this test proves that arm still parks
// (zero wakes) while genuinely idle and wakes promptly the instant an
// in-flight dispatch completes — the same property the deleted
// `ready_source_parks_when_idle_and_wakes_promptly_when_the_gate_fires`
// test proved for the generic trait's now-removed default arm.

/// Counts every `wake`/`wake_by_ref` call — the oracle for "did the loop
/// park (zero calls while genuinely idle) or self-wake (a busy-poll bug)".
#[derive(Default)]
struct CountingWake {
    count: AtomicUsize,
}

impl std::task::Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn handlers_arm_parks_when_idle_and_wakes_promptly_when_an_in_flight_dispatch_completes() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_park_not_spin_test)
        .expect("spawn big-stack test thread")
        .join()
        .expect("test body panicked");
}

fn run_park_not_spin_test() {
    let round_step = Duration::from_millis(5);
    let clock = MockClock::new();
    let bind: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4435);
    let peer: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 55_701);
    let socket = SharedSocket::new(bind);
    let factory = Arc::new(SharedFactory {
        socket: socket.clone(),
    });

    let gate = SlowGate::default();
    let calls_started = Arc::new(AtomicUsize::new(0));
    let dispatch = into_handle(GatedOk {
        gate: gate.clone(),
        calls_started: calls_started.clone(),
    });

    let protocol = proxima_http::http3::native::H3NativeListenProtocol::with_clock(clock.clone());
    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    });
    let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let mut serve = protocol.serve(bind, dispatch, &spec, context, shutdown_rx);
    let noop = noop_waker();
    let mut noop_cx = Context::from_waker(&noop);

    poll_serve(&mut serve, &mut noop_cx);

    let mut client = ClientHarness::new([0xC0; 8], [0xC1; 8], peer);
    socket.inject(drain_datagrams(&mut client.conn, client.now), peer);
    poll_serve(&mut serve, &mut noop_cx);

    for _ in 0..100 {
        if client.request_opened {
            break;
        }
        let out = client.round(&socket);
        if !out.is_empty() {
            socket.inject(out, peer);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut noop_cx);
    }
    assert!(client.request_opened, "client must reach a request open");

    for _ in 0..20 {
        if calls_started.load(Ordering::SeqCst) >= 1 {
            break;
        }
        let out = client.round(&socket);
        if !out.is_empty() {
            socket.inject(out, peer);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut noop_cx);
    }
    assert_eq!(
        calls_started.load(Ordering::SeqCst),
        1,
        "the dispatch must have started (and be parked on the gate)"
    );
    assert!(!client.saw_response_finished, "the response must still be gated");

    // HEADLINE: with the gate not yet released, no new datagram pending,
    // and the clock not advanced, the loop is genuinely idle. A poll must
    // return `Pending` WITHOUT calling `wake` on its own context — the
    // busy-poll antipattern is a future that immediately reschedules
    // itself instead of registering a real waker and parking.
    let counting_wake = Arc::new(CountingWake::default());
    let waker = Waker::from(Arc::clone(&counting_wake));
    let mut counting_cx = Context::from_waker(&waker);
    poll_serve(&mut serve, &mut counting_cx);
    assert_eq!(
        counting_wake.count.load(Ordering::SeqCst),
        0,
        "a genuinely idle loop (no datagram, no elapsed deadline, no completed dispatch) \
         must park rather than call wake on itself — a nonzero count here is the \
         busy-poll bug this seam exists to avoid"
    );

    // Firing the gate from OUTSIDE the loop — exactly the shape a real
    // `FuturesUnordered` task's own waker produces on completion — must
    // wake the loop promptly.
    gate.release();
    assert!(
        counting_wake.count.load(Ordering::SeqCst) >= 1,
        "the in-flight dispatch completing must wake the `handlers` arm promptly"
    );

    // Drain to completion (any waker works from here on).
    for _ in 0..100 {
        if client.saw_response_finished {
            break;
        }
        let out = client.round(&socket);
        if !out.is_empty() {
            socket.inject(out, peer);
        }
        clock.advance(round_step);
        poll_serve(&mut serve, &mut noop_cx);
    }
    assert!(client.saw_response_finished, "client never saw its H3 response");
    assert_eq!(client.response_status, Some(200));
    assert_eq!(client.response_body, b"ok");
}
