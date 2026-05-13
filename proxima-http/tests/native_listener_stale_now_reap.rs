//! Deterministic regression guard for the native-h3 listener's
//! "stale-now premature-reap" bug.
//!
//! The serve() loop used to sample `now` BEFORE its recv/timer/handlers
//! `.await` (which parks arbitrarily long on an idle socket), then reused that
//! stale value to anchor a freshly-accepted connection's
//! `handshake_completion_deadline`. When the await parked longer than the
//! deadline, the next reap pass's fresh `now` was already past it, so the
//! connection was reaped microseconds after acceptance — its SCID unregistered,
//! the client's in-flight packets misrouted onto a phantom Initial-state
//! connection, tripping `ProtocolViolation { "non-Initial packet received in
//! Initial state" }`. The fix splits `now` into `tick_start` (pre-await, sizes
//! the timer only) and a fresh `now` re-sampled AFTER the await.
//!
//! This test drives the REAL serve() loop over an in-memory datagram socket and
//! an injected mock [`Clock`] (a directly-constructed `MockDriver`). It parks
//! the loop on an idle socket, advances VIRTUAL time past a deliberately-small
//! `handshake_completion_micros`, THEN delivers the client's Initial — the
//! exact "long idle await then accept" the bug needs. It then completes the H3
//! GET/200 exchange end-to-end THROUGH the serve loop. A reaped connection can
//! serve nothing, so the response only arrives when the freshly-accepted
//! connection survives — i.e. only with the tick_start/now split in place.
//!
//! There is NO real wall-clock waiting anywhere: every "delay" is a
//! `MockDriver::advance` of virtual time.

#![cfg(feature = "http3-native")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::type_complexity)]

use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
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
use proxima_protocols::quic::tls::TlsProvider;
use proxima_protocols::quic::tls::rustls_provider::{RustlsClientProvider, RustlsConfig};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig as RustlsClientConfig, DigitallySignedStruct, SignatureScheme};

// ── injected mock clock over MockDriver ──────────────────────────────────────
//
// The one clock the serve loop reads `now` from AND drives its timer sleep
// from. Virtual time advances ONLY when the test calls `advance`.

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

// ── in-memory datagram socket the serve loop binds and the test drives ───────

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

    /// Queue datagrams for the serve loop to receive, waking any parked recv.
    fn inject(&self, datagrams: impl IntoIterator<Item = Vec<u8>>, from: SocketAddr) {
        let mut state = self.state.lock().expect("socket lock");
        for bytes in datagrams {
            state.inbound.push_back((bytes, from));
        }
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }

    /// Drain every datagram the serve loop has sent since the last call.
    fn take_sent(&self) -> Vec<Vec<u8>> {
        let mut state = self.state.lock().expect("socket lock");
        state.sent.drain(..).map(|(bytes, _peer)| bytes).collect()
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

// ── dispatch: a 200 OK the surviving connection can serve ─────────────────────

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = proxima_core::ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, Self::Err>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(b"ok"))) }
    }
}

// ── native client harness (mirrors tests/native_round_trip.rs) ────────────────

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

/// Poll the serve future once, asserting it stays alive (parked), never exits.
fn poll_serve(
    serve: &mut Pin<Box<dyn Future<Output = Result<(), proxima_core::ProximaError>> + Send + '_>>,
    cx: &mut Context<'_>,
) {
    match serve.as_mut().poll(cx) {
        Poll::Pending => {}
        Poll::Ready(other) => panic!("serve loop exited early: {other:?}"),
    }
}

fn drain_datagrams<P: TlsProvider>(conn: &mut Connection<P>, now: ProtoInstant) -> Vec<Vec<u8>> {
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

/// The bug reproduction. A connection accepted AFTER the loop's await advanced
/// the injected clock past a small `handshake_completion` deadline must NOT be
/// reaped microseconds after acceptance — it must survive to complete its
/// handshake and serve the GET. On the buggy single-`now` code the server reaps
/// the connection on the very next pass, no response ever arrives, and this
/// test times out its bounded round loop and fails.
#[test]
fn accept_after_idle_await_survives_and_serves_not_reaped() {
    // A handshake-completion deadline far SMALLER than the idle gap below, so a
    // stale pre-await `now` guarantees the reap. 2s virtual; idle gap is 5s.
    let handshake_completion_micros: u64 = 2_000_000;
    let idle_gap = Duration::from_secs(5);
    let round_step = Duration::from_millis(25);

    let clock = MockClock::new();
    let bind: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4433);
    let client_peer: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 55_555);
    let socket = SharedSocket::new(bind);
    let factory = Arc::new(SharedFactory {
        socket: socket.clone(),
    });

    let protocol = proxima_http::http3::native::H3NativeListenProtocol::with_clock(clock.clone());
    let dispatch = into_handle(ConstantOk);
    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
        "handshake_completion_micros": handshake_completion_micros,
    });
    let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
    // Held for the whole test — dropping the sender signals shutdown.
    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let mut serve = protocol.serve(bind, dispatch, &spec, context, shutdown_rx);
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    // 1) Prime the loop: it binds the socket and parks on an empty recv (no
    //    connections yet, so no timer is armed).
    poll_serve(&mut serve, &mut cx);

    // 2) The "long idle await": advance virtual time PAST the handshake
    //    deadline while the loop is parked. No connection exists, so nothing
    //    wakes — exactly the SO_REUSEPORT-shard-sat-idle scenario.
    clock.advance(idle_gap);

    // 3) Build the native client and deliver its Initial NOW — after the gap.
    let mut client_now = ProtoInstant::from_micros(1_000_000);
    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let client_tp = encode_client_tp(&client_scid);
    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_rustls_config(),
            server_name: ServerName::try_from("localhost").expect("server name"),
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        client_now,
    )
    .expect("client conn");

    socket.inject(drain_datagrams(&mut client, client_now), client_peer);
    poll_serve(&mut serve, &mut cx);

    // 4) Complete the QUIC + H3 exchange THROUGH the serve loop. Every "delay"
    //    is a virtual-clock advance; there is no real sleeping.
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut client_state = DriverState::new();
    let mut saw_settings = false;
    let mut request_opened = false;
    let mut response_status: Option<u16> = None;
    let mut response_body: Vec<u8> = Vec::new();
    let mut saw_response_finished = false;

    for _ in 0..200 {
        // server → client
        for datagram in socket.take_sent() {
            let _ = client.handle_datagram(client_now, &datagram);
        }
        let _ = client.handle_timeout(client_now);

        if matches!(client.state(), ConnectionState::Established(_)) {
            let _ = drive_client_step(&mut client, &mut client_h3, &mut client_state);
            while let Some(event) = client_h3.poll_event() {
                match event {
                    H3ClientEvent::SettingsEstablished { .. } => saw_settings = true,
                    H3ClientEvent::ResponseHeaders { status, .. } => response_status = status,
                    H3ClientEvent::ResponseData { bytes, .. } => response_body.extend(bytes),
                    H3ClientEvent::ResponseFinished { .. } => saw_response_finished = true,
                    _ => {}
                }
            }
            if saw_settings && !request_opened {
                let request_id = client_h3
                    .open_request(&[
                        (b":method", b"GET"),
                        (b":scheme", b"https"),
                        (b":authority", b"localhost"),
                        (b":path", b"/"),
                    ])
                    .expect("open_request");
                client_h3.finish_request(request_id).expect("finish_request");
                request_opened = true;
            }
            let _ = drive_client_step(&mut client, &mut client_h3, &mut client_state);
        }

        if saw_response_finished && response_status.is_some() {
            break;
        }

        // client → server, then advance BOTH clocks a beat and drive serve.
        let client_out = drain_datagrams(&mut client, client_now);
        if !client_out.is_empty() {
            socket.inject(client_out, client_peer);
        }
        client_now = ProtoInstant::from_micros(client_now.as_micros() + 25_000);
        clock.advance(round_step);
        poll_serve(&mut serve, &mut cx);
    }

    assert!(
        saw_response_finished,
        "client never saw the H3 response — the connection accepted after the \
         idle await was reaped before it could serve (stale-now premature-reap)"
    );
    assert_eq!(
        response_status,
        Some(200),
        "surviving connection must serve 200; got {response_status:?}"
    );
    assert_eq!(response_body, b"ok", "response body mismatch");
}
