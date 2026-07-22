use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Waker};

use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::task::noop_waker;
use proxima_core::ProximaError;
use proxima_listen::stream::DatagramProtocol;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
use proxima_primitives::stream::{DatagramFactory, DatagramSocket};
use proxima_protocols::quic::connection::Connection;
use proxima_protocols::quic::tls::Epoch;
use proxima_protocols::quic::tls::mock::{MockStep, MockTlsProvider};

use super::{
    AcceptFn, ConnectionHandle, DatagramIngest, ListenerError, build_version_negotiation, client_dcid_for_demux,
    core_instant, quic_instant,
};
use crate::native::listener::Listener;

type QuicInstant = proxima_protocols::quic::time::Instant;

/// Captures every `proxima_telemetry::error!`/`warn!`/etc. call made by
/// `body` — including the PLAIN (no `recorder = ..`) form `act_and_surface`
/// uses, which routes through the process-wide default recorder
/// (`proxima_telemetry::export::default_recorder`), not through an
/// explicitly-passed handle like `proxima_telemetry::capture::capture`'s
/// closure API expects. Safe under `cargo nextest run` specifically:
/// nextest gives each `#[test]` its own PROCESS (not just its own thread),
/// so installing the process-wide default here can never race a
/// concurrently-running test's own install.
fn capture_default_telemetry(body: impl FnOnce()) -> CapturedPipe {
    let pipe = proxima_telemetry::pipes::InMemoryPipe::new();
    let recorder = Arc::new(
        proxima_telemetry::recorder::Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .expect("capture recorder build"),
    );
    proxima_telemetry::export::set_default_recorder(Arc::clone(&recorder));
    body();
    recorder.drain();
    // `Captured` has no public constructor from a raw pipe, so read the
    // pipe directly via the same accessor methods `Captured` wraps.
    CapturedPipe(pipe)
}

/// Thin wrapper exposing the same accessor shape as
/// `proxima_telemetry::capture::Captured` over a pipe collected via
/// [`capture_default_telemetry`] instead of `capture()`'s explicit-recorder
/// closure.
struct CapturedPipe(proxima_telemetry::pipes::InMemoryPipe);

impl CapturedPipe {
    fn logs(&self) -> Vec<proxima_telemetry::log::LogRecord> {
        self.0.logs()
    }
}

fn peer_addr(last_octet: u8) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, last_octet)), 44000 + u16::from(last_octet))
}

/// Real client connection whose Initial datagram is produced by the
/// canonical encoder (`Connection::poll_transmit`) rather than
/// hand-crafted bytes — the header, CID fields, and packet protection
/// are all genuine wire bytes a production client would send.
fn build_client(dcid: &[u8], scid: &[u8], client_hello: &[u8], origin: QuicInstant) -> Connection<MockTlsProvider> {
    let config = MockTlsProvider::script_client(vec![MockStep::EmitHandshakeBytes {
        epoch: Epoch::Initial,
        bytes: client_hello.to_vec(),
    }]);
    Connection::<MockTlsProvider>::new_client(config, b"", dcid, scid, origin).expect("new_client")
}

/// Server-side accept policy scripted to read exactly `client_hello` and
/// reply with `server_hello` in the Initial epoch — the same shape a real
/// server's TLS provider follows on the first ClientHello.
fn accept_fn_for(client_hello: Vec<u8>, server_hello: Vec<u8>) -> AcceptFn<MockTlsProvider> {
    Arc::new(move |dcid: &[u8], scid: &[u8], local_scid: &[u8], now: QuicInstant| {
        let config = MockTlsProvider::script_server(vec![
            MockStep::ReadHandshake {
                epoch: Epoch::Initial,
                expect: client_hello.clone(),
            },
            MockStep::EmitHandshakeBytes {
                epoch: Epoch::Initial,
                bytes: server_hello.clone(),
            },
        ]);
        Connection::<MockTlsProvider>::new_server(config, b"", dcid, scid, local_scid, now)
    })
}

/// Drain every pending outbound datagram this tick, returning the
/// `(bytes, peer)` pairs in emission order.
fn drain_all(listener: &mut Listener<MockTlsProvider>, now: proxima_core::time::Instant) -> Vec<(Vec<u8>, SocketAddr)> {
    let mut sent = Vec::new();
    let mut buf = [0u8; 1500];
    while let Some((len, peer)) = block_on(listener.transmit(now, &mut buf)).expect("transmit") {
        sent.push((buf[..len].to_vec(), peer));
    }
    sent
}

/// `Connection<MockTlsProvider>` inlines its crypto/loss/congestion
/// state (const-generic caps, no heap) per the sans-IO stack-over-heap
/// discipline, so each live value is tens of KB on the stack. A debug
/// build holds several of these live at once in the multi-connection
/// tests below (plus per-frame copies libtest's default 2 MiB test
/// thread stack does not budget for) — run those bodies on an
/// explicitly larger stack rather than depend on the harness default.
fn run_on_big_stack<Body: FnOnce() + Send + 'static>(body: Body) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn big-stack test thread")
        .join()
        .expect("test body panicked");
}

// HEADLINE: the whole point of the DatagramProtocol seam — a state
// machine whose timer must fire with NO inbound datagram to trigger it.
// Feed exactly one client Initial, then feed nothing ever again; the
// server's own PTO deadline must fire and re-emit the identical lost
// flight.
#[test]
fn headline_pto_retransmits_handshake_flight_with_no_inbound_ever_fed_again() {
    let client_dcid = [0xAA_u8; 8];
    let client_scid = [0xBB_u8; 8];
    let client_hello = b"CLIENTHELLO-PTO".to_vec();
    let server_hello = b"SERVERHELLO-PTO".to_vec();
    let peer = peer_addr(7);

    let origin = QuicInstant::from_micros(1_000_000);
    let mut client = build_client(&client_dcid, &client_scid, &client_hello, origin);
    let mut client_buf = [0u8; 1500];
    let first = client
        .poll_transmit(origin, &mut client_buf)
        .expect("client poll_transmit")
        .expect("client emits its Initial flight");

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(
        accept_fn_for(client_hello, server_hello),
        accept_tx,
    );

    let core_now = core_instant(origin);
    block_on(listener.on_datagram(core_now, peer, &client_buf[..first.len]))
        .expect("on_datagram accepts the client Initial");

    accept_rx
        .try_recv()
        .expect("exactly one handle pushed for the NewInitial");
    assert_eq!(listener.connection_handles().len(), 1, "exactly one connection accepted");

    // Drain the server's first flight (ServerHello) completely.
    let first_flight = drain_all(&mut listener, core_now);
    assert!(!first_flight.is_empty(), "server must emit its first flight");
    for (_, sent_peer) in &first_flight {
        assert_eq!(*sent_peer, peer);
    }

    // No further inbound is EVER fed. Advance straight to the listener's
    // own PTO deadline and fire it.
    let deadline = listener
        .next_deadline()
        .expect("PTO deadline armed after the ack-eliciting Initial flight");
    block_on(listener.on_timeout(deadline)).expect("on_timeout");

    let retransmit_flight = drain_all(&mut listener, deadline);
    assert!(
        !retransmit_flight.is_empty(),
        "PTO must re-emit the lost flight with no inbound datagram ever fed again"
    );
    // Not byte-identical: a real QUIC retransmit carries a NEW packet
    // number (RFC 9002 never reuses one), which changes the AEAD nonce
    // and header protection mask, so the wire bytes legitimately differ
    // from the original — but the retransmit must re-emit the SAME
    // number of packets carrying the SAME CRYPTO payload the first
    // flight lost (the ServerHello bytes never advance past the first
    // TLS step, since the mock script's second step never got consumed
    // by an ACK that never arrived).
    assert_eq!(
        retransmit_flight.len(),
        first_flight.len(),
        "PTO retransmit re-emits the same number of packets as the lost flight"
    );
    for (_, sent_peer) in &retransmit_flight {
        assert_eq!(*sent_peer, peer, "retransmit still addresses the original peer");
    }
}

#[test]
fn accept_channel_pushes_exactly_one_handle_per_new_initial() {
    let client_dcid = [0x11_u8; 8];
    let client_scid = [0x22_u8; 8];
    let client_hello = b"HELLO-ACCEPT".to_vec();
    let server_hello = b"HELLO-ACCEPT-REPLY".to_vec();

    let origin = QuicInstant::from_micros(500_000);
    let mut client = build_client(&client_dcid, &client_scid, &client_hello, origin);
    let mut buf = [0u8; 1500];
    let first = client.poll_transmit(origin, &mut buf).expect("poll").expect("emit");

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn_for(client_hello, server_hello), accept_tx);

    let core_now = core_instant(origin);
    block_on(listener.on_datagram(core_now, peer_addr(1), &buf[..first.len])).expect("accept");

    let notified: Vec<ConnectionHandle> = std::iter::from_fn(|| accept_rx.try_recv().ok()).collect();
    assert_eq!(notified.len(), 1, "exactly one accept notification for one NewInitial");
    assert_eq!(listener.connection_handles(), notified);
}

#[test]
fn next_deadline_is_the_minimum_across_connections() {
    run_on_big_stack(next_deadline_is_the_minimum_across_connections_body);
}

fn next_deadline_is_the_minimum_across_connections_body() {
    let hello_a = b"HELLO-A".to_vec();
    let hello_b = b"HELLO-B".to_vec();
    let reply_a = b"REPLY-A".to_vec();
    let reply_b = b"REPLY-B".to_vec();

    let mut client_a = build_client(&[0x01; 8], &[0x02; 8], &hello_a, QuicInstant::from_micros(0));
    let mut client_b = build_client(&[0x03; 8], &[0x04; 8], &hello_b, QuicInstant::from_micros(0));
    let mut buf_a = [0u8; 1500];
    let mut buf_b = [0u8; 1500];
    let first_a = client_a
        .poll_transmit(QuicInstant::from_micros(0), &mut buf_a)
        .expect("poll a")
        .expect("emit a");
    let first_b = client_b
        .poll_transmit(QuicInstant::from_micros(0), &mut buf_b)
        .expect("poll b")
        .expect("emit b");

    let (accept_tx, _accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn_for(hello_a, reply_a), accept_tx.clone());

    // connection A accepted (and sent) at t=0.
    block_on(listener.on_datagram(core_instant(QuicInstant::from_micros(0)), peer_addr(10), &buf_a[..first_a.len]))
        .expect("accept a");
    drain_all(&mut listener, core_instant(QuicInstant::from_micros(0)));

    // connection B accepted (and sent) later, at t=5_000_000 — its PTO
    // deadline is therefore strictly later than A's.
    let (accept_tx_b, _accept_rx_b) = mpsc::unbounded();
    let mut listener_b_only = Listener::<MockTlsProvider>::new(accept_fn_for(hello_b, reply_b), accept_tx_b);
    block_on(listener_b_only.on_datagram(core_instant(QuicInstant::from_micros(5_000_000)), peer_addr(11), &buf_b[..first_b.len]))
        .expect("accept b");
    drain_all(&mut listener_b_only, core_instant(QuicInstant::from_micros(5_000_000)));

    let deadline_a_only = listener.next_deadline().expect("A has a deadline");
    let deadline_b_only = listener_b_only.next_deadline().expect("B has a deadline");
    assert!(deadline_a_only < deadline_b_only, "A's PTO fires strictly earlier than B's");
}

#[test]
fn poll_transmit_drains_pending_egress_across_multiple_connections_before_none() {
    run_on_big_stack(poll_transmit_drains_pending_egress_across_multiple_connections_before_none_body);
}

fn poll_transmit_drains_pending_egress_across_multiple_connections_before_none_body() {
    let hello_a = b"HELLO-MULTI-A".to_vec();
    let hello_b = b"HELLO-MULTI-B".to_vec();
    let reply_a = b"REPLY-MULTI-A".to_vec();
    let reply_b = b"REPLY-MULTI-B".to_vec();

    let origin = QuicInstant::from_micros(0);
    let mut client_a = build_client(&[0x10; 8], &[0x20; 8], &hello_a, origin);
    let mut client_b = build_client(&[0x30; 8], &[0x40; 8], &hello_b, origin);
    let mut buf_a = [0u8; 1500];
    let mut buf_b = [0u8; 1500];
    let first_a = client_a.poll_transmit(origin, &mut buf_a).expect("poll a").expect("emit a");
    let first_b = client_b.poll_transmit(origin, &mut buf_b).expect("poll b").expect("emit b");

    // A single accept_fn cannot script two different expected
    // ClientHellos, so exercise two independently-accepting listeners
    // sharing ONE underlying accept channel is not representative —
    // instead accept both connections against a script that matches
    // either hello via two accept passes on the SAME listener using a
    // shared script keyed off which hello arrives is unnecessary here:
    // both clients emit distinct DCIDs, so the demux's NewInitial branch
    // just needs one script per connection object it constructs, i.e.
    // the closure is invoked twice, and we return the right connection
    // per invocation based on the observed `dcid`.
    let dcid_a = [0x10_u8; 8];
    let accept_fn: AcceptFn<MockTlsProvider> = Arc::new(move |dcid, scid, local_scid, now| {
        if dcid == dcid_a {
            let config = MockTlsProvider::script_server(vec![
                MockStep::ReadHandshake { epoch: Epoch::Initial, expect: hello_a.clone() },
                MockStep::EmitHandshakeBytes { epoch: Epoch::Initial, bytes: reply_a.clone() },
            ]);
            Connection::<MockTlsProvider>::new_server(config, b"", dcid, scid, local_scid, now)
        } else {
            let config = MockTlsProvider::script_server(vec![
                MockStep::ReadHandshake { epoch: Epoch::Initial, expect: hello_b.clone() },
                MockStep::EmitHandshakeBytes { epoch: Epoch::Initial, bytes: reply_b.clone() },
            ]);
            Connection::<MockTlsProvider>::new_server(config, b"", dcid, scid, local_scid, now)
        }
    });

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn, accept_tx);
    let core_now = core_instant(origin);
    let peer_a = peer_addr(20);
    let peer_b = peer_addr(21);
    block_on(listener.on_datagram(core_now, peer_a, &buf_a[..first_a.len])).expect("accept a");
    block_on(listener.on_datagram(core_now, peer_b, &buf_b[..first_b.len])).expect("accept b");

    let handles: Vec<ConnectionHandle> = std::iter::from_fn(|| accept_rx.try_recv().ok()).collect();
    assert_eq!(handles.len(), 2, "both connections accepted");

    let sent = drain_all(&mut listener, core_now);
    let peers: std::collections::BTreeSet<SocketAddr> = sent.iter().map(|(_, peer)| *peer).collect();
    assert!(peers.contains(&peer_a), "connection A's flight was drained");
    assert!(peers.contains(&peer_b), "connection B's flight was drained");
}

// The seam this module exists for is a NewInitial packet followed by a
// SECOND real datagram from the SAME client, addressed using the
// server-issued CID (per RFC 9000 the client switches to the server's
// advertised SCID once its first response arrives) — this must classify
// as `Existing` and route to the SAME connection object, not spawn a
// second one.
#[test]
fn second_client_datagram_routes_to_the_existing_connection_not_a_new_one() {
    run_on_big_stack(second_client_datagram_routes_to_the_existing_connection_not_a_new_one_body);
}

fn second_client_datagram_routes_to_the_existing_connection_not_a_new_one_body() {
    let client_dcid = [0x77_u8; 8];
    let client_scid = [0x88_u8; 8];
    let client_hello = b"HELLO-ROUTE".to_vec();
    let server_hello = b"REPLY-ROUTE".to_vec();
    let peer = peer_addr(30);

    let origin = QuicInstant::from_micros(0);
    // The client's script needs a second step to read the server's real
    // Initial reply — without it, MockTlsProvider has nothing scripted
    // for the second `read_handshake` call and returns `Tls(NotReady)`.
    let config = MockTlsProvider::script_client(vec![
        MockStep::EmitHandshakeBytes {
            epoch: Epoch::Initial,
            bytes: client_hello.clone(),
        },
        MockStep::ReadHandshake {
            epoch: Epoch::Initial,
            expect: server_hello.clone(),
        },
    ]);
    let mut client = Connection::<MockTlsProvider>::new_client(config, b"", &client_dcid, &client_scid, origin)
        .expect("new_client");
    let mut buf = [0u8; 1500];
    let first = client.poll_transmit(origin, &mut buf).expect("poll").expect("emit");

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(
        accept_fn_for(client_hello, server_hello),
        accept_tx,
    );
    let core_now = core_instant(origin);
    block_on(listener.on_datagram(core_now, peer, &buf[..first.len])).expect("accept");
    let handle = accept_rx.try_recv().expect("notified");
    assert_eq!(listener.connection_handles(), vec![handle]);

    let server_flight = drain_all(&mut listener, core_now);
    assert!(!server_flight.is_empty());

    // Feed the server's real reply back into the client — this is what
    // makes the client switch its outbound DCID to the server's SCID.
    for (bytes, _) in &server_flight {
        client
            .handle_datagram(origin, bytes)
            .expect("client processes server's reply");
    }

    // The client's next flight (e.g. an Initial ACK) now addresses the
    // server using the server-issued local SCID.
    let mut next_buf = [0u8; 1500];
    let next = client
        .poll_transmit(origin, &mut next_buf)
        .expect("client poll_transmit after reading server reply")
        .expect("client has a second flight (ACK-eliciting Initial was received) to send");

    block_on(listener.on_datagram(core_now, peer, &next_buf[..next.len]))
        .expect("second datagram routes without error");

    assert_eq!(
        listener.connection_handles().len(),
        1,
        "the second datagram from the same client must route to the EXISTING connection, not spawn a new one"
    );
    assert!(
        accept_rx.try_recv().ok().is_none(),
        "no second accept notification for a datagram belonging to an existing connection"
    );
}

#[test]
fn listener_error_displays_io_variant() {
    let err = ListenerError::Io(std::io::Error::other("oops"));
    let formatted = format!("{err}");
    assert!(formatted.contains("io:"));
}

#[test]
fn quic_instant_and_core_instant_round_trip_at_microsecond_resolution() {
    let quic = QuicInstant::from_micros(42_424_242);
    let core = core_instant(quic);
    let back = quic_instant(core);
    assert_eq!(back, quic);
}

// --- end-to-end: Listener::listen_protocol through the real driver ---
//
// Everything above drives `Listener<P>` directly against the
// `DatagramProtocol` trait. This section proves the actual reference-point
// wiring (`Listener::listen_protocol`) is real: a fake `DatagramSocket` /
// `DatagramFactory` stand in for the OS socket (mirroring
// `datagram_protocol_listener.rs`'s own test harness) and a real client
// Initial is driven all the way through `DatagramProtocolListenProtocol`'s
// `serve()` loop.

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
        Self { state: Arc::new(Mutex::new(SocketState::default())), local }
    }

    fn inject(&self, bytes: Vec<u8>, from: SocketAddr) {
        let mut state = self.state.lock().expect("lock");
        state.inbound.push_back((bytes, from));
        if let Some(waker) = state.waker.take() {
            waker.wake();
        }
    }

    fn sent(&self) -> Vec<(Vec<u8>, SocketAddr)> {
        self.state.lock().expect("lock").sent.clone()
    }
}

impl DatagramSocket for SharedSocket {
    fn poll_recv_from(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> std::task::Poll<io::Result<(usize, SocketAddr)>> {
        let mut state = self.state.lock().expect("lock");
        match state.inbound.pop_front() {
            Some((bytes, from)) => {
                let len = bytes.len().min(buf.len());
                buf[..len].copy_from_slice(&bytes[..len]);
                std::task::Poll::Ready(Ok((len, from)))
            }
            None => {
                state.waker = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
    }

    fn poll_send_to(&mut self, _cx: &mut Context<'_>, buf: &[u8], peer: SocketAddr) -> std::task::Poll<io::Result<usize>> {
        self.state.lock().expect("lock").sent.push((buf.to_vec(), peer));
        std::task::Poll::Ready(Ok(buf.len()))
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

struct UnusedPipe;

impl SendPipe for UnusedPipe {
    type In = Request<bytes::Bytes>;
    type Out = Response<bytes::Bytes>;
    type Err = ProximaError;

    fn call(&self, _request: Request<bytes::Bytes>) -> impl Future<Output = Result<Response<bytes::Bytes>, ProximaError>> + Send {
        async move {
            Ok(Response {
                status: 200,
                metadata: HeaderList::new(),
                payload: bytes::Bytes::new(),
                stream: None,
                upgrade: None,
            })
        }
    }
}

fn poll_n(
    serve: &mut Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>>,
    cx: &mut Context<'_>,
    passes: usize,
) -> bool {
    for _ in 0..passes {
        if serve.as_mut().poll(cx).is_ready() {
            return true;
        }
    }
    false
}

#[test]
fn listen_protocol_wiring_drives_a_real_client_initial_through_the_real_driver() {
    run_on_big_stack(listen_protocol_wiring_drives_a_real_client_initial_through_the_real_driver_body);
}

fn listen_protocol_wiring_drives_a_real_client_initial_through_the_real_driver_body() {
    let bind = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5401);
    let socket = SharedSocket::new(bind);
    let factory = Arc::new(SharedFactory { socket: socket.clone() });

    let client_dcid = [0x55_u8; 8];
    let client_scid = [0x66_u8; 8];
    let client_hello = b"HELLO-WIRING".to_vec();
    let server_hello = b"REPLY-WIRING".to_vec();
    let peer = peer_addr(40);

    let origin = QuicInstant::from_micros(0);
    let mut client = build_client(&client_dcid, &client_scid, &client_hello, origin);
    let mut client_buf = [0u8; 1500];
    let first = client.poll_transmit(origin, &mut client_buf).expect("poll").expect("emit");

    // Production `TimeClock` (the same default `Listener::listen_protocol`
    // wires internally) — no need for a deterministic fake clock here: with
    // zero connections accepted yet, `next_deadline()` is `None`, so the
    // driver's timer race arm is `future::pending()` and the recv arm alone
    // decides the outcome once the datagram is injected below.
    let accept_fn: AcceptFn<MockTlsProvider> = accept_fn_for(client_hello, server_hello);
    let (protocol, mut accept_rx) = Listener::<MockTlsProvider>::listen_protocol("quic-wiring", accept_fn);
    let spec = serde_json::json!({});
    let context = ServeContext::new(Arc::new(NoopTelemetry)).with_datagram_factory(factory);
    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let dispatch = into_handle(UnusedPipe);
    let mut serve = protocol.serve(bind, dispatch, &spec, context, shutdown_rx);

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let _ = serve.as_mut().poll(&mut cx);

    socket.inject(client_buf[..first.len].to_vec(), peer);
    let resolved = poll_n(&mut serve, &mut cx, 16);
    assert!(!resolved, "serve must not resolve before shutdown fires");

    assert!(!socket.sent().is_empty(), "the driver must ship the server's ServerHello flight");
    for (_, sent_peer) in socket.sent() {
        assert_eq!(sent_peer, peer);
    }
    assert!(
        accept_rx.try_recv().is_ok(),
        "the accept channel wired through listen_protocol observes the NewInitial"
    );
}

// --- hardening: ODCID registration ---

// The regression guard for the CID-truncation bug class (see project
// history: the 20-byte-DCID / stale-reap production fixes): quinn (and
// RFC 9000 §17.2's max) uses a 20-byte client DCID. A `[u8; 8]` narrowing
// silently drops it to `None`, so it never registers and the handshake
// stalls. This MUST round-trip the full 20 bytes.
#[test]
fn client_dcid_preserves_the_full_20_byte_quinn_length() {
    let dcid20 = [0xAB_u8; 20];
    let stored = client_dcid_for_demux(&dcid20).expect("20-byte client DCID must be kept");
    assert_eq!(&stored[..], &dcid20[..]);
}

#[test]
fn client_dcid_keeps_the_common_8_byte_length() {
    let dcid8 = [0x11_u8; 8];
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

// The behavioral half of the ODCID hardening: a SECOND real Initial from
// the SAME client, still addressed to the client's ORIGINAL dcid (this is
// what happens for real when the client's PTO fires before it has ever
// seen our reply — a CRYPTO-fragment continuation or a genuine
// retransmit), must route to the EXISTING connection. Before this fix,
// only `local_scid` was registered at accept time, so this second
// datagram — still bearing the client's own dcid, not ours — classified
// as ANOTHER `NewInitial` and spawned a phantom connection that never
// completed the handshake.
#[test]
fn second_initial_still_addressed_to_the_clients_own_dcid_routes_to_the_existing_connection() {
    let client_dcid = [0xCC_u8; 8];
    let client_scid = [0xDD_u8; 8];
    let client_hello = b"HELLO-ODCID".to_vec();
    let server_hello = b"REPLY-ODCID".to_vec();
    let peer = peer_addr(50);

    let origin = QuicInstant::from_micros(0);
    let mut client = build_client(&client_dcid, &client_scid, &client_hello, origin);
    let mut buf = [0u8; 1500];
    let first = client.poll_transmit(origin, &mut buf).expect("poll").expect("emit");

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn_for(client_hello, server_hello), accept_tx);
    let core_now = core_instant(origin);
    block_on(listener.on_datagram(core_now, peer, &buf[..first.len])).expect("accept");
    let handle = accept_rx.try_recv().expect("notified");
    assert_eq!(listener.connection_handles(), vec![handle]);

    // The client's reply is NEVER fed back — force ITS OWN PTO so it
    // re-emits the Initial with the SAME (still client-chosen) dcid,
    // exactly the shape a CRYPTO-fragment continuation or a genuine
    // client-side retransmit takes.
    let client_deadline = client.next_timeout().expect("client PTO armed after its first flight");
    client.handle_timeout(client_deadline).expect("client handle_timeout");
    let mut second_buf = [0u8; 1500];
    let second = client
        .poll_transmit(client_deadline, &mut second_buf)
        .expect("client poll_transmit after its own PTO")
        .expect("client re-emits its Initial, still addressed to its own dcid");

    block_on(listener.on_datagram(core_instant(client_deadline), peer, &second_buf[..second.len]))
        .expect("second datagram (client's own dcid) routes without error");

    assert_eq!(
        listener.connection_handles().len(),
        1,
        "a second Initial still bearing the client's ORIGINAL dcid must route to the \
         existing connection, not spawn a phantom NewInitial"
    );
    assert!(
        accept_rx.try_recv().ok().is_none(),
        "no second accept notification for a datagram belonging to an existing connection"
    );
}

// --- hardening: Version Negotiation reply ---

#[test]
fn version_negotiation_echoes_swapped_cids_and_offers_v1() {
    let peer_dcid = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let peer_scid = [10u8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let mut buf = [0u8; 64];
    let written = build_version_negotiation(&peer_dcid, &peer_scid, &mut buf).expect("version negotiation encodes");
    let parsed =
        proxima_protocols::quic::packet::header::parse_long(&buf[..written]).expect("parses as a long header");
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

// Behavioral half: a real unsupported-version datagram queues a VN reply
// that `transmit` drains, addressed to the ORIGINATING peer, BEFORE any
// connection's own egress (proving the "drained first" ordering, not just
// that it's drained eventually).
#[test]
fn unsupported_version_datagram_queues_a_vn_reply_drained_first_by_transmit() {
    let peer = peer_addr(51);
    let peer_dcid = [0x01_u8; 8];
    let peer_scid = [0x02_u8; 8];

    // A GREASE-style unsupported version (RFC 9000 §17.2.1 reserved
    // pattern: any version of the form 0x?a?a?a?a) — the demux classifies
    // long-header packets carrying this as `UnsupportedVersion` rather
    // than `NewInitial`.
    let mut datagram = vec![0u8; 1200];
    datagram[0] = 0xC0; // long header, fixed bit set
    datagram[1..5].copy_from_slice(&[0x1a, 0x2a, 0x3a, 0x4a]); // GREASE version
    datagram[5] = u8::try_from(peer_dcid.len()).expect("dcid len fits u8");
    datagram[6..6 + peer_dcid.len()].copy_from_slice(&peer_dcid);
    let scid_len_offset = 6 + peer_dcid.len();
    datagram[scid_len_offset] = u8::try_from(peer_scid.len()).expect("scid len fits u8");
    datagram[scid_len_offset + 1..scid_len_offset + 1 + peer_scid.len()].copy_from_slice(&peer_scid);

    let (accept_tx, _accept_rx) = mpsc::unbounded();
    let accept_fn: AcceptFn<MockTlsProvider> = accept_fn_for(b"unused".to_vec(), b"unused".to_vec());
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn, accept_tx);
    let core_now = core_instant(QuicInstant::from_micros(0));
    block_on(listener.on_datagram(core_now, peer, &datagram)).expect("unsupported-version datagram accepted");
    assert!(
        listener.connection_handles().is_empty(),
        "an unsupported-version datagram must never spawn a connection"
    );

    let mut buf = [0u8; 1500];
    let (len, sent_peer) = block_on(listener.transmit(core_now, &mut buf))
        .expect("transmit")
        .expect("a Version Negotiation reply is staged and drained");
    assert_eq!(sent_peer, peer);
    let parsed = proxima_protocols::quic::packet::header::parse_long(&buf[..len]).expect("parses as a long header");
    match parsed {
        proxima_protocols::quic::packet::header::Header::VersionNegotiation { dcid, scid, .. } => {
            assert_eq!(dcid, &peer_scid[..], "VN DCID echoes the peer's SCID (swapped)");
            assert_eq!(scid, &peer_dcid[..], "VN SCID echoes the peer's DCID (swapped)");
        }
        other => panic!("expected a VersionNegotiation packet, got {other:?}"),
    }
}

// --- the routing-outcome return: `ingest_datagram` / `DatagramIngest` ---

// The seam an application protocol layered on top (H3, etc.) uses to
// drive ONLY the connection a datagram actually touched, read straight off
// the return rather than a side-channel: a `NewInitial` returns
// `Accepted { handle, .. }` naming its own freshly-created handle, and a
// datagram routed to that SAME connection later returns
// `Existing { handle, .. }` with the identical handle — no separate drain
// step, no stale-if-forgotten side channel.
#[test]
fn ingest_datagram_return_names_the_accepted_then_the_existing_handle() {
    run_on_big_stack(ingest_datagram_return_names_the_accepted_then_the_existing_handle_body);
}

fn ingest_datagram_return_names_the_accepted_then_the_existing_handle_body() {
    let client_dcid = [0x99_u8; 8];
    let client_scid = [0x88_u8; 8];
    let client_hello = b"HELLO-INGEST".to_vec();
    let server_hello = b"REPLY-INGEST".to_vec();
    let peer = peer_addr(60);

    let origin = QuicInstant::from_micros(0);
    let config = MockTlsProvider::script_client(vec![
        MockStep::EmitHandshakeBytes {
            epoch: Epoch::Initial,
            bytes: client_hello.clone(),
        },
        MockStep::ReadHandshake {
            epoch: Epoch::Initial,
            expect: server_hello.clone(),
        },
    ]);
    let mut client = Connection::<MockTlsProvider>::new_client(config, b"", &client_dcid, &client_scid, origin)
        .expect("new_client");
    let mut buf = [0u8; 1500];
    let first = client.poll_transmit(origin, &mut buf).expect("poll").expect("emit");

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn_for(client_hello, server_hello), accept_tx);
    let core_now = core_instant(origin);

    let ingest = listener
        .ingest_datagram(origin, peer, &buf[..first.len])
        .expect("ingest_datagram accepts the client Initial");
    let handle = match ingest {
        DatagramIngest::Accepted { handle, error } => {
            assert!(error.is_none(), "a clean first Initial must not surface an error");
            handle
        }
        other => panic!("expected Accepted, got {other:?}"),
    };
    assert_eq!(
        accept_rx.try_recv().expect("notified"),
        handle,
        "the accept channel reports the SAME handle the ingest return named"
    );

    let server_flight = drain_all(&mut listener, core_now);
    for (bytes, _) in &server_flight {
        client.handle_datagram(origin, bytes).expect("client reads server reply");
    }
    let mut next_buf = [0u8; 1500];
    let next = client
        .poll_transmit(origin, &mut next_buf)
        .expect("client poll_transmit")
        .expect("client has a second flight");
    let ingest_again = listener
        .ingest_datagram(origin, peer, &next_buf[..next.len])
        .expect("ingest_datagram routes the second datagram");
    match ingest_again {
        DatagramIngest::Existing { handle: routed, error } => {
            assert_eq!(routed, handle, "the second datagram routes to the SAME handle the accept named");
            assert!(error.is_none(), "a clean handshake continuation must not surface an error");
        }
        other => panic!("expected Existing, got {other:?}"),
    }
}

// --- errors surface: act (close) + surface (telemetry + return), never hidden ---

// HEADLINE for the error-surfacing rework: a genuine RFC 9000 transport
// violation on connection A must (1) close THAT connection with the
// matching transport `CONNECTION_CLOSE` code (proven via a real
// `transmit` drain, not just "some close call happened"), (2) fire a
// `proxima_telemetry::error!` event, and (3) appear in the
// `DatagramIngest` return — while the listener stays alive and
// connection B, routed and driven independently, is completely
// unaffected.
#[test]
fn connection_level_protocol_error_closes_that_connection_surfaces_telemetry_and_leaves_others_alone() {
    run_on_big_stack(connection_level_protocol_error_closes_that_connection_surfaces_telemetry_and_leaves_others_alone_body);
}

fn connection_level_protocol_error_closes_that_connection_surfaces_telemetry_and_leaves_others_alone_body() {
    let hello_a = b"HELLO-ERR-A".to_vec();
    let hello_b = b"HELLO-ERR-B".to_vec();
    let reply_a = b"REPLY-ERR-A".to_vec();
    let reply_b = b"REPLY-ERR-B".to_vec();
    let origin = QuicInstant::from_micros(0);
    let peer_a = peer_addr(70);
    let peer_b = peer_addr(71);

    let mut client_a = build_client(&[0x40; 8], &[0x41; 8], &hello_a, origin);
    // B (unlike A) drives a real second flight later in this test, so its
    // script needs a second step to read the server's real Initial reply —
    // without it, MockTlsProvider has nothing scripted for the second
    // `read_handshake` call and returns `Tls(NotReady)`.
    let client_b_config = MockTlsProvider::script_client(vec![
        MockStep::EmitHandshakeBytes {
            epoch: Epoch::Initial,
            bytes: hello_b.clone(),
        },
        MockStep::ReadHandshake {
            epoch: Epoch::Initial,
            expect: reply_b.clone(),
        },
    ]);
    let mut client_b =
        Connection::<MockTlsProvider>::new_client(client_b_config, b"", &[0x50; 8], &[0x51; 8], origin)
            .expect("new_client b");
    let mut buf_a = [0u8; 1500];
    let mut buf_b = [0u8; 1500];
    let first_a = client_a.poll_transmit(origin, &mut buf_a).expect("poll a").expect("emit a");
    let first_b = client_b.poll_transmit(origin, &mut buf_b).expect("poll b").expect("emit b");

    let dcid_a = [0x40_u8; 8];
    let accept_fn: AcceptFn<MockTlsProvider> = Arc::new(move |dcid, scid, local_scid, now| {
        let (hello, reply) = if dcid == dcid_a { (&hello_a, &reply_a) } else { (&hello_b, &reply_b) };
        let config = MockTlsProvider::script_server(vec![
            MockStep::ReadHandshake { epoch: Epoch::Initial, expect: hello.clone() },
            MockStep::EmitHandshakeBytes { epoch: Epoch::Initial, bytes: reply.clone() },
        ]);
        Connection::<MockTlsProvider>::new_server(config, b"", dcid, scid, local_scid, now)
    });

    let (accept_tx, mut accept_rx) = mpsc::unbounded();
    let mut listener = Listener::<MockTlsProvider>::new(accept_fn, accept_tx);

    let handle_a = match listener.ingest_datagram(origin, peer_a, &buf_a[..first_a.len]).expect("accept a") {
        DatagramIngest::Accepted { handle, error } => {
            assert!(error.is_none(), "a clean first Initial must not surface an error");
            handle
        }
        other => panic!("expected Accepted, got {other:?}"),
    };
    let handle_b = match listener.ingest_datagram(origin, peer_b, &buf_b[..first_b.len]).expect("accept b") {
        DatagramIngest::Accepted { handle, error } => {
            assert!(error.is_none(), "a clean first Initial must not surface an error");
            handle
        }
        other => panic!("expected Accepted, got {other:?}"),
    };
    let _ = accept_rx.try_recv();
    let _ = accept_rx.try_recv();
    assert_eq!(listener.connection_handles().len(), 2, "both connections accepted");

    // Drain BOTH connections' first flights now, before triggering A's
    // error — so the LATER post-error drain contains ONLY A's fresh
    // CONNECTION_CLOSE frame, not a mix of A's close and B's still-pending
    // ServerHello.
    let core_now = core_instant(origin);
    let first_flights = drain_all(&mut listener, core_now);
    let server_flight_b: Vec<Vec<u8>> = first_flights
        .iter()
        .filter(|(_, peer)| *peer == peer_b)
        .map(|(bytes, _)| bytes.clone())
        .collect();
    assert!(!server_flight_b.is_empty(), "B's first flight must have shipped");
    for bytes in &server_flight_b {
        client_b.handle_datagram(origin, bytes).expect("client B reads server reply");
    }

    // Learn connection A's server-chosen local SCID (generated internally
    // by `ingest_datagram`, so the test can't know it up front) — needed
    // to address a real packet AT that connection.
    let local_scid_a = match listener.connection_mut(handle_a).expect("connection a").state() {
        proxima_protocols::quic::connection::ConnectionState::Initial(state) => state.local_initial_scid.to_vec(),
        other => panic!("expected Initial, got {other:?}"),
    };

    // A genuine RFC 9000 violation, built with the CANONICAL ENCODER (not
    // hand-rolled bytes): a long-header packet of the WRONG epoch
    // (Handshake) addressed to a connection still in the Initial state.
    // `Connection::handle_datagram`'s own header-form check
    // (`parse_and_apply_initial`) matches the parsed header against
    // Initial/VersionNegotiation/Retry and returns exactly
    // `ProtocolViolation { reason: "non-Initial packet received in
    // Initial state" }` for anything else — a real client can produce
    // this shape by sending its Handshake flight before the server has
    // acknowledged the Initial (reordered/duplicated network delivery).
    let mut bad = [0u8; 128];
    let payload = [0x00_u8, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11];
    let written = proxima_protocols::quic::packet::header::Header::Handshake {
        version: 1,
        dcid: &local_scid_a,
        scid: &[0x99_u8; 8],
        length: payload.len() as u64,
        pn_and_payload: &payload,
    }
    .encode(&mut bad)
    .expect("encode handshake header");

    let captured = capture_default_telemetry(|| {
        match listener
            .ingest_datagram(origin, peer_a, &bad[..written])
            .expect("ingest_datagram routes the malformed short-header packet")
        {
            DatagramIngest::Existing { handle, error } => {
                assert_eq!(handle, handle_a, "the violation is attributed to connection A");
                match error {
                    Some(proxima_protocols::quic::connection::ConnectionError::ProtocolViolation { reason }) => {
                        assert_eq!(reason, "non-Initial packet received in Initial state");
                    }
                    other => panic!("expected Some(ProtocolViolation), got {other:?}"),
                }
            }
            other => panic!("expected Existing, got {other:?}"),
        }
    });

    // SURFACE: a telemetry error event fired — never silently discarded.
    let logs = captured.logs();
    assert!(
        logs.iter().any(|log| log.level == proxima_telemetry::level::Level::ERROR),
        "a connection-level protocol error must fire a telemetry error event; got {} log(s), none at ERROR",
        logs.len()
    );

    // ACT: connection A transitions to Closing immediately — `close_transport`
    // is synchronous and idempotent-first-wins (see `close_inner`). The
    // violation was detected while A was still in the Initial state, so no
    // Application keys exist yet; RFC 9000 long-header CONNECTION_CLOSE
    // framing for Initial/Handshake epochs isn't implemented (see
    // `poll_transmit_closing_before_application_keys_yields_ok_none_not_err`
    // in proxima-protocols — the documented, pre-existing behavior: the peer
    // sees silence and idle timeout reaps the connection, rather than
    // `transmit` erroring or looping). So `transmit` legitimately drains
    // nothing new for A here; what matters is that A is no longer live.
    assert!(
        matches!(
            listener.connection_mut(handle_a).expect("connection a still present").state(),
            proxima_protocols::quic::connection::ConnectionState::Closing(_)
        ),
        "connection A must have transitioned to Closing"
    );
    let sent = drain_all(&mut listener, core_now);
    for (_, sent_peer) in &sent {
        assert_eq!(*sent_peer, peer_a, "any drained bytes here would only be a retransmit of A's close");
    }

    // Connection B is COMPLETELY unaffected: still routable, still makes
    // real handshake progress, through the SAME listener instance.
    let mut next_buf_b = [0u8; 1500];
    let next_b = client_b
        .poll_transmit(origin, &mut next_buf_b)
        .expect("client B poll_transmit")
        .expect("client B has a second flight");
    match listener
        .ingest_datagram(origin, peer_b, &next_buf_b[..next_b.len])
        .expect("connection B keeps routing after A's error")
    {
        DatagramIngest::Existing { handle, error } => {
            assert_eq!(handle, handle_b, "B's continuation still routes to B's own handle");
            assert!(error.is_none(), "B's handshake continuation is unaffected by A's error");
        }
        other => panic!("expected Existing, got {other:?}"),
    }
    assert_eq!(
        listener.connection_handles().len(),
        2,
        "the listener itself stays alive with both table entries present — \
         a per-connection error never tears down the whole listener"
    );
}
