//! Native HTTP/3 round-trip test — pure proto + rustls bridge, no
//! quinn, no h3-quinn, no sockets. Validates the per-connection H3
//! driver routes QUIC stream bytes through the sans-IO H3 state
//! machine and back.
//!
//! Mirrors the contract proxima's legacy `tests/listener_h3.rs`
//! asserts on the quinn-compat stack: client opens a GET /, server
//! responds 200 + body "ok".

#![cfg(feature = "http3-native")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use proxima_http::http3::native::{DriverState, drive_client_step, drive_server_step};
use proxima_protocols::http3_codec::client::{ClientConnection, H3ClientEvent};
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::quic::connection::state::MAX_BIDI_STREAMS;
use proxima_protocols::quic::connection::{Connection, ConnectionState, TimerOutcome};
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;
use proxima_protocols::quic::tls::rustls_provider::{
    RustlsClientProvider, RustlsConfig, RustlsServerProvider,
};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::{ClientConfig as RustlsClientConfig, ServerConfig as RustlsServerConfig};

fn install_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

fn build_configs() -> (Arc<RustlsServerConfig>, Arc<RustlsClientConfig>) {
    install_provider();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("rcgen self-signed");
    let cert_der = cert.cert.der().clone();
    let key_pkcs8 = cert.signing_key.serialize_der();

    let cert_chain = vec![CertificateDer::from(cert_der.to_vec())];
    let server_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pkcs8));
    let mut server_config =
        RustlsServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_no_client_auth()
            .with_single_cert(cert_chain.clone(), server_key)
            .expect("server config");
    server_config.alpn_protocols = vec![b"h3".to_vec()];

    let mut roots = rustls::RootCertStore::empty();
    roots.add(cert_chain[0].clone()).expect("trust root");
    let mut client_config =
        RustlsClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_root_certificates(roots)
            .with_no_client_auth();
    client_config.alpn_protocols = vec![b"h3".to_vec()];

    (Arc::new(server_config), Arc::new(client_config))
}

fn encode_tp(scid: &[u8], odcid: Option<&[u8]>) -> Vec<u8> {
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
        original_destination_connection_id: odcid,
        ..Default::default()
    }
    .encode(&mut buf)
    .expect("encode tp");
    buf.truncate(written);
    buf
}

fn drain_datagrams<P: TlsProvider>(conn: &mut Connection<P>, now: Instant) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    loop {
        let mut buf = [0u8; 2048];
        match conn.poll_transmit(now, &mut buf) {
            Ok(Some(write)) => out.push(buf[..write.len].to_vec()),
            Ok(None) => break,
            Err(err) => panic!("poll_transmit: {err:?}"),
        }
    }
    out
}

fn pump_io<PC: TlsProvider, PS: TlsProvider>(
    client: &mut Connection<PC>,
    server: &mut Connection<PS>,
    now: &mut Instant,
) {
    *now = Instant::from_micros(now.as_micros() + 25_000);
    for d in drain_datagrams(client, *now) {
        server.handle_datagram(*now, &d).expect("srv handle");
    }
    for d in drain_datagrams(server, *now) {
        client.handle_datagram(*now, &d).expect("cli handle");
    }
    let _ = client.handle_timeout(*now);
    let _ = server.handle_timeout(*now);
    let _ = TimerOutcome::Continue;
}

fn drive_to_established(
    client: &mut Connection<RustlsClientProvider>,
    server: &mut Connection<RustlsServerProvider>,
    now: &mut Instant,
) {
    for _ in 0..32 {
        pump_io(client, server, now);
        if matches!(client.state(), ConnectionState::Established(_))
            && matches!(server.state(), ConnectionState::Established(_))
        {
            return;
        }
    }
    panic!("handshake didn't complete");
}

#[test]
fn native_h3_get_round_trip_returns_200_ok() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");

    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let server_local_scid = [0xAAu8; 8];
    let client_tp = encode_tp(&client_scid, None);
    let server_tp = encode_tp(&server_local_scid, Some(&client_dcid));
    let mut now = Instant::from_micros(1_000_000);

    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_config,
            server_name,
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        now,
    )
    .expect("client conn");
    let mut server = Connection::<RustlsServerProvider>::new_server(
        RustlsConfig::Server {
            config: server_config,
        },
        &server_tp,
        &client_dcid,
        &client_scid,
        &server_local_scid,
        now,
    )
    .expect("server conn");

    drive_to_established(&mut client, &mut server, &mut now);

    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_state = DriverState::new();
    let mut client_state = DriverState::new();

    // SETTINGS exchange — six driver passes are enough for both
    // endpoints to ship their control stream + SETTINGS frame, reach
    // the peer, and observe the SettingsEstablished event.
    let mut saw_server_settings = false;
    let mut saw_client_settings = false;
    for _ in 0..6 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::SettingsEstablished { .. }) {
                saw_server_settings = true;
            }
        }
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::SettingsEstablished { .. }) {
                saw_client_settings = true;
            }
        }
        if saw_server_settings && saw_client_settings {
            break;
        }
    }
    assert!(saw_server_settings, "server saw SETTINGS");
    assert!(saw_client_settings, "client saw SETTINGS");

    // Client issues a GET /.
    let request_id = client_h3
        .open_request(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"localhost"),
            (b":path", b"/"),
        ])
        .expect("open_request");
    client_h3
        .finish_request(request_id)
        .expect("finish_request");

    let mut saw_request_headers = false;
    let mut saw_request_finished = false;
    for _ in 0..12 {
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            match event {
                H3ServerEvent::RequestHeaders { .. } => saw_request_headers = true,
                H3ServerEvent::RequestFinished { .. } => saw_request_finished = true,
                _ => {}
            }
        }
        if saw_request_headers && saw_request_finished {
            break;
        }
    }
    assert!(saw_request_headers, "server saw RequestHeaders");
    assert!(saw_request_finished, "server saw RequestFinished");

    // Server responds 200 OK + body "ok".
    server_h3
        .send_response_headers(
            proxima_protocols::http3_codec::server::StreamId(request_id.0),
            &[(b":status", b"200")],
        )
        .expect("send_response_headers");
    server_h3
        .send_response_data(proxima_protocols::http3_codec::server::StreamId(request_id.0), b"ok")
        .expect("send_response_data");
    server_h3
        .finish_response(proxima_protocols::http3_codec::server::StreamId(request_id.0))
        .expect("finish_response");

    let mut response_status: Option<Vec<u8>> = None;
    let mut response_body = Vec::new();
    let mut saw_finished_response = false;
    for _ in 0..12 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = client_h3.poll_event() {
            match event {
                H3ClientEvent::ResponseHeaders { status, .. } => {
                    response_status = status.map(|code| code.to_string().into_bytes());
                }
                H3ClientEvent::ResponseData { bytes, .. } => response_body.extend(bytes),
                H3ClientEvent::ResponseFinished { .. } => saw_finished_response = true,
                _ => {}
            }
        }
        if response_status.is_some() && saw_finished_response {
            break;
        }
    }
    assert_eq!(response_status.as_deref(), Some(&b"200"[..]));
    assert_eq!(response_body, b"ok");
    assert!(saw_finished_response);
}

/// One full GET/200 cycle on an already-established connection. Returns the
/// number of `pump_io` steps (each = 25ms of simulated time) the cycle cost.
/// A per-request PTO stall would surface as a large pump count here.
fn serve_one_get(
    client: &mut Connection<RustlsClientProvider>,
    server: &mut Connection<RustlsServerProvider>,
    client_h3: &mut ClientConnection,
    server_h3: &mut ServerConnection,
    client_state: &mut DriverState,
    server_state: &mut DriverState,
    now: &mut Instant,
) -> usize {
    let mut pumps = 0usize;
    let request_id = client_h3
        .open_request(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"localhost"),
            (b":path", b"/"),
        ])
        .expect("open_request");
    client_h3
        .finish_request(request_id)
        .expect("finish_request");

    let mut saw_finished = false;
    for _ in 0..80 {
        drive_client_step(client, client_h3, client_state).expect("driver cli");
        drive_server_step(server, server_h3, server_state).expect("driver srv");
        pump_io(client, server, now);
        pumps += 1;
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::RequestFinished { .. }) {
                saw_finished = true;
            }
        }
        if saw_finished {
            break;
        }
    }
    assert!(
        saw_finished,
        "server never saw RequestFinished after {pumps} pumps"
    );

    server_h3
        .send_response_headers(
            proxima_protocols::http3_codec::server::StreamId(request_id.0),
            &[(b":status", b"200")],
        )
        .expect("send_response_headers");
    server_h3
        .send_response_data(proxima_protocols::http3_codec::server::StreamId(request_id.0), b"ok")
        .expect("send_response_data");
    server_h3
        .finish_response(proxima_protocols::http3_codec::server::StreamId(request_id.0))
        .expect("finish_response");

    let mut saw_response = false;
    for _ in 0..80 {
        drive_server_step(server, server_h3, server_state).expect("driver srv");
        drive_client_step(client, client_h3, client_state).expect("driver cli");
        pump_io(client, server, now);
        pumps += 1;
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::ResponseFinished { .. }) {
                saw_response = true;
            }
        }
        if saw_response {
            break;
        }
    }
    assert!(
        saw_response,
        "client never saw ResponseFinished after {pumps} pumps"
    );
    pumps
}

/// RIGOR: a persistent connection serving back-to-back GETs must not pay a
/// per-request stall. rekt_h3 on a live socket crawls at ~0.80 rps (~1.25s
/// per request) AFTER the first — this deterministic harness reproduces (or
/// refutes) that at the protocol layer, with a controlled clock and no IO.
#[test]
fn native_h3_two_sequential_requests_no_per_request_stall() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");

    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let server_local_scid = [0xAAu8; 8];
    let client_tp = encode_tp(&client_scid, None);
    let server_tp = encode_tp(&server_local_scid, Some(&client_dcid));
    let mut now = Instant::from_micros(1_000_000);

    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_config,
            server_name,
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        now,
    )
    .expect("client conn");
    let mut server = Connection::<RustlsServerProvider>::new_server(
        RustlsConfig::Server {
            config: server_config,
        },
        &server_tp,
        &client_dcid,
        &client_scid,
        &server_local_scid,
        now,
    )
    .expect("server conn");

    drive_to_established(&mut client, &mut server, &mut now);

    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_state = DriverState::new();
    let mut client_state = DriverState::new();

    let mut saw_server_settings = false;
    let mut saw_client_settings = false;
    for _ in 0..6 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::SettingsEstablished { .. }) {
                saw_server_settings = true;
            }
        }
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::SettingsEstablished { .. }) {
                saw_client_settings = true;
            }
        }
        if saw_server_settings && saw_client_settings {
            break;
        }
    }
    assert!(
        saw_server_settings && saw_client_settings,
        "SETTINGS exchanged"
    );

    let pumps_1 = serve_one_get(
        &mut client,
        &mut server,
        &mut client_h3,
        &mut server_h3,
        &mut client_state,
        &mut server_state,
        &mut now,
    );
    let pumps_2 = serve_one_get(
        &mut client,
        &mut server,
        &mut client_h3,
        &mut server_h3,
        &mut client_state,
        &mut server_state,
        &mut now,
    );
    eprintln!("STALL-PROBE pumps_1={pumps_1} pumps_2={pumps_2} (each pump = 25ms sim time)");
    assert!(
        pumps_2 <= pumps_1 + 4,
        "request 2 stalled vs request 1: pumps_1={pumps_1} pumps_2={pumps_2} (each pump=25ms)"
    );
}

/// Regression: when a server response body exceeds the per-stream QUIC
/// send buffer (STREAM_SEND_INLINE = 1024 bytes by default), the driver
/// must retain the unaccepted suffix and ship it on subsequent passes.
/// Prior to the fix, `send_all` broke out of the loop on `accepted == 0`
/// and silently dropped the suffix; the client only observed a truncated
/// body and the FIN-after-truncation. This test ships a 4096-byte body
/// and asserts the client reassembles the full payload byte-for-byte.
#[test]
fn native_h3_large_body_survives_per_stream_backpressure() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");

    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let server_local_scid = [0xAAu8; 8];
    let client_tp = encode_tp(&client_scid, None);
    let server_tp = encode_tp(&server_local_scid, Some(&client_dcid));
    let mut now = Instant::from_micros(1_000_000);

    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_config,
            server_name,
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        now,
    )
    .expect("client conn");
    let mut server = Connection::<RustlsServerProvider>::new_server(
        RustlsConfig::Server {
            config: server_config,
        },
        &server_tp,
        &client_dcid,
        &client_scid,
        &server_local_scid,
        now,
    )
    .expect("server conn");

    drive_to_established(&mut client, &mut server, &mut now);

    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_state = DriverState::new();
    let mut client_state = DriverState::new();

    for _ in 0..6 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while client_h3.poll_event().is_some() {}
        while server_h3.poll_event().is_some() {}
    }

    let request_id = client_h3
        .open_request(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"localhost"),
            (b":path", b"/large"),
        ])
        .expect("open_request");
    client_h3
        .finish_request(request_id)
        .expect("finish_request");

    let mut saw_request_finished = false;
    for _ in 0..12 {
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::RequestFinished { .. }) {
                saw_request_finished = true;
            }
        }
        if saw_request_finished {
            break;
        }
    }
    assert!(saw_request_finished, "server did not see RequestFinished");

    // 4096 bytes deterministically — larger than STREAM_SEND_INLINE
    // (1024 default per proxima-quic-proto.toml), forces partial-accept
    // on the first send_application.
    let body: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    server_h3
        .send_response_headers(
            proxima_protocols::http3_codec::server::StreamId(request_id.0),
            &[(b":status", b"200")],
        )
        .expect("send_response_headers");
    server_h3
        .send_response_data(proxima_protocols::http3_codec::server::StreamId(request_id.0), &body)
        .expect("send_response_data");
    server_h3
        .finish_response(proxima_protocols::http3_codec::server::StreamId(request_id.0))
        .expect("finish_response");

    let mut response_body: Vec<u8> = Vec::new();
    let mut saw_finished = false;
    for _ in 0..48 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = client_h3.poll_event() {
            match event {
                H3ClientEvent::ResponseData { bytes, .. } => response_body.extend(bytes),
                H3ClientEvent::ResponseFinished { .. } => saw_finished = true,
                _ => {}
            }
        }
        if saw_finished && response_body.len() == body.len() {
            break;
        }
    }
    assert!(saw_finished, "client did not see ResponseFinished");
    assert_eq!(
        response_body.len(),
        body.len(),
        "body length mismatch — driver silently dropped backpressured bytes"
    );
    assert_eq!(response_body, body, "body bytes mismatch");
}

/// Regression: the per-stream `credit_recv` guard introduced at
/// connection/mod.rs (FLOW_CONTROL_ERROR rejection) is meaningful only
/// when `MAX_STREAM_DATA` is also emitted as the application drains.
/// Without the grant, any legitimate request/response body larger than
/// the initial `initial_max_stream_data_*` advertisement (65,536 bytes
/// per the test TPs) gets rejected at byte 65,537 as a self-DoS.
///
/// This test ships a 96 KiB response body — well past the 64 KiB
/// initial credit. It passes only when the proto layer:
///   1. accounts consumed bytes against `entry.flow.recv_offset` on
///      `read_stream` (driving `should_emit_max_stream_data` past
///      its threshold);
///   2. enqueues a `FrameIntent::MaxStreamData { stream_id, maximum }`
///      with the new credit in `poll_transmit_established`;
///   3. encodes RFC 9000 §19.10 frame type 0x11 on the wire;
///   4. applies `entry.flow.grant_recv_credit(new_credit)` once the
///      frame is accepted into a packet.
#[test]
fn native_h3_response_body_larger_than_initial_per_stream_credit() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");

    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let server_local_scid = [0xAAu8; 8];
    let client_tp = encode_tp(&client_scid, None);
    let server_tp = encode_tp(&server_local_scid, Some(&client_dcid));
    let mut now = Instant::from_micros(1_000_000);

    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_config,
            server_name,
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        now,
    )
    .expect("client conn");
    let mut server = Connection::<RustlsServerProvider>::new_server(
        RustlsConfig::Server {
            config: server_config,
        },
        &server_tp,
        &client_dcid,
        &client_scid,
        &server_local_scid,
        now,
    )
    .expect("server conn");

    drive_to_established(&mut client, &mut server, &mut now);

    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_state = DriverState::new();
    let mut client_state = DriverState::new();

    for _ in 0..6 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while client_h3.poll_event().is_some() {}
        while server_h3.poll_event().is_some() {}
    }

    let request_id = client_h3
        .open_request(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"localhost"),
            (b":path", b"/large"),
        ])
        .expect("open_request");
    client_h3
        .finish_request(request_id)
        .expect("finish_request");

    let mut saw_request_finished = false;
    for _ in 0..16 {
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::RequestFinished { .. }) {
                saw_request_finished = true;
            }
        }
        if saw_request_finished {
            break;
        }
    }
    assert!(saw_request_finished, "server did not see RequestFinished");

    // 96 KiB > 65,536 — must traverse at least one MAX_STREAM_DATA grant
    // round-trip.
    let body: Vec<u8> = (0..98_304u32).map(|i| (i % 251) as u8).collect();
    server_h3
        .send_response_headers(
            proxima_protocols::http3_codec::server::StreamId(request_id.0),
            &[(b":status", b"200")],
        )
        .expect("send_response_headers");
    server_h3
        .send_response_data(proxima_protocols::http3_codec::server::StreamId(request_id.0), &body)
        .expect("send_response_data");
    server_h3
        .finish_response(proxima_protocols::http3_codec::server::StreamId(request_id.0))
        .expect("finish_response");

    let mut response_body: Vec<u8> = Vec::new();
    let mut saw_finished = false;
    // Larger iteration cap than the 4 KiB test because every
    // MAX_STREAM_DATA round-trip costs an RTT and the body needs
    // multiple grants to drain.
    for _ in 0..256 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = client_h3.poll_event() {
            match event {
                H3ClientEvent::ResponseData { bytes, .. } => response_body.extend(bytes),
                H3ClientEvent::ResponseFinished { .. } => saw_finished = true,
                _ => {}
            }
        }
        if saw_finished && response_body.len() == body.len() {
            break;
        }
    }
    assert!(
        saw_finished,
        "client did not see ResponseFinished after {} bytes",
        response_body.len()
    );
    assert_eq!(
        response_body.len(),
        body.len(),
        "body length mismatch — likely MAX_STREAM_DATA grant never fired"
    );
    assert_eq!(response_body, body, "body bytes mismatch");
}

/// An established, SETTINGS-exchanged H3 session — the shared fixture the
/// pump-based stall tests drive without re-deriving the handshake each time.
struct Session {
    client: Connection<RustlsClientProvider>,
    server: Connection<RustlsServerProvider>,
    client_h3: ClientConnection,
    server_h3: ServerConnection,
    client_state: DriverState,
    server_state: DriverState,
    now: Instant,
}

fn established_session() -> Session {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");
    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let server_local_scid = [0xAAu8; 8];
    let client_tp = encode_tp(&client_scid, None);
    let server_tp = encode_tp(&server_local_scid, Some(&client_dcid));
    let mut now = Instant::from_micros(1_000_000);
    let mut client = Connection::<RustlsClientProvider>::new_client(
        RustlsConfig::Client {
            config: client_config,
            server_name,
        },
        &client_tp,
        &client_dcid,
        &client_scid,
        now,
    )
    .expect("client conn");
    let mut server = Connection::<RustlsServerProvider>::new_server(
        RustlsConfig::Server {
            config: server_config,
        },
        &server_tp,
        &client_dcid,
        &client_scid,
        &server_local_scid,
        now,
    )
    .expect("server conn");
    drive_to_established(&mut client, &mut server, &mut now);
    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_state = DriverState::new();
    let mut client_state = DriverState::new();
    let mut saw_server = false;
    let mut saw_client = false;
    for _ in 0..6 {
        drive_server_step(&mut server, &mut server_h3, &mut server_state).expect("driver srv");
        drive_client_step(&mut client, &mut client_h3, &mut client_state).expect("driver cli");
        pump_io(&mut client, &mut server, &mut now);
        while let Some(event) = server_h3.poll_event() {
            if matches!(event, H3ServerEvent::SettingsEstablished { .. }) {
                saw_server = true;
            }
        }
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::SettingsEstablished { .. }) {
                saw_client = true;
            }
        }
        if saw_server && saw_client {
            break;
        }
    }
    assert!(saw_server && saw_client, "SETTINGS exchanged");
    Session {
        client,
        server,
        client_h3,
        server_h3,
        client_state,
        server_state,
        now,
    }
}

/// Open `count` GETs concurrently (all before any drive), then drive both
/// directions to completion. Returns total pumps. A concurrent-stream bug
/// surfaces as a `drive_*_step` error (panic) or an unmet completion count.
fn serve_concurrent_gets(session: &mut Session, count: usize) -> usize {
    let mut ids = Vec::new();
    for _ in 0..count {
        let id = session
            .client_h3
            .open_request(&[
                (b":method", b"GET"),
                (b":scheme", b"https"),
                (b":authority", b"localhost"),
                (b":path", b"/"),
            ])
            .expect("open_request");
        session
            .client_h3
            .finish_request(id)
            .expect("finish_request");
        ids.push(id);
    }
    let mut requests_seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut pumps = 0usize;
    for _ in 0..160 {
        drive_client_step(
            &mut session.client,
            &mut session.client_h3,
            &mut session.client_state,
        )
        .expect("driver cli");
        drive_server_step(
            &mut session.server,
            &mut session.server_h3,
            &mut session.server_state,
        )
        .expect("driver srv");
        pump_io(&mut session.client, &mut session.server, &mut session.now);
        pumps += 1;
        while let Some(event) = session.server_h3.poll_event() {
            if let H3ServerEvent::RequestFinished { stream_id } = event {
                requests_seen.insert(stream_id.0);
            }
        }
        if requests_seen.len() == count {
            break;
        }
    }
    assert_eq!(
        requests_seen.len(),
        count,
        "server saw only {} of {count} concurrent requests after {pumps} pumps",
        requests_seen.len()
    );
    for id in &ids {
        let sid = proxima_protocols::http3_codec::server::StreamId(id.0);
        session
            .server_h3
            .send_response_headers(sid, &[(b":status", b"200")])
            .expect("send_response_headers");
        session
            .server_h3
            .send_response_data(sid, b"ok")
            .expect("send_response_data");
        session
            .server_h3
            .finish_response(sid)
            .expect("finish_response");
    }
    let mut responses_seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for _ in 0..160 {
        drive_server_step(
            &mut session.server,
            &mut session.server_h3,
            &mut session.server_state,
        )
        .expect("driver srv");
        drive_client_step(
            &mut session.client,
            &mut session.client_h3,
            &mut session.client_state,
        )
        .expect("driver cli");
        pump_io(&mut session.client, &mut session.server, &mut session.now);
        pumps += 1;
        while let Some(event) = session.client_h3.poll_event() {
            if let H3ClientEvent::ResponseFinished { stream_id } = event {
                responses_seen.insert(stream_id.0);
            }
        }
        if responses_seen.len() == count {
            break;
        }
    }
    assert_eq!(
        responses_seen.len(),
        count,
        "client saw only {} of {count} responses after {pumps} pumps",
        responses_seen.len()
    );
    pumps
}

/// A2: concurrent streams on one connection (the live 1x4 case ERRORED) —
/// prove or disprove that the protocol itself breaks under concurrency.
#[test]
fn native_h3_four_concurrent_requests_all_succeed() {
    let mut session = established_session();
    let pumps = serve_concurrent_gets(&mut session, 4);
    eprintln!("STALL-PROBE concurrent=4 pumps={pumps} (each pump=25ms)");
    assert!(
        pumps <= 12,
        "4 concurrent requests cost {pumps} pumps (each 25ms) — concurrency stall"
    );
}

/// A3: the per-request stall accumulates over a long-lived connection —
/// prove or disprove by serving 20 sequential GETs and asserting each stays
/// cheap.
#[test]
fn native_h3_twenty_sequential_requests_stay_cheap() {
    let mut session = established_session();
    let mut counts = Vec::new();
    for _ in 0..20 {
        let pumps = serve_one_get(
            &mut session.client,
            &mut session.server,
            &mut session.client_h3,
            &mut session.server_h3,
            &mut session.client_state,
            &mut session.server_state,
            &mut session.now,
        );
        counts.push(pumps);
    }
    eprintln!("STALL-PROBE sequential20={counts:?} (each pump=25ms)");
    let worst = *counts.iter().max().expect("counts non-empty");
    assert!(
        worst <= 8,
        "a sequential request stalled: counts={counts:?} (each pump=25ms)"
    );
}

/// A4: an idle gap between requests triggers a PTO that stalls the next
/// request — prove or disprove by idling ~1.25s (50 pumps) between two GETs.
#[test]
fn native_h3_request_after_long_idle_stays_cheap() {
    let mut session = established_session();
    let first = serve_one_get(
        &mut session.client,
        &mut session.server,
        &mut session.client_h3,
        &mut session.server_h3,
        &mut session.client_state,
        &mut session.server_state,
        &mut session.now,
    );
    for _ in 0..50 {
        pump_io(&mut session.client, &mut session.server, &mut session.now);
    }
    let after_idle = serve_one_get(
        &mut session.client,
        &mut session.server,
        &mut session.client_h3,
        &mut session.server_h3,
        &mut session.client_state,
        &mut session.server_state,
        &mut session.now,
    );
    eprintln!("STALL-PROBE idle first={first} after_idle={after_idle} (each pump=25ms)");
    assert!(
        after_idle <= first + 4,
        "request after idle stalled: first={first} after_idle={after_idle}"
    );
}

/// A5: MAX_STREAMS replenishment — after serving more sequential requests than
/// the initial table cap (MAX_BIDI_STREAMS = 8), the server must have raised
/// its advertised `local_limit` above the initial cap.  Proves that the
/// `drain_peer_bidi_reaped_delta → record_peer_closed → should_emit_max_streams
/// → grant_local_max_streams` chain fires end-to-end without a real socket.
#[test]
fn native_h3_max_streams_replenishment_advances_local_limit() {
    let mut session = established_session();

    let request_count = MAX_BIDI_STREAMS * 3;
    for _ in 0..request_count {
        serve_one_get(
            &mut session.client,
            &mut session.server,
            &mut session.client_h3,
            &mut session.server_h3,
            &mut session.client_state,
            &mut session.server_state,
            &mut session.now,
        );
    }

    let local_limit = match session.server.state() {
        ConnectionState::Established(state) => state.max_streams_bidi.local_limit,
        other => panic!("expected Established, got {other:?}"),
    };

    assert!(
        local_limit > MAX_BIDI_STREAMS as u64,
        "local_limit={local_limit} did not advance past initial cap={MAX_BIDI_STREAMS}: \
         MAX_STREAMS replenishment did not fire"
    );
}
