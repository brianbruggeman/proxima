#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! End-to-end handshake smoke test through the rustls-backed
//! TlsProvider. This is the load-bearing test that proves the
//! provider's read_handshake / write_handshake / KeyChange wiring
//! actually drives a connection from Initial → Handshake →
//! Established without sockets — pure in-memory loopback between
//! Connection<RustlsClientProvider> and Connection<RustlsServerProvider>.
//!
//! If this test passes, the proto's External AEAD dispatch helpers
//! work and the bridge is ready for consumers. If it doesn't, the
//! bug is local + reproducible without sockets / proxima-h3 / tokio.

#![cfg(all(feature = "quic-tls-rustls", feature = "quic-mock-tls"))]

use std::sync::Arc;

use proxima_protocols::quic::connection::{Connection, TimerOutcome};
use proxima_protocols::quic::streams::StreamDirection;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::rustls_provider::{
    RustlsClientProvider, RustlsConfig, RustlsServerProvider,
};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::{ClientConfig as RustlsClientConfig, ServerConfig as RustlsServerConfig};

fn install_default_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

fn build_configs() -> (Arc<RustlsServerConfig>, Arc<RustlsClientConfig>) {
    install_default_provider();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("rcgen self-signed");
    let cert_der = cert.cert.der().clone();
    let key_pkcs8 = cert.signing_key.serialize_der();

    let server_cert_chain = vec![CertificateDer::from(cert_der.to_vec())];
    let server_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pkcs8.clone()));
    let mut server_config =
        RustlsServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_no_client_auth()
            .with_single_cert(server_cert_chain.clone(), server_key)
            .expect("rustls server config");
    server_config.alpn_protocols = vec![b"h3".to_vec()];

    // Client trusts our self-signed cert.
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(server_cert_chain[0].clone())
        .expect("add cert to roots");
    let mut client_config =
        RustlsClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_root_certificates(roots)
            .with_no_client_auth();
    client_config.alpn_protocols = vec![b"h3".to_vec()];

    (Arc::new(server_config), Arc::new(client_config))
}

/// Configs that FORCE a TLS HelloRetryRequest: the client offers an X25519
/// key_share (its first group) but the server supports only secp256r1, so the
/// server must reject the key_share and ask the client to retry with secp256r1
/// (RFC 8446 §4.1.4). This is exactly what a modern server preferring a group
/// the client didn't key_share for does (e.g. nginx preferring X25519MLKEM768).
fn build_configs_forcing_hrr() -> (Arc<RustlsServerConfig>, Arc<RustlsClientConfig>) {
    install_default_provider();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("rcgen self-signed");
    let cert_der = cert.cert.der().clone();
    let key_pkcs8 = cert.signing_key.serialize_der();
    let server_cert_chain = vec![CertificateDer::from(cert_der.to_vec())];
    let server_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pkcs8));

    let mut server_provider = rustls::crypto::aws_lc_rs::default_provider();
    server_provider.kx_groups = vec![rustls::crypto::aws_lc_rs::kx_group::SECP256R1];
    let mut server_config = RustlsServerConfig::builder_with_provider(Arc::new(server_provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("server tls13")
        .with_no_client_auth()
        .with_single_cert(server_cert_chain.clone(), server_key)
        .expect("rustls server config");
    server_config.alpn_protocols = vec![b"h3".to_vec()];

    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(server_cert_chain[0].clone())
        .expect("add cert to roots");
    let mut client_provider = rustls::crypto::aws_lc_rs::default_provider();
    client_provider.kx_groups = vec![
        rustls::crypto::aws_lc_rs::kx_group::X25519,
        rustls::crypto::aws_lc_rs::kx_group::SECP256R1,
    ];
    let mut client_config = RustlsClientConfig::builder_with_provider(Arc::new(client_provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("client tls13")
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

/// Pump a Connection: drain every datagram out into a Vec, return them.
fn drain_outbound<P: proxima_protocols::quic::tls::TlsProvider>(
    conn: &mut Connection<P>,
    now: Instant,
) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
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

#[test]
fn rustls_bridge_handshake_completes_over_in_memory_loopback() {
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
    .expect("client connection");

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
    .expect("server connection");

    // Drive both sides until both are in Established or we've spun too many times.
    for round in 0..32 {
        now = Instant::from_micros(1_000_000 + (round + 1) * 50_000);
        let client_out = drain_outbound(&mut client, now);
        for datagram in client_out {
            server
                .handle_datagram(now, &datagram)
                .unwrap_or_else(|err| panic!("round {round} server.handle: {err:?}"));
        }
        let server_out = drain_outbound(&mut server, now);
        for datagram in server_out {
            client
                .handle_datagram(now, &datagram)
                .unwrap_or_else(|err| panic!("round {round} client.handle: {err:?}"));
        }
        // Both sides tick their timers (PTO etc.).
        let _ = client.handle_timeout(now);
        let _ = server.handle_timeout(now);
        let client_state = client.state().label();
        let server_state = server.state().label();
        eprintln!("round {round}: client={client_state} server={server_state}");
        if matches!(
            client.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) && matches!(
            server.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) {
            return; // success
        }
        let _ = TimerOutcome::Continue;
    }
    panic!(
        "handshake never completed; client={} server={}",
        client.state().label(),
        server.state().label()
    );
}

/// Regression: the client must follow a HelloRetryRequest. Before the fix, the
/// client never pumped its Initial-epoch CRYPTO after construction, so the
/// second ClientHello the HRR demands was never sent and the handshake stalled
/// in Initial forever (the exact failure seen against nginx, which prefers
/// X25519MLKEM768 and HRRs our X25519-only key_share).
#[test]
fn rustls_bridge_handshake_completes_through_hello_retry_request() {
    let (server_config, client_config) = build_configs_forcing_hrr();
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
    .expect("client connection");

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
    .expect("server connection");

    for round in 0..32 {
        now = Instant::from_micros(1_000_000 + (round + 1) * 50_000);
        for datagram in drain_outbound(&mut client, now) {
            server
                .handle_datagram(now, &datagram)
                .unwrap_or_else(|err| panic!("round {round} server.handle: {err:?}"));
        }
        for datagram in drain_outbound(&mut server, now) {
            client
                .handle_datagram(now, &datagram)
                .unwrap_or_else(|err| panic!("round {round} client.handle: {err:?}"));
        }
        let _ = client.handle_timeout(now);
        let _ = server.handle_timeout(now);
        if matches!(
            client.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) && matches!(
            server.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) {
            return; // success: the HRR was followed and the handshake completed.
        }
    }
    panic!(
        "handshake never completed through HRR; client={} server={}",
        client.state().label(),
        server.state().label()
    );
}

/// Stream-round-trip smoke test: drive the bridge to Established, open a
/// client-initiated bidi stream, ship application bytes both ways, and
/// verify they emerge intact on the other side. Without this the
/// handshake test alone leaves "the AEAD-protected 1-RTT data path
/// actually works" unproven for the rustls-backed provider.
#[test]
fn rustls_bridge_stream_round_trip_after_handshake() {
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
    .expect("client connection");
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
    .expect("server connection");

    drive_both_until_established(&mut client, &mut server, &mut now);

    // Open a client-initiated bidi stream, ship "PING" + close_send
    // (mirrors what the H3 driver does on a one-shot request stream).
    let stream_id = client
        .open_stream(StreamDirection::Bidi)
        .expect("open_stream");
    let payload_c2s = b"PING";
    let accepted = client
        .send_application(stream_id, payload_c2s)
        .expect("send_application");
    assert_eq!(accepted, payload_c2s.len());
    client.close_send(stream_id).expect("close_send");

    // Drive both sides a few rounds so the STREAM frame ships to the
    // server's recv buffer.
    pump_rounds(&mut client, &mut server, &mut now, 4);

    // Server must have observed the bytes on the matching stream id.
    let mut out = [0u8; 16];
    let read = server
        .read_stream(stream_id, &mut out)
        .expect("read_stream srv");
    assert_eq!(&out[..read], payload_c2s, "server saw client's PING");
    assert!(
        server
            .stream_recv_finished(stream_id)
            .expect("stream_recv_finished"),
        "server saw FIN"
    );

    // Server replies on the same stream with "PONG".
    let payload_s2c = b"PONG";
    let accepted = server
        .send_application(stream_id, payload_s2c)
        .expect("send_application srv");
    assert_eq!(accepted, payload_s2c.len());

    pump_rounds(&mut client, &mut server, &mut now, 4);

    let mut out = [0u8; 16];
    let read = client
        .read_stream(stream_id, &mut out)
        .expect("read_stream cli");
    assert_eq!(&out[..read], payload_s2c, "client saw server's PONG");
}

/// Regression for the 1-RTT interop bug curl/ngtcp2 exposed: a peer that
/// issues a longer SCID than we issue (curl picks 20 bytes; we issue 8).
/// The inbound short-header DCID is the CID *we* issued, so the receiver
/// must key the PN/HP offset off its own SCID length — keying off the
/// peer's (longer) CID misplaces the header-protection sample and fails
/// every 1-RTT decrypt. The handshake (long headers carry explicit CID
/// lengths) still completes, so only a post-handshake data exchange
/// catches it. Matching-length CIDs (the other tests) hide the bug.
#[test]
fn rustls_bridge_1rtt_survives_mismatched_peer_cid_length() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");
    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    // 20-byte client SCID (max-length, like curl) vs our 8-byte server SCID.
    let client_scid = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x01, 0x02, 0x03, 0x04, 0x05,
    ];
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
    .expect("client connection");
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
    .expect("server connection");

    drive_both_until_established(&mut client, &mut server, &mut now);

    // 1-RTT data: client -> server. Pre-fix this never decrypts server-side.
    let stream_id = client
        .open_stream(StreamDirection::Bidi)
        .expect("open_stream");
    let payload = b"PING";
    client
        .send_application(stream_id, payload)
        .expect("send_application");
    client.close_send(stream_id).expect("close_send");
    pump_rounds(&mut client, &mut server, &mut now, 4);

    let mut out = [0u8; 16];
    let read = server
        .read_stream(stream_id, &mut out)
        .expect("read_stream srv");
    assert_eq!(
        &out[..read],
        payload,
        "server decrypted client's 1-RTT data despite the longer peer CID"
    );
}

/// Glue a side's per-call outbound packets into ONE datagram, exercising
/// RFC 9000 §12.2 coalesced-packet processing on the receiver. OpenSSL's
/// QUIC client coalesces its Handshake Finished with its first 1-RTT
/// packet and does NOT retransmit standalone, so a receiver that processes
/// only the first coalesced packet stalls the handshake forever.
fn coalesce(datagrams: Vec<Vec<u8>>) -> Vec<u8> {
    let mut one = Vec::new();
    for datagram in datagrams {
        one.extend_from_slice(&datagram);
    }
    one
}

fn is_established<P: proxima_protocols::quic::tls::TlsProvider>(conn: &Connection<P>) -> bool {
    matches!(
        conn.state(),
        proxima_protocols::quic::connection::ConnectionState::Established(_)
    )
}

#[test]
fn rustls_bridge_handshake_completes_with_coalesced_datagrams() {
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
    .expect("client connection");
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
    .expect("server connection");

    let mut completed = false;
    for _ in 0..40 {
        now = Instant::from_micros(now.as_micros() + 50_000);
        let from_client = coalesce(drain_outbound(&mut client, now));
        if !from_client.is_empty() {
            server
                .handle_datagram(now, &from_client)
                .expect("server handles coalesced datagram");
        }
        let from_server = coalesce(drain_outbound(&mut server, now));
        if !from_server.is_empty() {
            client
                .handle_datagram(now, &from_server)
                .expect("client handles coalesced datagram");
        }
        let _ = client.handle_timeout(now);
        let _ = server.handle_timeout(now);
        if is_established(&client) && is_established(&server) {
            completed = true;
            break;
        }
    }
    assert!(
        completed,
        "coalesced-datagram handshake never completed (client={}, server={})",
        client.state().label(),
        server.state().label()
    );
}

/// Regression for the OpenSSL-QUIC interop trait: its client picks a
/// ZERO-LENGTH source connection ID (`scid: []`). The server then
/// addresses 1-RTT packets to the peer with a zero-length DCID, and must
/// still ship + receive application bytes both ways. ngtcp2 uses a
/// non-empty SCID, so the other tests never exercise the empty-CID path.
#[test]
fn rustls_bridge_1rtt_round_trip_with_zero_length_peer_cid() {
    let (server_config, client_config) = build_configs();
    let server_name = ServerName::try_from("localhost").expect("server name");
    let client_dcid = [0xc0u8, 0xff, 0xee, 0xc0, 0xde, 0xba, 0xbe, 0x42];
    let client_scid: [u8; 0] = []; // zero-length, like OpenSSL's QUIC client
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
    .expect("client connection");
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
    .expect("server connection");

    drive_both_until_established(&mut client, &mut server, &mut now);

    let stream_id = client
        .open_stream(StreamDirection::Bidi)
        .expect("open_stream");
    client
        .send_application(stream_id, b"PING")
        .expect("send_application");
    client.close_send(stream_id).expect("close_send");
    pump_rounds(&mut client, &mut server, &mut now, 4);
    let mut out = [0u8; 16];
    let read = server
        .read_stream(stream_id, &mut out)
        .expect("read_stream srv");
    assert_eq!(
        &out[..read],
        b"PING",
        "server read client 1-RTT (zero-len peer CID)"
    );

    server
        .send_application(stream_id, b"PONG")
        .expect("send_application srv");
    pump_rounds(&mut client, &mut server, &mut now, 4);
    let mut out = [0u8; 16];
    let read = client
        .read_stream(stream_id, &mut out)
        .expect("read_stream cli");
    assert_eq!(
        &out[..read],
        b"PONG",
        "client read server 1-RTT to zero-len CID"
    );
}

/// Regression for the missing HANDSHAKE_DONE (RFC 9001 §4.1.2): the
/// server MUST confirm the handshake to the client. ngtcp2 tolerated its
/// absence; OpenSSL's QUIC client reached Established then stalled (kept
/// ACKing Handshake, never drove its 1-RTT h3 request) until the server
/// sent HANDSHAKE_DONE.
#[test]
fn rustls_bridge_server_sends_handshake_done() {
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
    .expect("client connection");
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
    .expect("server connection");

    drive_both_until_established(&mut client, &mut server, &mut now);
    // The server emits HANDSHAKE_DONE on its first 1-RTT flight after
    // Established; pump so it reaches the client.
    pump_rounds(&mut client, &mut server, &mut now, 4);

    assert!(
        client.received_handshake_done(),
        "client received the server's HANDSHAKE_DONE (RFC 9001 §4.1.2)"
    );
    assert!(
        !server.received_handshake_done(),
        "server never receives HANDSHAKE_DONE — it sends it (RFC 9000 §19.20)"
    );
}

fn drive_both_until_established(
    client: &mut Connection<RustlsClientProvider>,
    server: &mut Connection<RustlsServerProvider>,
    now: &mut Instant,
) {
    for round in 0..32 {
        *now = Instant::from_micros(now.as_micros() + 50_000);
        for datagram in drain_outbound(client, *now) {
            server.handle_datagram(*now, &datagram).expect("srv handle");
        }
        for datagram in drain_outbound(server, *now) {
            client.handle_datagram(*now, &datagram).expect("cli handle");
        }
        let _ = client.handle_timeout(*now);
        let _ = server.handle_timeout(*now);
        if matches!(
            client.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) && matches!(
            server.state(),
            proxima_protocols::quic::connection::ConnectionState::Established(_)
        ) {
            return;
        }
        let _ = round;
    }
    panic!("handshake never completed");
}

fn pump_rounds(
    client: &mut Connection<RustlsClientProvider>,
    server: &mut Connection<RustlsServerProvider>,
    now: &mut Instant,
    rounds: u32,
) {
    for _ in 0..rounds {
        *now = Instant::from_micros(now.as_micros() + 50_000);
        for datagram in drain_outbound(client, *now) {
            server.handle_datagram(*now, &datagram).expect("srv handle");
        }
        for datagram in drain_outbound(server, *now) {
            client.handle_datagram(*now, &datagram).expect("cli handle");
        }
        let _ = client.handle_timeout(*now);
        let _ = server.handle_timeout(*now);
    }
}
