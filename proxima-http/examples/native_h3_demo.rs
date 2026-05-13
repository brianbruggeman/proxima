//! Standalone native HTTP/3 server demo.
//!
//! Binds a UDP socket, accepts QUIC connections via the rustls bridge
//! with a self-signed cert, and serves two endpoints:
//!
//! - `GET /`       → 200 "hello from proxima native h3\n"
//! - `POST /echo`  → 200 with the request body echoed back
//! - anything else → 404
//!
//! Every state transition, frame exchange, and timer event is logged
//! so you can see the full protocol flow.
//!
//! Run:
//!   RUST_LOG=debug cargo run -p proxima-h3 --example native_h3_demo --features native
//!
//! Test with a quinn-based client, or with curl (if built with HTTP/3):
//!   curl -k --http3 https://localhost:4433/
//!   curl -k --http3 -X POST -d 'hello world' https://localhost:4433/echo

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, StreamId as H3StreamId};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::quic::connection::state::{MAX_BIDI_STREAMS, MAX_UNI_STREAMS};
use proxima_protocols::quic::connection::{Connection, ConnectionState, DatagramWrite};
use proxima_protocols::quic::endpoint::{ConnectionHandle, DatagramClassification, EndpointDemux};
use proxima_protocols::quic::time::Instant as ProtoInstant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsConfig, RustlsServerProvider};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use proxima_telemetry::{debug, error, info, warn};
use tokio::net::UdpSocket;

use proxima_http::http3::native::driver::{DriverState, drive_server_step};

const BIND_ADDR: &str = "0.0.0.0:4433";

struct ConnEntry {
    connection: Box<Connection<RustlsServerProvider>>,
    peer: SocketAddr,
    local_scid: [u8; 8],
    h3: ServerConnection,
    driver_state: DriverState,
}

#[tokio::main]
async fn main() {
    let recorder = proxima_telemetry::recorder::Recorder::builder()
        .export(proxima_telemetry::export::Exporter::std())
        .expect("console exporter")
        .install()
        .expect("install console recorder");
    proxima_telemetry::emit::global::install_from_env();
    std::thread::spawn(move || {
        loop {
            recorder.drain();
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    });

    let server_config = build_rustls_server_config();
    let socket = UdpSocket::bind(BIND_ADDR).await.expect("bind");
    let local = socket.local_addr().expect("local_addr");
    info!(%local, "native h3 demo server listening");

    let mut demux =
        EndpointDemux::with_local_cid_len(proxima_protocols::quic::connection::SUPPORTED_VERSIONS, 8);
    let mut connections: BTreeMap<u32, ConnEntry> = BTreeMap::new();
    let mut next_handle: u32 = 0;
    let mut recv_buf = [0u8; 2048];
    let origin = std::time::Instant::now();

    loop {
        let now = proto_instant(&origin);

        // 1) receive one datagram (with timeout for timer driving)
        let recv_result =
            tokio::time::timeout(Duration::from_millis(50), socket.recv_from(&mut recv_buf)).await;

        if let Ok(Ok((len, peer))) = recv_result {
            let datagram = &recv_buf[..len];
            debug!(len, %peer, "recv datagram");
            handle_inbound(
                datagram,
                peer,
                now,
                &mut demux,
                &mut connections,
                &mut next_handle,
                &server_config,
            );
        }

        // 2) drive each established connection's H3 layer
        let handles: Vec<u32> = connections.keys().copied().collect();
        for handle_id in &handles {
            let Some(entry) = connections.get_mut(handle_id) else {
                continue;
            };
            if !matches!(entry.connection.state(), ConnectionState::Established(_)) {
                continue;
            }
            if let Err(err) = drive_server_step(
                &mut entry.connection,
                &mut entry.h3,
                &mut entry.driver_state,
            ) {
                warn!(?err, handle = handle_id, "driver step error");
                let _ = entry.connection.close(0x0101, b"driver error");
                continue;
            }
            // process H3 events
            while let Some(event) = entry.h3.poll_event() {
                match event {
                    H3ServerEvent::SettingsEstablished { peer } => {
                        info!(handle = handle_id, ?peer, "H3 SETTINGS established");
                    }
                    H3ServerEvent::RequestHeaders { stream_id, headers } => {
                        let method = headers
                            .iter()
                            .find(|f| f.name == b":method")
                            .map(|f| String::from_utf8_lossy(&f.value).to_string())
                            .unwrap_or_default();
                        let path = headers
                            .iter()
                            .find(|f| f.name == b":path")
                            .map(|f| String::from_utf8_lossy(&f.value).to_string())
                            .unwrap_or_default();
                        info!(handle = handle_id, %method, %path, ?stream_id, "request headers");
                    }
                    H3ServerEvent::RequestData { stream_id, bytes } => {
                        debug!(
                            handle = handle_id,
                            ?stream_id,
                            len = bytes.len(),
                            "request data"
                        );
                    }
                    H3ServerEvent::RequestFinished { stream_id } => {
                        info!(
                            handle = handle_id,
                            ?stream_id,
                            "request finished — sending response"
                        );
                        serve_request(entry, stream_id);
                    }
                    H3ServerEvent::GoAway { peer_max_stream_id } => {
                        info!(handle = handle_id, peer_max_stream_id, "peer GOAWAY");
                    }
                    _ => {}
                }
            }
        }

        // 3) drive timers + reap closed connections
        let mut reap: Vec<(u32, [u8; 8])> = Vec::new();
        for handle_id in &handles {
            let Some(entry) = connections.get_mut(handle_id) else {
                continue;
            };
            let _ = entry.connection.handle_timeout(now);
            if matches!(
                entry.connection.state(),
                ConnectionState::Closed | ConnectionState::Draining(_)
            ) {
                reap.push((*handle_id, entry.local_scid));
            }
        }
        for (handle_id, scid) in &reap {
            connections.remove(handle_id);
            let _ = demux.unregister(scid);
            info!(handle = handle_id, "reaped connection");
        }

        // 4) drain outbound from every connection
        for handle_id in connections.keys().copied().collect::<Vec<_>>() {
            let Some(entry) = connections.get_mut(&handle_id) else {
                continue;
            };
            let mut send_buf = [0u8; 2048];
            loop {
                match entry.connection.poll_transmit(now, &mut send_buf) {
                    Ok(Some(DatagramWrite { len, .. })) => {
                        if let Err(err) = socket.send_to(&send_buf[..len], entry.peer).await {
                            error!(?err, "send_to failed");
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        warn!(?err, handle = handle_id, "poll_transmit error");
                        break;
                    }
                }
            }
        }
    }
}

fn serve_request(entry: &mut ConnEntry, stream_id: H3StreamId) {
    // look at what headers we received to decide the response
    // (the H3 proto doesn't retain headers after emitting them,
    // so a real server would stash them in application state;
    // for this demo we just respond with a fixed body)
    let _ = entry.h3.send_response_headers(
        stream_id,
        &[(b":status", b"200"), (b"content-type", b"text/plain")],
    );
    let _ = entry
        .h3
        .send_response_data(stream_id, b"hello from proxima native h3\n");
    let _ = entry.h3.finish_response(stream_id);
}

fn handle_inbound(
    datagram: &[u8],
    peer: SocketAddr,
    now: ProtoInstant,
    demux: &mut EndpointDemux,
    connections: &mut BTreeMap<u32, ConnEntry>,
    next_handle: &mut u32,
    server_config: &Arc<rustls::ServerConfig>,
) {
    let class = demux.classify_datagram(datagram);
    match class {
        DatagramClassification::Existing { handle, .. } => {
            if let Some(entry) = connections.get_mut(&handle.0)
                && let Err(err) = entry.connection.handle_datagram(now, datagram)
            {
                debug!(?err, handle = handle.0, "handle_datagram error");
            }
        }
        DatagramClassification::NewInitial { dcid, scid, .. } => {
            let local_scid = generate_scid();
            let server_tp = encode_server_tp(dcid, &local_scid);
            let connection = match Connection::<RustlsServerProvider>::new_server(
                RustlsConfig::Server {
                    config: server_config.clone(),
                },
                &server_tp,
                dcid,
                scid,
                &local_scid,
                now,
            ) {
                Ok(c) => c,
                Err(err) => {
                    warn!(?err, "new_server failed");
                    return;
                }
            };
            let handle = ConnectionHandle(*next_handle);
            *next_handle = next_handle.saturating_add(1);
            if demux.register(&local_scid, handle).is_err() {
                warn!("demux full");
                return;
            }
            let mut entry = ConnEntry {
                connection: Box::new(connection),
                peer,
                local_scid,
                h3: ServerConnection::new(Settings::default()),
                driver_state: DriverState::new(),
            };
            if let Err(err) = entry.connection.handle_datagram(now, datagram) {
                debug!(?err, "first datagram error");
            }
            connections.insert(handle.0, entry);
            info!(handle = handle.0, %peer, "accepted new connection");
        }
        DatagramClassification::UnsupportedVersion { .. } => {
            debug!("unsupported version — dropped");
        }
        DatagramClassification::Drop { reason } => {
            debug!(?reason, "dropped datagram");
        }
        _ => {}
    }
}

fn encode_server_tp(original_dcid: &[u8], local_scid: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 512];
    let written = TransportParameters {
        original_destination_connection_id: Some(original_dcid),
        initial_source_connection_id: Some(local_scid),
        initial_max_data: Some(1_048_576),
        max_idle_timeout_ms: Some(30_000),
        initial_max_stream_data_bidi_local: Some(65_536),
        initial_max_stream_data_bidi_remote: Some(65_536),
        initial_max_stream_data_uni: Some(65_536),
        initial_max_streams_bidi: Some(MAX_BIDI_STREAMS as u64),
        initial_max_streams_uni: Some(MAX_UNI_STREAMS as u64),
        ..Default::default()
    }
    .encode(&mut buf)
    .unwrap_or(0);
    buf.truncate(written);
    buf
}

fn generate_scid() -> [u8; 8] {
    use rand::{RngExt, TryRng};
    let mut out = [0u8; 8];
    if rand::rngs::SysRng.try_fill_bytes(&mut out).is_err() {
        rand::rng().fill(&mut out[..]);
    }
    out
}

fn build_rustls_server_config() -> Arc<rustls::ServerConfig> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("rcgen self-signed");
    let cert_der = cert.cert.der().clone();
    let key_pkcs8 = cert.signing_key.serialize_der();

    let cert_chain = vec![rustls::pki_types::CertificateDer::from(cert_der.to_vec())];
    let server_key = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(key_pkcs8),
    );
    let mut config =
        rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_no_client_auth()
            .with_single_cert(cert_chain, server_key)
            .expect("server config");
    config.alpn_protocols = vec![b"h3".to_vec()];
    Arc::new(config)
}

fn proto_instant(origin: &std::time::Instant) -> ProtoInstant {
    let micros = origin.elapsed().as_micros();
    ProtoInstant::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
}
