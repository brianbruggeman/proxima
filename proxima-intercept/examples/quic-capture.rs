//! Transparent QUIC/HTTP-3 capture server (observe-only). Terminates an inbound
//! QUIC connection with a per-SNI forged cert (the same CA the TCP proxy uses, via
//! [`ForgingResolver`]), decodes HTTP/3, and dumps each request (method / path /
//! authority / headers / body) for offline analysis — the UDP analog of the h2
//! relay's `relay_h2_capture`. It answers a minimal 200 so the client completes;
//! upstream re-origination (a true transparent proxy) is the follow-on.
//!
//! To see real traffic the OS must redirect UDP:443 to this port (root):
//!   echo "rdr pass on lo0 inet proto udp from any to any port 443 -> 127.0.0.1 port 4433" | sudo pfctl -ef -
//! then point the app at it. For a quick self-test, a curl built with HTTP/3:
//!   curl -k --http3-only https://localhost:4433/hello
//!
//! Run: cargo run -p proxima-intercept --example quic-capture --features quic-intercept

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use proxima_http::http3::native::driver::{DriverState, drive_server_step};
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, StreamId as H3StreamId};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_intercept::ca::{ForgingResolver, ca_cert_pem, ca_key_pem, generate_ca, load_ca};
use proxima_intercept::quic_relay::{H3Request, H3Response, reoriginate_h3, reoriginate_h3_to};
use proxima_protocols::quic::connection::{Connection, ConnectionState, DatagramWrite};
use proxima_protocols::quic::endpoint::{ConnectionHandle, DatagramClassification, EndpointDemux};
use proxima_protocols::quic::time::Instant as ProtoInstant;
use proxima_protocols::quic::tls::rustls_provider::{RustlsConfig, RustlsServerProvider};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use tokio::net::UdpSocket;
use tracing::{debug, error, info, warn};

const BIND_DEFAULT: &str = "0.0.0.0:4433";
static DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Accumulates one request's parts across H3 events (the proto doesn't retain
/// headers after emitting them, so we stash them here keyed by stream).
#[derive(Default)]
struct RequestAccum {
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct ConnEntry {
    connection: Box<Connection<RustlsServerProvider>>,
    peer: SocketAddr,
    local_scid: [u8; 8],
    h3: ServerConnection,
    driver_state: DriverState,
    pending: BTreeMap<String, RequestAccum>,
}

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".parse().unwrap()),
        )
        .init();

    let bind = std::env::var("PROXIMA_QUIC_BIND").unwrap_or_else(|_| BIND_DEFAULT.to_string());
    let dump_dir =
        std::env::var("PROXIMA_INTERCEPT_H2_DUMP").unwrap_or_else(|_| "/tmp".to_string());
    // explicit real-upstream addr to bypass a poisoned /etc/hosts redirect loop.
    let upstream_addr: Option<SocketAddr> = std::env::var("PROXIMA_QUIC_UPSTREAM_ADDR")
        .ok()
        .and_then(|value| value.parse().ok());
    let server_config = build_forging_config();
    let socket = UdpSocket::bind(&bind).await.expect("bind udp");
    info!(%bind, "quic capture server listening (h3, forged certs)");

    let mut demux =
        EndpointDemux::with_local_cid_len(proxima_protocols::quic::connection::SUPPORTED_VERSIONS, 8);
    let mut connections: BTreeMap<u32, ConnEntry> = BTreeMap::new();
    let mut next_handle: u32 = 0;
    let mut recv_buf = [0u8; 2048];
    let origin = std::time::Instant::now();

    loop {
        let now = proto_instant(&origin);

        let recv_result =
            tokio::time::timeout(Duration::from_millis(50), socket.recv_from(&mut recv_buf)).await;
        if let Ok(Ok((len, peer))) = recv_result {
            handle_inbound(
                &recv_buf[..len],
                peer,
                now,
                &mut demux,
                &mut connections,
                &mut next_handle,
                &server_config,
            );
        }

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
            drain_h3_events(entry, *handle_id, &dump_dir, upstream_addr).await;
        }

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
        }

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

async fn drain_h3_events(
    entry: &mut ConnEntry,
    handle_id: u32,
    dump_dir: &str,
    upstream_addr: Option<SocketAddr>,
) {
    while let Some(event) = entry.h3.poll_event() {
        match event {
            H3ServerEvent::RequestHeaders { stream_id, headers } => {
                let key = format!("{stream_id:?}");
                let accum = entry.pending.entry(key).or_default();
                for field in &headers {
                    accum.headers.push((
                        String::from_utf8_lossy(&field.name).into_owned(),
                        String::from_utf8_lossy(&field.value).into_owned(),
                    ));
                }
            }
            H3ServerEvent::RequestData { stream_id, bytes } => {
                let key = format!("{stream_id:?}");
                entry
                    .pending
                    .entry(key)
                    .or_default()
                    .body
                    .extend_from_slice(&bytes);
            }
            H3ServerEvent::RequestFinished { stream_id } => {
                let key = format!("{stream_id:?}");
                let accum = entry.pending.remove(&key).unwrap_or_default();
                dump_request(dump_dir, handle_id, &accum);
                let request = build_h3_request(&accum);
                let authority = request.authority.clone();
                if authority.is_empty() {
                    respond_minimal(entry, stream_id);
                    continue;
                }
                // transparent proxy: forward to the real upstream over a fresh QUIC
                // client (C20), relay the genuine response back to the intercepted
                // client. blocks this single-conn loop during the upstream RTT —
                // acceptable for an observe proxy; a pipelined proxy would spawn.
                let relayed = match upstream_addr {
                    Some(addr) => reoriginate_h3_to(&authority, addr, &request).await,
                    None => reoriginate_h3(&authority, &request).await,
                };
                match relayed {
                    Ok(response) => {
                        dump_response(dump_dir, handle_id, &authority, &response);
                        relay_response(entry, stream_id, &response);
                    }
                    Err(err) => {
                        warn!(?err, %authority, "upstream re-origination failed");
                        respond_minimal(entry, stream_id);
                    }
                }
            }
            _ => {}
        }
    }
}

fn build_h3_request(accum: &RequestAccum) -> H3Request {
    let header = |name: &str| {
        accum
            .headers
            .iter()
            .find(|(field, _)| field == name)
            .map_or(String::new(), |(_, value)| value.clone())
    };
    let forwarded: Vec<(String, String)> = accum
        .headers
        .iter()
        .filter(|(field, _)| !field.starts_with(':'))
        .cloned()
        .collect();
    H3Request {
        method: header(":method"),
        path: header(":path"),
        authority: header(":authority"),
        headers: forwarded,
        body: accum.body.clone(),
    }
}

fn relay_response(entry: &mut ConnEntry, stream_id: H3StreamId, response: &H3Response) {
    let status = response.status.to_string();
    let mut header_refs: Vec<(&[u8], &[u8])> = vec![(b":status".as_slice(), status.as_bytes())];
    for (name, value) in &response.headers {
        if name.starts_with(':') {
            continue;
        }
        header_refs.push((name.as_bytes(), value.as_bytes()));
    }
    let _ = entry.h3.send_response_headers(stream_id, &header_refs);
    let _ = entry.h3.send_response_data(stream_id, &response.body);
    let _ = entry.h3.finish_response(stream_id);
}

fn dump_response(dump_dir: &str, handle_id: u32, authority: &str, response: &H3Response) {
    let seq = DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let safe: String = authority
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '.' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let path_out = PathBuf::from(dump_dir).join(format!("proxima-h3-{safe}-{seq:03}-resp.txt"));
    let mut out = format!("status {}\n", response.status);
    for (name, value) in &response.headers {
        out.push_str(&format!("{name}: {value}\n"));
    }
    out.push_str("\n--- body ---\n");
    out.push_str(&String::from_utf8_lossy(&response.body));
    if let Err(err) = std::fs::write(&path_out, out) {
        warn!(?err, "h3 response dump write failed");
    } else {
        info!(
            handle = handle_id,
            status = response.status,
            "upstream response relayed + dumped"
        );
    }
}

fn dump_request(dump_dir: &str, handle_id: u32, accum: &RequestAccum) {
    let header = |name: &str| {
        accum
            .headers
            .iter()
            .find(|(field_name, _)| field_name == name)
            .map_or("", |(_, value)| value.as_str())
    };
    let method = header(":method");
    let path = header(":path");
    let authority = header(":authority");
    let content_type = header("content-type");
    info!(handle = handle_id, %method, %path, %authority, %content_type, body = accum.body.len(), "h3 request");

    let seq = DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let safe_authority: String = authority
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '.' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let path_out =
        PathBuf::from(dump_dir).join(format!("proxima-h3-{safe_authority}-{seq:03}.txt"));
    let mut out = String::new();
    out.push_str(&format!("{method} {path} (authority={authority})\n"));
    for (name, value) in &accum.headers {
        out.push_str(&format!("{name}: {value}\n"));
    }
    out.push_str("\n--- body ---\n");
    out.push_str(&String::from_utf8_lossy(&accum.body));
    if let Err(err) = std::fs::write(&path_out, out) {
        warn!(?err, "h3 dump write failed");
    } else {
        info!(dump = %path_out.display(), "h3 request dumped");
    }
}

fn respond_minimal(entry: &mut ConnEntry, stream_id: H3StreamId) {
    let _ = entry.h3.send_response_headers(
        stream_id,
        &[(b":status", b"200"), (b"content-type", b"text/plain")],
    );
    let _ = entry.h3.send_response_data(stream_id, b"observed\n");
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
    match demux.classify_datagram(datagram) {
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
                Ok(connection) => connection,
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
                pending: BTreeMap::new(),
            };
            if let Err(err) = entry.connection.handle_datagram(now, datagram) {
                debug!(?err, "first datagram error");
            }
            connections.insert(handle.0, entry);
            info!(handle = handle.0, %peer, "accepted quic connection");
        }
        DatagramClassification::UnsupportedVersion { .. } | DatagramClassification::Drop { .. } => {
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
        initial_max_streams_bidi: Some(100),
        initial_max_streams_uni: Some(100),
        ..Default::default()
    }
    .encode(&mut buf)
    .unwrap_or(0);
    buf.truncate(written);
    buf
}

fn generate_scid() -> [u8; 8] {
    use rand::RngExt;
    let mut out = [0u8; 8];
    rand::rng().fill(&mut out[..]);
    out
}

fn build_forging_config() -> Arc<rustls::ServerConfig> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let ca_dir =
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())).join(".proxima");
    let cert_path = ca_dir.join("ca.pem");
    let key_path = ca_dir.join("ca-key.pem");
    let ca = if cert_path.exists() && key_path.exists() {
        load_ca(&cert_path, &key_path).expect("load ca")
    } else {
        std::fs::create_dir_all(&ca_dir).expect("create ca dir");
        let generated = generate_ca().expect("generate ca");
        std::fs::write(&cert_path, ca_cert_pem(&generated).expect("ca pem")).expect("write ca");
        std::fs::write(&key_path, ca_key_pem(&generated)).expect("write key");
        generated
    };
    let mut config =
        rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(ForgingResolver::new(Arc::new(ca))));
    config.alpn_protocols = vec![b"h3".to_vec()];
    Arc::new(config)
}

fn proto_instant(origin: &std::time::Instant) -> ProtoInstant {
    let micros = origin.elapsed().as_micros();
    ProtoInstant::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
}
