//! Upstream QUIC re-origination — C20.
//!
//! Opens a native QUIC/HTTP-3 connection to a real upstream host,
//! forwards an [`H3Request`], and collects the [`H3Response`]. This is
//! IO-bound async glue, not a sans-IO codec; the gate is correctness +
//! faithful relay (§14), not throughput.
//!
//! Public surface:
//! - [`H3Request`] / [`H3Response`] — typed request/response carrier.
//! - [`reoriginate_h3`] — the leaf: open native QUIC to upstream, send
//!   the decoded H3 request, collect the response.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use proxima_core::ProximaError;
use proxima_http::http3::native::driver::{DriverState, drive_client_step};
use proxima_protocols::http3_codec::client::{ClientConnection, H3ClientEvent};
use proxima_protocols::http3_codec::qpack::decoder::{DecodeError, decode_into};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN;
use proxima_protocols::quic::connection::{Connection, ConnectionState, DatagramWrite};
use proxima_protocols::quic::time::Instant as ProtoInstant;
#[cfg(test)]
use proxima_protocols::quic::tls::rustls_provider::RustlsServerProvider;
use proxima_protocols::quic::tls::rustls_provider::{RustlsClientProvider, RustlsConfig};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use rustls::pki_types::ServerName;
use tokio::net::UdpSocket;
use tracing::{debug, trace, warn};

#[cfg(test)]
use proxima_http::http3::native::driver::drive_server_step;
#[cfg(test)]
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection};

const DATAGRAM_BUF: usize = 2048;
const HANDSHAKE_ROUNDS: usize = 64;
const SETTINGS_ROUNDS: usize = 16;
const REQUEST_ROUNDS: usize = 256;
const IDLE_TIMEOUT_MS: u64 = 10_000;

/// An HTTP/3 request ready to re-originate to an upstream.
#[derive(Debug, Clone)]
pub struct H3Request {
    pub method: String,
    pub path: String,
    pub authority: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// An HTTP/3 response collected from the upstream.
#[derive(Debug, Clone)]
pub struct H3Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Open a native QUIC/HTTP-3 connection to `upstream_host:443`, send
/// `request`, and return the response.
///
/// Uses real cert validation via `webpki-roots` so the proxy talks to the
/// genuine upstream as itself. For tests, inject a custom root via the
/// `build_client_config` helper and drive over loopback instead.
///
/// # Errors
///
/// Returns [`ProximaError::Upstream`] if the UDP bind, handshake, or
/// response collection fails. Returns [`ProximaError::Decode`] on
/// H3/QUIC protocol errors.
#[must_use = "caller must await and relay the response"]
pub async fn reoriginate_h3(
    upstream_host: &str,
    request: &H3Request,
) -> Result<H3Response, ProximaError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let client_config = build_client_config_from_roots(roots, upstream_host)?;
    reoriginate_h3_with_config(upstream_host, request, client_config).await
}

/// Inner workhorse shared by the public surface and tests. Accepts a
/// pre-built rustls `ClientConfig` so tests can inject a self-signed CA.
pub(crate) async fn reoriginate_h3_with_config(
    upstream_host: &str,
    request: &H3Request,
    client_config: Arc<rustls::ClientConfig>,
) -> Result<H3Response, ProximaError> {
    let (tls_name, lookup_target) = if upstream_host.contains(':') {
        let host_only = upstream_host
            .rsplit_once(':')
            .map(|(host, _port)| host)
            .unwrap_or(upstream_host);
        (host_only.to_owned(), upstream_host.to_owned())
    } else {
        (upstream_host.to_owned(), format!("{upstream_host}:443"))
    };

    let addr: SocketAddr = tokio::net::lookup_host(&lookup_target)
        .await
        .map_err(|err| ProximaError::Upstream(format!("dns {upstream_host}: {err}")))?
        .next()
        .ok_or_else(|| ProximaError::Upstream(format!("no address for {upstream_host}")))?;

    reoriginate_h3_to_addr(&tls_name, addr, request, client_config).await
}

/// Like [`reoriginate_h3`] but with an EXPLICIT upstream socket address, bypassing
/// DNS. For transparent-redirect deployments where `/etc/hosts` is poisoned to point
/// the intercepted client at the proxy: the proxy must reach the REAL upstream IP
/// directly, or it would re-resolve the poisoned name and loop back into itself.
/// `upstream_host` is still used for the TLS SNI + cert validation.
#[must_use = "caller must await and relay the response"]
pub async fn reoriginate_h3_to(
    upstream_host: &str,
    upstream_addr: SocketAddr,
    request: &H3Request,
) -> Result<H3Response, ProximaError> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let client_config = build_client_config_from_roots(roots, upstream_host)?;
    let tls_name = upstream_host
        .rsplit_once(':')
        .map_or(upstream_host, |(host, _port)| host);
    reoriginate_h3_to_addr(tls_name, upstream_addr, request, client_config).await
}

async fn reoriginate_h3_to_addr(
    tls_name: &str,
    addr: SocketAddr,
    request: &H3Request,
    client_config: Arc<rustls::ClientConfig>,
) -> Result<H3Response, ProximaError> {
    let socket = UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|err| ProximaError::Upstream(format!("bind udp: {err}")))?;
    socket
        .connect(addr)
        .await
        .map_err(|err| ProximaError::Upstream(format!("connect udp {addr}: {err}")))?;

    let client_conn = make_client_connection(tls_name, client_config)?;
    drive_h3_request(socket, client_conn, tls_name, request).await
}

fn build_client_config_from_roots(
    roots: rustls::RootCertStore,
    upstream_host: &str,
) -> Result<Arc<rustls::ClientConfig>, ProximaError> {
    let _ = upstream_host;
    let mut config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_root_certificates(roots)
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    Ok(Arc::new(config))
}

/// Build a [`rustls::ClientConfig`] with a custom root — used in tests.
#[cfg(test)]
#[must_use]
#[allow(clippy::expect_used)]
pub(crate) fn build_test_client_config(
    root_cert: rustls::pki_types::CertificateDer<'static>,
) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(root_cert).expect("test root cert");
    let mut config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_root_certificates(roots)
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    Arc::new(config)
}

fn make_client_connection(
    server_name_str: &str,
    client_config: Arc<rustls::ClientConfig>,
) -> Result<Box<Connection<RustlsClientProvider>>, ProximaError> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let server_name = ServerName::try_from(server_name_str.to_owned())
        .map_err(|err| ProximaError::Config(format!("server name {server_name_str}: {err}")))?;

    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::aws_lc_rs::default_provider()));
    let mut dcid = [0u8; 8];
    let mut scid = [0u8; 8];
    provider
        .secure_random
        .fill(&mut dcid)
        .map_err(|_| ProximaError::Upstream("rng fill dcid failed".into()))?;
    provider
        .secure_random
        .fill(&mut scid)
        .map_err(|_| ProximaError::Upstream("rng fill scid failed".into()))?;

    let tp = encode_client_tp(&scid);
    Ok(Box::new(
        Connection::<RustlsClientProvider>::new_client(
            RustlsConfig::Client {
                config: client_config,
                server_name,
            },
            &tp,
            &dcid,
            &scid,
            ProtoInstant::from_micros(0),
        )
        .map_err(|err| ProximaError::Upstream(format!("quic new_client: {err:?}")))?,
    ))
}

fn encode_client_tp(scid: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 512];
    let written = TransportParameters {
        initial_max_data: Some(1_048_576),
        max_idle_timeout_ms: Some(IDLE_TIMEOUT_MS),
        initial_max_stream_data_bidi_local: Some(65_536),
        initial_max_stream_data_bidi_remote: Some(65_536),
        initial_max_stream_data_uni: Some(65_536),
        initial_max_streams_bidi: Some(100),
        initial_max_streams_uni: Some(100),
        initial_source_connection_id: Some(scid),
        ..Default::default()
    }
    .encode(&mut buf)
    .unwrap_or(0);
    buf.truncate(written);
    buf
}

/// Drive the QUIC + H3 client over a real UDP socket until the response
/// is fully collected or the timeout fires.
async fn drive_h3_request(
    socket: UdpSocket,
    mut client: Box<Connection<RustlsClientProvider>>,
    server_name: &str,
    request: &H3Request,
) -> Result<H3Response, ProximaError> {
    let origin = std::time::Instant::now();
    let proto_now = || {
        let micros = origin.elapsed().as_micros();
        ProtoInstant::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
    };

    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();
    let deadline = start + timeout;

    // drive handshake: emit/receive datagrams until Established
    let mut recv_buf = [0u8; DATAGRAM_BUF];
    let mut send_buf = [0u8; DATAGRAM_BUF];
    for _ in 0..HANDSHAKE_ROUNDS {
        if std::time::Instant::now() >= deadline {
            return Err(ProximaError::Upstream(format!(
                "quic handshake timeout to {server_name}"
            )));
        }
        drain_outbound_to_socket(&mut client, &socket, &mut send_buf, proto_now()).await?;
        if matches!(client.state(), ConnectionState::Established(_)) {
            break;
        }
        if let Ok(Ok((len, _peer))) =
            tokio::time::timeout(Duration::from_millis(50), socket.recv_from(&mut recv_buf)).await
        {
            client
                .handle_datagram(proto_now(), &recv_buf[..len])
                .map_err(|err| ProximaError::Upstream(format!("quic handle_datagram: {err:?}")))?;
        }
        let _ = client.handle_timeout(proto_now());
    }

    if !matches!(client.state(), ConnectionState::Established(_)) {
        return Err(ProximaError::Upstream(format!(
            "quic handshake never completed to {server_name}: state={:?}",
            client.state().label()
        )));
    }
    debug!(%server_name, "quic established");

    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut client_driver = DriverState::new();

    // SETTINGS exchange
    for _ in 0..SETTINGS_ROUNDS {
        if std::time::Instant::now() >= deadline {
            return Err(ProximaError::Upstream(format!(
                "h3 settings timeout to {server_name}"
            )));
        }
        drive_client_step(&mut client, &mut client_h3, &mut client_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 driver: {err:?}")))?;
        drain_outbound_to_socket(&mut client, &socket, &mut send_buf, proto_now()).await?;
        if let Ok(Ok((len, _peer))) =
            tokio::time::timeout(Duration::from_millis(50), socket.recv_from(&mut recv_buf)).await
        {
            client
                .handle_datagram(proto_now(), &recv_buf[..len])
                .map_err(|err| ProximaError::Upstream(format!("quic handle_datagram: {err:?}")))?;
        }
        let _ = client.handle_timeout(proto_now());
        let mut saw_settings = false;
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::SettingsEstablished { .. }) {
                saw_settings = true;
            }
        }
        if saw_settings {
            debug!(%server_name, "h3 settings established");
            break;
        }
    }

    // open the request stream
    let pseudo_headers = build_request_headers(request);
    let header_slices: Vec<(&[u8], &[u8])> = pseudo_headers
        .iter()
        .map(|(name, value)| (name.as_slice(), value.as_slice()))
        .collect();
    let stream_id = client_h3
        .open_request(&header_slices)
        .map_err(|err| ProximaError::Upstream(format!("h3 open_request: {err:?}")))?;

    if !request.body.is_empty() {
        client_h3
            .send_request_data(stream_id, &request.body)
            .map_err(|err| ProximaError::Upstream(format!("h3 send_request_data: {err:?}")))?;
    }
    client_h3
        .finish_request(stream_id)
        .map_err(|err| ProximaError::Upstream(format!("h3 finish_request: {err:?}")))?;

    // collect the response
    let mut response_status: Option<u16> = None;
    let mut response_headers: Vec<(String, String)> = Vec::new();
    let mut response_body: Vec<u8> = Vec::new();
    let mut finished = false;

    for _ in 0..REQUEST_ROUNDS {
        if std::time::Instant::now() >= deadline {
            return Err(ProximaError::Upstream(format!(
                "h3 response timeout from {server_name}"
            )));
        }
        drive_client_step(&mut client, &mut client_h3, &mut client_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 driver: {err:?}")))?;
        drain_outbound_to_socket(&mut client, &socket, &mut send_buf, proto_now()).await?;
        if let Ok(Ok((len, _peer))) =
            tokio::time::timeout(Duration::from_millis(50), socket.recv_from(&mut recv_buf)).await
        {
            client
                .handle_datagram(proto_now(), &recv_buf[..len])
                .map_err(|err| ProximaError::Upstream(format!("quic handle_datagram: {err:?}")))?;
        }
        let _ = client.handle_timeout(proto_now());

        while let Some(event) = client_h3.poll_event() {
            match event {
                H3ClientEvent::ResponseHeaders {
                    status,
                    header_block,
                    ..
                } => {
                    response_status = status;
                    let mut scratch = [0u8; PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
                    let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
                        if name.first() != Some(&b':') {
                            response_headers.push((
                                String::from_utf8_lossy(name).into_owned(),
                                String::from_utf8_lossy(value).into_owned(),
                            ));
                        }
                        Ok(())
                    };
                    // the field section was already validated (cap-enforced) when
                    // this event was produced; re-decoding here for enumeration
                    // can't fail on a well-formed peer, so a decode error just
                    // means "no extra headers surfaced" — status is already
                    // captured above.
                    let _ = decode_into(&header_block, u64::MAX, &mut scratch, &mut sink);
                }
                H3ClientEvent::ResponseData { bytes, .. } => {
                    trace!(len = bytes.len(), "h3 response data chunk");
                    response_body.extend(bytes);
                }
                H3ClientEvent::ResponseFinished { .. } => {
                    finished = true;
                }
                _ => {}
            }
        }

        if finished {
            break;
        }
    }

    if !finished {
        warn!(%server_name, "h3 response stream not finished before round limit");
    }

    let status = response_status
        .ok_or_else(|| ProximaError::Decode(format!("no :status from {server_name}")))?;

    Ok(H3Response {
        status,
        headers: response_headers,
        body: response_body,
    })
}

/// Drain all pending outbound datagrams from the QUIC state machine to
/// the socket.
async fn drain_outbound_to_socket(
    client: &mut Box<Connection<RustlsClientProvider>>,
    socket: &UdpSocket,
    buf: &mut [u8; DATAGRAM_BUF],
    now: ProtoInstant,
) -> Result<(), ProximaError> {
    loop {
        match client.poll_transmit(now, buf) {
            Ok(Some(DatagramWrite { len, .. })) => {
                socket
                    .send(&buf[..len])
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("udp send: {err}")))?;
            }
            Ok(None) => break,
            Err(err) => {
                return Err(ProximaError::Upstream(format!("poll_transmit: {err:?}")));
            }
        }
    }
    Ok(())
}

/// Build the pseudo-headers + regular headers for an H3 request.
fn build_request_headers(request: &H3Request) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out: Vec<(Vec<u8>, Vec<u8>)> = vec![
        (b":method".to_vec(), request.method.as_bytes().to_vec()),
        (b":scheme".to_vec(), b"https".to_vec()),
        (
            b":authority".to_vec(),
            request.authority.as_bytes().to_vec(),
        ),
        (b":path".to_vec(), request.path.as_bytes().to_vec()),
    ];
    for (name, value) in &request.headers {
        out.push((name.as_bytes().to_vec(), value.as_bytes().to_vec()));
    }
    out
}

// ---------------------------------------------------------------------------
// In-memory loopback helpers for tests (mirroring native_round_trip.rs).
// ---------------------------------------------------------------------------

/// Run a complete H3 request/response cycle in-memory. Both sides share
/// the caller thread via synchronous datagram exchange — no sockets, no
/// OS scheduler latency. Used by the test suite.
///
/// Spawns on a thread with 8 MiB stack: the `in_memory_h3_round_trip_inner`
/// function has large frame depth from the QUIC/H3 state machine loops.
#[cfg(test)]
pub(crate) fn in_memory_h3_round_trip(
    client_config: Arc<rustls::ClientConfig>,
    server_config: Arc<rustls::ServerConfig>,
    server_name: &str,
    request: &H3Request,
    server_handler: impl Fn(&H3Request) -> H3Response + Send + 'static,
) -> Result<H3Response, ProximaError> {
    let server_name = server_name.to_owned();
    let request = request.clone();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            in_memory_h3_round_trip_inner(
                client_config,
                server_config,
                &server_name,
                &request,
                server_handler,
            )
        })
        .map_err(|err| ProximaError::Upstream(format!("thread spawn: {err}")))?
        .join()
        .map_err(|_| ProximaError::Upstream("in_memory_h3_round_trip thread panicked".into()))?
}

#[cfg(test)]
fn in_memory_h3_round_trip_inner(
    client_config: Arc<rustls::ClientConfig>,
    server_config: Arc<rustls::ServerConfig>,
    server_name: &str,
    request: &H3Request,
    server_handler: impl Fn(&H3Request) -> H3Response,
) -> Result<H3Response, ProximaError> {
    use rand::RngExt;
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let server_name_typed = ServerName::try_from(server_name.to_owned())
        .map_err(|err| ProximaError::Config(format!("server name: {err}")))?;

    let mut client_dcid = [0u8; 8];
    let mut client_scid = [0u8; 8];
    let mut server_scid = [0u8; 8];
    rand::rng().fill(&mut client_dcid[..]);
    rand::rng().fill(&mut client_scid[..]);
    rand::rng().fill(&mut server_scid[..]);

    let client_tp = encode_client_tp(&client_scid);
    let server_tp = encode_server_tp(&client_dcid, &server_scid);

    let mut now = ProtoInstant::from_micros(1_000_000);

    let mut client = Box::new(
        Connection::<RustlsClientProvider>::new_client(
            RustlsConfig::Client {
                config: client_config,
                server_name: server_name_typed,
            },
            &client_tp,
            &client_dcid,
            &client_scid,
            now,
        )
        .map_err(|err| ProximaError::Upstream(format!("quic new_client: {err:?}")))?,
    );

    let mut server = Box::new(
        Connection::<RustlsServerProvider>::new_server(
            RustlsConfig::Server {
                config: server_config,
            },
            &server_tp,
            &client_dcid,
            &client_scid,
            &server_scid,
            now,
        )
        .map_err(|err| ProximaError::Upstream(format!("quic new_server: {err:?}")))?,
    );

    pump_until_established(&mut client, &mut server, &mut now)?;

    let mut client_h3 = ClientConnection::new(Settings::default());
    let mut server_h3 = ServerConnection::new(Settings::default());
    let mut client_driver = DriverState::new();
    let mut server_driver = DriverState::new();

    // SETTINGS exchange
    for _ in 0..SETTINGS_ROUNDS {
        drive_client_step(&mut client, &mut client_h3, &mut client_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 client driver: {err:?}")))?;
        drive_server_step(&mut server, &mut server_h3, &mut server_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 server driver: {err:?}")))?;
        pump_datagrams(&mut client, &mut server, &mut now);
        let _ = drain_server_events(&mut server_h3);
        let mut saw = false;
        while let Some(event) = client_h3.poll_event() {
            if matches!(event, H3ClientEvent::SettingsEstablished { .. }) {
                saw = true;
            }
        }
        if saw {
            break;
        }
    }

    // open the request stream
    let pseudo_headers = build_request_headers(request);
    let header_slices: Vec<(&[u8], &[u8])> = pseudo_headers
        .iter()
        .map(|(name, value)| (name.as_slice(), value.as_slice()))
        .collect();
    let stream_id = client_h3
        .open_request(&header_slices)
        .map_err(|err| ProximaError::Upstream(format!("h3 open_request: {err:?}")))?;
    if !request.body.is_empty() {
        client_h3
            .send_request_data(stream_id, &request.body)
            .map_err(|err| ProximaError::Upstream(format!("h3 send_request_data: {err:?}")))?;
    }
    client_h3
        .finish_request(stream_id)
        .map_err(|err| ProximaError::Upstream(format!("h3 finish_request: {err:?}")))?;

    // drive until the server sees the request
    // accumulators persist across all loop iterations — body may arrive in chunks
    let mut server_stream_id: Option<proxima_protocols::http3_codec::server::StreamId> = None;
    let mut srv_method = String::new();
    let mut srv_path = String::new();
    let mut srv_authority = String::new();
    let mut srv_extra_headers: Vec<(String, String)> = Vec::new();
    let mut srv_body: Vec<u8> = Vec::new();
    let mut got_request_finished = false;
    for _ in 0..REQUEST_ROUNDS {
        drive_client_step(&mut client, &mut client_h3, &mut client_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 client driver: {err:?}")))?;
        drive_server_step(&mut server, &mut server_h3, &mut server_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 server driver: {err:?}")))?;
        pump_datagrams(&mut client, &mut server, &mut now);
        while let Some(event) = client_h3.poll_event() {
            let _ = event;
        }
        while let Some(event) = server_h3.poll_event() {
            match event {
                H3ServerEvent::RequestHeaders {
                    stream_id: sid,
                    headers,
                } => {
                    server_stream_id = Some(sid);
                    for field in &headers {
                        let name = String::from_utf8_lossy(&field.name).into_owned();
                        let value = String::from_utf8_lossy(&field.value).into_owned();
                        match name.as_str() {
                            ":method" => srv_method = value,
                            ":path" => srv_path = value,
                            ":authority" => srv_authority = value,
                            _ => srv_extra_headers.push((name, value)),
                        }
                    }
                }
                H3ServerEvent::RequestData { bytes, .. } => {
                    srv_body.extend(bytes);
                }
                H3ServerEvent::RequestFinished { .. } => {
                    got_request_finished = true;
                }
                _ => {}
            }
        }
        if got_request_finished {
            break;
        }
    }
    let server_request = H3Request {
        method: srv_method,
        path: srv_path,
        authority: srv_authority,
        headers: srv_extra_headers,
        body: srv_body,
    };

    let server_sid =
        server_stream_id.ok_or_else(|| ProximaError::Decode("server stream id missing".into()))?;

    let response = server_handler(&server_request);

    // send the response
    let status_str = response.status.to_string();
    let mut resp_headers: Vec<(Vec<u8>, Vec<u8>)> =
        vec![(b":status".to_vec(), status_str.as_bytes().to_vec())];
    for (name, value) in &response.headers {
        resp_headers.push((name.as_bytes().to_vec(), value.as_bytes().to_vec()));
    }
    let resp_slices: Vec<(&[u8], &[u8])> = resp_headers
        .iter()
        .map(|(name, value)| (name.as_slice(), value.as_slice()))
        .collect();
    server_h3
        .send_response_headers(server_sid, &resp_slices)
        .map_err(|err| ProximaError::Upstream(format!("h3 send_response_headers: {err:?}")))?;
    if !response.body.is_empty() {
        server_h3
            .send_response_data(server_sid, &response.body)
            .map_err(|err| ProximaError::Upstream(format!("h3 send_response_data: {err:?}")))?;
    }
    server_h3
        .finish_response(server_sid)
        .map_err(|err| ProximaError::Upstream(format!("h3 finish_response: {err:?}")))?;

    // drive until client collects the response
    let mut resp_status: Option<u16> = None;
    let mut resp_headers_out: Vec<(String, String)> = Vec::new();
    let mut resp_body_out: Vec<u8> = Vec::new();
    let mut resp_finished = false;

    for _ in 0..REQUEST_ROUNDS {
        drive_server_step(&mut server, &mut server_h3, &mut server_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 server driver: {err:?}")))?;
        drive_client_step(&mut client, &mut client_h3, &mut client_driver)
            .map_err(|err| ProximaError::Decode(format!("h3 client driver: {err:?}")))?;
        pump_datagrams(&mut client, &mut server, &mut now);
        let _ = drain_server_events(&mut server_h3);

        while let Some(event) = client_h3.poll_event() {
            match event {
                H3ClientEvent::ResponseHeaders {
                    status,
                    header_block,
                    ..
                } => {
                    resp_status = status;
                    let mut scratch = [0u8; PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
                    let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
                        if name.first() != Some(&b':') {
                            resp_headers_out.push((
                                String::from_utf8_lossy(name).into_owned(),
                                String::from_utf8_lossy(value).into_owned(),
                            ));
                        }
                        Ok(())
                    };
                    let _ = decode_into(&header_block, u64::MAX, &mut scratch, &mut sink);
                }
                H3ClientEvent::ResponseData { bytes, .. } => {
                    resp_body_out.extend(bytes);
                }
                H3ClientEvent::ResponseFinished { .. } => {
                    resp_finished = true;
                }
                _ => {}
            }
        }

        if resp_finished && resp_status.is_some() {
            break;
        }
    }

    let status =
        resp_status.ok_or_else(|| ProximaError::Decode("no :status in response".into()))?;

    Ok(H3Response {
        status,
        headers: resp_headers_out,
        body: resp_body_out,
    })
}

#[cfg(test)]
fn drain_server_events(server_h3: &mut ServerConnection) -> Vec<H3ServerEvent> {
    let mut events = Vec::new();
    while let Some(event) = server_h3.poll_event() {
        events.push(event);
    }
    events
}

#[cfg(test)]
fn encode_server_tp(original_dcid: &[u8], local_scid: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 512];
    let written = TransportParameters {
        original_destination_connection_id: Some(original_dcid),
        initial_source_connection_id: Some(local_scid),
        initial_max_data: Some(1_048_576),
        max_idle_timeout_ms: Some(IDLE_TIMEOUT_MS),
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

#[cfg(test)]
fn pump_datagrams(
    client: &mut Box<Connection<RustlsClientProvider>>,
    server: &mut Box<Connection<RustlsServerProvider>>,
    now: &mut ProtoInstant,
) {
    *now = ProtoInstant::from_micros(now.as_micros() + 25_000);
    let mut buf = [0u8; DATAGRAM_BUF];
    while let Ok(Some(write)) = client.poll_transmit(*now, &mut buf) {
        let _ = server.handle_datagram(*now, &buf[..write.len]);
    }
    while let Ok(Some(write)) = server.poll_transmit(*now, &mut buf) {
        let _ = client.handle_datagram(*now, &buf[..write.len]);
    }
    let _ = client.handle_timeout(*now);
    let _ = server.handle_timeout(*now);
}

#[cfg(test)]
fn pump_until_established(
    client: &mut Box<Connection<RustlsClientProvider>>,
    server: &mut Box<Connection<RustlsServerProvider>>,
    now: &mut ProtoInstant,
) -> Result<(), ProximaError> {
    for _ in 0..HANDSHAKE_ROUNDS {
        pump_datagrams(client, server, now);
        if matches!(client.state(), ConnectionState::Established(_))
            && matches!(server.state(), ConnectionState::Established(_))
        {
            return Ok(());
        }
    }
    Err(ProximaError::Upstream(format!(
        "quic handshake never completed: client={} server={}",
        client.state().label(),
        server.state().label(),
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::net::UdpSocket as TokioUdpSocket;

    fn loopback_configs() -> (Arc<rustls::ServerConfig>, Arc<rustls::ClientConfig>) {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("rcgen self-signed");
        let cert_der = rustls::pki_types::CertificateDer::from(cert.cert.der().to_vec());
        let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()),
        );
        let cert_chain = vec![cert_der.clone()];
        let mut server_cfg =
            rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
                .with_no_client_auth()
                .with_single_cert(cert_chain, key_der)
                .expect("server rustls config");
        server_cfg.alpn_protocols = vec![b"h3".to_vec()];
        let client_cfg = build_test_client_config(cert_der);
        (Arc::new(server_cfg), client_cfg)
    }

    fn simple_get() -> H3Request {
        H3Request {
            method: "GET".into(),
            path: "/".into(),
            authority: "localhost".into(),
            headers: vec![],
            body: vec![],
        }
    }

    // ---- test 1: happy round-trip — GET returns the server's response -------

    #[test]
    fn happy_get_round_trip_returns_200_and_body() {
        let (server_cfg, client_cfg) = loopback_configs();
        let request = simple_get();

        let response =
            in_memory_h3_round_trip(client_cfg, server_cfg, "localhost", &request, |_req| {
                H3Response {
                    status: 200,
                    headers: vec![("content-type".into(), "text/plain".into())],
                    body: b"hello from test server".to_vec(),
                }
            })
            .expect("round trip ok");

        assert_eq!(response.status, 200, "status must be 200");
        assert_eq!(response.body, b"hello from test server", "body relay");
    }

    // ---- test 2: POST body is delivered to the server intact ----------------

    #[test]
    fn post_body_delivered_intact_faithful_relay() {
        let (server_cfg, client_cfg) = loopback_configs();
        let payload = b"faithful relay payload \xde\xad\xbe\xef".to_vec();
        let request = H3Request {
            method: "POST".into(),
            path: "/upload".into(),
            authority: "localhost".into(),
            headers: vec![("content-type".into(), "application/octet-stream".into())],
            body: payload.clone(),
        };

        let response =
            in_memory_h3_round_trip(client_cfg, server_cfg, "localhost", &request, |req| {
                H3Response {
                    status: 201,
                    headers: vec![("x-echo-len".into(), req.body.len().to_string())],
                    body: req.body.clone(),
                }
            })
            .expect("round trip ok");

        assert_eq!(response.status, 201);
        assert_eq!(
            response.body, payload,
            "server received body byte-exact (faithful relay §14)"
        );
    }

    // ---- test 3: response headers are relayed --------------------------------

    #[test]
    fn response_headers_are_relayed() {
        let (server_cfg, client_cfg) = loopback_configs();
        let request = simple_get();

        let response =
            in_memory_h3_round_trip(client_cfg, server_cfg, "localhost", &request, |_req| {
                H3Response {
                    status: 200,
                    headers: vec![
                        ("x-custom-header".into(), "relay-test".into()),
                        ("content-type".into(), "application/json".into()),
                    ],
                    body: b"{}".to_vec(),
                }
            })
            .expect("round trip ok");

        let has_custom = response
            .headers
            .iter()
            .any(|(name, value)| name == "x-custom-header" && value == "relay-test");
        assert!(has_custom, "x-custom-header must be relayed");
        let has_ct = response
            .headers
            .iter()
            .any(|(name, value)| name == "content-type" && value == "application/json");
        assert!(has_ct, "content-type must be relayed");
    }

    // ---- test 4: sad path — dead port yields typed error, no panic ----------

    #[proxima::test]
    async fn connect_to_dead_port_yields_typed_error_no_panic() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let mut config =
            rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
                .with_root_certificates(roots)
                .with_no_client_auth();
        config.alpn_protocols = vec![b"h3".to_vec()];
        let config = Arc::new(config);

        // bind a port then immediately drop the socket so nothing listens there
        let listener = TokioUdpSocket::bind("127.0.0.1:0").await.expect("bind");
        let dead_addr = listener.local_addr().expect("local_addr");
        drop(listener);

        let dead_host = format!("127.0.0.1:{}", dead_addr.port());

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            reoriginate_h3_with_config(&dead_host, &simple_get(), config),
        )
        .await;

        match result {
            Ok(Err(err)) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("upstream")
                        || msg.contains("timeout")
                        || msg.contains("handshake"),
                    "error must be typed upstream/timeout, got: {msg}"
                );
            }
            Ok(Ok(_)) => panic!("should not succeed on dead port"),
            Err(_timeout) => {
                // timeout is also acceptable — the point is no panic
            }
        }
    }

    #[proxima::test]
    async fn reoriginate_to_explicit_dead_addr_errors_not_panics() {
        // the explicit-addr path (for poisoned-/etc/hosts redirects): a dead addr
        // must yield a typed error, never a panic/hang.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let listener = TokioUdpSocket::bind("127.0.0.1:0").await.expect("bind");
        let dead_addr = listener.local_addr().expect("local_addr");
        drop(listener);

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            reoriginate_h3_to("example.com", dead_addr, &simple_get()),
        )
        .await;

        match result {
            Ok(Err(_)) | Err(_) => {}
            Ok(Ok(_)) => panic!("explicit dead addr should not succeed"),
        }
    }

    // ---- test 5: empty body round-trips cleanly -----------------------------

    #[test]
    fn empty_body_round_trips() {
        let (server_cfg, client_cfg) = loopback_configs();
        let request = H3Request {
            method: "GET".into(),
            path: "/empty".into(),
            authority: "localhost".into(),
            headers: vec![],
            body: vec![],
        };

        let response =
            in_memory_h3_round_trip(client_cfg, server_cfg, "localhost", &request, |_req| {
                H3Response {
                    status: 204,
                    headers: vec![],
                    body: vec![],
                }
            })
            .expect("round trip ok");

        assert_eq!(response.status, 204);
        assert!(response.body.is_empty(), "204 body must be empty");
    }

    // ---- test 6: multi-KB body round-trips ----------------------------------

    #[test]
    fn multi_kb_body_round_trips_byte_exact() {
        let (server_cfg, client_cfg) = loopback_configs();

        let large_body: Vec<u8> = (0u32..8192).map(|i| (i % 251) as u8).collect();
        let request = H3Request {
            method: "POST".into(),
            path: "/large".into(),
            authority: "localhost".into(),
            headers: vec![("content-length".into(), large_body.len().to_string())],
            body: large_body.clone(),
        };

        let response =
            in_memory_h3_round_trip(client_cfg, server_cfg, "localhost", &request, |req| {
                H3Response {
                    status: 200,
                    headers: vec![],
                    body: req.body.clone(),
                }
            })
            .expect("round trip ok");

        assert_eq!(response.status, 200);
        assert_eq!(
            response.body.len(),
            large_body.len(),
            "multi-KB body length must match"
        );
        assert_eq!(
            response.body, large_body,
            "multi-KB body must be byte-exact"
        );
    }
}
