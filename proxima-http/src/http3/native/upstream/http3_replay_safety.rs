use super::{H3NativeUpstream, clock_now};
use bytes::Bytes;
use futures::channel::{mpsc, oneshot};
use futures::future::{Either, select};
use prime::os::{core_shard, net::UdpSocket};
use proxima_core::ProximaError;
use proxima_primitives::pipe::{Request, SendPipe};
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_protocols::quic::connection::{Connection, ConnectionState};
use proxima_protocols::quic::endpoint::ConnectionHandle;
use proxima_protocols::quic::tls::rustls_provider::{RustlsConfig, RustlsServerProvider};
use proxima_protocols::quic::transport_parameters::TransportParameters;
use proxima_quic::native::listener::{AcceptFn, DatagramIngest, Listener};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::collections::BTreeMap;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use crate::http3::native::{DriverState, drive_server_step};

const CHARGE_BODY: &[u8] = br#"{"task":"charge1"}"#;

#[derive(Debug)]
struct Receipt {
    connection: u32,
    stream: u64,
    headers: Vec<(Vec<u8>, Vec<u8>)>,
    body: Vec<u8>,
}

impl Receipt {
    fn header(&self, name: &[u8]) -> &[u8] {
        self.headers
            .iter()
            .find(|(field, _)| field == name)
            .map_or(&[], |(_, value)| value.as_slice())
    }
}

fn tls_configs() -> (Arc<rustls::ServerConfig>, rustls::ClientConfig) {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("native TLS certificate");
    let chain = vec![CertificateDer::from(cert.cert.der().to_vec())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()));
    let mut server =
        rustls::ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_no_client_auth()
            .with_single_cert(chain.clone(), key)
            .expect("server TLS");
    server.alpn_protocols = vec![b"h3".to_vec()];
    let mut roots = rustls::RootCertStore::empty();
    roots.add(chain[0].clone()).expect("native root");
    let mut client =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .with_root_certificates(roots)
            .with_no_client_auth();
    client.alpn_protocols = vec![b"h3".to_vec()];
    (Arc::new(server), client)
}

fn native_listener(
    config: Arc<rustls::ServerConfig>,
) -> (
    Listener<RustlsServerProvider>,
    mpsc::Receiver<ConnectionHandle>,
) {
    let accept: AcceptFn<RustlsServerProvider> = Arc::new(
        move |original_dcid, client_scid, local_scid, now| {
            let mut parameters = [0_u8; 512];
            let length = TransportParameters {
                initial_max_data: Some(1_048_576),
                max_idle_timeout_ms: Some(30_000),
                initial_max_stream_data_bidi_local: Some(65_536),
                initial_max_stream_data_bidi_remote: Some(65_536),
                initial_max_stream_data_uni: Some(65_536),
                initial_max_streams_bidi: Some(100),
                initial_max_streams_uni: Some(100),
                initial_source_connection_id: Some(local_scid),
                original_destination_connection_id: Some(original_dcid),
                ..Default::default()
            }
            .encode(&mut parameters)
            .expect("native transport parameters");
            println!(
                "h3_replay_accept_tls {}",
                serde_json::json!({
                    "original_dcid": original_dcid, "client_scid": client_scid, "local_scid": local_scid,
                })
            );
            Connection::new_server(
                RustlsConfig::Server {
                    config: Arc::clone(&config),
                },
                &parameters[..length],
                original_dcid,
                client_scid,
                local_scid,
                now,
            )
        },
    );
    let (accepted, receiver) = mpsc::channel(16);
    (Listener::new(accept, accepted), receiver)
}

async fn native_peer(
    mut socket: UdpSocket,
    config: Arc<rustls::ServerConfig>,
    mut finished: oneshot::Receiver<()>,
    close_targets: usize,
) -> Vec<Receipt> {
    let (mut listener, _accepted) = native_listener(config);
    let mut connections = BTreeMap::new();
    let mut pending = BTreeMap::<(u32, u64), Receipt>::new();
    let mut receipts = Vec::new();
    let mut targets = 0;
    let mut incoming = [0_u8; 65_536];
    let mut outgoing = [0_u8; 2048];
    loop {
        let next_timeout = listener
            .connection_handles()
            .into_iter()
            .filter_map(|handle| {
                listener
                    .connection_mut(handle)
                    .and_then(|connection| connection.next_timeout())
            })
            .min();
        let timer = async {
            if let Some(deadline) = next_timeout {
                let remaining = deadline.as_micros().saturating_sub(clock_now().as_micros());
                core_shard::timer_at(
                    core_shard::current_tick().saturating_add((remaining / 1000).max(1)),
                )
                .await;
            } else {
                futures::future::pending::<()>().await;
            }
        };
        let receive =
            poll_fn(|context| Pin::new(&mut socket).poll_recv_from(context, &mut incoming));
        futures::pin_mut!(timer, receive);
        let received = select(select(receive, timer), &mut finished).await;
        match received {
            Either::Right(_) => return receipts,
            Either::Left((Either::Left((packet, _)), _)) => {
                let (length, peer) = packet.expect("native peer receives UDP");
                println!(
                    "h3_replay_packet_rx {}",
                    serde_json::json!({"peer":peer.to_string(),"bytes":&incoming[..length]})
                );
                match listener
                    .ingest_datagram(clock_now(), peer, &incoming[..length])
                    .expect("native listener ingests datagram")
                {
                    DatagramIngest::Accepted { handle, error } => {
                        assert!(error.is_none(), "accepted protocol error: {error:?}");
                        println!(
                            "h3_replay_accepted {}",
                            serde_json::json!({"connection":handle.0,"peer":peer.to_string()})
                        );
                        connections.insert(
                            handle.0,
                            (
                                peer,
                                ServerConnection::new(Settings::default()),
                                DriverState::new(),
                            ),
                        );
                    }
                    DatagramIngest::Existing { handle, error } => {
                        if let Some(error) = error {
                            println!(
                                "h3_replay_ingest_error {}",
                                serde_json::json!({"connection":handle.0,"error":format!("{error:?}")})
                            );
                        }
                    }
                    _ => {}
                }
            }
            Either::Left((Either::Right(_), _)) => {
                for handle in listener.handle_timeout(clock_now()) {
                    listener.remove_connection(handle);
                }
            }
        }
        for handle in listener.connection_handles() {
            let (peer, server, driver) = connections.get_mut(&handle.0).expect("accepted H3 state");
            let connection = listener.connection_mut(handle).expect("native connection");
            if matches!(connection.state(), ConnectionState::Established(_)) {
                drive_server_step(connection, server, driver).expect("drive native request events");
                while let Some(event) = server.poll_event() {
                    match event {
                        H3ServerEvent::RequestHeaders { stream_id, headers } => {
                            let fields = headers
                                .into_iter()
                                .map(|field| (field.name, field.value))
                                .collect();
                            let receipt = Receipt {
                                connection: handle.0,
                                stream: stream_id.0,
                                headers: fields,
                                body: Vec::new(),
                            };
                            println!(
                                "h3_replay_headers {}",
                                serde_json::json!({"connection":handle.0,"stream":stream_id.0,"headers":receipt.headers})
                            );
                            assert!(pending.insert((handle.0, stream_id.0), receipt).is_none());
                        }
                        H3ServerEvent::RequestData { stream_id, bytes } => {
                            println!(
                                "h3_replay_data {}",
                                serde_json::json!({"connection":handle.0,"stream":stream_id.0,"bytes":bytes})
                            );
                            pending
                                .get_mut(&(handle.0, stream_id.0))
                                .expect("DATA after HEADERS")
                                .body
                                .extend_from_slice(&bytes);
                        }
                        H3ServerEvent::RequestFinished { stream_id } => {
                            let receipt = pending
                                .remove(&(handle.0, stream_id.0))
                                .expect("FIN after HEADERS and DATA");
                            let length = core::str::from_utf8(receipt.header(b"content-length"))
                                .expect("literal length utf8")
                                .parse::<usize>()
                                .expect("literal content length");
                            assert_eq!(receipt.body.len(), length, "full same-stream body at FIN");
                            let target = receipt.header(b":path") == b"/decide";
                            println!(
                                "h3_replay_complete {}",
                                serde_json::json!({
                                    "connection":receipt.connection,"stream":receipt.stream,"headers":receipt.headers,
                                    "body":receipt.body,"fin":true,"response_headers_generated":false,
                                })
                            );
                            receipts.push(receipt);
                            if target {
                                targets += 1;
                            }
                            if target && targets <= close_targets {
                                connection
                                    .close(0x102, b"consumed request before lost response")
                                    .expect("native close after FIN");
                                println!(
                                    "h3_replay_close_after_fin {}",
                                    serde_json::json!({"connection":handle.0,"stream":stream_id.0,"code":0x102})
                                );
                            } else {
                                server
                                    .send_response_headers(
                                        stream_id,
                                        &[(b":status", b"200"), (b"content-length", b"2")],
                                    )
                                    .expect("native response HEADERS");
                                server
                                    .send_response_data(stream_id, b"{}")
                                    .expect("native response DATA");
                                server
                                    .finish_response(stream_id)
                                    .expect("native response FIN");
                                println!(
                                    "h3_replay_response {}",
                                    serde_json::json!({"connection":handle.0,"stream":stream_id.0,"status":200,"body":b"{}"})
                                );
                            }
                        }
                        _ => {}
                    }
                }
                if matches!(connection.state(), ConnectionState::Established(_)) {
                    drive_server_step(connection, server, driver)
                        .expect("drive native response bytes");
                }
            }
            let closing = matches!(connection.state(), ConnectionState::Closing(_));
            while let Some(packet) = connection
                .poll_transmit(clock_now(), &mut outgoing)
                .expect("native outgoing packet")
            {
                let sent = poll_fn(|context| {
                    Pin::new(&mut socket).poll_send_to(context, &outgoing[..packet.len], *peer)
                })
                .await
                .expect("native UDP send");
                assert_eq!(sent, packet.len);
                println!(
                    "h3_replay_packet_tx {}",
                    serde_json::json!({"connection":handle.0,"closing":closing,"bytes":&outgoing[..sent]})
                );
            }
        }
    }
}

async fn send(
    client: &H3NativeUpstream,
    method: &str,
    path: &str,
    body: &[u8],
) -> Result<u16, ProximaError> {
    let request = Request::builder()
        .method(method)
        .path(Bytes::copy_from_slice(path.as_bytes()))
        .header("content-length", Bytes::from(body.len().to_string()))
        .body(Bytes::copy_from_slice(body))
        .build()
        .expect("native request");
    let response = SendPipe::call(client, request).await?;
    let status = response.status;
    let body = response
        .collect_body()
        .await
        .expect("complete native response body/FIN");
    println!(
        "h3_replay_client_response {}",
        serde_json::json!({"method":method,"path":path,"status":status,"body":body.as_ref()})
    );
    assert_eq!(body.as_ref(), b"{}");
    Ok(status)
}

async fn scenario(
    method: &str,
    reused: bool,
    close_targets: usize,
) -> (Result<u16, ProximaError>, Vec<Receipt>, bool) {
    assert!(
        core_shard::current_core().is_some(),
        "native fixture runs on a Prime core"
    );
    assert!(
        core_shard::with_current_reactor(|_| ()).is_some(),
        "native fixture has a Prime reactor"
    );
    println!("h3_replay_runtime prime_core_and_reactor");
    assert_eq!(CHARGE_BODY.len(), 18, "locked paper body");
    let (server_config, client_config) = tls_configs();
    let socket = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("native UDP bind");
    let address = socket.local_addr().expect("native address");
    let (finish, finished) = oneshot::channel();
    let client = H3NativeUpstream::with_client_config(address, "localhost", client_config);
    let peer = native_peer(socket, server_config, finished, close_targets);
    let caller = async {
        if reused {
            assert_eq!(
                send(&client, "GET", "/preflight", b"")
                    .await
                    .expect("warmup complete"),
                200
            );
            assert!(
                client.conn.lock().await.is_some(),
                "actual warmup cached connection"
            );
            println!("h3_replay_warmup_cache true");
        }
        let body = if method == "GET" {
            b"".as_slice()
        } else {
            CHARGE_BODY
        };
        let outcome = send(&client, method, "/decide", body).await;
        let cleared = client.conn.lock().await.is_none();
        println!(
            "h3_replay_client_outcome {}",
            serde_json::json!({"method":method,"reused":reused,"outcome":format!("{outcome:?}"),"cache_cleared":cleared})
        );
        assert_eq!(
            send(&client, "POST", "/independent", b"new-body")
                .await
                .expect("independent native request"),
            200
        );
        finish.send(()).expect("finish native peer");
        (outcome, cleared)
    };
    let (receipts, (outcome, cleared)) = futures::join!(peer, caller);
    let independent: Vec<_> = receipts
        .iter()
        .filter(|receipt| receipt.header(b":path") == b"/independent")
        .collect();
    assert_eq!(independent.len(), 1);
    assert_eq!(independent[0].body, b"new-body");
    let targets: Vec<_> = receipts
        .iter()
        .filter(|receipt| receipt.header(b":path") == b"/decide")
        .collect();
    if reused {
        let warmup = receipts
            .iter()
            .find(|receipt| receipt.header(b":path") == b"/preflight")
            .expect("warmup receipt");
        assert_eq!(
            warmup.connection, targets[0].connection,
            "target used actual warmed connection"
        );
        assert_ne!(warmup.stream, targets[0].stream);
    }
    if cleared {
        assert_ne!(
            targets.last().expect("target receipt").connection,
            independent[0].connection,
            "independent request reconnects after failure"
        );
    }
    (outcome, receipts, cleared)
}

// locked example: proxima-tensor/specs/http3-replay-safety/SPEC.md and external proxima-h3-replay-paper.md.
#[proxima::test(timeout = "30s")]
async fn ambiguous_post_not_replayed() {
    let (outcome, receipts, cleared) = scenario("POST", true, 1).await;
    let targets: Vec<_> = receipts
        .iter()
        .filter(|receipt| receipt.header(b":path") == b"/decide")
        .collect();
    assert_eq!(
        targets.len(),
        1,
        "one complete native POST, no ambiguous replay"
    );
    assert_eq!(targets[0].body, CHARGE_BODY);
    let warmup = receipts
        .iter()
        .find(|receipt| receipt.header(b":path") == b"/preflight")
        .expect("warmup receipt");
    assert_eq!(
        warmup.connection, targets[0].connection,
        "actual connection reuse"
    );
    assert_ne!(warmup.stream, targets[0].stream);
    assert!(cleared);
    original_failure(outcome);
}

fn original_failure(outcome: Result<u16, ProximaError>) {
    match outcome {
        Err(ProximaError::Upstream(message)) => assert_eq!(
            message, "h3-native: connection ended before a response arrived",
            "original missing-response failure is preserved"
        ),
        other => panic!("expected original native missing-response error, got {other:?}"),
    }
}

#[proxima::test(timeout = "30s")]
async fn idempotent_recovery_bounded() {
    for close_targets in [1, 2] {
        let (outcome, receipts, cleared) = scenario("GET", true, close_targets).await;
        let targets: Vec<_> = receipts
            .iter()
            .filter(|receipt| receipt.header(b":path") == b"/decide")
            .collect();
        assert_eq!(targets.len(), 2, "initial GET and one automatic recovery");
        assert_ne!(
            targets[0].connection, targets[1].connection,
            "distinct native connections"
        );
        assert_eq!(
            targets[0].headers, targets[1].headers,
            "identical replayed request HEADERS"
        );
        assert_eq!(
            targets[0].body, targets[1].body,
            "identical replayed request DATA"
        );
        assert_eq!(targets[0].header(b":method"), b"GET");
        assert_eq!(targets[0].body, b"");
        if close_targets == 1 {
            assert_eq!(outcome.expect("GET automatic recovery response"), 200);
            assert!(!cleared);
        } else {
            original_failure(outcome);
            assert!(
                cleared,
                "failed recovery clears cache without a third target"
            );
        }
    }
}

#[proxima::test(timeout = "30s")]
async fn fresh_and_unknown_not_replayed() {
    for method in ["GET", "POST"] {
        let (outcome, receipts, cleared) = scenario(method, false, 1).await;
        let targets: Vec<_> = receipts
            .iter()
            .filter(|receipt| receipt.header(b":path") == b"/decide")
            .collect();
        assert_eq!(targets.len(), 1, "fresh {method} never retries");
        assert_eq!(targets[0].header(b":method"), method.as_bytes());
        assert!(cleared);
        original_failure(outcome);
    }
    for method in ["POST", "PATCH", "CONNECT", "CHARGE", "get"] {
        let (outcome, receipts, cleared) = scenario(method, true, 1).await;
        let targets: Vec<_> = receipts
            .iter()
            .filter(|receipt| receipt.header(b":path") == b"/decide")
            .collect();
        assert_eq!(targets.len(), 1, "ambiguous {method} never resubmits");
        assert_eq!(targets[0].header(b":method"), method.as_bytes());
        assert_eq!(
            targets[0].body, CHARGE_BODY,
            "actual complete method-vector payload"
        );
        assert!(cleared);
        original_failure(outcome);
    }
}
