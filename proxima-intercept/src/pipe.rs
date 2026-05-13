use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "intercept-capture")]
use std::time::Instant;
use std::time::SystemTime;

#[cfg(feature = "intercept-capture")]
use crate::capture::Capture;

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
#[cfg(not(feature = "intercept-config"))]
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};
use proxima_primitives::pipe::{Request, Response};
use serde_json::Value;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt as TokioWrite;
use tokio::net::TcpStream;

use crate::ca::{CaKeyPair, build_tls_acceptor, generate_ca, load_ca};
use crate::interceptor::{HostPolicy, Interception};

pub fn ts() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() % 86400;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let seconds = secs % 60;
    let ms = now.subsec_millis();
    format!("{hours:02}:{mins:02}:{seconds:02}.{ms:03}")
}

pub struct InterceptPipe {
    ca: Arc<CaKeyPair>,
    // the proxy's own externally-reachable `host:port`.
    proxy_addr: Option<String>,
    #[cfg(feature = "delta-tee")]
    delta_tx: Option<proxima_primitives::sync::broadcast::Sender<bytes::Bytes>>,
    #[cfg(feature = "intercept-capture")]
    capture: Option<Capture>,
    // domain policy/transforms (request rewrite, observe, route, host gate); when
    // None the proxy forwards verbatim — the original observe-and-forward behaviour.
    interceptor: Option<Arc<dyn crate::interceptor::Interceptor>>,
}

impl InterceptPipe {
    pub fn new(ca: CaKeyPair) -> Self {
        Self {
            ca: Arc::new(ca),
            proxy_addr: None,
            #[cfg(feature = "delta-tee")]
            delta_tx: None,
            #[cfg(feature = "intercept-capture")]
            capture: None,
            interceptor: None,
        }
    }

    /// Inject the domain policy/transforms — rewrite the outbound request, observe
    /// the exchange, route elsewhere, gate the host. proxima keeps the MITM
    /// mechanics; the [`Interceptor`](crate::interceptor::Interceptor) keeps what
    /// is domain-specific.
    #[must_use]
    pub fn with_interceptor(
        mut self,
        interceptor: Arc<dyn crate::interceptor::Interceptor>,
    ) -> Self {
        self.interceptor = Some(interceptor);
        self
    }

    /// Set the proxy's own externally-reachable address (`host:port`).
    #[must_use]
    pub fn with_proxy_addr(mut self, addr: impl Into<String>) -> Self {
        self.proxy_addr = Some(addr.into());
        self
    }

    /// Attach a broadcast sender so every raw chunk byte is teed to subscribers
    /// as it is written to the client. Best-effort: send failures (no receivers,
    /// lagging receivers) are silently dropped and never fail the response path.
    /// Requires the `delta-tee` feature.
    #[cfg(feature = "delta-tee")]
    #[must_use]
    pub fn with_delta_tee(mut self, sender: proxima_primitives::sync::broadcast::Sender<bytes::Bytes>) -> Self {
        self.delta_tx = Some(sender);
        self
    }

    pub fn with_generated_ca() -> Result<Self, ProximaError> {
        Ok(Self::new(generate_ca()?))
    }

    pub fn with_ca_files(
        cert_path: &std::path::Path,
        key_path: &std::path::Path,
    ) -> Result<Self, ProximaError> {
        Ok(Self::new(load_ca(cert_path, key_path)?))
    }

    #[cfg(feature = "intercept-capture")]
    #[must_use]
    pub fn with_capture(mut self, capture: Capture) -> Self {
        self.capture = Some(capture);
        self
    }

    // test-only inspection of the capture terminal's armed state — lets the
    // config parity test assert both lowering paths build a disarmed terminal
    // without exposing the private `capture` field.
    #[cfg(all(test, feature = "intercept-config"))]
    pub(crate) fn capture_is_armed(&self) -> bool {
        self.capture
            .as_ref()
            .is_some_and(crate::capture::Capture::is_armed)
    }
}

impl SendPipe for InterceptPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let ca = Arc::clone(&self.ca);
        #[cfg(feature = "delta-tee")]
        let delta_tx = self.delta_tx.clone();
        #[cfg(feature = "intercept-capture")]
        let capture = self.capture.clone();
        let interceptor = self.interceptor.clone();
        async move {
            let path_str = String::from_utf8_lossy(&request.path);
            let (target_host, target_port) = parse_connect_target(&path_str);
            let target_addr = format!("{target_host}:{target_port}");
            let host_owned = target_host.to_string();

            eprintln!("{} [connect] {target_addr}", ts());

            let handler = UpgradeHandler::new(move |hijacked: HijackedSocket| {
                let ca = ca;
                let target_host = host_owned;
                let target_addr = target_addr;
                #[cfg(feature = "delta-tee")]
                let delta_tx = delta_tx;
                #[cfg(feature = "intercept-capture")]
                let capture = capture;
                let interceptor = interceptor;

                async move {
                    let HijackedSocket { stream, leftover } = hijacked;

                    // operator host policy runs BEFORE TLS termination: a refused
                    // host drops the tunnel; a passthrough host is raw-tunnelled
                    // with no MITM (we cannot un-terminate TLS after accepting).
                    if let Some(policy) = interceptor
                        .as_ref()
                        .map(|hooks| hooks.host_policy(&target_host))
                    {
                        match policy {
                            HostPolicy::Refuse => return Ok(()),
                            HostPolicy::Passthrough => {
                                let adapter = FuturesIoToTokio::new(stream, leftover);
                                let (reader, writer) = tokio::io::split(adapter);
                                return pipe_raw(&target_host, &target_addr, reader, writer).await;
                            }
                            HostPolicy::Intercept => {}
                        }
                    }

                    let acceptor = build_tls_acceptor(&ca, &target_host, wants_h2(&target_host))?;
                    let tokio_compat = FuturesIoToTokio::new(stream, leftover);

                    let client_tls = match acceptor.accept(tokio_compat).await {
                        Ok(tls) => tls,
                        Err(err) => {
                            // a forged-cert rejection here is the cert-pinning signal
                            // for an h2 host: the client refused our CA.
                            eprintln!("{} [tls-fail] {target_host}: {err}", ts());
                            return Err(ProximaError::Upstream(format!("tls accept: {err}")));
                        }
                    };

                    let negotiated_h2 = client_tls
                        .get_ref()
                        .1
                        .alpn_protocol()
                        .is_some_and(|proto| proto == b"h2");

                    let (client_reader, client_writer) = tokio::io::split(client_tls);
                    let mut client_reader = client_reader;
                    let mut client_writer = client_writer;

                    if is_telemetry_host(&target_host) {
                        return pipe_raw(&target_host, &target_addr, client_reader, client_writer)
                            .await;
                    }

                    // h2 capture path: the hand-rolled http/1.1 parser below cannot
                    // read h2 frames, so an h2-negotiated connection is relayed
                    // transparently (TLS-decrypted both legs) while the plaintext is
                    // teed to a dump for offline decode via proxima-h2-codec. Gated by
                    // wants_h2 so only opted-in hosts ever reach here.
                    if negotiated_h2 {
                        eprintln!(
                            "{} [h2] {target_host}: negotiated h2, relaying + capturing",
                            ts()
                        );
                        return relay_h2_capture(
                            &target_host,
                            &target_addr,
                            client_reader,
                            client_writer,
                        )
                        .await;
                    }

                    #[cfg(feature = "intercept-capture")]
                    let interaction_started = Instant::now();

                    let mut first_buf = vec![0u8; 16 * 1024];
                    let first_read =
                        match tokio::io::AsyncReadExt::read(&mut client_reader, &mut first_buf)
                            .await
                        {
                            Ok(0) => return Ok(()),
                            Ok(count) => count,
                            Err(_) => return Ok(()),
                        };
                    first_buf.truncate(first_read);

                    let first_text = String::from_utf8_lossy(&first_buf).to_string();
                    let is_websocket = request_is_websocket_upgrade(&first_text);

                    // operator route override: forward to a different address.
                    let forward_addr = interceptor
                        .as_ref()
                        .and_then(|hooks| hooks.upstream_override(&target_host))
                        .unwrap_or_else(|| target_addr.clone());
                    let upstream = connect_upstream(&target_host, &forward_addr).await?;
                    let (mut upstream_reader, mut upstream_writer) = tokio::io::split(upstream);

                    if is_websocket {
                        eprintln!("{} [ws-upgrade] {target_host}", ts());

                        upstream_writer
                            .write_all(&first_buf)
                            .await
                            .map_err(ProximaError::Io)?;
                        let _ = upstream_writer.flush().await;

                        #[cfg(feature = "intercept-capture")]
                        let ws_interaction = match capture.as_ref() {
                            Some(recorder) => match recorder
                                .begin(&target_host, &first_buf, interaction_started)
                                .await
                            {
                                Ok(interaction) => Some(interaction),
                                Err(err) => {
                                    eprintln!("{} [capture-begin-error-ws] {err}", ts());
                                    None
                                }
                            },
                            None => None,
                        };
                        #[cfg(feature = "intercept-capture")]
                        let req_pusher = ws_interaction
                            .as_ref()
                            .map(|interaction| interaction.request_pusher());
                        #[cfg(feature = "intercept-capture")]
                        let resp_pusher = ws_interaction
                            .as_ref()
                            .map(|interaction| interaction.response_pusher());

                        let client_to_upstream = async move {
                            #[cfg(feature = "intercept-capture")]
                            let req_pusher = req_pusher;
                            pump_ws_client_to_upstream(
                                &mut client_reader,
                                &mut upstream_writer,
                                |chunk| {
                                    #[cfg(feature = "intercept-capture")]
                                    if let Some(ref pusher) = req_pusher {
                                        pusher.push(Bytes::copy_from_slice(chunk));
                                    }
                                    let _ = chunk;
                                },
                            )
                            .await;
                        };

                        let upstream_to_client = async move {
                            #[cfg(feature = "intercept-capture")]
                            let resp_pusher = resp_pusher;
                            let ws_report = pump_ws_upstream_to_client(
                                &mut upstream_reader,
                                &mut client_writer,
                                |chunk| {
                                    #[cfg(feature = "intercept-capture")]
                                    if let Some(ref pusher) = resp_pusher {
                                        pusher.push(Bytes::copy_from_slice(chunk));
                                    }
                                    let _ = chunk;
                                },
                            )
                            .await;
                            #[cfg(feature = "intercept-capture")]
                            {
                                (ws_report.delta_accum, ws_report.real_head_wire)
                            }
                            #[cfg(not(feature = "intercept-capture"))]
                            {
                                ws_report.delta_accum
                            }
                        };

                        let (_, ws_output) = tokio::join!(client_to_upstream, upstream_to_client);

                        #[cfg(feature = "intercept-capture")]
                        let (full_response, real_head_wire) = ws_output;
                        #[cfg(not(feature = "intercept-capture"))]
                        let _full_response = ws_output;

                        #[cfg(feature = "intercept-capture")]
                        if let Some(interaction) = ws_interaction {
                            const SYNTHETIC_101_FALLBACK: &[u8] = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
                            let head_wire = if real_head_wire.is_empty() {
                                SYNTHETIC_101_FALLBACK
                            } else {
                                real_head_wire.as_slice()
                            };
                            if let Err(err) = interaction.finish(head_wire).await {
                                eprintln!("{} [capture-error-ws] {err}", ts());
                            } else {
                                eprintln!(
                                    "{} [captured-ws] {target_host} {} response chars (real head: {} bytes)",
                                    ts(),
                                    full_response.len(),
                                    real_head_wire.len()
                                );
                            }
                        }
                    } else {
                        let body_buf =
                            accumulate_request_body(&mut client_reader, &first_buf, &first_text)
                                .await;
                        let request_body = String::from_utf8_lossy(&body_buf).to_string();

                        let model = parse_request_model(&request_body);
                        eprintln!(
                            "{} [request] {target_host}: {model} ({} body bytes)",
                            ts(),
                            request_body.len()
                        );

                        #[cfg(not(feature = "intercept-capture"))]
                        {
                            let dump_path = format!(
                                "/tmp/proxima-request-{}.json",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis()
                            );
                            if !request_body.is_empty() {
                                let _ = std::fs::write(&dump_path, &request_body);
                                eprintln!("{} [request-body] dumped to {dump_path}", ts());
                            }
                        }

                        // the interceptor decides the outbound request's fate:
                        // forward it (optionally rewritten — e.g. injected memory),
                        // or short-circuit with a direct response (the swap path).
                        // with no interceptor, forward the verbatim wire as before.
                        if let Some(hooks) = interceptor.as_ref() {
                            match hooks.intercept(&target_host, &body_buf).await {
                                Interception::Forward(bytes) => {
                                    upstream_writer
                                        .write_all(&bytes)
                                        .await
                                        .map_err(ProximaError::Io)?;
                                }
                                Interception::Respond(response) => {
                                    let _ =
                                        TokioWrite::write_all(&mut client_writer, &response).await;
                                    let _ = TokioWrite::flush(&mut client_writer).await;
                                    let _ = client_writer.shutdown().await;
                                    return Ok(());
                                }
                            }
                        } else {
                            upstream_writer
                                .write_all(&first_buf)
                                .await
                                .map_err(ProximaError::Io)?;
                            let tail = request_body_tail(
                                first_buf.len(),
                                first_text.find("\r\n\r\n"),
                                request_body.as_bytes(),
                            );
                            if !tail.is_empty() {
                                upstream_writer
                                    .write_all(tail)
                                    .await
                                    .map_err(ProximaError::Io)?;
                            }
                        }
                        let _ = upstream_writer.flush().await;
                        // do NOT shutdown the upstream write half: on a TLS stream that
                        // sends close_notify, which some real HTTP/1.1 servers (e.g.
                        // behind a CDN) read as end-of-connection and tear down before
                        // replying — small/fast requests then come back empty. The
                        // Content-Length already frames the request.

                        #[cfg(feature = "intercept-capture")]
                        let interaction = match (capture.as_ref(), interaction_started) {
                            (Some(recorder), started) => {
                                match recorder.begin(&target_host, &first_buf, started).await {
                                    Ok(handle) => {
                                        handle.push_request(Bytes::copy_from_slice(
                                            request_body.as_bytes(),
                                        ));
                                        Some(handle)
                                    }
                                    Err(err) => {
                                        eprintln!("{} [capture-begin-error] {err}", ts());
                                        None
                                    }
                                }
                            }
                            (None, _) => None,
                        };

                        #[cfg(feature = "intercept-capture")]
                        let interaction_ref = interaction.as_ref();
                        let pump_report = pump_streaming_response(
                            &mut upstream_reader,
                            &mut client_writer,
                            |chunk| {
                                #[cfg(feature = "intercept-capture")]
                                if let Some(handle) = interaction_ref {
                                    handle.push_response(Bytes::copy_from_slice(chunk));
                                }
                                #[cfg(feature = "delta-tee")]
                                crate::tee::tee_chunk(&delta_tx, &Bytes::copy_from_slice(chunk));
                                let _ = chunk;
                            },
                        )
                        .await?;
                        if pump_report.is_sse {
                            eprintln!("{} [sse-stream-begin] {target_host}", ts());
                        }
                        let _ = client_writer.shutdown().await;

                        let response_head_wire = pump_report.response_head_wire;
                        let stream_buf = pump_report.stream_buf;

                        // observe the completed exchange (original request + full
                        // response wire) so the interceptor can persist / enrich.
                        if let Some(hooks) = interceptor.as_ref() {
                            let mut response_wire = response_head_wire.clone();
                            response_wire.extend_from_slice(&stream_buf);
                            hooks.observe(&target_host, &body_buf, &response_wire).await;
                        }
                        #[cfg(feature = "intercept-capture")]
                        let total_body_bytes = stream_buf.len();
                        #[cfg(feature = "intercept-capture")]
                        let is_sse = pump_report.is_sse;

                        #[cfg(not(feature = "intercept-capture"))]
                        {
                            let dump_path = format!(
                                "/tmp/proxima-response-{}.txt",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis()
                            );
                            let mut combined = response_head_wire.clone();
                            combined.extend_from_slice(&stream_buf);
                            let _ = std::fs::write(&dump_path, &combined);
                            eprintln!(
                                "{} [response-dump] {} bytes -> {dump_path}",
                                ts(),
                                combined.len()
                            );
                        }

                        #[cfg(feature = "intercept-capture")]
                        if let Some(handle) = interaction {
                            if let Err(err) = handle.finish(&response_head_wire).await {
                                eprintln!("{} [capture-error] {err}", ts());
                            } else {
                                let kind = if is_sse { "sse" } else { "post" };
                                eprintln!(
                                    "{} [captured-{kind}] {target_host} {} req + {} resp bytes",
                                    ts(),
                                    request_body.len(),
                                    total_body_bytes
                                );
                            }
                        }
                    }

                    Ok(())
                }
            });

            Ok(Response::new(200).with_upgrade(handler))
        }
    }
}


/// Hosts whose traffic is TLS-terminated as h2 and captured for offline decode.
/// True if `host` contains any comma-separated substring in `list`. The host
/// domains themselves are deployment data, never compiled in.
fn host_in_list(host: &str, list: &str) -> bool {
    list.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .any(|entry| host.contains(entry))
}

/// Some clients (h2 + protobuf) cannot be parsed by the http/1.1 pipeline and
/// must negotiate h2. Which hosts those are is deployment-specific, so the list
/// is opt-in via `PROXIMA_INTERCEPT_H2_HOSTS` (comma-separated substrings) — no
/// vendor domain is compiled in. Unset means no host negotiates h2.
fn wants_h2(host: &str) -> bool {
    std::env::var("PROXIMA_INTERCEPT_H2_HOSTS").is_ok_and(|list| host_in_list(host, &list))
}

/// Upstream TLS that advertises h2 in ALPN — the client leg of the h2 capture
/// relay, so the negotiated protocol matches what the intercepted client speaks.
async fn connect_upstream_h2(
    host: &str,
    addr: &str,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, ProximaError> {
    let upstream = TcpStream::connect(addr)
        .await
        .map_err(|err| ProximaError::Upstream(format!("connect {addr}: {err}")))?;
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut client_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    client_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls::pki_types::ServerName::try_from(host)
        .map_err(|err| ProximaError::Config(format!("server name: {err}")))?
        .to_owned();
    connector
        .connect(server_name, upstream)
        .await
        .map_err(|err| ProximaError::Upstream(format!("tls connect {addr}: {err}")))
}

/// Per-connection sequence so concurrent/sequential h2 connections to the same
/// host get distinct dump files instead of truncating each other (h2 multiplexes
/// many streams per connection, and a client opens several connections).
static H2_DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Transparent h2 relay that tees the decrypted plaintext of BOTH directions to
/// dump files, so the captured frames decode offline with proxima-h2-codec and
/// the Connect/protobuf vocab can be characterized from real bytes (§14). This
/// observes — it does not terminate h2; full termination/forwarding is the
/// follow-on once the wire is known. Dump dir: `PROXIMA_INTERCEPT_H2_DUMP` or /tmp.
async fn relay_h2_capture<Reader, Writer>(
    host: &str,
    addr: &str,
    client_reader: Reader,
    client_writer: Writer,
) -> Result<(), ProximaError>
where
    Reader: AsyncRead + Unpin,
    Writer: AsyncWrite + Unpin,
{
    let upstream = connect_upstream_h2(host, addr).await?;
    let (upstream_reader, upstream_writer) = tokio::io::split(upstream);

    let dump_dir =
        std::env::var("PROXIMA_INTERCEPT_H2_DUMP").unwrap_or_else(|_| "/tmp".to_string());
    let safe_host: String = host
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '.' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let seq = H2_DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let c2s_path =
        PathBuf::from(&dump_dir).join(format!("proxima-h2-{safe_host}-{seq:03}-c2s.bin"));
    let s2c_path =
        PathBuf::from(&dump_dir).join(format!("proxima-h2-{safe_host}-{seq:03}-s2c.bin"));
    eprintln!(
        "{} [h2-dump] {host} c2s={} s2c={}",
        ts(),
        c2s_path.display(),
        s2c_path.display()
    );

    let (forward, backward) = tokio::join!(
        copy_tee(client_reader, upstream_writer, c2s_path),
        copy_tee(upstream_reader, client_writer, s2c_path),
    );
    forward?;
    backward?;
    Ok(())
}

/// Copy `from` → `to`, appending every chunk to `dump_path` for offline analysis.
async fn copy_tee<Reader, Writer>(
    mut from: Reader,
    mut to: Writer,
    dump_path: PathBuf,
) -> Result<(), ProximaError>
where
    Reader: AsyncRead + Unpin,
    Writer: AsyncWrite + Unpin,
{
    let mut dump = tokio::fs::File::create(&dump_path)
        .await
        .map_err(|err| ProximaError::Upstream(format!("h2 dump create: {err}")))?;
    let mut buffer = vec![0u8; 32 * 1024];
    loop {
        let count = tokio::io::AsyncReadExt::read(&mut from, &mut buffer)
            .await
            .map_err(|err| ProximaError::Upstream(format!("h2 relay read: {err}")))?;
        if count == 0 {
            break;
        }
        TokioWrite::write_all(&mut to, &buffer[..count])
            .await
            .map_err(|err| ProximaError::Upstream(format!("h2 relay write: {err}")))?;
        TokioWrite::flush(&mut to)
            .await
            .map_err(|err| ProximaError::Upstream(format!("h2 relay flush: {err}")))?;
        TokioWrite::write_all(&mut dump, &buffer[..count])
            .await
            .map_err(|err| ProximaError::Upstream(format!("h2 dump write: {err}")))?;
    }
    TokioWrite::flush(&mut dump).await.ok();
    Ok(())
}

async fn connect_upstream(
    host: &str,
    addr: &str,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, ProximaError> {
    let upstream = TcpStream::connect(addr)
        .await
        .map_err(|err| ProximaError::Upstream(format!("connect {addr}: {err}")))?;
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut client_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    // offer http/1.1 via ALPN: the h1 pipeline forwards h1 upstream, and
    // CDN-fronted endpoints close TLS connections that negotiate no protocol.
    client_config.alpn_protocols = vec![b"http/1.1".to_vec()];
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
    let server_name = rustls::pki_types::ServerName::try_from(host)
        .map_err(|err| ProximaError::Config(format!("server name: {err}")))?
        .to_owned();
    connector
        .connect(server_name, upstream)
        .await
        .map_err(|err| ProximaError::Upstream(format!("tls connect {addr}: {err}")))
}

async fn pipe_raw<R, W>(
    host: &str,
    addr: &str,
    mut client_reader: R,
    mut client_writer: W,
) -> Result<(), ProximaError>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let upstream_tls = connect_upstream(host, addr).await?;
    let (mut upstream_reader, mut upstream_writer) = tokio::io::split(upstream_tls);
    let up = async {
        let _ = tokio::io::copy(&mut client_reader, &mut upstream_writer).await;
        let _ = upstream_writer.shutdown().await;
    };
    let down = async {
        let _ = tokio::io::copy(&mut upstream_reader, &mut client_writer).await;
        let _ = client_writer.shutdown().await;
    };
    tokio::join!(up, down);
    Ok(())
}

/// Parse the `model` field from a JSON request body, or "" if absent/unparseable.
#[must_use]
pub fn parse_request_model(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|val| val.get("model").and_then(Value::as_str).map(String::from))
        .unwrap_or_default()
}

/// The slice of the full request body that still needs to be forwarded to the
/// upstream AFTER `first_buf` (which already carried the request head plus
/// whatever body bytes arrived with the first read). `header_boundary` is the
/// index of the `\r\n\r\n` in `first_buf`'s text; None means no head boundary
/// was seen (nothing to forward beyond first_buf). Off-by-one-prone — isolated
/// so it can be unit-tested.
#[must_use]
pub fn request_body_tail(
    first_buf_len: usize,
    header_boundary: Option<usize>,
    full_body: &[u8],
) -> &[u8] {
    let Some(boundary) = header_boundary else {
        return &[];
    };
    let body_in_first = first_buf_len.saturating_sub(boundary + 4);
    if body_in_first < full_body.len() {
        &full_body[body_in_first..]
    } else {
        &[]
    }
}

/// Decode a raw response body: undo chunked transfer-encoding then gzip
/// content-encoding, both driven by the response head. Real upstreams return
/// SSE/JSON gzip-compressed and often chunked; the proxy forwards those bytes
/// verbatim to the client (which decodes them) but can decode for its own
/// inspection. Returns the decoded bytes, or the input unchanged if neither
/// encoding applies / fails.
#[must_use]
pub fn decode_response_body(stream_buf: &[u8], head_wire: &[u8]) -> Vec<u8> {
    let head = String::from_utf8_lossy(head_wire);
    let dechunked = if header_value_contains(&head, "transfer-encoding", "chunked") {
        decode_chunked_bytes(stream_buf)
    } else {
        stream_buf.to_vec()
    };
    if header_value_contains(&head, "content-encoding", "gzip") {
        gunzip(&dechunked).unwrap_or(dechunked)
    } else if header_value_contains(&head, "content-encoding", "br") {
        brotli_decode(&dechunked).unwrap_or(dechunked)
    } else if header_value_contains(&head, "content-encoding", "zstd") {
        zstd_decode(&dechunked).unwrap_or(dechunked)
    } else {
        dechunked
    }
}

fn header_value_contains(head: &str, name: &str, needle: &str) -> bool {
    head.lines()
        .filter_map(|line| line.split_once(':'))
        .any(|(field, value)| {
            field.trim().eq_ignore_ascii_case(name)
                && value.trim().to_ascii_lowercase().contains(needle)
        })
}

/// Bytes-level chunked transfer-encoding decode (a gzipped body is not valid
/// UTF-8, so a str-based decode can't be used here).
fn decode_chunked_bytes(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    let mut pos = 0;
    while pos < raw.len() {
        // find end of the chunk-size line
        let Some(crlf) = find_subslice(&raw[pos..], b"\r\n") else {
            break;
        };
        let size_line = &raw[pos..pos + crlf];
        let size_str = String::from_utf8_lossy(size_line);
        let chunk_size = match usize::from_str_radix(size_str.trim(), 16) {
            Ok(size) => size,
            Err(_) => break,
        };
        if chunk_size == 0 {
            break;
        }
        let data_start = pos + crlf + 2;
        let data_end = data_start + chunk_size;
        if data_end > raw.len() {
            out.extend_from_slice(&raw[data_start..]);
            break;
        }
        out.extend_from_slice(&raw[data_start..data_end]);
        pos = data_end;
        if raw[pos..].starts_with(b"\r\n") {
            pos += 2;
        }
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn gunzip(raw: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let mut decoder = flate2::read::GzDecoder::new(raw);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok().map(|_| out)
}

/// Decode a `content-encoding: br` (brotli) body. Codex and several LLM APIs
/// negotiate brotli; without this the compressed bytes get extracted + stored
/// verbatim and surface as garbled "memory" text downstream.
fn brotli_decode(raw: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read as _;
    let mut out = Vec::new();
    brotli::Decompressor::new(raw, 4096)
        .read_to_end(&mut out)
        .ok()
        .map(|_| out)
}

/// Decode a `content-encoding: zstd` body (zstandard).
fn zstd_decode(raw: &[u8]) -> Option<Vec<u8>> {
    zstd::decode_all(raw).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod decode_body_tests {
    use super::decode_response_body;

    // a real assistant-turn shape (what a production caller hands the decoder).
    const PAYLOAD: &[u8] =
        b"{\"role\":\"assistant\",\"content\":\"hello from a real captured turn\"}";

    #[test]
    fn decodes_brotli_content_encoding() {
        use std::io::Write as _;
        let mut compressed = Vec::new();
        {
            let mut encoder = brotli::CompressorWriter::new(&mut compressed, 4096, 5, 22);
            encoder.write_all(PAYLOAD).unwrap();
        }
        let head = b"HTTP/1.1 200 OK\r\ncontent-encoding: br\r\n\r\n";
        assert_eq!(decode_response_body(&compressed, head), PAYLOAD);
    }

    #[test]
    fn decodes_zstd_content_encoding() {
        let compressed = zstd::encode_all(PAYLOAD, 3).unwrap();
        let head = b"HTTP/1.1 200 OK\r\ncontent-encoding: zstd\r\n\r\n";
        assert_eq!(decode_response_body(&compressed, head), PAYLOAD);
    }

    #[test]
    fn passes_through_unencoded_body() {
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n";
        assert_eq!(decode_response_body(PAYLOAD, head), PAYLOAD);
    }
}

#[derive(Debug, Default)]
pub struct ResponsePumpReport {
    pub is_sse: bool,
    pub response_head_wire: Vec<u8>,
    pub stream_buf: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct WsPumpReport {
    // read-only without intercept-capture; allow dead_code per Principle 14
    // (incumbent parity: type must always expose the field even when the
    // feature path doesn't consume it, so tests can still inspect it).
    #[allow(dead_code)]
    pub real_head_wire: Vec<u8>,
    pub delta_accum: String,
}

/// Accumulate the full HTTP/1.1 request body from `client_reader`.
///
/// The first read from the client has already happened — `first_buf` holds
/// those bytes and `first_text` is the lossy UTF-8 view used for header
/// parsing. We find the `\r\n\r\n` header boundary, parse `Content-Length`
/// case-insensitively (defaulting to 0), then read more from
/// `client_reader` until the body buffer reaches that length. Returns the
/// full body bytes (empty Vec if the header boundary is not present).
///
/// Behavior on transport errors (`Err`) or premature EOF (`Ok(0)`): break
/// the read loop and return whatever has accumulated so far. This is the
/// pre-existing inline behavior; the proxy is a pass-through so a partial
/// body simply forwards a shorter request to the upstream.
async fn accumulate_request_body<R>(
    client_reader: &mut R,
    first_buf: &[u8],
    first_text: &str,
) -> Vec<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let body_start = match first_text.find("\r\n\r\n") {
        Some(idx) => idx,
        None => return Vec::new(),
    };
    let headers = &first_text[..body_start];
    let content_length: usize = headers
        .lines()
        .find(|line| line.to_lowercase().starts_with("content-length:"))
        .and_then(|line| {
            line.split_once(':')
                .map(|(_, val)| val.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);

    let body_so_far = &first_buf[body_start + 4..];
    let mut body_buf = body_so_far.to_vec();

    while body_buf.len() < content_length {
        let mut more = vec![0u8; content_length - body_buf.len()];
        match tokio::io::AsyncReadExt::read(client_reader, &mut more).await {
            Ok(0) => break,
            Ok(count) => body_buf.extend_from_slice(&more[..count]),
            Err(_) => break,
        }
    }
    body_buf
}

/// Pump bytes from client to upstream during a WebSocket session.
///
/// Contract: `on_request_chunk` is called for every successfully-forwarded
/// chunk. On read or write error, the loop breaks WITHOUT signalling the
/// callback — callers should treat "pump returned" as "stream may have
/// ended abnormally" and not assume all bytes were forwarded if they need
/// that guarantee.
pub async fn pump_ws_client_to_upstream<R, W, F>(
    client_reader: &mut R,
    upstream_writer: &mut W,
    mut on_request_chunk: F,
) where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
    F: FnMut(&[u8]),
{
    let mut buf = [0u8; 16 * 1024];
    loop {
        let read = match tokio::io::AsyncReadExt::read(client_reader, &mut buf).await {
            Ok(0) => break,
            Ok(count) => count,
            Err(_) => break,
        };
        if tokio::io::AsyncWriteExt::write_all(upstream_writer, &buf[..read])
            .await
            .is_err()
        {
            break;
        }
        let _ = tokio::io::AsyncWriteExt::flush(upstream_writer).await;
        on_request_chunk(&buf[..read]);
    }
    let _ = tokio::io::AsyncWriteExt::shutdown(upstream_writer).await;
}

/// Pump bytes from upstream to client during a WebSocket session, returning
/// the captured 101 head wire and accumulated text-delta payloads.
///
/// Contract: `on_response_chunk` is called only for bytes AFTER the 101
/// header boundary (the head wire is reported separately). On read or
/// write error the loop breaks WITHOUT signalling the callback — callers
/// should treat "pump returned" as "stream may have ended abnormally".
pub async fn pump_ws_upstream_to_client<R, W, F>(
    upstream_reader: &mut R,
    client_writer: &mut W,
    mut on_response_chunk: F,
) -> WsPumpReport
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
    F: FnMut(&[u8]),
{
    let mut buf = [0u8; 16 * 1024];
    let mut ws_started = false;
    let mut real_head_wire: Vec<u8> = Vec::new();
    let mut delta_accum = String::new();
    let mut last_logged_len: usize = 0;
    let mut frame_buf: Vec<u8> = Vec::new();

    loop {
        let read = match tokio::io::AsyncReadExt::read(upstream_reader, &mut buf).await {
            Ok(0) => break,
            Ok(count) => count,
            Err(_) => break,
        };
        if tokio::io::AsyncWriteExt::write_all(client_writer, &buf[..read])
            .await
            .is_err()
        {
            break;
        }
        let _ = tokio::io::AsyncWriteExt::flush(client_writer).await;

        let push_start_at: usize = if ws_started {
            0
        } else if let Some(body_start) = buf[..read]
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
        {
            real_head_wire.extend_from_slice(&buf[..body_start + 4]);
            body_start + 4
        } else {
            real_head_wire.extend_from_slice(&buf[..read]);
            read
        };
        if push_start_at < read {
            on_response_chunk(&buf[push_start_at..read]);
        }

        if !ws_started {
            if buf[..read.min(15)].starts_with(b"HTTP/1.1 101") {
                ws_started = true;
                if let Some(body_start) = buf[..read]
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                {
                    frame_buf.extend_from_slice(&buf[body_start + 4..read]);
                }
            }
        } else {
            frame_buf.extend_from_slice(&buf[..read]);
        }

        let mut got_close = false;
        if ws_started && !frame_buf.is_empty() {
            let mut offset = 0;
            while offset < frame_buf.len() {
                match proxima_protocols::websocket_frame::parse_frame(&frame_buf[offset..]) {
                    Ok((frame, consumed)) => {
                        let mut payload = frame.payload.to_vec();
                        if let Some(mask) = frame.mask {
                            proxima_protocols::websocket_frame::unmask_in_place(&mut payload, mask);
                        }
                        match frame.opcode {
                            proxima_protocols::websocket_frame::Opcode::Text => {
                                // generic text-delta sniff: any json frame carrying a
                                // non-empty string `delta` field. vocab-agnostic — the
                                // bytes are forwarded verbatim regardless; this only
                                // feeds the human-readable [stream] log + delta_accum.
                                if let Ok(value) = serde_json::from_slice::<Value>(&payload)
                                    && let Some(delta) = value.get("delta").and_then(Value::as_str)
                                    && !delta.is_empty()
                                {
                                    delta_accum.push_str(delta);
                                    if delta.ends_with('\n')
                                        || delta.ends_with('.')
                                        || delta.ends_with('!')
                                        || delta.ends_with('?')
                                    {
                                        let new_content = &delta_accum[last_logged_len..];
                                        let trimmed = new_content.trim();
                                        if trimmed.len() > 20 {
                                            eprintln!("{} [stream] {trimmed}", ts());
                                        }
                                        last_logged_len = delta_accum.len();
                                    }
                                }
                            }
                            proxima_protocols::websocket_frame::Opcode::Close => {
                                got_close = true;
                            }
                            _ => {}
                        }
                        offset += consumed;
                    }
                    Err(_) => break,
                }
            }
            frame_buf.drain(..offset);
        }
        if got_close {
            break;
        }
    }
    let _ = tokio::io::AsyncWriteExt::shutdown(client_writer).await;
    WsPumpReport {
        real_head_wire,
        delta_accum,
    }
}

impl ResponsePumpReport {
    #[must_use]
    #[allow(dead_code)]
    pub fn total_body_bytes(&self) -> usize {
        self.stream_buf.len()
    }
}

/// How an HTTP/1.1 response body's length is determined (RFC 7230 §3.3.3).
enum BodyFraming {
    /// 1xx, 204, 304 — no body regardless of headers (rule 1).
    None,
    /// `Content-Length: n` (rule 5).
    ContentLength(usize),
    /// `Transfer-Encoding: chunked` — body ends at the zero-length chunk (rule 3).
    Chunked,
    /// Neither signal present — body is delimited by connection close (rule 7).
    UntilClose,
}

/// Classify a response body's framing from its head wire bytes. `Transfer-
/// Encoding: chunked` takes precedence over `Content-Length` per RFC 7230 §3.3.3.
fn classify_response_body(head_wire: &[u8]) -> BodyFraming {
    let head = String::from_utf8_lossy(head_wire);
    let status = head
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok());
    if matches!(status, Some(code) if (100..200).contains(&code) || code == 204 || code == 304) {
        return BodyFraming::None;
    }
    let mut content_length = None;
    for line in head.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            return BodyFraming::Chunked;
        }
        if let Some(value) = lower.strip_prefix("content-length:") {
            content_length = value.trim().parse::<usize>().ok();
        }
    }
    match content_length {
        Some(length) => BodyFraming::ContentLength(length),
        None => BodyFraming::UntilClose,
    }
}

/// Whether the accumulated body bytes constitute a complete response per its
/// framing. `UntilClose` is never complete until the socket EOFs (handled by
/// the read loop's `Ok(0)` arm).
fn response_body_complete(body: &[u8], framing: &BodyFraming) -> bool {
    match framing {
        BodyFraming::None => true,
        BodyFraming::ContentLength(length) => body.len() >= *length,
        BodyFraming::Chunked => body.ends_with(b"0\r\n\r\n"),
        BodyFraming::UntilClose => false,
    }
}

pub async fn pump_streaming_response<R, W, F>(
    upstream_reader: &mut R,
    client_writer: &mut W,
    mut on_response_chunk: F,
) -> Result<ResponsePumpReport, ProximaError>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
    F: FnMut(&[u8]),
{
    let mut response_header_buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let mut header_end_idx: Option<usize> = None;
    let mut scratch = [0u8; 16 * 1024];
    while header_end_idx.is_none() {
        let read = match tokio::io::AsyncReadExt::read(upstream_reader, &mut scratch).await {
            Ok(0) => break,
            Ok(count) => count,
            // upstreams that close without a TLS close_notify (common for
            // CDN-fronted servers) surface UnexpectedEof; treat it as a clean
            // end-of-stream and forward whatever head bytes arrived, matching
            // the body loop below.
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(ProximaError::Io(err)),
        };
        response_header_buf.extend_from_slice(&scratch[..read]);
        header_end_idx = response_header_buf
            .windows(4)
            .position(|window| window == b"\r\n\r\n");
    }

    let split_at = match header_end_idx {
        Some(idx) => idx + 4,
        None => response_header_buf.len(),
    };
    let response_head_wire = response_header_buf[..split_at].to_vec();
    let is_sse = response_is_sse(&response_head_wire);
    let body_already = response_header_buf[split_at..].to_vec();

    // HTTP/1.1 response framing (RFC 7230 §3.3.3). A keep-alive upstream does
    // NOT close after the response, so reading to EOF blocks forever; honor the
    // body's length signal to know when it's complete. Flush every write so the
    // client sees bytes as they arrive even when the upstream keeps the socket
    // open (the original code only flushed for SSE, so buffered keep-alive
    // responses never reached the client).
    let framing = classify_response_body(&response_head_wire);

    tokio::io::AsyncWriteExt::write_all(client_writer, &response_head_wire)
        .await
        .map_err(ProximaError::Io)?;
    let _ = tokio::io::AsyncWriteExt::flush(client_writer).await;

    let mut stream_buf: Vec<u8> = Vec::with_capacity(body_already.len());
    if !body_already.is_empty() {
        tokio::io::AsyncWriteExt::write_all(client_writer, &body_already)
            .await
            .map_err(ProximaError::Io)?;
        let _ = tokio::io::AsyncWriteExt::flush(client_writer).await;
        on_response_chunk(&body_already);
        stream_buf.extend_from_slice(&body_already);
    }

    while !response_body_complete(&stream_buf, &framing) {
        let read = match tokio::io::AsyncReadExt::read(upstream_reader, &mut scratch).await {
            Ok(0) => break,
            Ok(count) => count,
            Err(_) => break,
        };
        if tokio::io::AsyncWriteExt::write_all(client_writer, &scratch[..read])
            .await
            .is_err()
        {
            break;
        }
        let _ = tokio::io::AsyncWriteExt::flush(client_writer).await;
        on_response_chunk(&scratch[..read]);
        stream_buf.extend_from_slice(&scratch[..read]);
    }
    let _ = tokio::io::AsyncWriteExt::flush(client_writer).await;

    Ok(ResponsePumpReport {
        is_sse,
        response_head_wire,
        stream_buf,
    })
}

/// Returns true if a captured HTTP/1.1 request looks like a WebSocket
/// upgrade per RFC 6455 §4.1. We only inspect the `Upgrade:` header
/// case-insensitively; a fully-correct check also validates `Connection:
/// upgrade` and `Sec-WebSocket-Version`, but the proxy is a passthrough
/// — we forward whatever the client sent — so this looser sniff just
/// chooses the per-protocol branch and any malformed-upgrade case is
/// rejected by the upstream server.
fn request_is_websocket_upgrade(request_text: &str) -> bool {
    request_text.lines().any(|line| {
        line.split_once(':').is_some_and(|(name, value)| {
            name.trim().eq_ignore_ascii_case("upgrade")
                && value.trim().eq_ignore_ascii_case("websocket")
        })
    })
}

fn response_is_sse(head_wire: &[u8]) -> bool {
    let text = match std::str::from_utf8(head_wire) {
        Ok(text_str) => text_str,
        Err(_) => return false,
    };
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .any(|(name, value)| {
            name.trim().eq_ignore_ascii_case("content-type")
                && value
                    .trim()
                    .to_ascii_lowercase()
                    .contains("text/event-stream")
        })
}

/// Split a CONNECT target (`host:port`) into host and port, defaulting to 443
/// when no port is present or the port is unparseable. Splits on the LAST
/// colon so bare IPv6 literals without a port keep their colons in the host
/// (a bracketed `[::1]:443` still splits correctly on the final colon).
#[must_use]
pub fn parse_connect_target(path: &str) -> (&str, u16) {
    match path.rsplit_once(':') {
        Some((host, port_str)) => {
            let port = port_str.parse().unwrap_or(443);
            (host, port)
        }
        None => (path, 443),
    }
}

/// Telemetry endpoints are raw-piped (not MITM-intercepted or captured): we
/// don't decrypt/record telemetry, just transparently forward it. Most carry
/// "telemetry" in the hostname (a generic, cross-cutting rule). Exceptions whose
/// hostname doesn't are supplied per-deployment via
/// `PROXIMA_INTERCEPT_TELEMETRY_HOSTS` (comma-separated) — no vendor domain is
/// compiled in.
#[must_use]
pub fn is_telemetry_host(host: &str) -> bool {
    host.contains("telemetry")
        || std::env::var("PROXIMA_INTERCEPT_TELEMETRY_HOSTS")
            .is_ok_and(|list| host_in_list(host, &list))
}

struct FuturesIoToTokio {
    inner: Box<dyn proxima_primitives::pipe::upgrade::HijackStream>,
    leftover: Vec<u8>,
    leftover_pos: usize,
}

impl FuturesIoToTokio {
    fn new(stream: Box<dyn proxima_primitives::pipe::upgrade::HijackStream>, leftover: Bytes) -> Self {
        Self {
            inner: stream,
            leftover: leftover.to_vec(),
            leftover_pos: 0,
        }
    }
}

impl tokio::io::AsyncRead for FuturesIoToTokio {
    fn poll_read(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        if self.leftover_pos < self.leftover.len() {
            let remaining = &self.leftover[self.leftover_pos..];
            let to_copy = remaining.len().min(buf.remaining());
            buf.put_slice(&remaining[..to_copy]);
            self.leftover_pos += to_copy;
            return std::task::Poll::Ready(Ok(()));
        }
        let inner = Pin::new(&mut *self.inner);
        match futures::io::AsyncRead::poll_read(inner, ctx, buf.initialize_unfilled()) {
            std::task::Poll::Ready(Ok(count)) => {
                buf.advance(count);
                std::task::Poll::Ready(Ok(()))
            }
            std::task::Poll::Ready(Err(err)) => std::task::Poll::Ready(Err(err)),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl tokio::io::AsyncWrite for FuturesIoToTokio {
    fn poll_write(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        let inner = Pin::new(&mut *self.inner);
        futures::io::AsyncWrite::poll_write(inner, ctx, buf)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        let inner = Pin::new(&mut *self.inner);
        futures::io::AsyncWrite::poll_flush(inner, ctx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        let inner = Pin::new(&mut *self.inner);
        futures::io::AsyncWrite::poll_close(inner, ctx)
    }
}

pub struct InterceptPipeFactory {
    /// Shared spigot the capture terminal is armed from at serve. Empty until
    /// the App turns it on; capture built here stays inert until then.
    // only the capture path consumes it; without that feature there is no sink.
    #[cfg_attr(not(feature = "intercept-capture"), allow(dead_code))]
    spigot: proxima_recording::pipe::DeferredRuntime,
}

impl InterceptPipeFactory {
    #[must_use]
    pub fn new(spigot: proxima_recording::pipe::DeferredRuntime) -> Self {
        Self { spigot }
    }
}

impl PipeFactory for InterceptPipeFactory {
    fn name(&self) -> &str {
        "intercept"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>>
    {
        let spec = spec.clone();
        #[cfg(feature = "intercept-config")]
        let spigot = Arc::clone(&self.spigot);
        Box::pin(async move {
            #[cfg(feature = "intercept-config")]
            {
                let config: crate::config::InterceptConfig = serde_json::from_value(spec)
                    .map_err(|err| ProximaError::Config(format!("intercept config: {err}")))?;
                config.into_handle(spigot)
            }
            #[cfg(not(feature = "intercept-config"))]
            {
                build_from_value(&spec, self)
            }
        })
    }
}

/// Hand-parsed fallback for builds without the `intercept-config` feature (which
/// brings in the serde derive / bon / conflaguration deps the typed
/// [`crate::config::InterceptConfig`] needs). Kept byte-for-byte equivalent to
/// the typed lowering: `(Some, Some)` CA paths load a persistent CA, otherwise
/// ephemeral; a `capture` block rejects `..` traversal and requires `data_path`.
#[cfg(not(feature = "intercept-config"))]
fn build_from_value(
    spec: &Value,
    factory: &InterceptPipeFactory,
) -> Result<PipeHandle, ProximaError> {
    let _ = factory;
    let ca_cert = spec
        .get("ca_cert")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let ca_key = spec
        .get("ca_key")
        .and_then(Value::as_str)
        .map(PathBuf::from);

    #[allow(unused_mut)]
    let mut pipe = match (ca_cert, ca_key) {
        (Some(cert_path), Some(key_path)) => InterceptPipe::with_ca_files(&cert_path, &key_path)?,
        _ => InterceptPipe::with_generated_ca()?,
    };

    #[cfg(feature = "intercept-capture")]
    if let Some(capture_value) = spec.get("capture") {
        let data_path = capture_value
            .get("data_path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| {
                ProximaError::Config(
                    "intercept spec capture.data_path is required when capture is set".into(),
                )
            })?;
        if data_path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(ProximaError::Config(
                "intercept spec capture.data_path must not contain `..` components".into(),
            ));
        }
        let capture = crate::capture::Capture::open(&data_path, Arc::clone(&factory.spigot))?;
        pipe = pipe.with_capture(capture);
    }

    #[cfg(not(feature = "intercept-capture"))]
    if spec.get("capture").is_some() {
        eprintln!(
            "{} [intercept-warn] spec has capture key but intercept-capture feature is not compiled in; capture is silently disabled",
            ts()
        );
    }

    Ok(into_handle(pipe))
}

pub fn factory_arc(spigot: proxima_recording::pipe::DeferredRuntime) -> Arc<dyn PipeFactory> {
    Arc::new(InterceptPipeFactory::new(spigot))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod response_sniff_tests {
    use super::response_is_sse;

    #[test]
    fn detects_text_event_stream_content_type() {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n";
        assert!(response_is_sse(wire));
    }

    #[test]
    fn detects_text_event_stream_with_charset_suffix() {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\r\n";
        assert!(response_is_sse(wire));
    }

    #[test]
    fn detects_case_insensitive_header_and_value() {
        let wire = b"HTTP/1.1 200 OK\r\nCONTENT-TYPE: TEXT/EVENT-STREAM\r\n\r\n";
        assert!(response_is_sse(wire));
    }

    #[test]
    fn rejects_json_response() {
        let wire =
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n";
        assert!(!response_is_sse(wire));
    }

    #[test]
    fn rejects_no_content_type() {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        assert!(!response_is_sse(wire));
    }

    #[test]
    fn handles_non_utf8_gracefully() {
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: \xff\xfe binary\r\n\r\n";
        assert!(!response_is_sse(wire));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod request_sniff_tests {
    use super::request_is_websocket_upgrade;

    #[test]
    fn detects_canonical_upgrade_header() {
        let request = "GET /chat HTTP/1.1\r\nHost: example.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
        assert!(request_is_websocket_upgrade(request));
    }

    #[test]
    fn detects_lowercase_upgrade_header_name() {
        let request = "GET /chat HTTP/1.1\r\nupgrade: websocket\r\nconnection: Upgrade\r\n\r\n";
        assert!(request_is_websocket_upgrade(request));
    }

    #[test]
    fn detects_mixed_case_websocket_value() {
        let request = "GET /chat HTTP/1.1\r\nUpgrade: WebSocket\r\n\r\n";
        assert!(request_is_websocket_upgrade(request));
    }

    #[test]
    fn rejects_post_request_without_upgrade() {
        let request =
            "POST /responses HTTP/1.1\r\nHost: example.com\r\nContent-Length: 5\r\n\r\nhello";
        assert!(!request_is_websocket_upgrade(request));
    }

    #[test]
    fn rejects_upgrade_to_non_websocket_protocol() {
        let request = "GET /chat HTTP/1.1\r\nUpgrade: h2c\r\n\r\n";
        assert!(!request_is_websocket_upgrade(request));
    }

    #[test]
    fn rejects_user_agent_string_that_contains_substring_websocket() {
        // earlier impl used a naive substring match (`contains("Upgrade: websocket")`)
        // which would correctly reject this but only by accident. the new
        // line-by-line check is robust against the body containing the same string.
        let request = "POST /responses HTTP/1.1\r\nUser-Agent: my-client/1.0 Upgrade: websocket faker\r\n\r\n";
        assert!(!request_is_websocket_upgrade(request));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod request_forward_tests {
    use super::{is_telemetry_host, parse_connect_target, parse_request_model, request_body_tail};

    #[test]
    fn connect_target_splits_host_and_port() {
        assert_eq!(
            parse_connect_target("api.example.com:443"),
            ("api.example.com", 443)
        );
        assert_eq!(
            parse_connect_target("example.com:8080"),
            ("example.com", 8080)
        );
    }

    #[test]
    fn connect_target_defaults_443_without_port() {
        assert_eq!(
            parse_connect_target("api.example.com"),
            ("api.example.com", 443)
        );
    }

    #[test]
    fn connect_target_defaults_443_on_unparseable_port() {
        assert_eq!(parse_connect_target("host:notaport"), ("host", 443));
    }

    #[test]
    fn connect_target_splits_on_last_colon_for_bracketed_ipv6() {
        // bracketed IPv6 with a port splits on the final colon, host keeps brackets
        assert_eq!(parse_connect_target("[::1]:8443"), ("[::1]", 8443));
    }

    #[test]
    fn telemetry_host_detected() {
        assert!(is_telemetry_host("telemetry.individual.example.com"));
        assert!(is_telemetry_host("telemetry.example.com"));
    }

    #[test]
    fn non_telemetry_hosts_intercepted() {
        assert!(!is_telemetry_host("api.example.com"));
        assert!(!is_telemetry_host("api.individual.example.com"));
    }

    #[test]
    fn host_in_list_matches_deployment_supplied_domains() {
        // routed domains live in deployment config, not source; substring match
        let configured = "metrics.example.com, edge.example.net, api2.example.org";
        assert!(super::host_in_list("metrics.example.com", configured));
        assert!(super::host_in_list("api2.example.org", configured));
        assert!(!super::host_in_list("api.example.io", configured));
        assert!(!super::host_in_list("metrics.example.com", ""));
    }

    #[test]
    fn parse_model_from_json_body() {
        assert_eq!(
            parse_request_model(r#"{"model":"model-nano","x":1}"#),
            "model-nano"
        );
        assert_eq!(
            parse_request_model(r#"{"model":"model-opus"}"#),
            "model-opus"
        );
    }

    #[test]
    fn parse_model_empty_when_absent_or_unparseable() {
        assert_eq!(parse_request_model(r#"{"no_model":true}"#), "");
        assert_eq!(parse_request_model("not json"), "");
        assert_eq!(parse_request_model(""), "");
    }

    #[test]
    fn tail_none_when_no_header_boundary() {
        // no \r\n\r\n seen → nothing to forward beyond first_buf
        assert_eq!(request_body_tail(100, None, b"the whole body"), b"");
    }

    #[test]
    fn tail_is_body_beyond_what_first_buf_carried() {
        // first_buf = 50 bytes; header boundary at index 30 → head+CRLFCRLF =
        // 34 bytes, so 16 body bytes already arrived in first_buf. the tail is
        // the remainder of the full body past those 16.
        let full_body = b"0123456789abcdefXXXXXXtail-bytes";
        let tail = request_body_tail(50, Some(30), full_body);
        assert_eq!(tail, &full_body[16..]);
    }

    #[test]
    fn tail_empty_when_first_buf_already_held_whole_body() {
        // body_in_first (20) >= full_body.len() (11) → nothing left to send
        let full_body = b"short-body!!";
        let tail = request_body_tail(54, Some(30), full_body); // 54-(30+4)=20
        assert_eq!(tail, b"");
    }

    #[test]
    fn tail_is_entire_body_when_first_buf_was_headers_only() {
        // first_buf length == boundary+4 → 0 body bytes in first_buf → tail is all
        let full_body = b"entire request body";
        let tail = request_body_tail(34, Some(30), full_body);
        assert_eq!(tail, full_body);
    }

    #[test]
    fn tail_saturates_when_first_buf_shorter_than_boundary() {
        // defensive: a malformed boundary larger than first_buf_len must not
        // underflow; saturating_sub yields 0 body-in-first → whole body is tail
        let full_body = b"body";
        let tail = request_body_tail(10, Some(40), full_body);
        assert_eq!(tail, full_body);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pump_streaming_tests {
    use super::pump_streaming_response;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tokio::io::AsyncWriteExt;

    type ChunkLog = Rc<RefCell<Vec<Vec<u8>>>>;

    fn capture_chunks() -> (ChunkLog, impl FnMut(&[u8])) {
        let captured: ChunkLog = Rc::new(RefCell::new(Vec::new()));
        let captured_clone = Rc::clone(&captured);
        let push = move |chunk: &[u8]| {
            captured_clone.borrow_mut().push(chunk.to_vec());
        };
        (captured, push)
    }

    #[proxima::test(runtime = "tokio")]
    async fn post_response_captures_full_body_in_one_chunk_when_arrived_intact() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        upstream_writer.write_all(wire).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        let report = pump_streaming_response(&mut upstream_reader, &mut client_writer, push)
            .await
            .expect("pump");
        client_writer.shutdown().await.unwrap();

        assert!(!report.is_sse);
        assert!(report.response_head_wire.ends_with(b"\r\n\r\n"));
        assert_eq!(report.stream_buf, b"{\"ok\":true}".to_vec());
        {
            let chunks = captured.borrow();
            assert_eq!(chunks.len(), 1);
            assert_eq!(&chunks[0], b"{\"ok\":true}");
        }

        let mut forwarded = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut client_reader, &mut forwarded)
            .await
            .unwrap();
        assert_eq!(forwarded, wire);
    }

    #[proxima::test(runtime = "tokio")]
    async fn sse_response_captures_each_event_separately_as_arrived() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);
        let head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n";
        upstream_writer.write_all(head).await.unwrap();
        let events = [
            &b"data: {\"delta\":\"hello\"}\n\n"[..],
            &b"data: {\"delta\":\" world\"}\n\n"[..],
            &b"data: [DONE]\n\n"[..],
        ];
        for event in events {
            upstream_writer.write_all(event).await.unwrap();
            upstream_writer.flush().await.unwrap();
            tokio::task::yield_now().await;
        }
        upstream_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        let report = pump_streaming_response(&mut upstream_reader, &mut client_writer, push)
            .await
            .expect("pump");

        assert!(report.is_sse, "is_sse must detect text/event-stream");
        let chunks = captured.borrow();
        let total_captured: usize = chunks.iter().map(Vec::len).sum();
        let expected_body: usize = events.iter().map(|event| event.len()).sum();
        assert_eq!(total_captured, expected_body);
        assert!(!chunks.is_empty(), "must capture at least one chunk");
        // bytes-per-chunk shape depends on tcp segmentation; the substantive
        // claim is that every event byte appears in the captured stream.
        let joined: Vec<u8> = chunks.iter().flatten().copied().collect();
        for event in events {
            assert!(
                joined.windows(event.len()).any(|window| window == event),
                "missing sse event in capture"
            );
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_handshake_response_routes_through_same_pump_with_is_sse_false() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);
        let wire = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: abc\r\n\r\n";
        upstream_writer.write_all(wire).await.unwrap();
        let ws_frame_payload = b"\x81\x10hello-from-server";
        upstream_writer.write_all(ws_frame_payload).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        let report = pump_streaming_response(&mut upstream_reader, &mut client_writer, push)
            .await
            .expect("pump");

        assert!(!report.is_sse, "ws handshake response is_sse must be false");
        let chunks = captured.borrow();
        assert!(
            !chunks.is_empty(),
            "ws frame payload must be captured as a chunk"
        );
        let joined: Vec<u8> = chunks.iter().flatten().copied().collect();
        assert!(
            joined
                .windows(ws_frame_payload.len())
                .any(|window| window == ws_frame_payload),
            "ws frame bytes must appear in captured chunks"
        );
    }

    // the spigot is built disarmed: factory.build wires capture but opens no
    // file until the App turns the spigot on at serve.
    #[cfg(feature = "intercept-capture")]
    #[proxima::test]
    async fn factory_build_with_capture_spec_stays_inert_until_armed() {
        use proxima_primitives::pipe::pipe_factory::PipeFactory;
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("from-spec.bin");
        let spec = serde_json::json!({
            "capture": {
                "data_path": data_path.to_string_lossy().to_string(),
            }
        });
        let factory = super::InterceptPipeFactory::new(proxima_recording::pipe::deferred_runtime());
        let _handle = factory.build(&spec, None).await.expect("build");
        assert!(
            !data_path.exists(),
            "disarmed capture must open no file at build"
        );
    }

    #[proxima::test]
    async fn factory_build_without_capture_spec_skips_capture() {
        use proxima_primitives::pipe::pipe_factory::PipeFactory;
        let spec = serde_json::json!({});
        let factory = super::InterceptPipeFactory::new(proxima_recording::pipe::deferred_runtime());
        let _handle = factory.build(&spec, None).await.expect("build");
        // no assertion beyond no-error — pipe is built without capture
    }

    #[cfg(feature = "intercept-capture")]
    #[proxima::test]
    async fn factory_build_with_capture_spec_rejects_parent_dir_traversal() {
        use proxima_primitives::pipe::pipe_factory::PipeFactory;
        let spec = serde_json::json!({
            "capture": { "data_path": "/var/data/../../etc/recording.bin" }
        });
        let factory = super::InterceptPipeFactory::new(proxima_recording::pipe::deferred_runtime());
        let result = factory.build(&spec, None).await;
        assert!(result.is_err(), "path with `..` must be rejected");
        if let Err(err) = result {
            let message = format!("{err}");
            assert!(
                message.contains(".."),
                "error should mention `..`, got: {message}"
            );
        }
    }

    #[cfg(feature = "intercept-capture")]
    #[proxima::test]
    async fn factory_build_with_capture_spec_propagates_unwritable_parent() {
        use proxima_primitives::pipe::pipe_factory::PipeFactory;
        let spec = serde_json::json!({
            "capture": { "data_path": "/nonexistent-parent-dir-xyz/recording.bin" }
        });
        let factory = super::InterceptPipeFactory::new(proxima_recording::pipe::deferred_runtime());
        let result = factory.build(&spec, None).await;
        assert!(
            result.is_err(),
            "missing parent dir must propagate as error"
        );
    }

    #[cfg(feature = "intercept-capture")]
    #[proxima::test]
    async fn factory_build_with_capture_spec_missing_data_path_errors() {
        use proxima_primitives::pipe::pipe_factory::PipeFactory;
        let spec = serde_json::json!({ "capture": {} });
        let factory = super::InterceptPipeFactory::new(proxima_recording::pipe::deferred_runtime());
        let result = factory.build(&spec, None).await;
        assert!(result.is_err(), "missing data_path must error");
        if let Err(err) = result {
            let message = format!("{err}");
            assert!(
                message.contains("data_path"),
                "error should mention data_path, got: {message}"
            );
        }
    }

    #[proxima::test]
    async fn accumulate_request_body_returns_full_body_when_in_first_buf() {
        let wire = b"POST /responses HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        let first_text = String::from_utf8_lossy(wire).to_string();
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        client_writer.shutdown().await.unwrap();
        let body = super::accumulate_request_body(&mut client_reader, wire, &first_text).await;
        assert_eq!(body, b"{\"ok\":true}");
    }

    #[proxima::test]
    async fn accumulate_request_body_reads_more_until_content_length_satisfied() {
        let head =
            b"POST /responses HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 20\r\n\r\n";
        let body_partial = b"hello";
        let mut wire = head.to_vec();
        wire.extend_from_slice(body_partial);
        let first_text = String::from_utf8_lossy(&wire).to_string();

        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        let trailing: &[u8] = b"-world-bytes!!!";
        client_writer.write_all(trailing).await.unwrap();
        client_writer.shutdown().await.unwrap();

        let body = super::accumulate_request_body(&mut client_reader, &wire, &first_text).await;
        assert_eq!(body, b"hello-world-bytes!!!");
        assert_eq!(body.len(), 20);
    }

    #[proxima::test]
    async fn accumulate_request_body_empty_when_no_header_boundary() {
        let wire = b"POST /responses HTTP/1.1\r\nContent-Length: 5\r\n"; // truncated, no \r\n\r\n
        let first_text = String::from_utf8_lossy(wire).to_string();
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        client_writer.shutdown().await.unwrap();
        let body = super::accumulate_request_body(&mut client_reader, wire, &first_text).await;
        assert!(body.is_empty(), "no header boundary must return empty body");
    }

    #[proxima::test]
    async fn accumulate_request_body_handles_premature_eof_gracefully() {
        let head = b"POST /responses HTTP/1.1\r\nContent-Length: 100\r\n\r\n";
        let first_text = String::from_utf8_lossy(head).to_string();
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        // upstream sends nothing and disconnects
        client_writer.shutdown().await.unwrap();
        let body = super::accumulate_request_body(&mut client_reader, head, &first_text).await;
        assert!(
            body.is_empty(),
            "premature eof must return what we have, not block"
        );
    }

    #[proxima::test]
    async fn accumulate_request_body_content_length_is_case_insensitive() {
        let wire = b"POST /x HTTP/1.1\r\nCONTENT-LENGTH: 5\r\n\r\nhello";
        let first_text = String::from_utf8_lossy(wire).to_string();
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        client_writer.shutdown().await.unwrap();
        let body = super::accumulate_request_body(&mut client_reader, wire, &first_text).await;
        assert_eq!(body, b"hello");
    }

    #[proxima::test]
    async fn accumulate_request_body_default_zero_when_content_length_missing() {
        let wire = b"GET /x HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let first_text = String::from_utf8_lossy(wire).to_string();
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        client_writer.shutdown().await.unwrap();
        let body = super::accumulate_request_body(&mut client_reader, wire, &first_text).await;
        assert!(
            body.is_empty(),
            "missing content-length must return empty body"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_pump_client_to_upstream_forwards_and_captures_every_chunk() {
        use tokio::io::AsyncWriteExt;
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);

        let frames = [
            &b"\x82\x05hello"[..],
            &b"\x82\x05world"[..],
            &b"\x82\x06!!ok!!"[..],
        ];
        for frame in frames {
            client_writer.write_all(frame).await.unwrap();
        }
        client_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        super::pump_ws_client_to_upstream(&mut client_reader, &mut upstream_writer, push).await;

        let mut forwarded = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut upstream_reader, &mut forwarded)
            .await
            .unwrap();
        let expected: Vec<u8> = frames.iter().flat_map(|f| f.iter().copied()).collect();
        assert_eq!(
            forwarded, expected,
            "client→upstream pump must forward every byte"
        );
        let chunks = captured.borrow();
        let captured_joined: Vec<u8> = chunks.iter().flatten().copied().collect();
        assert_eq!(
            captured_joined, expected,
            "client→upstream pump must capture every byte"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_pump_captures_real_101_head_and_post_headers_chunks() {
        use tokio::io::AsyncWriteExt;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);

        let real_101 = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: real-accept-value-here\r\n\r\n";
        upstream_writer.write_all(real_101).await.unwrap();
        // single small close frame: server-side, opcode 0x88, length 0
        upstream_writer.write_all(&[0x88, 0x00]).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        let report =
            super::pump_ws_upstream_to_client(&mut upstream_reader, &mut client_writer, push).await;

        assert!(
            !report.real_head_wire.is_empty(),
            "real_head_wire must capture the 101+headers"
        );
        assert!(
            report.real_head_wire.ends_with(b"\r\n\r\n"),
            "real_head_wire must end at the header boundary"
        );
        let head_text = std::str::from_utf8(&report.real_head_wire).unwrap();
        assert!(head_text.contains("Sec-WebSocket-Accept: real-accept-value-here"));
        // the 101 head should NOT appear in the captured response chunks
        let chunks = captured.borrow();
        let joined: Vec<u8> = chunks.iter().flatten().copied().collect();
        assert!(
            !joined
                .windows(real_101.len())
                .any(|window| window == real_101),
            "real_head_wire bytes must not duplicate into response chunks"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_pump_captures_101_header_split_across_multiple_reads() {
        use tokio::io::AsyncWriteExt;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);

        // split the 101 + headers across three writes with explicit flush + yield
        // between them. tokio::io::duplex respects write ordering when the reader
        // is awaited mid-stream, so the pump sees three distinct reads.
        let chunks = [
            &b"HTTP/1.1 101 Switching Protocols\r\n"[..],
            &b"Upgrade: websocket\r\nConnection: Upgrade\r\n"[..],
            &b"Sec-WebSocket-Accept: split-accept\r\n\r\n"[..],
        ];
        for piece in chunks {
            upstream_writer.write_all(piece).await.unwrap();
            upstream_writer.flush().await.unwrap();
            tokio::task::yield_now().await;
        }
        // close frame to terminate the pump
        upstream_writer.write_all(&[0x88, 0x00]).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let (captured, push) = capture_chunks();
        let report =
            super::pump_ws_upstream_to_client(&mut upstream_reader, &mut client_writer, push).await;

        let expected_head: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(
            report.real_head_wire, expected_head,
            "real_head_wire must accumulate all 101 + headers across N reads"
        );
        let head_text = std::str::from_utf8(&report.real_head_wire).unwrap();
        assert!(head_text.contains("Sec-WebSocket-Accept: split-accept"));
        // header bytes must NOT leak into ResponseChunk captures
        let response_chunks = captured.borrow();
        let captured_joined: Vec<u8> = response_chunks.iter().flatten().copied().collect();
        for piece in chunks {
            assert!(
                !captured_joined
                    .windows(piece.len())
                    .any(|window| window == piece),
                "header bytes must not appear in captured chunks"
            );
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_pump_accumulates_text_delta_payloads_across_frames() {
        use tokio::io::AsyncWriteExt;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);

        // build a json text-delta payload and frame it as a server text frame.
        // server text frame: 0x81 (FIN + text), then 7-bit length (no mask).
        let payload = br#"{"delta":"hi there.","type":"response.output_text.delta"}"#;
        let mut text_frame = vec![0x81u8, payload.len() as u8];
        text_frame.extend_from_slice(payload);
        let close_frame = [0x88u8, 0x00];

        upstream_writer.write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").await.unwrap();
        upstream_writer.write_all(&text_frame).await.unwrap();
        upstream_writer.write_all(&close_frame).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let (_captured, push) = capture_chunks();
        let report =
            super::pump_ws_upstream_to_client(&mut upstream_reader, &mut client_writer, push).await;

        assert!(
            report.delta_accum.contains("hi there."),
            "delta_accum must include the parsed text delta, got: {}",
            report.delta_accum
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn ws_pump_stops_on_close_frame_received_from_upstream() {
        use std::time::Duration;
        use tokio::io::AsyncWriteExt;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);

        upstream_writer
            .write_all(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
            .await
            .unwrap();
        upstream_writer.write_all(&[0x88, 0x00]).await.unwrap();
        // intentionally do not shutdown the writer — pump must exit on Close frame, not on EOF
        let (_captured, push) = capture_chunks();
        let pump =
            super::pump_ws_upstream_to_client(&mut upstream_reader, &mut client_writer, push);
        let result = tokio::time::timeout(Duration::from_secs(2), pump).await;
        assert!(
            result.is_ok(),
            "ws pump must stop on Close opcode, not block waiting for EOF"
        );
    }

    #[proxima::test]
    async fn pump_with_noop_closure_when_capture_disabled_still_forwards() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, mut client_reader) = tokio::io::duplex(64 * 1024);
        let wire = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        upstream_writer.write_all(wire).await.unwrap();
        upstream_writer.shutdown().await.unwrap();

        let report = pump_streaming_response(&mut upstream_reader, &mut client_writer, |_| {})
            .await
            .expect("pump");
        client_writer.shutdown().await.unwrap();

        assert_eq!(report.stream_buf, b"hello".to_vec());
        let mut forwarded = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut client_reader, &mut forwarded)
            .await
            .unwrap();
        assert_eq!(forwarded, wire);
    }

    // regression: a keep-alive upstream (api.anthropic.com) sends the response
    // then holds the socket open. Reading to EOF would block forever; the pump
    // must return at the Content-Length boundary. This is the bug that hung the
    // claude-CLI wrap.
    #[proxima::test]
    async fn keep_alive_content_length_completes_without_upstream_close() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);
        let wire = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
        upstream_writer.write_all(wire).await.unwrap();
        upstream_writer.flush().await.unwrap();

        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            pump_streaming_response(&mut upstream_reader, &mut client_writer, |_| {}),
        )
        .await
        .expect("pump must return via Content-Length, not block on the open keep-alive socket")
        .expect("pump");

        assert_eq!(report.stream_buf, b"{\"ok\":true}".to_vec());
        drop(upstream_writer);
    }

    // regression: anthropic streams the Messages SSE as chunked transfer-encoding
    // over a kept-alive socket; the pump must return at the zero-length chunk
    // terminator rather than waiting for a close.
    #[proxima::test]
    async fn keep_alive_chunked_completes_at_terminator_without_upstream_close() {
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let (mut client_writer, _client_reader) = tokio::io::duplex(64 * 1024);
        let wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
        upstream_writer.write_all(wire).await.unwrap();
        upstream_writer.flush().await.unwrap();

        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            pump_streaming_response(&mut upstream_reader, &mut client_writer, |_| {}),
        )
        .await
        .expect(
            "pump must return at the chunked terminator, not block on the open keep-alive socket",
        )
        .expect("pump");

        assert!(report.is_sse, "text/event-stream is SSE");
        assert!(
            report.stream_buf.ends_with(b"0\r\n\r\n"),
            "body ends at the zero chunk"
        );
        drop(upstream_writer);
    }

    #[test]
    fn classify_204_no_content_has_no_body() {
        assert!(matches!(
            super::classify_response_body(b"HTTP/1.1 204 No Content\r\nDate: x\r\n\r\n"),
            super::BodyFraming::None
        ));
    }

    #[test]
    fn classify_chunked_wins_over_content_length() {
        assert!(matches!(
            super::classify_response_body(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n"
            ),
            super::BodyFraming::Chunked
        ));
    }

    #[test]
    fn classify_no_length_signal_is_until_close() {
        assert!(matches!(
            super::classify_response_body(b"HTTP/1.1 200 OK\r\nServer: x\r\n\r\n"),
            super::BodyFraming::UntilClose
        ));
    }
}
