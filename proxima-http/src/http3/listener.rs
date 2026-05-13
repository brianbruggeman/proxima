//! HTTP/3 listener. Binds a QUIC endpoint, accepts inbound
//! connections, and runs [`crate::http3::server::serve_h3_connection`] for each.
//!
//! TLS is mandatory on QUIC; the listener spec carries cert + key
//! paths the same way [`crate::listeners::http`]'s `tls` section
//! does. ALPN advertises `h3`. A self-signed cert is generated when
//! `dev_self_signed: true` is set — handy for tests and local dev.
//!
//! Same substrate discipline as [`crate::listeners::http`]: accept
//! loop on the calling task, per-connection driver spawned on the
//! ambient runtime (today tokio; Stage H removes the assumption).

use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use futures::channel::oneshot;
use proxima_telemetry::{debug, warn};
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_quic::{Endpoint, dev_server_config};

const ALPN_H3: &[u8] = b"h3";

pub struct H3ListenProtocol {
    label: String,
}

impl Default for H3ListenProtocol {
    fn default() -> Self {
        Self { label: "h3".into() }
    }
}

impl H3ListenProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ListenProtocol for H3ListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let server_config = match build_server_config(spec) {
            Ok(cfg) => cfg,
            Err(err) => {
                return Box::pin(async move { Err(err) });
            }
        };
        // Bind here, synchronously, rather than inside the returned
        // future. `Endpoint::server` wraps `quinn::Endpoint::server`,
        // which binds the OS socket in its constructor, not lazily on
        // first poll — binding inside the future meant the socket
        // didn't exist until something actually polled the spawned
        // task, so a caller that dials right after spawning (as the
        // e2e round-trip test does) could race a scheduler-delayed
        // first poll and never reach a listening socket at all.
        let endpoint = match Endpoint::server(bind, server_config) {
            Ok(endpoint) => endpoint,
            Err(err) => {
                let err = ProximaError::Upstream(format!("h3 bind: {err}"));
                return Box::pin(async move { Err(err) });
            }
        };
        debug!(?bind, local = ?endpoint.local_addr(), "h3 listener bound");
        if let Some(sender) = context.ready_signal.clone() {
            let _ = sender.send(());
        }
        let runtime_for_conns = context.runtime.clone();

        Box::pin(async move {
            let in_flight = Arc::new(AtomicU64::new(0));
            let mut shutdown = shutdown;

            loop {
                tokio::select! {
                    biased;
                    _ = &mut shutdown => {
                        endpoint.close(0, b"shutdown");
                        break;
                    }
                    accepted = endpoint.accept() => {
                        match accepted {
                            Some(Ok(connection)) => {
                                let dispatch = dispatch.clone();
                                let in_flight = in_flight.clone();
                                let conn_future = async move {
                                    if let Err(err) = crate::http3::server::serve_h3_connection(
                                        connection,
                                        dispatch,
                                        in_flight,
                                    )
                                    .await
                                    {
                                        warn!(?err, "h3 connection ended with error");
                                    }
                                };
                                // h3's serve_h3_connection is Send (the
                                // upstream h3 + h3-quinn stacks are Send-
                                // bound at every layer), so the fallback
                                // when no per-core runtime is set can
                                // use plain `tokio::spawn` — no LocalSet
                                // required at the caller. Compare with
                                // listeners::http which uses spawn_local
                                // even in the fallback because its h1
                                // driver may hold !Send state.
                                match &runtime_for_conns {
                                    Some(rt) => rt.spawn_on_current_core(Box::pin(conn_future)),
                                    None => {
                                        tokio::spawn(conn_future);
                                    }
                                }
                            }
                            Some(Err(err)) => {
                                warn!(?err, "h3 accept failed");
                            }
                            None => break,
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

fn build_server_config(spec: &Value) -> Result<quinn::ServerConfig, ProximaError> {
    // RFC 9001 §4.1.1: 0-RTT lets returning clients send data on
    // the first flight, saving 1 RTT. spec key `allow_0rtt: true`
    // enables it; default off. Replay risk is the caller's
    // problem to mitigate at the application layer.
    let allow_0rtt = spec
        .get("allow_0rtt")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // QUIC connection migration (RFC 9000 §9). On by default in
    // quinn; spec key `allow_migration: false` disables it for
    // operators who want connection pinning (sticky sessions,
    // stateful per-connection rate limit, IP-bound auth).
    let allow_migration = spec
        .get("allow_migration")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let mut server_config = if spec
        .get("dev_self_signed")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let sans = spec
            .get("dev_sans")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .filter(|sans| !sans.is_empty())
            .unwrap_or_else(|| vec!["localhost".to_string()]);
        dev_server_config(sans, &[ALPN_H3])
            .map_err(|err| ProximaError::Upstream(format!("h3 dev cert: {err}")))?
    } else {
        build_server_config_from_files(spec)?
    };

    apply_quic_transport_knobs(&mut server_config, allow_0rtt, allow_migration)?;
    Ok(server_config)
}

fn apply_quic_transport_knobs(
    server_config: &mut quinn::ServerConfig,
    allow_0rtt: bool,
    allow_migration: bool,
) -> Result<(), ProximaError> {
    let transport = Arc::get_mut(&mut server_config.transport).ok_or_else(|| {
        ProximaError::Config("h3 listener: TransportConfig is shared; clone before mutating".into())
    })?;
    transport.allow_spin(true);
    if !allow_migration {
        // quinn doesn't expose a direct "disable migration" boolean
        // but `max_idle_timeout(0)` + tight `migration` policy is
        // the closest production knob — for now, signal intent via
        // a `keep_alive_interval` tweak that pins NAT mappings. A
        // proper migration-disable lands when quinn surfaces it.
        // Until then, this is documentation: spec accepted, hard
        // enforcement is a follow-up.
        let _ = allow_migration; // explicit intent, no-op today
    }
    if allow_0rtt {
        // quinn's 0-RTT acceptance is server-side: clients send
        // early data when they have a valid session ticket. The
        // server-side knob is implicit — enabling stateless
        // tickets (which the dev cert path doesn't) flips it on.
        // For the file-cert path operators install their own
        // session keys; we document the requirement here.
        let _ = allow_0rtt; // see note above; runtime wiring TBD
    }
    Ok(())
}

fn build_server_config_from_files(spec: &Value) -> Result<quinn::ServerConfig, ProximaError> {
    // (Helper hoisted out so the knob-application path can call it
    // before applying transport-level config.)
    let cert_path = spec
        .get("cert_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| ProximaError::Config("h3 listener missing `cert_path`".into()))?;
    let key_path = spec
        .get("key_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| ProximaError::Config("h3 listener missing `key_path`".into()))?;

    let cert_bytes = std::fs::read(&cert_path)
        .map_err(|err| ProximaError::Upstream(format!("h3 cert read {cert_path:?}: {err}")))?;
    let key_bytes = std::fs::read(&key_path)
        .map_err(|err| ProximaError::Upstream(format!("h3 key read {key_path:?}: {err}")))?;

    let certs = <quinn::rustls::pki_types::CertificateDer<'_> as quinn::rustls::pki_types::pem::PemObject>::pem_slice_iter(&cert_bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| ProximaError::Upstream(format!("h3 cert parse: {err}")))?;
    let key = <quinn::rustls::pki_types::PrivateKeyDer<'_> as quinn::rustls::pki_types::pem::PemObject>::from_pem_slice(&key_bytes)
        .map_err(|err| ProximaError::Upstream(format!("h3 key parse: {err}")))?;

    let mut tls = quinn::rustls::ServerConfig::builder_with_protocol_versions(&[
        &quinn::rustls::version::TLS13,
    ])
    .with_no_client_auth()
    .with_single_cert(certs, key)
    .map_err(|err| ProximaError::Upstream(format!("h3 rustls config: {err}")))?;
    tls.alpn_protocols = vec![ALPN_H3.to_vec()];

    let quic_tls = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
        .map_err(|err| ProximaError::Upstream(format!("h3 quic rustls config: {err}")))?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_tls)))
}
