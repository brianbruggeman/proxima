//! Substrate-side QUIC endpoint. Wraps the I/O-bound driver behind a
//! façade that hides whether the backend is the high-level [`quinn`]
//! crate (today) or a [`quinn_proto`]-driven loop on the substrate
//! `Runtime` (future).

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::Connection;

/// Endpoint façade. One per UDP socket. Accepts inbound QUIC
/// connections; each accepted handle becomes a [`Connection`] that
/// owns its stream multiplexer.
pub struct Endpoint {
    inner: quinn::Endpoint,
    local_addr: Option<SocketAddr>,
}

impl Endpoint {
    /// Bind a server endpoint to `addr` using the supplied TLS server
    /// config. ALPN protocols (e.g. `h3`) are configured on the
    /// `ServerConfig` by the caller.
    pub fn server(addr: SocketAddr, server_config: quinn::ServerConfig) -> io::Result<Self> {
        let inner = quinn::Endpoint::server(server_config, addr)?;
        let local_addr = inner.local_addr().ok();
        Ok(Self { inner, local_addr })
    }

    /// Local bind address after the OS resolved any ephemeral port.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    /// Accept the next inbound QUIC connection. Returns `None` once
    /// the endpoint is closed.
    pub async fn accept(&self) -> Option<io::Result<Connection>> {
        let incoming = self.inner.accept().await?;
        Some(match incoming.await {
            Ok(connection) => Ok(Connection::new(connection)),
            Err(err) => Err(io::Error::other(format!("quic handshake: {err}"))),
        })
    }

    /// Trigger a graceful close. In-flight connections drain before
    /// the endpoint future resolves.
    pub fn close(&self, error_code: u32, reason: &[u8]) {
        self.inner.close(error_code.into(), reason);
    }
}

/// Build a self-signed `ServerConfig` for tests / dev. Generates a
/// fresh certificate for the supplied SAN list and advertises the
/// supplied ALPN protocols (e.g. `b"h3"`).
pub fn dev_server_config(sans: Vec<String>, alpn: &[&[u8]]) -> io::Result<quinn::ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(sans)
        .map_err(|err| io::Error::other(format!("rcgen: {err}")))?;
    let cert_der = cert.cert.der().clone();
    let key_der =
        quinn::rustls::pki_types::PrivateKeyDer::Pkcs8(cert.signing_key.serialize_der().into());

    let mut tls = quinn::rustls::ServerConfig::builder_with_protocol_versions(&[
        &quinn::rustls::version::TLS13,
    ])
    .with_no_client_auth()
    .with_single_cert(vec![cert_der], key_der)
    .map_err(|err| io::Error::other(format!("rustls server config: {err}")))?;
    tls.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();

    let quic_tls = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
        .map_err(|err| io::Error::other(format!("quic rustls config: {err}")))?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_tls)))
}
