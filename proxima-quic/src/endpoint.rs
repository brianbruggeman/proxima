//! Core-side QUIC endpoint over the upstream `quinn` crate — the
//! `quinn-compat` half of this crate's dual surface. Accepted
//! connections are handed back as plain [`quinn::Connection`]s: this
//! surface exists to ride quinn directly, so wrapping its connection
//! handle would only rename it. The sans-IO alternative that owes
//! nothing to quinn is this crate's `native` module.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

/// Endpoint façade. One per UDP socket. Accepts inbound QUIC
/// connections; each accepted handle is a [`quinn::Connection`] that
/// owns its stream multiplexer.
pub struct Endpoint {
    inner: quinn::Endpoint,
}

impl Endpoint {
    /// Bind a server endpoint to `addr` using the supplied TLS server
    /// config. ALPN protocols (e.g. `h3`) are configured on the
    /// `ServerConfig` by the caller.
    ///
    /// # Errors
    ///
    /// Bubbles up the `bind(2)` failure.
    pub fn server(addr: SocketAddr, server_config: quinn::ServerConfig) -> io::Result<Self> {
        let inner = quinn::Endpoint::server(server_config, addr)?;
        Ok(Self { inner })
    }

    /// Local bind address after the OS resolved any ephemeral port.
    /// Same signature as the native facade's `Endpoint::local_addr`.
    ///
    /// # Errors
    ///
    /// Bubbles up [`std::io::Error`] from `getsockname(2)`.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Accept the next inbound QUIC connection, collapsing quinn's
    /// two-step `accept().await?.await` into one call and its handshake
    /// error into [`io::Error`]. Returns `None` once the endpoint is
    /// closed.
    pub async fn accept(&self) -> Option<io::Result<quinn::Connection>> {
        let incoming = self.inner.accept().await?;
        Some(
            incoming
                .await
                .map_err(|err| io::Error::other(format!("quic handshake: {err}"))),
        )
    }

    /// Trigger a graceful close. In-flight connections drain before
    /// the endpoint future resolves.
    pub fn close(&self, error_code: u32, reason: &[u8]) {
        self.inner.close(error_code.into(), reason);
    }
}

/// Build a self-signed `ServerConfig` for tests / dev. Generates a
/// fresh certificate for the supplied SAN list and advertises the
/// supplied ALPN protocols (e.g. `b"h3"`). Production plugs in real
/// certs instead.
///
/// # Errors
///
/// Certificate generation, key encoding, or rustls rejecting the
/// resulting cert/key pair.
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
