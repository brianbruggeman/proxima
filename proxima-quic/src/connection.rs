//! Substrate-side QUIC connection façade. Wraps a single accepted
//! QUIC connection. The h3 driver builds on this; future native-quic
//! work swaps the internals to a `quinn_proto::Connection` event pump
//! without changing the type's public shape.

use std::net::SocketAddr;

pub struct Connection {
    inner: quinn::Connection,
}

impl Connection {
    pub(crate) fn new(inner: quinn::Connection) -> Self {
        Self { inner }
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.inner.remote_address()
    }

    /// Negotiated ALPN protocol from the TLS handshake, if any.
    pub fn alpn_protocol(&self) -> Option<Vec<u8>> {
        self.inner
            .handshake_data()
            .and_then(|d| d.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
            .and_then(|d| d.protocol.clone())
    }

    /// Underlying [`quinn::Connection`] for bridge crates (h3-quinn).
    /// Public so `proxima-h3` can bridge; not part of the
    /// substrate-portable surface.
    pub fn quinn(&self) -> quinn::Connection {
        self.inner.clone()
    }
}
