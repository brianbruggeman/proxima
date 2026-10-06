//! TLS termination at the listener.
//!
//! Tier: tier marker. Today's `TlsConfig` uses `PathBuf` in every
//! variant (`TlsMode::Files`, `ClientAuth::Required.trust_anchors`,
//! `SniCert.cert/key`) which inherently needs std. Under no_std + alloc
//! the crate is a marker.
//!
//! A clean no_std + alloc `proxima-tls-rustls-core` extraction
//! (taking byte-payload cert/key inputs instead of PathBuf) is the
//! disciplined-component follow-on tracked as DC-TLS-CORE in the
//! no_std discipline log — multi-day disciplined-component effort
//! (reshapes the public config API), out of scope for this pass.
//!
//! Layers above TCP, layers below L7. The output is a runtime-neutral
//! `rustls::ServerConfig` (`build_server_config`) — a handshake wrapper
//! that turns an accepted stream into a TLS session implementing
//! `AsyncRead + AsyncWrite`. Two adapters wrap it: the default
//! tokio-free `futures_rustls::TlsAcceptor` (`build_acceptor_futures_io`,
//! `futures-io` feature) and `tokio_rustls::TlsAcceptor`
//! (`build_acceptor`, `tokio` feature) for callers on a tokio-native
//! socket. Whatever L7 stack consumes plaintext after that sees an
//! opaque byte stream.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
mod imp;

#[cfg(feature = "std")]
pub use imp::*;

#[cfg(all(test, feature = "futures-io"))]
#[path = "build_defaults.rs"]
mod build_defaults;

#[cfg(feature = "futures-io")]
mod connector;

#[cfg(feature = "futures-io")]
pub use connector::{
    RootSource, TlsClientConfig, TlsClientLayerBuilder, TlsConn, TlsStreamUpstream,
};
