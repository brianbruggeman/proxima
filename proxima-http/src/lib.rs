//! Runtime-agnostic HTTP/1.1, HTTP/2, HTTP/3, and WebSocket stacks for
//! proxima. Folded from `proxima-h1`, `proxima-h2`, `proxima-h3`, and
//! `proxima-websocket` into one crate, hyper-shaped, feature-gated per
//! protocol.
//!
//! Each stack composes the matching sans-IO codec from
//! `proxima-protocols` (`http1_codec`, `http2_codec`, `hpack`,
//! `http3_codec`) with the std transport edge (tokio, hyper). See each
//! module's own docs for the tier split.

/// Shared error->status/body rendering for the h1/h2/h3 server
/// drivers — kept in one spot so a rejection (`ProximaError::Forbidden`)
/// renders identically regardless of which protocol served it. Gated on
/// exactly the features whose server module calls it: `http1`/`http1-native`
/// (`http1::serve`), `http2-native` (`http2::server`, also reached via the
/// `http2` feature, which forwards into `http2-native`), and
/// `http3-quinn-compat` (`http3::server` — the quinn-based h3 bridge; see
/// `http3/mod.rs`). `http3-native`'s own server path (`http3::native`) does
/// not call this yet, so it is deliberately excluded — including it left
/// these fns dead-code under an `http3-native`-only build.
#[cfg(any(
    feature = "http1",
    feature = "http1-native",
    feature = "http2-native",
    feature = "http3-quinn-compat"
))]
mod error_render;

/// `Listener::any()` scaffolding: the h1/h2-prior-knowledge
/// `AnyProtocol` candidates plus `AnyListenProtocol`, the open registry
/// -driven sibling of [`listener::HttpListenProtocol`]. Needs `http-listener`
/// for the h1 candidate (`serve_h1_connection`) at minimum; the h2
/// candidate additionally needs `http2-native`.
#[cfg(feature = "http-listener")]
pub mod any_listener;
#[cfg(any(
    feature = "http1",
    feature = "http1-native",
    feature = "http1-stream-client"
))]
pub mod http1;
#[cfg(any(feature = "http2", feature = "http2-native"))]
pub mod http2;
#[cfg(any(feature = "http3-native", feature = "http3-quinn-compat"))]
pub mod http3;
/// ALPN-multiplexed h1+h2 listener combiner, folded in from the former
/// `proxima-listeners-http` crate.
#[cfg(feature = "http-listener")]
pub mod listener;
/// `{{var}}` string-template expansion, folded in from the former
/// `proxima-templates` crate. Used by [`http1::client`] and
/// [`http1::upstream`] for dynamic header injection.
#[cfg(any(feature = "http1", feature = "http1-stream-client"))]
pub mod templates;
#[cfg(feature = "websocket")]
pub mod websocket;
