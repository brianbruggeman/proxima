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
/// renders identically regardless of which protocol served it.
#[cfg(any(
    feature = "http1",
    feature = "http2-native",
    feature = "http3-native",
    feature = "http3-quinn-compat"
))]
mod error_render;

#[cfg(any(feature = "http1", feature = "http1-stream-client"))]
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
