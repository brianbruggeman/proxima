//! Runtime-agnostic HTTP/1.1, HTTP/2, HTTP/3, and WebSocket stacks for
//! proxima. Folded from `proxima-h1`, `proxima-h2`, `proxima-h3`, and
//! `proxima-websocket` into one crate, hyper-shaped, feature-gated per
//! protocol.
//!
//! Each stack composes the matching sans-IO codec from
//! `proxima-protocols` (`http1_codec`, `http2_codec`, `hpack`,
//! `http3_codec`) with the std transport edge (tokio, hyper). See each
//! module's own docs for the tier split.

// every module below documents itself with `//!` in its own file. an outer
// `///` here would not merely duplicate that — rustdoc resolves a merged
// module doc's intra-doc links in the DECLARING scope, so the module's own
// unqualified `[`Foo`]` links stop resolving.
#[cfg(any(
    feature = "http1",
    feature = "http1-native",
    feature = "http2-native",
    feature = "http3-quinn-compat"
))]
mod error_render;

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
#[cfg(feature = "http-listener")]
pub mod listener;
#[cfg(any(feature = "http1", feature = "http1-stream-client"))]
pub mod templates;
#[cfg(feature = "websocket")]
pub mod websocket;
