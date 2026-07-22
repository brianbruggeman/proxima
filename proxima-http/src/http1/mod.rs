//! HTTP/1.1 stack for proxima: wire-format parser, body framing,
//! connection state machine, response writer, and `shared_http` client
//! pool (hyper-legacy on tokio).
//!
//! Lives behind feature gates in `proxima-http`'s `[features]` table:
//!
//! - `http1-native` enables the sans-IO wire parser + connection state
//!   machine + the tokio-free `serve` driver (`serve_connection` /
//!   `serve_h1_connection`) — no hyper, no tokio.
//! - `http1` layers the legacy hyper/tokio client stack
//!   (`upstream`/`shared_http`/`client`) and the `H1ListenProtocol`
//!   sibling's tokio accept loop on top of `http1-native`.
//! - `http1-tls` enables the https-capable `shared_http` connector.
//!
//! Listeners are in the umbrella's `listeners/http.rs` for now (they
//! depend on the listener registry); a follow-on extraction pulls them
//! into this crate.

// Sans-IO codec lives in proxima-protocols::http1_codec; re-exported
// here so existing `proxima_http::http1::{h1, h1_body, h1_connection,
// h1_response}` call sites keep working.
pub use proxima_protocols::http1_codec::{h1, h1_body, h1_connection, h1_response};

#[cfg(feature = "http1-stream-client")]
pub mod client;
#[cfg(feature = "http1")]
pub mod hyper_body;
// pure config types shared by the hyper-backed `upstream` module AND the
// tokio-free prime client (`client`/`prime_upstream`) — see http_config.rs's
// doc comment. Compiles under either arm, so it's ungated.
pub mod http_config;
#[cfg(feature = "http1")]
pub mod listener;
#[cfg(feature = "http1-stream-client")]
pub mod prime_upstream;
// pure config types carried by the always-compiled `upstream` module, so they
// live outside the `http1-stream-client`-gated `client` module.
pub mod response_config;
#[cfg(feature = "http1-native")]
pub mod serve;
#[cfg(feature = "http1")]
pub mod shared_http;
#[cfg(feature = "http1")]
pub mod upstream;
#[cfg(all(target_os = "linux", feature = "http1-io-uring"))]
pub mod uring_transport;

#[cfg(feature = "http1-stream-client")]
pub use client::{H1ClientConfig, H1ClientUpstream};
pub use http_config::{HeaderForward, HttpConfig, HttpHeadersConfig, HttpUpstreamConfig};
#[cfg(feature = "http1")]
pub use listener::H1ListenProtocol;
#[cfg(feature = "http1-stream-client")]
pub use prime_upstream::PrimeHttpPipeFactory;
pub use response_config::{
    ResponseBodyMode, ResponseHandling, ResponseHandlingConfig, ResponseHeaderMode,
};
#[cfg(feature = "http1-native")]
pub use serve::{HttpListenerSpec, serve_connection, serve_h1_connection};
#[cfg(feature = "http1")]
pub use upstream::{HttpPipeFactory, HttpUpstream};
#[cfg(all(target_os = "linux", feature = "http1-io-uring"))]
pub use uring_transport::UringAsyncStream;
