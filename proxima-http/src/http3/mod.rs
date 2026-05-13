//! HTTP/3 protocol driver. **Two first-class surface layers** — both
//! permanently supported, gated independently:
//!
//! - **`native`** (default) — proxima's sans-IO HTTP/3 stack: the
//!   [`proxima_protocols::http3_codec`] state machine on top of
//!   [`proxima_quic::native`] (proxima-quic-proto +
//!   [`proxima_protocols::quic::tls::rustls_provider`]). Pulls in **rustls
//!   directly** without quinn/h3-quinn. Mounted at
//!   [`native::H3NativeListenProtocol`].
//!
//! - **`quinn-compat`** (default) — bridge over the upstream
//!   non-proxima crates (`quinn` + `h3` + `h3-quinn`). For consumers
//!   that want to ride the canonical implementations. Mounted at
//!   [`listener::H3ListenProtocol`].
//!
//! Both protocols are exported from [`proxima::listeners`]; consumers
//! pick by name in their listener spec (`"h3"` for legacy,
//! `"h3-native"` for the proxima stack). The dual surface is the
//! contract, not a transitional state.

#[cfg(feature = "http3-native")]
pub mod native;

#[cfg(feature = "http3-quinn-compat")]
pub mod listener;
#[cfg(feature = "http3-quinn-compat")]
pub mod server;
#[cfg(feature = "http3-quinn-compat")]
pub mod upstream;

#[cfg(feature = "http3-quinn-compat")]
pub use listener::H3ListenProtocol;
#[cfg(feature = "http3-quinn-compat")]
pub use server::serve_h3_connection;
#[cfg(feature = "http3-quinn-compat")]
pub use upstream::Http3Upstream;
