//! Runtime-agnostic QUIC I/O facade over
//! `proxima_protocols::quic`.
//!
//! **Two first-class surface layers** — both permanently supported,
//! gated independently:
//!
//! - **`native`** (default) — Future-shaped Endpoint+Connection over
//!   the sans-IO `proxima_protocols::quic` state machine. The TLS layer
//!   bridges directly to [`rustls`] via
//!   [`proxima_protocols::quic::tls::rustls_provider`]; **no quinn, no
//!   h3-quinn**. Drivable by any executor: prime in production, tokio
//!   via `tokio-compat`, embassy, or a hand-rolled poll loop. Per
//!   workspace principle 5: no tokio in production builds.
//!
//! - **`quinn-compat`** (default) — wraps the upstream non-proxima
//!   [`quinn`] + [`quinn-proto`] crates. For consumers that want to
//!   ride the canonical implementations directly.
//!
//! Both surfaces ship in the default feature set; consumers
//! `--no-default-features --features native` (or `--features
//! quinn-compat`) to pick one explicitly. The dual surface is the
//! contract, not a transitional state.

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "quinn-compat")]
pub mod connection;
#[cfg(feature = "quinn-compat")]
pub mod endpoint;
#[cfg(feature = "quinn-compat")]
pub mod stream_listener;

#[cfg(feature = "quinn-compat")]
pub use connection::Connection;
#[cfg(feature = "quinn-compat")]
pub use endpoint::{Endpoint, dev_server_config};
#[cfg(feature = "quinn-compat")]
pub use stream_listener::{QuicListener, QuicStreamConnection};
