//! Runtime-agnostic native QUIC facade over `proxima_protocols::quic`.
//!
//! The facade is shaped as `poll_*` methods so any executor with a
//! `Future` shape can drive it: prime in production, tokio behind the
//! `tokio-compat` feature flag, embassy, or a hand-rolled poll loop.
//!
//! # Layering
//!
//! - The sans-IO state machine ([`proxima_protocols::quic::Connection`])
//!   produces `Transmit` descriptors and consumes inbound datagrams.
//! - The facade wraps it with a UDP socket (today via
//!   [`prime::os::net::UdpSocket`]) + a monotonic clock + an executor-
//!   agnostic `poll_*` loop driver.
//! - Public configs derive `Builder + Serialize + Deserialize` so
//!   conflaguration loaders compose (principle 4).
//!
//! # Status
//!
//! C29 surface: `Endpoint::bind_client`, `Endpoint::poll_send`,
//! `Endpoint::poll_recv`, and the client-side `Connection` shape with
//! `poll_application_send` / `poll_application_recv` /
//! `poll_handshake`. Server-side `accept` lands in B1.1 alongside the
//! endpoint demux integration.

pub mod config;
pub mod endpoint;
pub mod listener;

#[cfg(feature = "tokio-compat")]
pub mod tokio_endpoint;

pub use config::{ClientConfig, EndpointConfig, ServerConfig};
pub use endpoint::{Endpoint, EndpointError};
pub use listener::{AcceptFn, DatagramIngest, Listener, ListenerError};

#[cfg(feature = "tokio-compat")]
pub use tokio_endpoint::TokioEndpoint;
