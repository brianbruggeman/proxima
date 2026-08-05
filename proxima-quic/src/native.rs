//! Runtime-agnostic native QUIC facade over `proxima_protocols::quic`.
//!
//! The facade is shaped as `poll_*` methods so any executor with a
//! `Future` shape can drive it: prime in production, tokio behind the
//! `tokio-compat` feature flag, embassy, or a hand-rolled poll loop.
//!
//! # Layering
//!
//! - The sans-IO state machine
//!   ([`proxima_protocols::quic::connection::Connection`]) produces
//!   `Transmit` descriptors and consumes inbound datagrams.
//! - The facade wraps it with a UDP socket (today via
//!   [`prime::os::net::UdpSocket`]) + a monotonic clock + an executor-
//!   agnostic `poll_*` loop driver.
//! - Public configs derive `Serialize + Deserialize` so conflaguration
//!   loaders compose (principle 4); the `Builder` + `Settings` derives
//!   layer on at the consumer crate (see [`config`]).
//!
//! # Surface
//!
//! - [`Endpoint`] — one socket, one connection, `poll_send` /
//!   `poll_recv` plus their `sendmmsg`/`recvmmsg` batch twins. The
//!   client-dial and single-inbound-connection shape.
//! - [`Listener`] — the multi-connection server side: an I/O-free,
//!   DCID-demuxed [`DatagramProtocol`](proxima_listen::stream::DatagramProtocol)
//!   state machine that the `proxima-listen` datagram driver serves.
//! - `TokioEndpoint` (feature `tokio-compat`) — the same
//!   single-connection shape over `tokio::net::UdpSocket`.

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
