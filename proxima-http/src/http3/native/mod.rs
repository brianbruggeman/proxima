//! Runtime-agnostic native HTTP/3 facade.
//!
//! Composes [`proxima_quic::native::Endpoint`] with the sans-IO
//! [`proxima_protocols::http3_codec::server::ServerConnection`] /
//! [`proxima_protocols::http3_codec::client::ClientConnection`] state machines.
//! The Future-shaped `poll_*` surface lets any executor drive it.

pub mod client;
pub mod config;
pub mod driver;
pub mod listen;
pub mod server;
#[cfg(feature = "http3-native-upstream")]
pub mod upstream;

pub use client::{Client, ClientError};
pub use config::{ClientConfig, ServerConfig};
pub use driver::{DriverState, drive_client_step, drive_server_step};
pub use listen::H3NativeListenProtocol;
pub use server::{Server, ServerError};
#[cfg(feature = "http3-part-source")]
pub use upstream::bench_multiplexed_part_source;
#[cfg(feature = "http3-native-upstream")]
pub use upstream::{H3NativeUpstream, bench_multiplexed};
