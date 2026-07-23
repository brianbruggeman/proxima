//! proxima's own MQTT client, built on the sans-IO codec in
//! `proxima_protocols::mqtt` — no third-party MQTT client crate.
//!
//! Two layers, transport-agnostic by construction:
//! - [`session::ClientSession`] — the sans-IO protocol state machine
//!   (`CONNECT`/`CONNACK` handshake, request/reply, the `SUBSCRIBE` push
//!   loop). Bytes in, bytes out; no socket (principle 11).
//! - [`pipe::MqttClientUpstream`] — the async Pipe + `PipeFactory` target,
//!   driving the same session over a futures-io transport, so
//!   `proxima::Client` speaks MQTT as just another registered protocol.

pub mod config;
pub mod pipe;
pub mod session;

pub use config::{MqttClientConfig, MqttConfigError};
pub use pipe::MqttClientUpstream;
pub use session::{ClientError, ClientSession, PushStep, Step};
