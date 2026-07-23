//! proxima's own Kafka client, built on the sans-IO wire codec
//! ([`crate::wire`] body layer over `proxima_protocols::kafka`'s framing +
//! header) — no `rdkafka`/`kafka-protocol` crate.
//!
//! Three layers, transport-agnostic by construction, mirroring
//! `proxima_redis::client`'s split:
//! - [`session::ClientSession`] — the sans-IO protocol state machine
//!   (`ApiVersions` handshake, request/reply). Bytes in, bytes out; no
//!   socket (principle 11).
//! - [`blocking::KafkaClient`] — a `std::io::Read + Write` driver around
//!   the session.
//! - [`pipe::KafkaClientUpstream`] — the async Pipe target, driving the
//!   same session over a futures-io transport, so `proxima::Client` speaks
//!   Kafka as just another registered protocol. Its typed
//!   [`pipe::KafkaClientUpstream::produce`]/[`pipe::KafkaClientUpstream::fetch`]
//!   are the ergonomic entry points; [`pipe::request_of`] exposes the raw
//!   `Request<Bytes>` convention underneath for a caller that wants
//!   `SendPipe::call` directly.

pub mod blocking;
pub mod config;
pub mod pipe;
pub mod session;

pub use blocking::KafkaClient;
pub use config::{KafkaClientConfig, KafkaConfigError};
pub use pipe::{KafkaClientUpstream, request_of};
pub use session::{ClientError, ClientSession, Step};
