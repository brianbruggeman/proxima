//! proxima's own Redis/Valkey client, built on the sans-IO RESP codec — no
//! `redis` crate.
//!
//! Three layers, transport-agnostic by construction:
//! - [`session::ClientSession`] — the sans-IO protocol state machine (handshake,
//!   request/reply, pub/sub push loop). Bytes in, bytes out; no socket
//!   (principle 11).
//! - [`blocking::RedisClient`] — a `std::io::Read + Write` driver around the
//!   session (the real-server parity harness uses it).
//! - [`pipe::RedisClientUpstream`] — the async Pipe + `PipeFactory` target,
//!   driving the same session over a futures-io transport, so `proxima::Client`
//!   speaks Redis/Valkey as just another registered protocol.

pub mod blocking;
pub mod config;
pub mod pipe;
pub mod session;

pub use blocking::RedisClient;
pub use config::{RedisClientConfig, RedisConfigError, RespProtocol};
pub use pipe::RedisClientUpstream;
pub use session::{ClientError, ClientSession, PushStep, Step};
