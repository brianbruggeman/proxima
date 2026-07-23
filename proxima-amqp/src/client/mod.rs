//! proxima's own AMQP 0-9-1 client, built on the sans-IO wire/method codec
//! in [`crate::wire`]/[`crate::method`]/[`crate::fsm`] — no `lapin`, no
//! `amq-protocol` in the production dependency graph (only as a dev-only
//! bench comparison baseline, see `proxima-protocols`).
//!
//! Three layers, transport-agnostic by construction — mirrors
//! `proxima_redis::client`'s own split:
//! - [`session::ClientSession`] — the sans-IO protocol state machine
//!   (handshake, `basic.publish`, `basic.consume`). Bytes in, bytes out; no
//!   socket (principle 11).
//! - [`blocking::AmqpClient`] — a `std::io::Read + Write` driver around the
//!   session.
//! - [`pipe::AmqpClientUpstream`] — the async `Pipe` target, driving the
//!   same session over a futures-io transport, so `proxima::Client` speaks
//!   AMQP as just another registered protocol.

pub mod blocking;
pub mod config;
pub mod pipe;
pub mod session;

pub use blocking::{AmqpClient, ClientDelivery};
pub use config::{AmqpClientConfig, AmqpConfigError};
pub use pipe::AmqpClientUpstream;
pub use session::{ClientError, ClientSession, Step};
