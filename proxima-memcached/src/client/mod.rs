//! proxima's own memcached client, built on the sans-IO text-protocol
//! codec — no `memcache`/`memcached-rs` crate.
//!
//! Three layers, transport-agnostic by construction:
//! - [`session::ClientSession`] — the sans-IO protocol state machine
//!   (request/reply). Bytes in, bytes out; no socket (principle 11).
//! - [`blocking::MemcachedClient`] — a `std::io::Read + Write` driver
//!   around the session.
//! - [`pipe::MemcachedClientUpstream`] — the async Pipe + `PipeFactory`
//!   target, driving the same session over a futures-io transport, so
//!   `proxima::Client` speaks memcached as just another registered
//!   protocol.

pub mod blocking;
pub mod config;
pub mod pipe;
pub mod session;

pub use blocking::MemcachedClient;
pub use config::{MemcachedClientConfig, MemcachedConfigError};
pub use pipe::MemcachedClientUpstream;
pub use session::{ClientError, ClientSession, Step};
