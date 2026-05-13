//! proxima's own PostgreSQL client, built on `proxima-pgwire-codec` +
//! [`crate::scram::ScramClient`] — no `tokio-postgres`.
//!
//! Three layers, transport-agnostic by construction:
//! - [`session::ClientSession`] — the sans-IO protocol state machine (startup,
//!   trust/cleartext/SCRAM auth, simple/extended query). Bytes in, bytes out;
//!   no socket (principle 11). The client-side mirror of the codec's server
//!   `Session`.
//! - [`blocking::PgClient`] — a `std::io::Read + Write` driver around the
//!   session (the real-PG parity harness uses it).
//! - the async Pipe + `PipeFactory` (see `crate::client::pipe`) drive the same
//!   session over a futures-io transport, so `proxima::Client` speaks pgwire as
//!   just another registered protocol.

pub mod blocking;
pub mod config;
#[cfg(feature = "client")]
pub mod pipe;
pub mod session;

pub use blocking::PgClient;
pub use config::{ConfigError, PgClientConfig};
#[cfg(feature = "client")]
pub use pipe::PgwireClientUpstream;
pub use session::{ClientError, ClientSession, Column, QueryResult, Step};
