//! proxima's own DNS resolver client, built on the sans-IO RFC 1035 codec —
//! no `hickory-resolver`/`trust-dns`. Two layers, transport-agnostic by
//! construction, mirroring `proxima_redis::client`'s split:
//! - [`session::DnsClientSession`] — the sans-IO protocol state (query
//!   encode, response decode, id tracking). Bytes in, bytes out; no socket
//!   (principle 11).
//! - [`pipe::DnsClientUpstream`] — the async driver over a
//!   [`proxima_primitives::stream::DatagramFactory`], so `proxima::Client`
//!   can speak DNS as just another registered protocol.

pub mod config;
pub mod pipe;
pub mod session;

pub use config::{DnsConfigError, DnsResolverConfig};
pub use pipe::DnsClientUpstream;
pub use session::DnsClientSession;
