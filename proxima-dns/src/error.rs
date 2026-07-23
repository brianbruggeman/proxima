//! Facade-level serve/resolve errors. Wire-level detail stays in
//! `proxima_protocols::dns` (`ParseError`/`EncodeError`); this layer adds
//! transport, buffer-policy, and configuration failures. Mirrors
//! `proxima_redis::error::RedisServeError` /
//! `proxima_pgwire::error::ServeError`.

use thiserror::Error;

/// Listener-side serve failure — surfaced by [`crate::DnsDatagramProtocol`]
/// (UDP) and [`crate::DnsAnyProtocol`] (TCP).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DnsServeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("handler pipe: {0}")]
    Pipe(#[from] proxima_core::ProximaError),
    #[error("inbound query exceeds the {limit}-byte message limit")]
    MessageTooLarge { limit: usize },
    #[error("dns wire error: {0}")]
    Wire(String),
    #[error("connection closed mid-message")]
    UnexpectedEof,
    #[error("config: {0}")]
    Config(String),
}

/// Resolver-client failure — surfaced by [`crate::client::DnsClientUpstream`].
/// A resolver-side RCODE (NXDOMAIN, SERVFAIL, …) is NOT one of these: it is a
/// successful protocol exchange with a negative answer, returned as
/// `Ok(DnsAnswer { rcode, .. })` for the caller to interpret — these variants
/// are transport/framing failures where no interpretable answer arrived.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DnsClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("dns wire error: {0}")]
    Wire(String),
    #[error("query timed out after {0}ms with no matching reply")]
    Timeout(u64),
    #[error("reply id {reply} does not match the outstanding query id {expected}")]
    IdMismatch { expected: u16, reply: u16 },
    #[error("config: {0}")]
    Config(String),
}
