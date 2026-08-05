//! Facade-level resolve errors. Wire-level detail stays in
//! `proxima_protocols::dns` (`ParseError`/`EncodeError`/`DnsTcpFrameError`);
//! this layer adds transport and configuration failures.
//!
//! There is no listener-side sibling: neither DNS server can fail. A
//! malformed, oversized, or unanswerable query is warned and dropped at the
//! point the decision is made — one bad query must never tear down a
//! connectionless listener — so `DnsDatagramProtocol` names
//! `core::convert::Infallible` as its error type and the DNS-over-TCP app
//! surfaces the codec's own `DnsTcpFrameError` unwrapped.

use thiserror::Error;

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
