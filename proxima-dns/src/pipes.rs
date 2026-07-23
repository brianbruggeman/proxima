//! Typed pipe surface for the DNS listener — the DNS sibling of
//! `proxima_redis::pipes` / `proxima_pgwire::pipes`. A handler pipe never
//! touches wire bytes, borrowed [`proxima_protocols::dns::codec_trait::Message`]
//! views, or the RFC 1035 §4.1.4 compression walk: [`DnsQuery`] /
//! [`DnsAnswer`] are owned, business-level types decoded/encoded once at the
//! wire edge (see [`crate::datagram_protocol::DnsDatagramProtocol::on_datagram`]).
//!
//! A query's question section is, in every real-world resolver and stub
//! client, exactly one [`super::proxima_protocols::dns::Question`] (RFC 1035
//! §4.1.2 permits more, but no deployed client sends more than one and no
//! deployed server answers more than one) — [`DnsQuery`] carries that single
//! question typed; a listener facing a multi-question packet drops it as
//! malformed rather than guessing which question the handler wants (see
//! [`crate::datagram_protocol`]'s module doc for the drop reasoning).

/// One decoded DNS query, owned. Built once per inbound datagram from the
/// zero-copy [`proxima_protocols::dns::codec_trait::Message`] so a handler
/// pipe works with plain, `'static` data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsQuery {
    /// Echoed back into the response header (RFC 1035 §4.1.1) so the asker
    /// can match reply to request.
    pub id: u16,
    /// `RD` bit of the query — echoed into the response's own `RD` bit.
    pub recursion_desired: bool,
    /// Dotted question name, e.g. `"example.com."`.
    pub name: String,
    /// RFC 1035 §3.2.2 / RFC 3596 query type (1 = A, 28 = AAAA, 5 = CNAME, …).
    pub qtype: u16,
    /// RFC 1035 §3.2.4 query class (1 = IN, virtually always).
    pub qclass: u16,
}

/// One answer-section record a handler wants in its reply. `rdata` is
/// already-encoded wire bytes — see
/// [`proxima_protocols::dns::encode::ipv4_rdata`] /
/// [`proxima_protocols::dns::encode::ipv6_rdata`] for the common cases.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DnsAnswerRecord {
    /// Record owner name — typically the query name itself for a direct
    /// answer, or an alias target's name when chaining a CNAME.
    pub name: String,
    pub rtype: u16,
    pub rclass: u16,
    pub ttl: u32,
    pub rdata: Vec<u8>,
}

/// A handler's full reply to one [`DnsQuery`]: the response code plus zero
/// or more answer records. `authoritative` / `recursion_available` set the
/// matching response-header bits (RFC 1035 §4.1.1); an empty `records` with
/// `rcode == 0` is a legal "no data" answer (NODATA), distinct from
/// `rcode == 3` (NXDOMAIN, "name does not exist").
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DnsAnswer {
    /// RFC 1035 §4.1.1 4-bit RCODE (0 = `NOERROR`, 3 = `NXDOMAIN`, …).
    pub rcode: u8,
    pub authoritative: bool,
    pub recursion_available: bool,
    pub records: Vec<DnsAnswerRecord>,
}

impl DnsAnswer {
    /// A `NOERROR` reply carrying `records` — the common case.
    #[must_use]
    pub fn ok(records: Vec<DnsAnswerRecord>) -> Self {
        Self {
            rcode: 0,
            authoritative: false,
            recursion_available: true,
            records,
        }
    }

    /// An `NXDOMAIN` reply (RFC 1035 §4.1.1 RCODE 3) with no records.
    #[must_use]
    pub fn name_error() -> Self {
        Self {
            rcode: 3,
            authoritative: false,
            recursion_available: true,
            records: Vec::new(),
        }
    }
}

/// Typed request carrying a [`DnsQuery`] as payload.
pub type DnsPipeRequest = proxima_primitives::pipe::request::Request<DnsQuery>;

/// Typed response carrying a [`DnsAnswer`] as payload.
pub type DnsPipeReply = proxima_primitives::pipe::request::Response<DnsAnswer>;

/// Runtime-erased handle for DNS query-handler pipes.
pub type DnsPipeHandle = proxima_primitives::pipe::alloc_tier::PipeHandle<DnsPipeRequest, DnsPipeReply>;

/// Wrap any DNS-compatible pipe in a [`DnsPipeHandle`] — the bridge between
/// a business handler you write (`impl SendPipe<In = DnsPipeRequest, Out =
/// DnsPipeReply>`) and every seam that wants the type-erased
/// [`DnsPipeHandle`] ([`crate::DnsAnyProtocol::new`],
/// [`crate::DnsDatagramProtocol::listen_protocol`],
/// `proxima::ListenerProtocolExt::dns`).
///
/// ```
/// use proxima_dns::{DnsPipeRequest, DnsPipeReply, into_dns_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct Resolver;
/// impl SendPipe for Resolver {
///     type In = DnsPipeRequest;
///     type Out = DnsPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
///         unreachable!("illustrative — no query is dispatched in this doctest")
///     }
/// }
///
/// let handle = into_dns_handle(Resolver);
/// # let _ = handle;
/// ```
pub use proxima_primitives::pipe::alloc_tier::into_handle as into_dns_handle;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ok_answer_defaults_to_noerror_recursion_available() {
        let answer = DnsAnswer::ok(Vec::new());
        assert_eq!(answer.rcode, 0);
        assert!(answer.recursion_available);
        assert!(!answer.authoritative);
        assert!(answer.records.is_empty());
    }

    #[test]
    fn name_error_answer_is_rcode_three_with_no_records() {
        let answer = DnsAnswer::name_error();
        assert_eq!(answer.rcode, 3);
        assert!(answer.records.is_empty());
    }
}
