//! proxima's own DNS resolver client + listener facade, mirroring
//! `proxima_redis`'s crate structure.
//!
//! The sans-IO RFC 1035 parser ([`Header`], [`Flags`], [`Question`],
//! [`Record`], [`RData`], [`Name`], [`ParseError`], [`parse_header`],
//! [`parse_question`], [`parse_record`]) plus the write-side encoder
//! ([`EncodeError`], [`EncodeQuestion`], [`AnswerRecord`], [`encode_query`],
//! [`encode_response`]) and the `proxima_codec::Datagram` impl
//! ([`DnsDatagramCodec`], [`Message`], [`parse_message`]) all live in
//! [`proxima_protocols::dns`] — see its docs for the wire layer. This crate
//! is the std client + listener built on top:
//!
//! - the async [`client::DnsClientUpstream`] resolver, driving the sans-IO
//!   [`client::DnsClientSession`] over a pluggable
//!   [`proxima_primitives::stream::DatagramFactory`] (prime, tokio, a fake
//!   test socket) — the `client` feature.
//! - [`DnsDatagramProtocol`] — the UDP server, a
//!   [`proxima_listen::stream::DatagramProtocol`] state machine driven by
//!   `DatagramProtocolListenProtocol`, dispatching each parsed query to a
//!   caller-supplied [`DnsPipeHandle`] and staging the encoded reply — the
//!   `listen` feature.
//! - [`DnsAnyProtocol`] — the DNS-over-TCP (RFC 1035 §4.2.2) sibling, an
//!   [`proxima_listen::any::AnyProtocol`] candidate for the open universal
//!   listener — also the `listen` feature. See its module doc for the
//!   2-byte length-prefix framing gap this module fills directly rather
//!   than extending the shared codec crate.

#[cfg(any(feature = "client", feature = "listen"))]
pub mod error;
#[cfg(any(feature = "client", feature = "listen"))]
pub mod pipes;
#[cfg(any(feature = "client", feature = "listen"))]
pub(crate) mod wire;

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "listen")]
pub mod any_protocol;
#[cfg(feature = "listen")]
pub mod config;
#[cfg(feature = "listen")]
pub mod datagram_protocol;
#[cfg(feature = "listen")]
pub mod framed_app;

pub use proxima_protocols::dns::{
    Flags, Header, Name, ParseError, Question, RData, Record, parse_header, parse_name,
    parse_question, parse_record,
};
pub use proxima_protocols::dns::encode::{
    AnswerRecord, EncodeError, EncodeQuestion, encode_name, encode_query, encode_response,
    ipv4_rdata, ipv6_rdata,
};

pub use proxima_protocols::dns::codec_trait::{DnsDatagramCodec, Message, QuestionIter, RecordIter, parse_message};

#[cfg(feature = "client")]
pub use client::{DnsClientUpstream, DnsConfigError, DnsResolverConfig};

#[cfg(any(feature = "client", feature = "listen"))]
pub use error::DnsClientError;
#[cfg(any(feature = "client", feature = "listen"))]
pub use pipes::{DnsAnswer, DnsAnswerRecord, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, DnsQuery, into_dns_handle};

// the server-side surface a DNS query handler builds against — re-exported
// so a caller imports everything from proxima-dns and never reaches past
// it into proxima-protocols/proxima-listen internals (teaching surface,
// workspace principle 2), mirroring proxima-redis's own top-level
// re-export shape.
#[cfg(feature = "listen")]
pub use any_protocol::DnsAnyProtocol;
#[cfg(feature = "listen")]
pub use config::DnsServerConfig;
#[cfg(feature = "listen")]
pub use datagram_protocol::DnsDatagramProtocol;
#[cfg(feature = "listen")]
pub use error::DnsServeError;
#[cfg(feature = "listen")]
pub use framed_app::{DnsFramedApp, DnsFramedAppError, DnsTcpOutcome};
