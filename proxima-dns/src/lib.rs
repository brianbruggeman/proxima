//! proxima's own DNS resolver client + listener facade.
//!
//! The sans-IO RFC 1035 wire layer lives in [`proxima_protocols::dns`] — the
//! parser ([`Header`], [`Flags`], [`Question`], [`Record`], [`RData`],
//! [`Name`], [`parse_header`], [`parse_question`], [`parse_record`]), the
//! encoder ([`EncodeQuestion`], [`AnswerRecord`], [`encode_query`],
//! [`encode_response`]), the `proxima_codec::Datagram` impl
//! ([`DnsDatagramCodec`], [`Message`], [`parse_message`]) and the
//! DNS-over-TCP `proxima_codec::FrameCodec` ([`DnsTcpCodec`],
//! [`DnsTcpOwnedFrame`]). All of it is re-exported here unconditionally, so a
//! caller imports everything from `proxima-dns` and never reaches past it into
//! `proxima-protocols` internals (teaching surface, workspace principle 2).
//!
//! On top of that wire layer this crate adds the std-tier client and server:
//!
//! - `client::DnsClientUpstream` — the async resolver, driving the sans-IO
//!   `client::session` encode/decode pair over a caller-injected
//!   `proxima_primitives::stream::DatagramFactory` (prime, tokio, a fake test
//!   socket), so `proxima::Client` speaks DNS as a registered protocol. The
//!   `client` feature.
//! - `DnsUdpAnyProtocol` / `DnsAnyProtocol` — the DNS-over-UDP (RFC 1035
//!   §4.2.1) and DNS-over-TCP (§4.2.2) `proxima_listen::any::AnyProtocol`
//!   candidates `proxima::ListenerProtocolExt::dns` registers together, so one
//!   `.any()`-fanned bind answers both transports on one port. The `listen`
//!   feature.
//! - `DnsDatagramProtocol` — a standalone, dedicated UDP-only listener (a
//!   `proxima_listen::stream::DatagramProtocol` state machine with its own
//!   batched recv/transmit tick) for a caller who wants exactly that and no
//!   TCP sibling. Also the `listen` feature.
//!
//! Every name behind `client` / `listen` above is deliberately unlinked: an
//! intra-doc link to a feature-gated item fails `cargo doc` at the tiers where
//! that item does not exist, including this crate's own default one.
//!
//! ## Scope
//!
//! **UDP and TCP, no DNS-over-QUIC (DoQ) or DNS-over-TLS/HTTPS (DoT/DoH).**
//! Classic UDP queries and DNS-over-TCP framing are both implemented; the
//! encrypted-transport variants are not — `proxima::ListenerProtocolExt::dns`'s
//! `.quic()` pairing is a named config error rather than a silent plaintext
//! fallback (see that method's doc).

#[cfg(feature = "client")]
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
#[cfg(feature = "listen")]
pub mod udp_any_protocol;

pub use proxima_protocols::dns::encode::{
    AnswerRecord, EncodeError, EncodeQuestion, encode_name, encode_query, encode_response,
    ipv4_rdata, ipv6_rdata,
};
pub use proxima_protocols::dns::{
    Flags, Header, Name, ParseError, Question, RData, Record, parse_header, parse_name,
    parse_question, parse_record,
};

pub use proxima_protocols::dns::codec_trait::{
    DnsDatagramCodec, Message, QuestionIter, RecordIter, parse_message,
};
pub use proxima_protocols::dns::frame_codec::{
    DnsTcpCodec, DnsTcpFrameError, DnsTcpOwnedFrame, DnsTcpQuery, DnsTcpViolation,
};

#[cfg(feature = "client")]
pub use client::{DnsClientUpstream, DnsConfigError, DnsResolverConfig};

#[cfg(feature = "client")]
pub use error::DnsClientError;
#[cfg(any(feature = "client", feature = "listen"))]
pub use pipes::{
    DnsAnswer, DnsAnswerRecord, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, DnsQuery,
    into_dns_handle,
};

#[cfg(feature = "listen")]
pub use any_protocol::DnsAnyProtocol;
#[cfg(feature = "listen")]
pub use config::DnsServerConfig;
#[cfg(feature = "listen")]
pub use datagram_protocol::DnsDatagramProtocol;
#[cfg(feature = "listen")]
pub use framed_app::{DnsFramedApp, DnsTcpOutcome};
#[cfg(feature = "listen")]
pub use udp_any_protocol::DnsUdpAnyProtocol;
