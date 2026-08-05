//! `DnsAnyProtocol` — DNS-over-TCP (RFC 1035 §4.2.2: each message prefixed
//! by a 2-byte big-endian length) as an [`AnyProtocol`] candidate for the
//! open universal listener, mirroring `RedisAnyProtocol` /
//! `proxima_pgwire`'s own `AnyProtocol` candidates.
//!
//! `drive` builds and delegates to a
//! [`proxima_listen::any::FramedAny<DnsTcpCodec, DnsFramedApp, _, _>`] — the
//! generic stateless `AnyProtocol` driver. `DnsAnyProtocol` itself stays a
//! thin, named constructor: it resolves the per-connection
//! [`DnsServerConfig`] from the listener spec and BUILDS a fresh `FramedAny`
//! per accepted connection, mirroring `MemcachedAnyProtocol`'s shape.
//!
//! **Gap this module fills, not one it hides:** `proxima_codec`'s only
//! length-delimited framer, [`proxima_codec::LengthDelimitedCodec`], is
//! hard-coded to a **4-byte** big-endian prefix (`LengthDelimitedCodec::
//! HEADER_BYTES == 4`). RFC 1035 §4.2.2's TCP framing is a **2-byte** prefix
//! — a different width the shared codec cannot express, so
//! [`proxima_protocols::dns::DnsTcpCodec`] reads and writes the 2-byte prefix
//! itself as a plain [`proxima_codec::FrameCodec`] impl. Widening
//! `LengthDelimitedCodec` to a generic prefix width is a change to a crate
//! four other codecs depend on; it is named here as the gap rather than
//! solved as a side effect of a DNS module.
//!
//! Positive-match probe: the 2-byte length prefix must declare at least a
//! header, and that header must open a query — `QR == 0`, `OPCODE == 0`,
//! `QDCOUNT == 1`. DNS has no sigil byte to sniff the way RESP's `*` or a
//! fixed magic number would give, so the header's own reserved and count
//! fields are the whole signal, and the DNS-over-UDP candidate classifies on
//! exactly the same predicate.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, ShedReason};
use proxima_listen::any::{AnyHandler, AnyProtocol, FramedAny, ProbeVerdict};
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use proxima_protocols::dns::{DnsTcpCodec, DnsTcpOwnedFrame};

use crate::config::{DnsServerConfig, resolve_config};
use crate::framed_app::{DnsFramedApp, DnsTcpOutcome, shed_reply};
use crate::pipes::DnsPipeHandle;
use crate::wire::header_is_query;

/// RFC 1035 §4.2.2's big-endian length prefix, and §4.1.1's fixed header —
/// the smallest a framed TCP message can legally be is one of each.
const TCP_LENGTH_PREFIX_BYTES: usize = 2;
const DNS_HEADER_BYTES: usize = 12;
const MIN_TCP_FRAME_PREFIX: usize = TCP_LENGTH_PREFIX_BYTES + DNS_HEADER_BYTES;

/// The concrete [`FramedAny`] instantiation DNS-over-TCP drives — `Probe`/
/// `Shed` are plain `fn` items (no captured state), so `DnsAnyProtocol`
/// needs no generic parameters of its own to name this type.
type DnsFramedAny = FramedAny<
    DnsTcpCodec,
    DnsFramedApp,
    fn(&[u8]) -> ProbeVerdict,
    fn(ShedReason, &DnsTcpOwnedFrame) -> DnsTcpOutcome,
>;

/// DNS-over-TCP wire candidate for the open universal listener. Two UDP
/// siblings exist, for two different callers: [`crate::DnsUdpAnyProtocol`]
/// is the `AnyProtocol` candidate `Listener::builder().dns(handler)`
/// registers ALONGSIDE this one, under the SAME `.any()`-fanned listener,
/// so one bind answers both transports on one port — see that type's module
/// doc. [`crate::DnsDatagramProtocol`] is a separate, standalone
/// `DatagramProtocolListenProtocol` for a caller who wants a dedicated
/// UDP-only listener with no TCP sibling at all; it shares the same
/// [`DnsPipeHandle`] contract but is never what `.dns(handler)` registers.
///
/// ```
/// use proxima_listen::any::AnyProtocol;
/// use proxima_dns::{DnsAnyProtocol, DnsPipeRequest, DnsPipeReply, into_dns_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct Unimplemented; // no client dials in this doctest
/// impl SendPipe for Unimplemented {
///     type In = DnsPipeRequest;
///     type Out = DnsPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
///         unreachable!()
///     }
/// }
///
/// let candidate = DnsAnyProtocol::new("dns", into_dns_handle(Unimplemented));
/// assert_eq!(candidate.name(), "dns");
/// ```
pub struct DnsAnyProtocol {
    label: String,
    handler: DnsPipeHandle,
    config: DnsServerConfig,
}

impl DnsAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: DnsPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: DnsServerConfig::default(),
        }
    }

    /// Replaces the default [`DnsServerConfig`]; a `dns` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: DnsServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the [`FramedAny`] this connection drives, from `config`
    /// (already resolved against the listener spec).
    fn build(&self, config: &DnsServerConfig) -> DnsFramedAny {
        FramedAny::new(
            self.label.clone(),
            DnsTcpCodec::new(config.max_message_bytes),
            DnsFramedApp::new(self.label.clone(), self.handler.clone()),
            probe as fn(&[u8]) -> ProbeVerdict,
            shed_reply as fn(ShedReason, &DnsTcpOwnedFrame) -> DnsTcpOutcome,
            MIN_TCP_FRAME_PREFIX,
        )
    }
}

fn probe(prefix: &[u8]) -> ProbeVerdict {
    if prefix.len() < MIN_TCP_FRAME_PREFIX {
        return ProbeVerdict::NeedMore {
            at_least: MIN_TCP_FRAME_PREFIX,
        };
    }
    let declared_len = usize::from(u16::from_be_bytes([prefix[0], prefix[1]]));
    if declared_len < DNS_HEADER_BYTES {
        // no framed DNS message is ever shorter than its own header.
        return ProbeVerdict::No;
    }
    if header_is_query(&prefix[TCP_LENGTH_PREFIX_BYTES..]) {
        ProbeVerdict::Match { consumed: 0 }
    } else {
        ProbeVerdict::No
    }
}

impl AnyProtocol for DnsAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        MIN_TCP_FRAME_PREFIX
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        probe(prefix)
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            let framed = self.build(&config);
            framed.drive(stream, handler, spec, peer, admission).await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::SendPipe;
    use proxima_protocols::dns::encode;

    fn framed_query(id: u16) -> Vec<u8> {
        let mut message = Vec::new();
        encode::encode_query(
            id,
            true,
            encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut message,
        )
        .unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&u16::try_from(message.len()).unwrap().to_be_bytes());
        framed.extend_from_slice(&message);
        framed
    }

    fn handler() -> DnsPipeHandle {
        struct EchoAnswer;
        impl SendPipe for EchoAnswer {
            type In = crate::pipes::DnsPipeRequest;
            type Out = crate::pipes::DnsPipeReply;
            type Err = ProximaError;

            async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
                Ok(crate::pipes::DnsPipeReply::typed(
                    200,
                    crate::pipes::DnsAnswer::ok(Vec::new()),
                ))
            }
        }
        crate::pipes::into_dns_handle(EchoAnswer)
    }

    #[test]
    fn probe_needs_more_below_the_minimum_frame() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        assert_eq!(
            protocol.probe(&[0u8; 4]),
            ProbeVerdict::NeedMore {
                at_least: MIN_TCP_FRAME_PREFIX
            }
        );
    }

    #[test]
    fn probe_matches_a_well_formed_single_question_frame() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let framed = framed_query(1234);
        assert_eq!(
            protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_rejects_a_multi_question_header() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let mut framed = framed_query(1234);
        // bump QDCOUNT (prefix-relative bytes 6..8) to 2.
        framed[7] = 2;
        assert_eq!(
            protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]),
            ProbeVerdict::No
        );
    }

    #[test]
    fn probe_rejects_a_framed_response_not_a_query() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let mut framed = framed_query(1234);
        // set QR (prefix-relative byte 4, header byte 2, high bit): a reply
        // arriving at a server is not this candidate's traffic.
        framed[4] |= 0b1000_0000;
        assert_eq!(
            protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]),
            ProbeVerdict::No
        );
    }

    #[test]
    fn probe_rejects_a_non_standard_opcode() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let mut framed = framed_query(1234);
        // OPCODE 5 (UPDATE, RFC 2136) — a legal DNS message this crate's
        // single-question query path cannot serve.
        framed[4] |= 5 << 3;
        assert_eq!(
            protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]),
            ProbeVerdict::No
        );
    }
}
