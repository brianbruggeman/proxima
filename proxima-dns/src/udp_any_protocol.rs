//! `DnsUdpAnyProtocol` — DNS-over-UDP (RFC 1035 §4.2.1: the raw message,
//! no length prefix) as an [`AnyProtocol`] candidate for the open universal
//! listener — the datagram-side sibling of [`crate::DnsAnyProtocol`]
//! (DNS-over-TCP). `wants_datagram` returns `true`, so
//! `Listener::builder().dns(handler)` registers BOTH candidates under
//! `.any()` and reaches DNS over either transport through the IDENTICAL
//! classify+drive path a TCP-only candidate already uses — `drive`'s `Box
//! <dyn StreamConnection>` needs no special case for a UDP-sourced
//! connection (see `proxima_listen::any::AnyProtocol::wants_datagram`'s own
//! doc: it is a one-shot adapter over the single already-received
//! datagram, `AsyncRead`/`AsyncWrite` all the way).
//!
//! This is deliberately a SEPARATE type from [`crate::DnsDatagramProtocol`]
//! (the standalone [`proxima_listen::stream::DatagramProtocol`] state
//! machine `DatagramProtocolListenProtocol` drives): the two solve the same
//! wire with different plumbing for different callers.
//! `DnsDatagramProtocol` stays available for a caller who wants a
//! dedicated UDP-only listener with no TCP sibling at all (its own
//! `listen_protocol` constructor, its own tests, unchanged by this module).
//! `DnsUdpAnyProtocol` is what `.dns(handler)` now registers so DNS-over-TCP
//! and DNS-over-UDP share one port number under `.any()`.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value;

use proxima_codec::Datagram;
use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use proxima_protocols::dns::codec_trait::DnsDatagramCodec;

use crate::config::DnsServerConfig;
use crate::pipes::DnsPipeHandle;
use crate::wire::{answer_to_wire, message_to_query};

/// A raw DNS message is never shorter than its own fixed 12-byte header
/// (RFC 1035 §4.1.1) — no 2-byte length prefix on this wire, unlike
/// [`crate::DnsAnyProtocol`]'s TCP framing.
const MIN_UDP_HEADER_BYTES: usize = 12;

/// DNS-over-UDP wire candidate for the open universal listener. See the
/// module doc for why this is a distinct type from [`crate::DnsDatagramProtocol`].
pub struct DnsUdpAnyProtocol {
    label: String,
    handler: DnsPipeHandle,
    config: DnsServerConfig,
}

impl DnsUdpAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: DnsPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: DnsServerConfig::default(),
        }
    }

    /// Replaces the default [`DnsServerConfig`]; a `dns` object in the
    /// listener spec still wins at drive time — mirrors
    /// [`crate::DnsAnyProtocol::with_config`].
    #[must_use]
    pub fn with_config(mut self, config: DnsServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(base: &DnsServerConfig, spec: &Value) -> Result<DnsServerConfig, ProximaError> {
    match spec.get("dns") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("dns spec: {error}"))),
    }
}

/// Positive-match probe over the RAW message (no length prefix): the fixed
/// header must be present and its `QDCOUNT` (header-relative bytes 4..6)
/// must be exactly `1` — the same single-question contract
/// [`crate::any_protocol::probe`] enforces for the TCP wire, just without
/// skipping a 2-byte prefix first.
fn probe(prefix: &[u8]) -> ProbeVerdict {
    if prefix.len() < MIN_UDP_HEADER_BYTES {
        return ProbeVerdict::NeedMore {
            at_least: MIN_UDP_HEADER_BYTES,
        };
    }
    let qdcount = u16::from_be_bytes([prefix[4], prefix[5]]);
    if qdcount == 1 {
        ProbeVerdict::Match { consumed: 0 }
    } else {
        ProbeVerdict::No
    }
}

/// Recovers the UDP sender's address from the [`PeerInfo`]
/// `.any()`'s datagram fan-in stamps on a UDP-sourced connection
/// (`PeerInfo::Tcp` reused for "an IP:port peer" regardless of L4 transport
/// — see `proxima_http::any_listener::DatagramAsStream::peer`'s own doc).
/// `None` (no peer info at all) can't happen on this path in practice; it
/// degrades to the unspecified address rather than failing the whole
/// query, matching this module's "never tear down the listener over one
/// malformed/unusual query" contract.
fn peer_addr(peer: Option<&PeerInfo>) -> SocketAddr {
    match peer {
        Some(PeerInfo::Tcp(addr)) => *addr,
        _ => SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED), 0),
    }
}

impl AnyProtocol for DnsUdpAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        MIN_UDP_HEADER_BYTES
    }

    fn wants_datagram(&self) -> bool {
        true
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        probe(prefix)
    }

    fn drive<'a>(
        &'a self,
        mut stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        _admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            // Carries its own engine from construction (`.dns(handler)`),
            // mirroring `DnsAnyProtocol`'s identical asymmetry — see
            // `AnyProtocol::drive`'s own doc for why this is a documented
            // pattern, not a shortcut: pgwire/redis candidates do the same.
            let dispatch = &self.handler;
            let peer = peer_addr(peer.as_ref());

            // The classifier already replayed however many bytes it read to
            // resolve the match at the front of `stream`; whatever remains
            // of the one already-received datagram follows immediately —
            // `read_to_end` reassembles the full original message exactly
            // once, then sees EOF (see `DatagramAsStream`'s doc).
            let mut datagram = Vec::new();
            stream.read_to_end(&mut datagram).await?;

            if datagram.len() > config.max_message_bytes {
                proxima_telemetry::warn!(
                    label = %self.label,
                    %peer,
                    len = datagram.len(),
                    limit = config.max_message_bytes,
                    "dns query exceeds message limit; dropping"
                );
                return Ok(());
            }
            let addressed = match DnsDatagramCodec.decode(peer, &datagram) {
                Ok(addressed) => addressed,
                Err(error) => {
                    proxima_telemetry::warn!(label = %self.label, %peer, ?error, "dns query failed to parse; dropping");
                    return Ok(());
                }
            };
            let Some(query) = message_to_query(&addressed.message) else {
                proxima_telemetry::warn!(label = %self.label, %peer, "dns query is not exactly one question; dropping");
                return Ok(());
            };

            let request = crate::pipes::DnsPipeRequest {
                method: proxima_primitives::pipe::Method::from_wire(bytes::Bytes::from_static(
                    b"DNS",
                )),
                path: bytes::Bytes::from_static(b"/"),
                query: proxima_primitives::pipe::header_list::HeaderList::new(),
                metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
                payload: query.clone(),
                stream: None,
                context: proxima_primitives::pipe::request::RequestContext::default(),
            };
            let reply = match SendPipe::call(dispatch, request).await {
                Ok(reply) => reply,
                Err(error) => {
                    proxima_telemetry::warn!(label = %self.label, %peer, ?error, "dns handler pipe failed; dropping");
                    return Ok(());
                }
            };

            let mut out = Vec::new();
            if let Err(error) = answer_to_wire(&query, &reply.payload, &mut out) {
                proxima_telemetry::warn!(label = %self.label, %peer, ?error, "dns answer failed to encode; dropping");
                return Ok(());
            }
            stream.write_all(&out).await?;
            stream.close().await?;
            Ok(())
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn probe_needs_more_below_the_fixed_header() {
        assert_eq!(
            probe(&[0_u8; 4]),
            ProbeVerdict::NeedMore {
                at_least: MIN_UDP_HEADER_BYTES
            }
        );
    }

    #[test]
    fn probe_matches_a_well_formed_single_question_header_with_no_length_prefix() {
        let mut message = Vec::new();
        proxima_protocols::dns::encode::encode_query(
            1234,
            true,
            proxima_protocols::dns::encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut message,
        )
        .unwrap();
        assert_eq!(
            probe(&message[..MIN_UDP_HEADER_BYTES]),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_rejects_a_multi_question_header() {
        let mut message = Vec::new();
        proxima_protocols::dns::encode::encode_query(
            1234,
            true,
            proxima_protocols::dns::encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut message,
        )
        .unwrap();
        // QDCOUNT sits at header-relative bytes 4..6 — bump it to 2.
        message[5] = 2;
        assert_eq!(probe(&message[..MIN_UDP_HEADER_BYTES]), ProbeVerdict::No);
    }

    #[test]
    fn peer_addr_recovers_the_tcp_shaped_peer_info() {
        let addr = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 5)),
            4000,
        );
        assert_eq!(peer_addr(Some(&PeerInfo::Tcp(addr))), addr);
    }

    #[test]
    fn peer_addr_degrades_to_unspecified_with_no_peer_info() {
        let addr = peer_addr(None);
        assert_eq!(
            addr.ip(),
            std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)
        );
    }
}
