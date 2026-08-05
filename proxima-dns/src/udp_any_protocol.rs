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
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::config::{DnsServerConfig, resolve_config};
use crate::pipes::DnsPipeHandle;
use crate::wire::{answer_datagram, header_is_query};

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

/// Positive-match probe over the RAW message (no length prefix): the fixed
/// header must be present and must open a query — the same predicate
/// [`crate::any_protocol`] enforces for the TCP wire, just without skipping a
/// 2-byte prefix first.
fn probe(prefix: &[u8]) -> ProbeVerdict {
    if prefix.len() < MIN_UDP_HEADER_BYTES {
        return ProbeVerdict::NeedMore {
            at_least: MIN_UDP_HEADER_BYTES,
        };
    }
    if header_is_query(prefix) {
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
        _ => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
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
            let peer = peer_addr(peer.as_ref());

            // The classifier already replayed however many bytes it read to
            // resolve the match at the front of `stream`; whatever remains
            // of the one already-received datagram follows immediately —
            // `read_to_end` reassembles the full original message exactly
            // once, then sees EOF (see `DatagramAsStream`'s doc).
            let mut datagram = Vec::new();
            stream.read_to_end(&mut datagram).await?;

            // The handler comes from construction (`.dns(handler)`), not from
            // the `AnyHandler` the classifier passes — see `AnyProtocol::drive`'s
            // own doc for why that asymmetry is the documented pattern here;
            // pgwire's and redis's candidates do the same.
            let reply = answer_datagram(
                &self.label,
                &self.handler,
                config.max_message_bytes,
                peer,
                &datagram,
            )
            .await;
            if let Some(bytes) = reply {
                stream.write_all(&bytes).await?;
            }
            stream.close().await?;
            Ok(())
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn example_com_query() -> Vec<u8> {
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
        message
    }

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
        let message = example_com_query();
        assert_eq!(
            probe(&message[..MIN_UDP_HEADER_BYTES]),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_rejects_a_multi_question_header() {
        let mut message = example_com_query();
        // QDCOUNT sits at header-relative bytes 4..6 — bump it to 2.
        message[5] = 2;
        assert_eq!(probe(&message[..MIN_UDP_HEADER_BYTES]), ProbeVerdict::No);
    }

    #[test]
    fn probe_rejects_a_response_not_a_query() {
        let mut message = example_com_query();
        // set QR (header byte 2, high bit): a reply arriving at a server is
        // not this candidate's traffic, however well-formed it looks.
        message[2] |= 0b1000_0000;
        assert_eq!(probe(&message[..MIN_UDP_HEADER_BYTES]), ProbeVerdict::No);
    }

    #[test]
    fn probe_rejects_a_non_standard_opcode() {
        let mut message = example_com_query();
        // OPCODE 4 (NOTIFY, RFC 1996) — legal DNS, not a query this crate
        // can answer.
        message[2] |= 4 << 3;
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
