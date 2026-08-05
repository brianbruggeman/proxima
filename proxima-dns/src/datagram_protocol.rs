//! `DnsDatagramProtocol` — DNS-over-UDP as a
//! [`proxima_listen::stream::DatagramProtocol`] state machine, the sans-IO
//! seam [`proxima_listen::stream::DatagramProtocolListenProtocol`] drives.
//! Mirrors `proxima_quic::native::listener::Listener<P>`'s own `impl
//! DatagramProtocol` (the reference implementation this crate's design was
//! read against) — but DNS query/response is stateless request/reply, not a
//! multi-round-trip connection: `on_timeout`/`next_deadline` are trivial
//! (no retransmit state to arm a timer for), and every unit of work starts
//! and ends inside one `on_datagram` call.
//!
//! `on_datagram` runs `crate::wire::answer_datagram` — the decode / dispatch
//! / encode unit both UDP listeners share — and stages the reply for
//! [`DatagramProtocol::transmit`] to drain. A malformed or oversized datagram,
//! or a handler failure, is logged and dropped there, which is why this
//! state machine's error type is [`Infallible`]: nothing about one bad query
//! can tear down a connectionless listener (the same contract
//! [`proxima_listen::stream::DatagramListenProtocol`] and the QUIC listener
//! both hold).
//!
//! Register with [`DnsDatagramProtocol::listen_protocol`], mirroring
//! `Listener::listen_protocol`'s single reference point wiring a
//! `DatagramProtocol` impl onto its driver.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use proxima_core::time::Instant;
use proxima_listen::stream::{DatagramProtocol, DatagramProtocolListenProtocol};
use proxima_telemetry::warn;

use crate::config::DnsServerConfig;
use crate::pipes::DnsPipeHandle;
use crate::wire::answer_datagram;

/// DNS-over-UDP query/response state machine. Construct fresh per `serve()`
/// via [`Self::listen_protocol`]; holds no per-peer state between calls —
/// `pending` is purely a one-tick staging queue between `on_datagram` and
/// `transmit`, never carried across ticks with meaningful backlog (a
/// well-behaved deployment drains it every tick, same as
/// `DatagramListenProtocol`'s reply staging).
pub struct DnsDatagramProtocol {
    label: String,
    handler: DnsPipeHandle,
    config: Arc<DnsServerConfig>,
    pending: VecDeque<(Bytes, SocketAddr)>,
}

impl DnsDatagramProtocol {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        handler: DnsPipeHandle,
        config: Arc<DnsServerConfig>,
    ) -> Self {
        Self {
            label: label.into(),
            handler,
            config,
            pending: VecDeque::new(),
        }
    }

    /// Build the [`DatagramProtocolListenProtocol`] driving a fresh
    /// [`DnsDatagramProtocol`] per `serve()` invocation — the single
    /// reference point wiring the DNS UDP server onto the runtime-agnostic
    /// [`DatagramProtocol`] seam, mirroring
    /// `proxima_quic::native::listener::Listener::listen_protocol`.
    #[must_use]
    pub fn listen_protocol(
        label: impl Into<String>,
        handler: DnsPipeHandle,
        config: DnsServerConfig,
    ) -> DatagramProtocolListenProtocol<impl Fn() -> Self + Send + Sync + 'static, Self> {
        let label = label.into();
        let config = Arc::new(config);
        let build = move || Self::new(label.clone(), handler.clone(), Arc::clone(&config));
        DatagramProtocolListenProtocol::new("dns", build)
    }
}

impl DatagramProtocol for DnsDatagramProtocol {
    type Err = Infallible;

    async fn on_datagram(
        &mut self,
        _now: Instant,
        peer: SocketAddr,
        datagram: &[u8],
    ) -> Result<(), Self::Err> {
        let reply = answer_datagram(
            &self.label,
            &self.handler,
            self.config.max_message_bytes,
            peer,
            datagram,
        )
        .await;
        if let Some(bytes) = reply {
            self.pending.push_back((Bytes::from(bytes), peer));
        }
        Ok(())
    }

    async fn on_timeout(&mut self, _now: Instant) -> Result<(), Self::Err> {
        // Stateless request/reply: nothing ever arms a deadline (see
        // `next_deadline`), so the driver never calls this.
        Ok(())
    }

    fn next_deadline(&self) -> Option<Instant> {
        // No retransmit state — an idle DNS listener costs zero wakeups
        // beyond the recv arm.
        None
    }

    async fn transmit(
        &mut self,
        _now: Instant,
        buf: &mut [u8],
    ) -> Result<Option<(usize, SocketAddr)>, Self::Err> {
        // `Ok(None)` ends the driver's whole drain loop for this tick, and
        // `next_deadline` never arms one — so skipping past an undeliverable
        // reply has to continue the drain, not return None, or every reply
        // staged behind it in the same batched tick waits for the next
        // inbound datagram to ship.
        while let Some((bytes, peer)) = self.pending.pop_front() {
            if bytes.len() > buf.len() {
                warn!(
                    label = %self.label,
                    %peer,
                    len = bytes.len(),
                    scratch = buf.len(),
                    "dns reply exceeds the listener's transmit scratch buffer; dropping"
                );
                continue;
            }
            let len = bytes.len();
            buf[..len].copy_from_slice(&bytes);
            return Ok(Some((len, peer)));
        }
        Ok(None)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;
    use std::task::{Context, Poll};

    use futures::task::noop_waker;
    use proxima_core::ProximaError;
    use proxima_primitives::pipe::SendPipe;
    use proxima_protocols::dns::encode;

    use super::*;
    use crate::pipes::{DnsAnswer, DnsAnswerRecord, DnsPipeReply, into_dns_handle};

    fn poll_once<Fut: Future>(future: Fut) -> Fut::Output {
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut pinned = Box::pin(future);
        loop {
            if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    fn query_bytes_for(id: u16, name: &str) -> Vec<u8> {
        let mut out = Vec::new();
        encode::encode_query(
            id,
            true,
            encode::EncodeQuestion {
                name,
                qtype: 1,
                qclass: 1,
            },
            &mut out,
        )
        .unwrap();
        out
    }

    fn example_com_query_bytes(id: u16) -> Vec<u8> {
        query_bytes_for(id, "example.com.")
    }

    struct StaticAnswerPipe;

    impl SendPipe for StaticAnswerPipe {
        type In = crate::pipes::DnsPipeRequest;
        type Out = DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, request: Self::In) -> Result<Self::Out, ProximaError> {
            let record = DnsAnswerRecord {
                name: request.payload.name.clone(),
                rtype: 1,
                rclass: 1,
                ttl: 60,
                rdata: encode::ipv4_rdata(core::net::Ipv4Addr::new(93, 184, 216, 34)).to_vec(),
            };
            Ok(DnsPipeReply::typed(200, DnsAnswer::ok(vec![record])))
        }
    }

    #[test]
    fn on_datagram_stages_a_reply_transmit_then_drains() {
        let handler = into_dns_handle(StaticAnswerPipe);
        let mut protocol =
            DnsDatagramProtocol::new("dns-test", handler, Arc::new(DnsServerConfig::default()));
        let peer = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 7)),
            40000,
        );
        let query_bytes = example_com_query_bytes(1234);

        let now = Instant::from_monotonic(core::time::Duration::ZERO);
        poll_once(protocol.on_datagram(now, peer, &query_bytes)).unwrap();

        let mut scratch = [0u8; 2048];
        let (len, sent_to) = poll_once(protocol.transmit(now, &mut scratch))
            .unwrap()
            .unwrap();
        assert_eq!(sent_to, peer);

        let message = proxima_protocols::dns::codec_trait::parse_message(&scratch[..len]).unwrap();
        assert_eq!(message.header.id, 1234);
        assert!(message.header.flags.is_response());
        assert_eq!(message.header.ancount, 1);

        assert!(
            poll_once(protocol.transmit(now, &mut scratch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn malformed_datagram_is_dropped_not_propagated() {
        let handler = into_dns_handle(StaticAnswerPipe);
        let mut protocol =
            DnsDatagramProtocol::new("dns-test", handler, Arc::new(DnsServerConfig::default()));
        let peer = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 9)),
            41000,
        );
        let now = Instant::from_monotonic(core::time::Duration::ZERO);

        let outcome = poll_once(protocol.on_datagram(now, peer, &[0u8; 4]));
        assert!(
            outcome.is_ok(),
            "a malformed datagram is dropped, not an Err"
        );

        let mut scratch = [0u8; 2048];
        assert!(
            poll_once(protocol.transmit(now, &mut scratch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn oversized_datagram_is_dropped() {
        let handler = into_dns_handle(StaticAnswerPipe);
        let config = DnsServerConfig::builder().max_message_bytes(20).build();
        let mut protocol = DnsDatagramProtocol::new("dns-test", handler, Arc::new(config));
        let peer = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 9)),
            41000,
        );
        let now = Instant::from_monotonic(core::time::Duration::ZERO);
        let query_bytes = example_com_query_bytes(7);
        assert!(query_bytes.len() > 20);

        poll_once(protocol.on_datagram(now, peer, &query_bytes)).unwrap();

        let mut scratch = [0u8; 2048];
        assert!(
            poll_once(protocol.transmit(now, &mut scratch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_reply_too_large_for_the_scratch_does_not_strand_the_next_one() {
        let handler = into_dns_handle(StaticAnswerPipe);
        let mut protocol =
            DnsDatagramProtocol::new("dns-test", handler, Arc::new(DnsServerConfig::default()));
        let peer = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(10, 0, 0, 11)),
            42000,
        );
        let now = Instant::from_monotonic(core::time::Duration::ZERO);

        // the handler echoes the query name into its answer, so a near-maximal
        // name makes a reply no 128-byte transmit scratch can carry. The
        // driver batches a whole recv burst before draining, so both replies
        // stage in the same tick.
        let long_label = "a".repeat(60);
        let long_name = format!("{long_label}.{long_label}.{long_label}.");
        poll_once(protocol.on_datagram(now, peer, &query_bytes_for(1, &long_name))).unwrap();
        poll_once(protocol.on_datagram(now, peer, &query_bytes_for(2, "b.io."))).unwrap();

        let mut scratch = [0u8; 128];
        let (len, _) = poll_once(protocol.transmit(now, &mut scratch))
            .unwrap()
            .expect("the second reply still ships once the oversized one is skipped");
        let message = proxima_protocols::dns::codec_trait::parse_message(&scratch[..len]).unwrap();
        assert_eq!(message.header.id, 2);

        assert!(
            poll_once(protocol.transmit(now, &mut scratch))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn next_deadline_is_always_none_stateless_server() {
        let handler = into_dns_handle(StaticAnswerPipe);
        let protocol =
            DnsDatagramProtocol::new("dns-test", handler, Arc::new(DnsServerConfig::default()));
        assert_eq!(protocol.next_deadline(), None);
    }
}
