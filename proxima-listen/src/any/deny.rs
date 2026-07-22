//! [`DenySignature`] — an [`AnyProtocol`] candidate that recognizes a fixed
//! malicious/scanner byte literal and, once matched, records a
//! [`Strike::Deny`] against the connecting peer and drops the connection —
//! no handler dispatch, ever. Registered ALONGSIDE the legit candidates
//! (h1, h2, pgwire, redis, ...) at the same open universal listener, so a
//! probe/scanner signature is reviewed by the SAME classifier every other
//! candidate is, rather than a bespoke pre-filter that could shadow real
//! traffic.
//!
//! `probe` is the identical literal-prefix compare
//! `H2PriorKnowledgeAnyProtocol::probe` (`proxima-http/src/any_listener.rs`)
//! already does — a fixed literal is a fixed literal, whether it is h2's
//! RFC 9113 preface or a scanner's known probe string.

use std::future::Future;
use std::pin::Pin;

use proxima_core::ProximaError;
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use serde_json::Value;

use super::probe::{AnyHandler, AnyProtocol, ProbeVerdict};
use crate::admission::{BlacklistTable, ConnAdmission, Strike};

/// One fixed byte-literal a `.deny(name, literal)` builder call registers.
/// Holds a concrete [`BlacklistTable`] (not a generic closure) — this
/// candidate is erased to `Arc<dyn AnyProtocol>` at registration anyway, so
/// there is no advantage to genericizing over the table's type, only cost.
pub struct DenySignature {
    name: String,
    literal: Vec<u8>,
    priority: u16,
    blacklist: BlacklistTable,
}

impl DenySignature {
    /// `priority` defaults to [`crate::sized::ANY_DENY_PRIORITY_DEFAULT`] —
    /// deliberately high, so a deny match is never HELD waiting on a
    /// lower-priority legit candidate still deciding (see
    /// [`crate::any::Classifier`]'s priority-ordered-wait rule): a
    /// positively-identified malicious literal should resolve and drop the
    /// connection as soon as it is seen, not after every other candidate
    /// has had its turn.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        literal: impl Into<Vec<u8>>,
        blacklist: BlacklistTable,
    ) -> Self {
        Self {
            name: name.into(),
            literal: literal.into(),
            priority: crate::sized::ANY_DENY_PRIORITY_DEFAULT,
            blacklist,
        }
    }

    /// Override the default priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u16) -> Self {
        self.priority = priority;
        self
    }
}

impl AnyProtocol for DenySignature {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u16 {
        self.priority
    }

    fn max_prefix_bytes(&self) -> usize {
        self.literal.len()
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(self.literal.len());
        if prefix[..compare_len] != self.literal[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < self.literal.len() {
            return ProbeVerdict::NeedMore {
                at_least: self.literal.len(),
            };
        }
        ProbeVerdict::Match {
            consumed: self.literal.len(),
        }
    }

    /// Record the strike and drop the stream — no handler dispatch, ever.
    /// `handler`/`spec`/`admission` are unused: a deny match has nothing to
    /// dispatch to and no request-level admission boundary of its own.
    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        _spec: &'a Value,
        peer: Option<PeerInfo>,
        _admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let peer_ip = crate::peer_ip(peer.as_ref());
            self.blacklist
                .record_strike(peer_ip, proxima_core::time::now(), Strike::Deny);
            drop(stream);
            Ok(())
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::admission::BlacklistConfig;
    use std::net::{IpAddr, Ipv4Addr};

    const PEER: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

    fn peer_info() -> PeerInfo {
        PeerInfo::Tcp(std::net::SocketAddr::new(PEER, 4242))
    }

    #[test]
    fn probe_matches_the_full_literal_and_rejects_a_divergent_prefix() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let deny = DenySignature::new("scanner", b"BADSCAN\r\n".to_vec(), table);
        assert!(matches!(
            deny.probe(b"BADSCAN\r\n"),
            ProbeVerdict::Match { consumed: 9 }
        ));
        assert!(matches!(
            deny.probe(b"GET / HTTP/1.1\r\n"),
            ProbeVerdict::No
        ));
    }

    #[test]
    fn probe_needs_more_bytes_on_a_live_prefix() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let deny = DenySignature::new("scanner", b"BADSCAN\r\n".to_vec(), table);
        match deny.probe(b"BADS") {
            ProbeVerdict::NeedMore { at_least } => assert_eq!(at_least, 9),
            other => panic!("expected NeedMore{{at_least: 9}}, got {other:?}"),
        }
    }

    #[test]
    fn default_priority_comes_from_the_sized_floor() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let deny = DenySignature::new("scanner", b"X".to_vec(), table);
        assert_eq!(deny.priority(), crate::sized::ANY_DENY_PRIORITY_DEFAULT);
    }

    #[test]
    fn with_priority_overrides_the_default() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let deny = DenySignature::new("scanner", b"X".to_vec(), table).with_priority(42);
        assert_eq!(deny.priority(), 42);
    }

    struct NeverPolled;
    impl futures::io::AsyncRead for NeverPolled {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            _buf: &mut [u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Ready(Ok(0))
        }
    }
    impl futures::io::AsyncWrite for NeverPolled {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &[u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }
        fn poll_close(
            self: Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }
    }
    impl StreamConnection for NeverPolled {
        fn peer(&self) -> Option<PeerInfo> {
            Some(peer_info())
        }
    }

    // Real-shape proof: `drive` records a `Strike::Deny` against the
    // peer's IP (not a handler dispatch) and the table then reports the
    // peer banned — end to end through `DenySignature`, not the table
    // directly.
    #[proxima::test]
    async fn drive_records_a_deny_strike_that_bans_the_peer() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let deny = DenySignature::new("scanner", b"BADSCAN\r\n".to_vec(), table.clone());
        let admission = ConnAdmission::unbounded();
        let spec = Value::Null;
        let outcome = deny
            .drive(
                Box::new(NeverPolled),
                crate::any::erase_handler(7_u8),
                &spec,
                Some(peer_info()),
                &admission,
            )
            .await;
        assert!(outcome.is_ok(), "drive must record + drop, never error");
        assert!(
            table.is_banned(PEER, proxima_core::time::now()),
            "one deny strike must ban the peer at the default threshold"
        );
    }
}
