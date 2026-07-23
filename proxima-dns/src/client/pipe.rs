//! `DnsClientUpstream` — the async driver over [`DnsClientSession`],
//! mirroring `proxima_redis::client::pipe::RedisClientUpstream`'s split
//! between the sans-IO session and the runtime-touching transport. Redis's
//! upstream is generic over a `StreamUpstream` (TCP-shaped); DNS's primary
//! transport is UDP, so this drives the runtime-agnostic
//! [`DatagramSocket`]/[`DatagramFactory`] pair instead — the same seam
//! [`crate::datagram_protocol::DnsDatagramProtocol`] binds server-side via
//! `ServeContext::datagram_factory`, injected here the identical way so a
//! caller can hand in prime's, tokio's, or a fake test factory without this
//! crate naming any concrete runtime.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::task::Poll;

use proxima_core::ProximaError;
use proxima_core::time::{now, timeout_at};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::DatagramFactory;

use crate::client::config::DnsResolverConfig;
use crate::client::session::DnsClientSession;
use crate::error::DnsClientError;
use crate::pipes::{DnsAnswer, DnsPipeReply, DnsPipeRequest, DnsQuery};

/// Receive-buffer size for one UDP reply. 4096 bytes covers every
/// EDNS0-negotiated response a stub resolver client advertises in
/// practice; a reply larger than this (rare, jumbo EDNS) is truncated by
/// the OS socket read the same way it would be by any fixed-size receive
/// buffer, and [`DnsClientSession::decode_response`] reports it as a wire
/// error rather than silently misinterpreting a partial message.
const MAX_UDP_REPLY_BYTES: usize = 4096;

/// Async resolver client: send a query, await the matching response.
/// Construct via [`Self::new`] with an injected [`DatagramFactory`] (the
/// same seam the listener side takes via `ServeContext::datagram_factory`)
/// and a [`DnsResolverConfig`].
pub struct DnsClientUpstream {
    factory: Arc<dyn DatagramFactory>,
    config: DnsResolverConfig,
}

impl DnsClientUpstream {
    #[must_use]
    pub fn new(factory: Arc<dyn DatagramFactory>, config: DnsResolverConfig) -> Self {
        Self { factory, config }
    }

    /// Send one query and await its matching reply, retrying up to
    /// `config.max_attempts` times on timeout or transport error (UDP has
    /// no delivery guarantee — see [`DnsResolverConfig::max_attempts`]'s
    /// doc). A resolver-side negative answer (NXDOMAIN, SERVFAIL) is not a
    /// retry trigger: it is a successful exchange, returned as
    /// `Ok(DnsAnswer { rcode, .. })`.
    ///
    /// # Errors
    /// [`DnsClientError::Timeout`] if every attempt's reply never arrives
    /// in time, or the last attempt's own [`DnsClientError`] (a transport
    /// or wire-decode failure) otherwise.
    pub async fn query(&self, name: &str, qtype: u16, qclass: u16) -> Result<DnsAnswer, DnsClientError> {
        let mut last_error = DnsClientError::Timeout(self.config.query_timeout_ms);
        for _ in 0..self.config.max_attempts.max(1) {
            match self.try_query(name, qtype, qclass).await {
                Ok(answer) => return Ok(answer),
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    async fn try_query(&self, name: &str, qtype: u16, qclass: u16) -> Result<DnsAnswer, DnsClientError> {
        let mut session = DnsClientSession::new();
        let (id, query_bytes) = session.encode_query(name, qtype, qclass, true)?;

        let local_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
        let mut socket = self.factory.bind(local_addr).map_err(DnsClientError::Io)?;
        let resolver_addr = self.config.resolver_addr()?;

        std::future::poll_fn(|cx| socket.poll_send_to(cx, &query_bytes, resolver_addr))
            .await
            .map_err(DnsClientError::Io)?;

        // One overall deadline for the whole exchange — computed once, so a
        // stray datagram from someone other than the resolver (discarded
        // below) can't reset the clock and starve the timeout.
        let deadline = now() + core::time::Duration::from_millis(self.config.query_timeout_ms);
        let mut buf = [0u8; MAX_UDP_REPLY_BYTES];
        loop {
            let recv = std::future::poll_fn(|cx| -> Poll<std::io::Result<(usize, SocketAddr)>> {
                socket.poll_recv_from(cx, &mut buf)
            });
            let (len, from) = timeout_at(deadline, recv)
                .await
                .map_err(|_elapsed| DnsClientError::Timeout(self.config.query_timeout_ms))?
                .map_err(DnsClientError::Io)?;
            if from != resolver_addr {
                // not our resolver's reply (stray/late packet) — keep
                // waiting against the same deadline.
                continue;
            }
            return session.decode_response(id, &buf[..len]);
        }
    }
}

impl SendPipe for DnsClientUpstream {
    type In = DnsPipeRequest;
    type Out = DnsPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: Self::In) -> Result<Self::Out, ProximaError> {
        let DnsQuery { name, qtype, qclass, .. } = request.payload;
        let answer = self
            .query(&name, qtype, qclass)
            .await
            .map_err(|error| ProximaError::Io(std::io::Error::other(error.to_string())))?;
        Ok(DnsPipeReply::typed(200, answer))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::io;
    use std::sync::Mutex;
    use std::task::{Context, Waker};

    use proxima_primitives::stream::DatagramSocket;
    use proxima_protocols::dns::codec_trait::parse_message;
    use proxima_protocols::dns::encode;

    use super::*;

    struct FakeResolverState {
        inbound: VecDeque<(Vec<u8>, SocketAddr)>,
        sent: Vec<(Vec<u8>, SocketAddr)>,
        waker: Option<Waker>,
        /// The address every subsequently-queued reply appears to come
        /// "from" — lets a test exercise the resolver-address mismatch
        /// guard by injecting a reply that claims a different sender, then
        /// flip it via [`FakeResolverSocket::set_reply_from`] and queue the
        /// "real" reply. Lives inside the shared state (not a per-clone
        /// field) so mutating it through the test's handle is visible to
        /// the internally-bound clone [`DnsClientUpstream`] actually polls.
        reply_from: SocketAddr,
    }

    #[derive(Clone)]
    struct FakeResolverSocket {
        state: Arc<Mutex<FakeResolverState>>,
        local: SocketAddr,
    }

    impl FakeResolverSocket {
        fn new(local: SocketAddr, reply_from: SocketAddr) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeResolverState {
                    inbound: VecDeque::new(),
                    sent: Vec::new(),
                    waker: None,
                    reply_from,
                })),
                local,
            }
        }

        fn set_reply_from(&self, addr: SocketAddr) {
            self.state.lock().unwrap().reply_from = addr;
        }

        /// Queue a reply built from the last sent query's id, echoing an A
        /// record answer for the queried name.
        fn queue_reply_to_last_query(&self) {
            let (query_bytes, reply_from) = {
                let state = self.state.lock().unwrap();
                let Some((query_bytes, _)) = state.sent.last().cloned() else {
                    return;
                };
                (query_bytes, state.reply_from)
            };
            let query_message = parse_message(&query_bytes).unwrap();
            let question = query_message.questions().next().unwrap().unwrap();
            let name = question.name.to_dotted();

            let mut response = Vec::new();
            let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
            let rdata = encode::ipv4_rdata(core::net::Ipv4Addr::new(93, 184, 216, 34));
            let record = encode::AnswerRecord {
                name: &name,
                rtype: 1,
                rclass: 1,
                ttl: 60,
                rdata: &rdata,
            };
            encode::encode_response(
                query_message.header.id,
                flags,
                encode::EncodeQuestion {
                    name: &name,
                    qtype: question.qtype,
                    qclass: question.qclass,
                },
                &[record],
                &mut response,
            )
            .unwrap();

            let mut state = self.state.lock().unwrap();
            state.inbound.push_back((response, reply_from));
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
        }
    }

    impl DatagramSocket for FakeResolverSocket {
        fn poll_recv_from(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<(usize, SocketAddr)>> {
            let mut state = self.state.lock().unwrap();
            match state.inbound.pop_front() {
                Some((bytes, from)) => {
                    let len = bytes.len().min(buf.len());
                    buf[..len].copy_from_slice(&bytes[..len]);
                    Poll::Ready(Ok((len, from)))
                }
                None => {
                    state.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }

        fn poll_send_to(&mut self, _cx: &mut Context<'_>, buf: &[u8], peer: SocketAddr) -> Poll<io::Result<usize>> {
            self.state.lock().unwrap().sent.push((buf.to_vec(), peer));
            Poll::Ready(Ok(buf.len()))
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(self.local)
        }
    }

    struct FakeResolverFactory {
        socket: FakeResolverSocket,
    }

    impl DatagramFactory for FakeResolverFactory {
        fn bind(&self, _addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
            Ok(Box::new(self.socket.clone()))
        }
    }

    fn resolver_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(198, 51, 100, 53)), 53)
    }

    #[proxima::test]
    async fn query_sends_and_decodes_the_matching_reply() {
        let socket = FakeResolverSocket::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            resolver_addr(),
        );
        let factory = Arc::new(FakeResolverFactory { socket: socket.clone() });
        let config = DnsResolverConfig::builder()
            .resolver_ip(resolver_addr().ip().to_string())
            .port(resolver_addr().port())
            .query_timeout_ms(200)
            .build();
        let client = DnsClientUpstream::new(factory, config);

        // race the query future against a background task that queues the
        // reply the instant a query has been sent.
        let query_future = client.query("example.com.", 1, 1);
        futures::pin_mut!(query_future);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        // first poll sends the query (and parks on recv).
        assert!(query_future.as_mut().poll(&mut cx).is_pending());
        socket.queue_reply_to_last_query();
        let answer = loop {
            match query_future.as_mut().poll(&mut cx) {
                Poll::Ready(result) => break result.unwrap(),
                Poll::Pending => continue,
            }
        };

        assert_eq!(answer.rcode, 0);
        assert_eq!(answer.records.len(), 1);
        assert_eq!(answer.records[0].name, "example.com.");
    }

    #[proxima::test]
    async fn a_reply_from_the_wrong_address_is_ignored_then_the_real_one_is_accepted() {
        // no sleeps: proves the mismatch guard filters and keeps polling by
        // queuing a stray reply from an off-target sender FIRST, then the
        // real resolver's reply — a client that accepted the first (wrong)
        // datagram would return its rcode/records instead of the correct
        // ones, so any pass here is a pass on the filter actually working.
        let wrong_sender = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 9)), 53);
        let socket = FakeResolverSocket::new(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0), wrong_sender);
        let factory = Arc::new(FakeResolverFactory { socket: socket.clone() });
        let config = DnsResolverConfig::builder()
            .resolver_ip(resolver_addr().ip().to_string())
            .port(resolver_addr().port())
            .query_timeout_ms(5_000)
            .max_attempts(1)
            .build();
        let client = DnsClientUpstream::new(factory, config);

        let query_future = client.query("example.com.", 1, 1);
        futures::pin_mut!(query_future);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(query_future.as_mut().poll(&mut cx).is_pending());

        // stray reply from `wrong_sender` — must be discarded, not returned.
        socket.queue_reply_to_last_query();
        assert!(
            query_future.as_mut().poll(&mut cx).is_pending(),
            "a reply from an unexpected sender must not resolve the query"
        );

        // now the real resolver answers.
        socket.set_reply_from(resolver_addr());
        socket.queue_reply_to_last_query();
        let answer = loop {
            match query_future.as_mut().poll(&mut cx) {
                Poll::Ready(result) => break result.unwrap(),
                Poll::Pending => continue,
            }
        };
        assert_eq!(answer.records.len(), 1);
        assert_eq!(answer.records[0].name, "example.com.");
    }
}
