//! `DnsClientUpstream` — the async driver over the sans-IO
//! [`crate::client::session`] pair, splitting the protocol from the
//! runtime-touching transport. DNS's primary transport is UDP, so this drives
//! the runtime-agnostic [`DatagramFactory`]/`DatagramSocket` pair — the same
//! seam the listener side binds via `ServeContext::datagram_factory`, injected
//! here the identical way so a caller can hand in prime's, tokio's, or a fake
//! test factory without this crate naming any concrete runtime.

use std::future::poll_fn;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::task::Poll;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use proxima_core::ProximaError;
use proxima_core::time::{now, timeout_at};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::{
    DatagramFactory, StreamConnection, StreamUpstream, StreamUpstreamExt,
};

use crate::client::config::DnsResolverConfig;
use crate::client::session::{decode_response_with_metadata, encode_query};
use crate::error::DnsClientError;
use crate::pipes::{DnsAnswer, DnsAnswerWithMetadata, DnsPipeReply, DnsPipeRequest, DnsQuery};

/// Receive-buffer size for one UDP reply. 4096 bytes covers every
/// EDNS0-negotiated response a stub resolver client advertises in
/// practice; a reply larger than this (rare, jumbo EDNS) is truncated by
/// the OS socket read the same way it would be by any fixed-size receive
/// buffer, and [`decode_response`] reports it as a wire error rather than
/// silently misinterpreting a partial message.
const MAX_UDP_REPLY_BYTES: usize = 4096;
/// DNS-over-TCP carries a two-byte length prefix (RFC 1035 §4.2.2).
const MAX_TCP_REPLY_BYTES: usize = u16::MAX as usize;

/// Async resolver client: send a query, await the matching response.
/// Construct via [`Self::new`] with an injected [`DatagramFactory`] (the
/// same seam the listener side takes via `ServeContext::datagram_factory`)
/// and a [`DnsResolverConfig`].
pub struct DnsClientUpstream {
    factory: Arc<dyn DatagramFactory>,
    tcp_upstream: Option<Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>>>,
    tcp_only: bool,
    config: DnsResolverConfig,
    /// Advances once per SEND — every query and every retransmission gets its
    /// own id. It lives here rather than per-call because `SendPipe::call`
    /// takes `&self`: a counter minted inside the call would restart at the
    /// same value for every query, which is how this crate previously put id
    /// 1 on every packet it ever sent.
    next_id: AtomicU16,
}

impl DnsClientUpstream {
    /// `new` never touches the network — a UDP socket only opens lazily on
    /// the first query. Building one is cheap and side-effect-free:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use proxima_dns::{DnsClientUpstream, DnsResolverConfig};
    /// use proxima_net::prime::PrimeDatagramFactory;
    ///
    /// let resolver = DnsClientUpstream::new(Arc::new(PrimeDatagramFactory), DnsResolverConfig::default());
    /// # let _ = resolver;
    /// ```
    #[must_use]
    pub fn new(factory: Arc<dyn DatagramFactory>, config: DnsResolverConfig) -> Self {
        Self {
            factory,
            tcp_upstream: None,
            tcp_only: false,
            config,
            // id 0 is legal (RFC 1035 places no restriction on it); starting
            // at 1 only keeps a fresh client's first query from reading like
            // an uninitialized field in a packet capture.
            next_id: AtomicU16::new(1),
        }
    }

    /// Attach an optional DNS-over-TCP dialer. When UDP returns a response
    /// with `TC=1`, [`Self::query`] retries the same exchange over this
    /// stream using DNS-over-TCP framing. The dialer is runtime-provided and
    /// may be backed by Prime, Tokio, or a deterministic test connection.
    #[must_use]
    pub fn with_tcp_upstream(
        mut self,
        tcp_upstream: Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>>,
    ) -> Self {
        self.tcp_upstream = Some(tcp_upstream);
        self
    }

    /// Use the attached stream transport for every exchange instead of
    /// sending a UDP probe first. This is the transport mode required by
    /// DNS-over-TLS and other stream-only resolver endpoints; the same
    /// bounded DNS-over-TCP framing and exchange deadline are retained.
    #[must_use]
    pub fn with_tcp_only(mut self) -> Self {
        self.tcp_only = true;
        self
    }

    fn next_id(&self) -> u16 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
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
    pub async fn query(
        &self,
        name: &str,
        qtype: u16,
        qclass: u16,
    ) -> Result<DnsAnswer, DnsClientError> {
        Ok(self.query_with_metadata(name, qtype, qclass).await?.answer)
    }

    /// Send one query and retain the response-envelope metadata needed for
    /// question validation and TCP fallback. The existing [`Self::query`]
    /// method remains the answer-only compatibility facade.
    pub async fn query_with_metadata(
        &self,
        name: &str,
        qtype: u16,
        qclass: u16,
    ) -> Result<DnsAnswerWithMetadata, DnsClientError> {
        let mut last_error = DnsClientError::Timeout(self.config.query_timeout_ms);
        for _ in 0..self.config.max_attempts.max(1) {
            if self.tcp_only {
                let id = self.next_id();
                let Some(tcp_upstream) = self.tcp_upstream.as_ref() else {
                    return Err(DnsClientError::Io(io::Error::new(
                        io::ErrorKind::NotConnected,
                        "stream-only DNS client has no stream upstream",
                    )));
                };
                match self
                    .try_tcp_query(tcp_upstream, id, name, qtype, qclass)
                    .await
                {
                    Ok(response) => return Ok(response),
                    Err(error) => last_error = error,
                }
                continue;
            }
            match self.try_query(name, qtype, qclass).await {
                Ok(response) if response.metadata.truncated => {
                    let Some(tcp_upstream) = self.tcp_upstream.as_ref() else {
                        return Ok(response);
                    };
                    let id = response
                        .metadata
                        .question
                        .as_ref()
                        .map_or(0, |question| question.id);
                    match self
                        .try_tcp_query(tcp_upstream, id, name, qtype, qclass)
                        .await
                    {
                        Ok(response) => return Ok(response),
                        Err(error) => last_error = error,
                    }
                }
                Ok(answer) => return Ok(answer),
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    async fn try_query(
        &self,
        name: &str,
        qtype: u16,
        qclass: u16,
    ) -> Result<DnsAnswerWithMetadata, DnsClientError> {
        let id = self.next_id();
        let query_bytes = encode_query(id, name, qtype, qclass, true)?;

        let local_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
        let mut socket = self.factory.bind(local_addr).map_err(DnsClientError::Io)?;
        let resolver_addr = self.config.resolver_addr()?;

        poll_fn(|cx| socket.poll_send_to(cx, &query_bytes, resolver_addr))
            .await
            .map_err(DnsClientError::Io)?;

        // One overall deadline for the whole exchange — computed once, so a
        // stray datagram from someone other than the resolver (discarded
        // below) can't reset the clock and starve the timeout.
        let deadline = now() + core::time::Duration::from_millis(self.config.query_timeout_ms);
        let mut buf = [0u8; MAX_UDP_REPLY_BYTES];
        loop {
            let recv = poll_fn(|cx| -> Poll<io::Result<(usize, SocketAddr)>> {
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
            return decode_response_with_metadata(id, &buf[..len]);
        }
    }

    async fn try_tcp_query(
        &self,
        tcp_upstream: &Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>>,
        id: u16,
        name: &str,
        qtype: u16,
        qclass: u16,
    ) -> Result<DnsAnswerWithMetadata, DnsClientError> {
        let deadline = now() + core::time::Duration::from_millis(self.config.query_timeout_ms);
        let query = encode_query(id, name, qtype, qclass, true)?;
        let frame_len = u16::try_from(query.len())
            .map_err(|_| DnsClientError::Wire("DNS-over-TCP query exceeds 65535 bytes".into()))?;
        let mut frame = Vec::with_capacity(query.len() + 2);
        frame.extend_from_slice(&frame_len.to_be_bytes());
        frame.extend_from_slice(&query);

        let mut connection = timeout_at(deadline, tcp_upstream.connect())
            .await
            .map_err(|_| DnsClientError::Timeout(self.config.query_timeout_ms))?
            .map_err(DnsClientError::Io)?;
        timeout_at(deadline, connection.write_all(&frame))
            .await
            .map_err(|_| DnsClientError::Timeout(self.config.query_timeout_ms))?
            .map_err(DnsClientError::Io)?;
        timeout_at(deadline, connection.flush())
            .await
            .map_err(|_| DnsClientError::Timeout(self.config.query_timeout_ms))?
            .map_err(DnsClientError::Io)?;

        let mut length = [0u8; 2];
        timeout_at(deadline, connection.read_exact(&mut length))
            .await
            .map_err(|_| DnsClientError::Timeout(self.config.query_timeout_ms))?
            .map_err(DnsClientError::Io)?;
        let response_len = usize::from(u16::from_be_bytes(length));
        if response_len == 0 || response_len > MAX_TCP_REPLY_BYTES {
            return Err(DnsClientError::Wire(
                "DNS-over-TCP response length is invalid".into(),
            ));
        }
        let mut response = vec![0u8; response_len];
        timeout_at(deadline, connection.read_exact(&mut response))
            .await
            .map_err(|_| DnsClientError::Timeout(self.config.query_timeout_ms))?
            .map_err(DnsClientError::Io)?;
        decode_response_with_metadata(id, &response)
    }
}

impl SendPipe for DnsClientUpstream {
    type In = DnsPipeRequest;
    type Out = DnsPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: Self::In) -> Result<Self::Out, ProximaError> {
        let DnsQuery {
            name,
            qtype,
            qclass,
            ..
        } = request.payload;
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

    use futures::io::{AsyncRead, AsyncWrite, Cursor};
    use proxima_primitives::stream::{DatagramSocket, PeerInfo, StreamConnection, StreamUpstream};
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

        fn queue_truncated_reply_to_last_query(&self) {
            let (query_bytes, reply_from) = {
                let state = self.state.lock().unwrap();
                let Some((query_bytes, _)) = state.sent.last().cloned() else {
                    return;
                };
                (query_bytes, state.reply_from)
            };
            let query_message = parse_message(&query_bytes).unwrap();
            let question = query_message.questions().next().unwrap().unwrap();
            let mut response = Vec::new();
            let base = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
            let flags = proxima_protocols::dns::Flags(base.0 | 0x0200);
            encode::encode_response(
                query_message.header.id,
                flags,
                encode::EncodeQuestion {
                    name: &question.name.to_dotted(),
                    qtype: question.qtype,
                    qclass: question.qclass,
                },
                &[],
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
        fn poll_recv_from(
            &mut self,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<(usize, SocketAddr)>> {
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

        fn poll_send_to(
            &mut self,
            _cx: &mut Context<'_>,
            buf: &[u8],
            peer: SocketAddr,
        ) -> Poll<io::Result<usize>> {
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

    struct FakeTcpConnection {
        response: Cursor<Vec<u8>>,
        writes: Arc<Mutex<Vec<u8>>>,
    }

    impl AsyncRead for FakeTcpConnection {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            std::pin::Pin::new(&mut self.get_mut().response).poll_read(cx, buf)
        }
    }

    impl AsyncWrite for FakeTcpConnection {
        fn poll_write(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.get_mut().writes.lock().unwrap().extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl StreamConnection for FakeTcpConnection {
        fn peer(&self) -> Option<PeerInfo> {
            None
        }
    }

    struct FakeTcpUpstream {
        response: Vec<u8>,
        writes: Arc<Mutex<Vec<u8>>>,
    }

    impl StreamUpstream for FakeTcpUpstream {
        type Conn = Box<dyn StreamConnection>;

        fn poll_connect(
            &self,
            _cx: &mut Context<'_>,
        ) -> Poll<io::Result<Self::Conn>> {
            Poll::Ready(Ok(Box::new(FakeTcpConnection {
                response: Cursor::new(self.response.clone()),
                writes: Arc::clone(&self.writes),
            })))
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
        let factory = Arc::new(FakeResolverFactory {
            socket: socket.clone(),
        });
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
    async fn truncated_udp_reply_falls_back_to_framed_tcp() {
        let socket = FakeResolverSocket::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            resolver_addr(),
        );
        let factory = Arc::new(FakeResolverFactory {
            socket: socket.clone(),
        });
        let tcp_response = {
            let mut message = Vec::new();
            let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
            encode::encode_response(
                1,
                flags,
                encode::EncodeQuestion {
                    name: "example.com.",
                    qtype: 1,
                    qclass: 1,
                },
                &[],
                &mut message,
            )
            .unwrap();
            let mut framed = Vec::new();
            framed.extend_from_slice(&(message.len() as u16).to_be_bytes());
            framed.extend_from_slice(&message);
            framed
        };
        let writes = Arc::new(Mutex::new(Vec::new()));
        let tcp = Arc::new(FakeTcpUpstream {
            response: tcp_response,
            writes: Arc::clone(&writes),
        });
        let config = DnsResolverConfig::builder()
            .resolver_ip(resolver_addr().ip().to_string())
            .port(resolver_addr().port())
            .query_timeout_ms(200)
            .max_attempts(1)
            .build();
        let client = DnsClientUpstream::new(factory, config).with_tcp_upstream(tcp);

        let query_future = client.query_with_metadata("example.com.", 1, 1);
        futures::pin_mut!(query_future);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(query_future.as_mut().poll(&mut cx).is_pending());
        socket.queue_truncated_reply_to_last_query();
        let response = loop {
            match query_future.as_mut().poll(&mut cx) {
                Poll::Ready(result) => break result.unwrap(),
                Poll::Pending => continue,
            }
        };

        assert_eq!(response.answer.rcode, 0);
        assert!(!response.metadata.truncated);
        let request = writes.lock().unwrap();
        let frame_len = usize::from(u16::from_be_bytes([request[0], request[1]]));
        assert_eq!(frame_len, request.len() - 2);
        assert_eq!(parse_message(&request[2..]).unwrap().header.id, 1);
    }

    #[proxima::test]
    async fn tcp_only_mode_uses_framed_stream_without_udp_probe() {
        let socket = FakeResolverSocket::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            resolver_addr(),
        );
        let factory = Arc::new(FakeResolverFactory {
            socket: socket.clone(),
        });
        let mut message = Vec::new();
        let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
        encode::encode_response(
            1,
            flags,
            encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &[],
            &mut message,
        )
        .unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(message.len() as u16).to_be_bytes());
        framed.extend_from_slice(&message);
        let writes = Arc::new(Mutex::new(Vec::new()));
        let tcp = Arc::new(FakeTcpUpstream {
            response: framed,
            writes: Arc::clone(&writes),
        });
        let config = DnsResolverConfig::builder()
            .resolver_ip(resolver_addr().ip().to_string())
            .port(resolver_addr().port())
            .query_timeout_ms(200)
            .max_attempts(1)
            .build();
        let client = DnsClientUpstream::new(factory, config)
            .with_tcp_upstream(tcp)
            .with_tcp_only();

        let response = client
            .query_with_metadata("example.com.", 1, 1)
            .await
            .unwrap();
        assert_eq!(response.answer.rcode, 0);
        assert!(socket.state.lock().unwrap().sent.is_empty());
        let request = writes.lock().unwrap();
        assert_eq!(parse_message(&request[2..]).unwrap().header.id, 1);
    }

    #[proxima::test]
    async fn successive_queries_carry_distinct_wire_ids() {
        // the id an upstream puts on the wire is what a reply must echo and
        // what an off-path spoofer has to guess; minting it per call instead
        // of per upstream stamped id 1 on every packet this client ever sent.
        let socket = FakeResolverSocket::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            resolver_addr(),
        );
        let factory = Arc::new(FakeResolverFactory {
            socket: socket.clone(),
        });
        let config = DnsResolverConfig::builder()
            .resolver_ip(resolver_addr().ip().to_string())
            .port(resolver_addr().port())
            .query_timeout_ms(200)
            .build();
        let client = DnsClientUpstream::new(factory, config);

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        for _ in 0..3 {
            let query_future = client.query("example.com.", 1, 1);
            futures::pin_mut!(query_future);
            assert!(query_future.as_mut().poll(&mut cx).is_pending());
            socket.queue_reply_to_last_query();
            loop {
                match query_future.as_mut().poll(&mut cx) {
                    Poll::Ready(result) => {
                        result.expect("the fake resolver answers every query");
                        break;
                    }
                    Poll::Pending => continue,
                }
            }
        }

        let sent_ids: Vec<u16> = socket
            .state
            .lock()
            .unwrap()
            .sent
            .iter()
            .map(|(bytes, _)| parse_message(bytes).unwrap().header.id)
            .collect();
        assert_eq!(sent_ids, vec![1, 2, 3], "each query gets its own id");
    }

    #[proxima::test]
    async fn a_reply_from_the_wrong_address_is_ignored_then_the_real_one_is_accepted() {
        // no sleeps: proves the mismatch guard filters and keeps polling by
        // queuing a stray reply from an off-target sender FIRST, then the
        // real resolver's reply — a client that accepted the first (wrong)
        // datagram would return its rcode/records instead of the correct
        // ones, so any pass here is a pass on the filter actually working.
        let wrong_sender = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 9)), 53);
        let socket = FakeResolverSocket::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            wrong_sender,
        );
        let factory = Arc::new(FakeResolverFactory {
            socket: socket.clone(),
        });
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
