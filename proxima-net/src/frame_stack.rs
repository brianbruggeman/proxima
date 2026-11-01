//! Device-independent TCP stack driver. The frame device is three pipes: a
//! source `() -> Option<Frame>` (`None` means nothing ready), a sink
//! `Frame -> ()`, and a mint `u16 -> Frame` that yields a blank frame of the
//! requested length. A dpdk port, a virtio-net queue pair, or an in-memory
//! channel all plug in by being those three pipes; the frame type is whatever
//! the mint yields and is only ever touched through `AsRef<[u8]>` /
//! `AsMut<[u8]>`, so a device that hands out owned buffers (a dpdk mbuf) is
//! parsed and rewritten in place with no copy at this seam.
//!
//! The driver answers ARP/ICMP inline, routes TCP into the sans-IO
//! [`TcpStack`], and refuses segments for closed ports with a RST so an active
//! open to a port nobody listens on fails fast. Each pipe is polled once per
//! call: a pipe that is not ready is [`DriverError::Stalled`], never a hang.

use crate::stack::{self, Action};
use crate::tcp_dial::{DialGuard, DialLease};
use crate::tcp_listener::{Endpoint, Inbound, OutSegment};
use crate::tcp_stack::{ConnId, Outbound, TcpStack};
use core::fmt::Debug;
use core::future::Future;
use core::pin::pin;
use futures::FutureExt;
use futures::io::{AsyncRead, AsyncWrite};
use proxima_primitives::pipe::Pipe;
use proxima_primitives::stream::{
    BindAddr, ConnectFuture, PeerInfo, StreamConnection, StreamListener, StreamUpstream,
};
use proxima_primitives::sync::{AsyncMutex, AsyncMutexGuard};
use proxima_protocols::inet::error::DecodeError;
use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
use proxima_protocols::inet::tcp::{self, TcpFlags, TcpHeader};
use proxima_protocols::tcp::time::Instant as TcpInstant;
use std::collections::HashMap;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant as StdInstant;
use thiserror::Error;

const BURST: usize = 32;
const ETH: usize = 14;
const IP: usize = 20;
const TCP: usize = 20;
const HEADERS: u16 = 54;
const TCP_LEN: u16 = 20;
const ARP_FRAME: u16 = 42;
const TTL: u8 = 64;

/// Why the driver could not move a frame or build a segment. The pipe errors
/// are carried whole so the log names which device failed and how.
#[derive(Debug, Error)]
pub enum DriverError<RxErr: Debug, TxErr: Debug, MintErr: Debug> {
    #[error("frame source failed: {0:?}")]
    Rx(RxErr),

    #[error("frame sink failed: {0:?}")]
    Tx(TxErr),

    #[error("frame mint failed for {length} bytes: {source:?}")]
    Mint { length: u16, source: MintErr },

    #[error("{pipe} pipe was not ready when polled")]
    Stalled { pipe: &'static str },

    #[error("segment payload of {length} bytes does not fit one frame")]
    PayloadTooLong { length: usize },

    #[error("mint returned {actual} bytes for a {expected} byte frame")]
    FrameSize { expected: usize, actual: usize },

    #[error("header encode failed: {0:?}")]
    Encode(DecodeError),

    #[error("arp request for {target:?} did not fit its frame")]
    ArpEncode { target: [u8; 4] },
}

pub type DriverFailure<Rx, Tx, Mint> =
    DriverError<<Rx as Pipe>::Err, <Tx as Pipe>::Err, <Mint as Pipe>::Err>;

fn into_io<RxErr: Debug, TxErr: Debug, MintErr: Debug>(
    error: DriverError<RxErr, TxErr, MintErr>,
) -> io::Error {
    io::Error::other(error.to_string())
}

struct OwnedInbound {
    mac: [u8; 6],
    ip: [u8; 4],
    port: u16,
    destination_port: u16,
    flags: TcpFlags,
    seq: u32,
    ack: u32,
    window: u16,
    payload: Vec<u8>,
}

impl OwnedInbound {
    fn borrow(&self) -> Inbound<'_> {
        Inbound {
            source_mac: self.mac,
            source_ip: self.ip,
            source_port: self.port,
            flags: self.flags,
            seq: self.seq,
            ack: self.ack,
            window: self.window,
            payload: &self.payload,
        }
    }

    fn refusal(&self) -> Option<Outbound> {
        if self.flags.rst {
            return None;
        }
        let consumed = u32::try_from(self.payload.len())
            .unwrap_or(u32::MAX)
            .wrapping_add(u32::from(self.flags.syn))
            .wrapping_add(u32::from(self.flags.fin));
        let (seq, ack, flags) = if self.flags.ack {
            (self.ack, 0, TcpFlags { rst: true, ..TcpFlags::default() })
        } else {
            (
                0,
                self.seq.wrapping_add(consumed),
                TcpFlags { rst: true, ack: true, ..TcpFlags::default() },
            )
        };
        let peer = Endpoint {
            mac: self.mac,
            ip: self.ip,
            port: self.port,
        };
        Some((peer, OutSegment { flags, seq, ack, window: 0, payload: Vec::new() }))
    }
}

fn classify_tcp(frame: &[u8], our_ip: [u8; 4]) -> Option<OwnedInbound> {
    let eth = EthernetFrame::parse(frame).ok()?;
    if eth.ether_type() != EtherType::Ipv4 {
        return None;
    }
    let ip = Ipv4Header::parse(eth.payload()).ok()?;
    if ip.protocol() != Ipv4Protocol::Tcp || ip.destination() != our_ip {
        return None;
    }
    let segment = TcpHeader::parse(ip.payload()).ok()?;
    Some(OwnedInbound {
        mac: eth.source(),
        ip: ip.source(),
        port: segment.source_port(),
        destination_port: segment.destination_port(),
        flags: segment.flags(),
        seq: segment.sequence(),
        ack: segment.acknowledgement(),
        window: segment.window(),
        payload: segment.payload().to_vec(),
    })
}

/// One TCP endpoint on one frame device: the sans-IO stack plus the ARP table
/// and the three frame pipes it reads, writes and mints through.
pub struct Driver<Rx, Tx, Mint> {
    rx: Rx,
    tx: Tx,
    mint: Mint,
    stack: TcpStack,
    start: StdInstant,
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    our_port: u16,
    arp: HashMap<[u8; 4], [u8; 6]>,
}

impl<Rx, Tx, Mint> Driver<Rx, Tx, Mint> {
    #[must_use]
    pub fn new(
        (rx, tx, mint): (Rx, Tx, Mint),
        our_mac: [u8; 6],
        local: SocketAddrV4,
        initial_sequence: u32,
    ) -> Self {
        let our_ip = local.ip().octets();
        let our_port = local.port();
        Self {
            rx,
            tx,
            mint,
            stack: TcpStack::new(our_ip, our_port, initial_sequence),
            start: StdInstant::now(),
            our_mac,
            our_ip,
            our_port,
            arp: HashMap::new(),
        }
    }

    fn now(&self) -> TcpInstant {
        TcpInstant::from_micros(u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX))
    }
}

impl<Rx, Tx, Mint> Driver<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16>,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>>,
    Tx: Pipe<In = Mint::Out, Out = ()>,
{
    fn settle<Output>(
        future: impl Future<Output = Output>,
        pipe: &'static str,
    ) -> Result<Output, DriverFailure<Rx, Tx, Mint>> {
        future.now_or_never().ok_or(DriverError::Stalled { pipe })
    }

    fn transmit(&self, frame: Mint::Out) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        Self::settle(self.tx.call(frame), "tx")?.map_err(DriverError::Tx)
    }

    fn blank_frame(&self, length: u16) -> Result<Mint::Out, DriverFailure<Rx, Tx, Mint>> {
        let frame = Self::settle(self.mint.call(length), "mint")?
            .map_err(|source| DriverError::Mint { length, source })?;
        let actual = frame.as_ref().len();
        if actual != usize::from(length) {
            return Err(DriverError::FrameSize { expected: usize::from(length), actual });
        }
        Ok(frame)
    }

    /// One rx burst: answer arp/icmp, route TCP through the stack, transmit replies.
    ///
    /// # Errors
    /// The first pipe or encode failure; frames already handled stay handled.
    pub fn pump(&mut self) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        let now = self.now();
        for _ in 0..BURST {
            let received = Self::settle(self.rx.call(()), "rx")?.map_err(DriverError::Rx)?;
            let Some(mut frame) = received else {
                return Ok(());
            };
            match stack::handle_frame(frame.as_mut(), self.our_mac, self.our_ip) {
                Action::Transmit => self.transmit(frame)?,
                Action::Drop => self.ingest(frame, now)?,
            }
        }
        Ok(())
    }

    fn ingest(&mut self, frame: Mint::Out, now: TcpInstant) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        if let Some((ip, mac)) = stack::parse_arp_reply(frame.as_ref()) {
            self.arp.insert(ip, mac);
            return Ok(());
        }
        let Some(inbound) = classify_tcp(frame.as_ref(), self.our_ip) else {
            return Ok(());
        };
        drop(frame);
        if inbound.destination_port == self.our_port {
            let outbound = self.stack.on_inbound(&inbound.borrow(), now);
            return self.send_segments(&outbound, self.our_port);
        }
        let refusals: Vec<Outbound> = inbound.refusal().into_iter().collect();
        self.send_segments(&refusals, inbound.destination_port)
    }

    fn send_arp_request(&self, target: [u8; 4]) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        let mut frame = self.blank_frame(ARP_FRAME)?;
        let written = stack::build_arp_request(frame.as_mut(), self.our_mac, self.our_ip, target);
        if written == 0 {
            return Err(DriverError::ArpEncode { target });
        }
        self.transmit(frame)
    }

    fn send_segments(
        &self,
        outbound: &[Outbound],
        source_port: u16,
    ) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        outbound
            .iter()
            .try_for_each(|(peer, segment)| self.send_segment(*peer, segment, source_port))
    }

    fn send_segment(
        &self,
        peer: Endpoint,
        segment: &OutSegment,
        source_port: u16,
    ) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        let length = segment.payload.len();
        let payload_len = u16::try_from(length).map_err(|_| DriverError::PayloadTooLong { length })?;
        let total = HEADERS
            .checked_add(payload_len)
            .ok_or(DriverError::PayloadTooLong { length })?;
        let mut frame = self.blank_frame(total)?;
        let buf = frame.as_mut();
        ethernet::write_header(&mut buf[..ETH], peer.mac, self.our_mac, EtherType::Ipv4)
            .map_err(DriverError::Encode)?;
        ipv4::write_header(
            &mut buf[ETH..ETH + IP],
            self.our_ip,
            peer.ip,
            Ipv4Protocol::Tcp,
            TTL,
            payload_len + TCP_LEN,
            0,
        )
        .map_err(DriverError::Encode)?;
        buf[ETH + IP + TCP..].copy_from_slice(&segment.payload);
        tcp::write_header(
            &mut buf[ETH + IP..],
            self.our_ip,
            peer.ip,
            source_port,
            peer.port,
            segment.seq,
            segment.ack,
            segment.flags,
            segment.window,
            &segment.payload,
        )
        .map_err(DriverError::Encode)?;
        self.transmit(frame)
    }
}

struct Shared<Rx, Tx, Mint> {
    inner: AsyncMutex<Driver<Rx, Tx, Mint>>,
    dial: Option<Arc<DialLease>>,
}

impl<Rx, Tx, Mint> Shared<Rx, Tx, Mint> {
    // manually polls the async gate's `lock()`: no thread parks, and on the
    // contended path the caller's waker is queued and re-woken on release.
    fn poll_lock(&self, cx: &mut Context<'_>) -> Poll<AsyncMutexGuard<'_, Driver<Rx, Tx, Mint>>> {
        let lock_future = pin!(self.inner.lock());
        match lock_future.poll(cx) {
            Poll::Ready(mut driver) => {
                if let Some(dial) = &self.dial {
                    dial.reclaim(&mut driver.stack);
                }
                Poll::Ready(driver)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

macro_rules! locked {
    ($shared:expr, $cx:expr) => {
        match $shared.poll_lock($cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        }
    };
}

/// A TCP listener bound to the driver's address.
pub struct FrameListener<Rx, Tx, Mint> {
    shared: Arc<Shared<Rx, Tx, Mint>>,
    local_addr: SocketAddr,
}

impl<Rx, Tx, Mint> FrameListener<Rx, Tx, Mint> {
    #[must_use]
    pub fn new(driver: Driver<Rx, Tx, Mint>, bind: SocketAddrV4) -> Self {
        Self {
            shared: Arc::new(Shared { inner: AsyncMutex::new(driver), dial: None }),
            local_addr: SocketAddr::V4(bind),
        }
    }
}

impl<Rx, Tx, Mint> StreamListener for FrameListener<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    type Conn = FrameConnection<Rx, Tx, Mint>;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let mut driver = locked!(self.shared, cx);
        if let Err(error) = driver.pump() {
            return Poll::Ready(Err(into_io(error)));
        }
        if let Some(id) = driver.stack.poll_accept() {
            let peer = driver.stack.peer(id).map_or(self.local_addr, |endpoint| {
                SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(endpoint.ip), endpoint.port))
            });
            return Poll::Ready(Ok(FrameConnection {
                shared: self.shared.clone(),
                id,
                peer,
                dial: None,
            }));
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }

    fn local_addr(&self) -> Option<BindAddr> {
        Some(BindAddr::Tcp(self.local_addr))
    }
}

/// A TCP connection: an `AsyncRead`/`AsyncWrite` byte stream over the shared stack.
pub struct FrameConnection<Rx, Tx, Mint> {
    shared: Arc<Shared<Rx, Tx, Mint>>,
    id: ConnId,
    peer: SocketAddr,
    dial: Option<DialGuard>,
}

impl<Rx, Tx, Mint> StreamConnection for FrameConnection<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Tcp(self.peer))
    }
}

impl<Rx, Tx, Mint> AsyncRead for FrameConnection<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut driver = locked!(this.shared, cx);
        if let Err(error) = driver.pump() {
            return Poll::Ready(Err(into_io(error)));
        }
        let read = match &this.dial {
            Some(dial) => dial.read(&mut driver.stack, buf)?,
            None => driver.stack.read(this.id, buf),
        };
        if read > 0 {
            return Poll::Ready(Ok(read));
        }
        if driver.stack.read_closed(this.id) {
            return Poll::Ready(Ok(0));
        }
        drop(driver);
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl<Rx, Tx, Mint> AsyncWrite for FrameConnection<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut driver = locked!(this.shared, cx);
        let now = driver.now();
        let outbound = match &this.dial {
            Some(dial) => dial.write(&mut driver.stack, buf, now)?,
            None => driver.stack.write(this.id, buf, now),
        };
        let sent = driver.send_segments(&outbound, driver.our_port);
        Poll::Ready(sent.map(|()| buf.len()).map_err(into_io))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let Some(dial) = &this.dial else {
            return Poll::Ready(Ok(()));
        };
        let driver = locked!(this.shared, cx);
        Poll::Ready(dial.flush(&driver.stack))
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let mut driver = locked!(this.shared, cx);
        let outbound = match &this.dial {
            Some(dial) => dial.close(&mut driver.stack)?,
            None => driver.stack.close(this.id),
        };
        let sent = driver.send_segments(&outbound, driver.our_port);
        Poll::Ready(sent.map_err(into_io))
    }
}

/// An active-open TCP client: ARP-resolves the peer, drives the handshake, and
/// yields a connected [`FrameConnection`].
pub struct FrameUpstream<Rx, Tx, Mint> {
    shared: Arc<Shared<Rx, Tx, Mint>>,
    peer: SocketAddrV4,
}

impl<Rx, Tx, Mint> FrameUpstream<Rx, Tx, Mint> {
    #[must_use]
    pub fn new(driver: Driver<Rx, Tx, Mint>, peer: SocketAddrV4) -> Self {
        let tuple = (peer.ip().octets(), peer.port());
        Self {
            shared: Arc::new(Shared {
                inner: AsyncMutex::new(driver),
                dial: Some(Arc::new(DialLease::new(tuple))),
            }),
            peer,
        }
    }
}

impl<Rx, Tx, Mint> StreamUpstream for FrameUpstream<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    type Conn = FrameConnection<Rx, Tx, Mint>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let mut operation = None;
        // ConnectFuture is the trait's boxed future type; the box is the trait's, not ours.
        Box::pin(core::future::poll_fn(move |context| {
            self.poll_dial(context, &mut operation)
        }))
    }
}

impl<Rx, Tx, Mint> FrameUpstream<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16>,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Option<Mint::Out>>,
    Tx: Pipe<In = Mint::Out, Out = ()>,
{
    fn poll_dial(
        &self,
        context: &mut Context<'_>,
        operation: &mut Option<DialGuard>,
    ) -> Poll<io::Result<FrameConnection<Rx, Tx, Mint>>> {
        let mut driver = locked!(self.shared, context);
        let Some(lease) = &self.shared.dial else {
            return Poll::Ready(Err(io::Error::other("missing upstream tuple lease")));
        };
        if operation.is_none() {
            *operation = Some(lease.claim(&mut driver.stack)?);
        }
        let Some(dial) = operation.as_mut() else {
            return Poll::Ready(Err(io::Error::other("missing dial ownership")));
        };
        if let Err(error) = driver.pump() {
            return Poll::Ready(Err(into_io(error)));
        }
        let peer_ip = self.peer.ip().octets();
        if let Some(id) = dial.connection() {
            if !dial.owns(&driver.stack) {
                return Poll::Ready(Err(io::Error::from(io::ErrorKind::ConnectionRefused)));
            }
            while let Some(connected) = driver.stack.poll_connected() {
                if connected == id && dial.connected(&driver.stack) {
                    let Some(dial) = operation.take() else {
                        return Poll::Ready(Err(io::Error::other("missing dial ownership")));
                    };
                    let dial = dial.complete(&driver.stack)?;
                    return Poll::Ready(Ok(FrameConnection {
                        shared: Arc::clone(&self.shared),
                        dial: Some(dial),
                        id,
                        peer: SocketAddr::V4(self.peer),
                    }));
                }
            }
        } else if let Some(&mac) = driver.arp.get(&peer_ip) {
            let endpoint = Endpoint { mac, ip: peer_ip, port: self.peer.port() };
            let outbound = dial.start(&mut driver.stack, endpoint)?;
            if let Err(error) = driver.send_segments(&outbound, driver.our_port) {
                return Poll::Ready(Err(into_io(error)));
            }
        } else if let Err(error) = driver.send_arp_request(peer_ip) {
            return Poll::Ready(Err(into_io(error)));
        }
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use crossbeam_queue::ArrayQueue;
    use futures::executor::block_on;
    use futures::future::join;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};
    use proxima_protocols::http1_codec::h1::HttpVersion;
    use proxima_protocols::http1_codec::h1_body::BodyFraming;
    use proxima_protocols::http1_codec::h1_client::encode_request_head;
    use proxima_protocols::http1_codec::h1_response::write_response_head;
    use std::task::Waker;

    type Wire = Arc<ArrayQueue<Vec<u8>>>;

    struct WireRx(Wire);
    struct WireTx(Wire);
    struct WireMint;

    #[derive(Debug, PartialEq, Eq)]
    enum WireFull {
        AtCapacity,
    }

    impl Pipe for WireRx {
        type In = ();
        type Out = Option<Vec<u8>>;
        type Err = Infallible;

        fn call(&self, _input: ()) -> impl Future<Output = Result<Option<Vec<u8>>, Infallible>> {
            async move { Ok(self.0.pop()) }
        }
    }

    impl Pipe for WireTx {
        type In = Vec<u8>;
        type Out = ();
        type Err = WireFull;

        fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<(), WireFull>> {
            async move { self.0.push(frame).map_err(|_| WireFull::AtCapacity) }
        }
    }

    impl Pipe for WireMint {
        type In = u16;
        type Out = Vec<u8>;
        type Err = Infallible;

        fn call(&self, length: u16) -> impl Future<Output = Result<Vec<u8>, Infallible>> {
            async move { Ok(vec![0; usize::from(length)]) }
        }
    }

    type WirePipes = (WireRx, WireTx, WireMint);

    const SERVER_MAC: [u8; 6] = [0x02, 0, 0, 0, 0, 0x01];
    const CLIENT_MAC: [u8; 6] = [0x02, 0, 0, 0, 0, 0x02];
    const SERVER: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 8080);
    const CLIENT: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 2), 40000);
    const BODY: &[u8] = b"ok\n";

    fn request_bytes() -> Vec<u8> {
        let mut request = Vec::new();
        let headers = [("Host", "guest.local"), ("User-Agent", "proxima"), ("Accept", "*/*")];
        encode_request_head("GET", "/healthz", &headers, &mut request);
        request
    }

    fn response_bytes() -> Vec<u8> {
        let mut response = Vec::new();
        let headers = [
            ("Content-Type".to_owned(), "text/plain".to_owned()),
            ("Connection".to_owned(), "close".to_owned()),
        ];
        let framing = BodyFraming::ContentLength(BODY.len() as u64);
        write_response_head(&mut response, HttpVersion::Http11, 200, "OK", &headers, framing);
        response.extend_from_slice(BODY);
        response
    }

    fn wired(capacity: usize) -> (WirePipes, WirePipes) {
        let to_server: Wire = Arc::new(ArrayQueue::new(capacity));
        let to_client: Wire = Arc::new(ArrayQueue::new(capacity));
        (
            (WireRx(to_server.clone()), WireTx(to_client.clone()), WireMint),
            (WireRx(to_client), WireTx(to_server), WireMint),
        )
    }

    fn endpoints(peer: SocketAddrV4) -> (FrameListener<WireRx, WireTx, WireMint>, FrameUpstream<WireRx, WireTx, WireMint>) {
        let (server_pipes, client_pipes) = wired(64);
        let listener = FrameListener::new(Driver::new(server_pipes, SERVER_MAC, SERVER, 0x1000), SERVER);
        let upstream = FrameUpstream::new(Driver::new(client_pipes, CLIENT_MAC, CLIENT, 0x2000), peer);
        (listener, upstream)
    }

    async fn serve(listener: &FrameListener<WireRx, WireTx, WireMint>, response: &[u8]) -> Vec<u8> {
        let mut connection = listener.accept().await.expect("server accepts the dialed connection");
        let mut request = Vec::new();
        let mut chunk = [0u8; 256];
        while !request.ends_with(b"\r\n\r\n") {
            let read = connection.read(&mut chunk).await.expect("server reads the request");
            assert!(read > 0, "client closed before finishing the request head");
            request.extend_from_slice(&chunk[..read]);
        }
        connection.write_all(response).await.expect("server writes the response");
        connection.close().await.expect("server closes");
        request
    }

    async fn call(upstream: &FrameUpstream<WireRx, WireTx, WireMint>, request: &[u8]) -> Vec<u8> {
        let mut connection = upstream.connect().await.expect("client completes the handshake");
        connection.write_all(request).await.expect("client writes the request");
        let mut response = Vec::new();
        connection.read_to_end(&mut response).await.expect("client reads to the server's FIN");
        connection.close().await.expect("client closes");
        response
    }

    #[test]
    fn http1_request_and_response_cross_two_drivers_and_both_sides_close() {
        let (listener, upstream) = endpoints(SERVER);

        let (request, response) = (request_bytes(), response_bytes());

        let (request_seen, response_seen) =
            block_on(join(serve(&listener, &response), call(&upstream, &request)));

        assert_eq!(request_seen, request);
        assert_eq!(response_seen, response);
    }

    #[test]
    fn dialing_a_closed_port_is_refused_within_a_bounded_number_of_polls() {
        const POLL_BUDGET: usize = 64;
        let closed_port = SocketAddrV4::new(*SERVER.ip(), 9999);
        let (listener, upstream) = endpoints(closed_port);
        let mut context = Context::from_waker(Waker::noop());
        let mut accept = pin!(listener.accept());
        let mut connect = pin!(upstream.connect());

        let outcome = (0..POLL_BUDGET).find_map(|_| {
            assert!(accept.as_mut().poll(&mut context).is_pending(), "nothing listens on the dialed port");
            match connect.as_mut().poll(&mut context) {
                Poll::Ready(result) => Some(result),
                Poll::Pending => None,
            }
        });

        let error = outcome
            .expect("dial resolves inside the poll budget")
            .err()
            .expect("dial to a closed port must fail");
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);
    }

    #[test]
    fn a_full_sink_surfaces_as_a_named_tx_error_not_a_dropped_segment() {
        let (_server_pipes, client_pipes) = wired(1);
        let driver = Driver::new(client_pipes, CLIENT_MAC, CLIENT, 0x2000);
        let endpoint = Endpoint { mac: SERVER_MAC, ip: SERVER.ip().octets(), port: SERVER.port() };
        let segment = OutSegment { flags: TcpFlags::default(), seq: 1, ack: 1, window: 1024, payload: b"x".to_vec() };

        driver.send_segment(endpoint, &segment, CLIENT.port()).expect("first frame fits the wire");
        let overflow = driver.send_segment(endpoint, &segment, CLIENT.port());

        assert!(matches!(overflow, Err(DriverError::Tx(WireFull::AtCapacity))), "second frame meets a full sink");
    }
}
