//! Device-independent TCP stack driver. The frame device is three pipes: a
//! source `() -> Frame` (pending until a frame is ready), a sink
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
//! call: the sink and mint must settle at once ([`DriverError::Stalled`]); the source parks the task on its waker.

use crate::stack::{self, Action};
use crate::tcp_dial::{DialError, DialGuard, DialLease, SynStep, wake_after};
use crate::tcp_listener::{Endpoint, Inbound, OutSegment};
use crate::tcp_stack::{ConnId, Outbound, TcpStack};
use crossbeam_queue::SegQueue;
use core::fmt::Debug;
use core::future::Future;
use core::pin::pin;
use futures::FutureExt;
use futures::io::{AsyncRead, AsyncWrite};
use futures::task::{ArcWake, waker};
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
use std::task::{Context, Poll, Waker};
use std::time::Duration;
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

pub use crate::tcp_dial::DEFAULT_RETRANSMIT;

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

/// Tasks parked on the rx source. The source holds one waker slot, so the
/// driver registers this fan-out waker there and every parked task here.
struct Waiters {
    parked: SegQueue<Waker>,
}

impl ArcWake for Waiters {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        while let Some(task) = arc_self.parked.pop() {
            task.wake();
        }
    }
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
    arp_sent: HashMap<[u8; 4], StdInstant>,
    retransmit: Duration,
    waiters: Arc<Waiters>,
    fan_out: Waker,
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
        let waiters = Arc::new(Waiters { parked: SegQueue::new() });
        Self {
            fan_out: waker(Arc::clone(&waiters)),
            waiters,
            arp_sent: HashMap::new(),
            retransmit: DEFAULT_RETRANSMIT,
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

    /// Interval before an unanswered ARP request is sent again.
    #[must_use]
    pub fn with_retransmit(mut self, interval: Duration) -> Self {
        self.retransmit = interval;
        self
    }

    fn now(&self) -> TcpInstant {
        TcpInstant::from_micros(u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX))
    }
}

impl<Rx, Tx, Mint> Driver<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16>,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Mint::Out>,
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

    fn poll_rx(&self) -> Poll<Result<Mint::Out, Rx::Err>> {
        let mut fan_out = Context::from_waker(&self.fan_out);
        pin!(self.rx.call(())).poll(&mut fan_out)
    }

    /// One rx burst: answer arp/icmp, route TCP through the stack, transmit
    /// replies. When the source has nothing the calling task is parked and
    /// woken by the next frame; a burst that moved frames wakes every parked
    /// task, since the frames may belong to another connection.
    ///
    /// # Errors
    /// The first pipe or encode failure; frames already handled stay handled.
    pub fn pump(&mut self, cx: &Context<'_>) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        self.waiters.parked.push(cx.waker().clone());
        let now = self.now();
        let mut moved = false;
        for _ in 0..BURST {
            let Poll::Ready(received) = self.poll_rx() else {
                break;
            };
            let mut frame = received.map_err(DriverError::Rx)?;
            moved = true;
            match stack::handle_frame(frame.as_mut(), self.our_mac, self.our_ip) {
                Action::Transmit => self.transmit(frame)?,
                Action::Drop => self.ingest(frame, now)?,
            }
        }
        if moved {
            ArcWake::wake_by_ref(&self.waiters);
        }
        Ok(())
    }

    fn retransmit_arp(&mut self, target: [u8; 4], cx: &mut Context<'_>) -> Result<(), DriverFailure<Rx, Tx, Mint>> {
        let due = self.arp_sent.get(&target).is_none_or(|sent| sent.elapsed() >= self.retransmit);
        if due {
            self.send_arp_request(target)?;
            self.arp_sent.insert(target, StdInstant::now());
        }
        let waited = self.arp_sent.get(&target).map_or(Duration::ZERO, StdInstant::elapsed);
        wake_after(self.retransmit.saturating_sub(waited), cx);
        Ok(())
    }

    fn retransmit_syn(&self, dial: &mut DialGuard, cx: &mut Context<'_>) -> io::Result<()> {
        let source_port = dial.connection().map_or(self.our_port, |(port, _, _)| port);
        let remaining = match dial.next_syn(self.retransmit) {
            SynStep::Send(segments) => {
                self.send_segments(segments, source_port).map_err(into_io)?;
                self.retransmit
            }
            SynStep::Wait(remaining) => remaining,
            SynStep::Exhausted => {
                let attempts = dial.syn_transmissions();
                return Err(io::Error::new(io::ErrorKind::TimedOut, DialError::SynUnanswered { attempts }));
            }
        };
        wake_after(remaining, cx);
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
        if self.stack.owns_port(inbound.destination_port) {
            let outbound = self
                .stack
                .on_inbound_to(inbound.destination_port, &inbound.borrow(), now);
            return self.send_segments(&outbound, inbound.destination_port);
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
    Rx: Pipe<In = (), Out = Mint::Out> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    type Conn = FrameConnection<Rx, Tx, Mint>;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let mut driver = locked!(self.shared, cx);
        if let Err(error) = driver.pump(cx) {
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
    Rx: Pipe<In = (), Out = Mint::Out> + Send + 'static,
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
    Rx: Pipe<In = (), Out = Mint::Out> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut driver = locked!(this.shared, cx);
        if let Err(error) = driver.pump(cx) {
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
        Poll::Pending
    }
}

impl<Rx, Tx, Mint> AsyncWrite for FrameConnection<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Mint::Out> + Send + 'static,
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
        let sent = driver.send_segments(&outbound, this.id.0);
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
        let sent = driver.send_segments(&outbound, this.id.0);
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
        Self {
            shared: Arc::new(Shared {
                inner: AsyncMutex::new(driver),
                dial: Some(Arc::new(DialLease::new())),
            }),
            peer,
        }
    }
}

impl<Rx, Tx, Mint> StreamUpstream for FrameUpstream<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16> + Send + 'static,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Mint::Out> + Send + 'static,
    Tx: Pipe<In = Mint::Out, Out = ()> + Send + 'static,
{
    type Conn = FrameConnection<Rx, Tx, Mint>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let mut operation = None;
        // ConnectFuture is the trait's boxed future type; the box is the trait's, not ours.
        Box::pin(core::future::poll_fn(move |cx| {
            self.poll_dial(cx, &mut operation)
        }))
    }
}

impl<Rx, Tx, Mint> FrameUpstream<Rx, Tx, Mint>
where
    Mint: Pipe<In = u16>,
    Mint::Out: AsRef<[u8]> + AsMut<[u8]>,
    Rx: Pipe<In = (), Out = Mint::Out>,
    Tx: Pipe<In = Mint::Out, Out = ()>,
{
    fn poll_dial(
        &self,
        cx: &mut Context<'_>,
        operation: &mut Option<DialGuard>,
    ) -> Poll<io::Result<FrameConnection<Rx, Tx, Mint>>> {
        let mut driver = locked!(self.shared, cx);
        let Some(lease) = &self.shared.dial else {
            return Poll::Ready(Err(io::Error::other("missing upstream tuple lease")));
        };
        if operation.is_none() {
            *operation = Some(lease.claim());
        }
        let Some(dial) = operation.as_mut() else {
            return Poll::Ready(Err(io::Error::other("missing dial ownership")));
        };
        if let Err(error) = driver.pump(cx) {
            return Poll::Ready(Err(into_io(error)));
        }
        let peer_ip = self.peer.ip().octets();
        if let Some(id) = dial.connection() {
            if !dial.owns(&driver.stack) {
                return Poll::Ready(Err(io::Error::from(io::ErrorKind::ConnectionRefused)));
            }
            if dial.connected(&driver.stack) {
                while driver.stack.poll_connected().is_some() {}
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
            if let Err(error) = driver.retransmit_syn(dial, cx) {
                return Poll::Ready(Err(error));
            }
        } else if let Some(&mac) = driver.arp.get(&peer_ip) {
            let endpoint = Endpoint { mac, ip: peer_ip, port: self.peer.port() };
            let outbound = dial.start(&mut driver.stack, endpoint)?;
            let source_port = dial.connection().map_or(driver.our_port, |(port, _, _)| port);
            if let Err(error) = driver.send_segments(&outbound, source_port) {
                return Poll::Ready(Err(into_io(error)));
            }
        } else if let Err(error) = driver.retransmit_arp(peer_ip, cx) {
            return Poll::Ready(Err(into_io(error)));
        }
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use crossbeam_queue::ArrayQueue;
    use futures::task::AtomicWaker;
    use futures::executor::block_on;
    use futures::future::{join, join_all};
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};
    use proxima_protocols::http1_codec::h1::HttpVersion;
    use proxima_protocols::http1_codec::h1_body::BodyFraming;
    use proxima_protocols::http1_codec::h1_client::encode_request_head;
    use proxima_protocols::http1_codec::h1_response::write_response_head;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Waker;

    struct WireQueue {
        frames: ArrayQueue<Vec<u8>>,
        waker: AtomicWaker,
    }

    impl WireQueue {
        fn new(capacity: usize) -> Wire {
            Arc::new(Self { frames: ArrayQueue::new(capacity), waker: AtomicWaker::new() })
        }

        fn deliver(&self, frame: Vec<u8>) -> Result<(), Vec<u8>> {
            let outcome = self.frames.push(frame);
            self.waker.wake();
            outcome
        }
    }

    type Wire = Arc<WireQueue>;

    struct WireRx(Wire);
    struct WireTx(Wire);
    struct WireMint;

    #[derive(Debug, PartialEq, Eq)]
    enum WireFull {
        AtCapacity,
    }

    impl Pipe for WireRx {
        type In = ();
        type Out = Vec<u8>;
        type Err = Infallible;

        fn call(&self, _input: ()) -> impl Future<Output = Result<Vec<u8>, Infallible>> {
            core::future::poll_fn(|cx| {
                if let Some(frame) = self.0.frames.pop() {
                    return Poll::Ready(Ok(frame));
                }
                self.0.waker.register(cx.waker());
                self.0.frames.pop().map_or(Poll::Pending, |frame| Poll::Ready(Ok(frame)))
            })
        }
    }

    impl Pipe for WireTx {
        type In = Vec<u8>;
        type Out = ();
        type Err = WireFull;

        fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<(), WireFull>> {
            async move { self.0.deliver(frame).map_err(|_| WireFull::AtCapacity) }
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
        let to_server: Wire = WireQueue::new(capacity);
        let to_client: Wire = WireQueue::new(capacity);
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

    fn payload_for(index: usize) -> Vec<u8> {
        format!("connection-{index:04}-payload").into_bytes()
    }

    async fn echo_accepted(listener: &FrameListener<WireRx, WireTx, WireMint>, count: usize) -> Vec<Vec<u8>> {
        let mut accepted = Vec::with_capacity(count);
        for _ in 0..count {
            accepted.push(listener.accept().await.expect("server accepts every dialed connection"));
        }
        join_all(accepted.into_iter().map(|mut connection| async move {
            let mut received = vec![0u8; payload_for(0).len()];
            connection.read_exact(&mut received).await.expect("server reads the payload");
            connection.write_all(&received).await.expect("server echoes the payload");
            connection.close().await.expect("server closes");
            received
        }))
        .await
    }

    async fn dial_and_echo(upstream: &FrameUpstream<WireRx, WireTx, WireMint>, index: usize) -> Vec<u8> {
        let mut connection = upstream.connect().await.expect("every concurrent dial completes its handshake");
        connection.write_all(&payload_for(index)).await.expect("client writes its payload");
        let mut echoed = Vec::new();
        connection.read_to_end(&mut echoed).await.expect("client reads the echo to the server's FIN");
        connection.close().await.expect("client closes");
        echoed
    }

    #[test]
    fn sixty_four_concurrent_dials_to_one_peer_endpoint_are_independent_byte_streams() {
        const DIALS: usize = 64;
        let (server_pipes, client_pipes) = wired(4096);
        let listener = FrameListener::new(Driver::new(server_pipes, SERVER_MAC, SERVER, 0x1000), SERVER);
        let upstream = FrameUpstream::new(Driver::new(client_pipes, CLIENT_MAC, CLIENT, 0x2000), SERVER);

        let (served, echoed) = block_on(join(
            echo_accepted(&listener, DIALS),
            join_all((0..DIALS).map(|index| dial_and_echo(&upstream, index))),
        ));

        let expected: Vec<Vec<u8>> = (0..DIALS).map(payload_for).collect();
        assert_eq!(echoed, expected, "each connection receives its own payload back");
        let mut served_sorted = served;
        served_sorted.sort();
        assert_eq!(served_sorted, expected, "the listener saw each payload exactly once");
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

    struct WakeCount(AtomicUsize);

    impl ArcWake for WakeCount {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct SilentDial {
        to_server: Wire,
        to_client: Wire,
        upstream: FrameUpstream<WireRx, WireTx, WireMint>,
        wakes: Arc<WakeCount>,
    }

    fn silent_dial(retransmit: Duration) -> SilentDial {
        let to_server = WireQueue::new(64);
        let to_client = WireQueue::new(64);
        let pipes = (WireRx(to_client.clone()), WireTx(to_server.clone()), WireMint);
        let driver = Driver::new(pipes, CLIENT_MAC, CLIENT, 0x2000).with_retransmit(retransmit);
        SilentDial {
            to_server,
            to_client,
            upstream: FrameUpstream::new(driver, SERVER),
            wakes: Arc::new(WakeCount(AtomicUsize::new(0))),
        }
    }

    fn polls_pending(dial: &SilentDial, polls: usize) {
        let task_waker = waker(Arc::clone(&dial.wakes));
        let mut context = Context::from_waker(&task_waker);
        let mut connect = pin!(dial.upstream.connect());
        (0..polls).for_each(|_| assert!(connect.as_mut().poll(&mut context).is_pending(), "no frame ever arrives"));
    }

    #[test]
    fn a_dial_with_no_frames_parks_without_waking_itself_and_sends_one_arp_before_the_deadline() {
        let dial = silent_dial(Duration::from_secs(3600));

        polls_pending(&dial, 5);

        assert_eq!(dial.wakes.0.load(Ordering::SeqCst), 0, "nothing woke the task: no frame, no deadline");
        assert_eq!(dial.to_server.frames.len(), 1, "one arp request until the retransmit deadline passes");
    }

    #[test]
    fn a_dial_resends_its_arp_request_once_the_retransmit_deadline_has_passed() {
        let dial = silent_dial(Duration::ZERO);

        polls_pending(&dial, 3);

        assert_eq!(dial.to_server.frames.len(), 3, "each poll past the deadline sends one arp request");
        assert!(dial.wakes.0.load(Ordering::SeqCst) > 0, "a passed deadline wakes the task");
    }

    #[test]
    fn a_frame_arriving_wakes_the_parked_dial_exactly_once() {
        let dial = silent_dial(Duration::from_secs(3600));
        polls_pending(&dial, 1);
        assert_eq!(dial.wakes.0.load(Ordering::SeqCst), 0, "parked, not spinning");

        dial.to_client.deliver(vec![0; 60]).expect("rx wire has room");

        assert_eq!(dial.wakes.0.load(Ordering::SeqCst), 1, "the push woke the parked task");
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
    struct SynCounts {
        seen: AtomicUsize,
        dropped: AtomicUsize,
    }

    struct SynDroppingTx {
        wire: Wire,
        drop_first: usize,
        counts: Arc<SynCounts>,
    }

    impl Pipe for SynDroppingTx {
        type In = Vec<u8>;
        type Out = ();
        type Err = WireFull;

        fn call(&self, frame: Vec<u8>) -> impl Future<Output = Result<(), WireFull>> {
            async move {
                let is_syn = classify_tcp(&frame, SERVER.ip().octets())
                    .is_some_and(|segment| segment.flags.syn && !segment.flags.ack);
                if is_syn {
                    let ordinal = self.counts.seen.fetch_add(1, Ordering::SeqCst);
                    if ordinal < self.drop_first {
                        self.counts.dropped.fetch_add(1, Ordering::SeqCst);
                        return Ok(());
                    }
                }
                self.wire.deliver(frame).map_err(|_| WireFull::AtCapacity)
            }
        }
    }

    struct LossyDial {
        listener: FrameListener<WireRx, WireTx, WireMint>,
        upstream: FrameUpstream<WireRx, SynDroppingTx, WireMint>,
        counts: Arc<SynCounts>,
    }

    fn lossy_dial(drop_first: usize, retransmit: Duration) -> LossyDial {
        let (server_pipes, (client_rx, client_tx, client_mint)) = wired(64);
        let counts = Arc::new(SynCounts { seen: AtomicUsize::new(0), dropped: AtomicUsize::new(0) });
        let lossy = SynDroppingTx { wire: client_tx.0, drop_first, counts: Arc::clone(&counts) };
        let driver = Driver::new((client_rx, lossy, client_mint), CLIENT_MAC, CLIENT, 0x2000).with_retransmit(retransmit);
        LossyDial {
            listener: FrameListener::new(Driver::new(server_pipes, SERVER_MAC, SERVER, 0x1000), SERVER),
            upstream: FrameUpstream::new(driver, SERVER),
            counts,
        }
    }

    #[test]
    fn a_dial_whose_first_syn_is_lost_completes_on_the_retransmitted_syn() {
        let dial = lossy_dial(1, Duration::ZERO);

        let (accepted, connected) = block_on(join(dial.listener.accept(), dial.upstream.connect()));

        accepted.expect("server accepts the retransmitted handshake");
        connected.expect("dial completes after the retransmit");
        assert_eq!(dial.counts.dropped.load(Ordering::SeqCst), 1, "exactly the first syn was lost");
        assert!(dial.counts.seen.load(Ordering::SeqCst) >= 2, "a second syn crossed the wire");
    }

    #[test]
    fn a_lost_syn_is_not_resent_before_the_retransmit_deadline() {
        let dial = lossy_dial(1, Duration::from_secs(3600));
        let mut context = Context::from_waker(Waker::noop());
        let mut accept = pin!(dial.listener.accept());
        let mut connect = pin!(dial.upstream.connect());

        (0..16).for_each(|_| {
            assert!(accept.as_mut().poll(&mut context).is_pending());
            assert!(connect.as_mut().poll(&mut context).is_pending(), "the lost syn stalls the dial");
        });

        assert_eq!(dial.counts.seen.load(Ordering::SeqCst), 1, "one syn until the deadline passes");
    }

    #[test]
    fn a_dial_whose_syn_is_never_answered_fails_with_a_named_error_after_the_retry_bound() {
        let dial = lossy_dial(usize::MAX, Duration::ZERO);
        let mut context = Context::from_waker(Waker::noop());
        let mut accept = pin!(dial.listener.accept());
        let mut connect = pin!(dial.upstream.connect());

        let outcome = (0..256).find_map(|_| {
            assert!(accept.as_mut().poll(&mut context).is_pending());
            match connect.as_mut().poll(&mut context) {
                Poll::Ready(result) => Some(result),
                Poll::Pending => None,
            }
        });

        let error = outcome.expect("dial gives up inside the poll budget").err().expect("an unanswered dial fails");
        let expected_attempts = proxima_protocols::tcp::retx::MaxRetransmit::ABANDON_THRESHOLD + 1;
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(
            matches!(
                error.get_ref().and_then(|source| source.downcast_ref::<DialError>()),
                Some(DialError::SynUnanswered { attempts }) if *attempts == expected_attempts
            ),
            "error names the exhausted syn retransmits: {error}"
        );
        assert_eq!(dial.counts.dropped.load(Ordering::SeqCst), usize::from(expected_attempts));
    }
}
