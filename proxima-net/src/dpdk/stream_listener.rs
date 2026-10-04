//! Prime-native TCP `StreamListener` over dpdk (`proxima_primitives::stream`). No tokio: the
//! poll-mode driver has no fd, so `poll_accept` / `poll_read` busy-poll the RX
//! ring (re-arming the waker on `Pending`) and pump the sans-IO [`TcpStack`].
//! Accepted connections are `AsyncRead`/`AsyncWrite` handles sharing the stack
//! under an [`AsyncMutex`]; each handle opportunistically pumps on poll (single
//! prime core, so the lock is uncontended). Lock acquisition is manually
//! polled (never a thread-parking mutex) per the workspace lock-discipline
//! rule: everything here is reached from `poll_*` futures, so a blocking
//! mutex is never the right primitive, contended or not. ARP/ICMP are
//! answered inline.
//!
//! EAL is process-global → one dpdk listener per process. The dpdk resources are
//! core-pinned; the `unsafe` Send/Sync assert single-core use.

use super::port::{self, Port};
use super::{DpdkError, Eal, Mempool, RawMbuf};
use crate::stack::{self, Action};
use crate::tcp_dial::{DialGuard, DialLease};
use crate::tcp_listener::{Endpoint, Inbound, OutSegment};
use crate::tcp_stack::{ConnId, TcpStack};
use core::future::Future;
use core::pin::pin;
use futures::io::{AsyncRead, AsyncWrite};
use proxima_primitives::stream::{
    BindAddr, ConnectFuture, PeerInfo, StreamConnection, StreamListener, StreamUpstream,
};
use proxima_primitives::sync::{AsyncMutex, AsyncMutexGuard};
use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
use proxima_protocols::inet::tcp::{self, TcpHeader};
use proxima_protocols::tcp::time::Instant as TcpInstant;
use std::collections::HashMap;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant as StdInstant;

const BURST: usize = 32;
const ETH: usize = 14;
const IP: usize = 20;
const TCP: usize = 20;

struct State {
    _eal: Eal,
    pool: Mempool,
    port: Port,
    stack: TcpStack,
    start: StdInstant,
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    our_port: u16,
    arp: HashMap<[u8; 4], [u8; 6]>,
}

struct Shared {
    inner: AsyncMutex<State>,
    dial: Option<Arc<DialLease>>,
}

// SAFETY: the dpdk resources are reached only under `inner`'s lock and the
// listener is driven from a single prime core (per-core rx/tx queues). Raw mbuf
// pointers never escape a pump.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    // manually polls the async gate mutex's `lock()` future: no thread ever
    // blocks, and on the (never, single-core) contended path the caller's own
    // waker is queued and re-woken on release instead of parking.
    fn poll_lock(&self, cx: &mut Context<'_>) -> Poll<AsyncMutexGuard<'_, State>> {
        let lock_future = pin!(self.inner.lock());
        match lock_future.poll(cx) {
            Poll::Ready(mut state) => {
                if let Some(dial) = &self.dial {
                    dial.reclaim(&mut state.stack);
                }
                Poll::Ready(state)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A dpdk-backed TCP listener bound to one address.
pub struct DpdkStreamListener {
    shared: Arc<Shared>,
    local_addr: SocketAddr,
}

impl DpdkStreamListener {
    /// Bring up a net_tap vdev and accept TCP connections on `bind`.
    ///
    /// # Errors
    /// Propagates EAL / pool / port bring-up failures as [`DpdkError`].
    pub fn bind(bind: SocketAddrV4, iface: &str, pmd_dir: &str) -> Result<Self, DpdkError> {
        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal = Eal::init(&[
            "proxima-net-dpdk",
            "-l",
            "0",
            "--no-pci",
            "-d",
            pmd_dir,
            &vdev,
        ])?;
        let pool = Mempool::create("pnd_tcp_pool", 8192, -1)?;
        let port = Port::init(0, &pool)?;
        let our_mac = port.mac()?;
        let our_ip = bind.ip().octets();
        let our_port = bind.port();
        let stack = TcpStack::new(our_ip, our_port, 0x1000);
        let state = State {
            _eal: eal,
            pool,
            port,
            stack,
            start: StdInstant::now(),
            our_mac,
            our_ip,
            our_port,
            arp: HashMap::new(),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                inner: AsyncMutex::new(state),
                dial: None,
            }),
            local_addr: SocketAddr::V4(bind),
        })
    }
}

impl State {
    fn now(&self) -> TcpInstant {
        TcpInstant::from_micros(u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX))
    }

    // one rx burst: answer arp/icmp, route TCP through the stack, transmit replies.
    fn pump(&mut self) {
        let now = self.now();
        let mut bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let received = usize::from(self.port.rx_burst(&mut bufs));
        for &mbuf in &bufs[..received] {
            let frame = unsafe { port::frame_bytes_mut(mbuf) };
            match stack::handle_frame(frame, self.our_mac, self.our_ip) {
                Action::Transmit => {
                    let mut one = [mbuf];
                    if self.port.tx_burst(&mut one) == 0 {
                        unsafe { port::free(mbuf) };
                    }
                }
                Action::Drop => {
                    if let Some((ip, mac)) = stack::parse_arp_reply(frame) {
                        // active-open clients learn the peer MAC here.
                        self.arp.insert(ip, mac);
                        unsafe { port::free(mbuf) };
                    } else if let Some(inbound) = classify_tcp(frame, self.our_ip, self.our_port) {
                        let outbound = self.stack.on_inbound(&inbound.borrow(), now);
                        unsafe { port::free(mbuf) };
                        for (peer, segment) in outbound {
                            self.tx_segment(peer, &segment);
                        }
                    } else {
                        unsafe { port::free(mbuf) };
                    }
                }
            }
        }
    }

    fn send_arp_request(&mut self, target_ip: [u8; 4]) {
        let mbuf = self.pool.alloc();
        if mbuf.is_null() {
            return;
        }
        let Some(buf) = (unsafe { port::frame_append(mbuf, 42) }) else {
            unsafe { port::free(mbuf) };
            return;
        };
        let written = stack::build_arp_request(buf, self.our_mac, self.our_ip, target_ip);
        if written == 0 {
            unsafe { port::free(mbuf) };
            return;
        }
        let mut one = [mbuf];
        if self.port.tx_burst(&mut one) == 0 {
            unsafe { port::free(mbuf) };
        }
    }

    fn tx_segment(&mut self, peer: Endpoint, segment: &OutSegment) {
        let payload_len = u16::try_from(segment.payload.len()).unwrap_or(0);
        let total = u16::try_from(ETH + IP + TCP).unwrap_or(0) + payload_len;
        let mbuf = self.pool.alloc();
        if mbuf.is_null() {
            return;
        }
        let Some(buf) = (unsafe { port::frame_append(mbuf, total) }) else {
            unsafe { port::free(mbuf) };
            return;
        };
        let _ = ethernet::write_header(&mut buf[..ETH], peer.mac, self.our_mac, EtherType::Ipv4);
        let l4_len = u16::try_from(TCP).unwrap_or(0) + payload_len;
        let _ = ipv4::write_header(
            &mut buf[ETH..ETH + IP],
            self.our_ip,
            peer.ip,
            Ipv4Protocol::Tcp,
            64,
            l4_len,
            0,
        );
        buf[ETH + IP + TCP..].copy_from_slice(&segment.payload);
        let _ = tcp::write_header(
            &mut buf[ETH + IP..],
            self.our_ip,
            peer.ip,
            self.our_port,
            peer.port,
            segment.seq,
            segment.ack,
            segment.flags,
            segment.window,
            &segment.payload,
        );
        let mut one = [mbuf];
        if self.port.tx_burst(&mut one) == 0 {
            unsafe { port::free(mbuf) };
        }
    }
}

// owned inbound so the source mbuf can be freed before replies are built.
struct OwnedInbound {
    mac: [u8; 6],
    ip: [u8; 4],
    port: u16,
    flags: tcp::TcpFlags,
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
}

fn classify_tcp(frame: &[u8], our_ip: [u8; 4], our_port: u16) -> Option<OwnedInbound> {
    let eth = EthernetFrame::parse(frame).ok()?;
    if eth.ether_type() != EtherType::Ipv4 {
        return None;
    }
    let ip = Ipv4Header::parse(eth.payload()).ok()?;
    if ip.protocol() != Ipv4Protocol::Tcp || ip.destination() != our_ip {
        return None;
    }
    let segment = TcpHeader::parse(ip.payload()).ok()?;
    if segment.destination_port() != our_port {
        return None;
    }
    Some(OwnedInbound {
        mac: eth.source(),
        ip: ip.source(),
        port: segment.source_port(),
        flags: segment.flags(),
        seq: segment.sequence(),
        ack: segment.acknowledgement(),
        window: segment.window(),
        payload: segment.payload().to_vec(),
    })
}

impl StreamListener for DpdkStreamListener {
    type Conn = DpdkStreamConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let mut state = match self.shared.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        state.pump();
        if let Some(id) = state.stack.poll_accept() {
            let peer = state.stack.peer(id).map_or(self.local_addr, |endpoint| {
                SocketAddr::V4(SocketAddrV4::new(
                    Ipv4Addr::from(endpoint.ip),
                    endpoint.port,
                ))
            });
            return Poll::Ready(Ok(DpdkStreamConnection {
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

/// An accepted dpdk TCP connection: an `AsyncRead`/`AsyncWrite` byte stream over
/// the shared [`TcpStack`].
pub struct DpdkStreamConnection {
    shared: Arc<Shared>,
    id: ConnId,
    peer: SocketAddr,
    dial: Option<DialGuard>,
}

impl StreamConnection for DpdkStreamConnection {
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Tcp(self.peer))
    }
}

impl AsyncRead for DpdkStreamConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut state = match this.shared.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        state.pump();
        let read = match &this.dial {
            Some(dial) => dial.read(&mut state.stack, buf)?,
            None => state.stack.read(this.id, buf),
        };
        if read > 0 {
            return Poll::Ready(Ok(read));
        }
        if state.stack.read_closed(this.id) {
            return Poll::Ready(Ok(0));
        }
        drop(state);
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl AsyncWrite for DpdkStreamConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut state = match this.shared.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        let now = state.now();
        let outbound = match &this.dial {
            Some(dial) => dial.write(&mut state.stack, buf, now)?,
            None => state.stack.write(this.id, buf, now),
        };
        for (peer, segment) in outbound {
            state.tx_segment(peer, &segment);
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let Some(dial) = &this.dial else {
            return Poll::Ready(Ok(()));
        };
        let state = match this.shared.poll_lock(context) {
            Poll::Ready(state) => state,
            Poll::Pending => return Poll::Pending,
        };
        Poll::Ready(dial.flush(&state.stack))
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // send our FIN (passive-close: the app finished reading + echoing). The
        // stack keeps the connection until the peer acks it; we do not block.
        let this = self.get_mut();
        let mut state = match this.shared.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        let outbound = match &this.dial {
            Some(dial) => dial.close(&mut state.stack)?,
            None => state.stack.close(this.id),
        };
        for (peer, segment) in outbound {
            state.tx_segment(peer, &segment);
        }
        Poll::Ready(Ok(()))
    }
}

/// A dpdk-backed active-open TCP client: ARP-resolves the peer, drives the
/// handshake, and yields a connected [`DpdkStreamConnection`]. One per process
/// (EAL is process-global).
pub struct DpdkStreamUpstream {
    shared: Arc<Shared>,
    peer_ip: [u8; 4],
    peer_port: u16,
}

impl DpdkStreamUpstream {
    /// Bring up a net_tap vdev and prepare to connect from `local` to `peer`.
    ///
    /// # Errors
    /// Propagates EAL / pool / port bring-up failures as [`DpdkError`].
    pub fn bind(
        local: SocketAddrV4,
        peer: SocketAddrV4,
        iface: &str,
        pmd_dir: &str,
    ) -> Result<Self, DpdkError> {
        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal = Eal::init(&[
            "proxima-net-dpdk",
            "-l",
            "0",
            "--no-pci",
            "-d",
            pmd_dir,
            &vdev,
        ])?;
        let pool = Mempool::create("pnd_up_pool", 8192, -1)?;
        let port = Port::init(0, &pool)?;
        let our_mac = port.mac()?;
        let our_ip = local.ip().octets();
        let our_port = local.port();
        let stack = TcpStack::new(our_ip, our_port, 0x2000);
        let state = State {
            _eal: eal,
            pool,
            port,
            stack,
            start: StdInstant::now(),
            our_mac,
            our_ip,
            our_port,
            arp: HashMap::new(),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                inner: AsyncMutex::new(state),
                dial: Some(Arc::new(DialLease::new((peer.ip().octets(), peer.port())))),
            }),
            peer_ip: peer.ip().octets(),
            peer_port: peer.port(),
        })
    }
}

impl StreamUpstream for DpdkStreamUpstream {
    type Conn = DpdkStreamConnection;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let mut operation = None;
        Box::pin(core::future::poll_fn(move |context| {
            self.poll_dial(context, &mut operation)
        }))
    }
}

impl DpdkStreamUpstream {
    fn poll_dial(
        &self,
        context: &mut Context<'_>,
        operation: &mut Option<DialGuard>,
    ) -> Poll<io::Result<DpdkStreamConnection>> {
        let mut state = match self.shared.poll_lock(context) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        let Some(lease) = &self.shared.dial else {
            return Poll::Ready(Err(io::Error::other("missing upstream tuple lease")));
        };
        if operation.is_none() {
            *operation = Some(lease.claim(&mut state.stack)?);
        }
        let Some(dial) = operation.as_mut() else {
            return Poll::Ready(Err(io::Error::other("missing dial ownership")));
        };
        state.pump();
        if let Some(id) = dial.connection() {
            if !dial.owns(&state.stack) {
                return Poll::Ready(Err(io::Error::from(io::ErrorKind::ConnectionReset)));
            }
            while let Some(connected) = state.stack.poll_connected() {
                if connected == id && dial.connected(&state.stack) {
                    let Some(dial) = operation.take() else {
                        return Poll::Ready(Err(io::Error::other("missing dial ownership")));
                    };
                    let dial = dial.complete(&state.stack)?;
                    return Poll::Ready(Ok(DpdkStreamConnection {
                        shared: Arc::clone(&self.shared),
                        dial: Some(dial),
                        id,
                        peer: SocketAddr::V4(SocketAddrV4::new(
                            Ipv4Addr::from(self.peer_ip),
                            self.peer_port,
                        )),
                    }));
                }
            }
        } else if let Some(&mac) = state.arp.get(&self.peer_ip) {
            let peer = Endpoint {
                mac,
                ip: self.peer_ip,
                port: self.peer_port,
            };
            let outbound = dial.start(&mut state.stack, peer)?;
            for (target, segment) in outbound {
                state.tx_segment(target, &segment);
            }
        } else {
            state.send_arp_request(self.peer_ip);
        }
        context.waker().wake_by_ref();
        Poll::Pending
    }
}
