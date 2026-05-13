//! Prime-native TCP `StreamListener`/`StreamUpstream` over AF_XDP
//! (`proxima_primitives::stream`). No tokio and no busy-poll: on a proxima worker the ONE
//! xsk fd is registered on the per-core reactor (via [`super::readiness`], the
//! same mechanism the UDP `PacketListener` uses) and every poll point
//! (`poll_accept`/`poll_read`/`poll_connect`) parks on it after fully draining
//! the RX ring (EPOLLET). Off a worker (plain `block_on`) it falls back to
//! busy-poll so the block_on examples still work. Accepted connections are
//! `AsyncRead`/`AsyncWrite` handles sharing the stack under a `Mutex`; ARP/ICMP
//! are answered inline via `proxima_net::stack`, and each outbound segment is
//! framed with `proxima-inet-codec` into a fresh UMEM frame pushed onto the TX
//! ring.
//!
//! Readiness sharing: one fd can only be registered once, so `State` holds a
//! single [`Readiness`] and a small set of parked wakers. Any poll point that
//! makes RX progress wakes the other parked wakers so a shared pump delivers
//! bytes to every connection's read task; the reactor's `POLLIN` re-drives the
//! last-armed waker, which fans the wake out to the rest.
//!
//! Unlike the dpdk backend this needs no `unsafe impl Send/Sync`: `State` holds
//! only the mmap-backed `XskSocket` (whose rings/UMEM are `Send`) and plain
//! data, so `Mutex<State>` is `Send + Sync` by construction.

use super::bpf::XdpProgram;
use super::error::XdpError;
use super::readiness::{Readiness, ReadyState};
use super::sized;
use super::sys;
use super::uapi::{self, xdp_desc};
use super::xsk::{RingSizes, UmemConfig, XskSocket};
use futures::io::{AsyncRead, AsyncWrite};
use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
use proxima_protocols::inet::tcp::{self, TcpHeader};
use crate::stack::{self, Action};
use crate::tcp_listener::{Endpoint, Inbound, OutSegment};
use crate::tcp_stack::{ConnId, TcpStack};
use proxima_primitives::stream::{BindAddr, PeerInfo, StreamConnection, StreamListener, StreamUpstream};
use proxima_protocols::tcp::time::Instant as TcpInstant;
use std::collections::HashMap;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Instant as StdInstant;

const ETH: usize = 14;
const IP: usize = 20;
const TCP: usize = 20;
const ARP_FRAME: usize = 42;

struct State {
    socket: XskSocket,
    _program: XdpProgram,
    readiness: Readiness,
    // parked wakers waiting on the shared xsk fd (accept + per-connection reads
    // + connect). One fd registration, many waiters — see the module docs.
    read_wakers: Vec<Waker>,
    stack: TcpStack,
    start: StdInstant,
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    our_port: u16,
    arp: HashMap<[u8; 4], [u8; 6]>,
}

struct Shared {
    inner: Mutex<State>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

// bring up the xsk socket + redirect program for `ifname`/`queue_id`: bind in
// copy+need-wakeup mode, seed the fill ring, load/attach the XDP redirect and
// point xskmap[queue] at the socket. Shared by the listener and the upstream.
fn setup_socket(ifname: &str, queue_id: u32) -> Result<(XskSocket, XdpProgram), XdpError> {
    let ifindex = sys::if_nametoindex(ifname)?;
    let umem_cfg = UmemConfig {
        frame_count: sized::UMEM_FRAME_COUNT,
        frame_size: sized::UMEM_FRAME_SIZE,
    };
    let ring_sizes = RingSizes {
        fill: sized::RINGS_FILL,
        completion: sized::RINGS_COMPLETION,
        rx: sized::RINGS_RX,
        tx: sized::RINGS_TX,
    };
    let mut socket = XskSocket::bind(
        ifname,
        queue_id,
        umem_cfg,
        ring_sizes,
        uapi::XDP_COPY | uapi::XDP_USE_NEED_WAKEUP,
    )?;
    for _ in 0..ring_sizes.fill {
        let Some(frame) = socket.umem_mut().alloc_frame() else {
            break;
        };
        if !socket.fill_mut().push(frame) {
            socket.umem_mut().free_frame(frame);
            break;
        }
    }
    let mut program = XdpProgram::load(queue_id + 1)?;
    program.update_map(queue_id, socket.fd())?;
    program.attach(ifindex)?;
    Ok((socket, program))
}

/// An AF_XDP-backed TCP listener bound to one address/queue.
pub struct XdpStreamListener {
    shared: Arc<Shared>,
    local_addr: SocketAddr,
}

impl XdpStreamListener {
    /// Bring up an AF_XDP socket on `ifname`/`queue_id` and accept TCP
    /// connections on `bind`. `our_mac` is the interface link address.
    ///
    /// # Errors
    /// Propagates socket / BPF bring-up failures as [`XdpError`].
    pub fn bind(
        bind: SocketAddrV4,
        ifname: &str,
        queue_id: u32,
        our_mac: [u8; 6],
    ) -> Result<Self, XdpError> {
        let (socket, program) = setup_socket(ifname, queue_id)?;
        let our_ip = bind.ip().octets();
        let our_port = bind.port();
        let stack = TcpStack::new(our_ip, our_port, 0x1000);
        let readiness = Readiness::new(socket.fd());
        let state = State {
            socket,
            _program: program,
            readiness,
            read_wakers: Vec::new(),
            stack,
            start: StdInstant::now(),
            our_mac,
            our_ip,
            our_port,
            arp: HashMap::new(),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(state),
            }),
            local_addr: SocketAddr::V4(bind),
        })
    }
}

impl State {
    fn now(&self) -> TcpInstant {
        TcpInstant::from_micros(u64::try_from(self.start.elapsed().as_micros()).unwrap_or(u64::MAX))
    }

    // fully drain the rx ring (EPOLLET requires it): answer arp/icmp, learn peer
    // macs, route TCP through the stack, transmit replies. Returns whether any
    // descriptor was processed, so a poll point can wake peers on RX progress.
    fn pump(&mut self) -> bool {
        let now = self.now();
        let mut progressed = false;
        while let Some(desc) = self.socket.rx_mut().pop() {
            progressed = true;
            let frame_len = desc.len as usize;
            // SAFETY: `desc.addr`/`desc.len` are a UMEM offset+length the kernel
            // just handed back on the RX ring, within a frame this socket owns
            // exclusively until it is freed or requeued below.
            let frame = unsafe {
                std::slice::from_raw_parts_mut(
                    self.socket.umem_mut().frame_ptr(desc.addr),
                    frame_len,
                )
            };
            match stack::handle_frame(frame, self.our_mac, self.our_ip) {
                Action::Transmit => {
                    let reply = xdp_desc {
                        addr: desc.addr,
                        len: desc.len,
                        options: 0,
                    };
                    if !self.socket.tx_mut().push(reply) {
                        self.socket.umem_mut().free_frame(desc.addr);
                    }
                }
                Action::Drop => {
                    if let Some((ip, mac)) = stack::parse_arp_reply(frame) {
                        self.arp.insert(ip, mac);
                        self.socket.umem_mut().free_frame(desc.addr);
                    } else if let Some(inbound) = classify_tcp(frame, self.our_ip, self.our_port) {
                        self.socket.umem_mut().free_frame(desc.addr);
                        let outbound = self.stack.on_inbound(&inbound.borrow(), now);
                        for (peer, segment) in outbound {
                            self.tx_segment(peer, &segment);
                        }
                    } else {
                        self.socket.umem_mut().free_frame(desc.addr);
                    }
                }
            }
        }
        let _ = self.socket.kick_tx();
        self.refill_fill_ring();
        progressed
    }

    // arm the ONE shared reactor registration for read-readiness and record the
    // caller's waker in the parked set (deduped). Returns the readiness outcome.
    fn arm(&mut self, cx: &Context<'_>) -> io::Result<ReadyState> {
        if !self
            .read_wakers
            .iter()
            .any(|waker| waker.will_wake(cx.waker()))
        {
            self.read_wakers.push(cx.waker().clone());
        }
        self.readiness.poll(cx)
    }

    // wake every parked waker except the current task's (the caller re-checks
    // its own condition inline). Called after a pump made RX progress so waiters
    // on other connections re-poll and read the bytes just delivered.
    fn wake_others(&mut self, cx: &Context<'_>) {
        let mut index = 0;
        while index < self.read_wakers.len() {
            if self.read_wakers[index].will_wake(cx.waker()) {
                index += 1;
            } else {
                self.read_wakers.swap_remove(index).wake();
            }
        }
    }

    fn refill_fill_ring(&mut self) {
        while let Some(addr) = self.socket.completion_mut().pop() {
            self.socket.umem_mut().free_frame(addr);
        }
        while let Some(frame) = self.socket.umem_mut().alloc_frame() {
            if !self.socket.fill_mut().push(frame) {
                self.socket.umem_mut().free_frame(frame);
                break;
            }
        }
        if self.socket.fill_needs_wakeup() {
            let _ = self.socket.wake_rx();
        }
    }

    fn send_arp_request(&mut self, target_ip: [u8; 4]) {
        let Some(addr) = self.socket.umem_mut().alloc_frame() else {
            return;
        };
        // SAFETY: `addr` was just allocated from the free list, so no other
        // reference exists and the frame holds at least `ARP_FRAME` bytes.
        let buf = unsafe {
            std::slice::from_raw_parts_mut(self.socket.umem_mut().frame_ptr(addr), ARP_FRAME)
        };
        let written = stack::build_arp_request(buf, self.our_mac, self.our_ip, target_ip);
        if written == 0 {
            self.socket.umem_mut().free_frame(addr);
            return;
        }
        let desc = xdp_desc {
            addr,
            len: u32::try_from(written).unwrap_or(0),
            options: 0,
        };
        if !self.socket.tx_mut().push(desc) {
            self.socket.umem_mut().free_frame(addr);
            return;
        }
        let _ = self.socket.kick_tx();
    }

    fn tx_segment(&mut self, peer: Endpoint, segment: &OutSegment) {
        let payload_len = u16::try_from(segment.payload.len()).unwrap_or(0);
        let total = ETH + IP + TCP + segment.payload.len();
        if total > sized::UMEM_FRAME_SIZE as usize {
            return;
        }
        let Some(addr) = self.socket.umem_mut().alloc_frame() else {
            return;
        };
        // SAFETY: `addr` was just allocated from the free list, so no other
        // reference exists and the frame holds at least `total` bytes.
        let buf = unsafe {
            std::slice::from_raw_parts_mut(self.socket.umem_mut().frame_ptr(addr), total)
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
        let desc = xdp_desc {
            addr,
            len: u32::try_from(total).unwrap_or(0),
            options: 0,
        };
        if !self.socket.tx_mut().push(desc) {
            self.socket.umem_mut().free_frame(addr);
            return;
        }
        let _ = self.socket.kick_tx();
    }
}

// owned inbound so the source frame can be freed before replies are built.
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

impl StreamListener for XdpStreamListener {
    type Conn = XdpStreamConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let mut state = self.shared.lock();
        loop {
            let progressed = state.pump();
            if let Some(id) = state.stack.poll_accept() {
                if progressed {
                    state.wake_others(cx);
                }
                let peer = state.stack.peer(id).map_or(self.local_addr, |endpoint| {
                    SocketAddr::V4(SocketAddrV4::new(
                        Ipv4Addr::from(endpoint.ip),
                        endpoint.port,
                    ))
                });
                return Poll::Ready(Ok(XdpStreamConnection {
                    shared: self.shared.clone(),
                    id,
                    peer,
                }));
            }
            if progressed {
                state.wake_others(cx);
                continue;
            }
            match state.arm(cx) {
                Ok(ReadyState::Retry) => continue,
                Ok(ReadyState::Parked) => return Poll::Pending,
                Ok(ReadyState::OffWorker) => {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        Some(BindAddr::Tcp(self.local_addr))
    }
}

/// An accepted AF_XDP TCP connection: an `AsyncRead`/`AsyncWrite` byte stream
/// over the shared [`TcpStack`].
pub struct XdpStreamConnection {
    shared: Arc<Shared>,
    id: ConnId,
    peer: SocketAddr,
}

impl StreamConnection for XdpStreamConnection {
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Tcp(self.peer))
    }
}

impl AsyncRead for XdpStreamConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut state = this.shared.lock();
        loop {
            let progressed = state.pump();
            let read = state.stack.read(this.id, buf);
            if read > 0 {
                if progressed {
                    state.wake_others(cx);
                }
                return Poll::Ready(Ok(read));
            }
            if state.stack.read_closed(this.id) {
                if progressed {
                    state.wake_others(cx);
                }
                return Poll::Ready(Ok(0));
            }
            if progressed {
                state.wake_others(cx);
                continue;
            }
            match state.arm(cx) {
                Ok(ReadyState::Retry) => continue,
                Ok(ReadyState::Parked) => return Poll::Pending,
                Ok(ReadyState::OffWorker) => {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
    }
}

impl AsyncWrite for XdpStreamConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut state = this.shared.lock();
        let now = state.now();
        let outbound = state.stack.write(this.id, buf, now);
        for (peer, segment) in outbound {
            state.tx_segment(peer, &segment);
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // send our FIN (passive-close: the app finished reading + echoing). The
        // stack keeps the connection until the peer acks it; we do not block.
        let this = self.get_mut();
        let mut state = this.shared.lock();
        let outbound = state.stack.close(this.id);
        for (peer, segment) in outbound {
            state.tx_segment(peer, &segment);
        }
        let _ = state.socket.kick_tx();
        Poll::Ready(Ok(()))
    }
}

enum ConnectPhase {
    Resolving,
    Connecting(ConnId),
}

/// An AF_XDP-backed active-open TCP client: ARP-resolves the peer, drives the
/// handshake, and yields a connected [`XdpStreamConnection`].
pub struct XdpStreamUpstream {
    shared: Arc<Shared>,
    peer_ip: [u8; 4],
    peer_port: u16,
    phase: Mutex<ConnectPhase>,
}

impl XdpStreamUpstream {
    /// Bring up an AF_XDP socket on `ifname`/`queue_id` and prepare to connect
    /// from `local` to `peer`. `our_mac` is the interface link address.
    ///
    /// # Errors
    /// Propagates socket / BPF bring-up failures as [`XdpError`].
    pub fn bind(
        local: SocketAddrV4,
        peer: SocketAddrV4,
        ifname: &str,
        queue_id: u32,
        our_mac: [u8; 6],
    ) -> Result<Self, XdpError> {
        let (socket, program) = setup_socket(ifname, queue_id)?;
        let our_ip = local.ip().octets();
        let our_port = local.port();
        let stack = TcpStack::new(our_ip, our_port, 0x2000);
        let readiness = Readiness::new(socket.fd());
        let state = State {
            socket,
            _program: program,
            readiness,
            read_wakers: Vec::new(),
            stack,
            start: StdInstant::now(),
            our_mac,
            our_ip,
            our_port,
            arp: HashMap::new(),
        };
        Ok(Self {
            shared: Arc::new(Shared {
                inner: Mutex::new(state),
            }),
            peer_ip: peer.ip().octets(),
            peer_port: peer.port(),
            phase: Mutex::new(ConnectPhase::Resolving),
        })
    }

    fn phase(&self) -> MutexGuard<'_, ConnectPhase> {
        match self.phase.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl StreamUpstream for XdpStreamUpstream {
    type Conn = XdpStreamConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let mut state = self.shared.lock();
        loop {
            let progressed = state.pump();
            let mut connected_id = None;
            {
                let mut phase = self.phase();
                match *phase {
                    ConnectPhase::Resolving => {
                        if let Some(&mac) = state.arp.get(&self.peer_ip) {
                            let peer = Endpoint {
                                mac,
                                ip: self.peer_ip,
                                port: self.peer_port,
                            };
                            let (id, outbound) = state.stack.connect(peer);
                            for (target, segment) in outbound {
                                state.tx_segment(target, &segment);
                            }
                            *phase = ConnectPhase::Connecting(id);
                        } else {
                            state.send_arp_request(self.peer_ip);
                        }
                    }
                    ConnectPhase::Connecting(id) => {
                        while let Some(done_id) = state.stack.poll_connected() {
                            if done_id == id {
                                connected_id = Some(id);
                            }
                        }
                    }
                }
            }
            if let Some(id) = connected_id {
                let peer = SocketAddr::V4(SocketAddrV4::new(
                    Ipv4Addr::from(self.peer_ip),
                    self.peer_port,
                ));
                return Poll::Ready(Ok(XdpStreamConnection {
                    shared: self.shared.clone(),
                    id,
                    peer,
                }));
            }
            if progressed {
                state.wake_others(cx);
                continue;
            }
            match state.arm(cx) {
                Ok(ReadyState::Retry) => continue,
                Ok(ReadyState::Parked) => return Poll::Pending,
                Ok(ReadyState::OffWorker) => {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
    }
}
