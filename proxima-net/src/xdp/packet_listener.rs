//! Prime-native UDP `PacketListener` over AF_XDP (`proxima_net::packet`). No
//! tokio: on a proxima worker `poll_recv` registers the xsk fd on the per-core
//! reactor for read-readiness and parks; off a worker it falls back to
//! busy-poll (re-arming the waker on `Pending`). It answers ARP/ICMP inline via
//! `proxima_net::stack::handle_frame`, learns peer MACs into a small ARP
//! cache, and frames outbound datagrams with `proxima-inet-codec`.
//!
//! An AF_XDP queue is single-producer/single-consumer, so the listener must
//! be polled from a single core — the same constraint the dpdk backend has.
//! Unlike dpdk this needs no manual `unsafe impl Send`: every field the
//! `Mutex` guards is itself `Send` (the rings only hold an mmap pointer plus
//! plain index arithmetic), so the auto-derived bound is exactly right.
//!
//! The reactor is edge-triggered, so `poll_rx` fully drains the RX ring before
//! `poll_recv` arms readiness — a residual descriptor would produce no fresh
//! edge and the wake would be lost.

use super::bpf::XdpProgram;
use super::error::XdpError;
use super::readiness::{Readiness, ReadyState};
use super::sized;
use super::sys;
use super::uapi::{self, xdp_desc};
use super::xsk::{RingSizes, UmemConfig, XskSocket};
use bytes::Bytes;
use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
use proxima_protocols::inet::udp::{self, UdpHeader};
use crate::packet::{Packet, PacketListener};
use crate::stack::{self, Action};
use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::Mutex;
use std::task::{Context, Poll};

const ETH: usize = 14;
const IP: usize = 20;
const UDP: usize = 8;

struct Inner {
    socket: XskSocket,
    readiness: Readiness,
    rx_queue: VecDeque<Packet>,
    arp: HashMap<[u8; 4], [u8; 6]>,
}

/// An AF_XDP-backed UDP datagram listener bound to one queue and address.
///
/// `_program` is kept alive for the listener's lifetime so the redirect XDP
/// program stays attached; its `Drop` detaches the program from the netdev.
pub struct XdpPacketListener {
    inner: Mutex<Inner>,
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    our_port: u16,
    local_addr: SocketAddr,
    _program: XdpProgram,
}

impl XdpPacketListener {
    /// Bring up an AF_XDP socket on `ifname`/`queue_id`, load and attach the
    /// redirect XDP program (SKB mode), point `xskmap[queue_id]` at the
    /// socket, and bind a UDP listener on `bind`. The kernel does not hand
    /// back the interface's link address on bind, so the caller supplies
    /// `our_mac`.
    ///
    /// # Errors
    /// Propagates socket bring-up ([`XskSocket::bind`]) and BPF load/attach
    /// failures as [`XdpError`].
    pub fn bind(
        ifname: &str,
        queue_id: u32,
        our_mac: [u8; 6],
        bind: SocketAddrV4,
    ) -> Result<Self, XdpError> {
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
        // copy mode is the reliable AF_XDP path on veth; need-wakeup lets the
        // kernel tell us when the fill/tx rings want a syscall kick.
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

        let readiness = Readiness::new(socket.fd());
        Ok(Self {
            inner: Mutex::new(Inner {
                socket,
                readiness,
                rx_queue: VecDeque::new(),
                arp: HashMap::new(),
            }),
            our_mac,
            our_ip: bind.ip().octets(),
            our_port: bind.port(),
            local_addr: SocketAddr::V4(bind),
            _program: program,
        })
    }
}

impl Inner {
    // fully drain the rx ring (edge-triggered reactor requires it) in batches:
    // answer arp/icmp, learn peer macs, queue our udp. One atomic release per
    // batch, one tx kick for the whole drain.
    fn poll_rx(&mut self, our_mac: [u8; 6], our_ip: [u8; 4], our_port: u16) {
        let mut descs = [xdp_desc::default(); sized::BATCH_RX_DRAIN];
        loop {
            let received = self.socket.rx_mut().peek_batch(&mut descs);
            if received == 0 {
                break;
            }
            for &desc in &descs[..received] {
                let frame_len = desc.len as usize;
                // SAFETY: `desc.addr` is a UMEM offset the kernel just handed
                // back via the RX ring, and `desc.len` never exceeds the frame
                // size we registered, so the slice stays within the frame this
                // socket owns exclusively until it is freed or requeued below.
                let frame = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.socket.umem_mut().frame_ptr(desc.addr),
                        frame_len,
                    )
                };
                match stack::handle_frame(frame, our_mac, our_ip) {
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
                        if let Some((src, src_mac, data)) = parse_udp_to_us(frame, our_ip, our_port)
                        {
                            self.arp.insert(src.ip().octets(), src_mac);
                            let dst =
                                SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(our_ip), our_port));
                            self.rx_queue.push_back(Packet {
                                src: SocketAddr::V4(src),
                                dst,
                                data,
                            });
                        }
                        self.socket.umem_mut().free_frame(desc.addr);
                    }
                }
            }
        }
        // poke the kernel to flush any arp/icmp replies we queued for tx.
        let _ = self.socket.kick_tx();
        self.refill_fill_ring();
    }

    // reclaim completed tx frames and top the fill ring back up in batches so
    // the kernel always has somewhere to receive into — one atomic release per
    // reclaimed batch, one atomic commit per refilled batch.
    fn refill_fill_ring(&mut self) {
        let mut completed = [0u64; sized::BATCH_RX_DRAIN];
        loop {
            let reclaimed = self.socket.completion_mut().pop_batch(&mut completed);
            if reclaimed == 0 {
                break;
            }
            for &addr in &completed[..reclaimed] {
                self.socket.umem_mut().free_frame(addr);
            }
        }
        let mut frames = [0u64; sized::BATCH_RX_DRAIN];
        loop {
            let mut count = 0;
            while count < sized::BATCH_RX_DRAIN {
                let Some(frame) = self.socket.umem_mut().alloc_frame() else {
                    break;
                };
                frames[count] = frame;
                count += 1;
            }
            if count == 0 {
                break;
            }
            let pushed = self.socket.fill_mut().push_batch(&frames[..count]);
            for &addr in &frames[pushed..count] {
                self.socket.umem_mut().free_frame(addr);
            }
            if pushed < count {
                break;
            }
        }
        // need-wakeup: if the kernel ran the fill ring dry it set the flag;
        // poke the rx path so it picks up the frames we just replenished.
        if self.socket.fill_needs_wakeup() {
            let _ = self.socket.wake_rx();
        }
    }

    fn send_udp(
        &mut self,
        dst_mac: [u8; 6],
        our_mac: [u8; 6],
        our_ip: [u8; 4],
        our_port: u16,
        dst: SocketAddrV4,
        payload: &[u8],
    ) -> io::Result<()> {
        let payload_len = u16::try_from(payload.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "datagram too large"))?;
        let total = ETH + IP + UDP + payload.len();
        if total > sized::UMEM_FRAME_SIZE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "datagram exceeds frame size",
            ));
        }
        let Some(addr) = self.socket.umem_mut().alloc_frame() else {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "no free umem frame",
            ));
        };
        // SAFETY: `addr` was just allocated from this Umem's free list, so no
        // other reference to this frame exists and it holds at least `total`
        // bytes (`sized::UMEM_FRAME_SIZE`, checked above).
        let buf = unsafe {
            std::slice::from_raw_parts_mut(self.socket.umem_mut().frame_ptr(addr), total)
        };
        let dst_ip = dst.ip().octets();
        let _ = ethernet::write_header(&mut buf[..ETH], dst_mac, our_mac, EtherType::Ipv4);
        let l4_len = u16::try_from(UDP).unwrap_or(0) + payload_len;
        let _ = ipv4::write_header(
            &mut buf[ETH..ETH + IP],
            our_ip,
            dst_ip,
            Ipv4Protocol::Udp,
            64,
            l4_len,
            0,
        );
        buf[ETH + IP + UDP..].copy_from_slice(payload);
        let _ = udp::write_header(
            &mut buf[ETH + IP..],
            our_ip,
            dst_ip,
            our_port,
            dst.port(),
            payload,
        );

        let desc = xdp_desc {
            addr,
            len: u32::try_from(total).unwrap_or(0),
            options: 0,
        };
        if !self.socket.tx_mut().push(desc) {
            self.socket.umem_mut().free_frame(addr);
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "tx ring full"));
        }
        // need-wakeup: only syscall when the kernel says the TX ring is idle and
        // needs a poke; while it is actively draining we skip the sendto.
        if self.socket.tx_needs_wakeup() {
            self.socket.kick_tx()?;
        }
        Ok(())
    }
}

// parse an inbound frame as a UDP datagram addressed to us; returns the peer
// address, its MAC, and the payload.
fn parse_udp_to_us(
    frame: &[u8],
    our_ip: [u8; 4],
    our_port: u16,
) -> Option<(SocketAddrV4, [u8; 6], Bytes)> {
    let eth = EthernetFrame::parse(frame).ok()?;
    if eth.ether_type() != EtherType::Ipv4 {
        return None;
    }
    let ip = Ipv4Header::parse(eth.payload()).ok()?;
    if ip.protocol() != Ipv4Protocol::Udp || ip.destination() != our_ip {
        return None;
    }
    let datagram = UdpHeader::parse(ip.payload()).ok()?;
    if datagram.destination_port() != our_port {
        return None;
    }
    let src = SocketAddrV4::new(Ipv4Addr::from(ip.source()), datagram.source_port());
    Some((
        src,
        eth.source(),
        Bytes::copy_from_slice(datagram.payload()),
    ))
}

impl PacketListener for XdpPacketListener {
    fn poll_recv(&self, cx: &mut Context<'_>, _buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        loop {
            if let Some(packet) = inner.rx_queue.pop_front() {
                return Poll::Ready(Ok(packet));
            }
            inner.poll_rx(self.our_mac, self.our_ip, self.our_port);
            if !inner.rx_queue.is_empty() {
                continue;
            }
            // ring fully drained and nothing for us: park on reactor readiness,
            // or busy-poll when off a proxima worker.
            match inner.readiness.poll(cx) {
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

    fn poll_send(&self, _cx: &mut Context<'_>, packet: &Packet) -> Poll<io::Result<()>> {
        // the trait's mesh convention: `packet.src` is the destination to send to.
        let SocketAddr::V4(dst) = packet.src else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "ipv6 not supported",
            )));
        };
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(&dst_mac) = inner.arp.get(&dst.ip().octets()) else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::AddrNotAvailable,
                "no arp entry for peer",
            )));
        };
        Poll::Ready(inner.send_udp(
            dst_mac,
            self.our_mac,
            self.our_ip,
            self.our_port,
            dst,
            &packet.data,
        ))
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        Some(self.local_addr)
    }
}
