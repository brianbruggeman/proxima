//! Prime-native UDP `PacketListener` over dpdk (`proxima_net::packet`). No
//! tokio: a poll-mode driver has no fd to register, so `poll_recv` busy-polls
//! the RX ring and re-arms the waker on `Pending` — which is exactly the dpdk
//! model (a PMD core spins on `rte_eth_rx_burst`). It answers ARP/ICMP inline so
//! the kernel can reach us, learns peer MACs into a small ARP cache, and frames
//! outbound datagrams with `proxima-inet-codec`.
//!
//! `Inner` sits behind an [`AsyncMutex`], manually polled (never a
//! thread-parking mutex) per the workspace lock-discipline rule: `poll_recv`/
//! `poll_send` are reached only from `poll_*` futures, so a blocking mutex is
//! never the right primitive, contended or not.
//!
//! EAL is process-global, so one dpdk listener per process. The dpdk resources
//! are core-pinned; the listener must be polled from a single core (the prime
//! per-core model) — the `unsafe` Send/Sync impls assert exactly that.

use super::port::{self, Port};
use super::{DpdkError, Eal, Mempool, RawMbuf};
use bytes::Bytes;
use core::future::Future;
use core::pin::pin;
use proxima_primitives::sync::{AsyncMutex, AsyncMutexGuard};
use proxima_protocols::inet::ethernet::{self, EtherType, EthernetFrame};
use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
use proxima_protocols::inet::udp::{self, UdpHeader};
use crate::packet::{Packet, PacketListener};
use crate::stack::{self, Action};
use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::task::{Context, Poll};

const BURST: usize = 32;
const ETH: usize = 14;
const IP: usize = 20;
const UDP: usize = 8;

struct Inner {
    _eal: Eal,
    pool: Mempool,
    port: Port,
    rx_queue: VecDeque<Packet>,
    arp: HashMap<[u8; 4], [u8; 6]>,
}

/// A dpdk-backed UDP datagram listener bound to one address.
pub struct DpdkPacketListener {
    inner: AsyncMutex<Inner>,
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    our_port: u16,
    local_addr: SocketAddr,
}

// SAFETY: every access to the dpdk resources goes through `inner`'s lock, and
// the listener is polled from a single prime core (the PMD's rx/tx queues are
// single-producer/consumer per core). The raw mbuf/port pointers never escape.
unsafe impl Send for DpdkPacketListener {}
unsafe impl Sync for DpdkPacketListener {}

impl DpdkPacketListener {
    /// Bring up a net_tap vdev and bind a UDP listener on `bind`. `iface` is the
    /// kernel-visible tap name; `pmd_dir` is dpdk's runtime PMD path.
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
        let pool = Mempool::create("pnd_udp_pool", 8192, -1)?;
        let port = Port::init(0, &pool)?;
        let our_mac = port.mac()?;
        let our_ip = bind.ip().octets();
        let our_port = bind.port();
        Ok(Self {
            inner: AsyncMutex::new(Inner {
                _eal: eal,
                pool,
                port,
                rx_queue: VecDeque::new(),
                arp: HashMap::new(),
            }),
            our_mac,
            our_ip,
            our_port,
            local_addr: SocketAddr::V4(bind),
        })
    }

    // manually polls the async gate mutex's `lock()` future: no thread ever
    // blocks, and on the (never, single-core) contended path the caller's own
    // waker is queued and re-woken on release instead of parking.
    fn poll_lock(&self, cx: &mut Context<'_>) -> Poll<AsyncMutexGuard<'_, Inner>> {
        let lock_future = pin!(self.inner.lock());
        lock_future.poll(cx)
    }
}

impl Inner {
    // drain one rx burst: answer arp/icmp, learn peer macs, queue our udp.
    fn poll_rx(&mut self, our_mac: [u8; 6], our_ip: [u8; 4], our_port: u16) {
        let mut bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let received = usize::from(self.port.rx_burst(&mut bufs));
        for &mbuf in &bufs[..received] {
            let frame = unsafe { port::frame_bytes_mut(mbuf) };
            match stack::handle_frame(frame, our_mac, our_ip) {
                Action::Transmit => {
                    let mut one = [mbuf];
                    if self.port.tx_burst(&mut one) == 0 {
                        unsafe { port::free(mbuf) };
                    }
                }
                Action::Drop => {
                    if let Some((src, src_mac, data)) = parse_udp_to_us(frame, our_ip, our_port) {
                        self.arp.insert(src.ip().octets(), src_mac);
                        let dst =
                            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::from(our_ip), our_port));
                        self.rx_queue.push_back(Packet {
                            src: SocketAddr::V4(src),
                            dst,
                            data,
                        });
                    }
                    unsafe { port::free(mbuf) };
                }
            }
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
        let total = u16::try_from(ETH + IP + UDP).unwrap_or(0) + payload_len;
        let mbuf = self.pool.alloc();
        if mbuf.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "mbuf pool exhausted",
            ));
        }
        let Some(buf) = (unsafe { port::frame_append(mbuf, total) }) else {
            unsafe { port::free(mbuf) };
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "no mbuf tailroom",
            ));
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
        let mut one = [mbuf];
        if self.port.tx_burst(&mut one) == 0 {
            unsafe { port::free(mbuf) };
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "tx ring full"));
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

impl PacketListener for DpdkPacketListener {
    fn poll_recv(&self, cx: &mut Context<'_>, _buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        let mut inner = match self.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
        };
        if inner.rx_queue.is_empty() {
            inner.poll_rx(self.our_mac, self.our_ip, self.our_port);
        }
        if let Some(packet) = inner.rx_queue.pop_front() {
            return Poll::Ready(Ok(packet));
        }
        // PMD has no fd: re-arm and re-poll next turn (busy-poll, the dpdk model).
        cx.waker().wake_by_ref();
        Poll::Pending
    }

    fn poll_send(&self, cx: &mut Context<'_>, packet: &Packet) -> Poll<io::Result<()>> {
        // the trait's mesh convention: `packet.src` is the destination to send to.
        let SocketAddr::V4(dst) = packet.src else {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "ipv6 not supported",
            )));
        };
        let mut inner = match self.poll_lock(cx) {
            Poll::Ready(guard) => guard,
            Poll::Pending => return Poll::Pending,
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
