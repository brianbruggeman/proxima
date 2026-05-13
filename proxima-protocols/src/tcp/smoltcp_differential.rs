//! Differential oracle: smoltcp 0.13 drives a real two-stack IPv4+TCP session
//! through a scripted byte sequence. The same bytes, split into segments at the
//! same sequence offsets, are replayed through our `DataPath` reassembler. The
//! test asserts that the application-layer content delivered by our side is
//! byte-for-byte identical to what smoltcp's receiver consumed.
//!
//! Seam used: `smoltcp::iface::Interface::poll` (the only stable public entry
//! point outside the smoltcp crate). `InterfaceInner::process_tcp` is
//! `pub(crate)` inside smoltcp and cannot be called here; we drive smoltcp
//! through its `Device` + `Interface::poll` public contract instead.
//!
//! Because smoltcp chooses its own ISN at connect time the comparison is at the
//! **application-byte level**: both sides must deliver the same N bytes of
//! content in order. The payload is a deterministic `(0..=255).cycle()` slice,
//! so a delivery-offset → byte-content mapping is always computable.
//!
//! The OOO scenario is exercised by feeding data segments to our DataPath in
//! reversed order while smoltcp sees them in-order — the cumulative delivery
//! count and content must still agree.

#![cfg(all(test, feature = "std"))]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::vec::Vec;

use smoltcp::iface::{Config, Interface, SocketSet, SocketStorage};
use smoltcp::phy::{Device, DeviceCapabilities, Medium};
use smoltcp::socket::tcp;
use smoltcp::time::Instant as SmolInstant;
use smoltcp::wire::{
    HardwareAddress, IpAddress, IpCidr, IpProtocol, Ipv4Address, Ipv4Packet, TcpPacket,
};

use super::super::congestion::Reno;
use super::super::connection::Segment as TcpSeg;
use super::super::seq::SeqNum;
use super::super::time::Instant;
use super::DataPath;

const ADDR_A: Ipv4Address = Ipv4Address::new(10, 0, 0, 1);
const ADDR_B: Ipv4Address = Ipv4Address::new(10, 0, 0, 2);
const LISTEN_PORT: u16 = 9000;
const SMSS: u32 = 1460;
const POLL_LIMIT: usize = 512;
const PAYLOAD_LEN: usize = 1200;

type Packets = Rc<RefCell<VecDeque<Vec<u8>>>>;
type Capture = Rc<RefCell<Vec<Vec<u8>>>>;

/// One side of a fake bidirectional IP link.
struct LinkSide {
    tx_into: Packets,
    rx_from: Packets,
    capture: Capture,
}

impl Device for LinkSide {
    type RxToken<'a>
        = OwnedBuf
    where
        Self: 'a;
    type TxToken<'a>
        = CaptureTx<'a>
    where
        Self: 'a;

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ip;
        caps.max_transmission_unit = 1500;
        caps
    }

    fn receive(&mut self, _ts: SmolInstant) -> Option<(OwnedBuf, CaptureTx<'_>)> {
        let buf = self.rx_from.borrow_mut().pop_front()?;
        Some((
            OwnedBuf(buf),
            CaptureTx {
                tx_into: &self.tx_into,
                capture: &self.capture,
            },
        ))
    }

    fn transmit(&mut self, _ts: SmolInstant) -> Option<CaptureTx<'_>> {
        Some(CaptureTx {
            tx_into: &self.tx_into,
            capture: &self.capture,
        })
    }
}

struct OwnedBuf(Vec<u8>);

impl smoltcp::phy::RxToken for OwnedBuf {
    fn consume<R, F: FnOnce(&[u8]) -> R>(self, f: F) -> R {
        f(&self.0)
    }
}

struct CaptureTx<'a> {
    tx_into: &'a Packets,
    capture: &'a Capture,
}

impl smoltcp::phy::TxToken for CaptureTx<'_> {
    fn consume<R, F: FnOnce(&mut [u8]) -> R>(self, len: usize, f: F) -> R {
        let mut buf = vec![0u8; len];
        let result = f(&mut buf);
        self.capture.borrow_mut().push(buf.clone());
        self.tx_into.borrow_mut().push_back(buf);
        result
    }
}

fn make_iface(addr: Ipv4Address, dev: &mut LinkSide) -> Interface {
    let config = Config::new(HardwareAddress::Ip);
    let mut iface = Interface::new(config, dev, SmolInstant::ZERO);
    iface.update_ip_addrs(|addrs| {
        addrs
            .push(IpCidr::new(IpAddress::Ipv4(addr), 8))
            .expect("ip addr");
    });
    iface
}

/// A parsed TCP data segment extracted from a raw IPv4 frame.
#[derive(Debug, Clone)]
struct TcpFrame {
    seq: u32,
    ack: u32,
    wnd: u32,
    is_syn: bool,
    is_ack: bool,
    payload: Vec<u8>,
}

/// Parse all IPv4+TCP frames captured at the wire, return one `TcpFrame` per packet.
fn parse_frames(frames: &[Vec<u8>]) -> Vec<TcpFrame> {
    let mut out = Vec::new();
    for frame in frames {
        let Ok(ipv4) = Ipv4Packet::new_checked(frame.as_slice()) else {
            continue;
        };
        if ipv4.next_header() != IpProtocol::Tcp {
            continue;
        }
        let Ok(tcp) = TcpPacket::new_checked(ipv4.payload()) else {
            continue;
        };
        out.push(TcpFrame {
            seq: tcp.seq_number().0 as u32,
            ack: tcp.ack_number().0 as u32,
            wnd: tcp.window_len() as u32,
            is_syn: tcp.syn(),
            is_ack: tcp.ack(),
            payload: tcp.payload().to_vec(),
        });
    }
    out
}

/// Build a DataPath in ESTABLISHED state.
/// `iss` = ISS of the local side (SND.UNA = SND.NXT = iss + 1 after SYN).
/// `irs` = IRS (SYN seq from peer; RCV.NXT = irs + 1 after SYN-ACK).
fn make_data_path(iss: SeqNum, irs: SeqNum, peer_wnd: u32) -> DataPath<8, 16, Reno> {
    let snd_start = iss.wrapping_add(1);
    let rcv_start = irs.wrapping_add(1);
    DataPath::established(snd_start, rcv_start, peer_wnd, 65535, SMSS, Reno::new(SMSS))
}

fn smol_ms(millis: u64) -> SmolInstant {
    SmolInstant::from_millis(millis as i64)
}

/// Poll both stacks until `done` returns true or `POLL_LIMIT` ticks pass.
fn poll_until<F>(
    ia: &mut Interface,
    da: &mut LinkSide,
    sa: &mut SocketSet<'_>,
    ib: &mut Interface,
    db: &mut LinkSide,
    sb: &mut SocketSet<'_>,
    start_ms: u64,
    mut done: F,
) -> u64
where
    F: FnMut(&mut SocketSet<'_>, &mut SocketSet<'_>) -> bool,
{
    let mut ms = start_ms;
    for _ in 0..POLL_LIMIT {
        if done(sa, sb) {
            break;
        }
        let now = smol_ms(ms);
        ia.poll(now, da, sa);
        ib.poll(now, db, sb);
        ms += 1;
    }
    ms
}

/// Reconstruct what application bytes correspond to delivered sequence space.
///
/// `delivered_ranges` is a list of `(seq_offset_from_data_start, byte_count)`.
/// `data_start_seq` is the first data-bearing sequence number (irs + 1).
/// `payload` is the original byte array sent by A.
fn bytes_from_ranges(
    delivered_ranges: &[(u32, u32)],
    data_start_seq: u32,
    payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    for (seq, count) in delivered_ranges {
        let offset = seq.wrapping_sub(data_start_seq) as usize;
        let end = offset + *count as usize;
        let end = end.min(payload.len());
        if offset < payload.len() {
            out.extend_from_slice(&payload[offset..end]);
        }
    }
    out
}

/// Session harness: sets up two smoltcp stacks, establishes a TCP connection,
/// sends `payload` from A to B, and returns:
/// - `smoltcp_received`: the bytes B's socket delivered
/// - `data_segments`: the TcpFrames carrying data that A transmitted
/// - `irs`: the peer's ISN (B's SYN seq number), derived from the first ACK sent by A
/// - `iss_syn`: A's own SYN sequence number
fn run_smoltcp_sender(source_port: u16, payload: &[u8]) -> (Vec<u8>, Vec<TcpFrame>, u32, u32) {
    let a_to_b: Packets = Rc::new(RefCell::new(VecDeque::new()));
    let b_to_a: Packets = Rc::new(RefCell::new(VecDeque::new()));
    let a_cap: Capture = Rc::new(RefCell::new(Vec::new()));

    let mut dev_a = LinkSide {
        tx_into: Rc::clone(&a_to_b),
        rx_from: Rc::clone(&b_to_a),
        capture: Rc::clone(&a_cap),
    };
    let mut dev_b = LinkSide {
        tx_into: Rc::clone(&b_to_a),
        rx_from: Rc::clone(&a_to_b),
        capture: Rc::new(RefCell::new(Vec::new())),
    };

    let mut ia = make_iface(ADDR_A, &mut dev_a);
    let mut ib = make_iface(ADDR_B, &mut dev_b);

    let mut store_a = [SocketStorage::EMPTY; 2];
    let mut sa = SocketSet::new(&mut store_a[..]);
    let mut store_b = [SocketStorage::EMPTY; 2];
    let mut sb = SocketSet::new(&mut store_b[..]);

    let handle_a = sa.add(tcp::Socket::new(
        tcp::SocketBuffer::new(vec![0u8; 65535]),
        tcp::SocketBuffer::new(vec![0u8; 65535]),
    ));
    let handle_b = sb.add(tcp::Socket::new(
        tcp::SocketBuffer::new(vec![0u8; 65535]),
        tcp::SocketBuffer::new(vec![0u8; 65535]),
    ));

    sb.get_mut::<tcp::Socket>(handle_b)
        .listen(LISTEN_PORT)
        .expect("listen");
    sa.get_mut::<tcp::Socket>(handle_a)
        .connect(
            ia.context(),
            (IpAddress::Ipv4(ADDR_B), LISTEN_PORT),
            source_port,
        )
        .expect("connect");

    let ms = poll_until(
        &mut ia,
        &mut dev_a,
        &mut sa,
        &mut ib,
        &mut dev_b,
        &mut sb,
        0,
        |sa, sb| {
            sa.get_mut::<tcp::Socket>(handle_a).may_send()
                && sb.get_mut::<tcp::Socket>(handle_b).may_recv()
        },
    );

    assert!(
        sa.get_mut::<tcp::Socket>(handle_a).may_send(),
        "A must be in ESTABLISHED"
    );
    assert!(
        sb.get_mut::<tcp::Socket>(handle_b).may_recv(),
        "B must be ready to receive"
    );

    let sent = sa
        .get_mut::<tcp::Socket>(handle_a)
        .send_slice(payload)
        .expect("send_slice");
    assert_eq!(sent, payload.len(), "full payload enqueued");

    let n = payload.len();
    let ms = poll_until(
        &mut ia,
        &mut dev_a,
        &mut sa,
        &mut ib,
        &mut dev_b,
        &mut sb,
        ms,
        |_sa, sb| sb.get_mut::<tcp::Socket>(handle_b).recv_queue() >= n,
    );
    let _ = ms;

    let mut received = vec![0u8; n];
    let got = sb
        .get_mut::<tcp::Socket>(handle_b)
        .recv_slice(&mut received)
        .expect("recv_slice");
    assert_eq!(got, n, "smoltcp must deliver all {n} bytes");

    let all_frames = parse_frames(&a_cap.borrow());

    // A's SYN: first frame with SYN set and no ACK.
    let iss_syn = all_frames
        .iter()
        .find(|f| f.is_syn && !f.is_ack)
        .map_or(0u32, |f| f.seq);

    // Data segments from A: non-SYN, with payload.
    let data_segs: Vec<TcpFrame> = all_frames
        .into_iter()
        .filter(|f| !f.is_syn && !f.payload.is_empty())
        .collect();

    assert!(!data_segs.is_empty(), "must have at least one data segment");

    // The first data segment's ACK field tells us rcv_nxt on A's side after the
    // handshake = irs + 1.  So irs = first_data_ack - 1.
    let irs = data_segs[0].ack.wrapping_sub(1);

    (received[..got].to_vec(), data_segs, irs, iss_syn)
}

/// Replay data segments into our DataPath and return the sequence ranges delivered.
/// Returns list of `(seq, delivered_count)` for in-order deliveries only.
fn replay_into_data_path(dp: &mut DataPath<8, 16, Reno>, segments: &[TcpFrame]) -> Vec<(u32, u32)> {
    let seg_ctrl = TcpSeg {
        ack: true,
        ..TcpSeg::default()
    };
    let mut ranges = Vec::new();
    for frame in segments {
        let out = dp.on_segment(
            seg_ctrl,
            SeqNum(frame.seq),
            SeqNum(frame.ack),
            frame.wnd,
            frame.payload.len() as u32,
            Instant::ZERO,
        );
        if out.delivered > 0 {
            // rcv_nxt after delivery; the delivered block ends here.
            let rcv_nxt_after = dp.rcv_nxt().0;
            let delivery_start_seq = rcv_nxt_after.wrapping_sub(out.delivered);
            ranges.push((delivery_start_seq, out.delivered));
        }
    }
    ranges
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn in_order_delivery_matches_smoltcp_oracle() {
    let payload: Vec<u8> = (0u8..=255).cycle().take(PAYLOAD_LEN).collect();
    let (smoltcp_received, data_segs, irs, iss_syn) = run_smoltcp_sender(49152, &payload);

    // build DataPath with ISS/IRS from smoltcp's actual handshake.
    let peer_wnd = data_segs[0].wnd;
    let mut dp = make_data_path(SeqNum(iss_syn), SeqNum(irs), peer_wnd);

    let ranges = replay_into_data_path(&mut dp, &data_segs);
    let data_start = irs.wrapping_add(1);
    let our_bytes = bytes_from_ranges(&ranges, data_start, &payload);

    assert_eq!(
        our_bytes.len(),
        smoltcp_received.len(),
        "in-order: DataPath must deliver same byte count as smoltcp oracle"
    );
    assert_eq!(
        our_bytes, smoltcp_received,
        "in-order: DataPath must deliver byte-identical content to smoltcp oracle"
    );
}

#[test]
fn out_of_order_then_gap_fill_matches_smoltcp_oracle() {
    // payload must exceed the effective MSS (1460 on medium-ip) so smoltcp emits
    // at least two data segments — this gives us a concrete OOO scenario to swap.
    let payload: Vec<u8> = (0u8..=255).cycle().take(2920).collect();
    let (smoltcp_received, data_segs, irs, iss_syn) = run_smoltcp_sender(49153, &payload);

    assert!(
        data_segs.len() >= 2,
        "need at least 2 data segments for OOO; got {}",
        data_segs.len()
    );

    let peer_wnd = data_segs[0].wnd;
    let mut dp = make_data_path(SeqNum(iss_syn), SeqNum(irs), peer_wnd);

    // reverse the first two segments to create OOO; all other segments remain in order.
    let mut reordered = data_segs.clone();
    reordered.swap(0, 1);

    let ranges = replay_into_data_path(&mut dp, &reordered);
    let data_start = irs.wrapping_add(1);
    let our_bytes = bytes_from_ranges(&ranges, data_start, &payload);

    assert_eq!(
        our_bytes.len(),
        smoltcp_received.len(),
        "OOO: DataPath must deliver same byte count as smoltcp oracle after gap fill"
    );
    assert_eq!(
        our_bytes, smoltcp_received,
        "OOO: DataPath must deliver byte-identical content to smoltcp oracle after gap fill"
    );
}

#[test]
fn duplicate_segment_delivers_no_extra_bytes() {
    let payload: Vec<u8> = (0u8..=255).cycle().take(800).collect();
    let (smoltcp_received, data_segs, irs, iss_syn) = run_smoltcp_sender(49154, &payload);

    let peer_wnd = data_segs[0].wnd;
    let mut dp = make_data_path(SeqNum(iss_syn), SeqNum(irs), peer_wnd);
    let seg_ctrl = TcpSeg {
        ack: true,
        ..TcpSeg::default()
    };

    let mut total_delivered: u64 = 0;

    for frame in &data_segs {
        let out1 = dp.on_segment(
            seg_ctrl,
            SeqNum(frame.seq),
            SeqNum(frame.ack),
            frame.wnd,
            frame.payload.len() as u32,
            Instant::ZERO,
        );
        total_delivered += out1.delivered as u64;

        // replay the same segment a second time — must deliver zero new bytes.
        let out2 = dp.on_segment(
            seg_ctrl,
            SeqNum(frame.seq),
            SeqNum(frame.ack),
            frame.wnd,
            frame.payload.len() as u32,
            Instant::ZERO,
        );
        assert_eq!(
            out2.delivered, 0,
            "duplicate of seq={} must deliver 0 new bytes",
            frame.seq
        );
    }

    assert_eq!(
        total_delivered,
        smoltcp_received.len() as u64,
        "duplicate-replay DataPath must deliver the same total byte count as smoltcp oracle"
    );
}
