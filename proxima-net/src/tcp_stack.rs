//! Sans-IO multi-connection TCP stack exposing byte streams (the generic form
//! of `tcp_listener`'s echo): segments in -> (accept events, readable bytes,
//! segments out). The app reads delivered bytes and writes bytes to send; the
//! stack runs the RFC 793 handshake + `DataPath` data phase + FIN/ACK close and
//! frames nothing itself (the backend driver serializes the `OutSegment`s). No
//! I/O — shared by every backend (dpdk, AF_XDP) via `proxima_net::tcp_stack`.
//!
//! A backend `StreamListener` drives it: `poll_accept` drains newly established
//! connections, `read`/`write` move bytes per connection, and `close` sends our
//! FIN once the app is done. Passive close is CLOSE_WAIT-correct: the peer's FIN
//! marks the read side at EOF and is acknowledged, but it never reaps the
//! connection while the app still has buffered RX bytes to drain — only once our
//! own FIN has been acked AND `recv` is empty does the connection leave the
//! table.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

use crate::tcp_listener::{Endpoint, Inbound, OutSegment};
use proxima_protocols::inet::tcp::TcpFlags;
use proxima_protocols::tcp::DataPath;
use proxima_protocols::tcp::congestion::Reno;
use proxima_protocols::tcp::connection::Segment as ControlBits;
use proxima_protocols::tcp::seq::SeqNum;
use proxima_protocols::tcp::time::Instant;

const OOO_GAPS: usize = 8;
const RETX_CAP: usize = 16;
const SMSS: u32 = 1460;
const OUR_WINDOW: u16 = 64240;
const ISN_STRIDE: u32 = 0x0004_0000;

type Path = DataPath<OOO_GAPS, RETX_CAP, Reno>;

/// The peer key a connection is addressed by: `(peer_ip, peer_port)`.
pub type ConnId = ([u8; 4], u16);

/// A segment to transmit, paired with the peer to send it to.
pub type Outbound = (Endpoint, OutSegment);

// P20 forbids Box; the connection table holds one entry per live peer, so the
// large `Open` variant is a fine size trade for staying heap-indirection-free.
#[allow(clippy::large_enum_variant)]
enum Entry {
    // passive open: we sent SYN-ACK, awaiting the peer's ACK.
    Handshake { peer: Endpoint, irs: u32, isn: u32 },
    // active open: we sent SYN, awaiting the peer's SYN-ACK.
    SynSent { peer: Endpoint, isn: u32 },
    Open(Conn),
}

// what an inbound segment did to a connection's lifecycle.
enum Lifecycle {
    None,
    Accepted,
    Connected,
    Drop,
}

struct Conn {
    peer: Endpoint,
    path: Path,
    iss_plus_one: u32,
    sent: u32,
    send_buf: Vec<u8>,
    recv: VecDeque<u8>,
    pending: BTreeMap<u32, u8>,
    deliver_seq: u32,
    peer_finished: bool,
    we_finished: bool,
    // the sequence number our own FIN occupies, set when the app calls close.
    // the connection is reaped only once the peer's ack covers this.
    our_fin_seq: Option<u32>,
    reset: bool,
}

/// A sans-IO TCP stack bound to one local `(ip, port)`.
pub struct TcpStack {
    our_ip: [u8; 4],
    our_port: u16,
    next_isn: u32,
    conns: BTreeMap<ConnId, Entry>,
    accepted: VecDeque<ConnId>,
    connected: VecDeque<ConnId>,
}

impl TcpStack {
    #[must_use]
    pub fn new(our_ip: [u8; 4], our_port: u16, our_isn: u32) -> Self {
        Self {
            our_ip,
            our_port,
            next_isn: our_isn,
            conns: BTreeMap::new(),
            accepted: VecDeque::new(),
            connected: VecDeque::new(),
        }
    }

    /// Actively open a connection to `peer`: registers a half-open entry and
    /// returns the SYN to transmit. The connection becomes usable once
    /// [`poll_connected`](Self::poll_connected) yields its id.
    pub fn connect(&mut self, peer: Endpoint) -> (ConnId, Vec<Outbound>) {
        let id: ConnId = (peer.ip, peer.port);
        let isn = self.next_isn;
        self.next_isn = self.next_isn.wrapping_add(ISN_STRIDE);
        self.conns.insert(id, Entry::SynSent { peer, isn });
        let syn = OutSegment {
            flags: TcpFlags {
                syn: true,
                ..TcpFlags::default()
            },
            seq: isn,
            ack: 0,
            window: OUR_WINDOW,
            payload: Vec::new(),
        };
        (id, vec![(peer, syn)])
    }

    /// Pop the next actively-opened connection whose handshake completed.
    pub fn poll_connected(&mut self) -> Option<ConnId> {
        self.connected.pop_front()
    }

    #[must_use]
    pub fn our_ip(&self) -> [u8; 4] {
        self.our_ip
    }

    #[must_use]
    pub fn our_port(&self) -> u16 {
        self.our_port
    }

    /// Pop the next connection that finished its handshake, if any.
    pub fn poll_accept(&mut self) -> Option<ConnId> {
        self.accepted.pop_front()
    }

    #[must_use]
    pub fn peer(&self, id: ConnId) -> Option<Endpoint> {
        match self.conns.get(&id)? {
            Entry::Handshake { peer, .. } | Entry::SynSent { peer, .. } => Some(*peer),
            Entry::Open(conn) => Some(conn.peer),
        }
    }

    /// True once the peer has sent FIN and all its data is drained — the app's
    /// read side is at EOF (also true once the connection has left the table).
    #[must_use]
    pub fn read_closed(&self, id: ConnId) -> bool {
        match self.conns.get(&id) {
            Some(Entry::Open(conn)) => conn.peer_finished && conn.recv.is_empty(),
            Some(Entry::Handshake { .. } | Entry::SynSent { .. }) => false,
            None => true,
        }
    }

    /// Drain up to `buf.len()` delivered bytes into `buf`; returns the count.
    pub fn read(&mut self, id: ConnId, buf: &mut [u8]) -> usize {
        let Some(Entry::Open(conn)) = self.conns.get_mut(&id) else {
            return 0;
        };
        let mut count = 0;
        while count < buf.len() {
            let Some(byte) = conn.recv.pop_front() else {
                break;
            };
            buf[count] = byte;
            count += 1;
        }
        count
    }

    /// Queue `data` for transmission and emit the segments it unblocks.
    pub fn write(&mut self, id: ConnId, data: &[u8], now: Instant) -> Vec<Outbound> {
        let Some(Entry::Open(conn)) = self.conns.get_mut(&id) else {
            return Vec::new();
        };
        conn.send_buf.extend_from_slice(data);
        let peer = conn.peer;
        conn.drain_send(now)
            .into_iter()
            .map(|segment| (peer, segment))
            .collect()
    }

    /// Close our write side: emit our FIN. The app calls this once it has
    /// finished reading and writing; the connection stays in the table (so a
    /// retransmitted peer ACK is still routed) until the peer acknowledges our
    /// FIN. Idempotent.
    pub fn close(&mut self, id: ConnId) -> Vec<Outbound> {
        let Some(Entry::Open(conn)) = self.conns.get_mut(&id) else {
            return Vec::new();
        };
        if conn.we_finished {
            return Vec::new();
        }
        conn.we_finished = true;
        conn.our_fin_seq = Some(conn.snd_nxt());
        let peer = conn.peer;
        vec![(peer, conn.fin())]
    }

    /// Process one inbound segment. Returns the segments to transmit; readable
    /// bytes and accept events are observed via `read` / `poll_accept`.
    pub fn on_inbound(&mut self, inbound: &Inbound, now: Instant) -> Vec<Outbound> {
        let id: ConnId = (inbound.source_ip, inbound.source_port);
        let peer = Endpoint {
            mac: inbound.source_mac,
            ip: inbound.source_ip,
            port: inbound.source_port,
        };

        if inbound.flags.rst {
            self.conns.remove(&id);
            return Vec::new();
        }

        if let Some(entry) = self.conns.get_mut(&id) {
            let (segments, lifecycle) = advance(entry, inbound, now);
            match lifecycle {
                Lifecycle::Accepted => self.accepted.push_back(id),
                Lifecycle::Connected => self.connected.push_back(id),
                Lifecycle::Drop => {
                    self.conns.remove(&id);
                }
                Lifecycle::None => {}
            }
            return segments
                .into_iter()
                .map(|segment| (peer, segment))
                .collect();
        }

        if is_initial_syn(inbound.flags) {
            let isn = self.next_isn;
            self.next_isn = self.next_isn.wrapping_add(ISN_STRIDE);
            self.conns.insert(
                id,
                Entry::Handshake {
                    peer,
                    irs: inbound.seq,
                    isn,
                },
            );
            return vec![(peer, synack(isn, inbound.seq))];
        }
        Vec::new()
    }
}

fn advance(entry: &mut Entry, inbound: &Inbound, now: Instant) -> (Vec<OutSegment>, Lifecycle) {
    match entry {
        Entry::Handshake { peer, irs, isn } => {
            if !is_bare_ack(inbound.flags) {
                return (Vec::new(), Lifecycle::None);
            }
            let mut conn = Conn::new(*peer, *isn, *irs, inbound.window);
            // the handshake-completing ACK can piggyback data (or a FIN).
            let segments = if inbound.payload.is_empty() && !inbound.flags.fin {
                Vec::new()
            } else {
                conn.on_segment(inbound, now)
            };
            *entry = Entry::Open(conn);
            (segments, Lifecycle::Accepted)
        }
        Entry::SynSent { peer, isn } => {
            // active open: the peer's SYN-ACK completes our handshake — ACK it and
            // hand the connection up via poll_connected.
            if !(inbound.flags.syn && inbound.flags.ack) {
                return (Vec::new(), Lifecycle::None);
            }
            let conn = Conn::new(*peer, *isn, inbound.seq, inbound.window);
            let ack = conn.bare_ack();
            *entry = Entry::Open(conn);
            (vec![ack], Lifecycle::Connected)
        }
        Entry::Open(conn) => {
            // a connection stays Open after the peer's FIN so the app can drain
            // its remaining bytes and observe EOF. It leaves the table on RST, or
            // once the peer ACKs the FIN *we* sent (via `close`) AND the app has
            // drained every buffered RX byte — never before.
            let segments = conn.on_segment(inbound, now);
            let acked_our_fin = conn.our_fin_seq.is_some_and(|fin_seq| {
                inbound.flags.ack && seq_geq(inbound.ack, fin_seq.wrapping_add(1))
            });
            let lifecycle = if conn.reset || (acked_our_fin && conn.recv.is_empty()) {
                Lifecycle::Drop
            } else {
                Lifecycle::None
            };
            (segments, lifecycle)
        }
    }
}

impl Conn {
    fn new(peer: Endpoint, our_isn: u32, irs: u32, peer_window: u16) -> Self {
        let iss_plus_one = our_isn.wrapping_add(1);
        let path = DataPath::established(
            SeqNum(iss_plus_one),
            SeqNum(irs.wrapping_add(1)),
            u32::from(peer_window),
            u32::from(OUR_WINDOW),
            SMSS,
            Reno::new(SMSS),
        );
        Self {
            peer,
            path,
            iss_plus_one,
            sent: 0,
            send_buf: Vec::new(),
            recv: VecDeque::new(),
            pending: BTreeMap::new(),
            deliver_seq: irs.wrapping_add(1),
            peer_finished: false,
            we_finished: false,
            our_fin_seq: None,
            reset: false,
        }
    }

    fn snd_nxt(&self) -> u32 {
        self.iss_plus_one.wrapping_add(self.sent)
    }

    // the ack we advertise: rcv_nxt, plus one for the peer's FIN once seen (a FIN
    // consumes one sequence number that DataPath's rcv_nxt does not count).
    fn rcv_ack(&self) -> u32 {
        self.path
            .rcv_nxt()
            .0
            .wrapping_add(u32::from(self.peer_finished))
    }

    // process one inbound segment: deliver in-order bytes to `recv`, ack the
    // peer's data/FIN, and emit any queued app data. Does NOT reflexively close
    // on the peer's FIN — the app closes when it is done (see `TcpStack::close`).
    fn on_segment(&mut self, inbound: &Inbound, now: Instant) -> Vec<OutSegment> {
        let control = to_control(inbound.flags);
        let payload_len = u32::try_from(inbound.payload.len()).unwrap_or(u32::MAX);
        for (index, byte) in inbound.payload.iter().enumerate() {
            let seq = inbound.seq.wrapping_add(u32::try_from(index).unwrap_or(0));
            self.pending.insert(seq, *byte);
        }
        let output = self.path.on_segment(
            control,
            SeqNum(inbound.seq),
            SeqNum(inbound.ack),
            u32::from(inbound.window),
            payload_len,
            now,
        );
        if output.connection_reset {
            self.reset = true;
            return Vec::new();
        }
        for _ in 0..output.delivered {
            if let Some(byte) = self.pending.remove(&self.deliver_seq) {
                self.recv.push_back(byte);
            }
            self.deliver_seq = self.deliver_seq.wrapping_add(1);
        }

        // set peer_finished before draining so every ack we emit here covers the
        // FIN's sequence number.
        if inbound.flags.fin {
            self.peer_finished = true;
        }
        let mut segments = self.drain_send(now);
        if inbound.flags.fin {
            // acknowledge the peer's FIN; keep the connection Open+readable.
            if segments.is_empty() {
                segments.push(self.bare_ack());
            }
        } else if segments.is_empty() && output.ack_required {
            segments.push(self.bare_ack());
        }
        segments
    }

    // pull queued app bytes through DataPath into data segments.
    fn drain_send(&mut self, now: Instant) -> Vec<OutSegment> {
        let mut segments = Vec::new();
        // sent bytes sit at the front of send_buf (never trimmed), so the next
        // unsent byte is at offset `sent`.
        let mut cursor = usize::try_from(self.sent).unwrap_or(usize::MAX);
        while cursor < self.send_buf.len() {
            let remaining = u32::try_from(self.send_buf.len() - cursor).unwrap_or(u32::MAX);
            let offset = u32::try_from(cursor).unwrap_or(u32::MAX);
            let Some(segment) = self.path.try_send(remaining, offset, now) else {
                break;
            };
            let len = segment.len as usize;
            let bytes = self.send_buf[cursor..cursor + len].to_vec();
            segments.push(OutSegment {
                flags: TcpFlags {
                    psh: true,
                    ack: true,
                    ..TcpFlags::default()
                },
                seq: segment.seq.0,
                ack: self.rcv_ack(),
                window: OUR_WINDOW,
                payload: bytes,
            });
            self.sent = self.sent.wrapping_add(segment.len);
            cursor += len;
        }
        segments
    }

    fn fin(&self) -> OutSegment {
        OutSegment {
            flags: TcpFlags {
                fin: true,
                ack: true,
                ..TcpFlags::default()
            },
            seq: self.snd_nxt(),
            ack: self.rcv_ack(),
            window: OUR_WINDOW,
            payload: Vec::new(),
        }
    }

    fn bare_ack(&self) -> OutSegment {
        OutSegment {
            flags: TcpFlags {
                ack: true,
                ..TcpFlags::default()
            },
            seq: self.snd_nxt(),
            ack: self.rcv_ack(),
            window: OUR_WINDOW,
            payload: Vec::new(),
        }
    }
}

fn to_control(flags: TcpFlags) -> ControlBits {
    ControlBits {
        syn: flags.syn,
        ack: flags.ack,
        fin: flags.fin,
        rst: flags.rst,
    }
}

fn is_initial_syn(flags: TcpFlags) -> bool {
    flags.syn && !flags.ack
}

fn is_bare_ack(flags: TcpFlags) -> bool {
    flags.ack && !flags.syn
}

// wrapping sequence comparison: is `a` at or beyond `b` within the 2^31 window.
fn seq_geq(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < 0x8000_0000
}

fn synack(isn: u32, peer_seq: u32) -> OutSegment {
    OutSegment {
        flags: TcpFlags {
            syn: true,
            ack: true,
            ..TcpFlags::default()
        },
        seq: isn,
        ack: peer_seq.wrapping_add(1),
        window: OUR_WINDOW,
        payload: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::cast_possible_truncation
    )]
    use super::*;

    const OUR_IP: [u8; 4] = [10, 0, 0, 2];
    const OUR_PORT: u16 = 80;
    const OUR_ISN: u32 = 0x1000;
    const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const PEER_IP: [u8; 4] = [10, 0, 0, 1];
    const PEER_PORT: u16 = 50000;
    const CLIENT_ISN: u32 = 0x5000;
    const ID: ConnId = (PEER_IP, PEER_PORT);

    fn seg(flags: TcpFlags, seq: u32, ack: u32, payload: &[u8]) -> Inbound<'_> {
        Inbound {
            source_mac: PEER_MAC,
            source_ip: PEER_IP,
            source_port: PEER_PORT,
            flags,
            seq,
            ack,
            window: 64240,
            payload,
        }
    }
    fn syn() -> TcpFlags {
        TcpFlags {
            syn: true,
            ..TcpFlags::default()
        }
    }
    fn ack() -> TcpFlags {
        TcpFlags {
            ack: true,
            ..TcpFlags::default()
        }
    }
    fn fin_ack() -> TcpFlags {
        TcpFlags {
            fin: true,
            ack: true,
            ..TcpFlags::default()
        }
    }

    fn established() -> TcpStack {
        let mut stack = TcpStack::new(OUR_IP, OUR_PORT, OUR_ISN);
        let out = stack.on_inbound(&seg(syn(), CLIENT_ISN, 0, &[]), Instant::ZERO);
        assert!(out[0].1.flags.syn && out[0].1.flags.ack);
        assert_eq!(out[0].1.seq, OUR_ISN);
        // before the final ACK there is nothing to accept.
        assert!(stack.poll_accept().is_none());
        stack.on_inbound(&seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1, &[]), Instant::ZERO);
        stack
    }

    #[test]
    fn handshake_then_accept_yields_the_connection() {
        let mut stack = established();
        assert_eq!(
            stack.poll_accept(),
            Some(ID),
            "established connection is acceptable"
        );
        assert!(stack.poll_accept().is_none(), "only once");
        assert_eq!(stack.peer(ID).map(|p| p.port), Some(PEER_PORT));
    }

    #[test]
    fn inbound_data_becomes_readable_bytes() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let out = stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"GET /\n"),
            Instant::from_micros(1),
        );
        // an ack is emitted; the bytes are now readable by the app.
        assert!(out.iter().any(|(_, s)| s.flags.ack));
        let mut buf = [0u8; 16];
        let n = stack.read(ID, &mut buf);
        assert_eq!(&buf[..n], b"GET /\n");
        assert_eq!(stack.read(ID, &mut buf), 0, "drained");
    }

    #[test]
    fn app_write_emits_a_data_segment() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let out = stack.write(ID, b"HTTP/1.1 200\n", Instant::from_micros(1));
        let data = out
            .iter()
            .find(|(_, s)| !s.payload.is_empty())
            .expect("a data segment");
        assert_eq!(data.1.payload, b"HTTP/1.1 200\n");
        assert_eq!(data.1.seq, OUR_ISN + 1, "first app byte at iss+1");
        assert!(data.1.flags.psh && data.1.flags.ack);
    }

    #[test]
    fn out_of_order_data_reads_back_in_order() {
        let mut stack = established();
        let _ = stack.poll_accept();
        // "DEF" at +3 arrives before "ABC" at +0.
        stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1 + 3, OUR_ISN + 1, b"DEF"),
            Instant::from_micros(1),
        );
        stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"ABC"),
            Instant::from_micros(2),
        );
        let mut buf = [0u8; 16];
        let n = stack.read(ID, &mut buf);
        assert_eq!(&buf[..n], b"ABCDEF");
    }

    #[test]
    fn active_open_connects_then_streams() {
        let mut stack = TcpStack::new(OUR_IP, OUR_PORT, OUR_ISN);
        let peer = Endpoint {
            mac: PEER_MAC,
            ip: PEER_IP,
            port: PEER_PORT,
        };
        let (id, out) = stack.connect(peer);
        // we emit a bare SYN with our ISN.
        assert!(out[0].1.flags.syn && !out[0].1.flags.ack);
        assert_eq!(out[0].1.seq, OUR_ISN);
        assert!(
            stack.poll_connected().is_none(),
            "not connected until the SYN-ACK"
        );

        // peer's SYN-ACK: seq=peer_isn, ack=our_isn+1.
        let synack = TcpFlags {
            syn: true,
            ack: true,
            ..TcpFlags::default()
        };
        let reply = stack.on_inbound(&seg(synack, CLIENT_ISN, OUR_ISN + 1, &[]), Instant::ZERO);
        assert!(
            reply[0].1.flags.ack && !reply[0].1.flags.syn,
            "we ACK the SYN-ACK"
        );
        assert_eq!(reply[0].1.ack, CLIENT_ISN + 1);
        assert_eq!(stack.poll_connected(), Some(id), "connection is now usable");

        // and it streams: write emits data, inbound data is readable.
        let data = stack.write(id, b"ping", Instant::from_micros(1));
        assert_eq!(
            data.iter()
                .find(|(_, s)| !s.payload.is_empty())
                .map(|(_, s)| s.payload.clone()),
            Some(b"ping".to_vec())
        );
        stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1 + 4, b"pong"),
            Instant::from_micros(2),
        );
        let mut buf = [0u8; 8];
        let read = stack.read(id, &mut buf);
        assert_eq!(&buf[..read], b"pong");
    }

    #[test]
    fn peer_fin_acks_without_reflexive_fin_and_keeps_data_readable() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let out = stack.on_inbound(
            &seg(fin_ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"bye"),
            Instant::from_micros(1),
        );
        // we ACK the peer's data+FIN, but must NOT reflexively send our own FIN
        // (the app has not closed yet).
        assert!(out.iter().any(|(_, s)| s.flags.ack), "we ack the fin");
        assert!(
            !out.iter().any(|(_, s)| s.flags.fin),
            "no reflexive FIN before the app closes"
        );
        assert_eq!(
            out.iter().find(|(_, s)| s.flags.ack).map(|(_, s)| s.ack),
            Some(CLIENT_ISN + 1 + 3 + 1),
            "the ack covers the data and the FIN's sequence number"
        );
        let mut buf = [0u8; 8];
        assert_eq!(stack.read(ID, &mut buf), 3, "final data still readable");
        assert!(
            stack.read_closed(ID),
            "EOF once data drained and peer FINned"
        );
    }

    #[test]
    fn half_close_delivers_buffered_data_before_our_close() {
        let mut stack = established();
        let _ = stack.poll_accept();
        // the peer sends data AND FIN in one burst (immediate half-close), exactly
        // what `send(data); SHUT_WR` produces on a fast local path.
        let out = stack.on_inbound(
            &seg(fin_ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"payload"),
            Instant::from_micros(1),
        );
        assert!(
            !out.iter().any(|(_, s)| s.flags.fin),
            "peer FIN must not reap the connection before the app drains recv"
        );
        // the buffered payload is delivered to the app AFTER the peer half-closed.
        let mut buf = [0u8; 16];
        let read = stack.read(ID, &mut buf);
        assert_eq!(
            &buf[..read],
            b"payload",
            "buffered bytes are readable across the peer's half-close"
        );
        assert!(stack.read_closed(ID), "EOF once drained");
        assert!(
            stack.peer(ID).is_some(),
            "connection stays Open across the app's read window"
        );

        // the app closes -> we emit our FIN.
        let closing = stack.close(ID);
        let fin = &closing[0].1;
        assert!(fin.flags.fin && fin.flags.ack, "close emits our fin+ack");
        let fin_seq = fin.seq;

        // the peer ACKs our FIN -> only now is the connection reaped.
        stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1 + 7 + 1, fin_seq.wrapping_add(1), &[]),
            Instant::from_micros(2),
        );
        assert!(
            stack.peer(ID).is_none(),
            "reaped only after our FIN is acked and recv is drained"
        );
    }
}
