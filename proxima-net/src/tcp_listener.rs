//! Sans-IO multi-connection TCP echo listener: the userspace TCP/IP stack
//! driven by `proxima-tcp` over a backend packet path (dpdk or AF_XDP). Pure logic — segments in,
//! segments out, no I/O — so the handshake / echo / close behaviour is
//! unit-tested without a backend (the backend driver just serializes the out-segments
//! into frames and transmits them). Inbound segments are demultiplexed onto a
//! per-peer connection table keyed by the peer's `(ip, port)`.
//!
//! The handshake is run through the RFC 793 control FSM
//! (`proxima_protocols::tcp::connection`); once ESTABLISHED the data phase is the composed
//! `DataPath` (window + reassembly + retransmit + congestion + RTT). Close is
//! emitted as a direct FIN+ACK on the peer's FIN (correct on the wire; the full
//! close-state accounting lives in the control FSM, exercised by its own tests).
//!
//! Limits (tracked in docs/net-dpdk/discipline.md (shared with the AF_XDP backend)): the echo reflects in-order
//! delivered bytes (out-of-order payload buffering is a follow-up — irrelevant
//! on a lossless local tap); the table grows unbounded (no idle eviction yet).

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

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

type Path = DataPath<OOO_GAPS, RETX_CAP, Reno>;

/// A peer's link/network/transport address, captured from its SYN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoint {
    pub mac: [u8; 6],
    pub ip: [u8; 4],
    pub port: u16,
}

/// One parsed inbound TCP segment addressed to the listener.
#[derive(Debug, Clone, Copy)]
pub struct Inbound<'segment> {
    pub source_mac: [u8; 6],
    pub source_ip: [u8; 4],
    pub source_port: u16,
    pub flags: TcpFlags,
    pub seq: u32,
    pub ack: u32,
    pub window: u16,
    pub payload: &'segment [u8],
}

/// A TCP segment the caller must serialize and transmit to [`Response::peer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutSegment {
    pub flags: TcpFlags,
    pub seq: u32,
    pub ack: u32,
    pub window: u16,
    pub payload: Vec<u8>,
}

/// The segments produced by one inbound segment, plus the peer to send them to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub peer: Endpoint,
    pub segments: Vec<OutSegment>,
}

// per-peer entry in the connection table.
// P20 forbids Box; the table holds one entry per live peer, so the large
// `Established` variant (Conn owns the full DataPath) is an acceptable trade
// for staying heap-indirection-free.
#[allow(clippy::large_enum_variant)]
enum ConnState {
    Handshake { irs: u32, isn: u32 },
    Established(Conn),
    Closing,
}

// the listener demultiplexes on the peer half of the 4-tuple (our ip:port fixed).
type ConnKey = ([u8; 4], u16);

// space successive connections' initial sequence numbers well apart.
const ISN_STRIDE: u32 = 0x0004_0000;

#[derive(Clone, Copy)]
enum Disposition {
    Open,
    Closing,
    Reset,
}

struct Conn {
    path: Path,
    iss_plus_one: u32,
    bytes_sent: u32,
    send_buf: Vec<u8>,
    // received payload bytes keyed by absolute sequence, so an out-of-order
    // segment's data can be echoed once the gap before it is filled. byte-keyed
    // for obvious correctness; a perf pass would store ranges (no wraparound
    // handling — connections are short-lived).
    pending: BTreeMap<u32, u8>,
    echo_seq: u32,
}

/// Multi-connection TCP echo server. Demultiplexes inbound segments onto a
/// per-peer connection table; drive it with [`EchoListener::on_inbound`].
pub struct EchoListener {
    our_ip: [u8; 4],
    our_port: u16,
    next_isn: u32,
    conns: BTreeMap<ConnKey, (Endpoint, ConnState)>,
}

impl EchoListener {
    #[must_use]
    pub fn new(our_ip: [u8; 4], our_port: u16, our_isn: u32) -> Self {
        Self {
            our_ip,
            our_port,
            next_isn: our_isn,
            conns: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn our_ip(&self) -> [u8; 4] {
        self.our_ip
    }

    #[must_use]
    pub fn our_port(&self) -> u16 {
        self.our_port
    }

    /// Number of connections currently in the table (handshaking, established,
    /// or closing).
    #[must_use]
    pub fn open_connections(&self) -> usize {
        self.conns.len()
    }

    /// Process one inbound segment and return the segments to transmit (if any).
    pub fn on_inbound(&mut self, inbound: &Inbound, now: Instant) -> Option<Response> {
        let key: ConnKey = (inbound.source_ip, inbound.source_port);
        let peer = Endpoint {
            mac: inbound.source_mac,
            ip: inbound.source_ip,
            port: inbound.source_port,
        };

        if inbound.flags.rst {
            self.conns.remove(&key);
            return None;
        }

        // existing connection: advance it in place. the established data path
        // mutates the boxed Conn directly (no per-packet realloc); only the table
        // entry is dropped when the connection finishes.
        if let Some((stored_peer, state)) = self.conns.get_mut(&key) {
            let stored_peer = *stored_peer;
            let (response, drop_connection) = advance(stored_peer, state, inbound, now);
            if drop_connection {
                self.conns.remove(&key);
            }
            return response;
        }

        // no entry: only an opening SYN starts a connection.
        if is_initial_syn(inbound.flags) {
            let isn = self.next_isn;
            self.next_isn = self.next_isn.wrapping_add(ISN_STRIDE);
            self.conns.insert(
                key,
                (
                    peer,
                    ConnState::Handshake {
                        irs: inbound.seq,
                        isn,
                    },
                ),
            );
            return Some(Response {
                peer,
                segments: vec![synack_segment(isn, inbound.seq)],
            });
        }
        None
    }
}

// advance one connection by a single segment, mutating its state in place.
// Returns the reply and whether the connection should be dropped from the table.
fn advance(
    peer: Endpoint,
    state: &mut ConnState,
    inbound: &Inbound,
    now: Instant,
) -> (Option<Response>, bool) {
    match state {
        ConnState::Handshake { irs, isn } => {
            if !is_bare_ack(inbound.flags) {
                return (None, false);
            }
            let mut conn = Conn::new(*isn, *irs, inbound.window);
            // the handshake-completing ACK can piggyback data (or a FIN).
            let (segments, disposition) = if inbound.payload.is_empty() && !inbound.flags.fin {
                (Vec::new(), Disposition::Open)
            } else {
                conn.handle(inbound, now)
            };
            match disposition {
                Disposition::Open => *state = ConnState::Established(conn),
                Disposition::Closing => *state = ConnState::Closing,
                Disposition::Reset => return (wrap(peer, segments), true),
            }
            (wrap(peer, segments), false)
        }
        ConnState::Established(conn) => {
            let (segments, disposition) = conn.handle(inbound, now);
            match disposition {
                Disposition::Open => (wrap(peer, segments), false),
                Disposition::Closing => {
                    *state = ConnState::Closing;
                    (wrap(peer, segments), false)
                }
                Disposition::Reset => (wrap(peer, segments), true),
            }
        }
        // CLOSING: the peer's ACK of our FIN finishes the close.
        ConnState::Closing => (None, inbound.flags.ack),
    }
}

fn wrap(peer: Endpoint, segments: Vec<OutSegment>) -> Option<Response> {
    if segments.is_empty() {
        None
    } else {
        Some(Response { peer, segments })
    }
}

fn synack_segment(isn: u32, peer_seq: u32) -> OutSegment {
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

impl Conn {
    fn new(our_isn: u32, irs: u32, peer_window: u16) -> Self {
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
            path,
            iss_plus_one,
            bytes_sent: 0,
            send_buf: Vec::new(),
            pending: BTreeMap::new(),
            echo_seq: irs.wrapping_add(1),
        }
    }

    fn snd_nxt(&self) -> u32 {
        self.iss_plus_one.wrapping_add(self.bytes_sent)
    }

    fn handle(&mut self, inbound: &Inbound, now: Instant) -> (Vec<OutSegment>, Disposition) {
        let control = to_control(inbound.flags);
        let payload_len = u32::try_from(inbound.payload.len()).unwrap_or(u32::MAX);

        // buffer the payload by sequence first, so an out-of-order segment's
        // bytes are available to echo once the reassembler later delivers them.
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
            return (Vec::new(), Disposition::Reset);
        }

        let mut segments = Vec::new();
        if output.delivered > 0 {
            let delivered = self.take_delivered(output.delivered);
            self.echo(&delivered, now, &mut segments);
        }

        if inbound.flags.fin {
            segments.push(self.fin_ack());
            return (segments, Disposition::Closing);
        }
        if segments.is_empty() && output.ack_required {
            segments.push(self.bare_ack());
        }
        (segments, Disposition::Open)
    }

    // pull the next `count` in-order bytes (which the reassembler just delivered)
    // out of the pending buffer, in sequence order.
    fn take_delivered(&mut self, count: u32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count as usize);
        for _ in 0..count {
            if let Some(byte) = self.pending.remove(&self.echo_seq) {
                bytes.push(byte);
            }
            self.echo_seq = self.echo_seq.wrapping_add(1);
        }
        bytes
    }

    // echo the delivered bytes back to the peer, gated by DataPath.
    fn echo(&mut self, delivered: &[u8], now: Instant, segments: &mut Vec<OutSegment>) {
        let base = self.send_buf.len();
        self.send_buf.extend_from_slice(delivered);

        let mut cursor = base;
        while cursor < base + delivered.len() {
            let remaining = u32::try_from(base + delivered.len() - cursor).unwrap_or(u32::MAX);
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
                ack: self.path.rcv_nxt().0,
                window: OUR_WINDOW,
                payload: bytes,
            });
            self.bytes_sent = self.bytes_sent.wrapping_add(segment.len);
            cursor += len;
        }
    }

    fn fin_ack(&self) -> OutSegment {
        OutSegment {
            flags: TcpFlags {
                fin: true,
                ack: true,
                ..TcpFlags::default()
            },
            seq: self.snd_nxt(),
            // ack the peer's data and its FIN (which consumes one sequence number).
            ack: self.path.rcv_nxt().0.wrapping_add(1),
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
            ack: self.path.rcv_nxt().0,
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

// a connection-opening SYN (no ACK).
fn is_initial_syn(flags: TcpFlags) -> bool {
    flags.syn && !flags.ack
}

// the handshake-completing ACK (no SYN).
fn is_bare_ack(flags: TcpFlags) -> bool {
    flags.ack && !flags.syn
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
    const OUR_PORT: u16 = 7;
    const OUR_ISN: u32 = 0x1000;
    const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const PEER_IP: [u8; 4] = [10, 0, 0, 1];
    const PEER_PORT: u16 = 40000;
    const CLIENT_ISN: u32 = 0x5000;

    fn inbound(flags: TcpFlags, seq: u32, ack: u32, payload: &[u8]) -> Inbound<'_> {
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

    fn established() -> EchoListener {
        let mut listener = EchoListener::new(OUR_IP, OUR_PORT, OUR_ISN);
        // SYN -> SYN-ACK
        let response = listener
            .on_inbound(&inbound(syn(), CLIENT_ISN, 0, &[]), Instant::ZERO)
            .expect("syn yields syn-ack");
        let synack = &response.segments[0];
        assert!(synack.flags.syn && synack.flags.ack);
        assert_eq!(synack.seq, OUR_ISN);
        assert_eq!(synack.ack, CLIENT_ISN + 1);
        // final ACK -> established, no segment
        assert!(
            listener
                .on_inbound(
                    &inbound(ack(), CLIENT_ISN + 1, OUR_ISN + 1, &[]),
                    Instant::ZERO
                )
                .is_none()
        );
        listener
    }

    #[test]
    fn handshake_emits_synack_to_the_peer() {
        let mut listener = EchoListener::new(OUR_IP, OUR_PORT, OUR_ISN);
        let response = listener
            .on_inbound(&inbound(syn(), CLIENT_ISN, 0, &[]), Instant::ZERO)
            .expect("syn-ack");
        assert_eq!(
            response.peer,
            Endpoint {
                mac: PEER_MAC,
                ip: PEER_IP,
                port: PEER_PORT
            }
        );
        assert_eq!(response.segments.len(), 1);
    }

    #[test]
    fn established_echoes_data_with_advancing_ack() {
        let mut listener = established();
        let payload = b"hello\n";
        let response = listener
            .on_inbound(
                &inbound(ack(), CLIENT_ISN + 1, OUR_ISN + 1, payload),
                Instant::from_micros(1),
            )
            .expect("data is echoed");
        let echo = &response.segments[0];
        assert_eq!(echo.payload, payload, "echoed bytes match");
        assert!(echo.flags.ack && echo.flags.psh);
        assert_eq!(echo.seq, OUR_ISN + 1, "our data starts at iss+1");
        assert_eq!(
            echo.ack,
            CLIENT_ISN + 1 + payload.len() as u32,
            "acks the received data"
        );
    }

    #[test]
    fn out_of_order_segment_is_echoed_after_gap_fill() {
        let mut listener = established();
        // segment 2 arrives first (bytes "DEF" at IRS+1+3), out of order.
        let early = listener.on_inbound(
            &inbound(ack(), CLIENT_ISN + 1 + 3, OUR_ISN + 1, b"DEF"),
            Instant::from_micros(1),
        );
        // nothing deliverable yet, so nothing is echoed (a dup-ack may be sent,
        // but it carries no payload).
        if let Some(resp) = &early {
            assert!(resp.segments.iter().all(|s| s.payload.is_empty()));
        }
        // segment 1 fills the gap; both segments are now in order.
        let filled = listener
            .on_inbound(
                &inbound(ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"ABC"),
                Instant::from_micros(2),
            )
            .expect("gap fill echoes everything");
        let echoed: Vec<u8> = filled
            .segments
            .iter()
            .flat_map(|s| s.payload.clone())
            .collect();
        assert_eq!(
            echoed, b"ABCDEF",
            "the full in-order stream is echoed, not just the gap-filler"
        );
    }

    #[test]
    fn second_data_segment_advances_our_sequence() {
        let mut listener = established();
        let first = b"abc";
        listener
            .on_inbound(
                &inbound(ack(), CLIENT_ISN + 1, OUR_ISN + 1, first),
                Instant::from_micros(1),
            )
            .expect("first echo");
        let second = b"de";
        let response = listener
            .on_inbound(
                &inbound(ack(), CLIENT_ISN + 1 + 3, OUR_ISN + 1 + 3, second),
                Instant::from_micros(2),
            )
            .expect("second echo");
        let echo = &response.segments[0];
        assert_eq!(
            echo.seq,
            OUR_ISN + 1 + 3,
            "second segment seq follows the first"
        );
        assert_eq!(echo.payload, second);
    }

    #[test]
    fn fin_yields_fin_ack_and_returns_to_listen() {
        let mut listener = established();
        let response = listener
            .on_inbound(
                &inbound(
                    TcpFlags {
                        fin: true,
                        ack: true,
                        ..TcpFlags::default()
                    },
                    CLIENT_ISN + 1,
                    OUR_ISN + 1,
                    &[],
                ),
                Instant::from_micros(1),
            )
            .expect("fin yields fin-ack");
        let fin = response.segments.last().expect("a segment");
        assert!(fin.flags.fin && fin.flags.ack);
        assert_eq!(fin.ack, CLIENT_ISN + 2, "acks the peer's FIN sequence");
        // the peer's final ACK closes us back to LISTEN, ready for the next peer.
        assert!(
            listener
                .on_inbound(
                    &inbound(ack(), CLIENT_ISN + 2, OUR_ISN + 2, &[]),
                    Instant::from_micros(2)
                )
                .is_none()
        );
        let next_syn =
            listener.on_inbound(&inbound(syn(), 0x9000, 0, &[]), Instant::from_micros(3));
        assert!(
            next_syn.is_some(),
            "listener accepts a new connection after close"
        );
    }

    #[test]
    fn rst_drops_back_to_listen() {
        let mut listener = established();
        let response = listener.on_inbound(
            &inbound(
                TcpFlags {
                    rst: true,
                    ..TcpFlags::default()
                },
                CLIENT_ISN + 1,
                OUR_ISN + 1,
                &[],
            ),
            Instant::from_micros(1),
        );
        assert!(response.is_none());
        assert!(
            listener
                .on_inbound(&inbound(syn(), 0x9000, 0, &[]), Instant::from_micros(2))
                .is_some()
        );
    }

    fn inbound_port(port: u16, flags: TcpFlags, seq: u32, ack: u32, payload: &[u8]) -> Inbound<'_> {
        let mut octets = PEER_MAC;
        octets[5] = (port & 0xff) as u8;
        Inbound {
            source_mac: octets,
            source_ip: PEER_IP,
            source_port: port,
            flags,
            seq,
            ack,
            window: 64240,
            payload,
        }
    }

    #[test]
    fn two_interleaved_connections_echo_independently() {
        let mut listener = EchoListener::new(OUR_IP, OUR_PORT, OUR_ISN);
        let (port_a, isn_a) = (40001, 0x1_0000);
        let (port_b, isn_b) = (40002, 0x2_0000);

        // both peers open; each SYN-ACK acks that peer's own ISN.
        let sa = listener
            .on_inbound(&inbound_port(port_a, syn(), isn_a, 0, &[]), Instant::ZERO)
            .expect("synack a");
        let sb = listener
            .on_inbound(&inbound_port(port_b, syn(), isn_b, 0, &[]), Instant::ZERO)
            .expect("synack b");
        assert_eq!(sa.segments[0].ack, isn_a + 1);
        assert_eq!(sb.segments[0].ack, isn_b + 1);
        let our_isn_a = sa.segments[0].seq;
        let our_isn_b = sb.segments[0].seq;
        assert_ne!(
            our_isn_a, our_isn_b,
            "distinct connections get distinct ISNs"
        );

        // complete both handshakes.
        listener.on_inbound(
            &inbound_port(port_a, ack(), isn_a + 1, our_isn_a + 1, &[]),
            Instant::ZERO,
        );
        listener.on_inbound(
            &inbound_port(port_b, ack(), isn_b + 1, our_isn_b + 1, &[]),
            Instant::ZERO,
        );
        assert_eq!(listener.open_connections(), 2);

        // interleave data; each peer gets its own bytes echoed.
        let ra = listener
            .on_inbound(
                &inbound_port(port_a, ack(), isn_a + 1, our_isn_a + 1, b"aaa"),
                Instant::from_micros(1),
            )
            .expect("echo a");
        let rb = listener
            .on_inbound(
                &inbound_port(port_b, ack(), isn_b + 1, our_isn_b + 1, b"bbbb"),
                Instant::from_micros(2),
            )
            .expect("echo b");
        assert_eq!(ra.peer.port, port_a);
        assert_eq!(ra.segments[0].payload, b"aaa");
        assert_eq!(rb.peer.port, port_b);
        assert_eq!(rb.segments[0].payload, b"bbbb");

        // closing A leaves B established.
        listener.on_inbound(
            &inbound_port(
                port_a,
                TcpFlags {
                    fin: true,
                    ack: true,
                    ..TcpFlags::default()
                },
                isn_a + 4,
                our_isn_a + 1,
                &[],
            ),
            Instant::from_micros(3),
        );
        listener.on_inbound(
            &inbound_port(port_a, ack(), isn_a + 5, our_isn_a + 4, &[]),
            Instant::from_micros(4),
        );
        assert_eq!(listener.open_connections(), 1, "A closed, B still open");
    }
}
