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

use crate::tcp_listener::{
    Endpoint, ISN_STRIDE, Inbound, OUR_WINDOW, OutSegment, Path, SMSS, is_bare_ack, is_initial_syn,
    synack_segment, to_control,
};
use proxima_protocols::inet::tcp::TcpFlags;
use proxima_protocols::tcp::DataPath;
use proxima_protocols::tcp::congestion::Reno;
use proxima_protocols::tcp::seq::SeqNum;
use proxima_protocols::tcp::time::Instant;

/// The 4-tuple key a connection is addressed by: `(local_port, peer_ip, peer_port)`;
/// our ip is the stack's single address so it is not part of the key.
pub type ConnId = (u16, [u8; 4], u16);

const EPHEMERAL_FIRST: u16 = 49152;
const EPHEMERAL_COUNT: usize = 16384;
const EPHEMERAL_WORDS: usize = EPHEMERAL_COUNT / 64;

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
    #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
    active_open: bool,
    path: Path,
    iss_plus_one: u32,
    sent: u32,
    send_buf: Vec<u8>,
    // Bounded to `OUR_WINDOW` (OverflowPolicy::Drop, RFC 793 §3.3 — a
    // receiver is never obligated to buffer bytes beyond its own advertised
    // window; the sender already sees that window in every segment we
    // emit, so a delivered byte landing here at capacity means the peer
    // sent outside the window we told it, a protocol violation on ITS
    // side). See `recv_window_drops` for the pressure observable.
    recv: VecDeque<u8>,
    held: Vec<(u32, Vec<u8>)>,
    trimmed: u32,
    deliver_seq: u32,
    peer_finished: bool,
    we_finished: bool,
    // the sequence number our own FIN occupies, set when the app calls close.
    // the connection is reaped only once the peer's ack covers this.
    our_fin_seq: Option<u32>,
    reset: bool,
    // pressure observable for `recv`'s OverflowPolicy::Drop.
    recv_window_drops: u32,
}

/// Default bound on `TcpStack::accepted`/`TcpStack::connected` — one entry
/// per connection whose handshake just completed, pending `poll_accept`/
/// `poll_connected`. This is a sans-IO, no_std/alloc-only crate with no
/// config-loading machinery (the backend driver owns that layer), so this
/// is the same documented-const-as-tunable shape `proxima-quic`'s
/// `DEFAULT_ACCEPT_CHANNEL_CAPACITY` uses. Generously above one poll cycle's
/// worth of handshake completions; override via
/// [`TcpStack::with_lifecycle_event_cap`].
pub const DEFAULT_LIFECYCLE_EVENT_CAP: usize = 1024;

/// A sans-IO TCP stack bound to one local `(ip, port)`.
pub struct TcpStack {
    our_ip: [u8; 4],
    our_port: u16,
    next_isn: u32,
    conns: BTreeMap<ConnId, Entry>,
    ephemeral_next: u32,
    ephemeral_free: VecDeque<u16>,
    ephemeral_used: [u64; EPHEMERAL_WORDS],
    // Bounded (OverflowPolicy::Drop — see `lifecycle_event_cap`'s doc): a
    // synchronous sans-IO state machine has no async surface to block on,
    // so the RFC 793 accept-backlog-overflow precedent applies (a listen
    // backlog that stops draining sheds new completions rather than
    // growing without bound). See `dropped_accepted`/`dropped_connected`
    // for the pressure observable.
    accepted: VecDeque<ConnId>,
    connected: VecDeque<ConnId>,
    lifecycle_event_cap: usize,
    dropped_accepted: u32,
    dropped_connected: u32,
}

impl TcpStack {
    #[must_use]
    pub fn new(our_ip: [u8; 4], our_port: u16, our_isn: u32) -> Self {
        Self::with_lifecycle_event_cap(our_ip, our_port, our_isn, DEFAULT_LIFECYCLE_EVENT_CAP)
    }

    /// [`Self::new`] with an explicit bound on `accepted`/`connected`
    /// instead of [`DEFAULT_LIFECYCLE_EVENT_CAP`].
    #[must_use]
    pub fn with_lifecycle_event_cap(
        our_ip: [u8; 4],
        our_port: u16,
        our_isn: u32,
        lifecycle_event_cap: usize,
    ) -> Self {
        Self {
            our_ip,
            our_port,
            next_isn: our_isn,
            conns: BTreeMap::new(),
            ephemeral_next: u32::from(EPHEMERAL_FIRST),
            ephemeral_free: VecDeque::new(),
            ephemeral_used: [0; EPHEMERAL_WORDS],
            accepted: VecDeque::new(),
            connected: VecDeque::new(),
            lifecycle_event_cap,
            dropped_accepted: 0,
            dropped_connected: 0,
        }
    }

    /// Actively open a connection to `peer`: registers a half-open entry and
    /// returns the SYN to transmit from a freshly allocated ephemeral local
    /// port (the first element of the id). The connection becomes usable once
    /// [`poll_connected`](Self::poll_connected) yields its id. `None` when all
    /// ephemeral ports are held by live connections.
    pub fn connect(&mut self, peer: Endpoint) -> Option<(ConnId, Vec<Outbound>)> {
        let local_port = self.allocate_port()?;
        let id: ConnId = (local_port, peer.ip, peer.port);
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
        Some((id, vec![(peer, syn)]))
    }

    fn allocate_port(&mut self) -> Option<u16> {
        loop {
            let port = match u16::try_from(self.ephemeral_next) {
                Ok(fresh) => {
                    self.ephemeral_next += 1;
                    fresh
                }
                Err(_) => self.ephemeral_free.pop_front()?,
            };
            if port != self.our_port {
                self.ephemeral_used[ephemeral_word(port)] |= ephemeral_bit(port);
                return Some(port);
            }
        }
    }

    fn release_port(&mut self, port: u16) {
        if port < EPHEMERAL_FIRST {
            return;
        }
        let word = &mut self.ephemeral_used[ephemeral_word(port)];
        if *word & ephemeral_bit(port) != 0 {
            *word &= !ephemeral_bit(port);
            self.ephemeral_free.push_back(port);
        }
    }

    fn remove_conn(&mut self, id: ConnId) {
        if self.conns.remove(&id).is_some() {
            self.release_port(id.0);
        }
    }

    /// True for the listener port and every ephemeral port a live active open holds.
    #[must_use]
    pub fn owns_port(&self, port: u16) -> bool {
        port == self.our_port
            || (port >= EPHEMERAL_FIRST
                && self.ephemeral_used[ephemeral_word(port)] & ephemeral_bit(port) != 0)
    }

    /// Pop the next actively-opened connection whose handshake completed.
    pub fn poll_connected(&mut self) -> Option<ConnId> {
        self.connected.pop_front()
    }

    #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
    pub(crate) fn active_connect_sequence(&self, id: ConnId) -> Option<u32> {
        match self.conns.get(&id) {
            Some(Entry::SynSent { isn, .. }) => Some(*isn),
            _ => self.active_connection_sequence(id),
        }
    }

    #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
    pub(crate) fn active_connection_sequence(&self, id: ConnId) -> Option<u32> {
        match self.conns.get(&id) {
            Some(Entry::Open(connection)) if connection.active_open => {
                Some(connection.iss_plus_one.wrapping_sub(1))
            }
            _ => None,
        }
    }

    #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
    pub(crate) fn cancel_connect(&mut self, id: ConnId, sequence: u32) {
        let owned = match self.conns.get(&id) {
            Some(Entry::SynSent { isn, .. }) => *isn == sequence,
            Some(Entry::Open(connection)) => {
                connection.active_open && connection.iss_plus_one == sequence.wrapping_add(1)
            }
            Some(Entry::Handshake { .. }) => false,
            None => true,
        };
        if owned {
            self.remove_conn(id);
            self.connected.retain(|connected| *connected != id);
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

    #[cfg(feature = "frame-stack")]
    pub(crate) fn has_pending_accept(&self) -> bool {
        !self.accepted.is_empty()
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

    /// Count of accept notifications dropped because `accepted` was at
    /// [`Self::with_lifecycle_event_cap`]'s bound — the pressure observable
    /// for that queue's `OverflowPolicy::Drop`.
    #[must_use]
    pub fn dropped_accepted(&self) -> u32 {
        self.dropped_accepted
    }

    /// Count of connect notifications dropped because `connected` was at
    /// [`Self::with_lifecycle_event_cap`]'s bound — the pressure observable
    /// for that queue's `OverflowPolicy::Drop`.
    #[must_use]
    pub fn dropped_connected(&self) -> u32 {
        self.dropped_connected
    }

    /// Count of delivered bytes dropped for connection `id` because `recv`
    /// was at the advertised window — the pressure observable for that
    /// connection's `OverflowPolicy::Drop` (see `Conn::recv`'s doc). `None`
    /// if `id` names no open connection.
    #[must_use]
    pub fn recv_window_drops(&self, id: ConnId) -> Option<u32> {
        match self.conns.get(&id)? {
            Entry::Open(conn) => Some(conn.recv_window_drops),
            Entry::Handshake { .. } | Entry::SynSent { .. } => None,
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
        let count = buf.len().min(conn.recv.len());
        let (front, back) = conn.recv.as_slices();
        let from_front = count.min(front.len());
        buf[..from_front].copy_from_slice(&front[..from_front]);
        buf[from_front..count].copy_from_slice(&back[..count - from_front]);
        conn.recv.drain(..count);
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
        self.on_inbound_to(self.our_port, inbound, now)
    }

    /// [`on_inbound`](Self::on_inbound) for a segment addressed to `local_port`:
    /// the listener port, or the ephemeral port of an active open.
    pub fn on_inbound_to(
        &mut self,
        local_port: u16,
        inbound: &Inbound,
        now: Instant,
    ) -> Vec<Outbound> {
        let id: ConnId = (local_port, inbound.source_ip, inbound.source_port);
        let peer = Endpoint {
            mac: inbound.source_mac,
            ip: inbound.source_ip,
            port: inbound.source_port,
        };

        if inbound.flags.rst {
            self.remove_conn(id);
            return Vec::new();
        }

        if let Some(entry) = self.conns.get_mut(&id) {
            let (segments, lifecycle) = advance(entry, inbound, now);
            match lifecycle {
                Lifecycle::Accepted => {
                    if self.accepted.len() < self.lifecycle_event_cap {
                        self.accepted.push_back(id);
                    } else {
                        self.dropped_accepted += 1;
                    }
                }
                Lifecycle::Connected => {
                    if self.connected.len() < self.lifecycle_event_cap {
                        self.connected.push_back(id);
                    } else {
                        self.dropped_connected += 1;
                    }
                }
                Lifecycle::Drop => self.remove_conn(id),
                Lifecycle::None => {}
            }
            return segments
                .into_iter()
                .map(|segment| (peer, segment))
                .collect();
        }

        if local_port == self.our_port && is_initial_syn(inbound.flags) {
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
            return vec![(peer, synack_segment(isn, inbound.seq))];
        }
        Vec::new()
    }
}

fn ephemeral_word(port: u16) -> usize {
    usize::from(port - EPHEMERAL_FIRST) / 64
}

fn ephemeral_bit(port: u16) -> u64 {
    1 << (usize::from(port - EPHEMERAL_FIRST) % 64)
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
            let conn = Conn {
                #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
                active_open: true,
                ..Conn::new(*peer, *isn, inbound.seq, inbound.window)
            };
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
            #[cfg(all(feature = "std", any(feature = "frame-stack", feature = "dpdk", feature = "xdp")))]
            active_open: false,
            path,
            iss_plus_one,
            sent: 0,
            send_buf: Vec::new(),
            recv: VecDeque::new(),
            held: Vec::new(),
            trimmed: 0,
            deliver_seq: irs.wrapping_add(1),
            peer_finished: false,
            we_finished: false,
            our_fin_seq: None,
            reset: false,
            recv_window_drops: 0,
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
        self.hold_ahead(inbound);
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
        self.deliver(inbound, output.delivered);

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

    // store an out-of-order payload whole, only while it sits ahead of
    // deliver_seq and inside the window we advertised; same-seq retransmit replaces.
    fn hold_ahead(&mut self, inbound: &Inbound) {
        let offset = inbound.seq.wrapping_sub(self.deliver_seq);
        let length = u32::try_from(inbound.payload.len()).unwrap_or(u32::MAX);
        let outside_window = offset.saturating_add(length) > u32::from(OUR_WINDOW);
        if length == 0 || offset == 0 || offset >= SEQ_HALF || outside_window {
            return;
        }
        match self.held.iter_mut().find(|(seq, _)| *seq == inbound.seq) {
            Some((_, bytes)) => {
                bytes.clear();
                bytes.extend_from_slice(inbound.payload);
            }
            None => self.held.push((inbound.seq, inbound.payload.to_vec())),
        }
    }

    // move `delivered` in-order bytes into recv: the inbound slice past any
    // prefix behind deliver_seq first, then held segments that now connect.
    fn deliver(&mut self, inbound: &Inbound, delivered: u32) {
        let mut remaining = usize::try_from(delivered).unwrap_or(usize::MAX);
        let behind = usize::try_from(self.deliver_seq.wrapping_sub(inbound.seq)).unwrap_or(usize::MAX);
        let fresh = inbound.payload.get(behind..).unwrap_or(&[]);
        let take = fresh.len().min(remaining);
        Self::push_capped(&mut self.recv, &mut self.recv_window_drops, &fresh[..take]);
        self.advance_deliver(take);
        remaining -= take;
        while remaining > 0 {
            remaining -= self.pull_held(remaining);
        }
        let deliver_seq = self.deliver_seq;
        self.held.retain(|(start, bytes)| {
            let length = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            let end_offset = start.wrapping_add(length).wrapping_sub(deliver_seq);
            end_offset != 0 && end_offset < SEQ_HALF
        });
    }

    // consume up to `limit` bytes from the held segment covering deliver_seq;
    // a byte no held segment covers was never stored (outside our window) and is
    // skipped one at a time, as the sequence space still advanced.
    fn pull_held(&mut self, limit: usize) -> usize {
        let deliver_seq = self.deliver_seq;
        let covering = self
            .held
            .iter()
            .enumerate()
            .find_map(|(index, (start, bytes))| {
                skip_into(*start, deliver_seq)
                    .filter(|skip| *skip < bytes.len())
                    .map(|skip| (index, skip))
            });
        let Some((index, skip)) = covering else {
            self.advance_deliver(1);
            return 1;
        };
        let bytes = &self.held[index].1;
        let end = bytes.len().min(skip.saturating_add(limit));
        Self::push_capped(&mut self.recv, &mut self.recv_window_drops, &bytes[skip..end]);
        self.advance_deliver(end - skip);
        end - skip
    }

    fn advance_deliver(&mut self, count: usize) {
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        self.deliver_seq = self.deliver_seq.wrapping_add(count);
    }

    // RFC 793 §3.3: a peer sending beyond the window WE advertised (in every
    // segment we emit) is the violation, not us; drop-and-count rather than
    // grow `recv` unbounded.
    fn push_capped(recv: &mut VecDeque<u8>, drops: &mut u32, bytes: &[u8]) {
        let room = usize::from(OUR_WINDOW).saturating_sub(recv.len());
        let take = room.min(bytes.len());
        recv.extend(&bytes[..take]);
        let dropped = u32::try_from(bytes.len() - take).unwrap_or(u32::MAX);
        *drops = drops.saturating_add(dropped);
    }

    // pull queued app bytes through DataPath into data segments.
    fn drain_send(&mut self, now: Instant) -> Vec<OutSegment> {
        let mut segments = Vec::new();
        self.release_acked();
        // sent-but-unacked bytes sit at the front of send_buf, so the next
        // unsent byte is at offset `sent - trimmed`.
        let mut cursor = usize::try_from(self.sent.wrapping_sub(self.trimmed)).unwrap_or(usize::MAX);
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

    // drop the acknowledged prefix of send_buf; an ack covering our FIN runs
    // one past the data, so the release is clamped to the bytes actually sent.
    fn release_acked(&mut self) {
        let acked = self.path.snd_una().0.wrapping_sub(self.iss_plus_one);
        let releasable = acked.wrapping_sub(self.trimmed);
        let in_buffer_sent = self.sent.wrapping_sub(self.trimmed);
        if releasable >= SEQ_HALF {
            return;
        }
        let release = releasable.min(in_buffer_sent);
        let count = usize::try_from(release).unwrap_or(0).min(self.send_buf.len());
        self.send_buf.drain(..count);
        self.trimmed = self.trimmed.wrapping_add(release);
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

const SEQ_HALF: u32 = 0x8000_0000;

// wrapping sequence comparison: is `a` at or beyond `b` within the 2^31 window.
fn seq_geq(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < SEQ_HALF
}

// how far `deliver_seq` sits past `start`, or None when `start` is ahead of it.
fn skip_into(start: u32, deliver_seq: u32) -> Option<usize> {
    let skip = deliver_seq.wrapping_sub(start);
    if skip < SEQ_HALF {
        usize::try_from(skip).ok()
    } else {
        None
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
    const ID: ConnId = (OUR_PORT, PEER_IP, PEER_PORT);

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
        established_with(CLIENT_ISN, OUR_ISN)
    }

    fn established_with(client_isn: u32, our_isn: u32) -> TcpStack {
        let mut stack = TcpStack::new(OUR_IP, OUR_PORT, our_isn);
        let out = stack.on_inbound(&seg(syn(), client_isn, 0, &[]), Instant::ZERO);
        assert!(out[0].1.flags.syn && out[0].1.flags.ack);
        assert_eq!(out[0].1.seq, our_isn);
        // before the final ACK there is nothing to accept.
        assert!(stack.poll_accept().is_none());
        stack.on_inbound(
            &seg(
                ack(),
                client_isn.wrapping_add(1),
                our_isn.wrapping_add(1),
                &[],
            ),
            Instant::ZERO,
        );
        stack
    }

    fn stream(length: usize) -> Vec<u8> {
        (0..length).map(|index| (index % 251) as u8).collect()
    }

    fn read_all(stack: &mut TcpStack) -> Vec<u8> {
        let mut collected = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let count = stack.read(ID, &mut buf);
            if count == 0 {
                return collected;
            }
            collected.extend_from_slice(&buf[..count]);
        }
    }

    fn send_data(stack: &mut TcpStack, client_isn: u32, our_isn: u32, offset: usize, bytes: &[u8], tick: u64) {
        let seq = client_isn.wrapping_add(1).wrapping_add(offset as u32);
        stack.on_inbound(
            &seg(ack(), seq, our_isn.wrapping_add(1), bytes),
            Instant::from_micros(tick),
        );
    }

    fn open_conn(stack: &TcpStack) -> &Conn {
        let Some(Entry::Open(conn)) = stack.conns.get(&ID) else {
            panic!("established connection must be Entry::Open");
        };
        conn
    }

    #[test]
    fn later_segments_first_then_the_gap_fill_read_back_in_order() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let bytes = stream(54 + 402 + 1460);

        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 54 + 402, &bytes[456..], 1);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 54, &bytes[54..456], 2);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 54, &bytes[54..456], 3);
        assert!(read_all(&mut stack).is_empty(), "nothing readable across the gap");
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 0, &bytes[..54], 4);

        assert_eq!(read_all(&mut stack), bytes);
        assert!(open_conn(&stack).held.is_empty(), "delivered segments are released");
    }

    #[test]
    fn duplicate_and_front_overlapping_segments_deliver_each_byte_once() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let bytes = stream(54 + 402 + 1460);

        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 0, &bytes[..54], 1);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 0, &bytes[..54], 2);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 30, &bytes[30..432], 3);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 400, &bytes[400..1860], 4);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 432, &bytes[432..], 5);

        assert_eq!(read_all(&mut stack), bytes);
        assert_eq!(stack.recv_window_drops(ID), Some(0));
    }

    #[test]
    fn held_segment_overlapped_by_the_gap_fill_is_consumed_without_duplication() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let bytes = stream(1460 + 402);

        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 1000, &bytes[1000..], 1);
        send_data(&mut stack, CLIENT_ISN, OUR_ISN, 0, &bytes[..1460], 2);

        assert_eq!(read_all(&mut stack), bytes);
        assert!(open_conn(&stack).held.is_empty());
    }

    #[test]
    fn send_buffer_stays_bounded_by_the_unacked_amount_across_many_acked_writes() {
        let mut stack = established();
        let _ = stack.poll_accept();
        let chunk = stream(1460);
        let writes = 200;
        let mut wire = Vec::new();

        for round in 0..writes {
            let out = stack.write(ID, &chunk, Instant::from_micros(round + 1));
            wire.extend(out.iter().flat_map(|(_, segment)| segment.payload.clone()));
            let acked = OUR_ISN + 1 + 1460 * (round as u32 + 1);
            stack.on_inbound(
                &seg(ack(), CLIENT_ISN + 1, acked, &[]),
                Instant::from_micros(round + 1),
            );
            assert!(
                open_conn(&stack).send_buf.len() <= 1460,
                "round {round}: send_buf holds more than the unacked amount"
            );
        }

        assert!(wire.len() > 4 * usize::from(OUR_WINDOW), "wrote far past the window");
        assert_eq!(wire, chunk.repeat(writes as usize), "acked bytes were transmitted intact");
        assert!(open_conn(&stack).send_buf.is_empty(), "everything acked, nothing retained");
    }

    #[test]
    fn inbound_stream_crossing_the_sequence_wrap_reads_back_in_order() {
        let client_isn = u32::MAX - 100;
        let mut stack = established_with(client_isn, OUR_ISN);
        let _ = stack.poll_accept();
        let bytes = stream(54 + 402 + 1460);

        send_data(&mut stack, client_isn, OUR_ISN, 54, &bytes[54..456], 1);
        send_data(&mut stack, client_isn, OUR_ISN, 456, &bytes[456..], 2);
        send_data(&mut stack, client_isn, OUR_ISN, 0, &bytes[..54], 3);

        assert_eq!(read_all(&mut stack), bytes);
        assert!(open_conn(&stack).held.is_empty());
    }

    #[test]
    fn outbound_stream_crossing_the_sequence_wrap_is_acked_and_released() {
        let our_isn = u32::MAX - 1000;
        let mut stack = established_with(CLIENT_ISN, our_isn);
        let _ = stack.poll_accept();
        let chunk = stream(1460);
        let mut sent_total = 0u32;
        let mut wire = Vec::new();

        for round in 0..4u64 {
            let out = stack.write(ID, &chunk, Instant::from_micros(round + 1));
            for (_, segment) in &out {
                assert_eq!(
                    segment.seq,
                    our_isn.wrapping_add(1).wrapping_add(sent_total),
                    "segment seq follows the wrapping send sequence"
                );
                sent_total = sent_total.wrapping_add(segment.payload.len() as u32);
                wire.extend_from_slice(&segment.payload);
            }
            let acked = our_isn.wrapping_add(1).wrapping_add(sent_total);
            stack.on_inbound(
                &seg(ack(), CLIENT_ISN + 1, acked, &[]),
                Instant::from_micros(round + 1),
            );
        }

        assert_eq!(wire, chunk.repeat(4));
        assert!(open_conn(&stack).send_buf.is_empty(), "acks across the wrap released the buffer");
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
    fn recv_buffer_at_window_capacity_drops_and_counts_further_delivered_bytes() {
        // arrange: an established connection whose `recv` is pre-filled to
        // exactly `OUR_WINDOW` bytes (bypassing DataPath's own accounting by
        // writing the private field directly — this test exercises the
        // capacity BACKSTOP `on_segment` enforces on `recv` itself, not
        // DataPath's window tracking).
        let mut stack = established();
        let _ = stack.poll_accept();
        {
            let Some(Entry::Open(conn)) = stack.conns.get_mut(&ID) else {
                panic!("established connection must be Entry::Open");
            };
            conn.recv.resize(usize::from(OUR_WINDOW), 0);
        }
        assert_eq!(stack.recv_window_drops(ID), Some(0), "no drops yet");

        // act: one more in-order byte arrives while `recv` is already at cap.
        stack.on_inbound(
            &seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1, b"X"),
            Instant::from_micros(1),
        );

        // assert: dropped and counted, not silently grown past the window.
        assert_eq!(
            stack.recv_window_drops(ID),
            Some(1),
            "byte beyond OUR_WINDOW is dropped and counted, not buffered"
        );
        let Some(Entry::Open(conn)) = stack.conns.get(&ID) else {
            panic!("established connection must be Entry::Open");
        };
        assert_eq!(
            conn.recv.len(),
            usize::from(OUR_WINDOW),
            "the queue never grew past the window it was pre-filled to"
        );
    }

    #[test]
    fn lifecycle_events_drop_and_count_past_the_configured_cap() {
        // arrange: a cap of 0 means the very first accept must drop.
        let mut stack = TcpStack::with_lifecycle_event_cap(OUR_IP, OUR_PORT, OUR_ISN, 0);
        assert_eq!(stack.dropped_accepted(), 0);

        // act: drive a full handshake to completion.
        let out = stack.on_inbound(&seg(syn(), CLIENT_ISN, 0, &[]), Instant::ZERO);
        assert!(out[0].1.flags.syn && out[0].1.flags.ack);
        stack.on_inbound(&seg(ack(), CLIENT_ISN + 1, OUR_ISN + 1, &[]), Instant::ZERO);

        // assert: the accept notification was dropped and counted, never
        // silently grown past the configured cap.
        assert_eq!(
            stack.dropped_accepted(),
            1,
            "accept event beyond the cap is dropped and counted"
        );
        assert_eq!(
            stack.poll_accept(),
            None,
            "the dropped notification never reaches poll_accept"
        );
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
        let (id, out) = stack.connect(peer).expect("ephemeral port available");
        assert!(id.0 >= 49152 && id.0 != OUR_PORT, "active open uses an ephemeral local port");
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
        let reply = stack.on_inbound_to(id.0, &seg(synack, CLIENT_ISN, OUR_ISN + 1, &[]), Instant::ZERO);
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
        stack.on_inbound_to(
            id.0,
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
