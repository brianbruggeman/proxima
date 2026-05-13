//! Per-connection stream table per [RFC 9000 §2].
//!
//! Bounded by `heapless::IndexMap` with const-generic caps from
//! `prime-runtime.toml [quic].max_concurrent_bidi_streams` /
//! `max_concurrent_uni_streams`.
//!
//! [RFC 9000 §2]: https://www.rfc-editor.org/rfc/rfc9000#section-2

use heapless::index_map::FnvIndexMap;

use crate::quic::side::Side;

use super::flow::StreamFlowControl;
use super::id::{StreamDirection, StreamId};
use super::state::{RecvState, SendState};

/// A single stream's combined state.
#[derive(Debug, Clone)]
pub struct Stream {
    pub id: StreamId,
    pub send: SendState,
    pub recv: RecvState,
    pub flow: StreamFlowControl,
}

impl Stream {
    /// Construct a fresh bidirectional stream.
    #[must_use]
    pub fn new_bidi(id: StreamId, flow: StreamFlowControl) -> Self {
        Self {
            id,
            send: SendState::new(),
            recv: RecvState::new(),
            flow,
        }
    }

    /// Construct a fresh local-initiated unidirectional stream
    /// (send-only — recv is in a terminal `DataRead{0}` state).
    #[must_use]
    pub fn new_local_uni(id: StreamId, flow: StreamFlowControl) -> Self {
        Self {
            id,
            send: SendState::new(),
            recv: RecvState::DataRead { offset_final: 0 },
            flow,
        }
    }

    /// Construct a fresh peer-initiated unidirectional stream
    /// (recv-only — send is in a terminal `DataRecvd{0}` state).
    #[must_use]
    pub fn new_peer_uni(id: StreamId, flow: StreamFlowControl) -> Self {
        Self {
            id,
            send: SendState::DataRecvd { offset_final: 0 },
            recv: RecvState::new(),
            flow,
        }
    }

    /// Fully closed per RFC 9000 §3 — both directions terminal (send
    /// `DataRecvd`/`ResetRecvd` AND recv `DataRead`/`ResetRead`), so there
    /// is no in-flight data to retransmit and nothing left to read. The
    /// table slot can be freed.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.send.is_terminal() && self.recv.is_terminal()
    }

    /// The application exchange on this bidi stream is complete: our recv
    /// side is terminal (request/response fully read) AND our send FIN/RESET is
    /// on the wire. We deliberately do NOT wait for our FIN to be ACKed — receiving the full response PROVES the peer got the request, so
    /// the ACK is moot for slot reuse. Without this, high-concurrency
    /// multiplexing leaves response-received-but-FIN-unacked streams pinned in
    /// the table until ACKs catch up, capping concurrency at MAX_BIDI.
    #[must_use]
    pub fn is_exchange_complete(&self) -> bool {
        self.recv.is_terminal() && (self.send.is_closed() || self.send.is_reset())
    }
}

/// First bidi stream id the PEER of `side` issues.
fn peer_bidi_base(side: Side) -> u64 {
    StreamId::next_local(None, side.peer(), StreamDirection::Bidi).as_u64()
}

/// Bounded per-direction stream table. Cap values are passed as
/// const generics so the type system enforces them at the proto
/// layer. Uses `heapless::FnvIndexMap` (O(1) hashed lookup; the caps
/// are build-time-enforced powers of two, FnvIndexMap's requirement)
/// so per-frame stream ops stop scaling with concurrency.
#[derive(Debug, Clone)]
pub struct StreamTable<const MAX_BIDI: usize, const MAX_UNI: usize> {
    bidi: FnvIndexMap<u64, Stream, MAX_BIDI>,
    uni: FnvIndexMap<u64, Stream, MAX_UNI>,
    next_local_bidi: Option<StreamId>,
    next_local_uni: Option<StreamId>,
    /// CONTIGUOUS closed prefix per bidi class: `Some(t)` means every bidi
    /// id `<= t` of that class has been reaped (RFC 9000 §3 closed). It
    /// advances ONLY over contiguous terminal streams from the base — never
    /// a max-seen watermark — so an unarrived lower id (reordering) blocks
    /// it and is never mistaken for closed. Only bidi: uni control/QPACK
    /// streams never close, so they are never reaped or resurrection-
    /// guarded.
    peer_bidi_reaped_thru: Option<u64>,
    /// Count of peer-initiated bidi streams reaped (prefix advanced) since
    /// the connection layer last called `drain_peer_bidi_reaped_delta`.
    /// Drives `MaxStreamsState::record_peer_closed` + `MAX_STREAMS` reissue
    /// without changing the `reap_closed_bidi` return type.
    peer_bidi_reaped_delta: u64,
}

/// Errors from the stream-table API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamTableError {
    /// At `MAX_BIDI` or `MAX_UNI` capacity for the requested direction.
    LimitReached,
    /// Caller asked about a stream ID that has no entry.
    UnknownStream,
}

impl<const MAX_BIDI: usize, const MAX_UNI: usize> Default for StreamTable<MAX_BIDI, MAX_UNI> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_BIDI: usize, const MAX_UNI: usize> StreamTable<MAX_BIDI, MAX_UNI> {
    /// Construct an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bidi: FnvIndexMap::new(),
            uni: FnvIndexMap::new(),
            next_local_bidi: None,
            next_local_uni: None,
            peer_bidi_reaped_thru: None,
            peer_bidi_reaped_delta: 0,
        }
    }

    /// Number of bidirectional streams.
    #[must_use]
    pub fn bidi_count(&self) -> usize {
        self.bidi.len()
    }

    /// Number of unidirectional streams.
    #[must_use]
    pub fn uni_count(&self) -> usize {
        self.uni.len()
    }

    /// Locally open a new stream of the given direction. Returns the
    /// assigned [`StreamId`].
    ///
    /// # Errors
    ///
    /// Returns [`StreamTableError::LimitReached`] when the per-direction
    /// table is at capacity.
    pub fn open_local(
        &mut self,
        side: Side,
        direction: StreamDirection,
        flow: StreamFlowControl,
    ) -> Result<StreamId, StreamTableError> {
        match direction {
            StreamDirection::Bidi => {
                // Free closed request slots first so a reused connection's
                // cap reflects CONCURRENT, not lifetime, bidi streams.
                self.reap_closed_bidi(side);
                let id = StreamId::next_local(self.next_local_bidi, side, direction);
                let stream = Stream::new_bidi(id, flow);
                self.bidi
                    .insert(id.as_u64(), stream)
                    .map_err(|_| StreamTableError::LimitReached)?;
                self.next_local_bidi = Some(id);
                Ok(id)
            }
            StreamDirection::Uni => {
                let id = StreamId::next_local(self.next_local_uni, side, direction);
                let stream = Stream::new_local_uni(id, flow);
                self.uni
                    .insert(id.as_u64(), stream)
                    .map_err(|_| StreamTableError::LimitReached)?;
                self.next_local_uni = Some(id);
                Ok(id)
            }
        }
    }

    /// Get-or-create a stream slot for an inbound peer-initiated
    /// frame referring to `id`.
    ///
    /// # Errors
    ///
    /// Returns [`StreamTableError::LimitReached`] when the per-direction
    /// table is at capacity.
    pub fn get_or_create_peer(
        &mut self,
        id: StreamId,
        flow: StreamFlowControl,
    ) -> Result<&mut Stream, StreamTableError> {
        match id.direction() {
            StreamDirection::Bidi => {
                if !self.bidi.contains_key(&id.as_u64()) {
                    let stream = Stream::new_bidi(id, flow);
                    self.bidi
                        .insert(id.as_u64(), stream)
                        .map_err(|_| StreamTableError::LimitReached)?;
                }
                self.bidi
                    .get_mut(&id.as_u64())
                    .ok_or(StreamTableError::UnknownStream)
            }
            StreamDirection::Uni => {
                if !self.uni.contains_key(&id.as_u64()) {
                    let stream = Stream::new_peer_uni(id, flow);
                    self.uni
                        .insert(id.as_u64(), stream)
                        .map_err(|_| StreamTableError::LimitReached)?;
                }
                self.uni
                    .get_mut(&id.as_u64())
                    .ok_or(StreamTableError::UnknownStream)
            }
        }
    }

    /// Borrow a stream slot if it exists.
    #[must_use]
    pub fn get(&self, id: StreamId) -> Option<&Stream> {
        match id.direction() {
            StreamDirection::Bidi => self.bidi.get(&id.as_u64()),
            StreamDirection::Uni => self.uni.get(&id.as_u64()),
        }
    }

    /// Borrow a stream slot mutably.
    pub fn get_mut(&mut self, id: StreamId) -> Option<&mut Stream> {
        match id.direction() {
            StreamDirection::Bidi => self.bidi.get_mut(&id.as_u64()),
            StreamDirection::Uni => self.uni.get_mut(&id.as_u64()),
        }
    }

    /// Remove a completed stream from the table, freeing its slot
    /// so the per-direction cap reflects **concurrent** streams, not
    /// lifetime total. The caller is responsible for ensuring the
    /// stream is fully terminal (both send and recv sides in a
    /// terminal state + all retransmittable intents drained) before
    /// calling — removing a stream with in-flight data would drop
    /// retransmit-on-loss intents.
    pub fn remove(&mut self, id: StreamId) -> Option<Stream> {
        match id.direction() {
            StreamDirection::Bidi => self.bidi.remove(&id.as_u64()),
            StreamDirection::Uni => self.uni.remove(&id.as_u64()),
        }
    }

    /// Free every bidi (request) stream that has reached the RFC 9000 §3
    /// closed state, advancing the per-class CONTIGUOUS closed prefix. A
    /// terminal stream ABOVE the prefix (a lower id still open or unarrived)
    /// is kept until the prefix reaches it — so a reaped id is never
    /// confused with an unarrived one. Uni control/QPACK streams are
    /// untouched (they never close). Reuse calls this before opening or
    /// admitting a stream so the cap reflects CONCURRENT, not lifetime,
    /// streams — letting one connection serve far more than `MAX_BIDI`.
    ///
    /// Each call accumulates reaped peer-bidi count into
    /// `peer_bidi_reaped_delta`; drain it via
    /// [`Self::drain_peer_bidi_reaped_delta`] in `poll_transmit` to credit
    /// `MaxStreamsState::record_peer_closed` and emit `MAX_STREAMS` frames.
    pub fn reap_closed_bidi(&mut self, side: Side) {
        // PEER-issued streams: contiguous prefix only. A lower id may still be
        // unarrived (network reordering) and would be indistinguishable from a
        // reaped one, so we never reap past a gap.
        let old_peer_thru = self.peer_bidi_reaped_thru;
        self.peer_bidi_reaped_thru =
            self.advance_bidi_prefix(self.peer_bidi_reaped_thru, peer_bidi_base(side));
        let peer_base = peer_bidi_base(side);
        let peer_reaped = match (old_peer_thru, self.peer_bidi_reaped_thru) {
            (None, Some(thru)) => (thru - peer_base) / 4 + 1,
            (Some(old), Some(new)) if new > old => (new - old) / 4,
            _ => 0,
        };
        self.peer_bidi_reaped_delta = self.peer_bidi_reaped_delta.saturating_add(peer_reaped);
        // LOCALLY-issued streams: we chose the ids, so there is no unarrived-
        // lower ambiguity — reap every fully-closed local bidi OUT OF ORDER.
        // The contiguous prefix stalls behind the lowest still-open stream;
        // under multiplexed concurrency requests complete out of order and the
        // table jams at MAX_BIDI (observed: client capped at ~1024 → table-full
        // ProtocolViolation). `is_reaped` resolves local ids by membership +
        // next-id, so no watermark is needed here.
        //
        // Guard on having opened ANY local bidi: a server only opens local uni
        // (its control stream), so its request streams are all peer-initiated.
        // Without this guard the server scanned the whole bidi table (each
        // entry ~1 KiB with inline buffers) on every STREAM frame for local
        // streams that never exist — ~27% of the core under load.
        if self.next_local_bidi.is_none() {
            return;
        }
        let mut closed: heapless::Vec<u64, MAX_BIDI> = heapless::Vec::new();
        for (id, stream) in &self.bidi {
            let stream_id = StreamId(*id);
            if stream_id.is_local(side) && stream.is_exchange_complete() {
                let _ = closed.push(*id);
            }
        }
        for id in closed {
            self.bidi.remove(&id);
        }
    }

    /// Consume and return the count of peer-initiated bidi streams reaped
    /// since the last call. The connection layer calls this in
    /// `poll_transmit_established` to credit
    /// `MaxStreamsState::record_peer_closed` and emit `MAX_STREAMS` frames
    /// that raise the peer's cumulative stream cap (RFC 9000 §4.6 / §19.11).
    pub fn drain_peer_bidi_reaped_delta(&mut self) -> u64 {
        core::mem::replace(&mut self.peer_bidi_reaped_delta, 0)
    }

    /// Advance one contiguous closed prefix: while the next id at the
    /// prefix boundary is a TERMINAL stream in the table, remove it and
    /// step the prefix by the RFC 9000 §2.1 stride (4). Stops at the first
    /// id that is open or absent (unarrived) — never skipping a gap.
    fn advance_bidi_prefix(&mut self, reaped_thru: Option<u64>, base: u64) -> Option<u64> {
        let mut thru = reaped_thru;
        loop {
            let next = thru.map_or(base, |prev| prev + 4);
            match self.bidi.get(&next) {
                // Relaxed from is_terminal (FIN ACKed) to is_exchange_complete
                // (response fully transferred + our FIN on the wire). Waiting
                // for the FIN ACK to advance the contiguous prefix pins every
                // completed stream for an extra RTT under load — at high
                // multiplexed concurrency the ACKs lag far enough that the
                // prefix never advances and the table jams at MAX_BIDI. The
                // contiguity (resurrection guard) is unchanged.
                Some(stream) if stream.is_exchange_complete() => {
                    self.bidi.remove(&next);
                    thru = Some(next);
                }
                _ => break,
            }
        }
        thru
    }

    /// Whether `id` is a reaped (closed) stream whose frames must be
    /// DROPPED rather than (re)opened (RFC 9000 §3). True only for a bidi
    /// id at or below the contiguous closed prefix; uni ids are never
    /// reaped (→ `false`, admit), and a bidi id above the prefix is new or
    /// unarrived (→ `false`, admit).
    #[must_use]
    pub fn is_reaped(&self, id: StreamId, side: Side) -> bool {
        if id.direction() != StreamDirection::Bidi {
            return false;
        }
        if id.is_local(side) {
            // We assigned this id; if it is at/below the highest id we have
            // handed out and is no longer in the table, it was reaped. This
            // admits out-of-order local reaping without a contiguous watermark.
            return self
                .next_local_bidi
                .is_some_and(|last| id.as_u64() <= last.as_u64())
                && !self.bidi.contains_key(&id.as_u64());
        }
        self.peer_bidi_reaped_thru
            .is_some_and(|thru| id.as_u64() <= thru)
    }

    /// Iterate over all streams (bidi + uni).
    pub fn iter(&self) -> impl Iterator<Item = &Stream> {
        self.bidi.values().chain(self.uni.values())
    }

    /// Iterate over all streams mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Stream> {
        self.bidi.values_mut().chain(self.uni.values_mut())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    type SmallTable = StreamTable<4, 4>;

    #[test]
    fn new_table_is_empty() {
        let table = SmallTable::new();
        assert_eq!(table.bidi_count(), 0);
        assert_eq!(table.uni_count(), 0);
    }

    #[test]
    fn open_local_bidi_assigns_client_zero_then_four() {
        let mut table = SmallTable::new();
        let id1 = table
            .open_local(
                Side::Client,
                StreamDirection::Bidi,
                StreamFlowControl::default(),
            )
            .expect("first slot");
        assert_eq!(id1, StreamId(0));
        let id2 = table
            .open_local(
                Side::Client,
                StreamDirection::Bidi,
                StreamFlowControl::default(),
            )
            .expect("second slot");
        assert_eq!(id2, StreamId(4));
        assert_eq!(table.bidi_count(), 2);
    }

    #[test]
    fn open_local_uni_assigns_client_two_then_six() {
        let mut table = SmallTable::new();
        let id1 = table
            .open_local(
                Side::Client,
                StreamDirection::Uni,
                StreamFlowControl::default(),
            )
            .expect("first slot");
        assert_eq!(id1, StreamId(2));
        let id2 = table
            .open_local(
                Side::Client,
                StreamDirection::Uni,
                StreamFlowControl::default(),
            )
            .expect("second slot");
        assert_eq!(id2, StreamId(6));
    }

    #[test]
    fn open_local_at_cap_returns_limit_reached() {
        let mut table: StreamTable<2, 2> = StreamTable::new();
        // 2-slot table → only one bidi-client can fit because the
        // LinearMap is power-of-two-sized; LimitReached when the
        // map is full.
        for _ in 0..2 {
            let _ = table.open_local(
                Side::Client,
                StreamDirection::Bidi,
                StreamFlowControl::default(),
            );
        }
        let result = table.open_local(
            Side::Client,
            StreamDirection::Bidi,
            StreamFlowControl::default(),
        );
        assert_eq!(result, Err(StreamTableError::LimitReached));
    }

    #[test]
    fn get_or_create_peer_creates_a_bidi_slot_on_first_access() {
        let mut table = SmallTable::new();
        // Peer's first server-bidi stream is StreamId(1).
        let stream = table
            .get_or_create_peer(StreamId(1), StreamFlowControl::default())
            .expect("create");
        assert_eq!(stream.id, StreamId(1));
        assert_eq!(table.bidi_count(), 1);
        // Subsequent access returns the same slot.
        let again = table
            .get_or_create_peer(StreamId(1), StreamFlowControl::default())
            .expect("again");
        assert_eq!(again.id, StreamId(1));
        assert_eq!(table.bidi_count(), 1);
    }

    #[test]
    fn get_returns_none_for_unknown_stream() {
        let table = SmallTable::new();
        assert!(table.get(StreamId(99)).is_none());
    }

    #[test]
    fn iter_visits_both_bidi_and_uni() {
        let mut table = SmallTable::new();
        table
            .open_local(
                Side::Client,
                StreamDirection::Bidi,
                StreamFlowControl::default(),
            )
            .expect("bidi");
        table
            .open_local(
                Side::Client,
                StreamDirection::Uni,
                StreamFlowControl::default(),
            )
            .expect("uni");
        let ids: alloc::vec::Vec<_> = table.iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&StreamId(0)));
        assert!(ids.contains(&StreamId(2)));
    }

    /// Admit a peer bidi stream `n` and drive it to the RFC 9000 §3 closed
    /// state (used by the reaping worked-example tests).
    fn admit_close(table: &mut StreamTable<8, 8>, n: u64) {
        table
            .get_or_create_peer(StreamId(n), StreamFlowControl::default())
            .expect("admit");
        let stream = table.get_mut(StreamId(n)).expect("stream");
        stream.send = SendState::DataRecvd { offset_final: 0 };
        stream.recv = RecvState::DataRead { offset_final: 0 };
    }

    /// Worked example (server side): the client (peer) opens bidi request
    /// streams serially; the CONTIGUOUS closed prefix frees each on close
    /// and drops late frames for it — but a reordering gap (a higher
    /// stream closing while a lower one is unarrived) does NOT advance the
    /// prefix, and uni control streams are never reaped. Encodes the
    /// algorithm-development proof for the C43 stream-reaping follow-on.
    #[test]
    fn contiguous_reaping_frees_serial_streams_and_blocks_resurrection() {
        let mut table: StreamTable<8, 8> = StreamTable::new();

        // Stream 0 closes → reap frees it; the prefix advances to 0.
        admit_close(&mut table, 0);
        table.reap_closed_bidi(Side::Server);
        assert_eq!(table.bidi_count(), 0, "closed stream 0 freed");
        assert!(
            table.is_reaped(StreamId(0), Side::Server),
            "0 reads as closed"
        );
        assert!(
            !table.is_reaped(StreamId(4), Side::Server),
            "4 not yet closed"
        );
        // A uni control stream is never reaped, so never resurrection-guarded.
        assert!(
            !table.is_reaped(StreamId(2), Side::Server),
            "uni never reaped"
        );

        // Reordering: stream 8 closes while stream 4 is unarrived. The
        // prefix must NOT jump the gap; 8 stays as a placeholder and an
        // unarrived 4 is NOT mistaken for closed.
        admit_close(&mut table, 8);
        table.reap_closed_bidi(Side::Server);
        assert_eq!(table.bidi_count(), 1, "8 kept above the gap");
        assert!(
            !table.is_reaped(StreamId(4), Side::Server),
            "unarrived 4 is NOT closed (the bug the naive watermark had)"
        );

        // Stream 4 now arrives + closes → the prefix advances 0→4→8,
        // freeing both contiguously.
        admit_close(&mut table, 4);
        table.reap_closed_bidi(Side::Server);
        assert_eq!(table.bidi_count(), 0, "4 and 8 freed once contiguous");
        assert!(table.is_reaped(StreamId(8), Side::Server), "8 now closed");
    }

    /// Fully reordered arrival (8, then 4, then 0): the prefix can't
    /// advance until the contiguous-lowest arrives, so NONE of them is ever
    /// mistaken for closed (every `is_reaped` is false while a lower id is
    /// unarrived); once 0 arrives + closes, the prefix sweeps 0→4→8 and
    /// frees all three. This is why the contiguous prefix needs no
    /// separate "implicit-open" — the gap conservatism handles reordering.
    #[test]
    fn reaping_handles_fully_reordered_arrival_without_dropping() {
        let mut table: StreamTable<8, 8> = StreamTable::new();
        admit_close(&mut table, 8);
        table.reap_closed_bidi(Side::Server);
        assert!(
            !table.is_reaped(StreamId(0), Side::Server),
            "0 unarrived, not closed"
        );
        assert!(
            !table.is_reaped(StreamId(4), Side::Server),
            "4 unarrived, not closed"
        );
        admit_close(&mut table, 4); // arrives out of order; must be admitted
        table.reap_closed_bidi(Side::Server);
        assert!(
            !table.is_reaped(StreamId(0), Side::Server),
            "0 still unarrived"
        );
        assert_eq!(
            table.bidi_count(),
            2,
            "4 and 8 held as placeholders above the gap"
        );
        admit_close(&mut table, 0); // the gap fills; the prefix sweeps 0→4→8
        table.reap_closed_bidi(Side::Server);
        assert_eq!(table.bidi_count(), 0, "all three reaped once contiguous");
        assert!(table.is_reaped(StreamId(8), Side::Server), "8 now closed");
    }

    #[test]
    fn contiguous_reaping_admits_far_more_streams_than_the_cap() {
        // 200 serial request streams over MAX_BIDI=8 — only possible
        // because each is reaped on close, freeing its slot.
        let mut table: StreamTable<8, 8> = StreamTable::new();
        for n in (0..800).step_by(4) {
            admit_close(&mut table, n);
            table.reap_closed_bidi(Side::Server);
        }
        assert_eq!(table.bidi_count(), 0, "all 200 reaped contiguously");
    }

    extern crate alloc;
}
