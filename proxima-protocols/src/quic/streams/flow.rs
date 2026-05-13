//! Connection-level + stream-level flow-control state per
//! [RFC 9000 §4].
//!
//! [RFC 9000 §4]: https://www.rfc-editor.org/rfc/rfc9000#section-4
//!
//! # C12.5 — dynamic-update helpers
//!
//! Beyond the initial MAX_DATA / MAX_STREAM_DATA values, the
//! flow-control loop runs in two directions:
//!
//! - **Recv-side** — when the application drains data we MUST emit
//!   MAX_DATA / MAX_STREAM_DATA to grow the peer's send window
//!   (RFC 9000 §4.1 / §4.5). Threshold: when remaining recv budget
//!   drops below half the originally-granted credit we double it.
//!   `should_emit_max_data()` / `should_emit_max_stream_data()`
//!   surface the new credit value the caller writes into the frame.
//! - **Send-side** — when we exhaust the peer's send credit we
//!   SHOULD emit DATA_BLOCKED / STREAM_DATA_BLOCKED for diagnostic
//!   back-pressure (RFC §19.12 / §19.13). `is_send_blocked()`
//!   returns true when `sent_offset == credit_send`.

/// Threshold ratio for auto-emission of MAX_DATA / MAX_STREAM_DATA.
/// When `recv_budget()` drops below `credit_recv / GRANT_THRESHOLD_DIVISOR`
/// we emit a new credit value of `credit_recv * GRANT_GROWTH_FACTOR`.
const GRANT_THRESHOLD_DIVISOR: u64 = 2;

/// Connection-level flow control per RFC 9000 §4.1.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectionFlowControl {
    /// Bytes peer has authorised us to send (via MAX_DATA).
    pub credit_send: u64,
    /// Bytes we've authorised peer (sent in MAX_DATA frames).
    pub credit_recv: u64,
    /// Initial recv credit advertised at construction. Used as the
    /// per-grant window size so `should_emit_max_data` advertises
    /// `recv_offset + initial_credit_recv` — bounding peer in-flight
    /// to one initial window regardless of how long the connection
    /// lives, and (critically) regardless of `recv_high_water`. The
    /// latter can advance for bytes we couldn't store
    /// (TransientRecvBufferFull); growing credit off high_water would
    /// advertise capacity we can't honour.
    pub initial_credit_recv: u64,
    /// Bytes we've actually sent (counts against `credit_send`).
    pub sent_offset: u64,
    /// Bytes the application has consumed (drives `should_emit_max_data`
    /// — credit grows in `recv_offset + initial_credit_recv` increments
    /// so the in-flight buffer stays bounded by the original
    /// advertisement).
    pub recv_offset: u64,
    /// Highest sum-across-streams of bytes the peer has sent us
    /// (charged against `credit_recv` for RFC 9000 §4.1 enforcement).
    /// Distinct from `recv_offset`: a retransmission charges the same
    /// (offset + len) and so contributes 0 here, while `recv_offset`
    /// only advances when the application drains.
    pub recv_high_water: u64,
}

impl ConnectionFlowControl {
    /// Construct with the local + peer initial-max-data transport
    /// parameters.
    #[must_use]
    pub const fn new(initial_credit_send: u64, initial_credit_recv: u64) -> Self {
        Self {
            credit_send: initial_credit_send,
            credit_recv: initial_credit_recv,
            initial_credit_recv,
            sent_offset: 0,
            recv_offset: 0,
            recv_high_water: 0,
        }
    }

    /// Bytes we can still send across all streams without blocking.
    #[must_use]
    pub fn send_budget(&self) -> u64 {
        self.credit_send.saturating_sub(self.sent_offset)
    }

    /// Bytes peer can still send to us without violating flow control.
    #[must_use]
    pub fn recv_budget(&self) -> u64 {
        self.credit_recv.saturating_sub(self.recv_offset)
    }

    /// Record that we've actually sent `bytes` bytes of stream data.
    pub fn record_sent(&mut self, bytes: u64) {
        self.sent_offset = self.sent_offset.saturating_add(bytes);
    }

    /// Record that the application has consumed `bytes` of stream data.
    pub fn record_consumed(&mut self, bytes: u64) {
        self.recv_offset = self.recv_offset.saturating_add(bytes);
    }

    /// Update the send-side credit upon receipt of a MAX_DATA frame.
    /// MAX_DATA values are monotonically non-decreasing — older values
    /// are silently dropped per RFC 9000 §19.9.
    pub fn observe_max_data(&mut self, peer_max_data: u64) {
        if peer_max_data > self.credit_send {
            self.credit_send = peer_max_data;
        }
    }

    /// Grow the recv-side credit (caller emits a MAX_DATA frame with
    /// `new_credit_recv`).
    pub fn grant_recv_credit(&mut self, new_credit_recv: u64) {
        if new_credit_recv > self.credit_recv {
            self.credit_recv = new_credit_recv;
        }
    }

    /// `true` when the local sender has exhausted the peer's send
    /// credit. Caller MAY emit a DATA_BLOCKED frame per RFC §19.12
    /// to inform the peer of the back-pressure.
    #[must_use]
    pub fn is_send_blocked(&self) -> bool {
        self.send_budget() == 0
    }

    /// If the peer's remaining recv budget has dropped below the
    /// grant threshold, returns the new `credit_recv` value the
    /// caller should advertise in a MAX_DATA frame. `None` if no
    /// update is warranted yet.
    ///
    /// Side-effect-free observer; caller calls
    /// [`Self::grant_recv_credit`] with the returned value AND
    /// writes the MAX_DATA frame.
    #[must_use]
    pub fn should_emit_max_data(&self) -> Option<u64> {
        if self.credit_recv == 0 {
            return None;
        }
        // Trigger: peer is about to hit the credit wall (high water
        // close to credit_recv). Using recv_offset (app-consumed)
        // would miss the case where peer races ahead of app.
        //
        // Value: `recv_offset + initial_credit_recv`. This bounds the
        // peer's in-flight to one initial window past whatever the
        // app has already drained — buffer footprint stays bounded
        // regardless of connection lifetime. Critically, the grant
        // does NOT grow off `recv_high_water`: bytes that arrived but
        // failed local storage (TransientRecvBufferFull) advance
        // high_water, but the grant grows only as the app actually
        // drains.
        let threshold = self.credit_recv / GRANT_THRESHOLD_DIVISOR;
        let remaining = self.credit_recv.saturating_sub(self.recv_high_water);
        if remaining < threshold {
            let proposed = self.recv_offset.saturating_add(self.initial_credit_recv);
            if proposed > self.credit_recv {
                Some(proposed)
            } else {
                // grant_recv_credit is monotonic; emitting a no-op
                // grant is wasted bandwidth. Skip when the app hasn't
                // drained enough to widen the window.
                None
            }
        } else {
            None
        }
    }
}

/// Per-stream flow control per RFC 9000 §4.5.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamFlowControl {
    pub credit_send: u64,
    pub credit_recv: u64,
    /// Initial per-stream recv credit advertised at construction;
    /// drives the grant value in `should_emit_max_stream_data` (see
    /// `ConnectionFlowControl::initial_credit_recv` for the
    /// rationale).
    pub initial_credit_recv: u64,
    pub sent_offset: u64,
    pub recv_offset: u64,
    /// Highest `offset + len` ever observed on an inbound STREAM
    /// frame for this stream. Used to compute the per-frame **delta**
    /// the inbound path charges against the connection-level
    /// recv-credit (RFC 9000 §4.1) — without this distinction a
    /// retransmission would double-charge the connection budget.
    pub recv_high_water: u64,
}

impl StreamFlowControl {
    /// Construct with the local + peer per-stream credit limits.
    #[must_use]
    pub const fn new(initial_credit_send: u64, initial_credit_recv: u64) -> Self {
        Self {
            credit_send: initial_credit_send,
            credit_recv: initial_credit_recv,
            initial_credit_recv,
            sent_offset: 0,
            recv_offset: 0,
            recv_high_water: 0,
        }
    }

    #[must_use]
    pub fn send_budget(&self) -> u64 {
        self.credit_send.saturating_sub(self.sent_offset)
    }

    #[must_use]
    pub fn recv_budget(&self) -> u64 {
        self.credit_recv.saturating_sub(self.recv_offset)
    }

    pub fn record_sent(&mut self, bytes: u64) {
        self.sent_offset = self.sent_offset.saturating_add(bytes);
    }

    pub fn record_consumed(&mut self, bytes: u64) {
        self.recv_offset = self.recv_offset.saturating_add(bytes);
    }

    pub fn observe_max_stream_data(&mut self, peer_max: u64) {
        if peer_max > self.credit_send {
            self.credit_send = peer_max;
        }
    }

    pub fn grant_recv_credit(&mut self, new_credit_recv: u64) {
        if new_credit_recv > self.credit_recv {
            self.credit_recv = new_credit_recv;
        }
    }

    /// `true` when the local sender has exhausted the peer's
    /// per-stream send credit. Caller MAY emit a STREAM_DATA_BLOCKED
    /// frame per RFC §19.13.
    #[must_use]
    pub fn is_send_blocked(&self) -> bool {
        self.send_budget() == 0
    }

    /// If the per-stream remaining recv budget has dropped below the
    /// grant threshold, returns the new `credit_recv` value the
    /// caller should advertise in a MAX_STREAM_DATA frame.
    ///
    /// Trigger uses peer-sent high water (RFC 9000 §4.5) so the gate
    /// fires when the peer is about to hit the wall. The grant value
    /// is `recv_offset + initial_credit_recv` (app-drained-bounded),
    /// not `credit_recv * 2` — the latter grows unboundedly with
    /// `recv_high_water` and is unsafe under TransientRecvBufferFull
    /// where peer-sent bytes don't land in our reassembly buffer.
    #[must_use]
    pub fn should_emit_max_stream_data(&self) -> Option<u64> {
        if self.credit_recv == 0 {
            return None;
        }
        let threshold = self.credit_recv / GRANT_THRESHOLD_DIVISOR;
        let remaining = self.credit_recv.saturating_sub(self.recv_high_water);
        if remaining < threshold {
            let proposed = self.recv_offset.saturating_add(self.initial_credit_recv);
            if proposed > self.credit_recv {
                Some(proposed)
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Per-direction MAX_STREAMS state per RFC 9000 §4.6.
///
/// Tracks (a) the peer-imposed limit on how many streams of a given
/// direction we may open, and (b) the local limit we've advertised
/// to the peer via MAX_STREAMS frames.
///
/// Note: this is a soft layer on top of the const-generic
/// `StreamTable<MAX_BIDI, MAX_UNI>` caps — the table caps are
/// absolute, while MaxStreamsState tracks the dynamic per-direction
/// permit count the peer may consume up to that cap.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxStreamsState {
    /// Largest stream-count the peer has authorized us to open via
    /// MAX_STREAMS (RFC §4.6). Monotonically non-decreasing.
    pub peer_limit: u64,
    /// Largest stream-count we've authorized the peer to open via
    /// our outbound MAX_STREAMS frames.
    pub local_limit: u64,
    /// Number of streams of this direction we've already opened
    /// locally (counts toward `peer_limit`).
    pub locally_opened: u64,
    /// Number of streams of this direction the peer has already
    /// opened (counts toward `local_limit`).
    pub peer_opened: u64,
    /// Number of peer-opened streams of this direction that have since
    /// closed (freed a stream-table slot). Drives `MAX_STREAMS` reissue.
    pub peer_closed: u64,
    /// The concurrent-stream window: how many more streams than the
    /// peer has CLOSED it may have open at once. Equals the fixed
    /// stream-table capacity (`initial_local_limit`), so the advertised
    /// cumulative cap never outruns what the table can physically hold.
    pub window: u64,
}

impl MaxStreamsState {
    /// Construct with the initial `initial_max_streams_*` transport-
    /// parameter values.
    #[must_use]
    pub const fn new(initial_peer_limit: u64, initial_local_limit: u64) -> Self {
        Self {
            peer_limit: initial_peer_limit,
            local_limit: initial_local_limit,
            locally_opened: 0,
            peer_opened: 0,
            peer_closed: 0,
            window: initial_local_limit,
        }
    }

    /// Apply an inbound MAX_STREAMS frame (RFC §19.11). Monotonic;
    /// older values silently dropped.
    pub fn observe_peer_max_streams(&mut self, peer_max: u64) {
        if peer_max > self.peer_limit {
            self.peer_limit = peer_max;
        }
    }

    /// Apply a locally-emitted MAX_STREAMS frame (caller stamps the
    /// new local_limit).
    pub fn grant_local_max_streams(&mut self, new_local_limit: u64) {
        if new_local_limit > self.local_limit {
            self.local_limit = new_local_limit;
        }
    }

    /// Record that we've opened a stream of this direction.
    pub fn record_locally_opened(&mut self) {
        self.locally_opened = self.locally_opened.saturating_add(1);
    }

    /// Record that the peer opened a stream of this direction.
    pub fn record_peer_opened(&mut self) {
        self.peer_opened = self.peer_opened.saturating_add(1);
    }

    /// Record that a peer-opened stream of this direction has closed
    /// and freed its table slot; drives `MAX_STREAMS` reissue.
    pub fn record_peer_closed(&mut self) {
        self.peer_closed = self.peer_closed.saturating_add(1);
    }

    /// If the peer is nearing the cumulative cap we advertised, returns
    /// the new `local_limit` to advertise in a `MAX_STREAMS` frame so it
    /// may open `window` more streams than it has closed — bounding
    /// concurrent open streams at the stream-table capacity while
    /// allowing unlimited total streams over the connection's life.
    /// `None` when no reissue is warranted yet. Side-effect-free; the
    /// caller calls [`Self::grant_local_max_streams`] with the returned
    /// value AND writes the `MAX_STREAMS` frame. Mirrors
    /// [`ConnectionFlowControl::should_emit_max_data`].
    #[must_use]
    pub fn should_emit_max_streams(&self) -> Option<u64> {
        if self.window == 0 {
            return None;
        }
        let target = self.peer_closed.saturating_add(self.window);
        let remaining = self.local_limit.saturating_sub(self.peer_opened);
        let threshold = self.window / GRANT_THRESHOLD_DIVISOR;
        if remaining <= threshold && target > self.local_limit {
            Some(target)
        } else {
            None
        }
    }

    /// `true` when we cannot open another local stream of this
    /// direction. Caller MAY emit STREAMS_BLOCKED per RFC §19.14.
    #[must_use]
    pub fn is_local_open_blocked(&self) -> bool {
        self.locally_opened >= self.peer_limit
    }

    /// `true` when the peer would exceed our advertised limit by
    /// opening another stream. Caller MUST emit a connection error
    /// per RFC §4.6 if this fires inbound.
    #[must_use]
    pub fn peer_would_exceed_local_limit(&self) -> bool {
        self.peer_opened >= self.local_limit
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn connection_send_budget_decreases_with_sent() {
        let mut fc = ConnectionFlowControl::new(1_000, 1_000);
        assert_eq!(fc.send_budget(), 1_000);
        fc.record_sent(300);
        assert_eq!(fc.send_budget(), 700);
    }

    #[test]
    fn connection_observe_max_data_is_monotonic() {
        let mut fc = ConnectionFlowControl::new(1_000, 1_000);
        fc.observe_max_data(500); // smaller — ignored
        assert_eq!(fc.credit_send, 1_000);
        fc.observe_max_data(5_000);
        assert_eq!(fc.credit_send, 5_000);
        fc.observe_max_data(2_000); // smaller — ignored
        assert_eq!(fc.credit_send, 5_000);
    }

    #[test]
    fn stream_send_budget_clips_at_zero_when_over_sent() {
        let mut fc = StreamFlowControl::new(100, 100);
        fc.record_sent(150);
        assert_eq!(fc.send_budget(), 0);
    }

    #[test]
    fn stream_recv_budget_grows_via_grant() {
        let mut fc = StreamFlowControl::new(0, 100);
        fc.record_consumed(50);
        assert_eq!(fc.recv_budget(), 50);
        fc.grant_recv_credit(500);
        assert_eq!(fc.recv_budget(), 450);
    }

    // ---- C12.5 — dynamic flow-control update helpers ----

    #[test]
    fn connection_is_send_blocked_when_credit_exhausted() {
        let mut fc = ConnectionFlowControl::new(1_000, 1_000);
        assert!(!fc.is_send_blocked());
        fc.record_sent(1_000);
        assert!(fc.is_send_blocked());
    }

    #[test]
    fn connection_should_emit_max_data_below_threshold() {
        // Trigger fires on peer-sent high water; grant value is
        // recv_offset + initial_credit_recv (consumption-bounded).
        // Need to advance BOTH high_water (trigger) AND recv_offset
        // (drives the grant value past the current credit_recv).
        let mut fc = ConnectionFlowControl::new(1_000, 1_000);
        assert!(fc.should_emit_max_data().is_none(), "no inbound yet");
        fc.recv_high_water = 400; // remaining = 600 > threshold (500)
        assert!(fc.should_emit_max_data().is_none());
        fc.recv_high_water = 550; // remaining = 450 < threshold (500)
        fc.recv_offset = 550; // consumed 550 → proposed = 550 + 1000 = 1550
        let next = fc.should_emit_max_data().expect("threshold breached");
        assert_eq!(next, 1_550, "grant = recv_offset + initial_credit_recv");
    }

    #[test]
    fn connection_should_emit_max_data_returns_none_at_zero_credit() {
        let fc = ConnectionFlowControl::new(0, 0);
        assert!(fc.should_emit_max_data().is_none());
    }

    #[test]
    fn stream_is_send_blocked_when_credit_exhausted() {
        let mut fc = StreamFlowControl::new(100, 100);
        assert!(!fc.is_send_blocked());
        fc.record_sent(100);
        assert!(fc.is_send_blocked());
    }

    #[test]
    fn stream_should_emit_max_stream_data_below_threshold() {
        // Mirror semantics to the connection-level test.
        let mut fc = StreamFlowControl::new(100, 100);
        assert!(fc.should_emit_max_stream_data().is_none());
        fc.recv_high_water = 60; // remaining = 40 < threshold (50)
        fc.recv_offset = 60; // consumed → proposed = 60 + 100 = 160
        let next = fc
            .should_emit_max_stream_data()
            .expect("threshold breached");
        assert_eq!(next, 160, "grant = recv_offset + initial_credit_recv");
    }

    // ---- MaxStreamsState ----

    #[test]
    fn max_streams_observe_peer_is_monotonic() {
        let mut state = MaxStreamsState::new(10, 10);
        state.observe_peer_max_streams(5); // smaller — ignored
        assert_eq!(state.peer_limit, 10);
        state.observe_peer_max_streams(20);
        assert_eq!(state.peer_limit, 20);
        state.observe_peer_max_streams(15); // smaller — ignored
        assert_eq!(state.peer_limit, 20);
    }

    #[test]
    fn max_streams_grant_local_is_monotonic() {
        let mut state = MaxStreamsState::new(10, 10);
        state.grant_local_max_streams(5);
        assert_eq!(state.local_limit, 10);
        state.grant_local_max_streams(20);
        assert_eq!(state.local_limit, 20);
    }

    #[test]
    fn max_streams_is_local_open_blocked_after_cap() {
        let mut state = MaxStreamsState::new(2, 100);
        assert!(!state.is_local_open_blocked());
        state.record_locally_opened();
        state.record_locally_opened();
        assert!(state.is_local_open_blocked());
    }

    #[test]
    fn max_streams_peer_would_exceed_local_limit() {
        let mut state = MaxStreamsState::new(100, 2);
        assert!(!state.peer_would_exceed_local_limit());
        state.record_peer_opened();
        state.record_peer_opened();
        assert!(state.peer_would_exceed_local_limit());
    }

    // ---- should_emit_max_streams: open→close→emit cycle ----

    /// Worked example: window=8 (table cap). Open 4 streams, close 1 →
    /// remaining drops to 4 (== threshold) and peer_closed bumps the target
    /// past local_limit → first MAX_STREAMS fires. Grant raises local_limit,
    /// subsequent call returns None (monotonicity / no-op).
    #[test]
    fn should_emit_max_streams_fires_after_close_grows_window() {
        let mut state = MaxStreamsState::new(100, 8);
        // Open 4 streams — remaining = 8-4 = 4 == threshold(4). target = 0+8=8 ==
        // local_limit → NOT emitted (no close yet; target == local_limit not ">").
        for _ in 0..4 {
            state.record_peer_opened();
        }
        assert!(
            state.should_emit_max_streams().is_none(),
            "no emit before any close"
        );
        // Close 1 — target = 1+8 = 9 > 8 → emit 9.
        state.record_peer_closed();
        let maximum = state
            .should_emit_max_streams()
            .expect("should emit after close grows target past limit");
        assert_eq!(maximum, 9, "new local_limit = peer_closed(1) + window(8)");
        // Grant and verify subsequent call returns None (grant is monotonic).
        state.grant_local_max_streams(maximum);
        assert_eq!(state.local_limit, 9);
        assert!(
            state.should_emit_max_streams().is_none(),
            "no re-emit after grant when remaining is still healthy"
        );
    }

    /// Concurrent-cap bound: the emitted `maximum` is always `peer_closed +
    /// window`. Opening many streams then closing many must NEVER advertise a
    /// limit higher than `peer_closed + window`, bounding concurrent open
    /// streams at `window`.
    #[test]
    fn should_emit_max_streams_concurrent_cap_stays_bounded_by_window() {
        let window = 8u64;
        let mut state = MaxStreamsState::new(1000, window);
        // Simulate 20 serial opens + closes (first 8 open, then 4 close,
        // triggering the first emission, then 4 more open, etc.).
        for round in 0u64..3 {
            for _ in 0..window {
                state.record_peer_opened();
            }
            for _ in 0..window {
                state.record_peer_closed();
            }
            if let Some(maximum) = state.should_emit_max_streams() {
                // maximum MUST equal peer_closed + window — never more.
                assert_eq!(
                    maximum,
                    state.peer_closed.saturating_add(window),
                    "round {round}: emitted maximum must equal peer_closed + window"
                );
                assert!(
                    maximum > state.local_limit,
                    "round {round}: only emit when target > current limit"
                );
                state.grant_local_max_streams(maximum);
            }
        }
        // After grants, local_limit must be == peer_closed + window.
        assert_eq!(state.local_limit, state.peer_closed.saturating_add(window));
    }

    /// Monotonicity: `should_emit_max_streams` must never return a value <=
    /// the already-granted `local_limit`. Without this guard, a retransmit of
    /// an old value would confuse the peer (RFC 9000 §19.11 monotonicity).
    #[test]
    fn should_emit_max_streams_monotone_grant_not_regressed() {
        let mut state = MaxStreamsState::new(1000, 8);
        // Drive to first emission.
        for _ in 0..5 {
            state.record_peer_opened();
        }
        for _ in 0..2 {
            state.record_peer_closed();
        }
        let first = state.should_emit_max_streams().expect("first emit");
        state.grant_local_max_streams(first);
        // Drive more opens/closes until another emission.
        for _ in 0..6 {
            state.record_peer_opened();
        }
        for _ in 0..4 {
            state.record_peer_closed();
        }
        let second = state.should_emit_max_streams().expect("second emit");
        assert!(
            second > first,
            "second emission must be strictly greater than first"
        );
        state.grant_local_max_streams(second);
        // A stale smaller value applied via grant must not lower local_limit.
        state.grant_local_max_streams(first);
        assert_eq!(
            state.local_limit, second,
            "grant_local_max_streams is monotone — stale smaller value dropped"
        );
    }

    /// Empty / zero-window guard: a freshly constructed state with zero
    /// initial_local_limit must not emit (prevents divide-by-zero and
    /// spurious MAX_STREAMS on connections that advertise 0 streams).
    #[test]
    fn should_emit_max_streams_no_emit_when_window_is_zero() {
        let mut state = MaxStreamsState::new(100, 0);
        state.record_peer_opened();
        state.record_peer_closed();
        assert!(
            state.should_emit_max_streams().is_none(),
            "zero window → threshold is 0 → remaining (0) <= threshold (0) but target (0+0=0) not > local_limit (0)"
        );
    }
}
