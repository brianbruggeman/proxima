//! Connection-level stream table + connection-level flow control
//! (RFC 7540 §5.1.1, §5.2.1).
//!
//! A single HTTP/2 connection multiplexes many streams. The
//! [`StreamTable`] owns those streams keyed by stream ID and enforces
//! the protocol's identifier discipline:
//!
//! - Client-initiated streams use **odd** IDs (1, 3, 5, ...).
//! - Server-initiated streams (push promise) use **even** IDs.
//! - Stream IDs are **monotonically increasing** within an origin
//!   half — a HEADERS for ID `N` implicitly closes streams `< N`
//!   that the peer never opened (RFC §5.1.1).
//!
//! Connection-level flow control sits beside per-stream windows:
//! DATA frames count against **both** the originating stream window
//! and the connection window. WINDOW_UPDATE on stream 0 increments
//! the connection window. The connection window's defaults and bounds
//! match per-stream:
//! [`DEFAULT_INITIAL_WINDOW_SIZE`](super::stream::DEFAULT_INITIAL_WINDOW_SIZE),
//! [`MAX_WINDOW_SIZE`](super::stream::MAX_WINDOW_SIZE).

use alloc::collections::BTreeMap;

use thiserror::Error;

use crate::http2_codec::stream::{
    DEFAULT_INITIAL_WINDOW_SIZE, MAX_WINDOW_SIZE, Stream, StreamError, StreamState,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TableError {
    #[error("stream id 0 is reserved for connection-level signals")]
    InvalidStreamIdZero,
    #[error("client streams must use odd ids; got {0}")]
    ClientIdNotOdd(u32),
    #[error("stream id {got} not greater than highest seen {highest}")]
    IdNotMonotonic { got: u32, highest: u32 },
    #[error("connection-level recv window exceeded: {len} bytes against {available} available")]
    ConnRecvWindowExceeded { len: u32, available: i64 },
    #[error("connection-level send window exceeded: {len} bytes against {available} available")]
    ConnSendWindowExceeded { len: u32, available: i64 },
    #[error("connection window grew beyond 2^31-1: current={current} increment={increment}")]
    ConnWindowOverflow { current: i64, increment: u32 },
    #[error("stream {0}: {1}")]
    Stream(u32, StreamError),
}

/// Connection-wide DATA flow-control accounting. One window each for
/// inbound (we owe peer credit via WINDOW_UPDATE) and outbound (peer
/// owes us credit). Mirrors per-stream [`Stream`] semantics minus the
/// state machine.
#[derive(Debug)]
pub struct ConnectionFlow {
    send_window: i64,
    recv_window: i64,
}

impl Default for ConnectionFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionFlow {
    #[must_use]
    pub fn new() -> Self {
        Self {
            send_window: i64::from(DEFAULT_INITIAL_WINDOW_SIZE),
            recv_window: i64::from(DEFAULT_INITIAL_WINDOW_SIZE),
        }
    }

    #[must_use]
    pub fn send_window(&self) -> i64 {
        self.send_window
    }

    #[must_use]
    pub fn recv_window(&self) -> i64 {
        self.recv_window
    }

    pub fn consume_send(&mut self, len: u32) -> Result<i64, TableError> {
        if i64::from(len) > self.send_window {
            return Err(TableError::ConnSendWindowExceeded {
                len,
                available: self.send_window,
            });
        }
        self.send_window -= i64::from(len);
        Ok(self.send_window)
    }

    pub fn consume_recv(&mut self, len: u32) -> Result<i64, TableError> {
        if i64::from(len) > self.recv_window {
            return Err(TableError::ConnRecvWindowExceeded {
                len,
                available: self.recv_window,
            });
        }
        self.recv_window -= i64::from(len);
        Ok(self.recv_window)
    }

    pub fn grant_send(&mut self, increment: u32) -> Result<i64, TableError> {
        let proposed = self.send_window.saturating_add(i64::from(increment));
        if proposed > MAX_WINDOW_SIZE {
            return Err(TableError::ConnWindowOverflow {
                current: self.send_window,
                increment,
            });
        }
        self.send_window = proposed;
        Ok(self.send_window)
    }

    pub fn grant_recv(&mut self, increment: u32) -> Result<i64, TableError> {
        let proposed = self.recv_window.saturating_add(i64::from(increment));
        if proposed > MAX_WINDOW_SIZE {
            return Err(TableError::ConnWindowOverflow {
                current: self.recv_window,
                increment,
            });
        }
        self.recv_window = proposed;
        Ok(self.recv_window)
    }
}

/// Owns all open streams on a connection plus the connection-level
/// flow-control state.
#[derive(Debug)]
pub struct StreamTable {
    streams: BTreeMap<u32, Stream>,
    flow: ConnectionFlow,
    highest_client_id: u32,
    initial_send_window: u32,
    initial_recv_window: u32,
}

impl Default for StreamTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTable {
    #[must_use]
    pub fn new() -> Self {
        Self::with_initial_windows(DEFAULT_INITIAL_WINDOW_SIZE, DEFAULT_INITIAL_WINDOW_SIZE)
    }

    #[must_use]
    pub fn with_initial_windows(send_initial: u32, recv_initial: u32) -> Self {
        Self {
            streams: BTreeMap::new(),
            flow: ConnectionFlow::new(),
            highest_client_id: 0,
            initial_send_window: send_initial,
            initial_recv_window: recv_initial,
        }
    }

    /// Register a new client-initiated stream after validating its ID.
    /// Returns a mutable handle. Errors on id 0, even ids, or
    /// non-monotonic ids (RFC §5.1.1). Used by the SERVER role (the peer opened
    /// it); the CLIENT role uses [`open_local_stream`](Self::open_local_stream).
    pub fn accept_client_stream(&mut self, id: u32) -> Result<&mut Stream, TableError> {
        self.register_odd_stream(id)
    }

    /// Register a new LOCALLY-initiated stream (the CLIENT role) — the mirror of
    /// [`accept_client_stream`]. The client opens odd, monotonically-increasing
    /// ids (RFC §5.1.1); this inserts the `Idle` stream so a subsequent
    /// `SendHeaders` can advance it. Same id validation, opposite initiator.
    pub fn open_local_stream(&mut self, id: u32) -> Result<&mut Stream, TableError> {
        self.register_odd_stream(id)
    }

    /// Shared odd-stream registration: id 0 / even / non-monotonic are errors
    /// (RFC §5.1.1); inserts an `Idle` stream at the connection's initial windows.
    fn register_odd_stream(&mut self, id: u32) -> Result<&mut Stream, TableError> {
        if id == 0 {
            return Err(TableError::InvalidStreamIdZero);
        }
        if id & 1 == 0 {
            return Err(TableError::ClientIdNotOdd(id));
        }
        if id <= self.highest_client_id {
            return Err(TableError::IdNotMonotonic {
                got: id,
                highest: self.highest_client_id,
            });
        }
        self.highest_client_id = id;
        let stream = Stream::with_windows(id, self.initial_send_window, self.initial_recv_window);
        Ok(self.streams.entry(id).or_insert(stream))
    }

    #[must_use]
    pub fn get(&self, id: u32) -> Option<&Stream> {
        self.streams.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Stream> {
        self.streams.get_mut(&id)
    }

    /// Number of streams currently tracked (open + half-closed +
    /// closed-but-not-yet-GC'd).
    #[must_use]
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Drop every stream in `Closed` state. Returns how many were
    /// removed.
    pub fn gc_closed(&mut self) -> usize {
        let before = self.streams.len();
        self.streams
            .retain(|_, stream| !matches!(stream.state(), StreamState::Closed));
        before - self.streams.len()
    }

    /// Apply a `SETTINGS_INITIAL_WINDOW_SIZE` change to every open
    /// stream's send-window per RFC §6.9.2.
    pub fn apply_initial_window_change(&mut self, new_size: u32) -> Result<(), TableError> {
        let delta = i64::from(new_size) - i64::from(self.initial_send_window);
        let delta_i32 = i32::try_from(delta).map_err(|_| TableError::ConnWindowOverflow {
            current: i64::from(self.initial_send_window),
            increment: new_size,
        })?;
        for (id, stream) in &mut self.streams {
            stream
                .apply_initial_window_delta(delta_i32)
                .map_err(|err| TableError::Stream(*id, err))?;
        }
        self.initial_send_window = new_size;
        Ok(())
    }

    /// Highest client-initiated stream id we've accepted. Used as
    /// the `last_stream_id` field of a GOAWAY frame so peers know
    /// which streams we've committed to processing.
    #[must_use]
    pub fn last_processed_id(&self) -> u32 {
        self.highest_client_id
    }

    /// Number of streams in active (non-Closed) states. Used to
    /// enforce SETTINGS_MAX_CONCURRENT_STREAMS (RFC §5.1.2).
    #[must_use]
    pub fn count_active(&self) -> usize {
        self.streams
            .values()
            .filter(|stream| !stream.is_closed())
            .count()
    }

    /// Connection flow-control handle (immutable).
    #[must_use]
    pub fn flow(&self) -> &ConnectionFlow {
        &self.flow
    }

    /// Connection flow-control handle (mutable).
    pub fn flow_mut(&mut self) -> &mut ConnectionFlow {
        &mut self.flow
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::http2_codec::stream::{StreamError, StreamEvent};

    #[test]
    fn connection_flow_defaults_to_initial_window() {
        let flow = ConnectionFlow::new();
        assert_eq!(flow.send_window(), 65_535);
        assert_eq!(flow.recv_window(), 65_535);
    }

    #[test]
    fn consume_send_drops_window() {
        let mut flow = ConnectionFlow::new();
        flow.consume_send(1024).unwrap();
        assert_eq!(flow.send_window(), 65_535 - 1024);
    }

    #[test]
    fn consume_send_over_window_errors() {
        let mut flow = ConnectionFlow::new();
        let err = flow.consume_send(100_000).unwrap_err();
        assert!(matches!(err, TableError::ConnSendWindowExceeded { .. }));
        assert_eq!(flow.send_window(), 65_535);
    }

    #[test]
    fn grant_recv_overflow_errors() {
        let mut flow = ConnectionFlow::new();
        flow.grant_recv(u32::MAX - 65_535).unwrap_err();
    }

    #[test]
    fn accept_client_stream_succeeds_for_odd_ids() {
        let mut table = StreamTable::new();
        let stream = table.accept_client_stream(1).unwrap();
        assert_eq!(stream.id, 1);
        assert_eq!(stream.state(), StreamState::Idle);
    }

    #[test]
    fn even_id_rejected() {
        let mut table = StreamTable::new();
        let err = table.accept_client_stream(2).unwrap_err();
        assert!(matches!(err, TableError::ClientIdNotOdd(2)));
    }

    #[test]
    fn zero_id_rejected() {
        let mut table = StreamTable::new();
        let err = table.accept_client_stream(0).unwrap_err();
        assert!(matches!(err, TableError::InvalidStreamIdZero));
    }

    #[test]
    fn non_monotonic_id_rejected() {
        let mut table = StreamTable::new();
        table.accept_client_stream(3).unwrap();
        let err = table.accept_client_stream(1).unwrap_err();
        assert!(matches!(
            err,
            TableError::IdNotMonotonic { got: 1, highest: 3 }
        ));
        let err = table.accept_client_stream(3).unwrap_err();
        assert!(matches!(
            err,
            TableError::IdNotMonotonic { got: 3, highest: 3 }
        ));
    }

    #[test]
    fn gc_closed_removes_only_closed_streams() {
        let mut table = StreamTable::new();
        table.accept_client_stream(1).unwrap();
        table.accept_client_stream(3).unwrap();
        table.accept_client_stream(5).unwrap();
        // Close stream 3 via RST.
        table
            .get_mut(3)
            .unwrap()
            .on_event(StreamEvent::RecvRst)
            .unwrap();
        assert_eq!(table.len(), 3);
        let removed = table.gc_closed();
        assert_eq!(removed, 1);
        assert_eq!(table.len(), 2);
        assert!(table.get(3).is_none());
        assert!(table.get(1).is_some());
        assert!(table.get(5).is_some());
    }

    #[test]
    fn apply_initial_window_change_shifts_open_streams() {
        let mut table = StreamTable::with_initial_windows(100, 100);
        table.accept_client_stream(1).unwrap();
        table.accept_client_stream(3).unwrap();
        // Bump initial to 500 → +400 delta on every send_window.
        table.apply_initial_window_change(500).unwrap();
        assert_eq!(table.get(1).unwrap().send_window(), 500);
        assert_eq!(table.get(3).unwrap().send_window(), 500);
        // Newly accepted streams use the new initial.
        table.accept_client_stream(5).unwrap();
        assert_eq!(table.get(5).unwrap().send_window(), 500);
    }

    #[test]
    fn apply_initial_window_change_can_go_negative() {
        let mut table = StreamTable::with_initial_windows(500, 500);
        let stream = table.accept_client_stream(1).unwrap();
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        stream
            .on_event(StreamEvent::SendData {
                end_stream: false,
                len: 100,
            })
            .unwrap();
        assert_eq!(stream.send_window(), 400);
        // Shrink initial to 50 → -450 delta. send_window goes to -50.
        table.apply_initial_window_change(50).unwrap();
        assert_eq!(table.get(1).unwrap().send_window(), -50);
    }

    #[test]
    fn flow_accessors_mutate_through_handle() {
        let mut table = StreamTable::new();
        table.flow_mut().consume_send(2000).unwrap();
        assert_eq!(table.flow().send_window(), 65_535 - 2000);
    }

    #[test]
    fn apply_initial_window_change_overflow_wraps_stream_error() {
        let mut table = StreamTable::with_initial_windows(1, 100);
        let stream = table.accept_client_stream(1).unwrap();
        // Bring send_window all the way up to the legal max.
        stream
            .grant_send_window((MAX_WINDOW_SIZE - 1) as u32)
            .unwrap();
        // Any positive delta on the initial window now overflows.
        let err = table.apply_initial_window_change(2).unwrap_err();
        assert!(matches!(
            err,
            TableError::Stream(1, StreamError::WindowOverflow { .. })
        ));
    }
}
