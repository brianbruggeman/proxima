//! HTTP/2 stream state machine + per-stream flow control (RFC 7540 §5).
//!
//! Each open stream on a connection has a lifecycle:
//!
//! ```text
//!     Idle  --recv HEADERS--> Open  --recv END_STREAM--> HalfClosedRemote
//!                              |                              |
//!                       send END_STREAM                 send END_STREAM
//!                              |                              |
//!                              v                              v
//!                       HalfClosedLocal --recv END_STREAM-> Closed
//!                              |
//!                       recv END_STREAM
//!                              |
//!                              v
//!                            Closed
//!
//!     RST_STREAM in either direction terminates immediately -> Closed.
//! ```
//!
//! Server perspective only: we never initiate via `send HEADERS` first,
//! and we don't issue `PUSH_PROMISE` yet (so Reserved states are absent).
//! When push promises land, they'll mark the *promised* stream as
//! `ReservedRemote` / `ReservedLocal` while the parent stream's state
//! is unaffected.
//!
//! ## Flow control
//!
//! Per RFC 7540 §5.2: each stream has independent **send** and **recv**
//! windows tracked alongside connection-level windows (handled by
//! [`super::flow_control`]). DATA frames consume sender window; the
//! receiver replenishes via `WINDOW_UPDATE`. Windows are signed `i64`
//! internally so a `SETTINGS_INITIAL_WINDOW_SIZE` delta that
//! retroactively shifts open streams' send-windows can legally drive
//! them negative — that's not an error per §6.9.2; we just stop
//! sending until WINDOW_UPDATE recovers.

use thiserror::Error;

/// RFC §5.2.1 — initial flow-control window for a newly created stream.
pub const DEFAULT_INITIAL_WINDOW_SIZE: u32 = 65_535;

/// RFC §6.9.1 — windows cannot exceed `2^31 - 1`. Crossing this is a
/// `FLOW_CONTROL_ERROR` on the stream (or connection for stream 0).
pub const MAX_WINDOW_SIZE: i64 = (1i64 << 31) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Created, not yet activated. Server streams start `Idle` and
    /// transition out on `recv HEADERS`.
    Idle,
    /// Active in both directions.
    Open,
    /// Peer signaled END_STREAM; we may still send response frames.
    HalfClosedRemote,
    /// We signaled END_STREAM; peer may still send (rare — typically
    /// pre-emptive close or HEAD-style). Reads still legal until
    /// peer END_STREAM or RST.
    HalfClosedLocal,
    /// Terminal — no further frames in either direction.
    Closed,
}

/// State-transition + flow-control events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEvent {
    RecvHeaders { end_stream: bool },
    RecvData { end_stream: bool, len: u32 },
    RecvRst,
    SendHeaders { end_stream: bool },
    SendData { end_stream: bool, len: u32 },
    SendRst,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StreamError {
    #[error("frame {event:?} illegal in state {state:?}")]
    InvalidState {
        event: StreamEvent,
        state: StreamState,
    },
    #[error("send blocked: need {needed} bytes but only {available} window")]
    SendWindowExceeded { needed: u32, available: i64 },
    #[error("recv overflowed window: {len} bytes against {available} available")]
    RecvWindowExceeded { len: u32, available: i64 },
    #[error("window grew beyond 2^31-1: current={current} increment={increment}")]
    WindowOverflow { current: i64, increment: u32 },
}

#[derive(Debug)]
pub struct Stream {
    pub id: u32,
    state: StreamState,
    /// How many DATA bytes we are still allowed to send (peer-granted).
    send_window: i64,
    /// How many DATA bytes the peer is allowed to send before we
    /// must replenish via WINDOW_UPDATE.
    recv_window: i64,
}

impl Stream {
    /// New `Idle` stream with windows at the default initial size.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self::with_windows(id, DEFAULT_INITIAL_WINDOW_SIZE, DEFAULT_INITIAL_WINDOW_SIZE)
    }

    /// New stream with explicit initial windows. Used when the
    /// connection's `SETTINGS_INITIAL_WINDOW_SIZE` differs from the
    /// spec default.
    #[must_use]
    pub fn with_windows(id: u32, send_initial: u32, recv_initial: u32) -> Self {
        Self {
            id,
            state: StreamState::Idle,
            send_window: i64::from(send_initial),
            recv_window: i64::from(recv_initial),
        }
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Remaining send-window credit. Negative is legal post-SETTINGS
    /// resize; the sender must wait for WINDOW_UPDATE to go positive.
    #[must_use]
    pub fn send_window(&self) -> i64 {
        self.send_window
    }

    /// Remaining recv-window credit.
    #[must_use]
    pub fn recv_window(&self) -> i64 {
        self.recv_window
    }

    /// `true` if no more frames may flow in either direction.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        matches!(self.state, StreamState::Closed)
    }

    /// Apply an event. Returns `Ok(new_state)` or `Err` describing
    /// what the peer (or our own send path) violated.
    pub fn on_event(&mut self, event: StreamEvent) -> Result<StreamState, StreamError> {
        match event {
            StreamEvent::RecvData { len, .. } if i64::from(len) > self.recv_window => {
                return Err(StreamError::RecvWindowExceeded {
                    len,
                    available: self.recv_window,
                });
            }
            StreamEvent::SendData { len, .. } if i64::from(len) > self.send_window => {
                return Err(StreamError::SendWindowExceeded {
                    needed: len,
                    available: self.send_window,
                });
            }
            _ => {}
        }
        let next = transition(self.state, event)?;
        match event {
            StreamEvent::RecvData { len, .. } => self.recv_window -= i64::from(len),
            StreamEvent::SendData { len, .. } => self.send_window -= i64::from(len),
            _ => {}
        }
        self.state = next;
        Ok(next)
    }

    /// Apply a WINDOW_UPDATE we received from the peer (grows our send
    /// credit). Errors if it would push the window over `2^31 - 1`.
    pub fn grant_send_window(&mut self, increment: u32) -> Result<i64, StreamError> {
        let proposed = self.send_window.saturating_add(i64::from(increment));
        if proposed > MAX_WINDOW_SIZE {
            return Err(StreamError::WindowOverflow {
                current: self.send_window,
                increment,
            });
        }
        self.send_window = proposed;
        Ok(self.send_window)
    }

    /// Apply a WINDOW_UPDATE we're emitting (grows recv credit we
    /// promise the peer).
    pub fn grant_recv_window(&mut self, increment: u32) -> Result<i64, StreamError> {
        let proposed = self.recv_window.saturating_add(i64::from(increment));
        if proposed > MAX_WINDOW_SIZE {
            return Err(StreamError::WindowOverflow {
                current: self.recv_window,
                increment,
            });
        }
        self.recv_window = proposed;
        Ok(self.recv_window)
    }

    /// Adjust send-window by the delta from a SETTINGS_INITIAL_WINDOW_SIZE
    /// change (RFC §6.9.2). Negative deltas are legal and may drive the
    /// window negative; positive deltas that would overflow are an error.
    pub fn apply_initial_window_delta(&mut self, delta: i32) -> Result<i64, StreamError> {
        let proposed = self.send_window + i64::from(delta);
        if proposed > MAX_WINDOW_SIZE {
            return Err(StreamError::WindowOverflow {
                current: self.send_window,
                increment: delta.unsigned_abs(),
            });
        }
        self.send_window = proposed;
        Ok(self.send_window)
    }
}

fn transition(state: StreamState, event: StreamEvent) -> Result<StreamState, StreamError> {
    use StreamEvent as E;
    use StreamState as S;
    let next = match (state, event) {
        (_, E::RecvRst | E::SendRst) => S::Closed,
        (S::Idle, E::RecvHeaders { end_stream: true }) => S::HalfClosedRemote,
        (S::Idle, E::RecvHeaders { end_stream: false }) => S::Open,
        // RFC 7540 §5.1: the CLIENT opens a stream by SENDING HEADERS from Idle —
        // the mirror of the server's recv-headers transitions above. `end_stream`
        // (a bodyless request, e.g. GET) half-closes our send side immediately.
        (S::Idle, E::SendHeaders { end_stream: true }) => S::HalfClosedLocal,
        (S::Idle, E::SendHeaders { end_stream: false }) => S::Open,
        (
            S::Open,
            E::RecvData {
                end_stream: true, ..
            }
            | E::RecvHeaders { end_stream: true },
        ) => S::HalfClosedRemote,
        (
            S::Open,
            E::RecvData {
                end_stream: false, ..
            }
            | E::RecvHeaders { end_stream: false },
        ) => S::Open,
        (
            S::Open,
            E::SendData {
                end_stream: true, ..
            }
            | E::SendHeaders { end_stream: true },
        ) => S::HalfClosedLocal,
        (
            S::Open,
            E::SendData {
                end_stream: false, ..
            }
            | E::SendHeaders { end_stream: false },
        ) => S::Open,
        (
            S::HalfClosedRemote,
            E::SendData {
                end_stream: true, ..
            }
            | E::SendHeaders { end_stream: true },
        ) => S::Closed,
        (
            S::HalfClosedRemote,
            E::SendData {
                end_stream: false, ..
            }
            | E::SendHeaders { end_stream: false },
        ) => S::HalfClosedRemote,
        (
            S::HalfClosedLocal,
            E::RecvData {
                end_stream: true, ..
            }
            | E::RecvHeaders { end_stream: true },
        ) => S::Closed,
        (
            S::HalfClosedLocal,
            E::RecvData {
                end_stream: false, ..
            }
            | E::RecvHeaders { end_stream: false },
        ) => S::HalfClosedLocal,
        _ => return Err(StreamError::InvalidState { event, state }),
    };
    Ok(next)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn idle_recv_headers_no_end_opens_stream() {
        let mut stream = Stream::new(1);
        let next = stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        assert_eq!(next, StreamState::Open);
    }

    // RFC 7540 §5.1 client stream lifecycle (the worked example for the client
    // role): Idle --send HEADERS--> Open --send END_STREAM DATA--> HalfClosedLocal
    // --recv response HEADERS--> HalfClosedLocal --recv END_STREAM DATA--> Closed.
    #[test]
    fn client_stream_lifecycle_open_send_recv_close() {
        let mut stream = Stream::new(1);
        assert_eq!(stream.state(), StreamState::Idle);
        assert_eq!(
            stream
                .on_event(StreamEvent::SendHeaders { end_stream: false })
                .unwrap(),
            StreamState::Open
        );
        assert_eq!(
            stream
                .on_event(StreamEvent::SendData {
                    end_stream: true,
                    len: 5
                })
                .unwrap(),
            StreamState::HalfClosedLocal
        );
        assert_eq!(
            stream
                .on_event(StreamEvent::RecvHeaders { end_stream: false })
                .unwrap(),
            StreamState::HalfClosedLocal
        );
        assert_eq!(
            stream
                .on_event(StreamEvent::RecvData {
                    end_stream: true,
                    len: 3
                })
                .unwrap(),
            StreamState::Closed
        );
        assert!(stream.is_closed());
    }

    // bodyless request (GET / grpc trailers-only): Idle --send HEADERS END_STREAM-->
    // HalfClosedLocal --recv response HEADERS END_STREAM--> Closed.
    #[test]
    fn client_bodyless_request_lifecycle() {
        let mut stream = Stream::new(1);
        assert_eq!(
            stream
                .on_event(StreamEvent::SendHeaders { end_stream: true })
                .unwrap(),
            StreamState::HalfClosedLocal
        );
        assert_eq!(
            stream
                .on_event(StreamEvent::RecvHeaders { end_stream: true })
                .unwrap(),
            StreamState::Closed
        );
    }

    #[test]
    fn idle_recv_headers_with_end_half_closes_remote() {
        let mut stream = Stream::new(1);
        let next = stream
            .on_event(StreamEvent::RecvHeaders { end_stream: true })
            .unwrap();
        assert_eq!(next, StreamState::HalfClosedRemote);
    }

    #[test]
    fn open_send_end_stream_half_closes_local() {
        let mut stream = Stream::new(1);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        let next = stream
            .on_event(StreamEvent::SendHeaders { end_stream: true })
            .unwrap();
        assert_eq!(next, StreamState::HalfClosedLocal);
    }

    #[test]
    fn half_closed_remote_send_end_closes() {
        let mut stream = Stream::new(1);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: true })
            .unwrap();
        assert_eq!(stream.state(), StreamState::HalfClosedRemote);
        let next = stream
            .on_event(StreamEvent::SendData {
                end_stream: true,
                len: 0,
            })
            .unwrap();
        assert_eq!(next, StreamState::Closed);
        assert!(stream.is_closed());
    }

    #[test]
    fn rst_from_either_side_closes() {
        let mut stream = Stream::new(1);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        stream.on_event(StreamEvent::RecvRst).unwrap();
        assert!(stream.is_closed());

        let mut stream = Stream::new(3);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        stream.on_event(StreamEvent::SendRst).unwrap();
        assert!(stream.is_closed());
    }

    // DATA from Idle is illegal in BOTH roles: a stream must be opened by HEADERS
    // first (RFC §5.1). (Sending HEADERS from Idle is now the legal client-open
    // transition, covered by `client_stream_lifecycle_open_send_recv_close`.)
    #[test]
    fn invalid_state_send_data_in_idle_errors() {
        let mut stream = Stream::new(1);
        let err = stream
            .on_event(StreamEvent::SendData {
                end_stream: false,
                len: 0,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            StreamError::InvalidState {
                state: StreamState::Idle,
                ..
            }
        ));
    }

    #[test]
    fn invalid_state_recv_data_after_remote_end_errors() {
        let mut stream = Stream::new(1);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: true })
            .unwrap();
        let err = stream
            .on_event(StreamEvent::RecvData {
                end_stream: false,
                len: 10,
            })
            .unwrap_err();
        assert!(matches!(err, StreamError::InvalidState { .. }));
    }

    #[test]
    fn send_data_consumes_window() {
        let mut stream = Stream::with_windows(1, 100, 100);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        stream
            .on_event(StreamEvent::SendData {
                end_stream: false,
                len: 40,
            })
            .unwrap();
        assert_eq!(stream.send_window(), 60);
        stream
            .on_event(StreamEvent::SendData {
                end_stream: false,
                len: 60,
            })
            .unwrap();
        assert_eq!(stream.send_window(), 0);
    }

    #[test]
    fn send_data_over_window_errors_and_preserves_state() {
        let mut stream = Stream::with_windows(1, 10, 65535);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        let err = stream
            .on_event(StreamEvent::SendData {
                end_stream: false,
                len: 20,
            })
            .unwrap_err();
        assert!(matches!(err, StreamError::SendWindowExceeded { .. }));
        assert_eq!(stream.send_window(), 10);
        assert_eq!(stream.state(), StreamState::Open);
    }

    #[test]
    fn recv_data_consumes_window() {
        let mut stream = Stream::with_windows(1, 100, 100);
        stream
            .on_event(StreamEvent::RecvHeaders { end_stream: false })
            .unwrap();
        stream
            .on_event(StreamEvent::RecvData {
                end_stream: false,
                len: 30,
            })
            .unwrap();
        assert_eq!(stream.recv_window(), 70);
    }

    #[test]
    fn grant_send_window_accumulates() {
        let mut stream = Stream::with_windows(1, 100, 100);
        stream.grant_send_window(50).unwrap();
        assert_eq!(stream.send_window(), 150);
        stream.grant_send_window(200).unwrap();
        assert_eq!(stream.send_window(), 350);
    }

    #[test]
    fn grant_window_overflow_errors() {
        let mut stream = Stream::with_windows(1, u32::MAX, 0);
        let err = stream.grant_send_window(1).unwrap_err();
        assert!(matches!(err, StreamError::WindowOverflow { .. }));
    }

    #[test]
    fn initial_window_delta_can_go_negative() {
        let mut stream = Stream::with_windows(1, 100, 100);
        stream.apply_initial_window_delta(-200).unwrap();
        assert_eq!(stream.send_window(), -100);
        // Recovery via WINDOW_UPDATE.
        stream.grant_send_window(300).unwrap();
        assert_eq!(stream.send_window(), 200);
    }

    #[test]
    fn initial_window_delta_overflow_errors() {
        let mut stream = Stream::with_windows(1, u32::MAX - 1, 0);
        let err = stream.apply_initial_window_delta(1000).unwrap_err();
        assert!(matches!(err, StreamError::WindowOverflow { .. }));
    }
}
