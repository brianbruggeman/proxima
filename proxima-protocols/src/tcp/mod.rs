//! Sans-IO TCP connection state machine (RFC 793 control FSM).
//!
//! A pure transition: feed a user call or an incoming segment's control bits,
//! get back the next state and the control action the caller must emit. No
//! I/O, no allocation, and deliberately no sequence-number, window, or
//! retransmission logic — that is the data-path layer, which consumes this.

pub mod congestion;
pub mod connection;
pub mod data_path;
pub mod reassembly;
pub mod retx;
pub mod rtt;
pub mod seq;
pub mod time;
pub mod window;

pub use congestion::{Reno, TcpCongestionControl};
pub use connection::{Action, Connection, Input, Segment, State};
pub use data_path::{DataPath, SegmentOutput};
pub use reassembly::{InsertOutcome, Reassembler};
pub use retx::{MaxRetransmit, RetransmitDecision, RetxQueue, RetxSegment};
pub use rtt::RtoEstimator;
pub use seq::{SeqNum, segment_acceptable};
pub use time::{Duration, Instant};
pub use window::{AckOutcome, WindowTracker};
