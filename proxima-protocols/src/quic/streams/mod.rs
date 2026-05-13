//! Stream multiplexing + flow control per [RFC 9000 §2-§4 + §19.8].
//!
//! Per-stream and connection-level flow control, the per-direction
//! StreamTable, ReassemblyQueue for out-of-order STREAM frames,
//! MAX_STREAMS state, and the per-stream send/recv state machines.
//!
//! [RFC 9000 §2-§4 + §19.8]: https://www.rfc-editor.org/rfc/rfc9000#section-2

pub mod flow;
pub mod id;
pub mod reassembly;
pub mod state;
pub mod table;

pub use flow::{ConnectionFlowControl, MaxStreamsState, StreamFlowControl};
pub use id::{StreamDirection, StreamId};
pub use reassembly::{Fragment, InsertOutcome, ReassemblyQueue};
pub use state::{RecvState, RecvStateError, STREAM_RECV_INLINE, STREAM_SEND_INLINE, SendState};
pub use table::{Stream, StreamTable, StreamTableError};
