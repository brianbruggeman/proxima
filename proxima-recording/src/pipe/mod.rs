//! Pipe-flavored recording layer: `LiveCaptureContext` (concrete
//! impl of `proxima-pipe::CaptureContext`), `BoundedRecordingSink`
//! with telemetry-counter back-pressure, and `Causal` (the record /
//! replay Pipe wrapper with causal index).
//!
//! Pure format pieces (event model, sink/source traits, bin + JSONL,
//! factory) live at the crate root (folded from `proxima-recording-core`).
//! Format-only consumers use the crate's default `std`/`alloc` surface;
//! pipe-flavored consumers additionally enable the `pipe` feature.
//!
//! Tier: std. Everything here holds a file handle or an OS thread, so the
//! `pipe` feature declares `std` rather than gating each item on it. The
//! concurrency is tokio-free: `proxima_primitives::sync::Notify` for the
//! drain handshake, `crossbeam-queue` for the lock-free ring, and
//! `futures::executor::block_on` on one dedicated thread per sink.

pub mod accumulate;
pub mod cap;
pub mod capture;
pub mod causality;
#[cfg(feature = "pipe-config")]
pub mod config;
pub mod dest;
pub mod event_sink;
pub mod fanout;
pub mod lazy;
pub mod log_pipe;
pub mod replay;
pub mod terminal_signal;

pub use accumulate::{AccumulatingSink, DEFAULT_BATCH_EVENTS};
pub use cap::{BoundedRecordingSink, DropReason, FailMode, RECORD_DROP_METRIC};
pub use capture::LiveCaptureContext;
pub use causality::{ByteRange, Causal, CausalEdge, CausalIndex};
#[cfg(feature = "pipe-config")]
pub use config::{RecorderConfig, RecordingConfig, SinkConfig};
pub use dest::{FormatKind, SinkSpec};
pub use event_sink::{AppendFuture, DynRecordingSink, EventTap, RecordingSink};
pub use fanout::FanOut;
pub use lazy::{DeferredRuntime, LazyFanOut, deferred_runtime};
pub use log_pipe::{AppendAck, AppendLog, ReplayChunk, ReplayLog};
pub use replay::{ReplayMode, TimedReplay};
pub use terminal_signal::TerminalSignal;
