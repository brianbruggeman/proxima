//! Pipe-flavored recording substrate: `LiveCaptureContext` (concrete
//! impl of `proxima-pipe::CaptureContext`), `BoundedRecordingSink`
//! with telemetry-counter back-pressure, and `Causal` (the record /
//! replay Pipe wrapper with causal index).
//!
//! Pure format pieces (event model, sink/source traits, bin + JSONL,
//! factory) live at the crate root (folded from `proxima-recording-core`).
//! Format-only consumers use the crate's default `std`/`alloc` surface;
//! pipe-flavored consumers additionally enable the `pipe` feature.
//!
//! Tier: real content is std-only. Recording sinks use tokio::sync +
//! crossbeam-queue + std::time. Enabling `pipe` without `std` compiles this
//! module as a marker (no public items) — same tier behaviour the former
//! `proxima-recording-pipe` crate had.

#[cfg(feature = "std")]
pub mod accumulate;
#[cfg(feature = "std")]
pub mod cap;
#[cfg(feature = "std")]
pub mod capture;
#[cfg(feature = "std")]
pub mod causality;
#[cfg(feature = "pipe-config")]
pub mod config;
#[cfg(feature = "std")]
pub mod dest;
#[cfg(feature = "std")]
pub mod event_sink;
#[cfg(feature = "std")]
pub mod fanout;
#[cfg(feature = "std")]
pub mod lazy;
#[cfg(feature = "std")]
pub mod log_pipe;
#[cfg(feature = "std")]
pub mod replay;
#[cfg(feature = "std")]
pub mod terminal_signal;

#[cfg(feature = "std")]
pub use accumulate::{AccumulatingSink, DEFAULT_BATCH_EVENTS};
#[cfg(feature = "std")]
pub use cap::{BoundedRecordingSink, DropReason, FailMode, RECORD_DROP_METRIC};
#[cfg(feature = "std")]
pub use capture::LiveCaptureContext;
#[cfg(feature = "std")]
pub use causality::{ByteRange, Causal, CausalEdge, CausalIndex};
#[cfg(feature = "pipe-config")]
pub use config::RecordingConfig;
#[cfg(feature = "std")]
pub use dest::{FormatKind, SinkSpec};
#[cfg(feature = "std")]
pub use event_sink::{AppendFuture, DynRecordingSink, EventTap, RecordingSink};
#[cfg(feature = "std")]
pub use fanout::FanOut;
#[cfg(feature = "std")]
pub use lazy::{DeferredRuntime, LazyFanOut, deferred_runtime};
#[cfg(feature = "std")]
pub use log_pipe::{AppendAck, AppendLog, ReplayChunk, ReplayLog};
#[cfg(feature = "std")]
pub use replay::{ReplayConfig, ReplayMode, TimedReplay};
#[cfg(feature = "std")]
pub use terminal_signal::TerminalSignal;
