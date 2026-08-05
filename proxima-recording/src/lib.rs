//! Recording layer for proxima: event model, source traits, binary and
//! JSONL on-disk formats, index files, and the source factory registry (the
//! base/format tier — folded from the former `proxima-recording-core`).
//!
//! No Pipe deps at the base tier. Pipe-flavored pieces (`Causal`,
//! `LiveCaptureContext`, `BoundedRecordingSink`, the record-wrapper Pipe;
//! folded from the former `proxima-recording-pipe`) live behind the `pipe`
//! feature, in `pipe`. Cassette replay-by-match-key (`ReplayUpstream`;
//! folded from the former `proxima-replay`) lives behind the `replay`
//! feature, in `replay`.
//!
//! Base-tier two tiers:
//!
//! - `alloc` (no_std + alloc): the event model (`event`), the binary block
//!   codec (`binary::frame`, `binary::wire`), and the runtime-agnostic
//!   blocking-I/O offload seam (`rt_fs`). No file handles, no `BufRead`, no
//!   registries — sans-IO by construction. This is the default surface a
//!   bare-metal target builds against (`--no-default-features --features
//!   alloc`). `replay`'s keying core (`replay::keying`, `replay::meta`)
//!   also lands here.
//! - `std` (default, forwards `alloc`): everything above plus the
//!   source/index/factory registries and the bin/jsonl format readers,
//!   which need `std::io::BufRead`, `PathBuf`, and `OnceLock`. `pipe`
//!   forwards `std` — every item it exposes owns a file handle or an OS
//!   thread.
//!
//! The `format::Format` trait (std tier) is the codec seam: it frames a batch
//! into the exact bytes appended to a log, and reads one unit back. Everything
//! above it — `pipe::AppendLog`, `pipe::ReplayLog`, the sources — is file
//! handling and scheduling around it.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod binary;
#[cfg(feature = "alloc")]
pub mod event;
#[cfg(feature = "std")]
pub mod factory;
#[cfg(feature = "std")]
pub mod format;
#[cfg(feature = "std")]
pub mod jsonl;
#[cfg(feature = "pipe")]
pub mod pipe;
#[cfg(feature = "replay")]
pub mod replay;
#[cfg(feature = "alloc")]
pub mod rt_fs;
#[cfg(feature = "std")]
pub mod source;

#[cfg(feature = "std")]
pub use binary::source::BinSourceFactory;
#[cfg(feature = "std")]
pub use binary::{BinFormat, BinSource, INDEX_RECORD_BYTES, IndexReader, IndexRecord, IndexWriter};
#[cfg(feature = "alloc")]
pub use event::{
    CacheOutcome, EventSource, FrameMetadata, HttpEvent, InteractionId, PipelineEvent,
    PipelineOutcome, ProcessEvent, ProtocolEvent, RECORDING_FORMAT_VERSION, RecordMeta,
    RecordingEvent, RequestHeader,
};
#[cfg(feature = "std")]
pub use factory::{
    DynRecordingSourceFactory, RecordingSourceFactory, RecordingSourceRegistry, SourceBuildFuture,
};
#[cfg(feature = "std")]
pub use format::Format;
#[cfg(feature = "std")]
pub use jsonl::JsonFormat;
#[cfg(feature = "std")]
pub use jsonl::JsonlSource;
#[cfg(feature = "std")]
pub use jsonl::source::JsonlSourceFactory;
#[cfg(feature = "std")]
pub use source::{DynRecordingSource, RecordingEventStream, RecordingSource};
