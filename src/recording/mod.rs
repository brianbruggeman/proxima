// Read/format substrate (event model, Format codecs, sources, source registry)
// stays at the proxima-recording crate root (folded from the former
// proxima-recording-core). The SINK half (the RecordingSink trait,
// BinSink/JsonlSink/BroadcastSink, the sink factory/registry) was deleted
// from recording-core and relocated onto the spigot substrate in
// proxima-recording's `pipe` module (folded from the former
// proxima-recording-pipe, C7). `sink` aliases that relocated module so
// `crate::recording::sink::{RecordingSink, DynRecordingSink, AppendFuture}`
// keeps resolving for existing consumers.
pub use proxima_recording::binary as bin;
pub use proxima_recording::jsonl;
pub use proxima_recording::{
    BinSource, BinSourceFactory, CacheOutcome, DynRecordingSource, DynRecordingSourceFactory,
    EventSource, FrameMetadata, HttpEvent, INDEX_RECORD_BYTES, IndexReader, IndexRecord,
    IndexWriter, InteractionId, JsonlSource, JsonlSourceFactory, PipelineEvent, PipelineOutcome,
    ProcessEvent, ProtocolEvent, ProtocolRenderer, RECORDING_FORMAT_VERSION, RecordMeta,
    RecordingEvent, RecordingEventStream, RecordingSource, RecordingSourceFactory,
    RecordingSourceRegistry, RequestHeader, SourceBuildFuture, event, factory, source,
};

// sink half relocated to the spigot substrate.
pub use proxima_recording::pipe::event_sink as sink;
pub use proxima_recording::pipe::{
    AccumulatingSink, AppendAck, AppendFuture, AppendLog, DeferredRuntime, DynRecordingSink,
    EventTap, FanOut, FormatKind, LazyFanOut, RecordingSink, ReplayLog, SinkSpec, TerminalSignal,
    deferred_runtime,
};

pub use proxima_recording::pipe::cap;
pub use proxima_recording::pipe::capture;
// bridges a `ProcessUpstream` (tokio::process-backed, no prime equivalent
// today) into the recording event stream — genuinely tokio-only.
#[cfg(feature = "tokio")]
pub mod process_bridge;

pub use crate::capture_surface::CaptureContext;
#[cfg(feature = "tokio")]
pub use process_bridge::{BridgeHandle, ProcessEventBridge};
pub use proxima_recording::pipe::cap::{
    BoundedRecordingSink, DropReason, FailMode, RECORD_DROP_METRIC,
};
pub use proxima_recording::pipe::capture::LiveCaptureContext;
