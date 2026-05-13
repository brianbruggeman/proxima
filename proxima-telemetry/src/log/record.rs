use crate::id::{SpanId, TraceFlags, TraceId};
use crate::level::Level;
use crate::log::body::LogBody;
use crate::tag::Tag;

/// A single emitted log record; the immutable snapshot handed to the sink.
///
/// `ts_ns` is the event timestamp; `observed_ts_ns` is when the record was
/// constructed (may differ in async or buffered pipelines).
#[derive(Clone)]
pub struct LogRecord {
    pub ts_ns: u64,
    pub observed_ts_ns: u64,
    pub level: Level,
    pub body: LogBody,
    pub attrs: smallvec::SmallVec<[Tag; 4]>,
    pub trace_id: Option<TraceId>,
    pub span_id: Option<SpanId>,
    pub trace_flags: TraceFlags,
    pub module_path: &'static str,
    pub file_line: (u32, u32),
}
