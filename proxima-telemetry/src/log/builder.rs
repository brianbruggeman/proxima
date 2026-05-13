use bytes::Bytes;

use crate::clock::Clock;
use crate::id::{SpanId, TraceFlags, TraceId};
use crate::level::Level;
use crate::log::body::LogBody;
use crate::log::record::LogRecord;
use crate::tag::{NestedValue, Tag, TagSink};

/// Ergonomic builder for a `LogRecord`.
///
/// Generic over the sink (`S`) and clock (`C`) to stay zero-alloc on the hot path.
/// v1: call sites construct directly via `LogBuilder::new`; C9 will wrap this
/// behind a recorder that provides the clock and sink from context.
pub struct LogBuilder<S, C>
where
    S: FnMut(LogRecord),
    C: Clock,
{
    level: Level,
    body: LogBody,
    attrs: smallvec::SmallVec<[Tag; 4]>,
    trace_id: Option<TraceId>,
    span_id: Option<SpanId>,
    trace_flags: TraceFlags,
    module_path: &'static str,
    file_line: (u32, u32),
    sink: S,
    clock: C,
}

impl<S, C> LogBuilder<S, C>
where
    S: FnMut(LogRecord),
    C: Clock,
{
    pub fn new(level: Level, sink: S, clock: C) -> Self {
        Self {
            level,
            body: LogBody::Empty,
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags::NOT_SAMPLED,
            module_path: "",
            file_line: (0, 0),
            sink,
            clock,
        }
    }

    pub fn message(mut self, text: &'static str) -> Self {
        self.body = LogBody::Text(text);
        self
    }

    pub fn body_bytes(mut self, bytes: Bytes) -> Self {
        self.body = LogBody::Owned(bytes);
        self
    }

    pub fn body_structured(mut self, value: NestedValue) -> Self {
        self.body = LogBody::Structured(value);
        self
    }

    pub fn trace(mut self, trace_id: TraceId, span_id: SpanId) -> Self {
        self.trace_id = Some(trace_id);
        self.span_id = Some(span_id);
        self
    }

    pub fn trace_flags(mut self, flags: TraceFlags) -> Self {
        self.trace_flags = flags;
        self
    }

    pub fn module_path(mut self, mod_path: &'static str) -> Self {
        self.module_path = mod_path;
        self
    }

    pub fn file_line(mut self, line: u32, col: u32) -> Self {
        self.file_line = (line, col);
        self
    }

    pub fn emit(mut self) {
        // correlate: with no span set explicitly, stamp the scoped current span
        // (pushed by the enclosing `SpanGuard`/`Spanned`). It is always a sampled,
        // active span — noop spans never enter the stack — so flag it sampled so
        // exporters keep it alongside its trace.
        if self.span_id.is_none()
            && let Some((trace_id, span_id)) = crate::current::current()
        {
            self.trace_id = Some(trace_id);
            self.span_id = Some(span_id);
            self.trace_flags = TraceFlags::SAMPLED;
            // mark records of a verbose-sampled trace so ElevationSink retains
            // them for a replay; the bit rides the record only, not the wire.
            #[cfg(feature = "elevation")]
            if crate::current::is_current_verbose() {
                self.trace_flags = self.trace_flags.with_verbose_buffered();
            }
        }
        // single clock read for synchronous emit — both timestamps land on the same tick.
        // async pipelines that split observed_ts from event_ts will set them explicitly.
        let ts_ns = self.clock.now_ns();
        let record = LogRecord {
            ts_ns,
            observed_ts_ns: ts_ns,
            level: self.level,
            body: self.body,
            attrs: self.attrs,
            trace_id: self.trace_id,
            span_id: self.span_id,
            trace_flags: self.trace_flags,
            module_path: self.module_path,
            file_line: self.file_line,
        };
        (self.sink)(record);
    }
}

impl<S, C> TagSink for LogBuilder<S, C>
where
    S: FnMut(LogRecord),
    C: Clock,
{
    fn push_tag(&mut self, tag: Tag) {
        self.attrs.push(tag);
    }
}
