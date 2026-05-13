use bytes::Bytes;
use proxima_primitives::pipe::header_list::HeaderList;

use crate::id::{SpanId, TraceFlags, TraceId, format_traceparent};
use crate::propagation::{TRACEPARENT, TRACESTATE};
use crate::tag::{ScalarValue, Tag, TagSink};
use crate::trace::clock::Clock;
use crate::trace::event::{EventBuilder, EventRecord};
use crate::trace::kind::SpanKind;
use crate::trace::link::{SpanLink, TagInline};
use crate::trace::status::Status;
use crate::trace::tracestate::TraceState;

#[derive(Clone)]
pub struct SpanRecord {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub name: &'static str,
    pub kind: SpanKind,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub status: Status,
    pub attrs: TagInline,
    pub events: smallvec::SmallVec<[EventRecord; 2]>,
    pub links: smallvec::SmallVec<[SpanLink; 1]>,
    pub tracestate: TraceState,
    pub module_path: &'static str,
    pub file_line: (u32, u32),
}

/// Where a finished span is delivered. A trait (not a bare `FnMut`) so the
/// recorder can hand the guard a concrete zero-size-ish sink (a struct holding
/// the per-core ring handle) instead of a `Box<dyn FnMut>` — no heap allocation
/// per span on the emit hot path. The blanket impl keeps closures (tests, the
/// tracing bridge) working unchanged.
pub trait SpanSink {
    fn emit(&mut self, record: SpanRecord);
}

impl<F: FnMut(SpanRecord)> SpanSink for F {
    fn emit(&mut self, record: SpanRecord) {
        self(record);
    }
}

pub struct SpanBuilder {
    name: &'static str,
    kind: SpanKind,
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: Option<SpanId>,
    attrs: TagInline,
    tracestate: TraceState,
    module_path: &'static str,
    file_line: (u32, u32),
}

impl SpanBuilder {
    pub fn new(name: &'static str, trace_id: TraceId, span_id: SpanId) -> Self {
        Self {
            name,
            kind: SpanKind::Internal,
            trace_id,
            span_id,
            parent_span_id: None,
            attrs: smallvec::SmallVec::new(),
            tracestate: TraceState::empty(),
            module_path: "",
            file_line: (0, 0),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn parent(mut self, parent: SpanId) -> Self {
        self.parent_span_id = Some(parent);
        self
    }

    pub fn tag(mut self, key: &'static str, value: impl Into<ScalarValue>) -> Self {
        self.attrs.push(Tag::Scalar {
            key,
            value: value.into(),
        });
        self
    }

    pub fn with_tracestate(mut self, ts: TraceState) -> Self {
        self.tracestate = ts;
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

    pub fn start<C, S>(self, clock: &C, sink: S) -> Span<S>
    where
        C: Clock,
        S: SpanSink,
    {
        let start_ns = clock.now_ns();
        Span {
            builder: self,
            start_ns,
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            status: Status::Unset,
            sink,
        }
    }
}

impl TagSink for SpanBuilder {
    fn push_tag(&mut self, tag: Tag) {
        self.attrs.push(tag);
    }
}

pub struct Span<S: SpanSink> {
    pub(crate) builder: SpanBuilder,
    pub(crate) start_ns: u64,
    pub(crate) events: smallvec::SmallVec<[EventRecord; 2]>,
    pub(crate) links: smallvec::SmallVec<[SpanLink; 1]>,
    pub(crate) status: Status,
    pub(crate) sink: S,
}

impl<S: SpanSink> Span<S> {
    pub fn id(&self) -> SpanId {
        self.builder.span_id
    }

    pub fn trace_id(&self) -> TraceId {
        self.builder.trace_id
    }

    pub fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    pub fn link(&mut self, trace_id: TraceId, span_id: SpanId) {
        self.links.push(SpanLink::new(trace_id, span_id));
    }

    pub fn event<C: Clock>(&mut self, name: &'static str, clock: &C) -> EventBuilder<'_> {
        let ts_ns = clock.now_ns();
        EventBuilder::new(self.builder.span_id, name, ts_ns, &mut self.events)
    }

    /// Consume the span, arm a RAII guard; the guard emits the record on drop.
    ///
    /// The clock is moved into the guard so that `Drop` can compute duration
    /// without a borrow. Use `MonotonicCounter` for tests; C9 will wire the
    /// recorder's platform clock here.
    ///
    /// Enters this span as the thread's current one (a log/metric emitted inside
    /// its synchronous scope correlates to it) and restores the parent on drop.
    /// For a span that rides an `async` future use [`enter_deferred`]: the future
    /// wrapper ([`Spanned`]) manages current-span per poll, so it must not be
    /// pre-entered here.
    ///
    /// [`enter_deferred`]: Span::enter_deferred
    /// [`Spanned`]: crate::spanned::Spanned
    pub fn enter<C: Clock>(self, clock: C) -> SpanGuard<S, C> {
        let current_parent = Some(crate::current::enter(self.trace_id(), self.id()));
        SpanGuard {
            span: Some(self),
            clock,
            current_parent,
        }
    }

    /// Arm the RAII guard WITHOUT entering the current-span stack — for a span
    /// whose current-scoping is driven per-poll by [`Spanned::scoped`]. Emits the
    /// record on drop exactly like [`enter`], but never touches the current-span
    /// stack, so wrapping the guard in a future cannot leak the span across an
    /// `.await`.
    ///
    /// [`enter`]: Span::enter
    /// [`Spanned::scoped`]: crate::spanned::Spanned::scoped
    pub fn enter_deferred<C: Clock>(self, clock: C) -> SpanGuard<S, C> {
        SpanGuard {
            span: Some(self),
            clock,
            current_parent: None,
        }
    }

    pub fn finish<C: Clock>(mut self, clock: &C) {
        let duration_ns = clock.now_ns().saturating_sub(self.start_ns);
        emit_record(
            &mut self.sink,
            self.builder,
            self.start_ns,
            duration_ns,
            self.status,
            self.events,
            self.links,
        );
    }
}

impl<S: SpanSink> TagSink for Span<S> {
    fn push_tag(&mut self, tag: Tag) {
        self.builder.attrs.push(tag);
    }
}

fn emit_record<S: SpanSink>(
    sink: &mut S,
    builder: SpanBuilder,
    start_ns: u64,
    duration_ns: u64,
    status: Status,
    events: smallvec::SmallVec<[EventRecord; 2]>,
    links: smallvec::SmallVec<[SpanLink; 1]>,
) {
    let record = SpanRecord {
        trace_id: builder.trace_id,
        span_id: builder.span_id,
        parent_span_id: builder.parent_span_id,
        name: builder.name,
        kind: builder.kind,
        start_ns,
        duration_ns,
        status,
        attrs: builder.attrs,
        events,
        links,
        tracestate: builder.tracestate,
        module_path: builder.module_path,
        file_line: builder.file_line,
    };
    sink.emit(record);
}

/// RAII span guard — emits the `SpanRecord` on drop via the owned clock.
pub struct SpanGuard<S: SpanSink, C: Clock> {
    span: Option<Span<S>>,
    clock: C,
    // `Some(parent)` when this guard entered the current-span (via `enter`) and so
    // must `restore(parent)` on drop; `None` for `enter_deferred` (Spanned scopes
    // per-poll and never entered here).
    current_parent: Option<Option<(TraceId, SpanId)>>,
}

impl<S: SpanSink, C: Clock> SpanGuard<S, C> {
    pub fn id(&self) -> Option<SpanId> {
        self.span.as_ref().map(|span| span.id())
    }

    pub fn set_status(&mut self, status: Status) {
        if let Some(span) = self.span.as_mut() {
            span.set_status(status);
        }
    }

    pub fn link(&mut self, trace_id: TraceId, span_id: SpanId) {
        if let Some(span) = self.span.as_mut() {
            span.link(trace_id, span_id);
        }
    }

    pub fn event(&mut self, name: &'static str) -> Option<EventBuilder<'_>> {
        let ts_ns = self.clock.now_ns();
        self.span
            .as_mut()
            .map(|span| EventBuilder::new(span.builder.span_id, name, ts_ns, &mut span.events))
    }

    /// Write this span's W3C context into outbound `headers` so a downstream
    /// hop becomes its child: `traceparent` from [`format_traceparent`], plus
    /// the inbound `tracestate` if one was carried. The INJECT half of
    /// [`Recorder::span_from_w3c`]'s EXTRACT. A sampled-out (noop) span has no
    /// allocated context and writes nothing.
    ///
    /// [`format_traceparent`]: crate::id::format_traceparent
    /// [`Recorder::span_from_w3c`]: crate::recorder::Recorder::span_from_w3c
    pub fn inject(&self, headers: &mut HeaderList) {
        let Some(span) = self.span.as_ref() else {
            return;
        };
        // an active span is by definition sampled — noop spans never reach here.
        let traceparent = format_traceparent(&span.trace_id(), &span.id(), TraceFlags::SAMPLED);
        headers.insert(TRACEPARENT, Bytes::copy_from_slice(&traceparent));
        if let Some(state) = span.builder.tracestate.0.as_ref() {
            headers.insert(TRACESTATE, state.clone());
        }
    }
}

impl<S: SpanSink, C: Clock> TagSink for SpanGuard<S, C> {
    fn push_tag(&mut self, tag: Tag) {
        if let Some(span) = self.span.as_mut() {
            span.push_tag(tag);
        }
    }
}

impl<S: SpanSink, C: Clock> Drop for SpanGuard<S, C> {
    fn drop(&mut self) {
        if let Some(span) = self.span.take() {
            let duration_ns = self.clock.now_ns().saturating_sub(span.start_ns);
            let Span {
                builder,
                start_ns,
                events,
                links,
                status,
                mut sink,
            } = span;
            emit_record(
                &mut sink,
                builder,
                start_ns,
                duration_ns,
                status,
                events,
                links,
            );
        }
        // restore the parent this guard displaced on `enter`; `enter_deferred`
        // never entered, so it restores nothing.
        if let Some(parent) = self.current_parent {
            crate::current::restore(parent);
        }
    }
}

impl<S: SpanSink, C: Clock> crate::spanned::SpanContext for SpanGuard<S, C> {
    fn span_context(&self) -> Option<(TraceId, SpanId)> {
        self.span
            .as_ref()
            .map(|span| (span.trace_id(), span.id()))
    }
}
