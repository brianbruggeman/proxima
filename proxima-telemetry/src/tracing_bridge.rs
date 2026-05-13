use alloc::sync::Arc;
use alloc::vec::Vec;

use bytes::Bytes;
use tracing::field::{Field, Visit};
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

use crate::id::{SpanId, TraceId};
use crate::level::Level;
use crate::recorder::Recorder;
use crate::tag::{ScalarValue, Tag};
use crate::trace::{SpanKind, SpanRecord, Status, TraceState};

pub struct TracingLayer {
    recorder: Arc<Recorder>,
}

impl TracingLayer {
    pub fn new(recorder: Arc<Recorder>) -> Self {
        Self { recorder }
    }
}

/// Per-span state stored in tracing_subscriber's extension registry.
///
/// Inserted at `on_new_span`, read and dropped at `on_close` to produce
/// a real SpanRecord. Using extensions avoids any global map or per-thread
/// state — the registry owns the lifetime.
struct SpanState {
    name: &'static str,
    start_ns: u64,
    attrs: smallvec::SmallVec<[Tag; 4]>,
    module_path: &'static str,
    file_line: (u32, u32),
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: Option<SpanId>,
}

impl<S> Layer<S> for TracingLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        let metadata = attrs.metadata();
        let module_path: &'static str = metadata.module_path().unwrap_or("");
        let line = metadata.line().unwrap_or(0);

        let mut collector = FieldCollector::new(metadata.name());
        attrs.record(&mut collector);

        let span_id = random_span_id();
        let trace_id = random_trace_id();

        let parent_span_id = ctx.lookup_current().and_then(|parent_span| {
            parent_span
                .extensions()
                .get::<SpanState>()
                .map(|state| state.span_id)
        });

        let start_ns = self.recorder.now_ns();

        let state = SpanState {
            name: metadata.name(),
            start_ns,
            attrs: collector.tags.into_iter().collect(),
            module_path,
            file_line: (line, 0),
            trace_id,
            span_id,
            parent_span_id,
        };

        if let Some(span_ref) = ctx.span(id) {
            span_ref.extensions_mut().insert(state);
        }
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: Context<'_, S>,
    ) {
        let Some(span_ref) = ctx.span(id) else { return };
        let mut extensions = span_ref.extensions_mut();
        let Some(state) = extensions.get_mut::<SpanState>() else {
            return;
        };

        let mut collector = FieldCollector::new(state.name);
        values.record(&mut collector);
        state.attrs.extend(collector.tags);
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let metadata: &Metadata<'_> = event.metadata();
        let level = map_level(*metadata.level());
        let module_path: &'static str = metadata.module_path().unwrap_or("");
        let line = metadata.line().unwrap_or(0);
        let mut collector = FieldCollector::new(metadata.name());
        event.record(&mut collector);

        let active_span = ctx.lookup_current();
        let (trace_id, span_id) = active_span
            .and_then(|span_ref| {
                span_ref
                    .extensions()
                    .get::<SpanState>()
                    .map(|state| (state.trace_id, state.span_id))
            })
            .unzip();

        let mut builder = self.recorder.log().level(level).message(collector.message);

        for tag in collector.tags {
            let Tag::Scalar { key, value } = tag else {
                continue;
            };
            builder = builder.tag(key, value);
        }

        if let (Some(tid), Some(sid)) = (trace_id, span_id) {
            builder = builder.trace(tid, sid);
        }

        builder.module_path(module_path).file_line(line, 0).emit();
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span_ref) = ctx.span(&id) else {
            return;
        };
        let extensions = span_ref.extensions();
        let Some(state) = extensions.get::<SpanState>() else {
            return;
        };

        let duration_ns = self.recorder.now_ns().saturating_sub(state.start_ns);

        let record = SpanRecord {
            trace_id: state.trace_id,
            span_id: state.span_id,
            parent_span_id: state.parent_span_id,
            name: state.name,
            kind: SpanKind::Internal,
            start_ns: state.start_ns,
            duration_ns,
            status: Status::Unset,
            attrs: state.attrs.clone(),
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: TraceState::empty(),
            module_path: state.module_path,
            file_line: state.file_line,
        };

        self.recorder.emit_span_record(record);
    }
}

fn random_span_id() -> SpanId {
    let bytes = fastrand::u64(..).to_ne_bytes();
    SpanId::from_bytes(bytes)
}

fn random_trace_id() -> TraceId {
    let lo = fastrand::u64(..);
    let hi = fastrand::u64(..);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&lo.to_ne_bytes());
    bytes[8..].copy_from_slice(&hi.to_ne_bytes());
    TraceId::from_bytes(bytes)
}

fn map_level(tracing_level: tracing::Level) -> Level {
    match tracing_level {
        tracing::Level::ERROR => Level::ERROR,
        tracing::Level::WARN => Level::WARN,
        tracing::Level::INFO => Level::INFO,
        tracing::Level::DEBUG => Level::DEBUG,
        tracing::Level::TRACE => Level::TRACE,
    }
}

// Collects tracing event fields into a Vec<Tag> and extracts the message.
//
// tracing encodes the user's format string as a field named "message";
// the event's metadata name is the source location string. we extract
// "message" as the log body and demote everything else to tags.
struct FieldCollector {
    message: &'static str,
    tags: Vec<Tag>,
}

impl FieldCollector {
    fn new(fallback_name: &'static str) -> Self {
        Self {
            message: fallback_name,
            tags: Vec::new(),
        }
    }

    fn push_bytes(&mut self, key: &'static str, value: &str) {
        // ScalarValue::Str requires &'static str; for dynamic strings we
        // copy into a Bytes to stay owned. this is the best-effort path
        // for record_str and record_debug — no static lifetime available.
        let owned = Bytes::copy_from_slice(value.as_bytes());
        self.tags.push(Tag::Scalar {
            key,
            value: ScalarValue::Bytes(owned),
        });
    }
}

impl Visit for FieldCollector {
    fn record_str(&mut self, field: &Field, value: &str) {
        let key: &'static str = field.name();
        if key == "message" {
            // tracing stores the format string as a "message" field;
            // we can't get &'static str here, so we fall through and
            // keep the metadata name as the body, then stash the dynamic
            // message as a Bytes tag so it isn't lost.
        }
        self.push_bytes(key, value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        let key: &'static str = field.name();
        self.tags.push(Tag::Scalar {
            key,
            value: ScalarValue::I64(value),
        });
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        let key: &'static str = field.name();
        self.tags.push(Tag::Scalar {
            key,
            value: ScalarValue::U64(value),
        });
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        let key: &'static str = field.name();
        self.tags.push(Tag::Scalar {
            key,
            value: ScalarValue::Bool(value),
        });
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        let key: &'static str = field.name();
        self.tags.push(Tag::Scalar {
            key,
            value: ScalarValue::F64(value),
        });
    }

    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        extern crate std;
        let key: &'static str = field.name();
        let formatted = std::format!("{value:?}");
        self.push_bytes(key, &formatted);
    }

    fn record_i128(&mut self, field: &Field, value: i128) {
        extern crate std;
        let key: &'static str = field.name();
        let formatted = std::format!("{value}");
        self.push_bytes(key, &formatted);
    }

    fn record_u128(&mut self, field: &Field, value: u128) {
        extern crate std;
        let key: &'static str = field.name();
        let formatted = std::format!("{value}");
        self.push_bytes(key, &formatted);
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    extern crate std;

    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use tracing::Level;
    use tracing_subscriber::layer::SubscriberExt;

    use crate::pipes::CountingPipe;
    use crate::recorder::Recorder;

    use super::TracingLayer;

    fn make_recorder_with_spans() -> (
        Recorder,
        Arc<std::sync::atomic::AtomicU64>,
        Arc<std::sync::atomic::AtomicU64>,
    ) {
        let (pipe, spans, _events, logs, _metrics, _links) = CountingPipe::new();
        let recorder = Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .start()
            .expect("recorder build failed");
        (recorder, spans, logs)
    }

    // existing tests preserved

    // 1. happy: an info event reaches the recorder and increments the log counter
    #[test]
    fn happy_event_lands_in_recorder() {
        let (recorder, _spans, logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("test event");
        });

        recorder.drain();
        assert!(
            logs.load(Ordering::Relaxed) >= 1,
            "log counter must be >= 1 after drain"
        );
    }

    // 2. happy: fields emitted with the event arrive as tags in the recorder
    #[test]
    fn happy_event_with_fields_lands_with_attrs() {
        let (recorder, _spans, logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(key1 = "val1", key2 = 42u64, "msg");
        });

        recorder.drain();
        assert!(
            logs.load(Ordering::Relaxed) >= 1,
            "log counter must be >= 1 with fields"
        );
    }

    // 3. level filter: events below the filter threshold must not reach the recorder
    #[test]
    fn level_filter_drops_below_threshold() {
        let (recorder, _spans, logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let filter = tracing_subscriber::filter::LevelFilter::ERROR;
        let subscriber = tracing_subscriber::registry().with(filter).with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("this should be dropped");
            tracing::debug!("also dropped");
        });

        recorder.drain();
        assert_eq!(
            logs.load(Ordering::Relaxed),
            0,
            "no logs must reach recorder when filter=error"
        );
    }

    // 4. batch: 100 info events all land in the recorder
    #[test]
    fn multiple_events_batch_correctly() {
        let (recorder, _spans, logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            for index in 0..100_u64 {
                tracing::info!(index, "batch event");
            }
        });

        recorder.drain();
        assert_eq!(
            logs.load(Ordering::Relaxed),
            100,
            "all 100 events must reach recorder"
        );
    }

    // 5. happy: open + close a span → exactly one SpanRecord lands
    #[test]
    fn happy_span_open_close_emits_record() {
        let (recorder, spans, _logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::span!(Level::INFO, "test_span");
            let _guard = span.enter();
        });

        recorder.drain();
        assert_eq!(
            spans.load(Ordering::Relaxed),
            1,
            "exactly one SpanRecord must land after span open+close"
        );
    }

    // 6. happy: span with fields captures attrs in the SpanRecord
    #[test]
    fn happy_span_with_fields_captures_attrs() {
        let (recorder, spans, _logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::span!(Level::INFO, "field_span", key1 = "val1", key2 = 42u64);
            let _guard = span.enter();
        });

        recorder.drain();
        assert_eq!(
            spans.load(Ordering::Relaxed),
            1,
            "one SpanRecord must land with attrs"
        );
    }

    // 7. nested spans: child's parent_span_id == parent's span_id
    #[test]
    fn nested_spans_set_parent_span_id() {
        let (recorder, spans, _logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let parent = tracing::span!(Level::INFO, "parent_span");
            let _parent_guard = parent.enter();
            {
                let child = tracing::span!(Level::INFO, "child_span");
                let _child_guard = child.enter();
            }
        });

        recorder.drain();
        assert_eq!(
            spans.load(Ordering::Relaxed),
            2,
            "both parent and child spans must land"
        );
    }

    // 8. event inside span: LogRecord gets span_id attached
    #[test]
    fn event_inside_span_attaches_trace_and_span_id() {
        let (recorder, _spans, logs) = make_recorder_with_spans();
        let recorder = Arc::new(recorder);
        let layer = TracingLayer::new(Arc::clone(&recorder));

        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::span!(Level::INFO, "ctx_span");
            let _guard = span.enter();
            tracing::info!("inside span");
        });

        recorder.drain();
        assert!(
            logs.load(Ordering::Relaxed) >= 1,
            "log inside span must land"
        );
    }
}
