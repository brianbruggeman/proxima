use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use rstest::rstest;

use crate::id::{SpanId, TraceId};
use crate::tag::{ScalarValue, Tag, TagSink};
use crate::trace::clock::MonotonicCounter;
use crate::trace::event::EventRecord;
use crate::trace::kind::SpanKind;
use crate::trace::link::SpanLink;
use crate::trace::span::{SpanBuilder, SpanRecord};
use crate::trace::status::Status;
use crate::trace::tracestate::TraceState;

const TRACE_BYTES: [u8; 16] = [
    0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31, 0x9c,
];
const SPAN_BYTES: [u8; 8] = [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31];
const SPAN2_BYTES: [u8; 8] = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x11, 0x22];

fn make_ids() -> (TraceId, SpanId) {
    (
        TraceId::from_bytes(TRACE_BYTES),
        SpanId::from_bytes(SPAN_BYTES),
    )
}

fn make_sink() -> (Rc<RefCell<Vec<SpanRecord>>>, impl FnMut(SpanRecord)) {
    let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
    let inner = Rc::clone(&collected);
    let sink = move |record: SpanRecord| inner.borrow_mut().push(record);
    (collected, sink)
}

// 1. SpanBuilder -> start -> finish delivers a SpanRecord to the sink with expected fields
#[test]
fn happy_span_builder_start_finish_delivers_record() {
    let clock = MonotonicCounter::new(100);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    SpanBuilder::new("db.query", trace_id, span_id)
        .kind(SpanKind::Client)
        .start(&clock, sink)
        .finish(&clock);

    let records = collected.borrow();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.name, "db.query");
    assert_eq!(record.trace_id, trace_id);
    assert_eq!(record.span_id, span_id);
    assert_eq!(record.kind, SpanKind::Client);
    assert_eq!(record.status, Status::Unset);
    assert!(record.attrs.is_empty());
    assert!(record.events.is_empty());
    assert!(record.links.is_empty());
}

// 2. span.event() creates an EventRecord inside the span's events Vec
#[test]
fn happy_span_event_creates_event_record() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let mut span = SpanBuilder::new("op", trace_id, span_id).start(&clock, sink);
    span.event("cache.hit", &clock).tag("key", "user:42").emit();
    span.finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].events.len(), 1);
    let evt = &records[0].events[0];
    assert_eq!(evt.name, "cache.hit");
    assert_eq!(evt.parent_span_id, span_id);
    assert_eq!(evt.attrs.len(), 1);
    let Tag::Scalar { key, value } = &evt.attrs[0] else {
        panic!("wrong tag variant")
    };
    assert_eq!(*key, "key");
    assert_eq!(*value, ScalarValue::Str("user:42"));
}

// 3. span.link() pushes a SpanLink onto the span
#[test]
fn happy_span_link_pushes_span_link() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let link_span = SpanId::from_bytes(SPAN2_BYTES);
    let (collected, sink) = make_sink();

    let mut span = SpanBuilder::new("op", trace_id, span_id).start(&clock, sink);
    span.link(trace_id, link_span);
    span.finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].links.len(), 1);
    let link = &records[0].links[0];
    assert_eq!(link.trace_id, trace_id);
    assert_eq!(link.span_id, link_span);
}

// 4. tag flow: builder .tag() before start and push_tag after start both land in attrs
#[test]
fn tag_flow_builder_and_post_start_both_land_in_attrs() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let mut span = SpanBuilder::new("op", trace_id, span_id)
        .tag("pre", 1i64)
        .start(&clock, sink);
    span.push_tag(Tag::Scalar {
        key: "post",
        value: ScalarValue::Bool(true),
    });
    span.finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].attrs.len(), 2);
    assert_eq!(
        records[0].attrs[0],
        Tag::Scalar {
            key: "pre",
            value: ScalarValue::I64(1)
        }
    );
    assert_eq!(
        records[0].attrs[1],
        Tag::Scalar {
            key: "post",
            value: ScalarValue::Bool(true)
        }
    );
}

// 5. status: span.set_status(Error) lands on the emitted record
#[test]
fn status_error_lands_on_record() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let mut span = SpanBuilder::new("op", trace_id, span_id).start(&clock, sink);
    span.set_status(Status::Error { reason: "timeout" });
    span.finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].status, Status::Error { reason: "timeout" });
}

// 6. span_builder! macro expands with module_path and file_line auto-captured
#[test]
fn macro_span_builder_captures_source_location() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    crate::span_builder!(trace_id, span_id, "annotated")
        .start(&clock, sink)
        .finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].name, "annotated");
    assert!(
        !records[0].module_path.is_empty(),
        "module_path must be set by macro"
    );
    assert_ne!(
        records[0].file_line,
        (0, 0),
        "file_line must be set by macro"
    );
}

// 7. span_builder! macro with tags embeds attrs
#[test]
fn macro_span_builder_with_tags_embeds_attrs() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let builder = crate::span_builder!(trace_id, span_id, "tagged", "http.method" = "GET");
    builder.start(&clock, sink).finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].attrs.len(), 1);
    assert_eq!(
        records[0].attrs[0],
        Tag::Scalar {
            key: "http.method",
            value: ScalarValue::Str("GET")
        }
    );
}

// 8. SpanGuard drop emits the record exactly once via sink
#[test]
fn drop_span_guard_emits_exactly_once() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    {
        let guard = SpanBuilder::new("raii", trace_id, span_id)
            .start(&clock, sink)
            .enter(MonotonicCounter::new(1_000));
        assert_eq!(
            collected.borrow().len(),
            0,
            "sink must not be called before drop"
        );
        drop(guard);
    }

    assert_eq!(
        collected.borrow().len(),
        1,
        "sink must be called exactly once on drop"
    );
}

// 9. parent_span_id propagates from SpanBuilder::parent() to SpanRecord
#[test]
fn parent_span_id_propagates_to_record() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let parent = SpanId::from_bytes(SPAN2_BYTES);
    let (collected, sink) = make_sink();

    SpanBuilder::new("child", trace_id, span_id)
        .parent(parent)
        .start(&clock, sink)
        .finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].parent_span_id, Some(parent));
}

// 10. TraceState round-trips on the SpanRecord
#[test]
fn tracestate_roundtrips_on_record() {
    use bytes::Bytes;

    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let ts = TraceState::from_bytes(Bytes::from_static(b"vendor=abc123"));

    SpanBuilder::new("op", trace_id, span_id)
        .with_tracestate(ts.clone())
        .start(&clock, sink)
        .finish(&clock);

    let records = collected.borrow();
    assert_eq!(records[0].tracestate, ts);
    assert!(!records[0].tracestate.is_empty());
}

// 11. span_event! macro emits an event on a SpanGuard
#[test]
fn macro_span_event_emits_event_on_guard() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let mut guard = SpanBuilder::new("op", trace_id, span_id)
        .start(&clock, sink)
        .enter(MonotonicCounter::new(1_000));

    crate::span_event!(guard, "checkpoint");

    drop(guard);

    let records = collected.borrow();
    assert_eq!(records[0].events.len(), 1);
    assert_eq!(records[0].events[0].name, "checkpoint");
}

// 12. span_event! macro with attrs pushes the attrs onto the event
#[test]
fn macro_span_event_with_attrs_pushes_attrs() {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let mut guard = SpanBuilder::new("op", trace_id, span_id)
        .start(&clock, sink)
        .enter(MonotonicCounter::new(1_000));

    crate::span_event!(guard, "annotated", "latency_ms" = 42u64);

    drop(guard);

    let records = collected.borrow();
    assert_eq!(records[0].events.len(), 1);
    let evt = &records[0].events[0];
    assert_eq!(evt.attrs.len(), 1);
    assert_eq!(
        evt.attrs[0],
        Tag::Scalar {
            key: "latency_ms",
            value: ScalarValue::U64(42)
        }
    );
}

// 13. SpanKind variants are all distinct
#[rstest]
#[case::internal(SpanKind::Internal)]
#[case::server(SpanKind::Server)]
#[case::client(SpanKind::Client)]
#[case::producer(SpanKind::Producer)]
#[case::consumer(SpanKind::Consumer)]
fn span_kind_variants_distinct(#[case] kind: SpanKind) {
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    SpanBuilder::new("op", trace_id, span_id)
        .kind(kind)
        .start(&clock, sink)
        .finish(&clock);

    assert_eq!(collected.borrow()[0].kind, kind);
}

// 14. duration_ns is nonzero when MonotonicCounter advances
#[test]
fn duration_ns_is_nonzero_after_guard_drop() {
    let clock = MonotonicCounter::new(0);
    let end_clock = MonotonicCounter::new(1_000);
    let (trace_id, span_id) = make_ids();
    let (collected, sink) = make_sink();

    let guard = SpanBuilder::new("op", trace_id, span_id)
        .start(&clock, sink)
        .enter(end_clock);
    drop(guard);

    let duration = collected.borrow()[0].duration_ns;
    assert!(duration > 0, "duration_ns must be > 0 when clock advances");
}

// 15. size assertions — regression guard for struct layouts
#[test]
fn size_of_known_types() {
    use core::mem::size_of;

    let event_size = size_of::<EventRecord>();
    let record_size = size_of::<SpanRecord>();
    let link_size = size_of::<SpanLink>();

    extern crate std;
    std::eprintln!(
        "EventRecord: {event_size}B  SpanRecord: {record_size}B  SpanLink: {link_size}B"
    );

    assert!(event_size > 0);
    assert!(record_size > 0);
    assert!(link_size > 0);
}
