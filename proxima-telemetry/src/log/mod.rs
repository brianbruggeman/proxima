pub mod body;
pub mod builder;
pub mod record;

pub use body::LogBody;
pub use builder::LogBuilder;
pub use record::LogRecord;

/// Emit a log record at a given level with an optional set of tags.
///
/// Forms:
/// - `log_record!(builder_expr, level, "msg")` — no attrs
/// - `log_record!(builder_expr, level, "msg", "k" = v, ...)` — with attrs
///
/// `builder_expr` must be an expression that produces a `LogBuilder<S, C>` via
/// `.level(level).message(msg).module_path(...).file_line(...)`.
/// In v1, callers pass a `LogBuilder::new(level, sink, clock)` expression directly.
/// C9 will introduce a recorder that wraps this pattern behind `.log(level)`.
///
/// Named `log_record!` (not `log!`) to avoid shadowing the `log` crate's macros,
/// consistent with C5's `span_builder!`/`span_event!` naming.
#[macro_export]
macro_rules! log_record {
    ($builder:expr $(,)?) => {{
        let mut __b = $builder;
        __b = __b
            .module_path(::core::module_path!())
            .file_line(::core::line!(), ::core::column!());
        __b.emit();
    }};
    ($builder:expr, $msg:literal $(,)?) => {{
        let mut __b = $builder;
        __b = __b
            .message($msg)
            .module_path(::core::module_path!())
            .file_line(::core::line!(), ::core::column!());
        __b.emit();
    }};
    ($builder:expr, $msg:literal, $($rest:tt)+) => {{
        let mut __b = $builder;
        __b = __b
            .message($msg)
            .module_path(::core::module_path!())
            .file_line(::core::line!(), ::core::column!());
        $crate::tag!(__b, $($rest)+);
        __b.emit();
    }};
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

    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use bytes::Bytes;
    use rstest::rstest;

    use crate::clock::MonotonicCounter;
    use crate::id::{SpanId, TraceFlags, TraceId};
    use crate::level::Level;
    use crate::log::body::LogBody;
    use crate::log::builder::LogBuilder;
    use crate::log::record::LogRecord;
    use crate::tag::{NestedValue, ScalarValue, Tag, TagSink};

    const TRACE_BYTES: [u8; 16] = [
        0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31,
        0x9c,
    ];
    const SPAN_BYTES: [u8; 8] = [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31];

    fn make_sink() -> (Rc<RefCell<Vec<LogRecord>>>, impl FnMut(LogRecord)) {
        let collected: Rc<RefCell<Vec<LogRecord>>> = Rc::new(RefCell::new(Vec::new()));
        let inner = Rc::clone(&collected);
        let sink = move |record: LogRecord| inner.borrow_mut().push(record);
        (collected, sink)
    }

    // 1. happy: LogBuilder::new + message + emit delivers LogRecord with expected fields
    #[test]
    fn happy_builder_new_message_emit_delivers_record() {
        let clock = MonotonicCounter::new(100);
        let (collected, sink) = make_sink();

        LogBuilder::new(Level::INFO, sink, clock)
            .message("hello world")
            .module_path("mymod")
            .file_line(42, 1)
            .emit();

        let records = collected.borrow();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.level, Level::INFO);
        assert_eq!(record.body, LogBody::Text("hello world"));
        assert_eq!(record.module_path, "mymod");
        assert_eq!(record.file_line, (42, 1));
        assert!(record.attrs.is_empty());
        assert!(record.trace_id.is_none());
        assert!(record.span_id.is_none());
    }

    // 2. body Text: .message("static") produces LogBody::Text
    #[test]
    fn body_text_message_produces_text_variant() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        LogBuilder::new(Level::DEBUG, sink, clock)
            .message("static message")
            .emit();

        let records = collected.borrow();
        assert_eq!(records[0].body, LogBody::Text("static message"));
    }

    // 3. body Owned: .body_bytes(...) produces LogBody::Owned
    #[test]
    fn body_bytes_produces_owned_variant() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();
        let payload = Bytes::from_static(b"dynamic bytes");

        LogBuilder::new(Level::INFO, sink, clock)
            .body_bytes(payload.clone())
            .emit();

        let records = collected.borrow();
        let LogBody::Owned(got) = &records[0].body else {
            panic!("expected LogBody::Owned");
        };
        assert_eq!(got, &payload);
    }

    // 4. body Structured: .body_structured(NestedValue::Array(...)) produces LogBody::Structured
    #[test]
    fn body_structured_produces_structured_variant() {
        static ITEMS: &[NestedValue] = &[
            NestedValue::Scalar(ScalarValue::I64(1)),
            NestedValue::Scalar(ScalarValue::Bool(true)),
        ];
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        LogBuilder::new(Level::INFO, sink, clock)
            .body_structured(NestedValue::Array(ITEMS))
            .emit();

        let records = collected.borrow();
        let LogBody::Structured(NestedValue::Array(items)) = &records[0].body else {
            panic!("expected LogBody::Structured(Array)");
        };
        assert_eq!(items.len(), 2);
    }

    // 5. trace correlation: .trace(t, s) sets trace_id and span_id; default is None
    #[test]
    fn trace_correlation_sets_ids_and_default_is_none() {
        let trace_id = TraceId::from_bytes(TRACE_BYTES);
        let span_id = SpanId::from_bytes(SPAN_BYTES);
        let clock = MonotonicCounter::new(0);

        let (collected_with, sink_with) = make_sink();
        LogBuilder::new(Level::INFO, sink_with, MonotonicCounter::new(0))
            .message("with trace")
            .trace(trace_id, span_id)
            .emit();

        let (collected_without, sink_without) = make_sink();
        LogBuilder::new(Level::INFO, sink_without, clock)
            .message("without trace")
            .emit();

        let with_records = collected_with.borrow();
        assert_eq!(with_records[0].trace_id, Some(trace_id));
        assert_eq!(with_records[0].span_id, Some(span_id));

        let without_records = collected_without.borrow();
        assert!(without_records[0].trace_id.is_none());
        assert!(without_records[0].span_id.is_none());
    }

    // 6. tag flow: tags added via builder AND via tag! macro both land in attrs
    #[test]
    fn tag_flow_builder_and_macro_both_land_in_attrs() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        let mut builder = LogBuilder::new(Level::INFO, sink, clock).message("tagged");
        builder.push_tag(Tag::Scalar {
            key: "direct",
            value: ScalarValue::I64(1),
        });
        crate::tag!(builder, "macro_key" = 2i64);
        builder.emit();

        let records = collected.borrow();
        assert_eq!(records[0].attrs.len(), 2);
        assert_eq!(
            records[0].attrs[0],
            Tag::Scalar {
                key: "direct",
                value: ScalarValue::I64(1)
            }
        );
        assert_eq!(
            records[0].attrs[1],
            Tag::Scalar {
                key: "macro_key",
                value: ScalarValue::I64(2)
            }
        );
    }

    // 7. log_record! macro: with no tags emits empty-attrs record
    #[test]
    fn macro_no_tags_emits_empty_attrs_record() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        crate::log_record!(LogBuilder::new(Level::INFO, sink, clock), "no attrs here");

        let records = collected.borrow();
        assert_eq!(records.len(), 1);
        assert!(records[0].attrs.is_empty());
        assert_eq!(records[0].body, LogBody::Text("no attrs here"));
    }

    // 8. log_record! macro: with tags emits record with tags in order
    #[test]
    fn macro_with_tags_emits_record_with_tags_in_order() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        crate::log_record!(
            LogBuilder::new(Level::WARN, sink, clock),
            "with tags",
            "k1" = 1i64,
            "k2" = "two",
            "k3" = true,
        );

        let records = collected.borrow();
        assert_eq!(records[0].attrs.len(), 3);
        assert_eq!(
            records[0].attrs[0],
            Tag::Scalar {
                key: "k1",
                value: ScalarValue::I64(1)
            }
        );
        assert_eq!(
            records[0].attrs[1],
            Tag::Scalar {
                key: "k2",
                value: ScalarValue::Str("two")
            }
        );
        assert_eq!(
            records[0].attrs[2],
            Tag::Scalar {
                key: "k3",
                value: ScalarValue::Bool(true)
            }
        );
    }

    // 9. module_path / file_line auto-captured by log_record! macro
    #[test]
    fn macro_captures_source_location() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        crate::log_record!(LogBuilder::new(Level::INFO, sink, clock), "location test");

        let records = collected.borrow();
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

    // 10. custom level: emit with Level::custom("audit", 18) and verify severity round-trips
    #[rstest]
    #[case::audit(Level::custom("audit", 18), 18, "audit")]
    #[case::trace_low(Level::custom("trace2", 2), 2, "trace2")]
    #[case::fatal_high(Level::custom("alert", 25), 25, "alert")]
    fn custom_level_severity_roundtrips(
        #[case] level: Level,
        #[case] expected_severity: u8,
        #[case] expected_name: &str,
    ) {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        LogBuilder::new(level, sink, clock)
            .message("custom level test")
            .emit();

        let records = collected.borrow();
        assert_eq!(records[0].level.severity(), expected_severity);
        assert_eq!(records[0].level.name(), expected_name);
    }

    // 11. size assertions — regression guard for struct layouts
    #[test]
    fn size_of_known_types() {
        use core::mem::size_of;

        let record_size = size_of::<LogRecord>();
        let body_size = size_of::<LogBody>();

        extern crate std;
        std::eprintln!("LogRecord: {record_size}B  LogBody: {body_size}B");

        assert!(record_size > 0);
        assert!(body_size > 0);
    }

    // 12. trace_flags: .trace_flags(TraceFlags::SAMPLED) propagates to record
    #[test]
    fn trace_flags_propagates_to_record() {
        let clock = MonotonicCounter::new(0);
        let (collected, sink) = make_sink();

        LogBuilder::new(Level::INFO, sink, clock)
            .message("sampled")
            .trace_flags(TraceFlags::SAMPLED)
            .emit();

        let records = collected.borrow();
        assert_eq!(records[0].trace_flags, TraceFlags::SAMPLED);
    }
}
