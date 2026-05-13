// integration tests for #[span] and #[derive(SpanCarrier)] macro expansion.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;

use proxima::telemetry::export::set_default_recorder;
use proxima::telemetry::id::{SpanId, TraceFlags, TraceId, format_traceparent};
use proxima::telemetry::pipes::{InMemoryPipe, NullPipe};
use proxima::telemetry::recorder::Recorder;
use proxima::telemetry::tag::{ScalarValue, Tag};
use proxima::telemetry::trace::{SpanCarrier, SpanKind, Status};
use proxima_macros::{SpanCarrier, instrument, span};
use rstest::rstest;

fn make_recorder() -> Recorder {
    Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed")
}

// ---- module-level annotated functions ----
// #[span] with recorder as an explicit parameter.

#[span(recorder = recorder)]
fn sync_fn(recorder: &Recorder, value: u64) -> u64 {
    value + 1
}

#[span(name = "explicit_name", recorder = recorder)]
fn explicitly_named_fn(recorder: &Recorder) -> &'static str {
    "ok"
}

#[span(level = "warn", recorder = recorder)]
fn warn_level_fn(recorder: &Recorder, input: u32) -> u32 {
    input * 2
}

#[span(recorder = recorder)]
async fn async_fn(recorder: &Recorder, value: u64) -> u64 {
    tokio::task::yield_now().await;
    value + 1
}

// 1. #[span] on a sync fn — span starts before body, record emitted after drop.
#[test]
fn span_on_sync_fn_emits_span_record() {
    let recorder = make_recorder();
    let result = sync_fn(&recorder, 41);
    assert_eq!(result, 42);
    let drained = recorder.drain();
    assert!(
        drained >= 1,
        "expected at least one span record, got {drained}"
    );
}

// 2. #[span] on an async fn — span survives across an await point.
#[tokio::test]
async fn span_on_async_fn_survives_await() {
    let recorder = make_recorder();
    let result = async_fn(&recorder, 41).await;
    assert_eq!(result, 42);
    let drained = recorder.drain();
    assert!(
        drained >= 1,
        "expected at least one span record, got {drained}"
    );
}

// 3. #[span(name = "explicit")] — explicit name compiles and runs.
#[test]
fn span_explicit_name_compiles_and_runs() {
    let recorder = make_recorder();
    let result = explicitly_named_fn(&recorder);
    assert_eq!(result, "ok");
    let drained = recorder.drain();
    assert!(drained >= 1);
}

// 4. #[span(level = "warn")] — explicit level is accepted without error.
#[test]
fn span_explicit_level_compiles_and_runs() {
    let recorder = make_recorder();
    let result = warn_level_fn(&recorder, 5);
    assert_eq!(result, 10);
    let drained = recorder.drain();
    assert!(drained >= 1);
}

// ---- ambient recorder: #[span] with no `recorder = ...` ----
// resolves the process-wide default installed via `set_default_recorder`; runs
// the body span-free when none is installed. This is the zero-wiring path.

#[span]
fn ambient_fn(value: u64) -> u64 {
    value + 1
}

#[test]
fn span_resolves_ambient_recorder() {
    // no recorder installed: the body still runs, just span-free (no panic).
    assert_eq!(ambient_fn(10), 11);

    let recorder = Arc::new(make_recorder());
    set_default_recorder(Arc::clone(&recorder));

    let result = ambient_fn(41);
    assert_eq!(result, 42);

    let drained = recorder.drain();
    assert!(
        drained >= 1,
        "expected a span via the ambient recorder, got {drained}"
    );
}

// #[instrument] is the unified-annotation alias for #[span] — same expansion.
#[instrument(recorder = recorder)]
fn instrumented_fn(recorder: &Recorder, value: u64) -> u64 {
    value + 2
}

#[test]
fn instrument_alias_records_span() {
    let recorder = make_recorder();
    let result = instrumented_fn(&recorder, 40);
    assert_eq!(result, 42);
    let drained = recorder.drain();
    assert!(
        drained >= 1,
        "expected a span via #[instrument], got {drained}"
    );
}

// #[span(budget = <ns>)] expands to a `.budget(<ns>)` chain (C5 tail-sampling).
// Under the default AlwaysOn sampler the span is head-kept, so this proves the
// arg compiles and threads through; the tail force-keep is covered in the
// recorder's own AlwaysOff tests.
#[span(recorder = recorder, budget = 1000)]
fn budgeted_fn(recorder: &Recorder, value: u64) -> u64 {
    value + 1
}

#[test]
fn span_budget_arg_compiles_and_records() {
    let recorder = make_recorder();
    assert_eq!(budgeted_fn(&recorder, 41), 42);
    let drained = recorder.drain();
    assert!(drained >= 1, "budgeted span still records, got {drained}");
}

// ---- enriched #[span]: kind, fields, arg capture, err ----

// a recorder whose drained records are inspectable (one handle in, one to read).
fn capturing() -> (Recorder, InMemoryPipe) {
    let pipe = InMemoryPipe::new();
    let recorder = Recorder::builder()
        .pipe(pipe.clone())
        .core_count(1)
        .start()
        .expect("recorder build failed");
    (recorder, pipe)
}

fn has_scalar(
    span: &proxima::telemetry::trace::SpanRecord,
    key: &str,
    value: &ScalarValue,
) -> bool {
    span.attrs.iter().any(|tag| {
        matches!(tag, Tag::Scalar { key: tag_key, value: tag_value } if *tag_key == key && tag_value == value)
    })
}

#[span(recorder = recorder, kind = "server")]
fn server_kind_fn(recorder: &Recorder) -> u8 {
    7
}

#[span(recorder = recorder, fields(component = "auth", attempt = 2u64))]
fn fielded_fn(recorder: &Recorder) -> u8 {
    0
}

#[span(recorder = recorder, fields(value))]
fn captures_arg_fn(recorder: &Recorder, value: u64) -> u64 {
    value
}

#[span(recorder = recorder, err)]
fn fallible_fn(recorder: &Recorder, ok: bool) -> Result<u32, &'static str> {
    if ok { Ok(1) } else { Err("boom") }
}

#[span(recorder = recorder, err)]
async fn fallible_async_fn(recorder: &Recorder, ok: bool) -> Result<u32, &'static str> {
    tokio::task::yield_now().await;
    if ok { Ok(1) } else { Err("boom") }
}

// kind = "server" lands on the SpanRecord as SpanKind::Server.
#[test]
fn span_kind_is_recorded() {
    let (recorder, pipe) = capturing();
    let _ = server_kind_fn(&recorder);
    recorder.drain();
    let spans = pipe.spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].kind, SpanKind::Server);
}

// fields(k = v) become typed scalar tags on the span.
#[test]
fn span_fields_are_tagged() {
    let (recorder, pipe) = capturing();
    let _ = fielded_fn(&recorder);
    recorder.drain();
    let spans = pipe.spans();
    assert_eq!(spans.len(), 1);
    assert!(has_scalar(
        &spans[0],
        "component",
        &ScalarValue::Str("auth")
    ));
    assert!(has_scalar(&spans[0], "attempt", &ScalarValue::U64(2)));
}

// fields(bare_ident) captures the named argument by value.
#[test]
fn span_captures_named_arg() {
    let (recorder, pipe) = capturing();
    let out = captures_arg_fn(&recorder, 99);
    assert_eq!(out, 99);
    recorder.drain();
    let spans = pipe.spans();
    assert!(has_scalar(&spans[0], "value", &ScalarValue::U64(99)));
}

// err: a returned Err flips the span status to Error.
#[test]
fn span_err_sets_error_status_on_err() {
    let (recorder, pipe) = capturing();
    let result = fallible_fn(&recorder, false);
    assert!(result.is_err());
    recorder.drain();
    let spans = pipe.spans();
    assert_eq!(spans.len(), 1);
    assert!(matches!(spans[0].status, Status::Error { .. }));
}

// err: an Ok return leaves the status unset (no false error).
#[test]
fn span_err_leaves_status_unset_on_ok() {
    let (recorder, pipe) = capturing();
    let result = fallible_fn(&recorder, true);
    assert_eq!(result, Ok(1));
    recorder.drain();
    let spans = pipe.spans();
    assert!(matches!(spans[0].status, Status::Unset));
}

// err threads through an await point on an async fn too.
#[tokio::test]
async fn span_err_async_sets_error_status() {
    let (recorder, pipe) = capturing();
    let result = fallible_async_fn(&recorder, false).await;
    assert!(result.is_err());
    recorder.drain();
    let spans = pipe.spans();
    assert!(matches!(spans[0].status, Status::Error { .. }));
}

// ---- explicit propagation: #[span(parent = ...)] ----
// proxima never reads an ambient/thread-local "current span" — a child link
// must be carried by hand as an `Option<&[u8]>` W3C traceparent.

#[span(recorder = recorder, parent = parent)]
fn parented_fn(recorder: &Recorder, parent: Option<&[u8]>) -> u8 {
    1
}

// `parent = Some(bytes)` continues the caller's trace: same trace_id, and the
// caller's span_id lands as parent_span_id.
#[test]
fn span_parent_arg_continues_the_trace() {
    let (recorder, pipe) = capturing();

    let parent_trace = TraceId::from_bytes([0x33; 16]);
    let parent_span = SpanId::from_bytes([0x44; 8]);
    let traceparent = format_traceparent(&parent_trace, &parent_span, TraceFlags::SAMPLED);

    let _ = parented_fn(&recorder, Some(&traceparent));
    recorder.drain();

    let spans = pipe.spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(
        spans[0].trace_id, parent_trace,
        "child inherits the caller's trace_id via the explicit parent arg"
    );
    assert_eq!(
        spans[0].parent_span_id,
        Some(parent_span),
        "child records the caller's span_id as its parent"
    );
}

// `parent = None` (no context carried) falls back to a fresh root, same as
// omitting the arg entirely.
#[test]
fn span_parent_arg_none_is_a_fresh_root() {
    let (recorder, pipe) = capturing();

    let _ = parented_fn(&recorder, None);
    recorder.drain();

    let spans = pipe.spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(
        spans[0].parent_span_id, None,
        "no parent carried -- fresh root, exactly like no `parent` arg at all"
    );
}

// 5. #[derive(SpanCarrier)] on a struct with `span_id: Option<SpanId>` — generated impl works.
#[derive(SpanCarrier)]
struct BasicEnvelope {
    span_id: Option<SpanId>,
    payload: Vec<u8>,
}

#[test]
fn span_carrier_default_field_name() {
    let mut envelope = BasicEnvelope {
        span_id: None,
        payload: vec![1, 2, 3],
    };

    assert_eq!(envelope.span_id(), None);
    assert_eq!(envelope.payload.len(), 3);

    let id = SpanId::from_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
    envelope.set_span_id(Some(id));
    assert_eq!(envelope.span_id(), Some(id));

    envelope.set_span_id(None);
    assert_eq!(envelope.span_id(), None);
}

// 6. #[derive(SpanCarrier)] with #[span_id] attribute on a differently-named field — works.
#[derive(SpanCarrier)]
struct AltEnvelope {
    #[span_id]
    trace_slot: Option<SpanId>,
    body: String,
}

#[test]
fn span_carrier_attr_annotated_field() {
    let mut envelope = AltEnvelope {
        trace_slot: None,
        body: String::from("hello"),
    };

    assert_eq!(envelope.span_id(), None);
    assert_eq!(envelope.body, "hello");

    let id = SpanId::from_bytes([0xaa, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44]);
    envelope.set_span_id(Some(id));
    assert_eq!(envelope.span_id(), Some(id));
}

// 7. SpanCarrier on a generic struct.
#[derive(SpanCarrier)]
struct GenericEnvelope<T> {
    span_id: Option<SpanId>,
    data: T,
}

#[rstest]
#[case::value_42(42u32)]
#[case::value_0(0u32)]
fn span_carrier_generic_struct(#[case] value: u32) {
    let mut env = GenericEnvelope {
        span_id: None,
        data: value,
    };
    let id = SpanId::from_bytes([1; 8]);
    env.set_span_id(Some(id));
    assert_eq!(env.span_id(), Some(id));
    assert_eq!(env.data, value);
}
