//! A span OBSERVES an operation's SCOPE (start..end, and its place in the
//! request tree) — `transform`'s `Tap` is the observe form over a single
//! value; a span is the same form aimed at a duration instead.
//!
//! Two spans below, parent and child, nest across a real async-task boundary
//! (`tokio::spawn`). proxima keeps span context as EXPLICIT DATA — a W3C
//! `traceparent`, the same value that crosses a network hop in
//! `distributed-trace` — never an ambient/thread-local "current span" the
//! way `tracing` has. A naive thread-local "current span" stack would
//! corrupt under proxima's own use case (many concurrently-interleaved
//! requests sharing one executor thread): task A opens a span, awaits and
//! yields the thread, task B runs and would see A's still-pushed span as
//! ITS parent. So the parent/child link a caller wants MUST be carried by
//! hand — `#[proxima::telemetry::instrument(parent = ...)]` is that explicit
//! seam: `parent = <expr>` takes an `Option<&[u8]>` W3C traceparent and,
//! when `Some`, continues that trace instead of opening a fresh root.
//! `validate_request` below is called TWICE to prove both ends of that seam:
//! once with `parent = None` (nothing carried -> its own root, same as
//! omitting the arg) and once with the caller's context threaded through
//! explicitly (-> a connected child).
//!
//! Run: cargo run --example traces

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use bytes::Bytes;
use proxima::HeaderList;
use proxima::telemetry::pipes::InMemoryPipe;
use proxima::telemetry::recorder::Recorder;
use proxima::telemetry::propagation::TRACEPARENT;

const REQUEST_KEY: &str = "user:42";

/// A leaf op instrumented with the sugar form PLUS the explicit `parent =`
/// seam: with `parent = None` it opens its own root (proven by the
/// `unparented` call in `main`); handed the caller's traceparent bytes it
/// joins the caller's trace instead — proxima's only propagation mechanism,
/// never ambient.
#[proxima::telemetry::instrument(name = "validate_request", parent = parent)]
fn validate_request(key: &str, parent: Option<&[u8]>) -> bool {
    !key.is_empty()
}

/// The parent scope. Opens the same way `#[instrument]` sugars over
/// (`recorder.span(name)...start()`), then hands its own context to a
/// same-scope callee via `parent =` AND across a `tokio::spawn` boundary as
/// an owned value before awaiting the child back.
async fn handle_request(recorder: Arc<Recorder>) -> u64 {
    let parent = recorder
        .span("handle_request")
        .tag("key", REQUEST_KEY)
        .start();
    println!("handle_request: opened span {:?}", parent.id());

    // capture the parent's context as a plain value BEFORE crossing any
    // boundary. This inject/extract pair is the only propagation mechanism
    // proxima has, in-process or over the wire — there is no ambient
    // "current span" for a callee to read instead.
    let mut headers = HeaderList::new();
    parent.inject(&mut headers);
    let traceparent: Bytes = headers
        .get(TRACEPARENT)
        .expect("an active span always injects a traceparent")
        .clone();

    // proof of the negative: called with nothing carried, the sugar form is
    // its own root, exactly like today's behavior with no `parent` arg.
    let _ = validate_request(REQUEST_KEY, None);

    // proof of the fix: the SAME sugar-form call, handed the parent's
    // context explicitly via `parent =`, joins the tree instead.
    let _ = validate_request(REQUEST_KEY, Some(traceparent.as_ref()));

    let result = tokio::spawn(run_query(Arc::clone(&recorder), traceparent))
        .await
        .expect("spawned query task did not panic");

    // dropping here (rather than letting scope-end do it implicitly) makes
    // the ordering explicit: the parent's recorded duration covers the
    // child's entire run, because it closes only after the child returns.
    drop(parent);
    result
}

/// The child scope, opened on the OTHER side of the `tokio::spawn` boundary
/// — a different task, possibly a different OS thread under the
/// multi-thread runtime — from a traceparent it was handed, not one it
/// inherited for free.
async fn run_query(recorder: Arc<Recorder>, traceparent: Bytes) -> u64 {
    let child = recorder
        .span_from_traceparent("run_query", traceparent.as_ref())
        .tag("table", "users")
        .start();
    println!(
        "  run_query:      opened span {:?}, carried in from handle_request",
        child.id()
    );
    42
}

#[tokio::main]
async fn main() {
    println!("proxima traces: spans across an async boundary\n");

    let pipe = InMemoryPipe::new();
    let recorder = Recorder::builder()
        .pipe(pipe.clone())
        .core_count(1)
        .install()
        .expect("recorder install failed");

    let value = handle_request(Arc::clone(&recorder)).await;
    println!("\nhandle_request -> {value}\n");

    recorder.drain();
    let spans = pipe.spans();

    println!("span tree ({} spans captured):", spans.len());
    for span in &spans {
        let parent = span
            .parent_span_id
            .map_or_else(|| "-".to_string(), |id| id.to_string());
        println!(
            "  name={:<17} trace_id={} span_id={} parent_span_id={parent}",
            span.name, span.trace_id, span.span_id
        );
    }

    let parent_span = spans
        .iter()
        .find(|span| span.name == "handle_request")
        .expect("parent span recorded");
    let child_span = spans
        .iter()
        .find(|span| span.name == "run_query")
        .expect("child span recorded, carried across tokio::spawn");
    let unparented_leaf = spans
        .iter()
        .find(|span| span.name == "validate_request" && span.parent_span_id.is_none())
        .expect("sugar-form leaf called with parent = None recorded");
    let parented_leaf = spans
        .iter()
        .find(|span| span.name == "validate_request" && span.parent_span_id.is_some())
        .expect("sugar-form leaf called with parent = Some(..) recorded");

    assert_eq!(
        child_span.trace_id, parent_span.trace_id,
        "child must inherit the parent's trace_id across the spawn boundary"
    );
    assert_eq!(
        child_span.parent_span_id,
        Some(parent_span.span_id),
        "child must record the parent's span_id -- proof the tree is connected"
    );
    assert_ne!(
        unparented_leaf.trace_id, parent_span.trace_id,
        "#[instrument(parent = None)] must be a fresh root -- propagation is explicit, never ambient"
    );
    assert_eq!(
        parented_leaf.trace_id, parent_span.trace_id,
        "#[instrument(parent = Some(..))] must inherit the caller's trace_id -- \
         proof the explicit seam connects the sugar form"
    );
    assert_eq!(
        parented_leaf.parent_span_id,
        Some(parent_span.span_id),
        "#[instrument(parent = Some(..))] must record the caller's span_id as its parent"
    );

    println!("\n-> run_query IS a child of handle_request: same trace_id, parent_span_id set");
    println!(
        "-> validate_request(parent = None) is its OWN root: no context carried, no auto-parent"
    );
    println!("-> validate_request(parent = Some(..)) IS a child: the explicit seam connects it");
    println!("\nPASS: span context survives both the tokio::spawn boundary and a same-scope call,");
    println!("      via explicit data in both cases -- never ambient.");
}
