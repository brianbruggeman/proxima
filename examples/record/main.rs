#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Wrap any `Pipe` in `RecordUpstream` and every (request, response)
//! interaction tees to a cassette as it flows — no separate "record mode",
//! the tee *is* the pipe. Builds on `transform`: recording is just another
//! `Pipe`, `In -> Out` unchanged, wrapped around the one you already wrote.
//!
//! The drainer that appends to the cassette runs off the hot path (a
//! background task, not this call), so the recording isn't durable the
//! instant `call` returns. `TerminalSignal` (see the `signal` example)
//! wraps the sink and fires once the terminal `Ended` event has been
//! appended AND flushed — `terminal.drained().await` replaces polling the
//! cassette from outside.
//!
//! Run: `cargo run --example record`

use std::path::Path;
use std::sync::Arc;

use futures::StreamExt;
use proxima::runtime::PrimeRuntime;
use proxima::{
    AccumulatingSink, DynRecordingSink, FormatKind, HttpEvent, JsonlSource, LazyFanOut,
    ProtocolEvent, RecordUpstream, RecordingEvent, RecordingSource, Request, Runtime, SendPipe,
    SinkSpec, SynthUpstream, TerminalSignal, deferred_runtime, into_handle,
};

#[tokio::main]
async fn main() {
    let cassette_dir = tempfile::tempdir().expect("tempdir");
    let cassette_path = cassette_dir.path().join("session.jsonl");

    // any Pipe can sit behind the recorder; this one is canned so the demo
    // has no real network dependency.
    let inner = into_handle(SynthUpstream::new("echo", 200, "hello from the wire"));

    // the LazyFanOut's spigot must be armed for the cassette to open at all —
    // disarmed, `RecordUpstream` still runs the call, it just writes nothing.
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
    let spigot = deferred_runtime();
    spigot.set(Arc::clone(&runtime)).ok();
    let durable = Arc::new(LazyFanOut::new(
        vec![SinkSpec::new(
            cassette_path.to_string_lossy(),
            FormatKind::Json,
        )],
        spigot,
    ));
    let accumulating: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));

    // TerminalSignal wraps the durable sink and fires once it has appended AND
    // flushed the interaction's terminal HTTP event — a real completion signal
    // instead of a disk-repolling loop.
    let terminal = Arc::new(TerminalSignal::new(
        accumulating,
        |event: &RecordingEvent| {
            matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. }))
        },
    ));
    let sink: DynRecordingSink = terminal.clone();

    let recorder = RecordUpstream::new("recorded", inner, sink, "echo");

    let request = Request::builder()
        .method("POST")
        .path("/v1/chat")
        .body("what is a cassette?")
        .build()
        .expect("request");

    let response = SendPipe::call(&recorder, request).await.expect("call");
    let status = response.status;
    let served_body = response
        .collect_body()
        .await
        .expect("collect response body");
    println!("--- capture: live traffic through RecordUpstream ---");
    println!(
        "served: {status} {:?}",
        String::from_utf8_lossy(&served_body)
    );

    // the drainer appends off the hot path (a background task, not this call),
    // so the cassette isn't durable the instant `call` returns — await the
    // Signal instead of polling for it: no loop, no retry count, no sleep.
    println!("--- replay: reading the cassette back ---");
    println!("  awaiting terminal.drained() (parked, not polled)...");
    terminal.drained().await;
    let events = read_cassette(&cassette_path, Arc::clone(&runtime)).await;

    let mut captured_body = Vec::new();
    for event in &events {
        match &event.event {
            ProtocolEvent::Http(HttpEvent::Started { request, .. }) => {
                println!("  Started:  {} {}", request.method, request.path);
            }
            ProtocolEvent::Http(HttpEvent::ResponseStarted { status, .. }) => {
                println!("  ResponseStarted: {status}");
            }
            ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => {
                captured_body.extend_from_slice(data);
                println!("  ResponseChunk: {} bytes", data.len());
            }
            ProtocolEvent::Http(HttpEvent::Ended { .. }) => {
                println!("  Ended");
            }
            _ => {}
        }
    }

    assert_eq!(
        &captured_body[..],
        &served_body[..],
        "the cassette must capture exactly the bytes served"
    );
    println!(
        "--- proof: {} bytes served == {} bytes captured ---",
        served_body.len(),
        captured_body.len()
    );
}

// `terminal.drained().await` already guarantees the terminal event is
// durable, so this is a single read, not a retry loop.
async fn read_cassette(path: &Path, runtime: Arc<dyn Runtime>) -> Vec<RecordingEvent> {
    let source = JsonlSource::new(path, runtime);
    let mut events = source.events();
    let mut collected = Vec::new();
    while let Some(event) = events.next().await {
        collected.push(event.expect("read recording event"));
    }
    collected
}
