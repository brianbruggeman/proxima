# Build a record/replay harness

**Prerequisites:** [Foundations](./00-foundations.md) — the **transform** and **observe** roles, `into_handle`, and `Signal` (section 12).
**You will:** record live traffic through any `Pipe` to a cassette, then replay it byte-identical for tests — no upstream call. The recorder is just another `Pipe` wrapped around yours; replay serves the cassette back.
**New concepts (in order):** record (`RecordUpstream` + a sink chain) · a fire-once completion `Signal` (durability await) · replay (`ReplayUpstream`, match key, typed miss).
**Answer keys:** [`examples/record/main.rs`](../../examples/record/main.rs), [`examples/replay/main.rs`](../../examples/replay/main.rs) — `cargo run --example record` / `--example replay`.

## Part A — Record: the tee is the pipe

`RecordUpstream` wraps any `Pipe` — a **transform**, `In -> Out` unchanged — and tees every (request, response) interaction to a cassette as it flows. There is no "record mode"; the recorder **is** a `Pipe` (`src/upstreams/record.rs:68, 103-108`):

```rust
use std::sync::Arc;

use proxima::runtime::PrimeRuntime;
use proxima::{
    AccumulatingSink, DynRecordingSink, FormatKind, HttpEvent, LazyFanOut, ProtocolEvent,
    RecordUpstream, RecordingEvent, Request, Runtime, SendPipe, SinkSpec, SynthUpstream,
    TerminalSignal, deferred_runtime, into_handle,
};

let cassette_dir = tempfile::tempdir().expect("tempdir");
let cassette_path = cassette_dir.path().join("session.jsonl");

// any Pipe can sit behind the recorder; this one is canned so the tutorial
// has no real network dependency (`SynthUpstream::new(name, status, body)`).
let inner = into_handle(SynthUpstream::new("echo", 200, "hello from the wire"));

// the LazyFanOut's spigot must be armed for the cassette to open at all —
// disarmed, `RecordUpstream` still runs the call, it just writes nothing.
let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
let spigot = deferred_runtime();
spigot.set(Arc::clone(&runtime)).ok();
let durable = Arc::new(LazyFanOut::new(
    vec![SinkSpec::new(cassette_path.to_string_lossy(), FormatKind::Json)],
    spigot,
));
let accumulating: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));

// `TerminalSignal` wraps the durable sink and fires once it has appended AND
// flushed the interaction's terminal HTTP event — a real completion signal,
// not a disk-repolling loop.
let terminal = Arc::new(TerminalSignal::new(
    accumulating,
    |event: &RecordingEvent| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })),
));
let sink: DynRecordingSink = terminal.clone();

// the trailing "echo" is a separate label: the pipe name RecordUpstream
// writes into every event it records, distinct from SynthUpstream's own
// "echo" name above.
let recorder = RecordUpstream::new("recorded", inner, sink, "echo");

let request = Request::builder()
    .method("POST")
    .path("/v1/chat")
    .body("what is a cassette?")
    .build()
    .expect("request");

let response = SendPipe::call(&recorder, request).await.expect("call"); // flows through, and tees
let served_status = response.status;
let served_body = response
    .collect_body()
    .await
    .expect("collect response body");
assert_eq!(served_status, 200);
```

`SynthUpstream` is `into_handle`d exactly like any other `Pipe` from Foundations — the recorder does not know or care that it is canned.

The sink `RecordUpstream` writes through is a small chain (`record/main.rs:42-59`):

- `LazyFanOut` — the drainer — writes to a JSONL cassette file (`SinkSpec::new(path, FormatKind::Json)`).
- `AccumulatingSink` batches the interaction's events.
- `TerminalSignal` wraps the sink and fires once the terminal `Ended` event has been appended **and** flushed.

Because the drainer appends off the hot path (a background task, not the `call`), the cassette is not durable the instant `call` returns. Instead of polling the file, **await the Signal** — the **observe** idea from Foundations plus a completion you can wait on (see [`examples/signal`](../../examples/signal)) — then read the cassette back as a stream of `RecordingEvent`s and confirm served bytes == captured bytes (`record/main.rs:86-120`):

```rust
use futures::StreamExt;
use proxima::{JsonlSource, RecordingSource};

terminal.drained().await; // parked, not polled — no loop, no retry count, no sleep

let source = JsonlSource::new(&cassette_path, Arc::clone(&runtime));
let mut events = source.events();
let mut captured_body = Vec::new();
while let Some(event) = events.next().await {
    let event = event.expect("read recording event");
    if let ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) = event.event {
        captured_body.extend_from_slice(&data);
    }
}

assert_eq!(
    &captured_body[..],
    &served_body[..],
    "the cassette must capture exactly the bytes served"
);
```

## Part B — Replay: serve the cassette byte-identical

`ReplayUpstream` loads a cassette and serves it back — same status, headers, and chunk framing — with **no** upstream call (`proxima-recording/src/replay/mod.rs:71-77, 85-92, 165-182`). Replay matches a request by method + path (+ query/body); `replay.known_keys()` lists what is captured:

```rust
use std::collections::BTreeMap;

use bytes::Bytes;
use futures::StreamExt;
use time::OffsetDateTime;

use proxima::{
    InteractionId, ProximaError, RecordMeta, RecordingRequestHeader, ReplayUpstream,
};
use proxima_recording::{Format, JsonFormat};

let cassette_dir = tempfile::tempdir().expect("tempdir");
let cassette_path = cassette_dir.path().join("chat.jsonl");

// build a cassette by hand — the same RecordingEvent/HttpEvent shape
// RecordUpstream's sink chain wrote in Part A — so this block stands on its
// own without re-running a live recording.
let response_headers = vec![
    ("content-type".to_string(), "application/json".to_string()),
    ("x-cassette".to_string(), "replay-demo".to_string()),
];
let response_chunks = vec![
    Bytes::from_static(br#"{"delta":"Hel"}"#),
    Bytes::from_static(br#"{"delta":"lo, "}"#),
    Bytes::from_static(br#"{"delta":"world"}"#),
];

let interaction = InteractionId::from_bytes([21; 16]);
let envelope = |ts_ms: u64, event: HttpEvent| RecordingEvent {
    id: interaction,
    ts_ms,
    parent: None,
    event: ProtocolEvent::Http(event),
};
let request_header = RecordingRequestHeader {
    method: "POST".to_string(),
    path: "/v1/chat/completions".to_string(),
    headers: BTreeMap::from([("accept".to_string(), "application/json".to_string())]),
    query: BTreeMap::from([("model".to_string(), "gpt-mini".to_string())]),
};
let mut events = vec![
    envelope(
        0,
        HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: "chat-upstream".to_string(),
            request: request_header,
            meta: None,
        },
    ),
    envelope(
        1,
        HttpEvent::RequestChunk {
            data: Bytes::from_static(br#"{"prompt":"say hi"}"#),
            metadata: Default::default(),
        },
    ),
    envelope(2, HttpEvent::RequestEnded),
    envelope(
        3,
        HttpEvent::ResponseStarted {
            status: 200,
            headers: response_headers.clone(),
        },
    ),
];
for (offset, chunk) in response_chunks.iter().enumerate() {
    events.push(envelope(
        4 + offset as u64,
        HttpEvent::ResponseChunk {
            data: chunk.clone(),
            metadata: Default::default(),
        },
    ));
}
events.push(envelope(
    4 + response_chunks.len() as u64,
    HttpEvent::Ended {
        latency_ms: 12,
        meta: RecordMeta::default(),
    },
));
let cassette_bytes = JsonFormat::new()
    .encode_block(events)
    .expect("encode cassette");
std::fs::write(&cassette_path, &cassette_bytes).expect("write cassette");

let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
let replay = ReplayUpstream::from_jsonl(&cassette_path, "chat-replay", runtime)
    .await
    .expect("load cassette"); // straight off disk
assert_eq!(
    replay.known_keys(),
    vec!["POST /v1/chat/completions?model=gpt-mini".to_string()]
);

let request = Request::builder()
    .method("POST")
    .path("/v1/chat/completions")
    .query_param("model", "gpt-mini")
    .build()
    .expect("request");
let response = SendPipe::call(&replay, request).await.expect("call");
assert_eq!(response.status, 200);

let replayed_headers: Vec<(String, String)> = response
    .metadata
    .iter()
    .map(|(name, value)| {
        (
            String::from_utf8_lossy(name.as_ref()).into_owned(),
            String::from_utf8_lossy(value.as_ref()).into_owned(),
        )
    })
    .collect();
assert_eq!(replayed_headers, response_headers);

let mut replayed_chunks = Vec::new();
let mut chunk_stream = response.into_chunk_stream();
while let Some(chunk) = chunk_stream.next().await {
    replayed_chunks.push(chunk.expect("chunk"));
}
assert_eq!(replayed_chunks, response_chunks); // byte-identical, chunk boundaries included

// replay never guesses: a request that was never captured is a typed miss
// (`proxima-recording/src/replay/mod.rs:174-179`), not a wrong-body 200 —
// the flip side of "byte-identical".
let unrecorded = Request::builder()
    .method("GET")
    .path("/v1/never-recorded")
    .build()
    .expect("request");

match SendPipe::call(&replay, unrecorded).await {
    Err(ProximaError::ReplayMiss { fingerprint }) => {
        // clean, typed miss — no invented body
        assert_eq!(fingerprint, "GET /v1/never-recorded?");
    }
    Err(other) => panic!("unexpected error: {other:?}"),
    Ok(_) => panic!("replay must not invent a response for an unrecorded request"),
}
```

`"chat-replay"` labels this `ReplayUpstream` pipe — the replay-side counterpart to the `"recorded"` label on `RecordUpstream::new` in Part A, not part of the cassette file itself.

The cassette `record` writes and `replay` reads is the same event-log format — Part B built one by hand above using exactly the `RecordingEvent`/`HttpEvent` shape Part A's sink chain wrote to `session.jsonl`.

## What you built

- **record** — `RecordUpstream` wraps any `Pipe`; the tee is the pipe; a `TerminalSignal` tells you when the cassette is durable (await, don't poll).
- **replay** — `ReplayUpstream` serves the cassette back byte-identical, and misses cleanly (`ReplayMiss`) on anything never captured.

Front a third-party API with `RecordUpstream` in one run, then swap in `ReplayUpstream` for your tests — same cassette, byte-identical, no network. Both are ordinary `Pipe`s; the harness is composition, not a mock framework.
