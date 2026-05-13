#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
// representative slice converted from tokio::test to proxima::test: these
// recording round-trips now drive on proxima's runtime (prime under test-prime).
// gated on the harness feature so the default build is unaffected; the
// proxima-test lane runs them with `--features test-prime`. See
// docs/proxima-test/discipline.md.
//
// also needs the full prime runtime bundle: `prime()` below constructs a
// `PrimeRuntime` directly, which is gated behind executor + inbox-alloc +
// reactor + bgpool together (see src/runtime.rs) — `test-support` alone
// only pulls in `macros`.
#![cfg(all(
    feature = "test-support",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use proxima::recording::sink::RecordingSink;
use proxima::recording::{BoundedRecordingSink, DynRecordingSink};
use proxima::{
    AccumulatingSink, BinSource, FailMode, FormatKind, HttpEvent, InteractionId, JsonlSource,
    LazyFanOut, ProtocolEvent, RecordMeta, RecordingEvent, RecordingRequestHeader, RecordingSource,
    SinkSpec,
};
use tempfile::tempdir;
use time::OffsetDateTime;

fn prime() -> Arc<dyn proxima::runtime::Runtime> {
    Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
}

fn make_sink(path: &std::path::Path, format: FormatKind) -> AccumulatingSink {
    let spigot = proxima::deferred_runtime();
    let _ = spigot.set(
        std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
            as std::sync::Arc<dyn proxima::runtime::Runtime>,
    );
    let durable = std::sync::Arc::new(LazyFanOut::new(
        vec![SinkSpec::new(path.to_string_lossy().into_owned(), format)],
        spigot,
    ));
    AccumulatingSink::new(durable, 1)
}

fn fixed_id(seed: u8) -> InteractionId {
    InteractionId::from_bytes([seed; 16])
}

fn sample_interaction(id: InteractionId) -> Vec<RecordingEvent> {
    let envelope = |ts_ms: u64, event: HttpEvent| RecordingEvent {
        id,
        ts_ms,
        parent: None,
        event: ProtocolEvent::Http(event),
    };
    vec![
        envelope(
            0,
            HttpEvent::Started {
                ts: OffsetDateTime::UNIX_EPOCH,
                pipe: "echo".into(),
                request: RecordingRequestHeader {
                    method: "POST".into(),
                    path: "/v1/chat/completions".into(),
                    headers: [("accept".into(), "application/json".into())]
                        .into_iter()
                        .collect(),
                    query: Default::default(),
                },
                meta: None,
            },
        ),
        envelope(
            3,
            HttpEvent::RequestChunk {
                data: Bytes::from_static(br#"{"x":1}"#),
                metadata: Default::default(),
            },
        ),
        envelope(4, HttpEvent::RequestEnded),
        envelope(
            42,
            HttpEvent::ResponseStarted {
                status: 200,
                headers: vec![("content-type".into(), "application/json".into())],
            },
        ),
        envelope(
            43,
            HttpEvent::ResponseChunk {
                data: Bytes::from_static(br#"{"y":2}"#),
                metadata: Default::default(),
            },
        ),
        envelope(
            80,
            HttpEvent::ResponseChunk {
                data: Bytes::from_static(br#"{"y":3}"#),
                metadata: Default::default(),
            },
        ),
        envelope(
            510,
            HttpEvent::Ended {
                latency_ms: 468,
                meta: RecordMeta {
                    cache: Some(proxima::CacheOutcome::Miss),
                    retries: 0,
                    upstream: Some("echo".into()),
                    instance_id: Some("abc".into()),
                    source: None,
                    extra: Default::default(),
                },
            },
        ),
    ]
}

async fn read_back<S: RecordingSource>(source: &S) -> Vec<RecordingEvent> {
    let stream = source.events();
    let mut events: Vec<RecordingEvent> = Vec::new();
    let mut iterator = stream;
    while let Some(item) = iterator.next().await {
        events.push(item.expect("event stream item"));
    }
    events
}

#[proxima::test]
async fn jsonl_sink_then_source_round_trips_interaction_byte_equivalent() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("echo.jsonl");
    let sink = make_sink(&path, FormatKind::Json);
    let original = sample_interaction(fixed_id(1));
    for event in &original {
        sink.append(event.clone()).await.expect("append");
    }
    sink.flush().await.expect("flush");

    let source = JsonlSource::new(&path, prime());
    let read = read_back(&source).await;
    assert_eq!(read, original);
}

#[proxima::test]
async fn bin_sink_then_source_round_trips_interaction_byte_equivalent() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("echo.bin");
    let sink = make_sink(&path, FormatKind::Bin);
    let original = sample_interaction(fixed_id(2));
    for event in &original {
        sink.append(event.clone()).await.expect("append");
    }
    sink.flush().await.expect("flush");

    let source = BinSource::new(&path, prime());
    let read = read_back(&source).await;
    assert_eq!(read, original);
}

#[proxima::test]
async fn interleaved_interactions_demux_correctly_on_jsonl() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("interleaved.jsonl");
    let sink = make_sink(&path, FormatKind::Json);
    let alpha = fixed_id(10);
    let beta = fixed_id(11);
    let alpha_events = sample_interaction(alpha);
    let beta_events = sample_interaction(beta);
    // simulate interleave: alpha-start, beta-start, alpha-req-chunk, beta-req-chunk, ...
    let mut interleaved: Vec<RecordingEvent> = Vec::new();
    for index in 0..alpha_events.len() {
        interleaved.push(alpha_events[index].clone());
        interleaved.push(beta_events[index].clone());
    }
    for event in &interleaved {
        sink.append(event.clone()).await.expect("append");
    }
    sink.flush().await.expect("flush");

    let source = JsonlSource::new(&path, prime());
    let read = read_back(&source).await;
    let alpha_recovered: Vec<RecordingEvent> = read
        .iter()
        .filter(|event| event.id() == alpha)
        .cloned()
        .collect();
    let beta_recovered: Vec<RecordingEvent> = read
        .iter()
        .filter(|event| event.id() == beta)
        .cloned()
        .collect();
    assert_eq!(alpha_recovered, alpha_events);
    assert_eq!(beta_recovered, beta_events);
}

// timing-sensitive: relies on the sink worker NOT draining between appends, which
// holds on a single-threaded runtime. Selects tokio explicitly — on prime the
// worker runs on the concurrent sister runtime and would drain first (no drop).
#[proxima::test(runtime = "tokio")]
async fn bounded_sink_drops_oldest_when_capacity_exceeded() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bounded.jsonl");
    let backend: DynRecordingSink = Arc::new(make_sink(&path, FormatKind::Json));
    let bounded = BoundedRecordingSink::new(backend.clone(), 2, FailMode::DropOldest);
    let id = fixed_id(20);
    // pre-fill the queue past capacity by appending three events without giving the worker
    // a chance to drain — third event must trigger an eviction. The sink's worker will then
    // observe the queue and write the survivors. We validate the on-disk count + drop counter.
    for ts in 0..3 {
        bounded
            .append(RecordingEvent {
                id,
                ts_ms: ts,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::RequestEnded),
            })
            .await
            .expect("append");
    }
    bounded.flush().await.expect("flush");
    let source = JsonlSource::new(&path, prime());
    let read = read_back(&source).await;
    let drained = bounded.drained();
    let dropped = bounded.dropped();
    assert_eq!(
        drained + dropped,
        3,
        "every appended event must be drained or dropped"
    );
    assert_eq!(
        read.len() as u64,
        drained,
        "on-disk count must match drained"
    );
    assert!(
        dropped >= 1,
        "at least one drop must have occurred when oversubscribing"
    );
}

#[proxima::test]
async fn golden_fixture_loads_six_events_in_order() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    // post-Phase-A: umbrella crate manifest is at workspace root; spec/ is a sibling.
    let fixture_path = Path::new(&manifest_dir).join("spec/examples/recording-http.jsonl");
    let source = JsonlSource::new(&fixture_path, prime());
    let events = read_back(&source).await;
    assert_eq!(events.len(), 6);
    assert!(matches!(
        events[0].event,
        ProtocolEvent::Http(HttpEvent::Started { .. })
    ));
    assert!(matches!(
        events[1].event,
        ProtocolEvent::Http(HttpEvent::RequestChunk { .. })
    ));
    assert!(matches!(
        events[2].event,
        ProtocolEvent::Http(HttpEvent::RequestEnded)
    ));
    assert!(matches!(
        events[3].event,
        ProtocolEvent::Http(HttpEvent::ResponseStarted { .. })
    ));
    assert!(matches!(
        events[4].event,
        ProtocolEvent::Http(HttpEvent::ResponseChunk { .. })
    ));
    assert!(matches!(
        events[5].event,
        ProtocolEvent::Http(HttpEvent::Ended { .. })
    ));
}
