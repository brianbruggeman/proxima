//! Integration test: capture → BinSink → replay round trip.
//!
//! Spans three crates without any network or live proxy:
//!   proxima-intercept   (Capture)            writes typed HttpEvents
//!   proxima-recording-core (BinSink/BinSource) durable on-disk format
//!   proxima-replay      (ReplayUpstream)      reads events back, serves as a Pipe
//!
//! This is the contract that makes capture *useful* — a recorded interaction
//! must be reconstructable into a Response that matches what the upstream
//! originally sent. Unit tests cover each crate in isolation; this proves the
//! composition end to end.
//!
//! Gated on `intercept-replay` (= capture + replay). A no-op when the feature
//! is off, so `cargo test` under default features still passes.
#![cfg(feature = "intercept-replay")]
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
use std::time::Instant;

use bytes::Bytes;
use futures::StreamExt;
use proxima_intercept::capture::Capture;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Request;
use proxima_recording::BinSource;
use proxima_recording::replay::ReplayUpstream;
use proxima_recording::source::DynRecordingSource;

// an armed spigot: capture opens + pumps once a runtime backs the off-core
// blocking I/O. a 1-core prime runtime is the default backend.
fn armed_spigot() -> proxima_recording::pipe::DeferredRuntime {
    let spigot = proxima_recording::pipe::deferred_runtime();
    spigot
        .set(
            std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
                as std::sync::Arc<dyn proxima::runtime::Runtime>,
        )
        .ok();
    spigot
}

fn prime() -> std::sync::Arc<dyn proxima::runtime::Runtime> {
    std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
}

#[proxima::test]
async fn captured_post_interaction_replays_with_matching_status_and_body() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("recording.bin");

    // record a synthetic POST interaction the way the proxy does
    let request_wire = b"POST /responses HTTP/1.1\r\n\
                         Host: api.individual.example.com\r\n\
                         Authorization: Bearer secret-token-value\r\n\
                         Content-Type: application/json\r\n\r\n";
    let response_wire = b"HTTP/1.1 200 OK\r\n\
                          Content-Type: application/json\r\n\
                          Content-Length: 24\r\n\r\n";
    let response_body = br#"{"output":[{"id":"42"}]}"#;

    {
        let capture = Capture::open(&data_path, armed_spigot()).expect("open capture");
        let recorder = capture
            .begin("api.individual.example.com", request_wire, Instant::now())
            .await
            .expect("begin");
        recorder.push_request(Bytes::from_static(br#"{"model":"model-nano"}"#));
        recorder.push_response(Bytes::from_static(response_body));
        recorder.finish(response_wire).await.expect("finish");
        // flush the sink by dropping the capture's last reference via scope exit;
        // BinSink flushes on append, but force a sync by reopening below.
    }

    // replay reads the recording back through the public proxima-replay Pipe
    let source: DynRecordingSource = Arc::new(BinSource::new(&data_path, prime()));
    let replay = ReplayUpstream::from_source(source, "round-trip-test")
        .await
        .expect("load replay");

    let keys = replay.known_keys();
    assert_eq!(
        keys.len(),
        1,
        "exactly one interaction recorded, got {keys:?}"
    );
    assert_eq!(
        keys[0], "POST /responses?",
        "match key must be method + path + empty query"
    );

    let request = Request::builder()
        .method(b"POST".as_slice())
        .path(b"/responses".to_vec())
        .build()
        .expect("build request");

    let response = replay.call(request).await.expect("replay call");
    assert_eq!(response.status, 200, "replayed status must match recorded");

    // headers carry the redacted-at-capture content-type (non-sensitive, kept verbatim)
    let content_type = response
        .metadata
        .iter()
        .find(|(name, _)| name.as_ref() == b"content-type")
        .map(|(_, value)| value.clone());
    assert!(
        content_type.is_some(),
        "content-type header must survive the round trip"
    );

    let mut body_stream = response.into_chunk_stream();
    let mut body_bytes = Vec::new();
    while let Some(item) = body_stream.next().await {
        body_bytes.extend_from_slice(&item.expect("body chunk"));
    }
    assert_eq!(
        body_bytes, response_body,
        "replayed body must byte-match the captured response body"
    );
}

#[proxima::test]
async fn replay_miss_for_unrecorded_request_path() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let data_path = temp_dir.path().join("recording.bin");

    let request_wire = b"POST /responses HTTP/1.1\r\nHost: api.example.com\r\n\r\n";
    let response_wire = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n";
    {
        let capture = Capture::open(&data_path, armed_spigot()).expect("open capture");
        let recorder = capture
            .begin("api.example.com", request_wire, Instant::now())
            .await
            .expect("begin");
        recorder.push_response(Bytes::from_static(b"ok"));
        recorder.finish(response_wire).await.expect("finish");
    }

    let source: DynRecordingSource = Arc::new(BinSource::new(&data_path, prime()));
    let replay = ReplayUpstream::from_source(source, "miss-test")
        .await
        .expect("load replay");

    // ask for a path that was never recorded — replay must signal a miss, not
    // serve the wrong interaction
    let request = Request::builder()
        .method(b"GET".as_slice())
        .path(b"/never-recorded".to_vec())
        .build()
        .expect("build request");

    let result = replay.call(request).await;
    assert!(
        result.is_err(),
        "unrecorded path must produce a replay miss, not a wrong-body hit"
    );
}
