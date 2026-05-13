//! Integration test (Work-Queue Row 2 / §16 substrate): re-prove the observe
//! path OFFLINE from vendored capture fixtures, every commit.
//!
//! The discipline log (C18) named `proxima_intercept_observe_paths_unregressed`
//! as the rollback gate; that unit test proves the config-level gate (swap-off →
//! observe byte-identical). The remaining §16 substrate it called out was
//! "replaying the 3 observe captures through the post-fork pipe … gated on
//! vendored captures for codex/claude/copilot, not yet in spec/examples/." This
//! test closes that: the captures are now checked in (`spec/examples/*-observe.jsonl`)
//! and CI replays them.
//!
//! The observe path, driven from the vendored files: `ReplayUpstream::from_jsonl`
//! serves each recorded turn; the replayed response must byte-match the recorded
//! one. The swap path moved to a downstream consumer with the vendor integration surfaces.
//!
//! The fixtures are REAL captured turns: each CLI (claude / copilot / codex) was
//! run through the intercept proxy (key-free — every tool authenticates its own
//! session) and the recorded request/response extracted. Account identifiers
//! (device/account/session ids, billing tokens) and the vendors' proprietary
//! system prompts are redacted — the same discipline the capture applies to
//! headers — while the real wire structure, the real user prompt, and the real
//! model output are preserved. The codex fixture stores the INFLATED
//! post-permessage-deflate payload, recovered via `WsInflater`.
//!
//! Gated on `intercept-replay`; a no-op under default features.
#![cfg(feature = "intercept-replay")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use futures::StreamExt;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Request;
use proxima_recording::JsonlSource;
use proxima_recording::event::{HttpEvent, ProtocolEvent};
use proxima_recording::replay::ReplayUpstream;
use proxima_recording::source::RecordingSource;

const COPILOT: &str = "copilot-responses-observe.jsonl";
const CLAUDE: &str = "claude-messages-observe.jsonl";
const CODEX: &str = "codex-response-create-observe.jsonl";

/// Resolve a vendored fixture against the workspace-root `spec/examples/`
/// (sibling of the canonical `recording-http.jsonl`). Test CWD is the package
/// dir, so anchor on `CARGO_MANIFEST_DIR` and step up one level.
fn fixture(name: &str) -> String {
    format!("{}/../spec/examples/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Concatenate the recorded request-body and response-body chunks from a vendored
/// capture — the inbound bytes the swap decodes and the outbound bytes the
/// observe path must reproduce.
fn prime() -> std::sync::Arc<dyn proxima::runtime::Runtime> {
    std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
}

async fn recorded_bodies(name: &str) -> (Vec<u8>, Vec<u8>) {
    let source = JsonlSource::new(fixture(name), prime());
    let mut events = source.events();
    let (mut request, mut response) = (Vec::new(), Vec::new());
    while let Some(item) = events.next().await {
        match item.expect("vendored event").event {
            ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. }) => {
                request.extend_from_slice(&data)
            }
            ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => {
                response.extend_from_slice(&data)
            }
            _ => {}
        }
    }
    (request, response)
}

/// Replay the recorded turn through the public `proxima-replay` Pipe and collect
/// the served response body — the observe path, offline.
async fn replayed_body(name: &str, method: &[u8], request_path: &[u8]) -> Vec<u8> {
    let replay = ReplayUpstream::from_jsonl(fixture(name), "observe-vendored", prime())
        .await
        .expect("load vendored capture");
    let request = Request::builder()
        .method(method)
        .path(request_path.to_vec())
        .build()
        .expect("build request");
    let response = replay.call(request).await.expect("replay call");
    let mut stream = response.into_chunk_stream();
    let mut body = Vec::new();
    while let Some(item) = stream.next().await {
        body.extend_from_slice(&item.expect("body chunk"));
    }
    body
}

#[proxima::test]
async fn copilot_observe_capture_replays_recorded_response() {
    let (_, recorded) = recorded_bodies(COPILOT).await;
    let served = replayed_body(COPILOT, b"POST", b"/responses").await;
    assert_eq!(
        served, recorded,
        "replayed body must byte-match the recorded copilot response"
    );
    let text = String::from_utf8(served).unwrap();
    assert!(
        text.contains("response.created"),
        "copilot observe body is a real Responses SSE stream"
    );
}

#[proxima::test]
async fn claude_observe_capture_replays_recorded_response() {
    let (_, recorded) = recorded_bodies(CLAUDE).await;
    let served = replayed_body(CLAUDE, b"POST", b"/v1/messages").await;
    assert_eq!(
        served, recorded,
        "replayed body must byte-match the recorded claude response"
    );
    let text = String::from_utf8(served).unwrap();
    // real Anthropic streaming SSE: message_start → text_delta(s) → message_delta
    assert!(
        text.contains("content_block_delta"),
        "claude observe body is a real Anthropic SSE stream"
    );
    assert!(
        text.contains("pong"),
        "the real model answer carries the echoed token"
    );
}

#[proxima::test]
async fn codex_observe_capture_replays_recorded_response() {
    let (_, recorded) = recorded_bodies(CODEX).await;
    let served = replayed_body(CODEX, b"POST", b"/backend-api/codex/responses").await;
    assert_eq!(
        served, recorded,
        "replayed body must byte-match the recorded codex response"
    );
    let text = String::from_utf8(served).unwrap();
    assert!(
        text.contains("response.created"),
        "codex observe body is a real Responses event stream"
    );
}
