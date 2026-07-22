//! Integration test: record a synthetic WebSocket session, replay it through
//! `WsReplayUpstream` as a genuine upgrade, and prove:
//!   1. the 101 response recomputes Sec-WebSocket-Accept from the replaying
//!      client's key (not the recorded one)
//!   2. the upgrade handler streams the recorded server frames back verbatim
//!
//! Spans the recording format base (BinFormat/BinSource) + the `replay`
//! feature (WsReplayUpstream) + proxima-pipe (upgrade machinery), no network.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "replay")]

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Request;
use proxima_primitives::pipe::upgrade::HijackedSocket;
use proxima_recording::event::{
    HttpEvent, InteractionId, ProtocolEvent, RecordingEvent, RequestHeader,
};
use proxima_recording::replay::ws::{WsReplayUpstream, compute_accept_key};
use proxima_recording::source::DynRecordingSource;
use proxima_recording::{BinFormat, BinSource, Format};
use proxima_runtime::Runtime;

fn prime() -> Arc<dyn Runtime> {
    Arc::new(prime::os::runtime::PrimeRuntime::new(1).expect("prime"))
}

/// An in-memory futures-io stream that captures everything written to it and
/// reads as immediate EOF. Send + Unpin, so it satisfies `HijackStream`.
#[derive(Clone)]
struct CaptureSocket {
    written: Arc<Mutex<Vec<u8>>>,
}

impl futures::io::AsyncRead for CaptureSocket {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Poll::Ready(Ok(0))
    }
}

impl futures::io::AsyncWrite for CaptureSocket {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.written.lock().unwrap().extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }
    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

fn http_event(id: InteractionId, event: HttpEvent) -> RecordingEvent {
    RecordingEvent {
        id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(event),
    }
}

async fn record_ws_session(path: &std::path::Path, frames: &[&[u8]]) {
    let id = InteractionId::new();
    let request = RequestHeader {
        method: "GET".into(),
        path: "/chat".into(),
        headers: std::collections::BTreeMap::new(),
        query: std::collections::BTreeMap::new(),
    };
    let mut events = vec![
        http_event(
            id,
            HttpEvent::Started {
                ts: time::OffsetDateTime::UNIX_EPOCH,
                pipe: "intercept".into(),
                request,
                meta: None,
            },
        ),
        http_event(id, HttpEvent::RequestEnded),
        http_event(
            id,
            HttpEvent::ResponseStarted {
                status: 101,
                headers: vec![
                    ("upgrade".into(), "websocket".into()),
                    ("connection".into(), "Upgrade".into()),
                ],
            },
        ),
    ];
    for frame in frames {
        events.push(http_event(
            id,
            HttpEvent::ResponseChunk {
                data: Bytes::copy_from_slice(frame),
                metadata: std::collections::BTreeMap::new(),
            },
        ));
    }
    events.push(http_event(
        id,
        HttpEvent::Ended {
            latency_ms: 1,
            meta: proxima_recording::event::RecordMeta::default(),
        },
    ));

    let bytes = BinFormat::new()
        .expect("bin format")
        .encode_block(events)
        .expect("encode");
    tokio::fs::write(path, bytes).await.expect("write");
}

#[proxima::test(runtime = "tokio")]
async fn ws_session_replays_as_upgrade_with_recomputed_accept_and_frames() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let path = temp_dir.path().join("ws.bin");

    // two server text frames: 0x81 (fin+text), len, payload
    let frame_a: &[u8] = b"\x81\x05hello";
    let frame_b: &[u8] = b"\x81\x05world";
    record_ws_session(&path, &[frame_a, frame_b]).await;

    let source: DynRecordingSource = Arc::new(BinSource::new(&path, prime()));
    let replay = WsReplayUpstream::from_source(source, "ws-replay-test")
        .await
        .expect("load ws replay");

    let keys = replay.known_keys();
    assert_eq!(
        keys,
        vec!["GET /chat?".to_string()],
        "one ws session indexed"
    );
    assert_eq!(replay.frame_count("GET /chat?"), Some(2));

    // a replaying client presents ITS OWN Sec-WebSocket-Key
    let client_key = "dGhlIHNhbXBsZSBub25jZQ==";
    let request = Request::builder()
        .method("GET")
        .path(b"/chat".to_vec())
        .header("sec-websocket-key", client_key.as_bytes().to_vec())
        .header("upgrade", b"websocket".to_vec())
        .build()
        .expect("build request");

    let response = replay.call(request).await.expect("replay call");
    assert_eq!(
        response.status, 101,
        "must be a switching-protocols upgrade"
    );

    // accept must be recomputed from THIS client's key, not the recorded one
    let accept = response
        .metadata
        .get_str("sec-websocket-accept")
        .expect("accept header present");
    assert_eq!(accept, compute_accept_key(client_key));
    assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", "RFC 6455 vector");

    // drive the upgrade handler against an in-memory socket; the recorded
    // server frames must come out verbatim, in order
    let upgrade = response.upgrade.expect("upgrade handler present");
    let written = Arc::new(Mutex::new(Vec::new()));
    let socket = CaptureSocket {
        written: Arc::clone(&written),
    };
    let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());
    upgrade.invoke(hijacked).await.expect("handler ran");

    let out = written.lock().unwrap().clone();
    let mut expected = Vec::new();
    expected.extend_from_slice(frame_a);
    expected.extend_from_slice(frame_b);
    assert_eq!(out, expected, "recorded server frames must replay verbatim");
}

#[proxima::test(runtime = "tokio")]
async fn non_ws_request_to_ws_replay_is_a_miss() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let path = temp_dir.path().join("ws.bin");
    record_ws_session(&path, &[b"\x81\x02hi"]).await;

    let source: DynRecordingSource = Arc::new(BinSource::new(&path, prime()));
    let replay = WsReplayUpstream::from_source(source, "miss-test")
        .await
        .expect("load");

    let request = Request::builder()
        .method("GET")
        .path(b"/not-recorded".to_vec())
        .header("sec-websocket-key", b"dGhlIHNhbXBsZSBub25jZQ==".to_vec())
        .build()
        .expect("build");
    assert!(
        replay.call(request).await.is_err(),
        "unrecorded path must miss"
    );
}

#[proxima::test(runtime = "tokio")]
async fn ws_replay_without_client_key_errors() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let path = temp_dir.path().join("ws.bin");
    record_ws_session(&path, &[b"\x81\x02hi"]).await;

    let source: DynRecordingSource = Arc::new(BinSource::new(&path, prime()));
    let replay = WsReplayUpstream::from_source(source, "no-key-test")
        .await
        .expect("load");

    // matching path but no Sec-WebSocket-Key — cannot complete the handshake
    let request = Request::builder()
        .method("GET")
        .path(b"/chat".to_vec())
        .build()
        .expect("build");
    assert!(
        replay.call(request).await.is_err(),
        "missing client key must error"
    );
}

#[proxima::test(runtime = "tokio")]
async fn http_post_recording_is_not_indexed_as_ws() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let path = temp_dir.path().join("http.bin");

    // a plain 200 POST interaction — WsReplayUpstream must ignore it
    let id = InteractionId::new();
    let events = vec![
        http_event(
            id,
            HttpEvent::Started {
                ts: time::OffsetDateTime::UNIX_EPOCH,
                pipe: "intercept".into(),
                request: RequestHeader {
                    method: "POST".into(),
                    path: "/responses".into(),
                    headers: std::collections::BTreeMap::new(),
                    query: std::collections::BTreeMap::new(),
                },
                meta: None,
            },
        ),
        http_event(
            id,
            HttpEvent::ResponseStarted {
                status: 200,
                headers: vec![],
            },
        ),
        http_event(
            id,
            HttpEvent::Ended {
                latency_ms: 1,
                meta: proxima_recording::event::RecordMeta::default(),
            },
        ),
    ];
    let bytes = BinFormat::new()
        .expect("bin format")
        .encode_block(events)
        .expect("encode");
    tokio::fs::write(&path, bytes).await.expect("write");

    let source: DynRecordingSource = Arc::new(BinSource::new(&path, prime()));
    let replay = WsReplayUpstream::from_source(source, "http-only")
        .await
        .expect("load");
    assert!(
        replay.known_keys().is_empty(),
        "200 interactions must not index as ws sessions"
    );
}
