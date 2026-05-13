//! Replay serves a cassette back byte-identical: no upstream call happens —
//! every status code, header, and response chunk comes straight off disk,
//! in the exact framing it was captured with. `record` (see
//! `examples/record.rs`) is what produces a cassette from live traffic;
//! this example builds one the same way — the same `RecordingEvent` /
//! `HttpEvent` model `record` emits — then proves `ReplayUpstream` plays
//! it back exactly, chunk boundaries included, and misses cleanly on
//! anything that was never captured.
//!
//! Run: `cargo run --example replay`

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use futures::StreamExt;
use time::OffsetDateTime;

use proxima::runtime::PrimeRuntime;
use proxima::{
    HttpEvent, InteractionId, ProtocolEvent, ProximaError, RecordMeta, RecordingEvent,
    RecordingRequestHeader, ReplayUpstream, Request, Runtime, SendPipe,
};
use proxima_recording::{Format, JsonFormat};

#[proxima::main(cores = 1)]
async fn main() -> Result<(), ProximaError> {
    let cassette_dir = tempfile::tempdir()?;
    let cassette_path = cassette_dir.path().join("chat.jsonl");

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
    let events = sample_interaction(interaction, &response_headers, &response_chunks);
    let cassette_bytes = JsonFormat::new()
        .encode_block(events)
        .map_err(|err| ProximaError::Record(format!("encode cassette: {err}")))?;
    std::fs::write(&cassette_path, &cassette_bytes)?;
    println!(
        "recorded 1 interaction ({} bytes) to {}",
        cassette_bytes.len(),
        cassette_path.display()
    );

    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1)?);
    let replay = ReplayUpstream::from_jsonl(&cassette_path, "chat-replay", runtime).await?;
    println!(
        "loaded cassette, known match keys: {:?}",
        replay.known_keys()
    );

    let request = Request::builder()
        .method("POST")
        .path("/v1/chat/completions")
        .query_param("model", "gpt-mini")
        .build()?;
    let response = SendPipe::call(&replay, request).await?;
    let replayed_status = response.status;

    if replayed_status != 200 {
        return Err(ProximaError::Record(format!(
            "replayed status {replayed_status} must match recorded status 200"
        )));
    }

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
    if replayed_headers != response_headers {
        return Err(ProximaError::Record(
            "replayed headers diverged from what was recorded".into(),
        ));
    }

    let mut replayed_chunks = Vec::new();
    let mut chunk_stream = response.into_chunk_stream();
    while let Some(chunk) = chunk_stream.next().await {
        replayed_chunks.push(chunk?);
    }
    if replayed_chunks != response_chunks {
        return Err(ProximaError::Record(
            "replayed chunk framing diverged from what was recorded".into(),
        ));
    }

    println!(
        "replayed status {replayed_status} and {} headers, byte-identical to what was recorded",
        replayed_headers.len()
    );
    for (index, chunk) in replayed_chunks.iter().enumerate() {
        println!("  chunk {index}: {:?}", String::from_utf8_lossy(chunk));
    }

    // replay never guesses: a request that was never captured is a typed
    // miss, not a wrong-body 200 — the flip side of "byte-identical".
    let unrecorded = Request::builder()
        .method("GET")
        .path("/v1/never-recorded")
        .build()?;
    match SendPipe::call(&replay, unrecorded).await {
        Err(ProximaError::ReplayMiss { fingerprint }) => {
            println!("unrecorded request correctly missed: {fingerprint}");
        }
        Err(other) => return Err(other),
        Ok(_) => {
            return Err(ProximaError::Record(
                "expected a replay miss for an unrecorded path".into(),
            ));
        }
    }

    Ok(())
}

/// One recorded HTTP interaction, in the exact `RecordingEvent` shape
/// `record`'s sink writes: `Started` → request chunk(s) → `RequestEnded` →
/// `ResponseStarted` → response chunk(s) → `Ended`.
fn sample_interaction(
    id: InteractionId,
    response_headers: &[(String, String)],
    response_chunks: &[Bytes],
) -> Vec<RecordingEvent> {
    let envelope = |ts_ms: u64, event: HttpEvent| RecordingEvent {
        id,
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
                headers: response_headers.to_vec(),
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

    events
}
