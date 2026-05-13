//! Re-provable disk-ratio measurement (Work-Queue Row 5) over the REAL vendored
//! capture payloads — not the niche repetitive-SSE best case.
//!
//! Finding (discipline C23): the per-interaction block compression is net-positive
//! on real traffic, and the lever that moves disk is the zstd LEVEL (already
//! runtime-configurable via `create_with_level`). The 256-byte block threshold
//! stays a const, not a build-time conflaguration axis — a measured call: raising
//! it to the 4 KiB LSM page floor was tried and reverted because the storage is a
//! packed append-log (no per-object page) and 4 KiB drops a typical streaming turn
//! under the bar, turning a ~14-23x compression into a raw-bytes EXPANSION. CPU is
//! a non-factor: streaming blocks compress at 1.7-7.7 GB/s (sub-µs/turn) and the
//! whole path is default-off + off the gateway's revenue path.
//!
//! This test re-proves two stable claims every commit: (a) compression is
//! net-positive across the varied real corpus (on-disk < raw chunk bytes), and
//! (b) a higher zstd level never costs disk (L19 <= L3). CPU is measured
//! out-of-band via `zstd -b`; a CPU assertion here would be host-fragile.
//!
//! Gated on `intercept-replay` (pulls recording-core + JsonlSource).
#![cfg(feature = "intercept-replay")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use futures::StreamExt;
use proxima::SendPipe;
use proxima_recording::JsonlSource;
use proxima_recording::binary::BinFormat;
use proxima_recording::event::{HttpEvent, ProtocolEvent, RecordingEvent};
use proxima_recording::pipe::AppendLog;
use proxima_recording::source::RecordingSource;

const FIXTURES: &[&str] = &[
    "copilot-responses-observe.jsonl",
    "claude-messages-observe.jsonl",
    "codex-response-create-observe.jsonl",
];

fn fixture(name: &str) -> String {
    format!("{}/../spec/examples/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn prime() -> std::sync::Arc<dyn proxima::runtime::Runtime> {
    std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
}

async fn events_of(name: &str) -> Vec<RecordingEvent> {
    let source = JsonlSource::new(fixture(name), prime());
    let mut stream = source.events();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.expect("vendored event"));
    }
    events
}

/// Raw payload bytes (the request + response chunk bodies) the recording carries
/// — the denominator for the disk ratio.
fn raw_chunk_bytes(events: &[RecordingEvent]) -> usize {
    events
        .iter()
        .map(|event| match &event.event {
            ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. })
            | ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => data.len(),
            _ => 0,
        })
        .sum()
}

/// Record `events` through a fresh AppendLog (bin format) at `level`, return the
/// on-disk size. (C7 replaced BinSink with the AppendLog pipe over a Format.)
async fn on_disk_bytes(events: &[RecordingEvent], level: i32) -> usize {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rec.bin");
    let format = Box::new(BinFormat::with_level(level).expect("bin format"));
    let sink = AppendLog::open(&path, format, prime()).expect("sink");
    sink.call(events.to_vec()).await.expect("append");
    sink.sync().await.expect("sync");
    std::fs::metadata(&path).expect("stat").len() as usize
}

#[proxima::test]
async fn block_compression_is_net_positive_on_real_vendored_payloads() {
    let mut total_raw = 0_usize;
    let mut total_disk = 0_usize;
    for name in FIXTURES {
        let events = events_of(name).await;
        let raw = raw_chunk_bytes(&events);
        let disk = on_disk_bytes(&events, 3).await;
        eprintln!(
            "{name}: raw chunks {raw}B -> on-disk {disk}B ({:.2}x)",
            raw as f64 / disk.max(1) as f64
        );
        total_raw += raw;
        total_disk += disk;
    }
    eprintln!(
        "corpus: raw {total_raw}B -> on-disk {total_disk}B ({:.2}x)",
        total_raw as f64 / total_disk.max(1) as f64
    );
    assert!(
        total_disk < total_raw,
        "block compression must be net-positive across the real corpus: {total_disk}B on-disk vs {total_raw}B raw"
    );
}

#[proxima::test]
async fn higher_zstd_level_never_costs_disk() {
    // the real tunable: level moves disk; the 256 threshold does not.
    let events = events_of("codex-response-create-observe.jsonl").await;
    let l3 = on_disk_bytes(&events, 3).await;
    let l19 = on_disk_bytes(&events, 19).await;
    eprintln!("codex payload: L3 {l3}B, L19 {l19}B");
    assert!(
        l19 <= l3,
        "a higher zstd level must not increase disk: L19 {l19}B > L3 {l3}B"
    );
}
