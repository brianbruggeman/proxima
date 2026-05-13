#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Integration tests for S1: durable WAL semantics on the spigot write
//! terminal + `BinSource` read side.
//!
//! Post-C7 the write terminal is `AccumulatingSink` over `LazyFanOut`
//! (recording-pipe), not the deleted `BinSink`; durability control is
//! `AccumulatingSink::sync` (fsync via the runtime offload). Per-event blocks
//! (batch=1) give one frame per append, so `BinSource::events_from_offset`
//! sees a distinct byte offset per event — the S1 offset-cursor contract.
//!
//! Covers:
//!   - `sync` succeeds on the happy path; the data file is visible after sync.
//!   - `events_from_offset(0)` yields the same sequence as `events()`, paired
//!     with strictly-increasing frame-start byte offsets.
//!   - `events_from_offset(N)` resumes from byte offset N.
//!   - Crash-point recovery: a data file truncated mid-frame (crash between
//!     write and sync) is detected as EOF on the partial frame; no corrupted
//!     event is yielded.

#![cfg(feature = "durable-wal")]

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use proxima::recording::sink::RecordingSink;
use proxima::{AccumulatingSink, FormatKind, LazyFanOut, SinkSpec};
use proxima_recording::binary::source::BinSource;
use proxima_recording::event::{InteractionId, ProtocolEvent, RecordingEvent};
use proxima_recording::source::RecordingSource;
use tempfile::TempDir;
use ulid::Ulid;

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn make_event(kind: &str) -> RecordingEvent {
    RecordingEvent {
        id: InteractionId::from_ulid(Ulid::new()),
        ts_ms: now_ms(),
        parent: None,
        event: ProtocolEvent::Custom {
            kind: kind.to_string(),
            payload: serde_json::json!({ "marker": kind }),
        },
    }
}

fn prime() -> Arc<dyn proxima::runtime::Runtime> {
    Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
}

// the durable write terminal: per-event bin blocks (batch=1 → one frame per
// append, the per-event-offset shape the S1 cursor needs) on an armed spigot.
fn durable_sink(path: &std::path::Path) -> AccumulatingSink {
    let spigot = proxima::deferred_runtime();
    let _ = spigot.set(
        Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
            as Arc<dyn proxima::runtime::Runtime>,
    );
    let durable = Arc::new(LazyFanOut::new(
        vec![SinkSpec::new(
            path.to_string_lossy().into_owned(),
            FormatKind::Bin,
        )],
        spigot,
    ));
    AccumulatingSink::new(durable, 1)
}

#[proxima::test]
async fn sync_returns_success_on_happy_path() {
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("durable_happy.bin");

    let sink = durable_sink(&bin_path);
    sink.append(make_event("first")).await.expect("append 1");
    sink.append(make_event("second")).await.expect("append 2");
    sink.sync().await.expect("sync should succeed");

    let data_meta = tokio::fs::metadata(&bin_path).await.expect("data exists");
    assert!(
        data_meta.len() > 0,
        "data file should have bytes after sync"
    );
}

#[proxima::test]
async fn events_from_offset_zero_yields_same_count_as_events_method() {
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("offset_zero.bin");

    let sink = durable_sink(&bin_path);
    for index in 0..5 {
        sink.append(make_event(&format!("event-{index}")))
            .await
            .expect("append");
    }
    sink.sync().await.expect("sync");
    drop(sink);

    let source = BinSource::new(&bin_path, prime());

    let baseline: Vec<RecordingEvent> = source
        .events()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|result| result.expect("event"))
        .collect();
    assert_eq!(baseline.len(), 5);

    let with_offsets: Vec<(u64, RecordingEvent)> = source
        .events_from_offset(0)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|result| result.expect("event"))
        .collect();
    assert_eq!(
        with_offsets.len(),
        5,
        "offset-paired stream should yield 5 events"
    );
    assert_eq!(with_offsets[0].0, 0, "first event starts at offset 0");

    for (idx, (_, event)) in with_offsets.iter().enumerate() {
        assert_eq!(event.id(), baseline[idx].id(), "event {idx} id mismatch");
    }
    for window in with_offsets.windows(2) {
        assert!(
            window[1].0 > window[0].0,
            "offsets must be strictly increasing: {} then {}",
            window[0].0,
            window[1].0
        );
    }
}

#[proxima::test]
async fn events_from_offset_n_resumes_from_specified_byte_position() {
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("offset_resume.bin");

    let sink = durable_sink(&bin_path);
    for index in 0..6 {
        sink.append(make_event(&format!("resume-event-{index}")))
            .await
            .expect("append");
    }
    sink.sync().await.expect("sync");
    drop(sink);

    let source = BinSource::new(&bin_path, prime());
    let all: Vec<(u64, RecordingEvent)> = source
        .events_from_offset(0)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|result| result.expect("event"))
        .collect();
    assert_eq!(all.len(), 6);

    let resume_offset = all[3].0;
    let resumed: Vec<(u64, RecordingEvent)> = source
        .events_from_offset(resume_offset)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(|result| result.expect("event"))
        .collect();
    assert_eq!(
        resumed.len(),
        3,
        "resuming from event 3's offset should yield 3 remaining events"
    );
    assert_eq!(
        resumed[0].0, resume_offset,
        "first resumed event at resume offset"
    );
    assert_eq!(
        resumed[0].1.id(),
        all[3].1.id(),
        "resumed 0 id matches all[3]"
    );
    assert_eq!(resumed[1].1.id(), all[4].1.id());
    assert_eq!(resumed[2].1.id(), all[5].1.id());
}

#[proxima::test]
async fn crash_mid_frame_yields_eof_not_corrupted_event() {
    // Crash between write and sync: the data file ends in a partial frame.
    // events_from_offset must NOT yield a corrupted event; it stops at the
    // partial-frame boundary.
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("crash_mid_frame.bin");

    let sink = durable_sink(&bin_path);
    for index in 0..3 {
        sink.append(make_event(&format!("durable-event-{index}")))
            .await
            .expect("append");
    }
    sink.sync().await.expect("sync");
    drop(sink);

    let source_pre = BinSource::new(&bin_path, prime());
    let pre_crash_count = source_pre
        .events_from_offset(0)
        .collect::<Vec<_>>()
        .await
        .len();
    assert_eq!(pre_crash_count, 3, "should have 3 events pre-crash");

    let full_len = tokio::fs::metadata(&bin_path).await.expect("meta").len();
    assert!(
        full_len > 8,
        "file should be big enough to truncate inside last frame"
    );
    let truncate_to = full_len - 2;
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(&bin_path)
        .await
        .expect("open for truncate");
    file.set_len(truncate_to).await.expect("truncate");
    drop(file);

    let source_post = BinSource::new(&bin_path, prime());
    let results: Vec<Result<(u64, RecordingEvent), _>> =
        source_post.events_from_offset(0).collect::<Vec<_>>().await;

    let successes: Vec<_> = results.iter().filter(|result| result.is_ok()).collect();
    assert_eq!(
        successes.len(),
        2,
        "exactly 2 fully-intact events should be readable after mid-frame truncate, got {}",
        successes.len()
    );

    for result in results.iter().flatten() {
        let event = &result.1;
        if let ProtocolEvent::Custom { kind, .. } = &event.event {
            assert!(
                kind.starts_with("durable-event-"),
                "no corrupted Ok event should slip through; got kind={kind}"
            );
        }
    }
}

#[proxima::test]
async fn sync_can_be_called_repeatedly_without_data_loss() {
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("repeat_sync.bin");

    let sink = durable_sink(&bin_path);
    sink.append(make_event("a")).await.expect("a");
    sink.sync().await.expect("sync 1");
    sink.append(make_event("b")).await.expect("b");
    sink.sync().await.expect("sync 2");
    sink.append(make_event("c")).await.expect("c");
    sink.sync().await.expect("sync 3");
    drop(sink);

    let source = BinSource::new(&bin_path, prime());
    let count = source.events_from_offset(0).collect::<Vec<_>>().await.len();
    assert_eq!(count, 3, "all 3 events should survive repeated syncs");
}

#[proxima::test]
async fn sync_completes_within_reasonable_time() {
    let dir = TempDir::new().expect("tempdir");
    let bin_path = dir.path().join("sync_throughput.bin");

    let sink = durable_sink(&bin_path);
    let started = std::time::Instant::now();
    for index in 0..100 {
        sink.append(make_event(&format!("throughput-{index}")))
            .await
            .expect("append");
        sink.sync().await.expect("sync");
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "100 sync calls should complete in <5s, took {elapsed:?}"
    );
}
