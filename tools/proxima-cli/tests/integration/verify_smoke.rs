#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;

use proxima::recording::event::{
    EventSource, HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent, RequestHeader,
};
use proxima::recording::sink::RecordingSink;
use proxima::recording::{AccumulatingSink, FormatKind, LazyFanOut, SinkSpec};
use tempfile::tempdir;
use time::OffsetDateTime;
use tokio::process::Command;

// a per-event bin recording sink on an armed spigot (batch=1 writes each event
// immediately, matching the old BinSink::create per-append semantics). the
// verify/replay CLI reads the .bin back through BinSource.
fn bin_sink(path: &Path) -> AccumulatingSink {
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

fn proxima_bin() -> &'static str {
    env!("CARGO_BIN_EXE_proxima")
}

#[proxima::test]
async fn verify_at_project_root_discovers_spec_and_passes() {
    let dir = tempdir().expect("tempdir");
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[upstreams.origin]
type    = "http"
url     = "https://api.example.com"
timeout = "5s"
"#,
    )
    .await
    .expect("write spec");

    let output = Command::new(proxima_bin())
        .arg("verify")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn verify");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "expected exit 0, got {output:?}");
    assert!(stdout.contains("PASS no_cycles"));
    assert!(stdout.contains("PASS all_upstreams_have_timeouts"));
}

#[proxima::test]
async fn verify_cycle_fails_with_exit_1() {
    let dir = tempdir().expect("tempdir");
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[pipes.a]
chain = ["b"]
[pipes.b]
chain = ["a"]
"#,
    )
    .await
    .expect("write spec");

    let output = Command::new(proxima_bin())
        .arg("verify")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn verify");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "expected exit 1");
    assert!(stdout.contains("FAIL no_cycles"));
    assert!(stdout.contains("a → b → a"));
}

#[proxima::test]
async fn verify_no_spec_exits_2() {
    let dir = tempdir().expect("tempdir");
    let output = Command::new(proxima_bin())
        .arg("verify")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn verify");

    assert_eq!(output.status.code(), Some(2), "expected exit 2");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no spec found"));
}

#[proxima::test]
async fn verify_json_format_emits_structured_doc() {
    let dir = tempdir().expect("tempdir");
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        "[upstreams.origin]\ntype = \"http\"\nurl = \"https://api\"\ntimeout = \"5s\"\n",
    )
    .await
    .expect("write spec");

    let output = Command::new(proxima_bin())
        .arg("verify")
        .arg("--format")
        .arg("json")
        .current_dir(dir.path())
        .output()
        .await
        .expect("spawn verify --format json");

    assert!(output.status.success());
    let doc: serde_json::Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert!(doc.get("entries").and_then(|v| v.as_array()).is_some());
    assert_eq!(doc.get("fail").and_then(serde_json::Value::as_u64), Some(0));
}

#[proxima::test]
async fn replay_at_project_root_discovers_recording_and_passes() {
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings)
        .await
        .expect("mkdir recordings");
    let bin_path = recordings.join("session.bin");
    write_minimal_recording(&bin_path).await;

    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "expected exit 0, got {output:?}");
    assert!(stdout.contains("PASS unauthorized_upstream_call"));
    assert!(stdout.contains("PASS replay.terminated_cleanly"));
    assert!(stdout.contains("PASS replay.recording_summary"));
}

#[proxima::test]
async fn replay_dangling_start_warns() {
    let dir = tempdir().expect("tempdir");
    let bin_path = dir.path().join("recordings").join("dangling.bin");
    tokio::fs::create_dir_all(bin_path.parent().expect("parent"))
        .await
        .expect("mkdir");
    write_dangling_recording(&bin_path).await;

    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "warn-only ⇒ exit 0");
    assert!(stdout.contains("WARN replay.terminated_cleanly"));

    let strict = Command::new(proxima_bin())
        .arg("replay")
        .arg("--strict")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .output()
        .await
        .expect("spawn replay --strict");
    assert_eq!(strict.status.code(), Some(1), "strict ⇒ exit 1");
}

#[proxima::test]
async fn replay_byte_drift_passes_when_live_pipe_matches_recording() {
    use proxima::recording::event::FrameMetadata;
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings).await.expect("mkdir");
    let bin_path = recordings.join("session.bin");

    // Recording: a single Started + ResponseChunk('hello') + Ended for
    // pipe `echo`. The bytes the live pipe is expected to produce are
    // the recorded chunk.
    let sink = bin_sink(&bin_path);
    let start_id = InteractionId::from_bytes([10; 16]);
    let chunk_id = InteractionId::from_bytes([11; 16]);
    let end_id = InteractionId::from_bytes([12; 16]);

    let mut req_header = proxima::recording::event::RequestHeader::default();
    req_header.method = "GET".into();
    req_header.path = "/".into();
    let started = RecordingEvent {
        id: start_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: "echo".into(),
            request: req_header,
            meta: None,
        }),
    };
    let chunk = RecordingEvent {
        id: chunk_id,
        ts_ms: 1,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
            data: bytes::Bytes::from_static(b"hello"),
            metadata: FrameMetadata::default(),
        }),
    };
    let ended = RecordingEvent {
        id: end_id,
        ts_ms: 2,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 1,
            meta: RecordMeta::default(),
        }),
    };
    sink.append(started).await.expect("append");
    sink.append(chunk).await.expect("append");
    sink.append(ended).await.expect("append");

    // Spec: a single `echo` pipe whose synth upstream returns "hello".
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[pipes.echo]
[pipes.echo.synth]
status = 200
body = "hello"
"#,
    )
    .await
    .expect("write spec");

    // Policy: flag `echo` for byte-identical replay.
    tokio::fs::write(
        dir.path().join("proxima.policy.toml"),
        r#"
[replay]
byte_identical_pipes = ["echo"]
"#,
    )
    .await
    .expect("write policy");

    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "expected exit 0, got stdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("PASS byte_drift"), "got: {stdout}");
}

#[proxima::test]
async fn replay_byte_drift_resolves_array_of_tables_pipe_form() {
    use proxima::recording::event::FrameMetadata;
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings).await.expect("mkdir");
    let bin_path = recordings.join("session.bin");

    let sink = bin_sink(&bin_path);
    let start_id = InteractionId::from_bytes([20; 16]);
    let chunk_id = InteractionId::from_bytes([21; 16]);
    let end_id = InteractionId::from_bytes([22; 16]);
    let mut req_header = proxima::recording::event::RequestHeader::default();
    req_header.method = "GET".into();
    req_header.path = "/".into();
    sink.append(RecordingEvent {
        id: start_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: "redact".into(),
            request: req_header,
            meta: None,
        }),
    })
    .await
    .expect("append start");
    sink.append(RecordingEvent {
        id: chunk_id,
        ts_ms: 1,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
            data: bytes::Bytes::from_static(b"redacted"),
            metadata: FrameMetadata::default(),
        }),
    })
    .await
    .expect("append chunk");
    sink.append(RecordingEvent {
        id: end_id,
        ts_ms: 2,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 1,
            meta: RecordMeta::default(),
        }),
    })
    .await
    .expect("append end");

    // The REAL proxima.toml shape: [[pipe]] array, not [pipes.x] map.
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[[pipe]]
name   = "redact"
[pipe.synth]
status = 200
body   = "redacted"
"#,
    )
    .await
    .expect("write spec");

    tokio::fs::write(
        dir.path().join("proxima.policy.toml"),
        r#"
[replay]
byte_identical_pipes = ["redact"]
"#,
    )
    .await
    .expect("write policy");

    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "expected exit 0, got stdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("PASS byte_drift"),
        "byte_drift must resolve [[pipe]] form; got: {stdout}"
    );
}

#[proxima::test]
async fn replay_byte_drift_warns_when_no_spec() {
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings).await.expect("mkdir");
    let bin_path = recordings.join("session.bin");
    write_minimal_recording(&bin_path).await;

    tokio::fs::write(
        dir.path().join("proxima.policy.toml"),
        r#"
[replay]
byte_identical_pipes = ["api"]
"#,
    )
    .await
    .expect("write policy");

    // Deliberately no spec file at the project root — discovery
    // returns None and the walker should emit WARN byte_drift.
    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "warn-only ⇒ exit 0, got: {stdout}");
    assert!(stdout.contains("WARN byte_drift"), "got: {stdout}");
    assert!(
        stdout.contains("byte_identical_pipes set but no --spec"),
        "got: {stdout}"
    );
}

#[proxima::test]
async fn replay_no_recording_exits_2() {
    let dir = tempdir().expect("tempdir");
    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no recording found"));
}

async fn write_minimal_recording(path: &std::path::Path) {
    let sink = bin_sink(path);
    let start_id = InteractionId::from_bytes([1; 16]);
    let end_id = InteractionId::from_bytes([2; 16]);

    let started = RecordingEvent {
        id: start_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: "api".into(),
            request: RequestHeader::default(),
            meta: None,
        }),
    };
    let ended = RecordingEvent {
        id: end_id,
        ts_ms: 1,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 1,
            meta: RecordMeta::default(),
        }),
    };

    sink.append(started).await.expect("append started");
    sink.append(ended).await.expect("append ended");
}

async fn write_recording_with_inferred_event(path: &std::path::Path, pipe: &str) {
    let sink = bin_sink(path);
    let start_id = InteractionId::from_bytes([7; 16]);
    let end_id = InteractionId::from_bytes([8; 16]);

    let started = RecordingEvent {
        id: start_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: pipe.into(),
            request: RequestHeader::default(),
            meta: None,
        }),
    };
    let mut inferred_meta = RecordMeta::default();
    inferred_meta.source = Some(EventSource::Inferred);
    let ended = RecordingEvent {
        id: end_id,
        ts_ms: 1,
        parent: Some(start_id),
        event: ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 1,
            meta: inferred_meta,
        }),
    };
    sink.append(started).await.expect("append started");
    sink.append(ended).await.expect("append ended");
}

#[proxima::test]
async fn replay_repair_reverts_must_derive_with_inferred_event() {
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings).await.expect("mkdir");
    let bin_path = recordings.join("session.bin");
    write_recording_with_inferred_event(&bin_path, "fetch").await;

    tokio::fs::write(
        dir.path().join("proxima.policy.toml"),
        r#"
[replay]
must_derive_from_record = ["fetch"]
"#,
    )
    .await
    .expect("write policy");

    let output = Command::new(proxima_bin())
        .arg("replay")
        .arg("--repair")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn replay --repair");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "post-repair replay should pass; got stdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("REPAIR dropped must_derive fetch"),
        "expected blame line, got: {stdout}"
    );
    assert!(stdout.contains("PASS inferred_not_recorded"));
    assert!(!stdout.contains("FAIL inferred_not_recorded"));
}

#[proxima::test]
async fn replay_without_repair_still_fails_on_inferred_event() {
    let dir = tempdir().expect("tempdir");
    let recordings = dir.path().join(".proxima").join("recordings");
    tokio::fs::create_dir_all(&recordings).await.expect("mkdir");
    let bin_path = recordings.join("session.bin");
    write_recording_with_inferred_event(&bin_path, "fetch").await;

    tokio::fs::write(
        dir.path().join("proxima.policy.toml"),
        r#"
[replay]
must_derive_from_record = ["fetch"]
"#,
    )
    .await
    .expect("write policy");

    let output = Command::new(proxima_bin())
        .arg("replay")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn replay");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected fail; got: {stdout}"
    );
    assert!(stdout.contains("FAIL inferred_not_recorded"));
}

#[proxima::test]
async fn verify_repair_turns_cycle_into_clean_exit() {
    let dir = tempdir().expect("tempdir");
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[pipes.a]
chain = ["b"]
[pipes.b]
chain = ["a"]
"#,
    )
    .await
    .expect("write spec");

    let output = Command::new(proxima_bin())
        .arg("verify")
        .arg("--repair")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn verify --repair");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "post-repair spec should pass; got {output:?}"
    );
    assert!(
        stdout.contains("REPAIR dropped chain_edge"),
        "expected blame line, got: {stdout}"
    );
    assert!(stdout.contains("PASS no_cycles"));
    assert!(!stdout.contains("FAIL no_cycles"));
}

#[proxima::test]
async fn verify_repair_json_emits_blame_field() {
    let dir = tempdir().expect("tempdir");
    tokio::fs::write(
        dir.path().join("proxima.toml"),
        r#"
[pipes.a]
chain = ["b"]
[pipes.b]
chain = ["a"]
"#,
    )
    .await
    .expect("write spec");

    let output = Command::new(proxima_bin())
        .arg("verify")
        .arg("--repair")
        .arg("--format")
        .arg("json")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn verify --repair --format json");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let document: serde_json::Value = serde_json::from_str(&stdout).expect("parse json output");
    let blame = document
        .get("blame")
        .and_then(serde_json::Value::as_array)
        .expect("blame array present");
    assert_eq!(blame.len(), 1, "exactly one edge dropped");
    let entry = blame[0].as_str().expect("blame entry is string");
    assert!(entry.starts_with("chain_edge "));
    assert_eq!(
        document.get("fail").and_then(serde_json::Value::as_u64),
        Some(0)
    );
}

async fn write_dangling_recording(path: &std::path::Path) {
    let sink = bin_sink(path);
    let start_id = InteractionId::from_bytes([3; 16]);
    let started = RecordingEvent {
        id: start_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: "lonely".into(),
            request: RequestHeader::default(),
            meta: None,
        }),
    };
    sink.append(started).await.expect("append started");
}
