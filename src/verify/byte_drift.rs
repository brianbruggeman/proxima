//! `byte_drift` — for each pipe flagged
//! `policy.replay.byte_identical_pipes`, dispatch each recorded
//! request through the *live* pipe built from the supplied spec and
//! assert the response bytes match the recorded response bytes
//! byte-for-byte.
//!
//! The walker needs three things: the recording (events), the policy
//! (allowlist of pipes to check), and the spec (to build live pipe
//! handles). The first two come from the existing replay-walker
//! plumbing; the spec arrives via `proxima replay --spec <path>`.
//!
//! Mismatch ⇒ FAIL `byte_drift` with detail describing the first
//! differing offset and the pipe name. A clean replay ⇒ PASS. When
//! `byte_identical_pipes` is non-empty but no spec is supplied,
//! the walker emits WARN `byte_drift` so a user running zero-arg
//! at the project root sees that the check was elided.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;

use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::load::{LoadContext, load};
use crate::pipe::PipeHandle;
use crate::recording::bin::source::BinSource;
use crate::recording::event::{HttpEvent, InteractionId, ProtocolEvent, RecordingEvent};
use crate::recording::jsonl::source::JsonlSource;
use crate::recording::source::DynRecordingSource;
use crate::request::{Request, Response};
use crate::runtime::Runtime;

use super::policy::Policy;
use super::report::{Report, ReportEntry};

use futures::StreamExt;
use serde_json::Value;

/// Run byte_drift over a recording. `spec` is the parsed spec value
/// (as produced by `LoadContext::config_formats.parse_with_hint`).
/// `context` is the LoadContext that builds live pipes from spec
/// fragments — typically `LoadContext::with_default_registry()`.
pub async fn verify_byte_drift(
    recording_path: &Path,
    policy: &Policy,
    spec: &Value,
    context: &LoadContext,
    runtime: &Arc<dyn Runtime>,
    report: &mut Report,
) -> Result<(), ProximaError> {
    if policy.replay.byte_identical_pipes.is_empty() {
        // Nothing to check, but emit a Pass entry so consumers see
        // the rule ran.
        report.push(ReportEntry::pass("byte_drift"));
        return Ok(());
    }

    let events = collect_events(recording_path, runtime).await?;
    let recorded = group_recorded(&events, policy);
    if recorded.is_empty() {
        report.push(ReportEntry::pass("byte_drift"));
        return Ok(());
    }

    let mut violations: Vec<String> = Vec::new();
    for entry in recorded {
        let RecordedInteraction {
            pipe_name,
            method,
            path,
            request_body,
            recorded_response,
        } = entry;
        let live_pipe = match build_pipe_for(&pipe_name, spec, context).await {
            Ok(handle) => handle,
            Err(err) => {
                violations.push(format!(
                    "{pipe_name}: cannot build live pipe from spec ({err})"
                ));
                continue;
            }
        };
        let request = match build_request(&method, &path, request_body.clone()) {
            Ok(request) => request,
            Err(err) => {
                violations.push(format!(
                    "{pipe_name}: cannot reconstruct recorded request ({err})"
                ));
                continue;
            }
        };
        let response = match SendPipe::call(&live_pipe, request).await {
            Ok(response) => response,
            Err(err) => {
                violations.push(format!("{pipe_name}: live dispatch errored ({err})"));
                continue;
            }
        };
        let live_bytes = match collect_response_bytes(response).await {
            Ok(bytes) => bytes,
            Err(err) => {
                violations.push(format!("{pipe_name}: collect live response bytes ({err})"));
                continue;
            }
        };
        if let Some(offset) = first_diff(&live_bytes, &recorded_response) {
            violations.push(format!(
                "{pipe_name}: bytes diverge at offset {offset} (recorded {} vs live {} bytes)",
                recorded_response.len(),
                live_bytes.len()
            ));
        }
    }

    if violations.is_empty() {
        report.push(ReportEntry::pass("byte_drift"));
    } else {
        report.push(ReportEntry::fail("byte_drift", violations.join("; ")));
    }
    Ok(())
}

/// Variant for callers without a spec: emits WARN explaining the
/// skip. Used when `proxima replay` runs with `--verify` set but no
/// `--spec`, since byte-identical checks require the spec to build
/// the live pipe.
pub fn skip_byte_drift_without_spec(policy: &Policy, report: &mut Report) {
    if policy.replay.byte_identical_pipes.is_empty() {
        report.push(ReportEntry::pass("byte_drift"));
        return;
    }
    report.push(ReportEntry::warn(
        "byte_drift",
        "skipped: byte_identical_pipes set but no --spec provided",
    ));
}

struct RecordedInteraction {
    pipe_name: String,
    method: String,
    path: String,
    request_body: Bytes,
    recorded_response: Bytes,
}

async fn collect_events(
    recording_path: &Path,
    runtime: &Arc<dyn Runtime>,
) -> Result<Vec<RecordingEvent>, ProximaError> {
    let source = source_for_recording(recording_path, runtime)?;
    let mut stream = source.events();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item?);
    }
    Ok(events)
}

/// Build a `DynRecordingSource` for `path`, dispatching on extension.
/// `.bin` → `BinSource`, `.jsonl` → `JsonlSource`. Mirrors the same
/// dispatch in `replay_walker::source_for_recording`.
fn source_for_recording(
    path: &Path,
    runtime: &Arc<dyn Runtime>,
) -> Result<DynRecordingSource, ProximaError> {
    let extension = path
        .extension()
        .and_then(|raw| raw.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("bin") => Ok(std::sync::Arc::new(BinSource::new(
            path,
            Arc::clone(runtime),
        ))),
        Some("jsonl") => Ok(std::sync::Arc::new(JsonlSource::new(
            path,
            Arc::clone(runtime),
        ))),
        Some(other) => Err(ProximaError::Config(format!(
            "byte_drift: unsupported recording extension '{other}' \
             (expected .bin or .jsonl)"
        ))),
        None => Err(ProximaError::Config(format!(
            "byte_drift: recording path {path:?} has no extension; \
             expected .bin or .jsonl"
        ))),
    }
}

fn group_recorded(events: &[RecordingEvent], policy: &Policy) -> Vec<RecordedInteraction> {
    let mut pipe_by_interaction: BTreeMap<InteractionId, String> = BTreeMap::new();
    let mut method_path: BTreeMap<InteractionId, (String, String)> = BTreeMap::new();
    let mut req_bodies: BTreeMap<InteractionId, Vec<u8>> = BTreeMap::new();
    let mut resp_bodies: BTreeMap<InteractionId, Vec<u8>> = BTreeMap::new();

    for event in events {
        match &event.event {
            ProtocolEvent::Http(HttpEvent::Started { pipe, request, .. }) => {
                pipe_by_interaction.insert(event.id, pipe.clone());
                method_path.insert(event.id, (request.method.clone(), request.path.clone()));
            }
            ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. }) => {
                if let Some(parent) = event.parent {
                    req_bodies
                        .entry(parent)
                        .or_default()
                        .extend_from_slice(data.as_ref());
                }
            }
            ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => {
                if let Some(parent) = event.parent {
                    resp_bodies
                        .entry(parent)
                        .or_default()
                        .extend_from_slice(data.as_ref());
                }
            }
            _ => {}
        }
    }

    let allowlist: std::collections::BTreeSet<&str> = policy
        .replay
        .byte_identical_pipes
        .iter()
        .map(String::as_str)
        .collect();

    let mut out = Vec::new();
    for (id, pipe) in pipe_by_interaction {
        if !allowlist.contains(pipe.as_str()) {
            continue;
        }
        let Some((method, path)) = method_path.remove(&id) else {
            continue;
        };
        let request_body = Bytes::from(req_bodies.remove(&id).unwrap_or_default());
        let recorded_response = Bytes::from(resp_bodies.remove(&id).unwrap_or_default());
        out.push(RecordedInteraction {
            pipe_name: pipe,
            method,
            path,
            request_body,
            recorded_response,
        });
    }
    out
}

/// Find the sub-spec for a named pipe, handling both proxima.toml
/// shapes:
///
/// - **Named-map form** (`[pipes.api] chain = [...]`) — `spec.pipes.<name>`
/// - **Array-of-tables form** (`[[pipe]] name = "api" ...`) — search
///   `spec.pipe[]` for an entry whose `name` matches.
///
/// The `name` field is normalized into the returned sub-spec
/// (existing `name` removed and re-inserted), matching the contract
/// `App::pipe` and `App::load_full` enforce before calling `load()`.
async fn build_pipe_for(
    pipe_name: &str,
    spec: &Value,
    context: &LoadContext,
) -> Result<PipeHandle, ProximaError> {
    let mut sub_spec = find_pipe_subspec(spec, pipe_name).ok_or_else(|| {
        ProximaError::Config(format!(
            "byte_drift: spec has no pipe named '{pipe_name}' \
             (looked in spec.pipes.<name> and spec.pipe[].name)"
        ))
    })?;
    if let Some(obj) = sub_spec.as_object_mut() {
        obj.remove("name");
        obj.insert("name".into(), Value::String(pipe_name.to_string()));
    }
    load(sub_spec, context).await
}

fn find_pipe_subspec(spec: &Value, pipe_name: &str) -> Option<Value> {
    // Named-map form: spec.pipes.<name>
    if let Some(named) = spec.get("pipes").and_then(|p| p.get(pipe_name)) {
        return Some(named.clone());
    }
    // Array-of-tables form: spec.pipe[].name == pipe_name
    if let Some(Value::Array(arr)) = spec.get("pipe") {
        for entry in arr {
            if entry.get("name").and_then(Value::as_str) == Some(pipe_name) {
                return Some(entry.clone());
            }
        }
    }
    None
}

fn build_request(method: &str, path: &str, body: Bytes) -> Result<Request<Bytes>, ProximaError> {
    Request::builder()
        .method(method)
        .path(path)
        .body(body)
        .build()
}

async fn collect_response_bytes(response: Response<Bytes>) -> Result<Bytes, ProximaError> {
    response.collect_body().await
}

fn first_diff(a: &Bytes, b: &Bytes) -> Option<usize> {
    if a == b {
        return None;
    }
    let common = a.len().min(b.len());
    for offset in 0..common {
        if a[offset] != b[offset] {
            return Some(offset);
        }
    }
    Some(common)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;
    use crate::recording::event::FrameMetadata;
    use crate::recording::event::{
        HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent, RequestHeader,
    };
    use time::OffsetDateTime;

    fn started(pipe: &str, method: &str, path: &str) -> ProtocolEvent {
        let mut header = RequestHeader::default();
        header.method = method.into();
        header.path = path.into();
        ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: pipe.into(),
            request: header,
            meta: None,
        })
    }

    fn response_chunk(data: &[u8]) -> ProtocolEvent {
        ProtocolEvent::Http(HttpEvent::ResponseChunk {
            data: Bytes::copy_from_slice(data),
            metadata: FrameMetadata::default(),
        })
    }

    fn ended() -> ProtocolEvent {
        ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 0,
            meta: RecordMeta::default(),
        })
    }

    fn make(
        id: InteractionId,
        parent: Option<InteractionId>,
        event: ProtocolEvent,
    ) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms: 0,
            parent,
            event,
        }
    }

    #[test]
    fn group_recorded_filters_to_allowlist_only() {
        let policy = Policy::parse_str(
            r#"
            [replay]
            byte_identical_pipes = ["target"]
        "#,
        )
        .expect("parse");
        let id_t = InteractionId::from_bytes([1; 16]);
        let id_o = InteractionId::from_bytes([2; 16]);
        let chunk_t = InteractionId::from_bytes([3; 16]);
        let chunk_o = InteractionId::from_bytes([4; 16]);

        let events = vec![
            make(id_t, None, started("target", "GET", "/x")),
            make(chunk_t, Some(id_t), response_chunk(b"hello")),
            make(id_o, None, started("other", "GET", "/y")),
            make(chunk_o, Some(id_o), response_chunk(b"world")),
        ];

        let grouped = group_recorded(&events, &policy);
        assert_eq!(grouped.len(), 1, "only target should be grouped");
        assert_eq!(grouped[0].pipe_name, "target");
        assert_eq!(&grouped[0].recorded_response[..], b"hello");
    }

    #[test]
    fn group_recorded_concatenates_chunks_in_order() {
        let policy = Policy::parse_str(
            r#"
            [replay]
            byte_identical_pipes = ["target"]
        "#,
        )
        .expect("parse");
        let id_t = InteractionId::from_bytes([1; 16]);
        let c1 = InteractionId::from_bytes([2; 16]);
        let c2 = InteractionId::from_bytes([3; 16]);

        let events = vec![
            make(id_t, None, started("target", "GET", "/")),
            make(c1, Some(id_t), response_chunk(b"hel")),
            make(c2, Some(id_t), response_chunk(b"lo")),
            make(InteractionId::from_bytes([4; 16]), Some(id_t), ended()),
        ];
        let grouped = group_recorded(&events, &policy);
        assert_eq!(&grouped[0].recorded_response[..], b"hello");
    }

    #[test]
    fn first_diff_identifies_offset() {
        let a = Bytes::from_static(b"hello world");
        let b = Bytes::from_static(b"hello porld");
        assert_eq!(first_diff(&a, &b), Some(6));
    }

    #[test]
    fn first_diff_matches_returns_none() {
        let a = Bytes::from_static(b"same");
        let b = Bytes::from_static(b"same");
        assert_eq!(first_diff(&a, &b), None);
    }

    #[test]
    fn find_pipe_subspec_handles_named_map_form() {
        let spec = serde_json::json!({
            "pipes": {
                "api": { "kv": "file" },
            }
        });
        let found = find_pipe_subspec(&spec, "api").expect("found");
        assert_eq!(found, serde_json::json!({ "kv": "file" }));
    }

    #[test]
    fn find_pipe_subspec_handles_array_of_tables_form() {
        let spec = serde_json::json!({
            "pipe": [
                { "name": "api", "kv": "file" },
                { "name": "other", "synth": {} },
            ]
        });
        let found = find_pipe_subspec(&spec, "api").expect("found");
        assert_eq!(found, serde_json::json!({ "name": "api", "kv": "file" }));
    }

    #[test]
    fn find_pipe_subspec_returns_none_for_missing() {
        let spec = serde_json::json!({ "pipes": { "other": {} } });
        assert!(find_pipe_subspec(&spec, "missing").is_none());
    }

    #[test]
    fn first_diff_length_mismatch_returns_shortest() {
        let a = Bytes::from_static(b"short");
        let b = Bytes::from_static(b"shorter");
        assert_eq!(first_diff(&a, &b), Some(5));
    }

    #[test]
    fn skip_byte_drift_without_spec_warns_when_pipes_flagged() {
        let policy = Policy::parse_str(
            r#"
            [replay]
            byte_identical_pipes = ["target"]
        "#,
        )
        .expect("parse");
        let mut report = Report::new();
        skip_byte_drift_without_spec(&policy, &mut report);
        let warns: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Warn))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(warns.contains(&"byte_drift"));
    }

    #[test]
    fn skip_byte_drift_passes_when_no_pipes_flagged() {
        let policy = Policy::default();
        let mut report = Report::new();
        skip_byte_drift_without_spec(&policy, &mut report);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.warn_count(), 0);
    }
}
