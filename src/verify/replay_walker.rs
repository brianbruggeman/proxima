//! Replay-recording assertions. Streams the event sequence from a
//! `.bin` recording, classifies each event, and emits a report
//! entry per enabled assertion.
//!
//! v1 ships four structural assertions over the recorded event
//! stream, named per the parking-lot spec — one rule name per check,
//! regardless of outcome:
//!
//! - **`unauthorized_upstream_call`** — every distinct pipe name in
//!   the recording must appear in `policy.replay.allowed_upstreams`
//!   when that list is non-empty. (The parking-lot schema names the
//!   field `allowed_upstreams` for historical reasons; in practice
//!   it is a pipe-identifier allowlist.) Violation ⇒ FAIL.
//! - **`inferred_not_recorded`** — every pipe in
//!   `policy.replay.must_derive_from_record` must produce only events
//!   whose [`RecordMeta::source`](crate::recording::EventSource) is
//!   `Recorded` (or `None`, treated as recorded for backward compat).
//!   An `Inferred` event for a flagged pipe ⇒ FAIL.
//! - **`replay.terminated_cleanly`** — every `HttpEvent::Started`
//!   must be matched by a corresponding `HttpEvent::Ended` (by
//!   `InteractionId`). A dangling start ⇒ WARN.
//! - **`replay.recording_summary`** — PASS with an event count and
//!   the distinct pipe names touched. Always informational.
//!
//! `byte_drift` (runtime spec-join + re-execution + byte diff) ships
//! separately in [`byte_drift`](super::byte_drift).
//!
//! The reframed-from-`nondeclared_nondeterminism` rule
//! `idempotence_contract` is a pure spec-policy join: it fires when
//! a pipe declared `idempotent = false` in the spec is also listed
//! in `policy.replay.must_derive_from_record` (a contradiction —
//! you cannot require deterministic recording-derivation from a
//! pipe declared non-idempotent). Needs `spec: Option<&Value>` at
//! the walker entry; absent spec ⇒ rule is skipped with a WARN.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use futures::StreamExt;
use serde_json::Value;

use std::sync::Arc;

use crate::error::ProximaError;
use crate::recording::bin::source::BinSource;
use crate::recording::event::{
    EventSource, HttpEvent, InteractionId, ProtocolEvent, RecordingEvent,
};
use crate::recording::jsonl::source::JsonlSource;
use crate::recording::source::DynRecordingSource;
use crate::runtime::Runtime;

use super::policy::Policy;
use super::report::{Report, ReportEntry};

/// Stream a `.bin` recording from disk and run replay assertions
/// against it. Passing a spec value enables the
/// `idempotence_contract` rule.
pub async fn verify_replay(
    recording_path: &Path,
    policy: &Policy,
    runtime: &Arc<dyn Runtime>,
) -> Result<Report, ProximaError> {
    verify_replay_with_spec(recording_path, policy, None, runtime).await
}

/// Same as [`verify_replay`] but with spec-aware rules enabled
/// (`idempotence_contract`). Spec is the parsed top-level value.
///
/// Recording format dispatches on extension: `.bin` → [`BinSource`],
/// `.jsonl` → [`JsonlSource`]. Other extensions error out — proxima
/// ships two recording formats today.
pub async fn verify_replay_with_spec(
    recording_path: &Path,
    policy: &Policy,
    spec: Option<&Value>,
    runtime: &Arc<dyn Runtime>,
) -> Result<Report, ProximaError> {
    let source = source_for_recording(recording_path, runtime)?;
    let mut stream = source.events();
    let mut events: Vec<RecordingEvent> = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item?);
    }
    Ok(verify_replay_events(&events, policy, spec))
}

/// Build a `DynRecordingSource` for `path`, dispatching on extension.
/// `.bin` → `BinSource`, `.jsonl` → `JsonlSource`.
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
            "verify_replay: unsupported recording extension '{other}' \
             (expected .bin or .jsonl)"
        ))),
        None => Err(ProximaError::Config(format!(
            "verify_replay: recording path {path:?} has no extension; \
             expected .bin or .jsonl"
        ))),
    }
}

/// Sync entry — run the assertions over an in-memory event list.
/// Used by tests; the async paths above collect into this.
pub(crate) fn verify_replay_events(
    events: &[RecordingEvent],
    policy: &Policy,
    spec: Option<&Value>,
) -> Report {
    let mut report = Report::new();

    let mut pipe_touches: BTreeMap<String, usize> = BTreeMap::new();
    let mut dangling_starts: BTreeMap<InteractionId, String> = BTreeMap::new();
    // Track the originating pipe per InteractionId so a downstream
    // `HttpEvent::Ended { meta }` can be attributed back to its pipe
    // for the `inferred_not_recorded` check (Ended carries the meta;
    // Started carries the pipe name).
    let mut pipe_by_interaction: BTreeMap<InteractionId, String> = BTreeMap::new();
    let mut inferred_violations: BTreeMap<String, usize> = BTreeMap::new();

    for event in events {
        match &event.event {
            ProtocolEvent::Http(HttpEvent::Started { pipe, meta, .. }) => {
                *pipe_touches.entry(pipe.clone()).or_default() += 1;
                dangling_starts.insert(event.id, pipe.clone());
                pipe_by_interaction.insert(event.id, pipe.clone());
                if let Some(meta) = meta
                    && meta.source == Some(EventSource::Inferred)
                    && is_must_derive_from_record(policy, pipe)
                {
                    *inferred_violations.entry(pipe.clone()).or_default() += 1;
                }
            }
            ProtocolEvent::Http(HttpEvent::Ended { meta, .. }) => {
                let interaction_id = event.parent.unwrap_or(event.id);
                dangling_starts.remove(&interaction_id);
                if meta.source == Some(EventSource::Inferred)
                    && let Some(pipe) = pipe_by_interaction.get(&interaction_id)
                    && is_must_derive_from_record(policy, pipe)
                {
                    *inferred_violations.entry(pipe.clone()).or_default() += 1;
                }
            }
            _ => {}
        }
    }

    run_pipes_in_allowlist(&pipe_touches, policy, &mut report);
    run_inferred_not_recorded(&inferred_violations, policy, &mut report);
    run_idempotence_contract(spec, policy, &mut report);
    run_terminated_cleanly(&dangling_starts, &mut report);
    run_recording_summary(events.len(), &pipe_touches, &mut report);

    report
}

/// Build a `pipe_name → idempotent: bool` map from the spec.
/// Reads both spec shapes — `[pipes.<name>] idempotent = false` and
/// `[[pipe]] name = "<name>" idempotent = false`. Pipes that omit
/// the field are treated as idempotent (the safe default).
fn idempotence_map(spec: &Value) -> BTreeMap<String, bool> {
    let mut map: BTreeMap<String, bool> = BTreeMap::new();

    if let Some(Value::Object(named)) = spec.get("pipes") {
        for (name, entry) in named {
            let value = entry
                .get("idempotent")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            map.insert(name.clone(), value);
        }
    }

    if let Some(Value::Array(arr)) = spec.get("pipe") {
        for entry in arr {
            let Some(name) = entry.get("name").and_then(Value::as_str) else {
                continue;
            };
            let value = entry
                .get("idempotent")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            map.insert(name.to_string(), value);
        }
    }

    map
}

/// `idempotence_contract` — every pipe in
/// `policy.replay.must_derive_from_record` must NOT be declared
/// `idempotent = false` in the spec. Requiring deterministic
/// recording-derivation from a non-idempotent pipe is a
/// contradiction.
///
/// Sanity-check rule: pure spec-policy join, no runtime detection.
/// Skipped (WARN) when no spec is supplied.
fn run_idempotence_contract(spec: Option<&Value>, policy: &Policy, report: &mut Report) {
    let must_derive = &policy.replay.must_derive_from_record;
    if must_derive.is_empty() {
        report.push(ReportEntry::pass("idempotence_contract"));
        return;
    }
    let Some(spec) = spec else {
        report.push(ReportEntry::warn(
            "idempotence_contract",
            "skipped: policy.replay.must_derive_from_record is non-empty \
             but no spec was provided to the walker",
        ));
        return;
    };
    let map = idempotence_map(spec);
    let mut contradictions: Vec<&str> = Vec::new();
    for pipe in must_derive {
        if matches!(map.get(pipe.as_str()), Some(false)) {
            contradictions.push(pipe.as_str());
        }
    }
    if contradictions.is_empty() {
        report.push(ReportEntry::pass("idempotence_contract"));
    } else {
        report.push(ReportEntry::fail(
            "idempotence_contract",
            format!(
                "pipes declared `idempotent = false` in spec cannot be \
                 required to derive from recording: {}",
                contradictions.join(", ")
            ),
        ));
    }
}

fn is_must_derive_from_record(policy: &Policy, pipe: &str) -> bool {
    policy
        .replay
        .must_derive_from_record
        .iter()
        .any(|name| name == pipe)
}

fn run_inferred_not_recorded(
    violations: &BTreeMap<String, usize>,
    policy: &Policy,
    report: &mut Report,
) {
    if policy.replay.must_derive_from_record.is_empty() {
        report.push(ReportEntry::pass("inferred_not_recorded"));
        return;
    }
    if violations.is_empty() {
        report.push(ReportEntry::pass("inferred_not_recorded"));
        return;
    }
    let detail: Vec<String> = violations
        .iter()
        .map(|(pipe, count)| format!("{pipe} ({count} inferred events)"))
        .collect();
    report.push(ReportEntry::fail(
        "inferred_not_recorded",
        format!(
            "flagged pipes produced inferred events: {}",
            detail.join(", ")
        ),
    ));
}

fn run_pipes_in_allowlist(
    pipe_touches: &BTreeMap<String, usize>,
    policy: &Policy,
    report: &mut Report,
) {
    let allowlist: &[String] = &policy.replay.allowed_upstreams;
    if allowlist.is_empty() {
        report.push(ReportEntry::pass("unauthorized_upstream_call"));
        return;
    }
    let allowed: BTreeSet<&str> = allowlist.iter().map(String::as_str).collect();
    let violations: Vec<&str> = pipe_touches
        .keys()
        .filter(|name| !allowed.contains(name.as_str()))
        .map(String::as_str)
        .collect();
    if violations.is_empty() {
        report.push(ReportEntry::pass("unauthorized_upstream_call"));
    } else {
        report.push(ReportEntry::fail(
            "unauthorized_upstream_call",
            format!("pipes outside allowlist: {}", violations.join(", ")),
        ));
    }
}

fn run_terminated_cleanly(dangling_starts: &BTreeMap<InteractionId, String>, report: &mut Report) {
    if dangling_starts.is_empty() {
        report.push(ReportEntry::pass("replay.terminated_cleanly"));
        return;
    }
    let pipes: Vec<&str> = dangling_starts.values().map(String::as_str).collect();
    report.push(ReportEntry::warn(
        "replay.terminated_cleanly",
        format!(
            "{} unterminated http interaction(s) for pipe(s): {}",
            dangling_starts.len(),
            pipes.join(", ")
        ),
    ));
}

fn run_recording_summary(
    event_count: usize,
    pipe_touches: &BTreeMap<String, usize>,
    report: &mut Report,
) {
    let pipes: Vec<String> = pipe_touches
        .iter()
        .map(|(name, count)| format!("{name}×{count}"))
        .collect();
    let detail = if pipes.is_empty() {
        format!("{event_count} events, no http pipes touched")
    } else {
        format!("{event_count} events, pipes: {}", pipes.join(", "))
    };
    let entry = ReportEntry {
        level: super::report::Level::Pass,
        rule: "replay.recording_summary".to_string(),
        detail,
    };
    report.push(entry);
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
    use crate::recording::event::{
        HttpEvent, ProtocolEvent, RecordMeta, RecordingEvent, RequestHeader,
    };
    use time::OffsetDateTime;

    fn make_event(
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

    fn http_started(pipe: &str) -> ProtocolEvent {
        ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: pipe.into(),
            request: RequestHeader::default(),
            meta: None,
        })
    }

    fn http_ended() -> ProtocolEvent {
        ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 0,
            meta: RecordMeta::default(),
        })
    }

    fn http_ended_with_source(source: EventSource) -> ProtocolEvent {
        let mut meta = RecordMeta::default();
        meta.source = Some(source);
        ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 0,
            meta,
        })
    }

    fn fresh_id(byte: u8) -> InteractionId {
        InteractionId::from_bytes([byte; 16])
    }

    fn parse_policy(text: &str) -> Policy {
        Policy::parse_str(text).expect("parse policy")
    }

    #[test]
    fn empty_recording_passes_with_summary() {
        let report = verify_replay_events(&[], &Policy::default(), None);
        let names: Vec<&str> = report.entries.iter().map(|e| e.rule.as_str()).collect();
        assert!(names.contains(&"replay.recording_summary"));
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn started_without_ended_warns() {
        let start_id = fresh_id(1);
        let events = vec![make_event(start_id, None, http_started("api"))];
        let report = verify_replay_events(&events, &Policy::default(), None);
        let warns: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Warn))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(warns.contains(&"replay.terminated_cleanly"));
    }

    #[test]
    fn matched_start_end_pair_passes_clean_termination() {
        let start_id = fresh_id(2);
        let end_id = fresh_id(3);
        let events = vec![
            make_event(start_id, None, http_started("api")),
            make_event(end_id, Some(start_id), http_ended()),
        ];
        let report = verify_replay_events(&events, &Policy::default(), None);
        assert_eq!(report.warn_count(), 0);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn pipe_outside_allowlist_fails() {
        let start_id = fresh_id(4);
        let end_id = fresh_id(5);
        let events = vec![
            make_event(start_id, None, http_started("rogue_pipe")),
            make_event(end_id, Some(start_id), http_ended()),
        ];
        let policy = parse_policy(
            r#"
            [replay]
            allowed_upstreams = ["safe_pipe"]
        "#,
        );
        let report = verify_replay_events(&events, &policy, None);
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(fails.contains(&"unauthorized_upstream_call"));
    }

    #[test]
    fn pipe_in_allowlist_passes() {
        let start_id = fresh_id(6);
        let end_id = fresh_id(7);
        let events = vec![
            make_event(start_id, None, http_started("safe_pipe")),
            make_event(end_id, Some(start_id), http_ended()),
        ];
        let policy = parse_policy(
            r#"
            [replay]
            allowed_upstreams = ["safe_pipe"]
        "#,
        );
        let report = verify_replay_events(&events, &policy, None);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn empty_allowlist_passes_any_pipe() {
        let start_id = fresh_id(8);
        let end_id = fresh_id(9);
        let events = vec![
            make_event(start_id, None, http_started("any_pipe")),
            make_event(end_id, Some(start_id), http_ended()),
        ];
        let report = verify_replay_events(&events, &Policy::default(), None);
        // empty allowlist = no allowlist enforcement
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn inferred_event_for_must_derive_pipe_fails() {
        let start_id = fresh_id(20);
        let end_id = fresh_id(21);
        let events = vec![
            make_event(start_id, None, http_started("fetch")),
            make_event(
                end_id,
                Some(start_id),
                http_ended_with_source(EventSource::Inferred),
            ),
        ];
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let report = verify_replay_events(&events, &policy, None);
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(fails.contains(&"inferred_not_recorded"), "got: {fails:?}");
    }

    #[test]
    fn recorded_event_for_must_derive_pipe_passes() {
        let start_id = fresh_id(22);
        let end_id = fresh_id(23);
        let events = vec![
            make_event(start_id, None, http_started("fetch")),
            make_event(
                end_id,
                Some(start_id),
                http_ended_with_source(EventSource::Recorded),
            ),
        ];
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let report = verify_replay_events(&events, &policy, None);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn empty_must_derive_list_passes() {
        let events = vec![make_event(fresh_id(24), None, http_started("any_pipe"))];
        let report = verify_replay_events(&events, &Policy::default(), None);
        let pass_rules: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Pass))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(pass_rules.contains(&"inferred_not_recorded"));
    }

    #[test]
    fn idempotence_contract_passes_when_no_must_derive() {
        let events = vec![make_event(fresh_id(30), None, http_started("p"))];
        let report = verify_replay_events(&events, &Policy::default(), None);
        let passes: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Pass))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(passes.contains(&"idempotence_contract"));
    }

    #[test]
    fn idempotence_contract_warns_when_spec_missing() {
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let events = vec![make_event(fresh_id(31), None, http_started("fetch"))];
        let report = verify_replay_events(&events, &policy, None);
        let warns: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Warn))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(warns.contains(&"idempotence_contract"));
    }

    #[test]
    fn idempotence_contract_fails_when_must_derive_pipe_is_non_idempotent_named_map() {
        let spec = serde_json::json!({
            "pipes": {
                "fetch": { "idempotent": false, "http": "https://example.com" }
            }
        });
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let events = vec![make_event(fresh_id(32), None, http_started("fetch"))];
        let report = verify_replay_events(&events, &policy, Some(&spec));
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(
            fails.contains(&"idempotence_contract"),
            "expected FAIL idempotence_contract, got: {fails:?}"
        );
    }

    #[test]
    fn idempotence_contract_fails_when_must_derive_pipe_is_non_idempotent_array_form() {
        let spec = serde_json::json!({
            "pipe": [
                { "name": "fetch", "idempotent": false, "http": "https://example.com" }
            ]
        });
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let events = vec![make_event(fresh_id(33), None, http_started("fetch"))];
        let report = verify_replay_events(&events, &policy, Some(&spec));
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(fails.contains(&"idempotence_contract"));
    }

    #[test]
    fn idempotence_contract_passes_when_idempotent_field_absent() {
        let spec = serde_json::json!({
            "pipes": {
                "fetch": { "http": "https://example.com" }
            }
        });
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let events = vec![make_event(fresh_id(34), None, http_started("fetch"))];
        let report = verify_replay_events(&events, &policy, Some(&spec));
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn idempotence_contract_passes_when_idempotent_explicitly_true() {
        let spec = serde_json::json!({
            "pipes": {
                "fetch": { "idempotent": true, "http": "https://example.com" }
            }
        });
        let policy = parse_policy(
            r#"
            [replay]
            must_derive_from_record = ["fetch"]
        "#,
        );
        let events = vec![make_event(fresh_id(35), None, http_started("fetch"))];
        let report = verify_replay_events(&events, &policy, Some(&spec));
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn summary_lists_touched_pipes_with_counts() {
        let id_one = fresh_id(10);
        let id_two = fresh_id(11);
        let id_three = fresh_id(12);
        let events = vec![
            make_event(id_one, None, http_started("api")),
            make_event(id_two, None, http_started("api")),
            make_event(id_three, None, http_started("admin")),
        ];
        let report = verify_replay_events(&events, &Policy::default(), None);
        let summary_entry = report
            .entries
            .iter()
            .find(|entry| entry.rule == "replay.recording_summary")
            .expect("summary entry present");
        assert!(summary_entry.detail.contains("api×2"));
        assert!(summary_entry.detail.contains("admin×1"));
    }
}
