//! Tier-2 test fixtures: `ConfigFixture`, `LivePlane`, `cassette_pipe`.
//! Tier-1 (runtime drivers, `TestCtx`, panic capture) lives in `proxima-test`.
//!
//! Cassettes: a `#[proxima::test(cassette = "name")]` body obtains a record-or-
//! replay `Handler` from [`cassette_pipe`]. Record uses a deterministic
//! synchronous tee ([`RecordingTee`]) over the recording-core event model +
//! `JsonlSink` (NOT the serving-path `RecordUpstream`, whose detached drainer
//! has no completion signal). Replay uses `ReplayUpstream::from_jsonl`. See
//! `docs/proxima-test/edges.md`.

// test-harness code: a panic IS the failure path here (mirrors proxima-macros),
// so expect in the harness lines is intentional.
#![allow(clippy::expect_used)]

pub use proxima_test::*;

pub use crate::cassette_config::{
    CassetteConfig, CassetteHooks, DuplicatePolicy, ModePolicy, RerecordDecision, RerecordPolicy,
    StaleDecision, StalenessPolicy,
};

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use time::OffsetDateTime;

use proxima_core::ProximaError;
use proxima_recording::event::{
    HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent, RequestHeader,
};
use proxima_recording::replay::{
    CASSETTE_META_KIND, CassetteMeta, MatchSpec, ReplayUpstream, content_digest,
    match_key_from_request_with,
};

use crate::recording::{
    AccumulatingSink, DynRecordingSink, FormatKind, LazyFanOut, SinkSpec, deferred_runtime,
};

use proxima_primitives::pipe::SendPipe;

use crate::app::{App, MountTarget};
use crate::control_plane::ControlPlane;
use crate::daemon_control_plane::{DaemonControlPlane, PipeConfig};
use crate::pipe::{Handler, PipeHandle, into_handle};
use crate::request::{Request, Response};

// ---------------------------------------------------------------------------
// cassette_pipe free fn (tier-2 replacement for TestCtx::cassette_pipe)
// ---------------------------------------------------------------------------

/// Return the record-or-replay `Handler` for this cassette test. Call it with
/// the test's real upstream `inner`: in record mode the interaction is teed
/// to the cassette and `inner` is exercised; in replay mode `inner` is
/// ignored and the recorded response is served. Panics if the test was not
/// declared with `cassette = "..."`.
///
/// Policy comes from [`CassetteConfig::resolve_for_dir`]: defaults ←
/// `tests/cassettes/config.toml` ← `PROXIMA_CASSETTE_*` env. Use
/// [`cassette_pipe_with`] / [`cassette_pipe_with_hooks`] for per-test
/// overrides.
///
/// # Errors
/// Propagates I/O / decode failures from creating the sink or loading the
/// cassette, plus policy failures (rerecord=fail, stale cassette, divergent
/// duplicate recordings).
pub async fn cassette_pipe<P: Handler>(
    cx: &proxima_test::TestCtx,
    inner: P,
) -> Result<PipeHandle, ProximaError> {
    let cassette = cx
        .cassette()
        .expect("cassette_pipe called on a test without `cassette = \"...\"`");
    let cassette_dir = cassette
        .path
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let config = CassetteConfig::resolve_for_dir(&cassette_dir)?;
    cassette_pipe_with_hooks(cx, inner, config, CassetteHooks::default()).await
}

/// [`cassette_pipe`] with an explicit config (fluent escape hatch).
///
/// # Errors
/// Same failure surface as [`cassette_pipe`].
pub async fn cassette_pipe_with<P: Handler>(
    cx: &proxima_test::TestCtx,
    inner: P,
    config: CassetteConfig,
) -> Result<PipeHandle, ProximaError> {
    cassette_pipe_with_hooks(cx, inner, config, CassetteHooks::default()).await
}

/// [`cassette_pipe`] with explicit config AND programmable hooks (the last
/// escape hatch — hooks win over the declarative policy per-decision).
///
/// # Errors
/// Same failure surface as [`cassette_pipe`].
pub async fn cassette_pipe_with_hooks<P: Handler>(
    cx: &proxima_test::TestCtx,
    inner: P,
    config: CassetteConfig,
    hooks: CassetteHooks,
) -> Result<PipeHandle, ProximaError> {
    let cassette = cx
        .cassette()
        .expect("cassette_pipe called on a test without `cassette = \"...\"`");
    let match_spec = MatchSpec {
        include_body: config.match_body,
    };
    match config.mode.resolve(cassette.mode) {
        proxima_test::Mode::Record => {
            if std::fs::exists(&cassette.path).unwrap_or(false) {
                let decision = match &hooks.on_rerecord {
                    Some(hook) => hook(&cassette.path),
                    None => config.rerecord.decision(),
                };
                match decision {
                    RerecordDecision::Truncate => {
                        let _ = std::fs::remove_file(&cassette.path);
                    }
                    RerecordDecision::Backup => {
                        let mut backup = cassette.path.clone().into_os_string();
                        backup.push(".bak");
                        std::fs::rename(&cassette.path, &backup).map_err(|error| {
                            ProximaError::Record(format!(
                                "backup cassette {} before rerecord: {error}",
                                cassette.path.display()
                            ))
                        })?;
                    }
                    RerecordDecision::Fail => {
                        return Err(ProximaError::Record(format!(
                            "cassette {} exists and rerecord policy is `fail`; delete it, \
                             relax PROXIMA_CASSETTE_RERECORD, or replay instead",
                            cassette.path.display()
                        )));
                    }
                    RerecordDecision::UseExisting => {
                        return replay_cassette(&cassette.path, &config, &hooks, match_spec).await;
                    }
                }
            }
            if let Some(parent) = cassette.path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // armed spigot + batch=1 so every teed event is written
            // immediately (cassette replay reads it back in-process).
            let spigot = deferred_runtime();
            let _ = spigot.set(crate::app::offline_runtime()?);
            let durable = std::sync::Arc::new(LazyFanOut::new(
                vec![SinkSpec::new(
                    cassette.path.to_string_lossy().into_owned(),
                    FormatKind::Json,
                )],
                spigot,
            ));
            let sink: DynRecordingSink = std::sync::Arc::new(AccumulatingSink::new(durable, 1));
            let meta = CassetteMeta {
                recorded_at_ms: now_unix_ms(),
                recorder: format!("proxima {}", env!("CARGO_PKG_VERSION")),
                request_bodies: true,
            };
            sink.append(RecordingEvent {
                id: InteractionId::new(),
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Custom {
                    kind: CASSETTE_META_KIND.to_string(),
                    payload: meta.to_payload(),
                },
            })
            .await?;
            sink.flush().await?;
            Ok(into_handle(RecordingTee {
                inner: into_handle(inner),
                sink,
                match_spec,
                duplicates: config.duplicates,
                seen: Arc::new(Mutex::new(HashMap::new())),
            }))
        }
        proxima_test::Mode::Replay => {
            replay_cassette(&cassette.path, &config, &hooks, match_spec).await
        }
    }
}

async fn replay_cassette(
    path: &Path,
    config: &CassetteConfig,
    hooks: &CassetteHooks,
    match_spec: MatchSpec,
) -> Result<PipeHandle, ProximaError> {
    let runtime: std::sync::Arc<dyn crate::runtime::Runtime> = crate::app::offline_runtime()?;
    let replay = ReplayUpstream::from_jsonl_with(path, "cassette", runtime, match_spec)
        .await
        .map_err(|error| {
            ProximaError::Record(format!(
                "load cassette {}: {error}; delete the file or run with \
                 PROXIMA_CASSETTE=record to re-record",
                path.display()
            ))
        })?;
    if config.match_body
        && let Some(meta) = replay.meta()
        && !meta.request_bodies
    {
        return Err(ProximaError::Record(format!(
            "cassette {} predates request-body capture but match_body is on; re-record it",
            path.display()
        )));
    }
    if let Some(max_age) = config.max_age() {
        let age_ms = replay
            .meta()
            .map(|meta| now_unix_ms().saturating_sub(meta.recorded_at_ms));
        let fresh = age_ms.is_some_and(|age| u128::from(age) <= max_age.as_millis());
        if !fresh {
            let decision = match &hooks.on_stale {
                Some(hook) => hook(path, replay.meta()),
                None => config.staleness.decision(),
            };
            let describe_age = age_ms.map_or_else(
                || "age unknown (no provenance stamp)".to_string(),
                |age| format!("age {age}ms exceeds max_age {}ms", max_age.as_millis()),
            );
            match decision {
                StaleDecision::Fail => {
                    return Err(ProximaError::Record(format!(
                        "stale cassette {}: {describe_age}; re-record with \
                         PROXIMA_CASSETTE=record or raise PROXIMA_CASSETTE_MAX_AGE_MS",
                        path.display()
                    )));
                }
                StaleDecision::Proceed => {
                    // eprintln, not telemetry: no recorder is wired in the
                    // test harness and the warning must reach the human.
                    eprintln!(
                        "warning: stale cassette {}: {describe_age}; replaying anyway",
                        path.display()
                    );
                }
            }
        }
    }
    Ok(into_handle(replay))
}

fn now_unix_ms() -> u64 {
    let nanos = OffsetDateTime::now_utc().unix_timestamp_nanos();
    u64::try_from(nanos / 1_000_000).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// config fixture
// ---------------------------------------------------------------------------

/// A typed-on-demand config value loaded through proxima's format-agnostic
/// registry. rstest has no config substrate; this lifts proxima's loader
/// (`proxima-config` + `crate::load`) into a fixture. Config is UNTYPED
/// `serde_json::Value` end-to-end in proxima, so [`Self::typed`] is a test-side
/// projection. [`Self::into_pipe`] realizes the config as a live `Handler`.
pub struct ConfigFixture {
    value: serde_json::Value,
}

impl ConfigFixture {
    /// Parse raw config text; `hint` is a format name (`"toml"`/`"json"`/`"yaml"`/…)
    /// or `None` to sniff.
    ///
    /// # Errors
    /// Propagates registry-build and parse failures.
    pub async fn from_raw(raw: &str, hint: Option<&str>) -> Result<Self, ProximaError> {
        let registry = crate::config_format::default_config_format_registry()?;
        Ok(Self {
            value: registry.parse_with_hint(raw, hint)?,
        })
    }

    /// Read + parse a config file relative to the consumer's `CARGO_MANIFEST_DIR`
    /// (same convention as cassettes). Format inferred from the extension.
    ///
    /// # Errors
    /// Propagates I/O and parse failures.
    pub async fn from_path(rel: &str, manifest_dir: &str) -> Result<Self, ProximaError> {
        let path = std::path::Path::new(manifest_dir).join(rel);
        let raw = std::fs::read_to_string(&path).map_err(|error| {
            ProximaError::Config(format!("read config {}: {error}", path.display()))
        })?;
        let hint = path.extension().and_then(|extension| extension.to_str());
        Self::from_raw(&raw, hint).await
    }

    /// Dispatch: a `.toml`/`.json`/… path-looking string → `from_path`, else raw (sniff).
    ///
    /// # Errors
    /// Propagates the underlying loader error.
    pub async fn from_raw_or_path(spec: &str, manifest_dir: &str) -> Result<Self, ProximaError> {
        let looks_like_path = spec.ends_with(".toml")
            || spec.ends_with(".json")
            || spec.ends_with(".yaml")
            || spec.ends_with(".yml");
        if looks_like_path {
            Self::from_path(spec, manifest_dir).await
        } else {
            Self::from_raw(spec, None).await
        }
    }

    #[must_use]
    pub fn value(&self) -> &serde_json::Value {
        &self.value
    }

    /// Project the untyped config into a typed struct (test-side only).
    ///
    /// # Errors
    /// Returns `Config` on a deserialize mismatch.
    pub fn typed<T: serde::de::DeserializeOwned>(&self) -> Result<T, ProximaError> {
        serde_json::from_value(self.value.clone())
            .map_err(|error| ProximaError::Config(format!("typed config projection: {error}")))
    }

    /// Deep-merge a patch (overlay): patch object keys override base recursively;
    /// non-objects replace.
    #[must_use]
    pub fn overlay(mut self, patch: serde_json::Value) -> Self {
        deep_merge(&mut self.value, patch);
        self
    }

    /// Deep-merge a patch given as raw text (any registered format, sniffed).
    ///
    /// # Panics
    /// Panics (test-failure path) if the registry can't build or the patch can't
    /// parse — surfaced through the harness's normal failure reporting.
    #[must_use]
    pub fn overlay_str(self, patch: &str) -> Self {
        let registry = crate::config_format::default_config_format_registry()
            .expect("proxima::test overlay: config-format registry");
        let value = registry
            .parse_with_hint(patch, None)
            .expect("proxima::test overlay: parse patch");
        self.overlay(value)
    }

    /// Realize the config as a live mounted `Handler`.
    ///
    /// # Errors
    /// Propagates loader/build failures.
    pub async fn into_pipe(self) -> Result<PipeHandle, ProximaError> {
        let context = crate::load::LoadContext::with_noop_telemetry()?;
        crate::load::load(crate::load::Spec::Inline(self.value), &context).await
    }
}

// ---------------------------------------------------------------------------
// live plane
// ---------------------------------------------------------------------------

/// A live, hot-swappable control plane as a fixture.
pub struct LivePlane {
    plane: DaemonControlPlane,
}

impl LivePlane {
    fn parse_spec(raw: &str) -> Result<serde_json::Value, ProximaError> {
        crate::config_format::default_config_format_registry()?.parse_with_hint(raw, None)
    }

    /// Build a plane hosting a single named pipe from a raw config spec.
    ///
    /// # Errors
    /// Propagates registry/parse/app-build failures.
    pub fn with_pipe(name: &str, spec_raw: &str) -> Result<Self, ProximaError> {
        let spec = Self::parse_spec(spec_raw)?;
        let app = App::new()?;
        let config = PipeConfig {
            name: name.to_string(),
            spec,
            requires: Vec::new(),
        };
        Ok(Self {
            plane: DaemonControlPlane::new(app, vec![config]),
        })
    }

    /// Start a configured pipe (walks its dependency graph).
    ///
    /// # Errors
    /// Propagates the control-plane start error.
    pub async fn start(&self, name: &str) -> Result<(), ProximaError> {
        ControlPlane::start(&self.plane, name)
            .await
            .map(|_status| ())
    }

    /// Mount `path` → the named pipe so the router routes to it.
    ///
    /// # Errors
    /// Propagates the mount error.
    pub async fn mount(&self, path: &str, name: &str) -> Result<(), ProximaError> {
        self.plane
            .mount(path, MountTarget::Named(name.to_string()))
            .await
    }

    /// Atomically hot-swap a pipe's spec (router reflects the new impl).
    ///
    /// # Errors
    /// Propagates the apply error.
    pub async fn hot_swap(&self, name: &str, spec_raw: &str) -> Result<(), ProximaError> {
        let spec = Self::parse_spec(spec_raw)?;
        ControlPlane::apply(&self.plane, name, spec)
            .await
            .map(|_status| ())
    }

    /// Drive a GET through the live router.
    ///
    /// # Errors
    /// Propagates request-build / call failures.
    pub async fn call(&self, path: &str) -> Result<Response<Bytes>, ProximaError> {
        let router = self.plane.router().await;
        let request = Request::builder().method("GET").path(path).build()?;
        SendPipe::call(&router, request).await
    }
}

fn deep_merge(base: &mut serde_json::Value, patch: serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(base_map), serde_json::Value::Object(patch_map)) => {
            for (key, patch_value) in patch_map {
                deep_merge(
                    base_map.entry(key).or_insert(serde_json::Value::Null),
                    patch_value,
                );
            }
        }
        (base_slot, patch_value) => *base_slot = patch_value,
    }
}

// ---------------------------------------------------------------------------
// recording tee (used by cassette_pipe in record mode)
// ---------------------------------------------------------------------------

struct RecordingTee {
    inner: PipeHandle,
    sink: DynRecordingSink,
    match_spec: MatchSpec,
    duplicates: DuplicatePolicy,
    // match key → response digest of the first recording for that key;
    // the record-time guard against silent last-wins collisions.
    seen: Arc<Mutex<HashMap<String, u64>>>,
}

fn header_map(headers: &crate::header_list::HeaderList) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect()
}

fn header_pairs(headers: &crate::header_list::HeaderList) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect()
}

impl SendPipe for RecordingTee {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        let sink = self.sink.clone();
        let match_key = match_key_from_request_with(&request, self.match_spec);
        let request_body = request.payload.clone();
        let duplicates = self.duplicates;
        let seen = self.seen.clone();
        async move {
            let interaction = InteractionId::new();
            let mut events = vec![RecordingEvent {
                id: interaction,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::Started {
                    ts: OffsetDateTime::UNIX_EPOCH,
                    pipe: "cassette".to_string(),
                    request: RequestHeader {
                        method: String::from_utf8_lossy(request.method.as_bytes()).into_owned(),
                        path: String::from_utf8_lossy(&request.path).into_owned(),
                        headers: header_map(&request.metadata),
                        query: header_map(&request.query),
                    },
                    meta: None,
                }),
            }];
            if !request_body.is_empty() {
                events.push(RecordingEvent {
                    id: interaction,
                    ts_ms: 0,
                    parent: None,
                    event: ProtocolEvent::Http(HttpEvent::RequestChunk {
                        data: request_body,
                        metadata: Default::default(),
                    }),
                });
            }
            events.push(RecordingEvent {
                id: interaction,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::RequestEnded),
            });

            let response = SendPipe::call(&inner, request).await?;
            let status = response.status;
            let pairs = header_pairs(&response.metadata);
            let body = response.collect_body().await?;
            events.push(RecordingEvent {
                id: interaction,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
                    status,
                    headers: pairs.clone(),
                }),
            });
            events.push(RecordingEvent {
                id: interaction,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                    data: body.clone(),
                    metadata: Default::default(),
                }),
            });
            events.push(RecordingEvent {
                id: interaction,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::Ended {
                    latency_ms: 0,
                    meta: RecordMeta::default(),
                }),
            });

            let response_digest = content_digest(&[&status.to_be_bytes(), &body]);
            let append = {
                let mut seen = seen.lock().expect("cassette seen-keys registry");
                match seen.get(&match_key) {
                    None => {
                        seen.insert(match_key.clone(), response_digest);
                        true
                    }
                    Some(&prior) if prior == response_digest => false,
                    Some(_) => match duplicates {
                        DuplicatePolicy::RejectDivergent => {
                            return Err(ProximaError::Record(format!(
                                "cassette match-key collision with divergent responses: \
                                 `{match_key}`; enable match_body, vary the request, or set \
                                 PROXIMA_CASSETTE_DUPLICATES=last-wins|first-wins"
                            )));
                        }
                        DuplicatePolicy::LastWins => {
                            seen.insert(match_key.clone(), response_digest);
                            true
                        }
                        DuplicatePolicy::FirstWins => false,
                    },
                }
            };
            if append {
                for event in events {
                    sink.append(event).await?;
                }
                sink.flush().await?;
            }

            let mut rebuilt = Response::new(status).with_body(body);
            for (name, value) in pairs {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}


// every test in this module drives cassette_pipe, which resolves a runtime
// via app::offline_runtime — matches offline_runtime's own gate exactly.
#[cfg(all(
    test,
    any(
        feature = "runtime-tokio",
        all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        )
    )
))]
mod tests {
    use super::*;
    use bytes::Bytes;

    struct SynthPipe;

    impl SendPipe for SynthPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async {
                Ok(Response::new(200)
                    .with_header("content-type", "application/json")
                    .with_body(Bytes::from_static(b"{\"ok\":1}")))
            }
        }
    }


    async fn drive_cassette_round_trip(path: std::path::PathBuf) {
        let record_ctx = proxima_test::TestCtx::__new_for_test(Some(proxima_test::CassetteCtx {
            path: path.clone(),
            mode: proxima_test::Mode::Record,
        }));
        let pipe = cassette_pipe(&record_ctx, SynthPipe)
            .await
            .expect("record pipe");
        let request = Request::builder()
            .method("GET")
            .path("/health")
            .build()
            .expect("request");
        let response = SendPipe::call(&pipe, request).await.expect("record call");
        assert_eq!(response.status, 200);
        assert_eq!(
            &response.collect_body().await.expect("record body")[..],
            b"{\"ok\":1}"
        );

        let replay_ctx = proxima_test::TestCtx::__new_for_test(Some(proxima_test::CassetteCtx {
            path,
            mode: proxima_test::Mode::Replay,
        }));
        let pipe = cassette_pipe(&replay_ctx, SynthPipe)
            .await
            .expect("replay pipe");
        let request = Request::builder()
            .method("GET")
            .path("/health")
            .build()
            .expect("request");
        let response = SendPipe::call(&pipe, request).await.expect("replay call");
        assert_eq!(response.status, 200);
        assert_eq!(
            &response.collect_body().await.expect("replay body")[..],
            b"{\"ok\":1}"
        );
    }

    #[test]
    fn cassette_records_then_replays_deterministically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("round_trip.jsonl");
        futures::executor::block_on(drive_cassette_round_trip(path));
    }

    // -- hardening: rerecord / duplicates / staleness / body matching --------

    struct ConstPipe(&'static [u8]);

    impl SendPipe for ConstPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let body = Bytes::from_static(self.0);
            async move { Ok(Response::new(200).with_body(body)) }
        }
    }


    /// Echoes the request payload back with a marker prefix, so replay
    /// correctness per-body is observable.
    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let mut body = b"echo:".to_vec();
            body.extend_from_slice(&request.payload);
            async move { Ok(Response::new(200).with_body(Bytes::from(body))) }
        }
    }


    /// Returns a different body on every call — the divergent-duplicate case.
    struct CountingPipe(std::sync::atomic::AtomicUsize);

    impl SendPipe for CountingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let count = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move { Ok(Response::new(200).with_body(Bytes::from(format!("resp-{count}")))) }
        }
    }


    fn ctx(path: &std::path::Path, mode: proxima_test::Mode) -> proxima_test::TestCtx {
        proxima_test::TestCtx::__new_for_test(Some(proxima_test::CassetteCtx {
            path: path.to_path_buf(),
            mode,
        }))
    }

    async fn drive(
        pipe: &PipeHandle,
        method: &str,
        path: &str,
        body: &'static [u8],
    ) -> Result<(u16, Bytes), ProximaError> {
        let request = Request::builder()
            .method(method)
            .path(path)
            .body(body)
            .build()?;
        let response = SendPipe::call(pipe, request).await?;
        let status = response.status;
        let collected = response.collect_body().await?;
        Ok((status, collected))
    }

    async fn record_once(path: &std::path::Path, body: &'static [u8]) {
        let record_ctx = ctx(path, proxima_test::Mode::Record);
        let pipe = cassette_pipe_with(&record_ctx, ConstPipe(body), CassetteConfig::default())
            .await
            .expect("record pipe");
        let (status, _body) = drive(&pipe, "GET", "/data", b"")
            .await
            .expect("record call");
        assert_eq!(status, 200);
    }

    #[test]
    fn rerecord_backup_preserves_prior_cassette() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("backup.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"one").await;
            let config = CassetteConfig::layered()
                .with_mode(ModePolicy::Record)
                .with_rerecord(RerecordPolicy::Backup)
                .build();
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with(&record_ctx, ConstPipe(b"two"), config)
                .await
                .expect("rerecord pipe");
            drive(&pipe, "GET", "/data", b"").await.expect("rerecord");

            let backup = path.with_extension("jsonl.bak");
            let backed_up = std::fs::read_to_string(&backup).expect("bak file");
            assert!(backed_up.contains("b25l"), "backup keeps `one` (b64)");
            let fresh = std::fs::read_to_string(&path).expect("cassette");
            assert!(fresh.contains("dHdv"), "new cassette holds `two` (b64)");
        });
    }

    #[test]
    fn rerecord_fail_refuses_to_destroy_existing_cassette() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("guarded.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"one").await;
            let config = CassetteConfig::layered()
                .with_mode(ModePolicy::Record)
                .with_rerecord(RerecordPolicy::Fail)
                .build();
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let outcome = cassette_pipe_with(&record_ctx, ConstPipe(b"two"), config).await;
            let Err(error) = outcome else {
                panic!("rerecord policy `fail` must refuse to overwrite");
            };
            assert!(format!("{error}").contains("rerecord policy is `fail`"));
        });
    }

    #[test]
    fn rerecord_hook_overrides_declarative_policy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("hooked.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"one").await;
            let config = CassetteConfig::layered()
                .with_mode(ModePolicy::Record)
                .with_rerecord(RerecordPolicy::Fail)
                .build();
            let hooks = CassetteHooks::new().with_on_rerecord(|_path| RerecordDecision::Truncate);
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with_hooks(&record_ctx, ConstPipe(b"two"), config, hooks)
                .await
                .expect("hook wins over fail policy");
            let (_status, body) = drive(&pipe, "GET", "/data", b"").await.expect("call");
            assert_eq!(&body[..], b"two");
        });
    }

    #[test]
    fn use_existing_hook_serves_recorded_data_instead_of_rerecording() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("protected.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"one").await;
            let before = std::fs::read_to_string(&path).expect("cassette");
            let config = CassetteConfig::layered()
                .with_mode(ModePolicy::Record)
                .build();
            let hooks =
                CassetteHooks::new().with_on_rerecord(|_path| RerecordDecision::UseExisting);
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with_hooks(&record_ctx, ConstPipe(b"two"), config, hooks)
                .await
                .expect("use-existing pipe");
            let (_status, body) = drive(&pipe, "GET", "/data", b"").await.expect("call");
            assert_eq!(&body[..], b"one", "served from the protected cassette");
            let after = std::fs::read_to_string(&path).expect("cassette");
            assert_eq!(after, before, "cassette bytes untouched");
        });
    }

    #[test]
    fn match_body_disambiguates_same_path_posts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bodies.jsonl");
        futures::executor::block_on(async {
            let config = CassetteConfig::layered().with_match_body(true).build();
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with(&record_ctx, EchoPipe, config.clone())
                .await
                .expect("record pipe");
            let (_s, alpha) = drive(&pipe, "POST", "/v1/chat", b"alpha").await.expect("a");
            let (_s, beta) = drive(&pipe, "POST", "/v1/chat", b"beta").await.expect("b");
            assert_eq!(&alpha[..], b"echo:alpha");
            assert_eq!(&beta[..], b"echo:beta");

            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let pipe = cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), config)
                .await
                .expect("replay pipe");
            let (_s, alpha) = drive(&pipe, "POST", "/v1/chat", b"alpha").await.expect("a");
            let (_s, beta) = drive(&pipe, "POST", "/v1/chat", b"beta").await.expect("b");
            assert_eq!(&alpha[..], b"echo:alpha");
            assert_eq!(&beta[..], b"echo:beta");
            let miss = drive(&pipe, "POST", "/v1/chat", b"gamma").await;
            assert!(matches!(miss, Err(ProximaError::ReplayMiss { .. })));
        });
    }

    #[test]
    fn divergent_duplicate_recordings_fail_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("divergent.jsonl");
        futures::executor::block_on(async {
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with(
                &record_ctx,
                CountingPipe(std::sync::atomic::AtomicUsize::new(0)),
                CassetteConfig::default(),
            )
            .await
            .expect("record pipe");
            drive(&pipe, "GET", "/poll", b"").await.expect("first");
            let outcome = drive(&pipe, "GET", "/poll", b"").await;
            let Err(error) = outcome else {
                panic!("divergent duplicate must fail");
            };
            assert!(format!("{error}").contains("divergent"));
        });
    }

    #[test]
    fn identical_repeats_are_deduplicated_not_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("polling.jsonl");
        futures::executor::block_on(async {
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe =
                cassette_pipe_with(&record_ctx, ConstPipe(b"same"), CassetteConfig::default())
                    .await
                    .expect("record pipe");
            drive(&pipe, "GET", "/poll", b"").await.expect("first");
            drive(&pipe, "GET", "/poll", b"")
                .await
                .expect("identical repeat");

            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let pipe =
                cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), CassetteConfig::default())
                    .await
                    .expect("replay pipe");
            let (_s, body) = drive(&pipe, "GET", "/poll", b"").await.expect("replay");
            assert_eq!(&body[..], b"same");
        });
    }

    #[test]
    fn last_wins_policy_restores_pre_hardening_behavior() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lastwins.jsonl");
        futures::executor::block_on(async {
            let config = CassetteConfig::layered()
                .with_duplicates(DuplicatePolicy::LastWins)
                .build();
            let record_ctx = ctx(&path, proxima_test::Mode::Record);
            let pipe = cassette_pipe_with(
                &record_ctx,
                CountingPipe(std::sync::atomic::AtomicUsize::new(0)),
                config,
            )
            .await
            .expect("record pipe");
            drive(&pipe, "GET", "/poll", b"").await.expect("first");
            drive(&pipe, "GET", "/poll", b"").await.expect("second");

            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let pipe =
                cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), CassetteConfig::default())
                    .await
                    .expect("replay pipe");
            let (_s, body) = drive(&pipe, "GET", "/poll", b"").await.expect("replay");
            assert_eq!(&body[..], b"resp-1", "last recording wins");
        });
    }

    /// Rewrite the provenance stamp so the cassette looks ancient.
    async fn age_cassette(path: &std::path::Path) {
        let text = std::fs::read_to_string(path).expect("read cassette");
        let aged: Vec<String> = text
            .lines()
            .map(|line| {
                if line.contains(CASSETTE_META_KIND) {
                    let mut value: serde_json::Value =
                        serde_json::from_str(line).expect("meta line json");
                    value["payload"]["recorded_at_ms"] = serde_json::Value::from(1_000u64);
                    serde_json::to_string(&value).expect("meta line rewrite")
                } else {
                    line.to_string()
                }
            })
            .collect();
        std::fs::write(path, aged.join("\n") + "\n").expect("write aged cassette");
    }

    #[test]
    fn stale_cassette_fails_replay_when_max_age_exceeded() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"old").await;
            age_cassette(&path).await;
            let config = CassetteConfig::layered()
                .with_max_age(std::time::Duration::from_secs(60))
                .build();
            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let outcome = cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), config).await;
            let Err(error) = outcome else {
                panic!("stale cassette must fail replay");
            };
            let rendered = format!("{error}");
            assert!(rendered.contains("stale cassette"));
            assert!(rendered.contains("re-record"));
        });
    }

    #[test]
    fn stale_cassette_warn_policy_proceeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale_warn.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"old").await;
            age_cassette(&path).await;
            let config = CassetteConfig::layered()
                .with_max_age(std::time::Duration::from_secs(60))
                .with_staleness(StalenessPolicy::Warn)
                .build();
            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let pipe = cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), config)
                .await
                .expect("warn proceeds");
            let (_s, body) = drive(&pipe, "GET", "/data", b"").await.expect("replay");
            assert_eq!(&body[..], b"old");
        });
    }

    #[test]
    fn stale_hook_grants_amnesty_over_fail_policy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("amnesty.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"old").await;
            age_cassette(&path).await;
            let config = CassetteConfig::layered()
                .with_max_age(std::time::Duration::from_secs(60))
                .build();
            let hooks = CassetteHooks::new().with_on_stale(|_path, _meta| StaleDecision::Proceed);
            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let pipe = cassette_pipe_with_hooks(&replay_ctx, ConstPipe(b"live"), config, hooks)
                .await
                .expect("hook amnesty");
            let (_s, body) = drive(&pipe, "GET", "/data", b"").await.expect("replay");
            assert_eq!(&body[..], b"old");
        });
    }

    #[test]
    fn cassette_without_provenance_is_stale_when_gate_enabled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("legacy.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"old").await;
            let text = std::fs::read_to_string(&path).expect("read");
            let stripped: Vec<&str> = text
                .lines()
                .filter(|line| !line.contains(CASSETTE_META_KIND))
                .collect();
            std::fs::write(&path, stripped.join("\n") + "\n").expect("strip meta");

            let config = CassetteConfig::layered()
                .with_max_age(std::time::Duration::from_secs(60))
                .build();
            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let outcome = cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), config).await;
            let Err(error) = outcome else {
                panic!("unknown age must fail the staleness gate");
            };
            assert!(format!("{error}").contains("age unknown"));
        });
    }

    #[test]
    fn unsupported_format_version_error_carries_rerecord_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("old_format.jsonl");
        futures::executor::block_on(async {
            record_once(&path, b"old").await;
            let text = std::fs::read_to_string(&path).expect("read");
            std::fs::write(&path, text.replace("\"v\":3", "\"v\":2")).expect("downgrade version");

            let replay_ctx = ctx(&path, proxima_test::Mode::Replay);
            let outcome =
                cassette_pipe_with(&replay_ctx, ConstPipe(b"live"), CassetteConfig::default())
                    .await;
            let Err(error) = outcome else {
                panic!("old format version must fail to load");
            };
            let rendered = format!("{error}");
            assert!(rendered.contains("unsupported recording version"));
            assert!(rendered.contains("re-record"));
        });
    }
}
