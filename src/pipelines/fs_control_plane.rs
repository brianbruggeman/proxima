use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use parking_lot::Mutex;
use proxima_primitives::sync::broadcast;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::io::AsyncWriteExt;
use tracing::error;

use crate::error::ProximaError;
use crate::pipelines::control_plane::{
    EventFilter, EventStream, ListFilter, PipelineControlPlane, PipelineRecord, PipelineStatus,
    PipelineSubmission, PipelineSummary,
};
use crate::pipelines::executor::PipelineExecutor;
use crate::pipelines::explain::{ExplainStep, explain_stage};
use crate::pipelines::replay::replay_pipeline_with_id;
use crate::pipelines::spec::{PipelineSpec, StageSpec};
use crate::recording::event::{
    InteractionId, PipelineEvent, PipelineOutcome, ProtocolEvent, RecordingEvent,
};
use crate::recording::jsonl::JsonlSource;
use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};
use crate::recording::source::RecordingSource;
use crate::recording::{AppendLog, DeferredRuntime, EventTap, FormatKind};
use futures::StreamExt;
use proxima_primitives::pipe::SendPipe;

/// Filesystem-backed `PipelineControlPlane`. Persists each submission
/// under `<root>/<pipeline_id>/` with the following layout:
///
/// ```text
/// <root>/
///   index.jsonl                  -- append-only summary log (last entry per id wins)
///   <pipeline_id>/
///     spec.toml                  -- submitted spec
///     recording.jsonl            -- full event stream (universal v3 envelope)
///     workspace/<stage_name>/    -- each stage's CWD; artifacts live here
/// ```
///
/// `open(root)` loads `index.jsonl` into the in-memory resolve/list
/// table so a fresh daemon process sees pipelines submitted in earlier
/// SSH sessions. Events for past pipelines stay on disk; `inspect`
/// returns the summary; `subscribe_events` for a terminal pipeline
/// streams historicals straight from `recording.jsonl`.
pub struct FsPipelineControlPlane {
    root: PathBuf,
    state: Arc<Mutex<FsState>>,
    /// The sink passed to spawned executors. Internally this is an
    /// `EventTap` wrapping the `FsRoutingSink` — events flow
    /// executor → routing (writes per-pipeline AppendLog + updates
    /// in-memory state) → tap (fanout to subscribers).
    sink: DynRecordingSink,
    /// Direct handle to the routing sink so submit can register a fresh
    /// per-pipeline AppendLog without going through the dyn wrapper.
    routing: Arc<FsRoutingSink>,
    broadcast: Arc<EventTap>,
}

struct FsState {
    pipelines: HashMap<InteractionId, FsPipelineEntry>,
    name_to_id: HashMap<String, InteractionId>,
}

struct FsPipelineEntry {
    summary: PipelineSummary,
    /// `None` for entries loaded from `index.jsonl` at startup. `Some` once
    /// the spec is in memory (after submit, or after a lazy `inspect`).
    spec: Option<PipelineSpec>,
    recording_path: PathBuf,
    /// Best-effort live event count. Authoritative count is in the
    /// recording file; this is the snapshot of what the routing sink
    /// has seen since the daemon started.
    live_events_recorded: usize,
}

/// One line per pipeline-status snapshot in `index.jsonl`. The last
/// line for a given `id` is authoritative.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    id: InteractionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    spec_hash_hex: String,
    started_at: OffsetDateTime,
    status: PipelineStatus,
}

impl FsPipelineControlPlane {
    pub const BROADCAST_CAPACITY: usize = 1024;

    /// Open or create a control plane backed by `root`. Loads existing
    /// `index.jsonl` so previously-submitted pipelines are visible
    /// immediately via `list` / `resolve` / `inspect`.
    pub async fn open(
        root: impl Into<PathBuf>,
        spigot: DeferredRuntime,
    ) -> Result<Self, ProximaError> {
        let root = root.into();
        tokio::fs::create_dir_all(&root).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "create proximad root {root:?}: {err}"
            )))
        })?;
        let state = Arc::new(Mutex::new(FsState {
            pipelines: HashMap::new(),
            name_to_id: HashMap::new(),
        }));
        let routing = Arc::new(FsRoutingSink {
            state: state.clone(),
            per_pipeline_sinks: Arc::new(Mutex::new(HashMap::new())),
            spigot: spigot.clone(),
            root: root.clone(),
        });
        let routing_dyn: DynRecordingSink = routing.clone();
        let broadcast = Arc::new(EventTap::new(routing_dyn, Self::BROADCAST_CAPACITY));
        let sink: DynRecordingSink = broadcast.clone();
        let plane = Self {
            root,
            state,
            sink,
            routing,
            broadcast,
        };
        plane.load_index().await?;
        Ok(plane)
    }

    async fn load_index(&self) -> Result<(), ProximaError> {
        let index_path = self.root.join("index.jsonl");
        if !tokio::fs::try_exists(&index_path).await.unwrap_or(false) {
            return Ok(());
        }
        let contents = tokio::fs::read_to_string(&index_path)
            .await
            .map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!(
                    "read index {index_path:?}: {err}"
                )))
            })?;
        let mut latest: HashMap<InteractionId, IndexEntry> = HashMap::new();
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: IndexEntry = serde_json::from_str(trimmed)
                .map_err(|err| ProximaError::Decode(format!("index.jsonl line: {err}")))?;
            latest.insert(entry.id, entry);
        }
        let mut guard = self.state.lock();
        for (id, entry) in latest {
            if let Some(name) = &entry.name {
                guard.name_to_id.insert(name.clone(), id);
            }
            let recording_path = self.root.join(id.to_string()).join("recording.jsonl");
            guard.pipelines.insert(
                id,
                FsPipelineEntry {
                    summary: PipelineSummary {
                        id,
                        name: entry.name,
                        spec_hash_hex: entry.spec_hash_hex,
                        started_at: entry.started_at,
                        status: entry.status,
                    },
                    spec: None,
                    recording_path,
                    live_events_recorded: 0,
                },
            );
        }
        Ok(())
    }

    async fn append_index(&self, entry: &IndexEntry) -> Result<(), ProximaError> {
        let index_path = self.root.join("index.jsonl");
        let line = serde_json::to_string(entry)
            .map_err(|err| ProximaError::Encode(format!("serialize index entry: {err}")))?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)
            .await
            .map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!(
                    "open index for append {index_path:?}: {err}"
                )))
            })?;
        file.write_all(line.as_bytes()).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("append index: {err}")))
        })?;
        file.write_all(b"\n").await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "append index newline: {err}"
            )))
        })?;
        file.flush().await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("flush index: {err}")))
        })?;
        Ok(())
    }

    async fn load_spec(&self, id: InteractionId) -> Result<PipelineSpec, ProximaError> {
        let path = self.root.join(id.to_string()).join("spec.toml");
        let text = tokio::fs::read_to_string(&path).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("read spec {path:?}: {err}")))
        })?;
        toml::from_str(&text)
            .map_err(|err| ProximaError::Decode(format!("parse spec {path:?}: {err}")))
    }

    async fn read_historical_events(
        &self,
        recording_path: &Path,
    ) -> Result<Vec<RecordingEvent>, ProximaError> {
        if !tokio::fs::try_exists(recording_path).await.unwrap_or(false) {
            return Ok(Vec::new());
        }
        let runtime = self.routing.spigot.get().cloned().ok_or_else(|| {
            ProximaError::Config("fs control plane: recording spigot not armed".into())
        })?;
        let source = JsonlSource::new(recording_path, runtime);
        let mut stream = source.events();
        let mut output = Vec::new();
        while let Some(item) = stream.next().await {
            output.push(item?);
        }
        Ok(output)
    }

    fn workspace_root(&self) -> PathBuf {
        self.root.clone()
    }
}

impl PipelineControlPlane for FsPipelineControlPlane {
    fn submit<'lifetime>(
        &'lifetime self,
        spec: PipelineSpec,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineSubmission, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            spec.validate()?;
            let pipeline_id = InteractionId::new();
            let pipeline_dir = self.root.join(pipeline_id.to_string());
            tokio::fs::create_dir_all(&pipeline_dir)
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "create pipeline dir {pipeline_dir:?}: {err}"
                    )))
                })?;
            let workspace = pipeline_dir.join("workspace");
            tokio::fs::create_dir_all(&workspace).await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!(
                    "create workspace {workspace:?}: {err}"
                )))
            })?;
            let spec_path = pipeline_dir.join("spec.toml");
            let spec_toml = toml::to_string_pretty(&spec)
                .map_err(|err| ProximaError::Encode(format!("serialize spec to toml: {err}")))?;
            tokio::fs::write(&spec_path, spec_toml)
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "write spec {spec_path:?}: {err}"
                    )))
                })?;
            let recording_path = pipeline_dir.join("recording.jsonl");
            let runtime = self.routing.spigot.get().cloned().ok_or_else(|| {
                ProximaError::Config("fs control plane: recording spigot not armed".into())
            })?;
            let pipeline_sink: Arc<AppendLog> = Arc::new(AppendLog::open(
                &recording_path,
                FormatKind::Json.codec()?,
                runtime,
            )?);
            let spec_hash_hex = hex_encode(&spec.spec_hash());
            let started_at = OffsetDateTime::now_utc();
            let summary = PipelineSummary {
                id: pipeline_id,
                name: spec.name.clone(),
                spec_hash_hex: spec_hash_hex.clone(),
                started_at,
                status: PipelineStatus::Running,
            };
            // register per-pipeline sink BEFORE spawning executor so the
            // routing sink can find it on the first event.
            {
                let mut per = self.routing.per_pipeline_sinks.lock();
                per.insert(pipeline_id, pipeline_sink);
            }
            {
                let mut guard = self.state.lock();
                if let Some(name) = &spec.name {
                    guard.name_to_id.insert(name.clone(), pipeline_id);
                }
                guard.pipelines.insert(
                    pipeline_id,
                    FsPipelineEntry {
                        summary: summary.clone(),
                        spec: Some(spec.clone()),
                        recording_path,
                        live_events_recorded: 0,
                    },
                );
            }
            self.append_index(&IndexEntry {
                id: pipeline_id,
                name: spec.name.clone(),
                spec_hash_hex,
                started_at,
                status: PipelineStatus::Running,
            })
            .await?;
            let executor =
                PipelineExecutor::new(self.sink.clone()).with_workspace_root(self.workspace_root());
            tokio::spawn(async move {
                if let Err(error) = executor.run_with_id(spec, pipeline_id).await {
                    error!(?error, %pipeline_id, "pipeline run errored");
                }
            });
            // best-effort: emit a status-update entry to index.jsonl when the
            // routing sink sees PipelineEvent::Ended (handled inside FsRoutingSink
            // by appending through this same plane's append_index path — see
            // FsRoutingSink::record_terminal).
            Ok(PipelineSubmission { pipeline_id })
        })
    }

    fn list<'lifetime>(
        &'lifetime self,
        filter: ListFilter,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipelineSummary>, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let guard = self.state.lock();
            let mut output: Vec<PipelineSummary> = guard
                .pipelines
                .values()
                .filter(|entry| match &filter.name {
                    Some(name) => entry.summary.name.as_deref() == Some(name.as_str()),
                    None => true,
                })
                .filter(|entry| match &filter.spec_hash_hex {
                    Some(hex) => &entry.summary.spec_hash_hex == hex,
                    None => true,
                })
                .map(|entry| entry.summary.clone())
                .collect();
            output.sort_by_key(|summary| std::cmp::Reverse(summary.started_at));
            Ok(output)
        })
    }

    fn resolve<'lifetime>(
        &'lifetime self,
        query: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<InteractionId, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            let guard = self.state.lock();
            if let Ok(ulid) = query.parse::<ulid::Ulid>() {
                let id = InteractionId::from_ulid(ulid);
                if guard.pipelines.contains_key(&id) {
                    return Ok(id);
                }
            }
            if let Some(id) = guard.name_to_id.get(query) {
                return Ok(*id);
            }
            let query_upper = query.to_uppercase();
            let mut prefix_matches: Vec<InteractionId> = guard
                .pipelines
                .keys()
                .filter(|id| id.to_string().starts_with(&query_upper))
                .copied()
                .collect();
            prefix_matches.extend(
                guard
                    .name_to_id
                    .iter()
                    .filter(|(name, _)| name.starts_with(query))
                    .map(|(_, id)| *id),
            );
            prefix_matches.sort();
            prefix_matches.dedup();
            match prefix_matches.len() {
                0 => Err(ProximaError::NotFound(format!(
                    "no pipeline matches `{query}`"
                ))),
                1 => Ok(prefix_matches[0]),
                _ => Err(ProximaError::Config(format!(
                    "ambiguous query `{query}` matches {} pipelines",
                    prefix_matches.len()
                ))),
            }
        })
    }

    fn inspect<'lifetime>(
        &'lifetime self,
        id: InteractionId,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineRecord, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let (summary, spec, event_count) = {
                let guard = self.state.lock();
                let entry = guard
                    .pipelines
                    .get(&id)
                    .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
                (
                    entry.summary.clone(),
                    entry.spec.clone(),
                    entry.live_events_recorded,
                )
            };
            let spec = match spec {
                Some(spec) => spec,
                None => self.load_spec(id).await?,
            };
            Ok(PipelineRecord {
                summary,
                spec,
                event_count,
            })
        })
    }

    fn subscribe_events<'lifetime>(
        &'lifetime self,
        filter: EventFilter,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            match filter {
                EventFilter::Pipeline(pipeline_id) => {
                    let (recording_path, terminal, mut live_rx) = {
                        let guard = self.state.lock();
                        let entry = guard.pipelines.get(&pipeline_id).ok_or_else(|| {
                            ProximaError::NotFound(format!("unknown pipeline `{pipeline_id}`"))
                        })?;
                        (
                            entry.recording_path.clone(),
                            !matches!(entry.summary.status, PipelineStatus::Running),
                            self.broadcast.subscribe(),
                        )
                    };
                    let historicals = self.read_historical_events(&recording_path).await?;
                    let owner_filter = pipeline_id;
                    let stream = stream! {
                        for event in historicals {
                            yield event;
                        }
                        if terminal {
                            return;
                        }
                        loop {
                            match live_rx.recv().await {
                                Ok(event) => {
                                    let owner = event_owner(&event);
                                    if owner == Some(owner_filter) {
                                        let is_terminal_event = matches!(
                                            &event.event,
                                            ProtocolEvent::Pipeline(PipelineEvent::Ended { .. })
                                        ) && event.id == owner_filter;
                                        yield event;
                                        if is_terminal_event {
                                            break;
                                        }
                                    }
                                }
                                Err(broadcast::error::RecvError::Closed) => break,
                                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                            }
                        }
                    };
                    Ok(Box::pin(stream) as EventStream)
                }
                EventFilter::AllEvents => {
                    let mut live_rx = self.broadcast.subscribe();
                    let stream = stream! {
                        loop {
                            match live_rx.recv().await {
                                Ok(event) => yield event,
                                Err(broadcast::error::RecvError::Closed) => break,
                                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                            }
                        }
                    };
                    Ok(Box::pin(stream) as EventStream)
                }
            }
        })
    }

    fn explain<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        stage: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ExplainStep>, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            // try the in-memory cache first; fall back to spec.toml on disk
            let cached = {
                let guard = self.state.lock();
                let entry = guard
                    .pipelines
                    .get(&id)
                    .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
                entry.spec.clone()
            };
            let spec = match cached {
                Some(spec) => spec,
                None => self.load_spec(id).await?,
            };
            explain_stage(&spec, stage)
        })
    }

    fn replay<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        substitutes: BTreeMap<String, StageSpec>,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineSubmission, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let (recording_path, cached_spec) = {
                let guard = self.state.lock();
                let entry = guard
                    .pipelines
                    .get(&id)
                    .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
                (entry.recording_path.clone(), entry.spec.clone())
            };
            let spec = match cached_spec {
                Some(spec) => spec,
                None => self.load_spec(id).await?,
            };
            let events = self.read_historical_events(&recording_path).await?;

            let new_pipeline_id = InteractionId::new();
            let pipeline_dir = self.root.join(new_pipeline_id.to_string());
            tokio::fs::create_dir_all(&pipeline_dir)
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "create replay dir {pipeline_dir:?}: {err}"
                    )))
                })?;
            tokio::fs::create_dir_all(pipeline_dir.join("workspace"))
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "create replay workspace: {err}"
                    )))
                })?;
            let spec_path = pipeline_dir.join("spec.toml");
            let spec_toml = toml::to_string_pretty(&spec)
                .map_err(|err| ProximaError::Encode(format!("serialize replay spec: {err}")))?;
            tokio::fs::write(&spec_path, spec_toml)
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "write replay spec {spec_path:?}: {err}"
                    )))
                })?;
            let new_recording_path = pipeline_dir.join("recording.jsonl");
            let runtime = self.routing.spigot.get().cloned().ok_or_else(|| {
                ProximaError::Config("fs control plane: recording spigot not armed".into())
            })?;
            let pipeline_sink: Arc<AppendLog> = Arc::new(AppendLog::open(
                &new_recording_path,
                FormatKind::Json.codec()?,
                runtime,
            )?);
            let spec_hash_hex = hex_encode(&spec.spec_hash());
            let started_at = OffsetDateTime::now_utc();
            let summary = PipelineSummary {
                id: new_pipeline_id,
                name: spec.name.clone(),
                spec_hash_hex: spec_hash_hex.clone(),
                started_at,
                status: PipelineStatus::Running,
            };
            {
                let mut per = self.routing.per_pipeline_sinks.lock();
                per.insert(new_pipeline_id, pipeline_sink);
            }
            {
                let mut guard = self.state.lock();
                guard.pipelines.insert(
                    new_pipeline_id,
                    FsPipelineEntry {
                        summary: summary.clone(),
                        spec: Some(spec.clone()),
                        recording_path: new_recording_path,
                        live_events_recorded: 0,
                    },
                );
            }
            self.append_index(&IndexEntry {
                id: new_pipeline_id,
                name: spec.name.clone(),
                spec_hash_hex,
                started_at,
                status: PipelineStatus::Running,
            })
            .await?;
            let sink = self.sink.clone();
            tokio::spawn(async move {
                if let Err(error) =
                    replay_pipeline_with_id(&spec, &events, &substitutes, sink, new_pipeline_id)
                        .await
                {
                    error!(?error, %new_pipeline_id, "fs replay errored");
                }
            });
            Ok(PipelineSubmission {
                pipeline_id: new_pipeline_id,
            })
        })
    }

    fn artifact_path<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        stage: &'lifetime str,
        relative: &'lifetime std::path::Path,
    ) -> Pin<Box<dyn Future<Output = Result<std::path::PathBuf, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            // ensure the pipeline exists at all — otherwise the next layer
            // would happily look up a non-existent dir and surface an io error.
            {
                let guard = self.state.lock();
                if !guard.pipelines.contains_key(&id) {
                    return Err(ProximaError::NotFound(format!("unknown pipeline `{id}`")));
                }
            }
            let stage_workspace = self.root.join(id.to_string()).join("workspace").join(stage);
            let candidate = stage_workspace.join(relative);
            // canonicalize both before the prefix check so symlinks and `..`
            // are resolved to their target paths. If candidate doesn't exist,
            // canonicalize errors — surface that as NotFound.
            let canon_candidate = tokio::fs::canonicalize(&candidate).await.map_err(|err| {
                ProximaError::NotFound(format!("artifact {candidate:?} not found: {err}"))
            })?;
            let canon_root = tokio::fs::canonicalize(&stage_workspace)
                .await
                .map_err(|err| {
                    ProximaError::NotFound(format!(
                        "stage workspace {stage_workspace:?} not found: {err}"
                    ))
                })?;
            if !canon_candidate.starts_with(&canon_root) {
                return Err(ProximaError::Config(format!(
                    "artifact path escapes stage workspace: {canon_candidate:?} not under {canon_root:?}"
                )));
            }
            Ok(canon_candidate)
        })
    }
}

struct FsRoutingSink {
    state: Arc<Mutex<FsState>>,
    per_pipeline_sinks: Arc<Mutex<HashMap<InteractionId, Arc<AppendLog>>>>,
    spigot: DeferredRuntime,
    root: PathBuf,
}

impl RecordingSink for FsRoutingSink {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        Box::pin(async move {
            let owner = event_owner(&event);
            let Some(pipeline_id) = owner else {
                return Ok(());
            };
            // write to the per-pipeline AppendLog first so on-disk recording
            // never lags the in-memory state.
            let sink = {
                let guard = self.per_pipeline_sinks.lock();
                guard.get(&pipeline_id).cloned()
            };
            if let Some(sink) = sink {
                SendPipe::call(sink.as_ref(), vec![event.clone()])
                    .await
                    .map(|_ack| ())?;
            }
            // update in-memory state: bump event count + handle status transition
            // on PipelineEvent::Ended.
            let mut status_change: Option<(PipelineStatus, PipelineSummary)> = None;
            {
                let mut guard = self.state.lock();
                if let Some(entry) = guard.pipelines.get_mut(&pipeline_id) {
                    entry.live_events_recorded += 1;
                    if let ProtocolEvent::Pipeline(PipelineEvent::Ended { outcome }) = &event.event
                    {
                        entry.summary.status = match outcome {
                            PipelineOutcome::Completed => PipelineStatus::Completed,
                            PipelineOutcome::Failed { .. } => PipelineStatus::Failed,
                            PipelineOutcome::Cancelled => PipelineStatus::Cancelled,
                        };
                        status_change = Some((entry.summary.status, entry.summary.clone()));
                    }
                }
            }
            if let Some((_, summary)) = status_change {
                // mirror the index update FsPipelineControlPlane::append_index
                // does on submit. duplicated inline because the routing sink
                // doesn't hold a back-reference to the plane.
                let _ = append_status_line(&self.root, &summary).await;
            }
            Ok(())
        })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        Box::pin(async {
            let sinks: Vec<Arc<AppendLog>> = {
                let guard = self.per_pipeline_sinks.lock();
                guard.values().cloned().collect()
            };
            for sink in sinks {
                sink.flush().await?;
            }
            Ok(())
        })
    }
}

async fn append_status_line(root: &Path, summary: &PipelineSummary) -> Result<(), ProximaError> {
    let index_path = root.join("index.jsonl");
    let entry = IndexEntry {
        id: summary.id,
        name: summary.name.clone(),
        spec_hash_hex: summary.spec_hash_hex.clone(),
        started_at: summary.started_at,
        status: summary.status,
    };
    let line = serde_json::to_string(&entry)
        .map_err(|err| ProximaError::Encode(format!("serialize index entry: {err}")))?;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index_path)
        .await
        .map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "open index for append {index_path:?}: {err}"
            )))
        })?;
    file.write_all(line.as_bytes())
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("append index: {err}"))))?;
    file.write_all(b"\n").await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "append index newline: {err}"
        )))
    })?;
    Ok(())
}

fn event_owner(event: &RecordingEvent) -> Option<InteractionId> {
    match &event.event {
        ProtocolEvent::Pipeline(_) => Some(event.id),
        _ => event.parent,
    }
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

#[cfg(all(
    test,
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::spec::StageSpec;
    use futures::StreamExt;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    fn armed_spigot() -> crate::recording::DeferredRuntime {
        let spigot = crate::recording::deferred_runtime();
        spigot
            .set(
                std::sync::Arc::new(crate::runtime::PrimeRuntime::new(1).expect("prime runtime"))
                    as std::sync::Arc<dyn crate::runtime::Runtime>,
            )
            .ok();
        spigot
    }

    fn shell_stage(name: &str, script: &str, deps: &[&str]) -> StageSpec {
        let (cmd, flag) = if cfg!(windows) {
            ("cmd", "/c")
        } else {
            ("/bin/sh", "-c")
        };
        StageSpec {
            name: name.into(),
            command: cmd.into(),
            args: vec![flag.into(), script.into()],
            env: BTreeMap::new(),
            cwd: None,
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
        }
    }

    async fn wait_for_terminal(
        plane: &FsPipelineControlPlane,
        id: InteractionId,
    ) -> PipelineStatus {
        // poll the in-memory status (a cheap mutex read that cannot miss a
        // terminal transition, unlike a one-shot broadcast event a lagging
        // receiver may drop) and idle between polls by awaiting the next event
        // with a short cap. the idle is load-bearing: a busy `yield_now` loop
        // never lets a single-threaded runtime go idle, so its reactor never
        // observes the pipeline subprocess's exit and this hangs; the cap
        // re-checks state even if the terminal event was dropped by lag.
        let mut live_events = plane.broadcast.subscribe();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            loop {
                let status = {
                    let guard = plane.state.lock();
                    guard.pipelines.get(&id).map(|entry| entry.summary.status)
                };
                if let Some(status) = status
                    && !matches!(status, PipelineStatus::Running)
                {
                    return status;
                }
                let _ =
                    tokio::time::timeout(std::time::Duration::from_millis(25), live_events.recv())
                        .await;
            }
        })
        .await
        .expect("pipeline did not reach a terminal state within 30s")
    }

    #[proxima::test]
    async fn submit_persists_spec_and_recording_to_disk() {
        let dir = tempdir().expect("tempdir");
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("open");
        let spec = PipelineSpec {
            name: Some("persisted".into()),
            stages: vec![shell_stage("only", "echo hi; exit 0", &[])],
        };
        let submission = plane.submit(spec).await.expect("submit");
        let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
        let pipeline_dir = dir.path().join(submission.pipeline_id.to_string());
        assert!(pipeline_dir.join("spec.toml").exists(), "spec.toml written");
        assert!(
            pipeline_dir.join("recording.jsonl").exists(),
            "recording.jsonl written"
        );
        assert!(
            pipeline_dir.join("workspace").exists(),
            "workspace dir created"
        );
        assert!(
            dir.path().join("index.jsonl").exists(),
            "index.jsonl written"
        );
    }

    #[proxima::test]
    async fn reopen_loads_existing_pipelines_from_index() {
        let dir = tempdir().expect("tempdir");
        let submission_id = {
            let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
                .await
                .expect("open");
            let submission = plane
                .submit(PipelineSpec {
                    name: Some("survives-restart".into()),
                    stages: vec![shell_stage("a", "exit 0", &[])],
                })
                .await
                .expect("submit");
            let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
            submission.pipeline_id
        };
        // simulate restart: drop the first plane, open a fresh one on the same root.
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("reopen");
        let all = plane.list(ListFilter::default()).await.expect("list");
        assert_eq!(all.len(), 1, "reopened plane sees the prior submission");
        assert_eq!(all[0].id, submission_id);
        let resolved = plane
            .resolve("survives-restart")
            .await
            .expect("resolve by name");
        assert_eq!(resolved, submission_id);
    }

    #[proxima::test]
    async fn inspect_lazy_loads_spec_for_historical_pipeline() {
        let dir = tempdir().expect("tempdir");
        let submission_id = {
            let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
                .await
                .expect("open");
            let submission = plane
                .submit(PipelineSpec {
                    name: Some("inspect-after-restart".into()),
                    stages: vec![shell_stage("a", "exit 0", &[])],
                })
                .await
                .expect("submit");
            let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
            submission.pipeline_id
        };
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("reopen");
        let record = plane.inspect(submission_id).await.expect("inspect");
        assert_eq!(record.spec.name.as_deref(), Some("inspect-after-restart"));
        assert_eq!(record.spec.stages.len(), 1);
    }

    #[proxima::test]
    async fn subscribe_terminal_yields_historicals_from_disk() {
        let dir = tempdir().expect("tempdir");
        let submission_id = {
            let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
                .await
                .expect("open");
            let submission = plane
                .submit(PipelineSpec {
                    name: Some("subscribe-historicals".into()),
                    stages: vec![shell_stage("a", "echo x; exit 0", &[])],
                })
                .await
                .expect("submit");
            let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
            submission.pipeline_id
        };
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("reopen");
        let mut stream = plane
            .subscribe_events(EventFilter::Pipeline(submission_id))
            .await
            .expect("subscribe");
        let mut events: Vec<RecordingEvent> = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        assert!(
            !events.is_empty(),
            "historical recording must be readable across restarts"
        );
    }

    #[proxima::test]
    async fn artifact_path_resolves_under_stage_workspace() {
        let dir = tempdir().expect("tempdir");
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("open");
        let spec = PipelineSpec {
            name: Some("artifact-target".into()),
            stages: vec![shell_stage(
                "produce",
                "echo report > report.txt; exit 0",
                &[],
            )],
        };
        let submission = plane.submit(spec).await.expect("submit");
        let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
        // give the post-exit drain a moment in case the file is still
        // buffered
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let resolved = plane
            .artifact_path(
                submission.pipeline_id,
                "produce",
                std::path::Path::new("report.txt"),
            )
            .await
            .expect("artifact_path");
        let contents = tokio::fs::read_to_string(&resolved)
            .await
            .expect("read artifact");
        assert!(
            contents.contains("report"),
            "artifact contents: {contents:?}"
        );
    }

    #[proxima::test]
    async fn artifact_path_rejects_traversal_above_workspace() {
        let dir = tempdir().expect("tempdir");
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("open");
        let spec = PipelineSpec {
            name: Some("traversal-target".into()),
            stages: vec![shell_stage("produce", "echo a > a.txt; exit 0", &[])],
        };
        let submission = plane.submit(spec).await.expect("submit");
        let _ = wait_for_terminal(&plane, submission.pipeline_id).await;
        // craft a traversal-up path that points at something outside the
        // stage workspace (the pipeline's spec.toml lives one level up).
        let outcome = plane
            .artifact_path(
                submission.pipeline_id,
                "produce",
                std::path::Path::new("../../spec.toml"),
            )
            .await;
        assert!(
            matches!(
                outcome,
                Err(ProximaError::Config(_) | ProximaError::NotFound(_))
            ),
            "traversal must be rejected; got {outcome:?}"
        );
    }

    #[proxima::test]
    async fn artifact_path_unknown_pipeline_returns_not_found() {
        let dir = tempdir().expect("tempdir");
        let plane = FsPipelineControlPlane::open(dir.path(), armed_spigot())
            .await
            .expect("open");
        let bogus = InteractionId::new();
        let outcome = plane
            .artifact_path(bogus, "produce", std::path::Path::new("report.txt"))
            .await;
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }
}
