use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use futures::Stream;
use parking_lot::Mutex;
use proxima_primitives::sync::broadcast;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tracing::error;

use crate::error::ProximaError;
use crate::pipelines::executor::PipelineExecutor;
use crate::pipelines::explain::{ExplainStep, explain_stage};
use crate::pipelines::replay::replay_pipeline_with_id;
use crate::pipelines::spec::{PipelineSpec, StageSpec};
use crate::recording::EventTap;
use crate::recording::event::{
    InteractionId, PipelineEvent, PipelineOutcome, ProtocolEvent, RecordingEvent,
};
use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};

/// Returned by `submit`. Carries the freshly allocated pipeline id so
/// callers can immediately tail/inspect/explain by id.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PipelineSubmission {
    pub pipeline_id: InteractionId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSummary {
    pub id: InteractionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub spec_hash_hex: String,
    pub started_at: OffsetDateTime,
    pub status: PipelineStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRecord {
    pub summary: PipelineSummary,
    pub spec: PipelineSpec,
    /// Events collected for this pipeline, in arrival order. Inspect
    /// returns a snapshot; live observation goes through `subscribe`.
    pub event_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_hash_hex: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum EventFilter {
    AllEvents,
    Pipeline(InteractionId),
}

pub type EventStream = Pin<Box<dyn Stream<Item = RecordingEvent> + Send>>;

/// Daemon control surface for pipeline submission and observation. One
/// trait for both transports (local UDS today, SSH-stdio later) so
/// every endpoint can share the same handler set. See `ControlPlane`
/// (rust/src/control_plane.rs) for the parallel pipe-lifecycle trait
/// this mirrors.
pub trait PipelineControlPlane: Send + Sync + 'static {
    fn submit<'lifetime>(
        &'lifetime self,
        spec: PipelineSpec,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineSubmission, ProximaError>> + Send + 'lifetime>>;

    fn list<'lifetime>(
        &'lifetime self,
        filter: ListFilter,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipelineSummary>, ProximaError>> + Send + 'lifetime>>;

    /// Resolve a query string to a canonical pipeline id. Order: exact
    /// ULID parse → exact name match → unique id prefix match.
    /// Ambiguous prefix matches return `Config` listing candidates.
    fn resolve<'lifetime>(
        &'lifetime self,
        query: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<InteractionId, ProximaError>> + Send + 'lifetime>>;

    fn inspect<'lifetime>(
        &'lifetime self,
        id: InteractionId,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineRecord, ProximaError>> + Send + 'lifetime>>;

    /// Stream of recording events. For running pipelines the stream
    /// yields historicals collected so far, then live events until the
    /// pipeline terminates. For terminal pipelines the stream yields
    /// historicals only and then ends. `EventFilter::AllEvents` skips
    /// the historical replay (no global history maintained) and yields
    /// only future events.
    fn subscribe_events<'lifetime>(
        &'lifetime self,
        filter: EventFilter,
    ) -> Pin<Box<dyn Future<Output = Result<EventStream, ProximaError>> + Send + 'lifetime>>;

    /// Trace a stage's `depends_on` closure. Returns the queried stage
    /// at depth 0 followed by its ancestors in spec-declaration order.
    /// Walks the spec rather than the recording — the recording's
    /// `parent` edge is pipeline→stage (hierarchical), while the
    /// inter-stage DAG lives in the spec.
    fn explain<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        stage: &'lifetime str,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<crate::pipelines::explain::ExplainStep>, ProximaError>>
                + Send
                + 'lifetime,
        >,
    >;

    /// Replay a recorded pipeline under a fresh pipeline id, optionally
    /// substituting selected stages with new `StageSpec`s. Non-substituted
    /// stages re-emit their recorded `ProcessEvent` stream (bit-identical
    /// observable behavior modulo `ts_ms`); substituted stages run live
    /// against fresh `ProcessUpstream`s. Returns the new pipeline's
    /// submission id so the caller can immediately tail/inspect/explain it.
    fn replay<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        substitutes: BTreeMap<String, crate::pipelines::spec::StageSpec>,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineSubmission, ProximaError>> + Send + 'lifetime>>;

    /// Resolve a stage-relative artifact path to an absolute disk path
    /// on the daemon host. Resolves `<workspace_root>/<id>/workspace/<stage>/<relative>`
    /// after canonicalization + prefix check (rejects `../` traversal).
    /// In-memory planes (no workspace on disk) return `NotFound`.
    fn artifact_path<'lifetime>(
        &'lifetime self,
        id: InteractionId,
        stage: &'lifetime str,
        relative: &'lifetime std::path::Path,
    ) -> Pin<Box<dyn Future<Output = Result<std::path::PathBuf, ProximaError>> + Send + 'lifetime>>
    {
        let _ = (id, stage, relative);
        Box::pin(async move {
            Err(ProximaError::NotFound(
                "this control plane does not expose on-disk artifacts".into(),
            ))
        })
    }
}

pub type DynPipelineControlPlane = Arc<dyn PipelineControlPlane>;

/// In-memory `PipelineControlPlane`: per-pipeline events live in a
/// `Vec<RecordingEvent>` guarded by a Mutex; live observation rides a
/// `broadcast::Sender<RecordingEvent>`. Used directly for tests and as
/// the default backing for the daemon while the FS store (G4-next) is
/// pending.
pub struct InMemoryPipelineControlPlane {
    state: Arc<Mutex<InnerState>>,
    sink: DynRecordingSink,
    broadcast: Arc<EventTap>,
}

struct InnerState {
    pipelines: HashMap<InteractionId, PipelineEntry>,
    name_to_id: HashMap<String, InteractionId>,
}

struct PipelineEntry {
    summary: PipelineSummary,
    spec: PipelineSpec,
    events: Vec<RecordingEvent>,
}

impl InMemoryPipelineControlPlane {
    /// Channel capacity for the live broadcast. Each subscriber gets an
    /// independent queue; slow subscribers see `RecvError::Lagged` and
    /// skip dropped events (the historical snapshot remains
    /// authoritative for catch-up).
    pub const BROADCAST_CAPACITY: usize = 1024;

    #[must_use]
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(InnerState {
            pipelines: HashMap::new(),
            name_to_id: HashMap::new(),
        }));
        let routing: DynRecordingSink = Arc::new(RoutingSink {
            state: state.clone(),
        });
        let broadcast = Arc::new(EventTap::new(routing, Self::BROADCAST_CAPACITY));
        let sink: DynRecordingSink = broadcast.clone();
        Self {
            state,
            sink,
            broadcast,
        }
    }
}

impl Default for InMemoryPipelineControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineControlPlane for InMemoryPipelineControlPlane {
    fn submit<'lifetime>(
        &'lifetime self,
        spec: PipelineSpec,
    ) -> Pin<Box<dyn Future<Output = Result<PipelineSubmission, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            spec.validate()?;
            let pipeline_id = InteractionId::new();
            let summary = PipelineSummary {
                id: pipeline_id,
                name: spec.name.clone(),
                spec_hash_hex: hex_encode(&spec.spec_hash()),
                started_at: OffsetDateTime::now_utc(),
                status: PipelineStatus::Running,
            };
            {
                let mut guard = self.state.lock();
                if let Some(name) = &spec.name {
                    guard.name_to_id.insert(name.clone(), pipeline_id);
                }
                guard.pipelines.insert(
                    pipeline_id,
                    PipelineEntry {
                        summary,
                        spec: spec.clone(),
                        events: Vec::new(),
                    },
                );
            }
            let executor = PipelineExecutor::new(self.sink.clone());
            tokio::spawn(async move {
                if let Err(error) = executor.run_with_id(spec, pipeline_id).await {
                    error!(?error, %pipeline_id, "pipeline run errored");
                }
            });
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
            // newest-first is what `list` users want by default.
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
            // 1: exact ULID parse
            if let Ok(ulid) = query.parse::<ulid::Ulid>() {
                let id = InteractionId::from_ulid(ulid);
                if guard.pipelines.contains_key(&id) {
                    return Ok(id);
                }
            }
            // 2: exact name match
            if let Some(id) = guard.name_to_id.get(query) {
                return Ok(*id);
            }
            // 3: unique id prefix (and name prefix)
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
                    "ambiguous query `{query}` matches {} pipelines: {:?}",
                    prefix_matches.len(),
                    prefix_matches
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
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
            let guard = self.state.lock();
            let entry = guard
                .pipelines
                .get(&id)
                .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
            Ok(PipelineRecord {
                summary: entry.summary.clone(),
                spec: entry.spec.clone(),
                event_count: entry.events.len(),
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
                    // capture historicals + a receiver under one lock so the
                    // historical replay and live subscription splice cleanly.
                    let (historicals, terminal, mut live_rx) = {
                        let guard = self.state.lock();
                        let entry = guard.pipelines.get(&pipeline_id).ok_or_else(|| {
                            ProximaError::NotFound(format!("unknown pipeline `{pipeline_id}`"))
                        })?;
                        let historicals = entry.events.clone();
                        let terminal = !matches!(entry.summary.status, PipelineStatus::Running);
                        let rx = self.broadcast.subscribe();
                        (historicals, terminal, rx)
                    };
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
                                        let is_terminal = matches!(
                                            &event.event,
                                            ProtocolEvent::Pipeline(PipelineEvent::Ended { .. })
                                        ) && event.id == owner_filter;
                                        yield event;
                                        if is_terminal {
                                            break;
                                        }
                                    }
                                }
                                Err(broadcast::error::RecvError::Closed) => break,
                                Err(broadcast::error::RecvError::Lagged(_)) => {
                                    // slow subscriber: skip dropped events;
                                    // catch up via inspect if needed.
                                    continue;
                                }
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
            let spec = {
                let guard = self.state.lock();
                let entry = guard
                    .pipelines
                    .get(&id)
                    .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
                entry.spec.clone()
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
            let (original_spec, original_events) = {
                let guard = self.state.lock();
                let entry = guard
                    .pipelines
                    .get(&id)
                    .ok_or_else(|| ProximaError::NotFound(format!("unknown pipeline `{id}`")))?;
                (entry.spec.clone(), entry.events.clone())
            };
            let new_pipeline_id = InteractionId::new();
            // pre-register so the routing sink picks up events flowing through the replay
            let summary = PipelineSummary {
                id: new_pipeline_id,
                name: original_spec.name.clone(),
                spec_hash_hex: hex_encode(&original_spec.spec_hash()),
                started_at: OffsetDateTime::now_utc(),
                status: PipelineStatus::Running,
            };
            {
                let mut guard = self.state.lock();
                guard.pipelines.insert(
                    new_pipeline_id,
                    PipelineEntry {
                        summary,
                        spec: original_spec.clone(),
                        events: Vec::new(),
                    },
                );
            }
            let sink = self.sink.clone();
            tokio::spawn(async move {
                if let Err(error) = replay_pipeline_with_id(
                    &original_spec,
                    &original_events,
                    &substitutes,
                    sink,
                    new_pipeline_id,
                )
                .await
                {
                    error!(?error, %new_pipeline_id, "replay errored");
                }
            });
            Ok(PipelineSubmission {
                pipeline_id: new_pipeline_id,
            })
        })
    }
}

/// Inner sink that routes each event to per-pipeline state.
/// `EventTap` wraps this and additionally forwards to the live
/// broadcast channel.
struct RoutingSink {
    state: Arc<Mutex<InnerState>>,
}

impl RecordingSink for RoutingSink {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        Box::pin(async move {
            let owner = event_owner(&event);
            if let Some(pipeline_id) = owner {
                let mut guard = self.state.lock();
                if let Some(entry) = guard.pipelines.get_mut(&pipeline_id) {
                    if let ProtocolEvent::Pipeline(PipelineEvent::Ended { outcome }) = &event.event
                    {
                        entry.summary.status = match outcome {
                            PipelineOutcome::Completed => PipelineStatus::Completed,
                            PipelineOutcome::Failed { .. } => PipelineStatus::Failed,
                            PipelineOutcome::Cancelled => PipelineStatus::Cancelled,
                        };
                    }
                    entry.events.push(event);
                }
            }
            Ok(())
        })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        Box::pin(async { Ok(()) })
    }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::spec::StageSpec;
    use futures::StreamExt;
    use std::collections::BTreeMap;

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
        plane: &InMemoryPipelineControlPlane,
        id: InteractionId,
    ) -> PipelineStatus {
        for _ in 0..500 {
            let record = plane.inspect(id).await.expect("inspect");
            if !matches!(record.summary.status, PipelineStatus::Running) {
                return record.summary.status;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("pipeline did not terminate within 5 seconds");
    }

    #[proxima::test]
    async fn submit_returns_id_then_completes_in_background() {
        let plane = InMemoryPipelineControlPlane::new();
        let spec = PipelineSpec {
            name: Some("alpha".into()),
            stages: vec![shell_stage("a", "exit 0", &[])],
        };
        let submission = plane.submit(spec).await.expect("submit");
        let status = wait_for_terminal(&plane, submission.pipeline_id).await;
        assert_eq!(status, PipelineStatus::Completed);
    }

    #[proxima::test]
    async fn resolve_by_id_name_and_prefix() {
        let plane = InMemoryPipelineControlPlane::new();
        let spec = PipelineSpec {
            name: Some("named-pipeline".into()),
            stages: vec![shell_stage("a", "exit 0", &[])],
        };
        let submission = plane.submit(spec).await.expect("submit");
        let id_string = submission.pipeline_id.to_string();
        // 1: exact id
        let resolved = plane.resolve(&id_string).await.expect("by id");
        assert_eq!(resolved, submission.pipeline_id);
        // 2: exact name
        let resolved = plane.resolve("named-pipeline").await.expect("by name");
        assert_eq!(resolved, submission.pipeline_id);
        // 3: id prefix (first 10 chars of the ULID)
        let prefix = &id_string[..10];
        let resolved = plane.resolve(prefix).await.expect("by prefix");
        assert_eq!(resolved, submission.pipeline_id);
    }

    #[proxima::test]
    async fn resolve_unknown_returns_not_found() {
        let plane = InMemoryPipelineControlPlane::new();
        let outcome = plane.resolve("does-not-exist").await;
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }

    #[proxima::test]
    async fn list_filters_by_name_and_returns_newest_first() {
        let plane = InMemoryPipelineControlPlane::new();
        let first = plane
            .submit(PipelineSpec {
                name: Some("alpha".into()),
                stages: vec![shell_stage("s", "exit 0", &[])],
            })
            .await
            .expect("submit");
        // tiny delay so started_at differs measurably
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let second = plane
            .submit(PipelineSpec {
                name: Some("beta".into()),
                stages: vec![shell_stage("s", "exit 0", &[])],
            })
            .await
            .expect("submit");
        let _ = wait_for_terminal(&plane, first.pipeline_id).await;
        let _ = wait_for_terminal(&plane, second.pipeline_id).await;
        let all = plane.list(ListFilter::default()).await.expect("list");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].name.as_deref(), Some("beta")); // newer first
        let only_alpha = plane
            .list(ListFilter {
                name: Some("alpha".into()),
                spec_hash_hex: None,
            })
            .await
            .expect("filter list");
        assert_eq!(only_alpha.len(), 1);
        assert_eq!(only_alpha[0].id, first.pipeline_id);
    }

    #[proxima::test]
    async fn subscribe_pipeline_filter_yields_only_that_pipelines_events() {
        let plane = InMemoryPipelineControlPlane::new();
        let alpha = plane
            .submit(PipelineSpec {
                name: Some("alpha".into()),
                stages: vec![shell_stage("s", "echo a; exit 0", &[])],
            })
            .await
            .expect("submit alpha");
        let _ = wait_for_terminal(&plane, alpha.pipeline_id).await;
        // subscribe AFTER terminal — should replay historicals and end
        let mut stream = plane
            .subscribe_events(EventFilter::Pipeline(alpha.pipeline_id))
            .await
            .expect("subscribe");
        let mut events: Vec<RecordingEvent> = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        assert!(!events.is_empty(), "expected historical events");
        for event in &events {
            let owner = event_owner(event);
            assert_eq!(
                owner,
                Some(alpha.pipeline_id),
                "every yielded event must belong to alpha"
            );
        }
    }
}
