use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use proxima_primitives::sync::task::JoinSet;
use time::OffsetDateTime;
use tokio::sync::watch;

use crate::error::ProximaError;
use crate::log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LogBuffer};
use crate::pipelines::dag::topological_order;
use crate::pipelines::spec::{PipelineSpec, StageSpec};
use crate::recording::event::{
    InteractionId, PipelineEvent, PipelineOutcome, ProcessEvent, ProtocolEvent, RecordingEvent,
};
use crate::recording::process_bridge::ProcessEventBridge;
use crate::recording::sink::DynRecordingSink;
use crate::upstreams::{ProcessSpec, ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal};

/// Replay a recorded pipeline into a new sink under a new pipeline id,
/// optionally substituting selected stages with fresh `StageSpec`s.
///
/// Non-substituted stages: their recorded `ProcessEvent` stream is
/// re-emitted with `id` / `parent` rewritten to the new pipeline /
/// stage ids. Bit-identical observable behavior (same exit code, same
/// stdout/stderr bytes, same per-line ordering) modulo `ts_ms`.
///
/// Substituted stages: a fresh `ProcessUpstream` + `ProcessEventBridge`
/// runs the substitute spec live and records its events. Downstream
/// stages see the substituted exit code via the strict-failure rule:
/// if a substituted stage exits non-zero, its descendants skip just
/// like in the original executor.
///
/// `original_events` must be the complete recorded event stream for
/// the source pipeline (caller already filtered to that pipeline's
/// `(event.id, event.parent)` set). The new pipeline id is allocated
/// inside this function and returned in the report.
pub async fn replay_pipeline(
    original_spec: &PipelineSpec,
    original_events: &[RecordingEvent],
    substitutes: &BTreeMap<String, StageSpec>,
    sink: DynRecordingSink,
) -> Result<ReplayReport, ProximaError> {
    replay_pipeline_with_id(
        original_spec,
        original_events,
        substitutes,
        sink,
        InteractionId::new(),
    )
    .await
}

/// Same as [`replay_pipeline`] but with a caller-supplied id, so a
/// `PipelineControlPlane` impl can pre-register the new pipeline in
/// its state (under that id) before any events flow through the sink.
pub async fn replay_pipeline_with_id(
    original_spec: &PipelineSpec,
    original_events: &[RecordingEvent],
    substitutes: &BTreeMap<String, StageSpec>,
    sink: DynRecordingSink,
    new_pipeline_id: InteractionId,
) -> Result<ReplayReport, ProximaError> {
    let order = topological_order(original_spec)?;
    let stages_by_name: BTreeMap<String, StageSpec> = original_spec
        .stages
        .iter()
        .map(|stage| (stage.name.clone(), stage.clone()))
        .collect();

    // map: original stage_id (from ProcessEvent::Started events) → stage_name.
    // we walk Process::Started events; command+args matching gives the binding
    // because the executor always emits Started with the StageSpec's command+args.
    let original_stage_id_by_name = derive_original_stage_ids(original_events, original_spec)?;
    let new_stage_ids: BTreeMap<String, InteractionId> = order
        .iter()
        .map(|name| (name.clone(), InteractionId::new()))
        .collect();

    let mut completion_tx: BTreeMap<String, watch::Sender<StageCompletion>> = BTreeMap::new();
    let mut completion_rx: BTreeMap<String, watch::Receiver<StageCompletion>> = BTreeMap::new();
    for name in &order {
        let (tx, rx) = watch::channel(StageCompletion::Pending);
        completion_tx.insert(name.clone(), tx);
        completion_rx.insert(name.clone(), rx);
    }

    let started_ts = OffsetDateTime::now_utc();
    let pipeline_started = Instant::now();
    sink.append(RecordingEvent {
        id: new_pipeline_id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Pipeline(PipelineEvent::Started {
            ts: started_ts,
            spec_hash: original_spec.spec_hash(),
            name: original_spec.name.clone(),
        }),
    })
    .await?;

    let mut tasks: JoinSet<()> = JoinSet::new();
    for name in &order {
        let stage_spec = stages_by_name[name].clone();
        let new_stage_id = new_stage_ids[name];
        let substitute = substitutes.get(name).cloned();
        let parent_receivers: Vec<watch::Receiver<StageCompletion>> = stage_spec
            .depends_on
            .iter()
            .map(|parent_name| completion_rx[parent_name].clone())
            .collect();
        let completion = completion_tx.remove(name).ok_or_else(|| {
            ProximaError::Config(format!("missing completion sender for stage `{name}`"))
        })?;
        let sink = sink.clone();
        let original_stage_id = original_stage_id_by_name.get(name).copied();
        let events_for_stage: Vec<RecordingEvent> = match original_stage_id {
            Some(id) => original_events
                .iter()
                .filter(|event| event.id == id)
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        let stage_name_owned = name.clone();
        tasks.spawn(async move {
            let outcome = run_or_replay_stage(
                stage_name_owned,
                new_stage_id,
                new_pipeline_id,
                stage_spec,
                substitute,
                events_for_stage,
                parent_receivers,
                sink,
            )
            .await;
            let completion_value = match outcome {
                Ok(StageOutcome::Completed { exit_code }) => {
                    StageCompletion::Completed { exit_code }
                }
                Ok(StageOutcome::Skipped) => StageCompletion::Skipped,
                Err(_) => StageCompletion::Skipped,
            };
            let _ = completion.send(completion_value);
        });
    }
    while tasks.join_next().await.is_some() {}

    let mut all_ok = true;
    let mut first_failure: Option<String> = None;
    let mut stage_report: BTreeMap<String, StageCompletion> = BTreeMap::new();
    for name in &order {
        let final_state = *completion_rx[name].borrow();
        stage_report.insert(name.clone(), final_state);
        match final_state {
            StageCompletion::Completed { exit_code: Some(0) } => {}
            other => {
                all_ok = false;
                if first_failure.is_none() {
                    first_failure = Some(format!("stage `{name}` final state: {other:?}"));
                }
            }
        }
    }
    let outcome = if all_ok {
        PipelineOutcome::Completed
    } else {
        PipelineOutcome::Failed {
            reason: first_failure.unwrap_or_else(|| "unknown failure".into()),
        }
    };
    sink.append(RecordingEvent {
        id: new_pipeline_id,
        ts_ms: pipeline_started.elapsed().as_millis() as u64,
        parent: None,
        event: ProtocolEvent::Pipeline(PipelineEvent::Ended {
            outcome: outcome.clone(),
        }),
    })
    .await?;

    Ok(ReplayReport {
        pipeline_id: new_pipeline_id,
        outcome,
        stages: new_stage_ids,
    })
}

#[derive(Debug)]
pub struct ReplayReport {
    pub pipeline_id: InteractionId,
    pub outcome: PipelineOutcome,
    pub stages: BTreeMap<String, InteractionId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageCompletion {
    Pending,
    Skipped,
    Completed { exit_code: Option<i32> },
}

enum StageOutcome {
    Completed { exit_code: Option<i32> },
    Skipped,
}

/// Match recorded `ProcessEvent::Started` events to spec stage names by
/// (command, args) — the executor always emits Started with the spec's
/// literal command + args, so this is unambiguous. Stages without a
/// matching Started in the recording (e.g. they were skipped on the
/// original run) get no entry and replay will run them live if no
/// substitute was provided… but they can't run because there's nothing
/// to replay either. v1: skip them.
fn derive_original_stage_ids(
    events: &[RecordingEvent],
    spec: &PipelineSpec,
) -> Result<BTreeMap<String, InteractionId>, ProximaError> {
    let mut output: BTreeMap<String, InteractionId> = BTreeMap::new();
    for event in events {
        let ProtocolEvent::Process(ProcessEvent::Started { command, args, .. }) = &event.event
        else {
            continue;
        };
        for stage in &spec.stages {
            if &stage.command == command && &stage.args == args && !output.contains_key(&stage.name)
            {
                output.insert(stage.name.clone(), event.id);
                break;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
async fn run_or_replay_stage(
    stage_name: String,
    new_stage_id: InteractionId,
    new_pipeline_id: InteractionId,
    original_stage_spec: StageSpec,
    substitute: Option<StageSpec>,
    recorded_events: Vec<RecordingEvent>,
    mut parents: Vec<watch::Receiver<StageCompletion>>,
    sink: DynRecordingSink,
) -> Result<StageOutcome, ProximaError> {
    for receiver in &mut parents {
        loop {
            let current = *receiver.borrow();
            if !matches!(current, StageCompletion::Pending) {
                break;
            }
            if receiver.changed().await.is_err() {
                return Ok(StageOutcome::Skipped);
            }
        }
        match *receiver.borrow() {
            StageCompletion::Completed { exit_code: Some(0) } => continue,
            _ => return Ok(StageOutcome::Skipped),
        }
    }

    if let Some(substitute_spec) = substitute {
        // run live — same path as the live executor
        return run_stage_live(
            stage_name,
            new_stage_id,
            new_pipeline_id,
            substitute_spec,
            sink,
        )
        .await;
    }
    if recorded_events.is_empty() {
        // no recorded events and no substitute — nothing to do; skip.
        return Ok(StageOutcome::Skipped);
    }
    replay_recorded_stage(
        new_stage_id,
        new_pipeline_id,
        recorded_events,
        sink,
        original_stage_spec,
    )
    .await
}

async fn run_stage_live(
    _stage_name: String,
    new_stage_id: InteractionId,
    new_pipeline_id: InteractionId,
    spec: StageSpec,
    sink: DynRecordingSink,
) -> Result<StageOutcome, ProximaError> {
    let process_spec = ProcessSpec {
        command: spec.command,
        args: spec.args,
        working_dir: spec.cwd,
        env: spec.env,
        restart: RestartPolicy::Never,
        restart_delay_ms: 0,
        max_restart_attempts: 0,
        ready_probe: ReadyProbe::None,
        shutdown_signal: ShutdownSignal::Term,
        shutdown_timeout_ms: 5_000,
    };
    let stdout_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
    let stderr_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
    let upstream = Arc::new(
        ProcessUpstream::spawn_with_buffers(
            format!("__replay_{new_pipeline_id}_stage_{new_stage_id}"),
            process_spec,
            Some(stdout_buf),
            Some(stderr_buf),
            None,
        )
        .await?,
    );
    let handle =
        ProcessEventBridge::attach(upstream.clone(), new_stage_id, Some(new_pipeline_id), sink)
            .await?;
    let exit_code = handle.wait_for_exit().await;
    for _ in 0..32 {
        proxima_primitives::sync::task::yield_now().await;
    }
    drop(upstream);
    Ok(StageOutcome::Completed { exit_code })
}

async fn replay_recorded_stage(
    new_stage_id: InteractionId,
    new_pipeline_id: InteractionId,
    events: Vec<RecordingEvent>,
    sink: DynRecordingSink,
    _original_stage_spec: StageSpec,
) -> Result<StageOutcome, ProximaError> {
    let mut exit_code: Option<Option<i32>> = None;
    for event in events {
        // rewrite the (id, parent) header so the replay carries the
        // new pipeline_id / new_stage_id but preserves all protocol
        // payload bytes verbatim.
        let rewritten = RecordingEvent {
            id: new_stage_id,
            ts_ms: event.ts_ms,
            parent: Some(new_pipeline_id),
            event: event.event.clone(),
        };
        if let ProtocolEvent::Process(ProcessEvent::Exited { exit_code: code }) = &rewritten.event {
            exit_code = Some(*code);
        }
        sink.append(rewritten).await?;
    }
    Ok(StageOutcome::Completed {
        exit_code: exit_code.unwrap_or(None),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::executor::PipelineExecutor;
    use crate::pipelines::spec::StageSpec;
    use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct MemorySink {
        events: StdMutex<Vec<RecordingEvent>>,
    }
    impl RecordingSink for MemorySink {
        fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
            Box::pin(async move {
                self.events.lock().expect("lock").push(event);
                Ok(())
            })
        }
        fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
            Box::pin(async { Ok(()) })
        }
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

    async fn run_to_collect_events(spec: PipelineSpec) -> Vec<RecordingEvent> {
        let sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_sink: DynRecordingSink = sink.clone();
        let executor = PipelineExecutor::new(dyn_sink);
        let _report = executor.run(spec).await.expect("run");
        sink.events.lock().expect("lock").clone()
    }

    fn filter_for_pipeline(
        events: &[RecordingEvent],
        pipeline_id: InteractionId,
    ) -> Vec<RecordingEvent> {
        events
            .iter()
            .filter(|event| event.id == pipeline_id || event.parent == Some(pipeline_id))
            .cloned()
            .collect()
    }

    #[proxima::test]
    async fn replay_without_substitutes_reproduces_recorded_stdout_per_stage() {
        let spec = PipelineSpec {
            name: Some("replay-roundtrip".into()),
            stages: vec![
                shell_stage("a", "echo aaa; exit 0", &[]),
                shell_stage("b", "echo bbb; exit 0", &["a"]),
            ],
        };
        let original_events = run_to_collect_events(spec.clone()).await;
        let original_pipeline_id = original_events
            .iter()
            .find_map(|event| match &event.event {
                ProtocolEvent::Pipeline(PipelineEvent::Started { .. }) => Some(event.id),
                _ => None,
            })
            .expect("original Pipeline::Started");
        let scoped = filter_for_pipeline(&original_events, original_pipeline_id);

        let replay_sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_replay: DynRecordingSink = replay_sink.clone();
        let report = replay_pipeline(&spec, &scoped, &BTreeMap::new(), dyn_replay)
            .await
            .expect("replay");
        assert!(matches!(report.outcome, PipelineOutcome::Completed));

        let replay_events = replay_sink.events.lock().expect("lock").clone();
        // every Process::Started in the replay must carry the new pipeline_id as parent
        // and have its id remapped to one of the new stage ids
        let new_stage_ids: std::collections::HashSet<InteractionId> =
            report.stages.values().copied().collect();
        for event in &replay_events {
            if let ProtocolEvent::Process(_) = &event.event {
                assert_eq!(event.parent, Some(report.pipeline_id));
                assert!(
                    new_stage_ids.contains(&event.id),
                    "process event must use a remapped stage id"
                );
            }
        }
        // stdout payloads must match between original and replay, per stage
        let original_stdout: Vec<Vec<u8>> = scoped
            .iter()
            .filter_map(|event| match &event.event {
                ProtocolEvent::Process(ProcessEvent::Stdout(bytes)) => Some(bytes.to_vec()),
                _ => None,
            })
            .collect();
        let replay_stdout: Vec<Vec<u8>> = replay_events
            .iter()
            .filter_map(|event| match &event.event {
                ProtocolEvent::Process(ProcessEvent::Stdout(bytes)) => Some(bytes.to_vec()),
                _ => None,
            })
            .collect();
        assert_eq!(original_stdout, replay_stdout, "stdout bytes preserved");
    }

    #[proxima::test]
    async fn replay_with_substitute_runs_substituted_stage_live_and_propagates_failure() {
        let spec = PipelineSpec {
            name: Some("replay-with-failure".into()),
            stages: vec![
                shell_stage("a", "echo a; exit 0", &[]),
                shell_stage("b", "echo b; exit 0", &["a"]),
                shell_stage("c", "echo c; exit 0", &["b"]),
            ],
        };
        let original_events = run_to_collect_events(spec.clone()).await;
        let original_pipeline_id = original_events
            .iter()
            .find_map(|event| match &event.event {
                ProtocolEvent::Pipeline(PipelineEvent::Started { .. }) => Some(event.id),
                _ => None,
            })
            .expect("original Pipeline::Started");
        let scoped = filter_for_pipeline(&original_events, original_pipeline_id);

        // substitute stage `b` with a failing variant
        let mut substitutes = BTreeMap::new();
        substitutes.insert("b".to_string(), shell_stage("b", "exit 3", &["a"]));

        let replay_sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_replay: DynRecordingSink = replay_sink.clone();
        let report = replay_pipeline(&spec, &scoped, &substitutes, dyn_replay)
            .await
            .expect("replay");
        assert!(matches!(report.outcome, PipelineOutcome::Failed { .. }));

        // c must have been skipped — no Process events with c's new stage id
        let c_stage_id = report.stages["c"];
        let snapshot = replay_sink.events.lock().expect("lock").clone();
        let c_event_count = snapshot
            .iter()
            .filter(|event| {
                event.id == c_stage_id && matches!(event.event, ProtocolEvent::Process(_))
            })
            .count();
        assert_eq!(c_event_count, 0, "stage c must be skipped after b fails");
    }
}
