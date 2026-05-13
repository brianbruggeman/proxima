use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use proxima_primitives::sync::task::JoinSet;
use time::OffsetDateTime;
use tokio::sync::watch;
use tracing::error;

use crate::error::ProximaError;
use crate::log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LogBuffer};
use crate::pipelines::dag::topological_order;
use crate::pipelines::spec::{PipelineSpec, StageSpec};
use crate::recording::event::{
    InteractionId, PipelineEvent, PipelineOutcome, ProtocolEvent, RecordingEvent,
};
use crate::recording::process_bridge::ProcessEventBridge;
use crate::recording::sink::DynRecordingSink;
use crate::upstreams::{ProcessSpec, ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal};

/// Strict-mode DAG executor: stages run when ALL their parents have
/// completed successfully (`Completed { exit_code: Some(0) }`); if any
/// parent failed or was skipped, the downstream stage is itself
/// `Skipped` (no process spawn, no `Started` event). The whole pipeline
/// is `Completed` only when every stage completed with exit code 0.
pub struct PipelineExecutor {
    sink: DynRecordingSink,
    workspace_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageCompletion {
    Pending,
    Skipped,
    Completed { exit_code: Option<i32> },
}

impl PipelineExecutor {
    #[must_use]
    pub fn new(sink: DynRecordingSink) -> Self {
        Self {
            sink,
            workspace_root: None,
        }
    }

    /// When set, stages whose `StageSpec.cwd` is `None` run in
    /// `<workspace_root>/<pipeline_id>/workspace/<stage_name>/`. The dir
    /// is created during `run_with_id` if missing. Stages with an
    /// explicit `cwd` keep using it (the user's choice wins). Without a
    /// workspace_root, unset cwds inherit proxima's working directory.
    #[must_use]
    pub fn with_workspace_root(mut self, root: PathBuf) -> Self {
        self.workspace_root = Some(root);
        self
    }

    pub async fn run(&self, spec: PipelineSpec) -> Result<PipelineRunReport, ProximaError> {
        self.run_with_id(spec, InteractionId::new()).await
    }

    /// Run the pipeline under a caller-supplied `pipeline_id`. Used by
    /// the control plane so `submit()` can return the id immediately
    /// while execution runs in a spawned task.
    pub async fn run_with_id(
        &self,
        spec: PipelineSpec,
        pipeline_id: InteractionId,
    ) -> Result<PipelineRunReport, ProximaError> {
        spec.validate()?;
        let order = topological_order(&spec)?;
        let stages_by_name: BTreeMap<String, StageSpec> = spec
            .stages
            .iter()
            .map(|stage| (stage.name.clone(), stage.clone()))
            .collect();
        let pipeline_started_ts = OffsetDateTime::now_utc();
        let pipeline_started = Instant::now();

        let stage_ids: BTreeMap<String, InteractionId> = order
            .iter()
            .map(|name| (name.clone(), InteractionId::new()))
            .collect();

        // per-stage completion channels; receivers cloned to downstream
        // stages so each child can await its parents.
        let mut completion_tx: BTreeMap<String, watch::Sender<StageCompletion>> = BTreeMap::new();
        let mut completion_rx: BTreeMap<String, watch::Receiver<StageCompletion>> = BTreeMap::new();
        for name in &order {
            let (tx, rx) = watch::channel(StageCompletion::Pending);
            completion_tx.insert(name.clone(), tx);
            completion_rx.insert(name.clone(), rx);
        }

        self.sink
            .append(RecordingEvent {
                id: pipeline_id,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Pipeline(PipelineEvent::Started {
                    ts: pipeline_started_ts,
                    spec_hash: spec.spec_hash(),
                    name: spec.name.clone(),
                }),
            })
            .await?;

        let mut stage_tasks: JoinSet<()> = JoinSet::new();
        for name in &order {
            let mut stage_spec = stages_by_name
                .get(name)
                .ok_or_else(|| {
                    ProximaError::Config(format!("stage `{name}` missing from spec lookup"))
                })?
                .clone();
            // workspace default: if the executor has a workspace_root set and
            // the stage didn't pick a cwd, materialize per-stage workspace
            // under <root>/<pipeline_id>/workspace/<stage_name>/.
            if stage_spec.cwd.is_none()
                && let Some(root) = &self.workspace_root
            {
                let workspace = root
                    .join(pipeline_id.to_string())
                    .join("workspace")
                    .join(&stage_spec.name);
                tokio::fs::create_dir_all(&workspace).await.map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "create stage workspace {workspace:?}: {err}"
                    )))
                })?;
                stage_spec.cwd = Some(workspace);
            }
            let stage_id = stage_ids[name];
            let parent_receivers: Vec<watch::Receiver<StageCompletion>> = stage_spec
                .depends_on
                .iter()
                .map(|parent_name| completion_rx[parent_name].clone())
                .collect();
            let completion = completion_tx.remove(name).ok_or_else(|| {
                ProximaError::Config(format!("missing completion sender for stage `{name}`"))
            })?;
            let sink = self.sink.clone();
            let stage_name = name.clone();
            stage_tasks.spawn(async move {
                let outcome =
                    run_stage(stage_id, pipeline_id, stage_spec, parent_receivers, sink).await;
                if let Err(error) = &outcome {
                    error!(stage = %stage_name, ?error, "stage execution failed");
                }
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
        while stage_tasks.join_next().await.is_some() {}

        // collect final stage outcomes for the report
        let mut stage_report: BTreeMap<String, StageCompletion> = BTreeMap::new();
        let mut all_completed_ok = true;
        let mut failed_stage: Option<String> = None;
        for name in &order {
            let final_state = *completion_rx[name].borrow();
            stage_report.insert(name.clone(), final_state);
            match final_state {
                StageCompletion::Completed { exit_code: Some(0) } => {}
                StageCompletion::Completed { exit_code } => {
                    all_completed_ok = false;
                    if failed_stage.is_none() {
                        failed_stage =
                            Some(format!("stage `{name}` exited with code {:?}", exit_code));
                    }
                }
                StageCompletion::Skipped => {
                    all_completed_ok = false;
                    if failed_stage.is_none() {
                        failed_stage = Some(format!("stage `{name}` skipped"));
                    }
                }
                StageCompletion::Pending => {
                    all_completed_ok = false;
                    if failed_stage.is_none() {
                        failed_stage = Some(format!("stage `{name}` never completed"));
                    }
                }
            }
        }
        let outcome = if all_completed_ok {
            PipelineOutcome::Completed
        } else {
            PipelineOutcome::Failed {
                reason: failed_stage.unwrap_or_else(|| "unknown failure".into()),
            }
        };

        self.sink
            .append(RecordingEvent {
                id: pipeline_id,
                ts_ms: pipeline_started.elapsed().as_millis() as u64,
                parent: None,
                event: ProtocolEvent::Pipeline(PipelineEvent::Ended {
                    outcome: outcome.clone(),
                }),
            })
            .await?;

        let report = PipelineRunReport {
            pipeline_id,
            outcome,
            stages: stage_ids
                .into_iter()
                .map(|(name, id)| {
                    let completion = stage_report
                        .get(&name)
                        .copied()
                        .unwrap_or(StageCompletion::Pending);
                    (name, StageReport { id, completion })
                })
                .collect(),
        };
        Ok(report)
    }
}

#[derive(Debug)]
pub struct PipelineRunReport {
    pub pipeline_id: InteractionId,
    pub outcome: PipelineOutcome,
    pub stages: BTreeMap<String, StageReport>,
}

#[derive(Debug, Clone, Copy)]
pub struct StageReport {
    pub id: InteractionId,
    completion: StageCompletion,
}

impl StageReport {
    #[must_use]
    pub fn exit_code(&self) -> Option<Option<i32>> {
        match self.completion {
            StageCompletion::Completed { exit_code } => Some(exit_code),
            _ => None,
        }
    }

    #[must_use]
    pub fn was_skipped(&self) -> bool {
        matches!(self.completion, StageCompletion::Skipped)
    }
}

enum StageOutcome {
    Completed { exit_code: Option<i32> },
    Skipped,
}

async fn run_stage(
    stage_id: InteractionId,
    pipeline_id: InteractionId,
    spec: StageSpec,
    mut parents: Vec<watch::Receiver<StageCompletion>>,
    sink: DynRecordingSink,
) -> Result<StageOutcome, ProximaError> {
    // strict policy: every parent must complete with exit_code Some(0). any
    // non-success short-circuits to Skipped without spawning the child.
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
            format!("__pipeline_{pipeline_id}_stage_{stage_id}"),
            process_spec,
            Some(stdout_buf),
            Some(stderr_buf),
            None,
        )
        .await?,
    );
    let bridge_handle =
        ProcessEventBridge::attach(upstream.clone(), stage_id, Some(pipeline_id), sink).await?;
    let exit_code = bridge_handle.wait_for_exit().await;
    // hold the upstream alive a few ticks so drain tasks can flush any
    // remaining lines emitted just before exit (the bridge tests do the
    // same dance).
    for _ in 0..32 {
        proxima_primitives::sync::task::yield_now().await;
    }
    drop(upstream);
    Ok(StageOutcome::Completed { exit_code })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::recording::event::{ProcessEvent, ProtocolEvent};
    use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    #[derive(Default)]
    struct MemorySink {
        events: StdMutex<Vec<RecordingEvent>>,
    }

    impl RecordingSink for MemorySink {
        fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
            Box::pin(async move {
                self.events.lock().expect("memory sink").push(event);
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

    async fn run(spec: PipelineSpec) -> (PipelineRunReport, Vec<RecordingEvent>) {
        let sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_sink: DynRecordingSink = sink.clone();
        let executor = PipelineExecutor::new(dyn_sink);
        let report = executor.run(spec).await.expect("run");
        let events = sink.events.lock().expect("lock").clone();
        (report, events)
    }

    #[proxima::test]
    async fn linear_pipeline_runs_stages_in_order_and_completes() {
        let spec = PipelineSpec {
            name: Some("linear".into()),
            stages: vec![
                shell_stage("a", "echo a; exit 0", &[]),
                shell_stage("b", "echo b; exit 0", &["a"]),
                shell_stage("c", "echo c; exit 0", &["b"]),
            ],
        };
        let (report, events) = run(spec).await;
        assert!(matches!(report.outcome, PipelineOutcome::Completed));
        assert_eq!(report.stages.len(), 3);
        for stage in report.stages.values() {
            assert_eq!(stage.exit_code(), Some(Some(0)));
            assert!(!stage.was_skipped());
        }
        // first event is Pipeline::Started; last is Pipeline::Ended
        assert!(matches!(
            events.first().expect("at least one event").event,
            ProtocolEvent::Pipeline(PipelineEvent::Started { .. })
        ));
        assert!(matches!(
            events.last().expect("at least one event").event,
            ProtocolEvent::Pipeline(PipelineEvent::Ended { .. })
        ));
    }

    #[proxima::test]
    async fn diamond_pipeline_runs_independent_stages_concurrently() {
        let latch_dir = TempDir::new().expect("tempdir");
        let latch_path = latch_dir.path().to_str().expect("utf8 path");

        let b_file = format!("{latch_path}/b_alive");
        let c_file = format!("{latch_path}/c_alive");

        // B signals it is alive, then busy-waits for C's signal before exiting.
        // C does the same in the other direction. Both must be alive at the same
        // instant, so their ProcessEvent::Started events will both appear before
        // either ProcessEvent::Exited event in the recorded stream.
        let b_script = format!("touch {b_file} && until [ -f {c_file} ]; do :; done");
        let c_script = format!("touch {c_file} && until [ -f {b_file} ]; do :; done");

        let spec = PipelineSpec {
            name: Some("diamond".into()),
            stages: vec![
                shell_stage("a", "exit 0", &[]),
                shell_stage("b", &b_script, &["a"]),
                shell_stage("c", &c_script, &["a"]),
                shell_stage("d", "exit 0", &["b", "c"]),
            ],
        };
        let (report, events) = run(spec).await;
        assert!(matches!(report.outcome, PipelineOutcome::Completed));

        let stage_b_id = report.stages["b"].id;
        let stage_c_id = report.stages["c"].id;

        let position = |id: InteractionId, is_started: bool| {
            events.iter().position(|event| {
                event.id == id
                    && matches!(
                        (&event.event, is_started),
                        (ProtocolEvent::Process(ProcessEvent::Started { .. }), true)
                            | (ProtocolEvent::Process(ProcessEvent::Exited { .. }), false)
                    )
            })
        };

        let b_started = position(stage_b_id, true).expect("b started event");
        let b_exited = position(stage_b_id, false).expect("b exited event");
        let c_started = position(stage_c_id, true).expect("c started event");
        let c_exited = position(stage_c_id, false).expect("c exited event");

        assert!(
            b_started < c_exited,
            "b must have started before c exited (b_started={b_started}, c_exited={c_exited})"
        );
        assert!(
            c_started < b_exited,
            "c must have started before b exited (c_started={c_started}, b_exited={b_exited})"
        );
    }

    #[proxima::test]
    async fn nonzero_exit_marks_pipeline_failed_and_skips_downstream() {
        let spec = PipelineSpec {
            name: Some("fail".into()),
            stages: vec![
                shell_stage("a", "exit 0", &[]),
                shell_stage("b", "exit 3", &["a"]),
                shell_stage("c", "echo never; exit 0", &["b"]),
            ],
        };
        let (report, events) = run(spec).await;
        assert!(matches!(report.outcome, PipelineOutcome::Failed { .. }));
        assert_eq!(report.stages["a"].exit_code(), Some(Some(0)));
        assert_eq!(report.stages["b"].exit_code(), Some(Some(3)));
        assert!(report.stages["c"].was_skipped());
        // c's process must never have been spawned (no ProcessEvent::Started for it).
        let stage_c_id = report.stages["c"].id;
        let c_started = events.iter().any(|event| {
            event.id == stage_c_id
                && matches!(
                    event.event,
                    ProtocolEvent::Process(ProcessEvent::Started { .. })
                )
        });
        assert!(!c_started, "c must not have started after b failed");
    }

    #[proxima::test]
    async fn every_process_event_carries_pipeline_id_as_parent() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![shell_stage("only", "exit 0", &[])],
        };
        let (report, events) = run(spec).await;
        let process_events: Vec<&RecordingEvent> = events
            .iter()
            .filter(|event| matches!(event.event, ProtocolEvent::Process(_)))
            .collect();
        assert!(!process_events.is_empty(), "expected process events");
        for event in process_events {
            assert_eq!(event.parent, Some(report.pipeline_id));
        }
    }

    #[proxima::test]
    async fn cycle_in_spec_returns_typed_error_before_running_anything() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                shell_stage("a", "exit 0", &["b"]),
                shell_stage("b", "exit 0", &["a"]),
            ],
        };
        let sink: Arc<MemorySink> = Arc::new(MemorySink::default());
        let dyn_sink: DynRecordingSink = sink.clone();
        let executor = PipelineExecutor::new(dyn_sink);
        let outcome = executor.run(spec).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
        // no events should have been recorded — validation happens before
        // Pipeline::Started is emitted.
        assert!(sink.events.lock().expect("lock").is_empty());
    }
}
