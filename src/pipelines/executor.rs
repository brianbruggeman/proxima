use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use proxima_http::templates::{TemplateContext, expand_strict};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::sync::task::JoinSet;
use proxima_telemetry::error;
use serde_json::{Map, Value};
use time::OffsetDateTime;
use tokio::sync::watch;

use crate::PipeHandle;
use crate::error::ProximaError;
use crate::log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LiveTailReceiver, LogBuffer};
use crate::pipelines::dag::topological_order;
use crate::pipelines::spec::{PipelineSpec, StageBody, StageRequest, StageSpec};
use crate::recording::event::{
    InteractionId, PipelineEvent, PipelineOutcome, ProtocolEvent, RecordingEvent,
};
use crate::recording::process_bridge::{BridgeHandle, ProcessEventBridge};
use crate::recording::sink::DynRecordingSink;
use crate::request::Request;
use crate::upstreams::{ProcessSpec, ProcessUpstream, ReadyProbe, RestartPolicy, ShutdownSignal};

/// Strict-mode DAG executor: stages run when ALL their parents have
/// completed successfully (`Completed { exit_code: Some(0) }`); if any
/// parent failed or was skipped, the downstream stage is itself
/// `Skipped` (no process spawn, no `Started` event). The whole pipeline
/// is `Completed` only when every stage completed with exit code 0.
pub struct PipelineExecutor {
    sink: DynRecordingSink,
    workspace_root: Option<PathBuf>,
    pipes: Arc<BTreeMap<String, PipeHandle>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageCompletion {
    Pending,
    Skipped,
    Failed,
    Completed { exit_code: Option<i32> },
}

type OutputReceiver = watch::Receiver<Option<Arc<Value>>>;
type CompletionSender = watch::Sender<StageCompletion>;
type OutputSender = watch::Sender<Option<Arc<Value>>>;
type Failures = BTreeMap<String, String>;
type StageFailure = Option<(String, String)>;

/// What a stage needs from the run around it: its parents' channels and
/// the pipeline-wide pipes and input.
struct StageLinks {
    parents: Vec<ParentLink>,
    pipes: Arc<BTreeMap<String, PipeHandle>>,
    input: Arc<Value>,
}

struct ParentLink {
    name: String,
    completion: watch::Receiver<StageCompletion>,
    output: OutputReceiver,
}

/// Everything one `execute` call carries between its phases: the stage
/// order and ids, plus the per-stage completion and output channels whose
/// receivers are cloned to downstream stages so each child can await its
/// parents.
struct RunPlan {
    pipeline_id: InteractionId,
    input: Arc<Value>,
    order: Vec<String>,
    stage_ids: BTreeMap<String, InteractionId>,
    stages_by_name: BTreeMap<String, StageSpec>,
    completion_tx: BTreeMap<String, CompletionSender>,
    completion_rx: BTreeMap<String, watch::Receiver<StageCompletion>>,
    output_tx: BTreeMap<String, OutputSender>,
    output_rx: BTreeMap<String, OutputReceiver>,
}

impl RunPlan {
    fn new(
        spec: &PipelineSpec,
        pipeline_id: InteractionId,
        input: Value,
    ) -> Result<Self, ProximaError> {
        let order = topological_order(spec)?;
        let mut plan = Self {
            pipeline_id,
            input: Arc::new(input),
            stage_ids: assign_ids(&order),
            stages_by_name: index_stages(spec),
            order,
            completion_tx: BTreeMap::new(),
            completion_rx: BTreeMap::new(),
            output_tx: BTreeMap::new(),
            output_rx: BTreeMap::new(),
        };
        plan.open_channels();
        Ok(plan)
    }

    fn open_channels(&mut self) {
        for name in &self.order {
            let (tx, rx) = watch::channel(StageCompletion::Pending);
            self.completion_tx.insert(name.clone(), tx);
            self.completion_rx.insert(name.clone(), rx);
            let (tx, rx) = watch::channel(None);
            self.output_tx.insert(name.clone(), tx);
            self.output_rx.insert(name.clone(), rx);
        }
    }

    fn links_for(&self, spec: &StageSpec, pipes: &Arc<BTreeMap<String, PipeHandle>>) -> StageLinks {
        let parents = spec
            .depends_on
            .iter()
            .map(|parent| ParentLink {
                name: parent.clone(),
                completion: self.completion_rx[parent].clone(),
                output: self.output_rx[parent].clone(),
            })
            .collect();
        StageLinks {
            parents,
            pipes: pipes.clone(),
            input: self.input.clone(),
        }
    }

    fn take_senders(
        &mut self,
        name: &str,
    ) -> Result<(CompletionSender, OutputSender), ProximaError> {
        let completion = self.completion_tx.remove(name).ok_or_else(|| {
            ProximaError::Config(format!("missing completion sender for stage `{name}`"))
        })?;
        let output = self.output_tx.remove(name).ok_or_else(|| {
            ProximaError::Config(format!("missing output sender for stage `{name}`"))
        })?;
        Ok((completion, output))
    }

    // reads each stage's final completion; a non-zero exit is recorded as a
    // failure reason, and the first non-success decides the pipeline outcome.
    fn fold_outcome(
        &self,
        failures: &mut Failures,
    ) -> (BTreeMap<String, StageCompletion>, PipelineOutcome) {
        let mut stage_report = BTreeMap::new();
        let mut first_failure: Option<String> = None;
        for name in &self.order {
            let state = *self.completion_rx[name].borrow();
            stage_report.insert(name.clone(), state);
            note_exit_failure(name, state, failures);
            first_failure = first_failure.or_else(|| failure_reason(name, state, failures));
        }
        let outcome = first_failure.map_or(PipelineOutcome::Completed, |reason| {
            PipelineOutcome::Failed { reason }
        });
        (stage_report, outcome)
    }

    fn outputs(&self) -> BTreeMap<String, Value> {
        self.order
            .iter()
            .filter_map(|name| {
                let value = self.output_rx[name].borrow().clone()?;
                Some((name.clone(), Value::clone(&value)))
            })
            .collect()
    }

    fn into_report(
        self,
        outcome: PipelineOutcome,
        stage_report: &BTreeMap<String, StageCompletion>,
        failures: Failures,
    ) -> PipelineRunReport {
        let outputs = self.outputs();
        let stages = stage_reports(self.stage_ids, stage_report);
        PipelineRunReport {
            pipeline_id: self.pipeline_id,
            outcome,
            stages,
            outputs,
            failures,
        }
    }
}

fn assign_ids(order: &[String]) -> BTreeMap<String, InteractionId> {
    order
        .iter()
        .map(|name| (name.clone(), InteractionId::new()))
        .collect()
}

fn index_stages(spec: &PipelineSpec) -> BTreeMap<String, StageSpec> {
    let stages = spec.stages.iter();
    stages
        .map(|stage| (stage.name.clone(), stage.clone()))
        .collect()
}

fn stage_reports(
    stage_ids: BTreeMap<String, InteractionId>,
    finals: &BTreeMap<String, StageCompletion>,
) -> BTreeMap<String, StageReport> {
    stage_ids
        .into_iter()
        .map(|(name, id)| {
            let completion = finals
                .get(&name)
                .copied()
                .unwrap_or(StageCompletion::Pending);
            (name, StageReport { id, completion })
        })
        .collect()
}

fn note_exit_failure(name: &str, state: StageCompletion, failures: &mut Failures) {
    if let StageCompletion::Completed { exit_code } = state
        && exit_code != Some(0)
    {
        failures.insert(name.to_owned(), format!("exited with code {exit_code:?}"));
    }
}

// the reason a stage's final state is not a clean success, None when it is.
fn failure_reason(name: &str, state: StageCompletion, failures: &Failures) -> Option<String> {
    match state {
        StageCompletion::Completed { exit_code: Some(0) } => None,
        StageCompletion::Completed { exit_code } => {
            Some(format!("stage `{name}` exited with code {exit_code:?}"))
        }
        StageCompletion::Skipped => Some(format!("stage `{name}` skipped")),
        StageCompletion::Failed => {
            let reason = failures.get(name).map_or("", String::as_str);
            Some(format!("stage `{name}` failed: {reason}"))
        }
        StageCompletion::Pending => Some(format!("stage `{name}` never completed")),
    }
}

/// One stage ready to run: its spec and links plus the two senders it
/// publishes completion and output through.
struct StageTask {
    name: String,
    stage_id: InteractionId,
    pipeline_id: InteractionId,
    spec: StageSpec,
    links: StageLinks,
    sink: DynRecordingSink,
    completion: CompletionSender,
    output: OutputSender,
}

impl StageTask {
    async fn run(self) -> StageFailure {
        let outcome = run_stage(
            self.stage_id,
            self.pipeline_id,
            self.spec,
            self.links,
            self.sink,
        )
        .await;
        if let Err(error) = &outcome {
            error!(stage = %self.name, ?error, "stage execution failed");
        }
        let (completion_value, reason) = settle(outcome, &self.output);
        let _ = self.completion.send(completion_value);
        reason.map(|text| (self.name, text))
    }
}

impl PipelineExecutor {
    #[must_use]
    pub fn new(sink: DynRecordingSink) -> Self {
        Self {
            sink,
            workspace_root: None,
            pipes: Arc::new(BTreeMap::new()),
        }
    }

    /// Name the pipes a pipe stage can call (`StageBody::Pipe { pipe, .. }`).
    /// Each is an ordinary [`PipeHandle`], so retry / rate-limit / transform
    /// middleware is whatever the caller already wrapped it in.
    #[must_use]
    pub fn with_pipes(mut self, pipes: BTreeMap<String, PipeHandle>) -> Self {
        self.pipes = Arc::new(pipes);
        self
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
        self.execute(spec, pipeline_id, Value::Null).await
    }

    /// Run the pipeline with `input` exposed to pipe stages as `{{body.in...}}`.
    pub async fn run_with_input(
        &self,
        spec: PipelineSpec,
        input: Value,
    ) -> Result<PipelineRunReport, ProximaError> {
        self.execute(spec, InteractionId::new(), input).await
    }

    async fn execute(
        &self,
        spec: PipelineSpec,
        pipeline_id: InteractionId,
        input: Value,
    ) -> Result<PipelineRunReport, ProximaError> {
        spec.validate()?;
        let mut plan = RunPlan::new(&spec, pipeline_id, input)?;
        let started = Instant::now();
        self.record_started(pipeline_id, &spec).await?;
        let mut failures = self.run_stages(&mut plan).await?;
        let (stage_report, outcome) = plan.fold_outcome(&mut failures);
        self.record_ended(pipeline_id, started, &outcome).await?;
        Ok(plan.into_report(outcome, &stage_report, failures))
    }

    async fn record_started(
        &self,
        pipeline_id: InteractionId,
        spec: &PipelineSpec,
    ) -> Result<(), ProximaError> {
        let event = PipelineEvent::Started {
            ts: OffsetDateTime::now_utc(),
            spec_hash: spec.spec_hash(),
            name: spec.name.clone(),
        };
        self.record(pipeline_id, 0, event).await
    }

    async fn record_ended(
        &self,
        pipeline_id: InteractionId,
        started: Instant,
        outcome: &PipelineOutcome,
    ) -> Result<(), ProximaError> {
        let outcome = outcome.clone();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        self.record(pipeline_id, elapsed_ms, PipelineEvent::Ended { outcome })
            .await
    }

    async fn record(
        &self,
        id: InteractionId,
        ts_ms: u64,
        event: PipelineEvent,
    ) -> Result<(), ProximaError> {
        let event = ProtocolEvent::Pipeline(event);
        let recording = RecordingEvent {
            id,
            ts_ms,
            parent: None,
            event,
        };
        self.sink.append(recording).await?;
        Ok(())
    }

    async fn run_stages(&self, plan: &mut RunPlan) -> Result<Failures, ProximaError> {
        let mut stage_tasks: JoinSet<StageFailure> = JoinSet::new();
        for name in plan.order.clone() {
            let task = self.stage_task(plan, &name).await?;
            stage_tasks.spawn(task.run());
        }
        let mut failures = Failures::new();
        while let Some(joined) = stage_tasks.join_next().await {
            if let Ok(Some((name, reason))) = joined {
                failures.insert(name, reason);
            }
        }
        Ok(failures)
    }

    async fn stage_task(&self, plan: &mut RunPlan, name: &str) -> Result<StageTask, ProximaError> {
        let spec = self.prepare_stage(plan, name).await?;
        let links = plan.links_for(&spec, &self.pipes);
        let (completion, output) = plan.take_senders(name)?;
        Ok(StageTask {
            name: name.to_owned(),
            stage_id: plan.stage_ids[name],
            pipeline_id: plan.pipeline_id,
            spec,
            links,
            sink: self.sink.clone(),
            completion,
            output,
        })
    }

    async fn prepare_stage(&self, plan: &RunPlan, name: &str) -> Result<StageSpec, ProximaError> {
        let mut stage_spec = plan
            .stages_by_name
            .get(name)
            .ok_or_else(|| {
                ProximaError::Config(format!("stage `{name}` missing from spec lookup"))
            })?
            .clone();
        // workspace default: if the executor has a workspace_root set and
        // the stage didn't pick a cwd, materialize per-stage workspace
        // under <root>/<pipeline_id>/workspace/<stage_name>/.
        if let StageBody::Process { cwd, .. } = &mut stage_spec.body
            && cwd.is_none()
            && let Some(root) = &self.workspace_root
        {
            *cwd = Some(create_workspace(root, plan.pipeline_id, &stage_spec.name).await?);
        }
        Ok(stage_spec)
    }
}

async fn create_workspace(
    root: &Path,
    pipeline_id: InteractionId,
    stage_name: &str,
) -> Result<PathBuf, ProximaError> {
    let workspace = root
        .join(pipeline_id.to_string())
        .join("workspace")
        .join(stage_name);
    tokio::fs::create_dir_all(&workspace).await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "create stage workspace {workspace:?}: {err}"
        )))
    })?;
    Ok(workspace)
}

/// `outputs` holds each succeeded pipe stage's value; `failures` holds the
/// reason of every pipe stage that failed (process stages report through
/// [`StageReport::exit_code`]).
#[derive(Debug)]
pub struct PipelineRunReport {
    pub pipeline_id: InteractionId,
    pub outcome: PipelineOutcome,
    pub stages: BTreeMap<String, StageReport>,
    pub outputs: BTreeMap<String, Value>,
    pub failures: BTreeMap<String, String>,
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

    #[must_use]
    pub fn was_failed(&self) -> bool {
        matches!(self.completion, StageCompletion::Failed)
    }
}

enum StageOutcome {
    Completed { exit_code: Option<i32> },
    Succeeded { output: Value },
    Failed { reason: String },
    Skipped,
}

fn settle(
    outcome: Result<StageOutcome, ProximaError>,
    output: &watch::Sender<Option<Arc<Value>>>,
) -> (StageCompletion, Option<String>) {
    match outcome {
        Ok(StageOutcome::Completed { exit_code }) => {
            (StageCompletion::Completed { exit_code }, None)
        }
        Ok(StageOutcome::Succeeded { output: value }) => {
            let _ = output.send(Some(Arc::new(value)));
            (StageCompletion::Completed { exit_code: Some(0) }, None)
        }
        Ok(StageOutcome::Failed { reason }) => (StageCompletion::Failed, Some(reason)),
        Ok(StageOutcome::Skipped) | Err(_) => (StageCompletion::Skipped, None),
    }
}

async fn run_stage(
    stage_id: InteractionId,
    pipeline_id: InteractionId,
    spec: StageSpec,
    mut links: StageLinks,
    sink: DynRecordingSink,
) -> Result<StageOutcome, ProximaError> {
    if !parents_succeeded(&mut links.parents).await {
        return Ok(StageOutcome::Skipped);
    }
    match spec.body {
        StageBody::Process {
            command,
            args,
            env,
            cwd,
        } => run_process_stage(stage_id, pipeline_id, (command, args, env, cwd), sink).await,
        StageBody::Pipe { pipe, request } => {
            Ok(run_pipe_stage(&spec.name, &pipe, &request, &links).await)
        }
    }
}

// strict policy: every parent must complete with exit_code Some(0). any
// non-success short-circuits to Skipped without running the stage.
async fn parents_succeeded(parents: &mut [ParentLink]) -> bool {
    for parent in parents {
        loop {
            if !matches!(*parent.completion.borrow(), StageCompletion::Pending) {
                break;
            }
            if parent.completion.changed().await.is_err() {
                return false;
            }
        }
        if !matches!(
            *parent.completion.borrow(),
            StageCompletion::Completed { exit_code: Some(0) }
        ) {
            return false;
        }
    }
    true
}

async fn run_pipe_stage(
    stage: &str,
    pipe: &str,
    request: &StageRequest,
    links: &StageLinks,
) -> StageOutcome {
    match pipe_stage_output(stage, pipe, request, links).await {
        Ok(output) => StageOutcome::Succeeded { output },
        Err(reason) => StageOutcome::Failed { reason },
    }
}

async fn pipe_stage_output(
    stage: &str,
    pipe: &str,
    request: &StageRequest,
    links: &StageLinks,
) -> Result<Value, String> {
    let scope = build_scope(&links.input, &links.parents);
    let built = build_request(request, &scope).map_err(|placeholder| {
        format!("stage `{stage}` placeholder `{placeholder}` did not resolve")
    })?;
    let handle = links
        .pipes
        .get(pipe)
        .ok_or_else(|| format!("stage `{stage}` uses unknown pipe `{pipe}`"))?;
    call_pipe(handle, built)
        .await
        .map_err(|reason| format!("stage `{stage}` pipe `{pipe}`: {reason}"))
}

fn build_scope(input: &Value, parents: &[ParentLink]) -> Value {
    let mut scope = Map::new();
    scope.insert("in".into(), input.clone());
    for parent in parents {
        if let Some(output) = parent.output.borrow().as_ref() {
            scope.insert(parent.name.clone(), Value::clone(output));
        }
    }
    Value::Object(scope)
}

// Err carries the unresolved placeholder key, or the builder's reason.
fn build_request(request: &StageRequest, scope: &Value) -> Result<Request<Bytes>, String> {
    let context = TemplateContext {
        body: Some(scope),
        ..Default::default()
    };
    let expand = |text: &str| expand_strict(text, &context);
    let mut builder = Request::builder()
        .method(expand(&request.method)?.as_str())
        .path(expand(&request.path)?);
    for (name, value) in &request.headers {
        builder = builder.header(name.as_str(), expand(value)?);
    }
    for (name, value) in &request.query {
        builder = builder.query_param(name.as_str(), expand(value)?);
    }
    if let Some(body) = &request.body {
        builder = builder.body(Bytes::from(expand(body)?));
    }
    builder.build().map_err(|err| err.to_string())
}

async fn call_pipe(handle: &PipeHandle, request: Request<Bytes>) -> Result<Value, String> {
    let response = SendPipe::call(handle, request)
        .await
        .map_err(|err| err.to_string())?;
    let status = response.status;
    let success = response.is_success();
    let body = response
        .collect_body()
        .await
        .map_err(|err| err.to_string())?;
    if !success {
        return Err(format!("returned status {status}"));
    }
    Ok(serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).into_owned())))
}

type ProcessParts = (
    String,
    Vec<String>,
    BTreeMap<String, String>,
    Option<PathBuf>,
);
type LogBuffers = (Arc<LogBuffer>, Arc<LogBuffer>);
type LogTails = (LiveTailReceiver, LiveTailReceiver);

fn process_spec((command, args, env, cwd): ProcessParts) -> ProcessSpec {
    ProcessSpec {
        command,
        args,
        working_dir: cwd,
        env,
        restart: RestartPolicy::Never,
        restart_delay_ms: 0,
        max_restart_attempts: 0,
        ready_probe: ReadyProbe::None,
        shutdown_signal: ShutdownSignal::Term,
        shutdown_timeout_ms: 5_000,
    }
}

async fn spawn_upstream(
    stage_id: InteractionId,
    pipeline_id: InteractionId,
    parts: ProcessParts,
    (stdout_buf, stderr_buf): LogBuffers,
) -> Result<Arc<ProcessUpstream>, ProximaError> {
    let upstream = ProcessUpstream::spawn_with_buffers(
        format!("__pipeline_{pipeline_id}_stage_{stage_id}"),
        process_spec(parts),
        Some(stdout_buf),
        Some(stderr_buf),
        None,
    )
    .await?;
    Ok(Arc::new(upstream))
}

async fn attach_bridge(
    upstream: &Arc<ProcessUpstream>,
    (stdout_rx, stderr_rx): LogTails,
    (stage_id, pipeline_id): (InteractionId, InteractionId),
    sink: DynRecordingSink,
) -> Result<BridgeHandle, ProximaError> {
    ProcessEventBridge::attach(
        upstream.clone(),
        stdout_rx,
        stderr_rx,
        stage_id,
        Some(pipeline_id),
        sink,
    )
    .await
}

async fn run_process_stage(
    stage_id: InteractionId,
    pipeline_id: InteractionId,
    parts: ProcessParts,
    sink: DynRecordingSink,
) -> Result<StageOutcome, ProximaError> {
    let stdout_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
    let stderr_buf = Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY));
    let tails = (stdout_buf.subscribe(), stderr_buf.subscribe());
    let upstream = spawn_upstream(stage_id, pipeline_id, parts, (stdout_buf, stderr_buf)).await?;
    let bridge = attach_bridge(&upstream, tails, (stage_id, pipeline_id), sink).await?;
    let exit_code = bridge.wait_for_exit().await;
    drop(upstream);
    Ok(StageOutcome::Completed { exit_code })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::recording::event::{ProcessEvent, ProtocolEvent};
    use crate::recording::sink::{AppendFuture, DynRecordingSink, RecordingSink};
    use crate::request::Response;
    use proxima_http::templates::expand;
    use proxima_primitives::pipe::into_handle;
    use serde_json::json;
    use std::future::Future;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
            body: StageBody::Process {
                command: cmd.into(),
                args: vec![flag.into(), script.into()],
                env: BTreeMap::new(),
                cwd: None,
            },
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

    #[derive(Clone, Default)]
    struct Probe {
        calls: Arc<AtomicUsize>,
        paths: Arc<StdMutex<Vec<String>>>,
    }

    struct StubPipe {
        status: u16,
        body: &'static [u8],
        probe: Probe,
    }

    impl SendPipe for StubPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            self.probe.calls.fetch_add(1, Ordering::SeqCst);
            let path = String::from_utf8_lossy(&request.path).into_owned();
            self.probe.paths.lock().expect("paths").push(path);
            let response = Response::new(self.status).with_body(Bytes::from_static(self.body));
            async move { Ok(response) }
        }
    }

    const TICKET_BODY: &[u8] = br#"{"repo":"brianbruggeman/proxima","pr":42}"#;

    fn stub(status: u16, body: &'static [u8]) -> (PipeHandle, Probe) {
        let probe = Probe::default();
        let pipe = StubPipe {
            status,
            body,
            probe: probe.clone(),
        };
        (into_handle(pipe), probe)
    }

    fn pipe_stage(name: &str, pipe: &str, path: &str, deps: &[&str]) -> StageSpec {
        let request = StageRequest {
            method: "GET".into(),
            path: path.into(),
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
            body: None,
        };
        StageSpec {
            name: name.into(),
            body: StageBody::Pipe {
                pipe: pipe.into(),
                request,
            },
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
        }
    }

    async fn run_pipes(
        stages: Vec<StageSpec>,
        pipes: Vec<(&str, PipeHandle)>,
        input: Value,
    ) -> PipelineRunReport {
        let sink: DynRecordingSink = Arc::new(MemorySink::default());
        let pipes = pipes
            .into_iter()
            .map(|(name, handle)| (name.to_string(), handle))
            .collect();
        let executor = PipelineExecutor::new(sink).with_pipes(pipes);
        let spec = PipelineSpec { name: None, stages };
        executor.run_with_input(spec, input).await.expect("run")
    }

    struct Observed {
        received: Vec<String>,
        calls: usize,
        failure: Option<String>,
    }

    fn observe(probe: &Probe, report: &PipelineRunReport, stage: &str) -> Observed {
        Observed {
            received: probe.paths.lock().expect("paths").clone(),
            calls: probe.calls.load(Ordering::SeqCst),
            failure: report.failures.get(stage).cloned(),
        }
    }

    enum Expect {
        Path(&'static str),
        Missing(&'static str),
    }

    fn check_template(observed: &Observed, expect: &Expect) -> Result<(), String> {
        match expect {
            Expect::Path(path) if observed.failure.is_none() && observed.received == [*path] => {
                Ok(())
            }
            Expect::Missing(key)
                if observed.calls == 0
                    && observed
                        .failure
                        .as_deref()
                        .is_some_and(|reason| reason.contains(key)) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "unexpected outcome: received {:?}, calls {}, failure {:?}",
                observed.received, observed.calls, observed.failure
            )),
        }
    }

    fn check_failure(
        report: &PipelineRunReport,
        failing: &str,
        dependent: &str,
        reason: &str,
    ) -> Result<(), String> {
        let recorded = report.failures.get(failing).map(String::as_str);
        if !recorded.is_some_and(|text| text.contains(reason)) {
            return Err(format!(
                "stage `{failing}` reason {recorded:?} lacks `{reason}`"
            ));
        }
        if !report.stages[dependent].was_skipped() {
            return Err(format!("stage `{dependent}` was not skipped"));
        }
        Ok(())
    }

    fn check_output(actual: &Value, expected: &Value) -> Result<(), String> {
        if actual == expected {
            return Ok(());
        }
        Err(format!("output {actual} differs from {expected}"))
    }

    fn ticket_scope() -> Value {
        json!({"in": {"ticket": "PROXIMA-24"}, "ticket": {"repo": "brianbruggeman/proxima", "pr": 42}})
    }

    const PR_PATH: &str = "/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}";
    const PR_EXPECTED: &str = "/repos/brianbruggeman/proxima/pulls/42";

    async fn run_ticket_then(path: &str) -> (PipelineRunReport, Probe) {
        let (ticket, _) = stub(200, TICKET_BODY);
        let (github, probe) = stub(200, b"{}");
        let stages = vec![
            pipe_stage("ticket", "vault", "/PROXIMA-24/state.json", &[]),
            pipe_stage("pr", "github", path, &["ticket"]),
        ];
        let pipes = vec![("vault", ticket), ("github", github)];
        let report = run_pipes(stages, pipes, json!({"ticket": "PROXIMA-24"})).await;
        (report, probe)
    }

    #[proxima::test]
    async fn stage_template_input_path_reaches_the_pipe() {
        let (vault, probe) = stub(200, TICKET_BODY);
        let stages = vec![pipe_stage(
            "ticket",
            "vault",
            "/{{body.in.ticket}}/state.json",
            &[],
        )];
        let report = run_pipes(
            stages,
            vec![("vault", vault)],
            json!({"ticket": "PROXIMA-24"}),
        )
        .await;
        let observed = observe(&probe, &report, "ticket");
        assert_eq!(
            check_template(&observed, &Expect::Path("/PROXIMA-24/state.json")),
            Ok(())
        );
    }

    #[proxima::test]
    async fn stage_template_dependency_output_fills_the_path() {
        let (report, probe) = run_ticket_then(PR_PATH).await;
        let observed = observe(&probe, &report, "pr");
        assert_eq!(
            check_template(&observed, &Expect::Path(PR_EXPECTED)),
            Ok(())
        );
    }

    #[proxima::test]
    async fn stage_template_missing_placeholder_fails_without_calling_the_pipe() {
        let (report, probe) = run_ticket_then("/repos/{{body.ticket.missing}}/pulls").await;
        let observed = observe(&probe, &report, "pr");
        assert_eq!(
            check_template(&observed, &Expect::Missing("body.ticket.missing")),
            Ok(())
        );
    }

    #[proxima::test]
    async fn stage_output_json_body_becomes_the_value() {
        let (vault, _) = stub(200, TICKET_BODY);
        let stages = vec![pipe_stage("ticket", "vault", "/state.json", &[])];
        let report = run_pipes(stages, vec![("vault", vault)], Value::Null).await;
        let expected = json!({"repo": "brianbruggeman/proxima", "pr": 42});
        assert_eq!(check_output(&report.outputs["ticket"], &expected), Ok(()));
    }

    #[proxima::test]
    async fn stage_output_plain_text_becomes_a_json_string() {
        let (llm, _) = stub(200, b"plain text review");
        let stages = vec![pipe_stage("review", "llm", "/v1/messages", &[])];
        let report = run_pipes(stages, vec![("llm", llm)], Value::Null).await;
        let expected = Value::String("plain text review".into());
        assert_eq!(check_output(&report.outputs["review"], &expected), Ok(()));
    }

    #[proxima::test]
    async fn stage_output_non_2xx_fails_the_stage_and_skips_dependents() {
        let (vault, _) = stub(503, b"unavailable");
        let (github, probe) = stub(200, b"{}");
        let stages = vec![
            pipe_stage("ticket", "vault", "/state.json", &[]),
            pipe_stage("pr", "github", "/pulls", &["ticket"]),
        ];
        let report = run_pipes(
            stages,
            vec![("vault", vault), ("github", github)],
            Value::Null,
        )
        .await;
        assert_eq!(check_failure(&report, "ticket", "pr", "503"), Ok(()));
        assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    }

    #[proxima::test]
    async fn ctl_template_incumbent_expand_passes_the_present_path_checks() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        for (template, expected) in [
            ("/{{body.in.ticket}}/state.json", "/PROXIMA-24/state.json"),
            (PR_PATH, PR_EXPECTED),
        ] {
            let received = vec![expand(template, &context)];
            let observed = Observed {
                received,
                calls: 1,
                failure: None,
            };
            assert_eq!(check_template(&observed, &Expect::Path(expected)), Ok(()));
        }
    }

    #[proxima::test]
    async fn ctl_template_lenient_expand_is_detected_as_the_missing_path_defect() {
        let scope = ticket_scope();
        let context = TemplateContext {
            body: Some(&scope),
            ..Default::default()
        };
        let path = expand("/repos/{{body.ticket.missing}}/pulls", &context);
        let (github, probe) = stub(200, b"{}");
        let request = Request::builder()
            .method("GET")
            .path(path)
            .build()
            .expect("request");
        SendPipe::call(&github, request).await.expect("call");
        let observed = Observed {
            received: probe.paths.lock().expect("paths").clone(),
            calls: probe.calls.load(Ordering::SeqCst),
            failure: None,
        };
        assert_eq!(observed.received, ["/repos//pulls"]);
        assert_eq!(observed.calls, 1);
        assert!(check_template(&observed, &Expect::Missing("body.ticket.missing")).is_err());
    }

    #[proxima::test]
    async fn ctl_template_hand_built_correct_failure_passes_the_check() {
        let observed = Observed {
            received: vec![],
            calls: 0,
            failure: Some("placeholder `body.ticket.missing` did not resolve".into()),
        };
        assert_eq!(
            check_template(&observed, &Expect::Missing("body.ticket.missing")),
            Ok(())
        );
    }

    #[proxima::test]
    async fn ctl_output_process_exit_one_passes_the_failure_check() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                shell_stage("a", "exit 1", &[]),
                shell_stage("b", "exit 0", &["a"]),
            ],
        };
        let (report, _) = run(spec).await;
        assert_eq!(
            check_failure(&report, "a", "b", "exited with code Some(1)"),
            Ok(())
        );
    }

    #[proxima::test]
    async fn ctl_output_all_zero_exit_run_is_rejected_by_the_failure_check() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                shell_stage("a", "exit 0", &[]),
                shell_stage("b", "exit 0", &["a"]),
            ],
        };
        let (report, _) = run(spec).await;
        assert!(check_failure(&report, "a", "b", "exited with code Some(1)").is_err());
    }

    #[proxima::test]
    async fn ctl_output_known_good_json_literal_passes_the_output_check() {
        let expected = json!({"repo": "brianbruggeman/proxima", "pr": 42});
        assert_eq!(
            check_output(
                &json!({"repo": "brianbruggeman/proxima", "pr": 42}),
                &expected
            ),
            Ok(())
        );
    }

    #[proxima::test]
    async fn ctl_output_empty_object_is_rejected_by_the_json_check() {
        let expected = json!({"repo": "brianbruggeman/proxima", "pr": 42});
        assert!(check_output(&json!({}), &expected).is_err());
    }

    #[proxima::test]
    async fn ctl_output_known_good_text_literal_passes_the_output_check() {
        let expected = Value::String("plain text review".into());
        assert_eq!(
            check_output(&Value::String("plain text review".into()), &expected),
            Ok(())
        );
    }

    #[proxima::test]
    async fn ctl_output_empty_string_is_rejected_by_the_text_check() {
        let expected = Value::String("plain text review".into());
        assert!(check_output(&Value::String(String::new()), &expected).is_err());
    }
}
