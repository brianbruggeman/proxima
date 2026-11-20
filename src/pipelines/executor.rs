use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use crate::scenarios::cel::Program;
use proxima_http::templates::{TemplateContext, expand_strict, expand_value_strict};
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
use crate::scenarios::cel::evaluate_guard;
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
    guards: Arc<BTreeMap<String, Arc<Program>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageCompletion {
    Pending,
    Skipped,
    Failed,
    Declined,
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
    guard: Option<Arc<Program>>,
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

    fn links_for(&self, spec: &StageSpec, executor: &PipelineExecutor) -> StageLinks {
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
            pipes: executor.pipes.clone(),
            guard: executor.guards.get(&spec.name).cloned(),
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
        StageCompletion::Declined => None,
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
            guards: Arc::new(BTreeMap::new()),
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

    /// Supply the compiled `when` guards, keyed by stage name. The executor
    /// never compiles CEL: compile once with `scenarios::cel::compile_expression`
    /// and reuse. A run whose spec has a `when` stage without a guard is refused.
    #[must_use]
    pub fn with_guards(mut self, guards: BTreeMap<String, Arc<Program>>) -> Self {
        self.guards = Arc::new(guards);
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
        self.require_guards(&spec)?;
        let mut plan = RunPlan::new(&spec, pipeline_id, input)?;
        let started = Instant::now();
        self.record_started(pipeline_id, &spec).await?;
        let mut failures = self.run_stages(&mut plan).await?;
        let (stage_report, outcome) = plan.fold_outcome(&mut failures);
        self.record_ended(pipeline_id, started, &outcome).await?;
        Ok(plan.into_report(outcome, &stage_report, failures))
    }

    fn require_guards(&self, spec: &PipelineSpec) -> Result<(), ProximaError> {
        let unguarded = spec
            .stages
            .iter()
            .find(|stage| stage.when.is_some() && !self.guards.contains_key(&stage.name));
        match unguarded {
            Some(stage) => Err(ProximaError::Config(format!(
                "stage `{}` has `when` but no compiled guard was supplied",
                stage.name
            ))),
            None => Ok(()),
        }
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
        let links = plan.links_for(&spec, self);
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

    #[must_use]
    pub fn was_declined(&self) -> bool {
        matches!(self.completion, StageCompletion::Declined)
    }
}

enum StageOutcome {
    Completed { exit_code: Option<i32> },
    Succeeded { output: Value },
    Failed { reason: String },
    Skipped,
    Declined,
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
        Ok(StageOutcome::Declined) => (StageCompletion::Declined, None),
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
    if let Some(early) = pre_run_outcome(&spec.name, &mut links).await {
        return Ok(early);
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

// Some ends the stage before its body runs.
async fn pre_run_outcome(stage: &str, links: &mut StageLinks) -> Option<StageOutcome> {
    match settle_parents(&mut links.parents).await {
        ParentGate::Skipped => Some(StageOutcome::Skipped),
        ParentGate::Declined => Some(StageOutcome::Declined),
        ParentGate::Ready => guard_verdict(stage, links),
    }
}

enum ParentGate {
    Ready,
    Declined,
    Skipped,
}

// a failed, skipped or nonzero parent skips the stage and wins over a
// declined one wherever it sits in depends_on; a declined parent declines it.
async fn settle_parents(parents: &mut [ParentLink]) -> ParentGate {
    let mut gate = ParentGate::Ready;
    for parent in parents {
        match settled_completion(parent).await {
            Some(StageCompletion::Completed { exit_code: Some(0) }) => {}
            Some(StageCompletion::Declined) => gate = ParentGate::Declined,
            _ => return ParentGate::Skipped,
        }
    }
    gate
}

// None when the parent's task ended without ever settling.
async fn settled_completion(parent: &mut ParentLink) -> Option<StageCompletion> {
    while matches!(*parent.completion.borrow(), StageCompletion::Pending) {
        parent.completion.changed().await.ok()?;
    }
    let state = *parent.completion.borrow();
    Some(state)
}

// None runs the stage: no guard, or a guard that held.
fn guard_verdict(stage: &str, links: &StageLinks) -> Option<StageOutcome> {
    let guard = links.guard.as_ref()?;
    match eval_guard(guard, links) {
        Ok(true) => None,
        Ok(false) => Some(StageOutcome::Declined),
        Err(reason) => Some(StageOutcome::Failed {
            reason: format!("stage `{stage}` when: {reason}"),
        }),
    }
}

fn eval_guard(guard: &Program, links: &StageLinks) -> Result<bool, String> {
    let outputs: Vec<(&str, Arc<Value>)> = links
        .parents
        .iter()
        .filter_map(|parent| Some((parent.name.as_str(), parent.output.borrow().clone()?)))
        .collect();
    let mut bindings: Vec<(&str, &Value)> = vec![("input", links.input.as_ref())];
    bindings.extend(outputs.iter().map(|(name, value)| (*name, value.as_ref())));
    evaluate_guard(guard, &bindings)
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
    builder = match &request.body {
        None => builder,
        Some(Value::String(text)) => builder.body(Bytes::from(expand(text)?)),
        Some(value) => {
            let expanded = expand_value_strict(value, &context)?;
            let bytes = serde_json::to_vec(&expanded).map_err(|err| err.to_string())?;
            builder.body(Bytes::from(bytes))
        }
    };
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
    use crate::scenarios::cel::compile_expression;
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
            when: None,
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
        bodies: Arc<StdMutex<Vec<Bytes>>>,
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
            let received = request.payload.clone();
            self.probe.bodies.lock().expect("bodies").push(received);
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
            when: None,
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

    fn check_received(probe: &Probe, expected: &Value) -> Result<(), String> {
        let bodies = probe.bodies.lock().expect("bodies");
        let [body] = bodies.as_slice() else {
            return Err(format!("expected exactly one body, got {}", bodies.len()));
        };
        let parsed: Value = serde_json::from_slice(body).map_err(|err| err.to_string())?;
        if &parsed == expected {
            return Ok(());
        }
        Err(format!("received {parsed} differs from {expected}"))
    }

    fn payload_stage(body: Value) -> Vec<StageSpec> {
        let mut stage = pipe_stage("pr", "github", "/pr", &["ticket"]);
        if let StageBody::Pipe { request, .. } = &mut stage.body {
            request.body = Some(body);
        }
        vec![pipe_stage("ticket", "vault", "/PROXIMA-24/state.json", &[]), stage]
    }

    async fn run_payload(body: Value) -> (PipelineRunReport, Probe) {
        let (ticket, _) = stub(200, TICKET_BODY);
        let (github, probe) = stub(200, b"{}");
        let pipes = vec![("vault", ticket), ("github", github)];
        let report = run_pipes(
            payload_stage(body),
            pipes,
            json!({"ticket": "PROXIMA-24"}),
        )
        .await;
        (report, probe)
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

    const NO_PR_BODY: &[u8] = br#"{"repo":"brianbruggeman/proxima","pr":null}"#;
    const HAS_PR: &str = "ticket.pr != null";
    const NO_PR: &str = "ticket.pr == null";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Want {
        Completed,
        Declined,
        Skipped,
        Failed,
    }

    struct Branch<'a> {
        stages: &'a [(&'a str, Want)],
        calls: &'a [(&'a str, usize)],
        run_failed: bool,
    }

    fn want_of(completion: StageCompletion) -> Want {
        match completion {
            StageCompletion::Declined => Want::Declined,
            StageCompletion::Skipped => Want::Skipped,
            StageCompletion::Failed => Want::Failed,
            StageCompletion::Completed { .. } | StageCompletion::Pending => Want::Completed,
        }
    }

    fn check_branch(
        report: &PipelineRunReport,
        calls: &[(&str, usize)],
        expected: &Branch<'_>,
    ) -> Result<(), String> {
        for (name, want) in expected.stages {
            let got = report.stages.get(*name).map(|stage| want_of(stage.completion));
            if got != Some(*want) {
                return Err(format!("stage `{name}` is {got:?}, wanted {want:?}"));
            }
        }
        if calls != expected.calls {
            return Err(format!("calls {calls:?}, wanted {:?}", expected.calls));
        }
        let run_failed = matches!(report.outcome, PipelineOutcome::Failed { .. });
        if run_failed != expected.run_failed {
            return Err(format!("run outcome {:?}", report.outcome));
        }
        Ok(())
    }

    fn tally(probes: &[(&'static str, &Probe)]) -> Vec<(&'static str, usize)> {
        let count = |probe: &Probe| probe.calls.load(Ordering::SeqCst);
        probes.iter().map(|(name, probe)| (*name, count(probe))).collect()
    }

    fn guarded(stage: StageSpec, when: &str) -> StageSpec {
        StageSpec {
            when: Some(when.into()),
            ..stage
        }
    }

    fn compiled(stages: &[StageSpec]) -> BTreeMap<String, Arc<Program>> {
        stages
            .iter()
            .filter_map(|stage| {
                let program = compile_expression(stage.when.as_deref()?).expect("compile");
                Some((stage.name.clone(), Arc::new(program)))
            })
            .collect()
    }

    async fn run_guarded(
        stages: Vec<StageSpec>,
        pipes: Vec<(&str, PipeHandle)>,
        guards: BTreeMap<String, Arc<Program>>,
        input: Value,
    ) -> Result<PipelineRunReport, ProximaError> {
        let sink: DynRecordingSink = Arc::new(MemorySink::default());
        let pipes = pipes
            .into_iter()
            .map(|(name, handle)| (name.to_string(), handle))
            .collect();
        let executor = PipelineExecutor::new(sink)
            .with_pipes(pipes)
            .with_guards(guards);
        let spec = PipelineSpec { name: None, stages };
        executor.run_with_input(spec, input).await
    }

    struct Rig {
        report: PipelineRunReport,
        github: Probe,
        llm: Probe,
    }

    impl Rig {
        fn calls(&self) -> Vec<(&'static str, usize)> {
            tally(&[("github", &self.github), ("llm", &self.llm)])
        }
    }

    // ticket -> pr(when) plus extra stages; github and llm count their calls.
    async fn run_rig(ticket_body: &'static [u8], pr_when: &str, extra: Vec<StageSpec>) -> Rig {
        let (ticket, _) = stub(200, ticket_body);
        let (github, github_probe) = stub(200, b"{}");
        let (llm, llm_probe) = stub(200, b"{}");
        let (broken, _) = stub(500, b"boom");
        let pr = pipe_stage("pr", "github", PR_PATH, &["ticket"]);
        let mut stages = vec![
            pipe_stage("ticket", "vault", "/PROXIMA-24/state.json", &[]),
            guarded(pr, pr_when),
        ];
        stages.extend(extra);
        let guards = compiled(&stages);
        let pipes = vec![("vault", ticket), ("github", github), ("llm", llm), ("broken", broken)];
        let report = run_guarded(stages, pipes, guards, json!({"ticket": "PROXIMA-24"}))
            .await
            .expect("run");
        Rig {
            report,
            github: github_probe,
            llm: llm_probe,
        }
    }

    fn broken_stage() -> StageSpec {
        pipe_stage("broken", "broken", "/x", &["ticket"])
    }

    fn join_stage(parents: &[&str]) -> StageSpec {
        pipe_stage("join", "llm", "/join", parents)
    }

    fn process_guarded(script: &str, when: &str) -> StageSpec {
        guarded(shell_stage("proc", script, &[]), when)
    }

    async fn run_process_guard(script: &str, input: Value) -> PipelineRunReport {
        let stages = vec![process_guarded(script, "input.go == true")];
        let guards = compiled(&stages);
        run_guarded(stages, vec![], guards, input)
            .await
            .expect("run")
    }

    async fn check_join(parents: &[&str], extra: Vec<StageSpec>, want: &[(&str, Want)]) {
        let mut stages = vec![join_stage(parents)];
        stages.extend(extra);
        let rig = run_rig(NO_PR_BODY, HAS_PR, stages).await;
        let expected = Branch {
            stages: want,
            calls: &[("github", 0), ("llm", 0)],
            run_failed: true,
        };
        assert_eq!(check_branch(&rig.report, &rig.calls(), &expected), Ok(()));
    }

    #[proxima::test]
    async fn guard_runs_pipe_stage_when_ticket_has_a_pr() {
        let rig = run_rig(TICKET_BODY, HAS_PR, vec![]).await;
        let expected = Branch {
            stages: &[("ticket", Want::Completed), ("pr", Want::Completed)],
            calls: &[("github", 1), ("llm", 0)],
            run_failed: false,
        };
        assert_eq!(check_branch(&rig.report, &rig.calls(), &expected), Ok(()));
    }

    #[proxima::test]
    async fn guard_runs_process_stage_when_input_says_go() {
        let report = run_process_guard("exit 0", json!({"go": true})).await;
        let expected = Branch {
            stages: &[("proc", Want::Completed)],
            calls: &[],
            run_failed: false,
        };
        assert_eq!(check_branch(&report, &[], &expected), Ok(()));
        assert_eq!(report.stages["proc"].exit_code(), Some(Some(0)));
    }

    #[proxima::test]
    async fn guard_declines_pipe_stage_when_ticket_has_no_pr() {
        let rig = run_rig(NO_PR_BODY, HAS_PR, vec![]).await;
        let expected = Branch {
            stages: &[("pr", Want::Declined)],
            calls: &[("github", 0), ("llm", 0)],
            run_failed: false,
        };
        assert_eq!(check_branch(&rig.report, &rig.calls(), &expected), Ok(()));
    }

    #[proxima::test]
    async fn guard_declines_child_of_declined_stage_without_evaluating_its_own_when() {
        let review = guarded(pipe_stage("review", "llm", "/r", &["pr"]), "ticket.missing == 1");
        let rig = run_rig(NO_PR_BODY, HAS_PR, vec![review]).await;
        let expected = Branch {
            stages: &[("pr", Want::Declined), ("review", Want::Declined)],
            calls: &[("github", 0), ("llm", 0)],
            run_failed: false,
        };
        assert_eq!(check_branch(&rig.report, &rig.calls(), &expected), Ok(()));
    }

    #[proxima::test]
    async fn guard_declines_never_hides_a_failed_parent_declined_first() {
        let want = [("pr", Want::Declined), ("broken", Want::Failed), ("join", Want::Skipped)];
        check_join(&["pr", "broken"], vec![broken_stage()], &want).await;
    }

    #[proxima::test]
    async fn guard_declines_never_hides_a_failed_parent_declined_last() {
        let want = [("pr", Want::Declined), ("broken", Want::Failed), ("join", Want::Skipped)];
        check_join(&["broken", "pr"], vec![broken_stage()], &want).await;
    }

    #[proxima::test]
    async fn guard_declines_process_stage_and_never_spawns_it() {
        let marker_dir = TempDir::new().expect("tempdir");
        let marker = marker_dir.path().join("ran");
        let script = format!("touch {}", marker.display());
        let report = run_process_guard(&script, json!({"go": false})).await;
        let expected = Branch {
            stages: &[("proc", Want::Declined)],
            calls: &[],
            run_failed: false,
        };
        assert_eq!(check_branch(&report, &[], &expected), Ok(()));
        assert!(!marker.exists(), "declined process must not run");
    }

    #[proxima::test]
    async fn guard_declines_loses_to_a_skipped_parent() {
        let after = pipe_stage("after", "llm", "/after", &["broken"]);
        let want = [("pr", Want::Declined), ("after", Want::Skipped), ("join", Want::Skipped)];
        check_join(&["pr", "after"], vec![broken_stage(), after], &want).await;
    }

    #[proxima::test]
    async fn guard_declines_loses_to_a_nonzero_exit_parent() {
        let exit3 = shell_stage("exit3", "exit 3", &[]);
        let want = [("pr", Want::Declined), ("exit3", Want::Completed), ("join", Want::Skipped)];
        check_join(&["pr", "exit3"], vec![exit3], &want).await;
    }

    async fn failed_reason(ticket_body: &'static [u8], when: &str) -> String {
        let rig = run_rig(ticket_body, when, vec![]).await;
        let expected = Branch {
            stages: &[("pr", Want::Failed)],
            calls: &[("github", 0), ("llm", 0)],
            run_failed: true,
        };
        assert_eq!(check_branch(&rig.report, &rig.calls(), &expected), Ok(()));
        rig.report.failures["pr"].clone()
    }

    #[proxima::test]
    async fn guard_eval_fails_on_an_undeclared_field() {
        let reason = failed_reason(TICKET_BODY, "ticket.missing == 1").await;
        assert!(reason.starts_with("stage `pr` when"), "{reason}");
    }

    #[proxima::test]
    async fn guard_eval_fails_on_a_non_bool_result() {
        let reason = failed_reason(TICKET_BODY, "ticket.repo").await;
        assert!(reason.starts_with("stage `pr` when"), "{reason}");
        assert!(reason.contains("bool"), "{reason}");
    }

    #[proxima::test]
    async fn guard_eval_fails_on_a_parent_without_output() {
        let prep = shell_stage("prep", "true", &[]);
        let child = guarded(pipe_stage("child", "github", "/c", &["prep"]), "prep == null");
        let stages = vec![prep, child];
        let guards = compiled(&stages);
        let (github, probe) = stub(200, b"{}");
        let report = run_guarded(stages, vec![("github", github)], guards, Value::Null)
            .await
            .expect("run");
        let expected = Branch {
            stages: &[("prep", Want::Completed), ("child", Want::Failed)],
            calls: &[("github", 0)],
            run_failed: true,
        };
        assert_eq!(check_branch(&report, &tally(&[("github", &probe)]), &expected), Ok(()));
        assert!(report.failures["child"].starts_with("stage `child` when"));
    }

    #[proxima::test]
    async fn guard_compiled_run_without_supplied_guard_is_refused() {
        let (ticket, ticket_probe) = stub(200, TICKET_BODY);
        let (github, github_probe) = stub(200, b"{}");
        let pr = guarded(pipe_stage("pr", "github", PR_PATH, &["ticket"]), HAS_PR);
        let stages = vec![pipe_stage("ticket", "vault", "/t", &[]), pr];
        let pipes = vec![("vault", ticket), ("github", github)];
        let outcome = run_guarded(stages, pipes, BTreeMap::new(), Value::Null).await;
        let message = outcome.expect_err("refused").to_string();
        assert!(message.contains("stage `pr`") && message.contains("guard"), "{message}");
        let calls = tally(&[("vault", &ticket_probe), ("github", &github_probe)]);
        assert_eq!(calls, [("vault", 0), ("github", 0)]);
    }

    #[proxima::test]
    async fn guard_compiled_supplied_program_decides_not_the_when_string() {
        let (ticket, _) = stub(200, TICKET_BODY);
        let (github, probe) = stub(200, b"{}");
        let pr = guarded(pipe_stage("pr", "github", PR_PATH, &["ticket"]), "ticket.pr !=");
        let stages = vec![pipe_stage("ticket", "vault", "/t", &[]), pr];
        let supplied = compile_expression(NO_PR).expect("compile");
        let guards = BTreeMap::from([("pr".to_string(), Arc::new(supplied))]);
        let pipes = vec![("vault", ticket), ("github", github)];
        let report = run_guarded(stages, pipes, guards, Value::Null)
            .await
            .expect("run");
        let expected = Branch {
            stages: &[("pr", Want::Declined)],
            calls: &[("github", 0)],
            run_failed: false,
        };
        assert_eq!(check_branch(&report, &tally(&[("github", &probe)]), &expected), Ok(()));
    }

    fn hand_report(pr: StageCompletion, outcome: PipelineOutcome) -> PipelineRunReport {
        let stage = |completion| StageReport {
            id: InteractionId::new(),
            completion,
        };
        PipelineRunReport {
            pipeline_id: InteractionId::new(),
            outcome,
            stages: BTreeMap::from([
                ("ticket".to_string(), stage(StageCompletion::Completed { exit_code: Some(0) })),
                ("pr".to_string(), stage(pr)),
            ]),
            outputs: BTreeMap::new(),
            failures: BTreeMap::new(),
        }
    }

    fn declined_branch() -> Branch<'static> {
        Branch {
            stages: &[("ticket", Want::Completed), ("pr", Want::Declined)],
            calls: &[("github", 0)],
            run_failed: false,
        }
    }

    #[proxima::test]
    async fn ctl_branch_hand_built_declined_report_passes_the_check() {
        let report = hand_report(StageCompletion::Declined, PipelineOutcome::Completed);
        assert_eq!(check_branch(&report, &[("github", 0)], &declined_branch()), Ok(()));
    }

    #[proxima::test]
    async fn ctl_branch_skipped_instead_of_declined_is_rejected() {
        let report = hand_report(StageCompletion::Skipped, PipelineOutcome::Completed);
        assert!(check_branch(&report, &[("github", 0)], &declined_branch()).is_err());
    }

    #[proxima::test]
    async fn ctl_branch_failed_outcome_is_rejected() {
        let outcome = PipelineOutcome::Failed { reason: "stage `pr` failed".into() };
        let report = hand_report(StageCompletion::Declined, outcome);
        assert!(check_branch(&report, &[("github", 0)], &declined_branch()).is_err());
    }

    #[proxima::test]
    async fn stage_payload_whole_placeholder_keeps_the_number() {
        let (_, probe) = run_payload(json!({"pr": "{{body.ticket.pr}}"})).await;
        assert_eq!(check_received(&probe, &json!({"pr": 42})), Ok(()));
    }

    #[proxima::test]
    async fn stage_payload_embedded_placeholders_render_as_text() {
        let body = json!({"title": "#{{body.ticket.pr}} in {{body.ticket.repo}}"});
        let (_, probe) = run_payload(body).await;
        let expected = json!({"title": "#42 in brianbruggeman/proxima"});
        assert_eq!(check_received(&probe, &expected), Ok(()));
    }

    #[proxima::test]
    async fn stage_payload_whole_placeholder_takes_an_object() {
        let (_, probe) = run_payload(json!({"ticket": "{{body.ticket}}"})).await;
        let expected = json!({"ticket": {"pr": 42, "repo": "brianbruggeman/proxima"}});
        assert_eq!(check_received(&probe, &expected), Ok(()));
    }

    #[proxima::test]
    async fn stage_payload_plain_leaves_pass_unchanged() {
        let body = json!({"n": 3, "ok": true, "none": null});
        let (_, probe) = run_payload(body.clone()).await;
        assert_eq!(check_received(&probe, &body), Ok(()));
    }

    #[proxima::test]
    async fn stage_payload_nested_list_expands_every_leaf() {
        let body = json!(["{{body.in.ticket}}", {"deep": ["{{body.ticket.repo}}"]}]);
        let (_, probe) = run_payload(body).await;
        let expected = json!(["PROXIMA-24", {"deep": ["brianbruggeman/proxima"]}]);
        assert_eq!(check_received(&probe, &expected), Ok(()));
    }

    #[proxima::test]
    async fn stage_payload_missing_path_fails_without_calling_the_pipe() {
        let (report, probe) = run_payload(json!({"x": "{{body.missing.path}}"})).await;
        let observed = observe(&probe, &report, "pr");
        assert_eq!(
            check_template(&observed, &Expect::Missing("body.missing.path")),
            Ok(())
        );
    }

    #[proxima::test]
    async fn stage_payload_padded_braces_resolve_and_outside_text_makes_text() {
        let body = json!({"a": "{{ body.ticket.pr }}", "b": " {{body.ticket.pr}}"});
        let (_, probe) = run_payload(body).await;
        assert_eq!(
            check_received(&probe, &json!({"a": 42, "b": " 42"})),
            Ok(())
        );
    }

    #[proxima::test]
    async fn ctl_payload_string_body_keeps_the_text_path() {
        let body = Value::String(r#"{"pr":"{{body.ticket.pr}}"}"#.to_string());
        let (_, probe) = run_payload(body).await;
        let received = probe.bodies.lock().expect("bodies").clone();
        assert_eq!(received, vec![Bytes::from_static(br#"{"pr":"42"}"#)]);
        assert_eq!(check_received(&probe, &json!({"pr": "42"})), Ok(()));
        assert!(check_received(&probe, &json!({"pr": 42})).is_err());
    }
}
