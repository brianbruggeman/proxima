use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use crate::error::ProximaError;
use crate::load::{LoadContext, Spec, load};
use crate::request::{Request, Response};
use crate::runtime::{CoreId, Runtime};
use crate::scenarios::spec::{
    CompareOp, Expectation, OrchestrationMode, Scenario, ScenarioPipeSpec, WorkloadSpec,
};
// orchestrator stays Send-only: scenario bootstrap builds Pipes through
// the factory registry, which yields `PipeHandle` (Arc<dyn DynPipe>);
// per-thread Pipes aren't constructable from JSON specs in v1, and
// scenarios are themselves a cross-thread harness. Stage 10 fork lives in
// the wrappers a scenario *composes*, not in the orchestrator itself.
use proxima_primitives::pipe::SendPipe;

use crate::pipe::PipeHandle;
use crate::scenarios::spec::WorkloadMode;
use crate::telemetry::{Labels, Metrics, MetricsSnapshot, TelemetryHandle};

pub struct ScenarioReport {
    pub completed: u64,
    pub successes: u64,
    pub failures: u64,
    pub failed_expectations: Vec<String>,
    pub metrics_snapshot: MetricsSnapshot,
    /// per-second `MetricsSnapshot` windows captured during an open-loop
    /// run. empty for closed-loop scenarios (existing behavior).
    pub windows: Vec<MetricsSnapshot>,
}

impl ScenarioReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failed_expectations.is_empty()
    }
}

pub async fn run_scenario(
    scenario: &Scenario,
    context: &LoadContext,
) -> Result<ScenarioReport, ProximaError> {
    run_scenario_with_sink(scenario, context, None).await
}

/// `run_scenario` with an optional sink for per-second `MetricsSnapshot`s
/// emitted during open-loop runs. CLI uses this to stream live windows
/// to a printer thread; in-process tests pass `None`.
///
/// `sink` is ignored for closed-loop scenarios (no per-second windows).
pub async fn run_scenario_with_sink(
    scenario: &Scenario,
    context: &LoadContext,
    sink: Option<std::sync::mpsc::Sender<MetricsSnapshot>>,
) -> Result<ScenarioReport, ProximaError> {
    match scenario.mode {
        OrchestrationMode::InProcess => run_in_process(scenario, context, sink).await,
        OrchestrationMode::Isolated => run_isolated(scenario, context).await,
    }
}

async fn run_in_process(
    scenario: &Scenario,
    context: &LoadContext,
    sink: Option<std::sync::mpsc::Sender<MetricsSnapshot>>,
) -> Result<ScenarioReport, ProximaError> {
    let pipes = build_pipes(scenario, context).await?;
    let target = pipes.get(&scenario.workload.target_pipe).ok_or_else(|| {
        ProximaError::Config(format!(
            "scenario workload targets unknown pipe `{}`",
            scenario.workload.target_pipe,
        ))
    })?;
    let (completed, successes, failures, windows) = match scenario.workload.mode()? {
        WorkloadMode::ClosedLoop => {
            // closed-loop runs inline on the caller — no runtime needed.
            let (completed, successes, failures) = drive_workload(
                target.clone(),
                &scenario.workload,
                context.telemetry.clone(),
            )
            .await?;
            (completed, successes, failures, Vec::new())
        }
        WorkloadMode::OpenLoop => {
            let metrics = context.metrics.as_ref().cloned().ok_or_else(|| {
                ProximaError::Config(
                    "open-loop scenarios require LoadContext.metrics to be set".into(),
                )
            })?;
            // open-loop awaits `timer_at`, which prime serves only on its own
            // worker, so it runs on a worker core. `offline_runtime` picks prime
            // or tokio per the build's runtime features.
            let runtime = crate::app::offline_runtime()?;
            let summary = drive_workload_open(
                target.clone(),
                &scenario.workload,
                context.telemetry.clone(),
                metrics,
                runtime,
                sink,
            )
            .await?;
            (
                summary.completed,
                summary.successes,
                summary.failures,
                summary.windows,
            )
        }
    };
    let snapshot = context
        .metrics
        .as_ref()
        .map(|metrics| metrics.snapshot())
        .unwrap_or_else(|| MetricsSnapshot {
            counters: Vec::new(),
            gauges: Vec::new(),
            histograms: Vec::new(),
        });
    let failed_expectations = evaluate_expectations(
        &scenario.expectations,
        &snapshot,
        successes,
        completed,
        failures,
    );
    Ok(ScenarioReport {
        completed,
        successes,
        failures,
        failed_expectations,
        metrics_snapshot: snapshot,
        windows,
    })
}

/// Run a scenario driver future on a runtime worker core and bridge its result
/// back to the (possibly outer-tokio) calling thread.
///
/// The open-loop driver awaits [`Runtime::timer_at`], which `PrimeRuntime` only
/// serves on one of its own worker threads — so the driver cannot run on the
/// arbitrary calling thread. We inject it onto core 0 via
/// [`Runtime::spawn_factory_on_core`] (the same per-core-loop-injection the
/// listeners use, see `App::run_until_signal`), where `timer_at` and
/// `spawn_on_current_core` are both valid, and ship the `Out` back over a
/// `oneshot`. This makes the driver runtime-agnostic: it runs unchanged on
/// `TokioPerCoreRuntime` and the prime-native `PrimeRuntime`.
async fn drive_on_core<Build, Fut, Out>(
    runtime: Arc<dyn Runtime>,
    build: Build,
) -> Result<Out, ProximaError>
where
    Build: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Out> + 'static,
    Out: Send + 'static,
{
    let (result_tx, result_rx) = futures::channel::oneshot::channel::<Out>();
    let factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static> =
        Box::new(move || {
            Box::pin(async move {
                let _ = result_tx.send(build().await);
            })
        });
    runtime
        .spawn_factory_on_core(CoreId(0), factory)
        .map_err(|err| ProximaError::Config(format!("spawn scenario driver on core 0: {err}")))?;
    result_rx.await.map_err(|_| {
        ProximaError::Upstream("scenario driver core dropped before completion".into())
    })
}

async fn run_isolated(
    scenario: &Scenario,
    context: &LoadContext,
) -> Result<ScenarioReport, ProximaError> {
    let target = scenario
        .pipes
        .iter()
        .find(|pipe| pipe.name == scenario.workload.target_pipe)
        .ok_or_else(|| {
            ProximaError::Config(format!(
                "scenario workload targets unknown pipe `{}`",
                scenario.workload.target_pipe,
            ))
        })?;
    let runtime_dir = TempDir::new().map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("scenario tempdir: {err}")))
    })?;
    let mut children: Vec<IsolatedChild> = Vec::with_capacity(scenario.pipes.len());
    let mut target_addr: Option<SocketAddr> = None;
    for pipe in &scenario.pipes {
        let mut child = spawn_isolated_pipe(pipe, runtime_dir.path()).await?;
        if pipe.name == target.name {
            target_addr = Some(child.addr);
        }
        // owning the Child keeps the process alive until end-of-scope; recorded for cleanup.
        children.push(IsolatedChild {
            process: child.process.take(),
        });
    }
    let target_addr = target_addr.ok_or_else(|| {
        ProximaError::Config("isolated scenario produced no child for target pipe".into())
    })?;
    let outcome = drive_isolated_workload(target_addr, &scenario.workload, context).await;
    // best-effort cleanup; SIGKILL is safe since children are scenario-only daemons.
    for child in &mut children {
        if let Some(mut process) = child.process.take() {
            let _ = process.start_kill();
            let _ = process.wait().await;
        }
    }
    let (completed, successes, failures) = outcome?;
    let snapshot = context
        .metrics
        .as_ref()
        .map(|metrics| metrics.snapshot())
        .unwrap_or_else(|| MetricsSnapshot {
            counters: Vec::new(),
            gauges: Vec::new(),
            histograms: Vec::new(),
        });
    let failed_expectations = evaluate_expectations(
        &scenario.expectations,
        &snapshot,
        successes,
        completed,
        failures,
    );
    Ok(ScenarioReport {
        completed,
        successes,
        failures,
        failed_expectations,
        metrics_snapshot: snapshot,
        windows: Vec::new(),
    })
}

struct IsolatedChild {
    process: Option<Child>,
}

struct SpawnedChild {
    addr: SocketAddr,
    process: ChildHolder,
}

struct ChildHolder {
    inner: Option<Child>,
}

impl ChildHolder {
    fn take(&mut self) -> Option<Child> {
        self.inner.take()
    }
}

async fn spawn_isolated_pipe(
    pipe: &ScenarioPipeSpec,
    runtime_dir: &std::path::Path,
) -> Result<SpawnedChild, ProximaError> {
    // 1. write the pipe spec to a temp toml file under the scenario runtime dir.
    let mut toml_table = serde_json::Map::new();
    if let Some(object) = pipe.spec.as_object() {
        for (key, value) in object {
            toml_table.insert(key.clone(), value.clone());
        }
    } else {
        return Err(ProximaError::Config(format!(
            "isolated pipe `{}` spec must be an object",
            pipe.name
        )));
    }
    if !toml_table.contains_key("name") {
        toml_table.insert("name".into(), serde_json::Value::String(pipe.name.clone()));
    }
    let toml_string = json_value_to_toml(&serde_json::Value::Object(toml_table))?;
    let config_path: PathBuf = runtime_dir.join(format!("{}.toml", pipe.name));
    tokio::fs::write(&config_path, toml_string)
        .await
        .map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("write child toml: {err}")))
        })?;
    // 2. pre-bind a port to grab a free address, drop the listener, hand the addr to the child.
    let pre_bind = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("pre-bind: {err}"))))?;
    let addr = pre_bind
        .local_addr()
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("local_addr: {err}"))))?;
    drop(pre_bind);
    // 3. spawn `proxima serve --config <toml> --addr <addr>` and wait for the READY line.
    let bin = std::env::var("PROXIMA_CLI")
        .map(PathBuf::from)
        .map_err(|_| {
            ProximaError::Config(
                "isolated scenarios require the PROXIMA_CLI env var to point at a `proxima` binary"
                    .into(),
            )
        })?;
    let mut command = Command::new(&bin);
    command
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--addr")
        .arg(addr.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = command.spawn().map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("spawn proxima serve: {err}")))
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        ProximaError::Io(std::io::Error::other(
            "child has no stdout pipe".to_string(),
        ))
    })?;
    wait_for_ready(stdout, addr).await?;
    Ok(SpawnedChild {
        addr,
        process: ChildHolder { inner: Some(child) },
    })
}

async fn wait_for_ready(
    stdout: tokio::process::ChildStdout,
    addr: SocketAddr,
) -> Result<(), ProximaError> {
    let mut reader = BufReader::new(stdout).lines();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let line_future = reader.next_line();
        let line_result = match tokio::time::timeout(Duration::from_millis(500), line_future).await
        {
            Ok(line_outcome) => line_outcome,
            Err(_) if std::time::Instant::now() < deadline => continue,
            Err(_) => {
                return Err(ProximaError::Io(std::io::Error::other(format!(
                    "proxima serve at {addr} did not announce READY within 10s"
                ))));
            }
        };
        match line_result {
            Ok(Some(line)) => {
                if line.starts_with("READY ") {
                    return Ok(());
                }
            }
            Ok(None) => {
                return Err(ProximaError::Io(std::io::Error::other(format!(
                    "proxima serve at {addr} closed stdout before READY"
                ))));
            }
            Err(err) => {
                return Err(ProximaError::Io(std::io::Error::other(format!(
                    "read child stdout: {err}"
                ))));
            }
        }
    }
}

fn json_value_to_toml(value: &serde_json::Value) -> Result<String, ProximaError> {
    let toml_value = json_to_toml(value);
    toml::to_string(&toml_value)
        .map_err(|err| ProximaError::Config(format!("encode pipe toml: {err}")))
}

fn json_to_toml(value: &serde_json::Value) -> toml::Value {
    match value {
        serde_json::Value::Null => toml::Value::String(String::new()),
        serde_json::Value::Bool(flag) => toml::Value::Boolean(*flag),
        serde_json::Value::Number(number) => {
            if let Some(int_value) = number.as_i64() {
                toml::Value::Integer(int_value)
            } else if let Some(float_value) = number.as_f64() {
                toml::Value::Float(float_value)
            } else {
                toml::Value::String(number.to_string())
            }
        }
        serde_json::Value::String(text) => toml::Value::String(text.clone()),
        serde_json::Value::Array(items) => {
            toml::Value::Array(items.iter().map(json_to_toml).collect())
        }
        serde_json::Value::Object(map) => {
            let mut table = toml::map::Map::new();
            for (key, sub) in map {
                table.insert(key.clone(), json_to_toml(sub));
            }
            toml::Value::Table(table)
        }
    }
}

async fn drive_isolated_workload(
    target_addr: SocketAddr,
    workload: &WorkloadSpec,
    _context: &LoadContext,
) -> Result<(u64, u64, u64), ProximaError> {
    // The target is an EXTERNAL `proxima serve` child process, so drive it with
    // a plain tokio HTTP/1.1 client, NOT proxima's prime `http` upstream: that
    // connector must be constructed on a prime worker core (CURRENT_REACTOR),
    // but the scenario driver runs off-core, so the prime upstream cannot
    // connect from here. A raw client has no such requirement.
    let total = workload.requests.max(1) as u64;
    let concurrency = workload.concurrency.max(1);
    let successes = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let issued = Arc::new(AtomicU64::new(0));

    let workers = (0..concurrency).map(|_| {
        let workload = workload.clone();
        let successes = successes.clone();
        let failures = failures.clone();
        let issued = issued.clone();
        async move {
            loop {
                let next = issued.fetch_add(1, Ordering::SeqCst);
                if next >= total {
                    break;
                }
                match isolated_http_request(target_addr, &workload).await {
                    Ok(status) if (200..400).contains(&status) => {
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });
    futures::future::join_all(workers).await;

    Ok((
        successes.load(Ordering::Relaxed) + failures.load(Ordering::Relaxed),
        successes.load(Ordering::Relaxed),
        failures.load(Ordering::Relaxed),
    ))
}

// minimal HTTP/1.1 client for driving an isolated child: one request per
// connection (Connection: close), returns the response status line's code.
async fn isolated_http_request(
    addr: SocketAddr,
    workload: &WorkloadSpec,
) -> Result<u16, ProximaError> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut target = workload.path.clone();
    if !workload.query.is_empty() {
        let pairs: Vec<String> = workload
            .query
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect();
        target = format!("{target}?{}", pairs.join("&"));
    }
    let body = workload.body.clone().unwrap_or_default();
    let mut request = format!(
        "{method} {target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {len}\r\n",
        method = workload.method,
        len = body.len(),
    );
    for (name, value) in &workload.headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");

    let mut stream = tokio::net::TcpStream::connect(addr)
        .await
        .map_err(|err| ProximaError::Upstream(format!("isolated connect {addr}: {err}")))?;
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|err| ProximaError::Upstream(format!("isolated write head: {err}")))?;
    if !body.is_empty() {
        stream
            .write_all(&body)
            .await
            .map_err(|err| ProximaError::Upstream(format!("isolated write body: {err}")))?;
    }
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|err| ProximaError::Upstream(format!("isolated read: {err}")))?;

    // status line: `HTTP/1.1 <code> <reason>`
    let head = response.split(|byte| *byte == b'\r').next().unwrap_or(&[]);
    let text = core::str::from_utf8(head)
        .map_err(|_| ProximaError::Upstream("isolated response status not utf-8".into()))?;
    text.split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| ProximaError::Upstream(format!("isolated bad status line: {text:?}")))
}

async fn build_pipes(
    scenario: &Scenario,
    context: &LoadContext,
) -> Result<HashMap<String, PipeHandle>, ProximaError> {
    let mut handles: HashMap<String, PipeHandle> = HashMap::with_capacity(scenario.pipes.len());
    for pipe in &scenario.pipes {
        let handle = load(Spec::Inline(pipe.spec.clone()), context).await?;
        handles.insert(pipe.name.clone(), handle);
    }
    Ok(handles)
}

pub async fn drive_workload(
    pipe: PipeHandle,
    workload: &WorkloadSpec,
    telemetry: TelemetryHandle,
) -> Result<(u64, u64, u64), ProximaError> {
    let total = workload.requests.max(1) as u64;
    let concurrency = workload.concurrency.max(1);
    let successes = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let issued = Arc::new(AtomicU64::new(0));

    // closed-loop has no cadence timer, so it runs INLINE on the caller's
    // executor — N concurrent worker futures `join_all`'d, no spawn, no
    // cross-thread dispatch (the same shape as `drive_workload_isolated`).
    // it does NOT go through `drive_on_core`: that injects onto a worker via
    // `spawn_factory_on_core` + a `oneshot`, which is pure overhead for a
    // timer-free loop (~2x on microsecond runs).
    let workers = (0..concurrency).map(|_| {
        let pipe = pipe.clone();
        let workload = workload.clone();
        let telemetry = telemetry.clone();
        let successes = successes.clone();
        let failures = failures.clone();
        let issued = issued.clone();
        async move {
            loop {
                let next = issued.fetch_add(1, Ordering::SeqCst);
                if next >= total {
                    break;
                }
                match call_one(&pipe, &workload, telemetry.clone()).await {
                    Ok(_) => {
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });
    futures::future::join_all(workers).await;

    let successes_count = successes.load(Ordering::Relaxed);
    let failures_count = failures.load(Ordering::Relaxed);
    Ok((
        successes_count + failures_count,
        successes_count,
        failures_count,
    ))
}

async fn call_one(
    pipe: &PipeHandle,
    workload: &WorkloadSpec,
    telemetry: TelemetryHandle,
) -> Result<Response<Bytes>, ProximaError> {
    let mut builder = Request::builder()
        .method(workload.method.as_str())
        .path(workload.path.clone())
        .telemetry(telemetry);
    for (name, value) in &workload.headers {
        builder = builder.header(name.clone(), value.clone());
    }
    for (name, value) in &workload.query {
        builder = builder.query_param(name.clone(), value.clone());
    }
    if let Some(body_bytes) = workload.body.as_ref() {
        builder = builder.body(body_bytes.clone());
    }
    let request = builder.build()?;
    let response = SendPipe::call(pipe, request).await?;
    if !(200..400).contains(&response.status) {
        return Err(ProximaError::Upstream(format!(
            "scenario request returned status {}",
            response.status
        )));
    }
    // drain the body so streaming write-back / recording / tap_complete callbacks fire.
    let status = response.status;
    let headers = response.metadata.clone();
    let drained = response.collect_body().await?;
    let mut rebuilt = Response::new(status).with_body(drained);
    for (name, value) in headers {
        rebuilt = rebuilt.with_header(name, value);
    }
    Ok(rebuilt)
}

fn evaluate_expectations(
    expectations: &[Expectation],
    snapshot: &MetricsSnapshot,
    successes: u64,
    completed: u64,
    failures_count: u64,
) -> Vec<String> {
    let mut failures = Vec::new();
    for expectation in expectations {
        match expectation {
            Expectation::Counter {
                metric,
                labels,
                op,
                expected,
            } => {
                let target_labels = labels_from_map(labels);
                let observed = snapshot
                    .counters
                    .iter()
                    .find(|(name, label_set, _)| name == metric && label_set == &target_labels)
                    .map(|(_, _, value)| *value)
                    .unwrap_or(0);
                if !compare(observed, *op, *expected) {
                    failures.push(format!(
                        "counter `{metric}` observed={observed} expected {op:?} {expected}"
                    ));
                }
            }
            Expectation::HistogramP99LeMs {
                metric,
                labels,
                max_ms,
            } => {
                let target_labels = labels_from_map(labels);
                let observed = snapshot
                    .histograms
                    .iter()
                    .find(|(name, label_set, _)| name == metric && label_set == &target_labels)
                    .map(|(_, _, summary)| summary.p99);
                match observed {
                    Some(p99) if p99 <= *max_ms => {}
                    Some(p99) => failures.push(format!(
                        "histogram `{metric}` p99={p99}ms exceeds max {max_ms}ms"
                    )),
                    None => failures.push(format!("histogram `{metric}` not present in snapshot")),
                }
            }
            Expectation::SuccessRateGe { ratio } => {
                if completed == 0 {
                    failures.push("success_rate undefined: no requests completed".into());
                    continue;
                }
                let observed = (successes as f64) / (completed as f64);
                if observed < *ratio {
                    failures.push(format!(
                        "success_rate observed={observed:.3} expected >= {ratio:.3}"
                    ));
                }
            }
            Expectation::Cel { expression } => {
                match crate::scenarios::cel::evaluate(
                    expression,
                    &CelBindings {
                        successes,
                        completed,
                        failures: failures_count,
                        snapshot,
                    },
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        failures.push(format!("cel expression `{expression}` evaluated to false"))
                    }
                    Err(error) => {
                        failures.push(format!("cel expression `{expression}` failed: {error}"))
                    }
                }
            }
            Expectation::Diff {
                identical,
                max_first_diff_offset: _,
            } => {
                let empty_labels = Labels::empty();
                let identical_count = snapshot
                    .counters
                    .iter()
                    .find(|(name, label_set, _)| {
                        name == "proxima.diff.identical_total" && label_set == &empty_labels
                    })
                    .map(|(_, _, value)| *value)
                    .unwrap_or(0);
                let divergent_count = snapshot
                    .counters
                    .iter()
                    .find(|(name, label_set, _)| {
                        name == "proxima.diff.divergent_total" && label_set == &empty_labels
                    })
                    .map(|(_, _, value)| *value)
                    .unwrap_or(0);
                if identical_count == 0 && divergent_count == 0 {
                    failures.push(
                        "diff expectation: no `proxima.diff.*_total` counters observed; \
                         was a `diff` middleware on the pipe path?"
                            .into(),
                    );
                    continue;
                }
                if *identical && divergent_count > 0 {
                    failures.push(format!(
                        "diff expectation: identical=true but observed \
                         {divergent_count} divergent calls"
                    ));
                }
                if !*identical && identical_count > 0 && divergent_count == 0 {
                    failures.push(
                        "diff expectation: identical=false but all observed calls were identical"
                            .into(),
                    );
                }
            }
        }
    }
    failures
}

pub struct CelBindings<'snapshot> {
    pub successes: u64,
    pub completed: u64,
    pub failures: u64,
    pub snapshot: &'snapshot MetricsSnapshot,
}

fn labels_from_map(map: &BTreeMap<String, String>) -> Labels {
    let pairs: Vec<(&str, &str)> = map
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    Labels::from_pairs(&pairs)
}

fn compare(observed: u64, op: CompareOp, expected: u64) -> bool {
    match op {
        CompareOp::Eq => observed == expected,
        CompareOp::Ge => observed >= expected,
        CompareOp::Le => observed <= expected,
    }
}

/// Summary of a single open-loop driver run. `windows` is a per-second
/// vector of cumulative `MetricsSnapshot`s captured during the run;
/// callers compute deltas by diffing adjacent windows.
#[derive(Debug, Clone)]
pub struct OpenLoopSummary {
    pub completed: u64,
    pub successes: u64,
    pub failures: u64,
    /// dispatches skipped because in-flight had hit the concurrency cap
    /// at tick time. surfaces saturation without stalling the cadence
    /// clock — at zero this means the driver hit `target_rps` cleanly.
    pub skipped_coordinated_omission: u64,
    pub windows: Vec<MetricsSnapshot>,
}

/// Open-loop sibling of [`drive_workload`]. Dispatches at the cadence
/// declared by `workload.target_rps + workload.duration` (or by
/// `workload.profile` when non-empty), bounded by `workload.concurrency`.
/// Records coordinated-omission-corrected latency under
/// `proxima.workload.co_latency_ms` (submit-time to response-time, including
/// queue delay). Emits per-second `MetricsSnapshot`s into `windows` and,
/// when `snapshot_sink` is `Some`, through the channel as well.
///
/// Cadence rides [`Runtime::timer_at`] — no `tokio::time::interval` /
/// `tokio::sync::Semaphore`. The whole driver runs on a runtime worker core
/// (via [`drive_on_core`]) and dispatches each request with
/// `Runtime::spawn_on_current_core`, so it is runtime-agnostic: identical
/// behaviour on `TokioPerCoreRuntime` and the prime-native `PrimeRuntime`.
pub async fn drive_workload_open(
    pipe: PipeHandle,
    workload: &WorkloadSpec,
    telemetry: TelemetryHandle,
    metrics: Arc<Metrics>,
    runtime: Arc<dyn Runtime>,
    snapshot_sink: Option<std::sync::mpsc::Sender<MetricsSnapshot>>,
) -> Result<OpenLoopSummary, ProximaError> {
    let schedule = resolve_schedule(workload)?;
    let concurrency_cap = workload.concurrency.max(1);
    let total_duration: Duration = schedule.iter().map(|step| step.duration).sum();

    let successes = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let skipped = Arc::new(AtomicU64::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let windows: Arc<Mutex<Vec<MetricsSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
    let workload_template = Arc::new(workload.clone());

    let run_start = Instant::now();
    let run_end = run_start + total_duration;

    drive_on_core(runtime.clone(), move || async move {
        let snapshot_done = spawn_snapshot_ticker(
            runtime.clone(),
            metrics.clone(),
            windows.clone(),
            snapshot_sink.clone(),
            run_start,
            run_end,
        );

        run_dispatch_loop(
            &runtime,
            &pipe,
            &workload_template,
            &telemetry,
            &successes,
            &failures,
            &skipped,
            &in_flight,
            concurrency_cap,
            run_start,
            &schedule,
        )
        .await;

        // dispatch can finish slightly before `run_end` (cadence rounds
        // down on the last tick). hold here until run_end so the snapshot
        // ticker has a chance to fire its final tick before drain + abort.
        if Instant::now() < run_end {
            runtime.timer_at(run_end).await;
        }

        let drain_until = run_end + Duration::from_secs(30);
        while in_flight.load(Ordering::Acquire) > 0 {
            if Instant::now() > drain_until {
                break;
            }
            runtime
                .timer_at(Instant::now() + Duration::from_millis(10))
                .await;
        }

        // the ticker loop exits on its own at `run_end`, which the drain
        // above has already passed; await its completion so its final tick
        // lands in `windows` before we snapshot.
        let _ = snapshot_done.await;

        // guarantee one final snapshot even for short runs where the
        // periodic ticker raced shutdown — windows always reflects the
        // final state after drain, and the sink sees the same snapshot
        // the summary records.
        let final_snapshot = metrics.snapshot();
        if let Ok(mut guard) = windows.lock() {
            guard.push(final_snapshot.clone());
        }
        if let Some(sender) = snapshot_sink.as_ref() {
            let _ = sender.send(final_snapshot);
        }

        let successes_count = successes.load(Ordering::Acquire);
        let failures_count = failures.load(Ordering::Acquire);
        let skipped_count = skipped.load(Ordering::Acquire);
        let windows_vec = windows
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        Ok::<OpenLoopSummary, ProximaError>(OpenLoopSummary {
            completed: successes_count + failures_count,
            successes: successes_count,
            failures: failures_count,
            skipped_coordinated_omission: skipped_count,
            windows: windows_vec,
        })
    })
    .await?
}

#[derive(Debug, Clone, Copy)]
struct ScheduleStep {
    rate: u64,
    duration: Duration,
}

fn resolve_schedule(workload: &WorkloadSpec) -> Result<Vec<ScheduleStep>, ProximaError> {
    if !workload.profile.is_empty() {
        return Ok(workload
            .profile
            .iter()
            .map(|step| ScheduleStep {
                rate: step.rate,
                duration: step.duration.as_duration(),
            })
            .collect());
    }
    let rate = workload
        .target_rps
        .ok_or_else(|| ProximaError::Config("open-loop workload missing `target_rps`".into()))?;
    let duration = workload
        .duration
        .ok_or_else(|| ProximaError::Config("open-loop workload missing `duration`".into()))?;
    Ok(vec![ScheduleStep {
        rate,
        duration: duration.as_duration(),
    }])
}

// spawns the per-second snapshot ticker on the current worker core and returns
// a receiver that fires once the ticker has emitted its final tick (at
// `run_end`). the caller awaits it instead of aborting a JoinHandle — the loop
// terminates on its own at `run_end`.
fn spawn_snapshot_ticker(
    runtime: Arc<dyn Runtime>,
    metrics: Arc<Metrics>,
    windows: Arc<Mutex<Vec<MetricsSnapshot>>>,
    snapshot_sink: Option<std::sync::mpsc::Sender<MetricsSnapshot>>,
    run_start: Instant,
    run_end: Instant,
) -> futures::channel::oneshot::Receiver<()> {
    let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
    let timer_runtime = runtime.clone();
    runtime.spawn_on_current_core(Box::pin(async move {
        let mut next = run_start + Duration::from_secs(1);
        while next <= run_end {
            timer_runtime.timer_at(next).await;
            let snapshot = metrics.snapshot();
            if let Ok(mut guard) = windows.lock() {
                guard.push(snapshot.clone());
            }
            if let Some(sender) = snapshot_sink.as_ref() {
                // receiver may have hung up; that's allowed — drop silently.
                let _ = sender.send(snapshot);
            }
            next += Duration::from_secs(1);
        }
        let _ = done_tx.send(());
    }));
    done_rx
}

#[allow(clippy::too_many_arguments)]
async fn run_dispatch_loop(
    runtime: &Arc<dyn Runtime>,
    pipe: &PipeHandle,
    workload: &Arc<WorkloadSpec>,
    telemetry: &TelemetryHandle,
    successes: &Arc<AtomicU64>,
    failures: &Arc<AtomicU64>,
    skipped: &Arc<AtomicU64>,
    in_flight: &Arc<AtomicUsize>,
    concurrency_cap: usize,
    run_start: Instant,
    schedule: &[ScheduleStep],
) {
    let mut step_start = run_start;
    for step in schedule {
        let step_end = step_start + step.duration;
        let period = cadence_period(step.rate);
        let mut next_dispatch = step_start;
        while next_dispatch < step_end {
            runtime.timer_at(next_dispatch).await;

            // CO timestamp captured at the *dispatch* point — includes any
            // queue delay if the dispatch had to wait. fires before the skip
            // check so skipped slots count toward the saturation surface.
            let submit_at = Instant::now();

            let current = in_flight.load(Ordering::Acquire);
            if current >= concurrency_cap {
                skipped.fetch_add(1, Ordering::Relaxed);
                next_dispatch += period;
                continue;
            }
            in_flight.fetch_add(1, Ordering::AcqRel);

            let pipe = pipe.clone();
            let workload = workload.clone();
            let telemetry_inner = telemetry.clone();
            let successes = successes.clone();
            let failures = failures.clone();
            let in_flight = in_flight.clone();

            runtime.spawn_on_current_core(Box::pin(async move {
                let outcome = call_one(&pipe, &workload, telemetry_inner.clone()).await;
                let co_latency_ms = submit_at.elapsed().as_secs_f64() * 1_000.0;
                telemetry_inner.histogram_record(
                    "proxima.workload.co_latency_ms",
                    &Labels::empty(),
                    co_latency_ms,
                );
                match outcome {
                    Ok(_) => {
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
                in_flight.fetch_sub(1, Ordering::AcqRel);
            }));

            next_dispatch += period;
        }
        step_start = step_end;
    }
}

fn cadence_period(rate: u64) -> Duration {
    let rate = rate.max(1);
    let period_nanos = 1_000_000_000_u128 / u128::from(rate);
    Duration::from_nanos(period_nanos.min(u128::from(u64::MAX)) as u64)
}

// the open-loop driver tests build a `TokioPerCoreRuntime` directly (the
// driver is runtime-agnostic, but this fixture is tokio-specific and runs
// under `#[proxima::test]`). gated on `runtime-tokio`; run them with
// `--features runtime-tokio`.
#[cfg(all(test, feature = "runtime-tokio"))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
mod tests {
    use super::*;
    use crate::pipe::into_handle;
    use crate::runtime::TokioPerCoreRuntime;
    use crate::scenarios::spec::{DurationSpec, ProfileStep};
    use crate::telemetry::NoopTelemetry;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            async move { Ok(Response::ok(bytes::Bytes::from_static(b"ok"))) }
        }
    }


    fn fixture_workload(target_rps: u64, secs: u64) -> WorkloadSpec {
        WorkloadSpec::new_open_loop("echo", target_rps, DurationSpec::from_secs(secs))
            .with_concurrency(64)
    }

    fn build_runtime() -> Arc<dyn Runtime> {
        Arc::new(TokioPerCoreRuntime::new(1).expect("build tokio per-core runtime"))
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_drives_at_target_rps_for_duration() {
        let pipe = into_handle(EchoPipe);
        let workload = fixture_workload(200, 2);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime = build_runtime();

        let summary = drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None)
            .await
            .expect("driver completes");

        let expected_floor = 200_u64 * 2 / 4;
        assert!(
            summary.completed >= expected_floor,
            "expected at least {expected_floor} completed, got {}",
            summary.completed
        );
        assert!(summary.failures == 0, "no failures expected with echo pipe");
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_emits_per_second_snapshot_windows() {
        let pipe = into_handle(EchoPipe);
        let workload = fixture_workload(50, 3);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime = build_runtime();

        let summary = drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None)
            .await
            .expect("driver completes");

        assert!(
            summary.windows.len() >= 2,
            "expected at least 2 windows over 3s, got {}",
            summary.windows.len()
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_streams_snapshots_through_sink_when_provided() {
        let pipe = into_handle(EchoPipe);
        let workload = fixture_workload(40, 2);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime = build_runtime();
        let (sender, receiver) = std::sync::mpsc::channel();

        let summary =
            drive_workload_open(pipe, &workload, telemetry, metrics, runtime, Some(sender))
                .await
                .expect("driver completes");

        let streamed: Vec<MetricsSnapshot> = receiver.try_iter().collect();
        assert!(
            !streamed.is_empty(),
            "expected at least one streamed snapshot"
        );
        assert_eq!(streamed.len(), summary.windows.len());
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_profile_walks_each_step_in_order() {
        let pipe = into_handle(EchoPipe);
        let mut workload = WorkloadSpec::new_open_loop("echo", 100, DurationSpec::from_secs(1));
        workload.target_rps = None;
        workload.duration = None;
        workload.concurrency = 32;
        workload.profile = vec![
            ProfileStep {
                rate: 50,
                duration: DurationSpec::from_secs(1),
            },
            ProfileStep {
                rate: 150,
                duration: DurationSpec::from_secs(1),
            },
        ];
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime = build_runtime();

        let summary = drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None)
            .await
            .expect("driver completes");

        let total_target = 50_u64 + 150_u64;
        assert!(
            summary.completed >= total_target / 4,
            "two-step profile should run both steps; got completed={}",
            summary.completed
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_surfaces_saturation_via_skipped_count() {
        let pipe = into_handle(SlowPipe::default());
        let workload = WorkloadSpec::new_open_loop("slow", 200, DurationSpec::from_secs(1))
            .with_concurrency(2);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = NoopTelemetry::handle();
        let runtime = build_runtime();

        let summary = drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None)
            .await
            .expect("driver completes");

        assert!(
            summary.skipped_coordinated_omission > 0,
            "low concurrency + slow pipe should produce skipped slots; \
             summary={summary:?}"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn open_loop_rejects_workload_missing_open_loop_fields() {
        let pipe = into_handle(EchoPipe);
        let workload = WorkloadSpec::new("echo", 5);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime = build_runtime();

        let outcome = drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test(runtime = "tokio")]
    async fn run_scenario_dispatches_open_loop_via_workload_mode() {
        let scenario = Scenario::new_programmatic(
            WorkloadSpec::new_open_loop("echo", 100, DurationSpec::from_secs(1))
                .with_concurrency(16),
        )
        .with_pipe(
            "echo",
            serde_json::json!({ "synth": { "status": 200, "body": "ok" } }),
        );
        let context = LoadContext::with_default_registry().expect("load context");

        let report = run_scenario(&scenario, &context)
            .await
            .expect("scenario runs");

        assert!(
            !report.windows.is_empty(),
            "open-loop run_in_process branch must populate ScenarioReport.windows"
        );
        let floor = 100_u64 / 4;
        assert!(
            report.completed >= floor,
            "open-loop run completed too few requests: {} < {floor}",
            report.completed
        );
        assert_eq!(report.failures, 0, "synth pipe should not fail");
    }

    #[proxima::test(runtime = "tokio")]
    async fn run_scenario_closed_loop_leaves_windows_empty() {
        let scenario =
            Scenario::new_programmatic(WorkloadSpec::new("echo", 25).with_concurrency(4))
                .with_pipe(
                    "echo",
                    serde_json::json!({ "synth": { "status": 200, "body": "ok" } }),
                );
        let context = LoadContext::with_default_registry().expect("load context");

        let report = run_scenario(&scenario, &context)
            .await
            .expect("scenario runs");

        assert!(
            report.windows.is_empty(),
            "closed-loop must leave windows empty"
        );
        assert_eq!(report.completed, 25);
    }

    #[derive(Default)]
    struct SlowPipe;

    impl SendPipe for SlowPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(Response::ok(bytes::Bytes::from_static(b"slow")))
            }
        }
    }

}

// the rewrite's whole point, proven on the prime-native runtime with zero
// tokio in the path: the open-loop driver — which awaits `timer_at` and
// dispatches per-request work via `spawn_on_current_core` — runs to completion
// on a `PrimeRuntime`. `run_prime` drives the outer async fn on its own
// prime core; the workload under test runs on the dedicated `PrimeRuntime::new(1)`
// passed in. before this rewrite the loop hard-errored off `runtime-tokio`.
#[cfg(all(
    test,
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod prime_tests {
    use super::*;
    use crate::pipe::into_handle;
    use crate::scenarios::spec::DurationSpec;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            async move { Ok(Response::ok(bytes::Bytes::from_static(b"ok"))) }
        }
    }


    #[test]
    fn open_loop_drives_on_prime_runtime() {
        let pipe = into_handle(EchoPipe);
        let workload = WorkloadSpec::new_open_loop("echo", 100, DurationSpec::from_secs(2))
            .with_concurrency(64);
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let runtime: Arc<dyn Runtime> =
            Arc::new(crate::runtime::PrimeRuntime::new(1).expect("build prime per-core runtime"));

        // this test deliberately pins the prime backend to prove the open-loop
        // driver runs prime-native; the edge-driver guardrail is not for it.
        #[allow(clippy::disallowed_methods)]
        let summary = crate::runtime::run_prime(async move {
            drive_workload_open(pipe, &workload, telemetry, metrics, runtime, None).await
        })
        .expect("run_prime drives the driver future")
        .expect("prime-driven open-loop driver completes");

        let expected_floor = 100_u64 * 2 / 4;
        assert!(
            summary.completed >= expected_floor,
            "expected at least {expected_floor} completed on prime, got {}",
            summary.completed
        );
        assert_eq!(
            summary.failures, 0,
            "no failures expected with echo pipe on prime"
        );
    }
}
