//! Control-plane trait surface for proxima: introspect / manage running
//! Pipes at runtime.
//!
//! Folded from the former `proxima-control-plane` crate.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::sync::Arc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::future::Future;
#[cfg(feature = "alloc")]
use core::pin::Pin;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use proxima_core::ProximaError;
#[cfg(feature = "std")]
use proxima_core::live::{Live, LiveControl, live};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::MetricsSnapshot;

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipeState {
    Running,
    Stopped,
    Failed,
    Starting,
    Stopping,
    Unknown,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipeStatus {
    pub name: String,
    pub state: PipeState,
    #[serde(default)]
    pub uptime_ms: Option<u64>,
    #[serde(default)]
    pub restart_count: u64,
    #[serde(default)]
    pub last_message: Option<String>,
}

/// Control surface for the daemon. Read-only ops have default impls
/// returning `NotFound` so inspection-only planes (tests, static) can
/// skip mutation entirely.
#[cfg(feature = "alloc")]
pub trait ControlPlane: Send + Sync + 'static {
    fn list_pipes<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipeStatus>, ProximaError>> + Send + 'lifetime>>;

    fn status<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>>;

    fn snapshot_metrics<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<MetricsSnapshot, ProximaError>> + Send + 'lifetime>>;

    /// Walks the dep graph and starts every required pipe in
    /// topological order before the requested one.
    fn start<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(format!(
                "control plane does not support `start`; received `{name}`"
            )))
        })
    }

    fn stop<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(format!(
                "control plane does not support `stop`; received `{name}`"
            )))
        })
    }

    fn restart<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(format!(
                "control plane does not support `restart`; received `{name}`"
            )))
        })
    }

    fn reload<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(
                "control plane does not support `reload`".into(),
            ))
        })
    }

    /// Trigger graceful shutdown of the underlying server. Idempotent
    /// — calling more than once is harmless. Read-only planes (tests,
    /// static) return `NotFound` from the default impl.
    fn shutdown<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(
                "control plane does not support `shutdown`".into(),
            ))
        })
    }

    /// Hot-swap a registered pipe to a new spec. The daemon rebuilds the
    /// pipe from `spec` and atomically updates every mount pointing at
    /// the old handle to the new one. In-flight requests on the old handle
    /// complete; new requests hit the new impl. Default impl returns
    /// `NotFound` so read-only planes don't have to wire this up.
    fn apply<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
        _spec: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            Err(ProximaError::NotFound(format!(
                "control plane does not support `apply`; received `{name}`"
            )))
        })
    }

    /// Oldest-first log snapshot; `max_lines` keeps the most-recent N,
    /// `None` returns all retained.
    fn logs<'lifetime>(
        &'lifetime self,
        _name: &'lifetime str,
        _max_lines: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
}

#[cfg(feature = "alloc")]
pub type DynControlPlane = Arc<dyn ControlPlane>;

/// Read-only `ControlPlane` for tests — state set by the harness.
/// Lock-free: `Live<BTreeMap<...>>` with copy-on-write (rcu) on `upsert`.
/// `Live` is arc-swap-backed, and arc-swap has no no_std path (its no_std
/// tier needs an unstable nightly feature — see proxima-config's schema module for the same
/// call), so this type stays `std`-only even though `ControlPlane`,
/// `ControlPlanePipe`, and the rest of the trait surface reach alloc.
#[cfg(feature = "std")]
pub struct StaticControlPlane {
    inner: Live<BTreeMap<String, PipeStatus>>,
    control: LiveControl<BTreeMap<String, PipeStatus>>,
}

#[cfg(feature = "std")]
impl StaticControlPlane {
    #[must_use]
    pub fn new(initial: Vec<PipeStatus>) -> Self {
        let inner: BTreeMap<String, PipeStatus> = initial
            .into_iter()
            .map(|status| (status.name.clone(), status))
            .collect();
        let (inner, control) = live(inner);
        Self { inner, control }
    }

    pub fn upsert(&self, status: PipeStatus) {
        self.control.update(|current| {
            let mut next = current.clone();
            next.insert(status.name.clone(), status.clone());
            next
        });
    }
}

#[cfg(feature = "std")]
impl ControlPlane for StaticControlPlane {
    fn list_pipes<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipeStatus>, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let mut pipes: Vec<PipeStatus> = self
                .inner
                .read(|snapshot| snapshot.values().cloned().collect());
            pipes.sort_by(|left, right| left.name.cmp(&right.name));
            Ok(pipes)
        })
    }

    fn status<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            self.inner
                .read(|snapshot| snapshot.get(name).cloned())
                .ok_or_else(|| ProximaError::NotFound(format!("pipe `{name}`")))
        })
    }

    fn snapshot_metrics<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<MetricsSnapshot, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            Ok(MetricsSnapshot {
                counters: Vec::new(),
                gauges: Vec::new(),
                histograms: Vec::new(),
            })
        })
    }
}

/// `Pipe` adapter mapping `(method, path)` onto `ControlPlane`
/// methods. Listeners (http, mcp, direct-socket) translate their
/// wire format into a `Request` and dispatch through here.
#[cfg(feature = "alloc")]
pub struct ControlPlanePipe {
    plane: DynControlPlane,
    label: String,
}

#[cfg(feature = "alloc")]
impl ControlPlanePipe {
    #[must_use]
    pub fn new(plane: DynControlPlane) -> Self {
        Self {
            plane,
            label: "control_plane".into(),
        }
    }

    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

#[cfg(feature = "alloc")]
impl SendPipe for ControlPlanePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let plane = self.plane.clone();
        async move {
            // method and path are bytes; convert at this control-plane
            // edge for the &str-typed route matcher. control-plane URLs
            // are always ASCII so lossy is acceptable.
            let method_string =
                String::from_utf8_lossy(request.method.as_bytes()).to_ascii_uppercase();
            let path_string = String::from_utf8_lossy(&request.path).into_owned();
            let method = method_string.as_str();
            let path = path_string.as_str();
            match (method, path) {
                ("GET", "/pipes") => {
                    let pipes = plane.list_pipes().await?;
                    json_response(200, &pipes)
                }
                ("GET", "/metrics") => {
                    let snapshot = plane.snapshot_metrics().await?;
                    json_response(200, &snapshot_envelope(snapshot))
                }
                ("GET", path) if let Some(name) = strip_logs_prefix(path) => {
                    let max_lines = request
                        .query
                        .get_str("max_lines")
                        .and_then(|raw| raw.parse::<usize>().ok());
                    match plane.logs(name, max_lines).await {
                        Ok(lines) => json_response(200, &lines),
                        Err(ProximaError::NotFound(_)) => {
                            Ok(Response::not_found()
                                .with_body(format!("logs for `{name}` not found")))
                        }
                        Err(error) => Err(error),
                    }
                }
                ("POST", path) if let Some(name) = strip_apply_suffix(path) => {
                    // body is the new pipe spec as JSON; deserialize at this edge.
                    let (_meta, body_bytes) = request.body_bytes().await?;
                    let spec: serde_json::Value = if body_bytes.is_empty() {
                        return Ok(Response::new(400)
                            .with_body("apply requires a JSON body with the new pipe spec"));
                    } else {
                        serde_json::from_slice(&body_bytes).map_err(|err| {
                            ProximaError::Config(format!("apply body must be JSON: {err}"))
                        })?
                    };
                    match plane.apply(name, spec).await {
                        Ok(status) => json_response(200, &status),
                        Err(ProximaError::NotFound(_)) => {
                            Ok(Response::not_found().with_body(format!("pipe `{name}` not found")))
                        }
                        Err(error) => Err(error),
                    }
                }
                ("POST", path) if let Some((name, action)) = strip_action_suffix(path) => {
                    let outcome = match action {
                        "start" => plane.start(name).await,
                        "stop" => plane.stop(name).await,
                        "restart" => plane.restart(name).await,
                        _ => {
                            return Ok(Response::not_found()
                                .with_body(format!("unknown action `{action}` on `{name}`")));
                        }
                    };
                    match outcome {
                        Ok(status) => json_response(200, &status),
                        Err(ProximaError::NotFound(_)) => {
                            Ok(Response::not_found().with_body(format!("pipe `{name}` not found")))
                        }
                        Err(error) => Err(error),
                    }
                }
                ("POST", "/reload") => match plane.reload().await {
                    Ok(()) => Ok(Response::no_data()),
                    Err(ProximaError::NotFound(message)) => {
                        Ok(Response::new(501).with_body(message))
                    }
                    Err(error) => Err(error),
                },
                ("POST", "/shutdown") => match plane.shutdown().await {
                    Ok(()) => Ok(Response::no_data()),
                    Err(ProximaError::NotFound(message)) => {
                        Ok(Response::new(501).with_body(message))
                    }
                    Err(error) => Err(error),
                },
                ("GET", path) if let Some(name) = strip_pipe_prefix(path) => {
                    match plane.status(name).await {
                        Ok(status) => json_response(200, &status),
                        Err(ProximaError::NotFound(_)) => {
                            Ok(Response::not_found().with_body(format!("pipe `{name}` not found")))
                        }
                        Err(error) => Err(error),
                    }
                }
                _ => Ok(Response::not_found()
                    .with_body(format!("unknown control-plane route: {method} {path}"))),
            }
        }
    }
}

#[cfg(feature = "alloc")]
fn strip_pipe_prefix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipes/")?;
    if remainder.is_empty() || remainder.contains('/') {
        return None;
    }
    Some(remainder)
}

#[cfg(feature = "alloc")]
fn strip_logs_prefix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipes/")?;
    let suffix = remainder.strip_suffix("/logs")?;
    if suffix.is_empty() || suffix.contains('/') {
        return None;
    }
    Some(suffix)
}

#[cfg(feature = "alloc")]
fn strip_action_suffix(path: &str) -> Option<(&str, &str)> {
    let remainder = path.strip_prefix("/pipes/")?;
    let (name, action) = remainder.rsplit_once('/')?;
    if name.is_empty() || name.contains('/') {
        return None;
    }
    if !matches!(action, "start" | "stop" | "restart") {
        return None;
    }
    Some((name, action))
}

#[cfg(feature = "alloc")]
fn strip_apply_suffix(path: &str) -> Option<&str> {
    let remainder = path.strip_prefix("/pipes/")?;
    let suffix = remainder.strip_suffix("/apply")?;
    if suffix.is_empty() || suffix.contains('/') {
        return None;
    }
    Some(suffix)
}

#[cfg(feature = "alloc")]
fn json_response<T: Serialize>(status: u16, value: &T) -> Result<Response<Bytes>, ProximaError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|err| ProximaError::Encode(format!("control plane json: {err}")))?;
    Ok(Response::new(status)
        .with_header("content-type", "application/json")
        .with_body(bytes::Bytes::from(bytes)))
}

#[cfg(feature = "alloc")]
#[derive(Debug, Serialize)]
struct SnapshotEnvelope {
    counters: Vec<SnapshotCounter>,
    gauges: Vec<SnapshotGauge>,
    histograms: Vec<SnapshotHistogram>,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Serialize)]
struct SnapshotCounter {
    metric: String,
    labels: Vec<(String, String)>,
    value: u64,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Serialize)]
struct SnapshotGauge {
    metric: String,
    labels: Vec<(String, String)>,
    value: i64,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Serialize)]
struct SnapshotHistogram {
    metric: String,
    labels: Vec<(String, String)>,
    count: u64,
    p50: f64,
    p99: f64,
}

#[cfg(feature = "alloc")]
fn snapshot_envelope(snapshot: MetricsSnapshot) -> SnapshotEnvelope {
    SnapshotEnvelope {
        counters: snapshot
            .counters
            .into_iter()
            .map(|(metric, labels, value)| SnapshotCounter {
                metric,
                labels: labels.entries().to_vec(),
                value,
            })
            .collect(),
        gauges: snapshot
            .gauges
            .into_iter()
            .map(|(metric, labels, value)| SnapshotGauge {
                metric,
                labels: labels.entries().to_vec(),
                value,
            })
            .collect(),
        histograms: snapshot
            .histograms
            .into_iter()
            .map(|(metric, labels, summary)| SnapshotHistogram {
                metric,
                labels: labels.entries().to_vec(),
                count: summary.count,
                p50: summary.p50,
                p99: summary.p99,
            })
            .collect(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::request::Request;

    fn fixture_plane() -> Arc<StaticControlPlane> {
        Arc::new(StaticControlPlane::new(vec![
            PipeStatus {
                name: "cart_api".into(),
                state: PipeState::Running,
                uptime_ms: Some(12_345),
                restart_count: 0,
                last_message: None,
            },
            PipeStatus {
                name: "cart_www".into(),
                state: PipeState::Stopped,
                uptime_ms: None,
                restart_count: 1,
                last_message: Some("manual stop".into()),
            },
        ]))
    }

    #[proxima::test]
    async fn get_pipes_returns_all_known_pipes_sorted() {
        let plane = fixture_plane();
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("GET")
            .path("/pipes")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        let pipes: Vec<PipeStatus> = serde_json::from_slice(&body).expect("parse");
        assert_eq!(pipes.len(), 2);
        assert_eq!(pipes[0].name, "cart_api");
        assert_eq!(pipes[1].name, "cart_www");
    }

    #[proxima::test]
    async fn get_pipe_by_name_returns_status_json() {
        let plane = fixture_plane();
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("GET")
            .path("/pipes/cart_api")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        let status: PipeStatus = serde_json::from_slice(&body).expect("parse");
        assert_eq!(status.name, "cart_api");
        assert_eq!(status.state, PipeState::Running);
    }

    #[proxima::test]
    async fn get_unknown_pipe_returns_404() {
        let plane = fixture_plane();
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("GET")
            .path("/pipes/nope")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn get_metrics_returns_envelope_shape() {
        let plane = fixture_plane();
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("GET")
            .path("/metrics")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        let envelope: serde_json::Value = serde_json::from_slice(&body).expect("parse");
        assert!(envelope.get("counters").is_some());
        assert!(envelope.get("gauges").is_some());
        assert!(envelope.get("histograms").is_some());
    }

    #[proxima::test]
    async fn unknown_route_returns_404_with_body() {
        let plane = fixture_plane();
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("POST")
            .path("/random")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 404);
        let body = response.collect_body().await.expect("collect");
        assert!(String::from_utf8_lossy(&body).contains("unknown control-plane route"));
    }

    struct LoggingPlane {
        lines: Vec<String>,
    }

    impl ControlPlane for LoggingPlane {
        fn list_pipes<'lifetime>(
            &'lifetime self,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<PipeStatus>, ProximaError>> + Send + 'lifetime>>
        {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn status<'lifetime>(
            &'lifetime self,
            _name: &'lifetime str,
        ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>>
        {
            Box::pin(async { Err(ProximaError::NotFound("static".into())) })
        }

        fn snapshot_metrics<'lifetime>(
            &'lifetime self,
        ) -> Pin<Box<dyn Future<Output = Result<MetricsSnapshot, ProximaError>> + Send + 'lifetime>>
        {
            Box::pin(async {
                Ok(MetricsSnapshot {
                    counters: Vec::new(),
                    gauges: Vec::new(),
                    histograms: Vec::new(),
                })
            })
        }

        fn logs<'lifetime>(
            &'lifetime self,
            _name: &'lifetime str,
            _max_lines: Option<usize>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, ProximaError>> + Send + 'lifetime>>
        {
            let lines = self.lines.clone();
            Box::pin(async move { Ok(lines) })
        }
    }

    #[proxima::test]
    async fn get_pipe_logs_returns_jsonl_array() {
        let plane: Arc<dyn ControlPlane> = Arc::new(LoggingPlane {
            lines: vec!["one".into(), "two".into(), "three".into()],
        });
        let pipe = ControlPlanePipe::new(plane);
        let request = Request::builder()
            .method("GET")
            .path("/pipes/cart_api/logs")
            .query_param("max_lines", "10")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        let lines: Vec<String> = serde_json::from_slice(&body).expect("parse");
        assert_eq!(lines, vec!["one", "two", "three"]);
    }

    #[proxima::test]
    async fn upsert_then_status_reflects_new_value() {
        let plane = fixture_plane();
        plane.upsert(PipeStatus {
            name: "cart_api".into(),
            state: PipeState::Failed,
            uptime_ms: None,
            restart_count: 7,
            last_message: Some("oom".into()),
        });
        let pipe = ControlPlanePipe::new(plane.clone());
        let request = Request::builder()
            .method("GET")
            .path("/pipes/cart_api")
            .build()
            .expect("request");
        let response = pipe.call(request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        let status: PipeStatus = serde_json::from_slice(&body).expect("parse");
        assert_eq!(status.state, PipeState::Failed);
        assert_eq!(status.restart_count, 7);
    }
}
