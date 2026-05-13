use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use proxima_primitives::sync::Mutex as TokioMutex;
use tracing::debug;

use crate::app::{App, MountTarget};
use crate::control_plane::{ControlPlane, PipeState, PipeStatus};
use crate::error::ProximaError;
use crate::load::Spec;
use crate::log_buffer::LogBufferRegistry;
use crate::telemetry::MetricsSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipeConfig {
    /// pipe registration name. example: `"cart_api"`
    pub name: String,
    /// inline pipe spec (same shape as `[[pipe]]` in TOML)
    #[serde(flatten)]
    pub spec: Value,
    /// dependency names; topological order on `start`, cycles error.
    /// example: `["cart_db"]`. default: `[]`
    #[serde(default)]
    pub requires: Vec<String>,
}

/// Clone-and-swap on every mutation — readers (`list`, `status`, log
/// queries) take no lock and never see torn updates.
#[derive(Clone, Default, Debug)]
pub struct DaemonState {
    pub configs: HashMap<String, PipeConfig>,
    pub running: HashSet<String>,
    pub started_at_unix_ms: HashMap<String, u64>,
    pub restart_counts: HashMap<String, u64>,
}

/// `ArcSwap` (not `Arc<ArcSwap<...>>`) — the plane is already shared
/// via `Arc<DaemonControlPlane>`, so an outer Arc would double-indirect.
pub struct DaemonControlPlane {
    app: Arc<TokioMutex<App>>,
    state: ArcSwap<DaemonState>,
    log_buffers: Arc<LogBufferRegistry>,
}

impl DaemonControlPlane {
    #[must_use]
    pub fn new(app: App, configs: Vec<PipeConfig>) -> Self {
        let log_buffers = app.load_context().log_buffers.clone();
        let mut map: HashMap<String, PipeConfig> = HashMap::new();
        for config in configs {
            map.insert(config.name.clone(), config);
        }
        let state = DaemonState {
            configs: map,
            running: HashSet::new(),
            started_at_unix_ms: HashMap::new(),
            restart_counts: HashMap::new(),
        };
        Self {
            app: Arc::new(TokioMutex::new(app)),
            state: ArcSwap::from_pointee(state),
            log_buffers,
        }
    }

    /// Mount `path` to a (named or handle) target on the underlying app — lets a
    /// caller drive the router directly (embedding / `LivePlane` test fixtures).
    ///
    /// # Errors
    /// Propagates the app's mount error.
    pub async fn mount(&self, path: &str, target: MountTarget) -> Result<(), ProximaError> {
        self.app.lock().await.mount(path, target)
    }

    /// Borrow the router handle for direct `Pipe::call`s (testing / diagnostics).
    pub async fn router(&self) -> crate::pipe::PipeHandle {
        self.app.lock().await.router_handle()
    }

    pub fn upsert_config(&self, config: PipeConfig) -> Result<(), ProximaError> {
        self.state.rcu(|previous| {
            let mut next = (**previous).clone();
            next.configs.insert(config.name.clone(), config.clone());
            Arc::new(next)
        });
        Ok(())
    }

    fn topological_order(&self, root: &str) -> Result<Vec<String>, ProximaError> {
        let snapshot = self.state.load();
        let mut order: Vec<String> = Vec::new();
        let mut visiting: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();
        topological_walk(
            root,
            &snapshot.configs,
            &mut order,
            &mut visiting,
            &mut visited,
        )?;
        Ok(order)
    }

    fn is_running(&self, name: &str) -> bool {
        self.state.load().running.contains(name)
    }

    fn mark_running(&self, name: &str) -> Result<(), ProximaError> {
        self.state.rcu(|previous| {
            let mut next = (**previous).clone();
            next.running.insert(name.to_string());
            next.started_at_unix_ms
                .insert(name.to_string(), now_unix_ms());
            Arc::new(next)
        });
        Ok(())
    }

    fn mark_stopped(&self, name: &str) -> Result<(), ProximaError> {
        self.state.rcu(|previous| {
            let mut next = (**previous).clone();
            next.running.remove(name);
            next.started_at_unix_ms.remove(name);
            Arc::new(next)
        });
        Ok(())
    }

    fn bump_restart_count(&self, name: &str) -> Result<(), ProximaError> {
        self.state.rcu(|previous| {
            let mut next = (**previous).clone();
            *next.restart_counts.entry(name.to_string()).or_insert(0) += 1;
            Arc::new(next)
        });
        Ok(())
    }

    fn build_status(&self, name: &str) -> PipeStatus {
        let snapshot = self.state.load();
        let state = if snapshot.running.contains(name) {
            PipeState::Running
        } else if snapshot.configs.contains_key(name) {
            PipeState::Stopped
        } else {
            PipeState::Unknown
        };
        let uptime_ms = snapshot
            .started_at_unix_ms
            .get(name)
            .copied()
            .map(|started| now_unix_ms().saturating_sub(started));
        let restart_count = snapshot.restart_counts.get(name).copied().unwrap_or(0);
        PipeStatus {
            name: name.to_string(),
            state,
            uptime_ms,
            restart_count,
            last_message: None,
        }
    }

    async fn spawn_pipe(&self, name: &str) -> Result<(), ProximaError> {
        if self.is_running(name) {
            return Ok(());
        }
        let config = {
            let snapshot = self.state.load();
            snapshot
                .configs
                .get(name)
                .cloned()
                .ok_or_else(|| ProximaError::NotFound(format!("pipe `{name}`")))?
        };
        let mut app = self.app.lock().await;
        let handle = app
            .pipe(&config.name, Spec::Inline(config.spec.clone()))
            .await?;
        let mount_path = format!("/{}/{{*path}}", config.name);
        if let Err(error) = app.mount(&mount_path, MountTarget::Handle(handle)) {
            // drop the just-registered pipe to avoid leaking a half-state
            let _ = app.remove_pipe(&config.name);
            return Err(error);
        }
        drop(app);
        self.mark_running(name)?;
        debug!(pipe = %name, "pipe started");
        Ok(())
    }

    async fn drop_pipe(&self, name: &str) -> Result<(), ProximaError> {
        if !self.is_running(name) {
            return Ok(());
        }
        let mount_path = format!("/{name}/{{*path}}");
        let app = self.app.lock().await;
        app.unmount(&mount_path);
        drop(app);
        let mut app = self.app.lock().await;
        match app.remove_pipe(name) {
            Ok(_) => {}
            Err(ProximaError::NotFound(_)) => {}
            Err(err) => return Err(err),
        }
        drop(app);
        self.mark_stopped(name)?;
        debug!(pipe = %name, "pipe stopped");
        Ok(())
    }
}

impl ControlPlane for DaemonControlPlane {
    fn list_pipes<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipeStatus>, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let snapshot = self.state.load();
            let mut names: Vec<String> = snapshot.configs.keys().cloned().collect();
            drop(snapshot);
            names.sort();
            Ok(names.iter().map(|name| self.build_status(name)).collect())
        })
    }

    fn status<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            if !self.state.load().configs.contains_key(name) {
                return Err(ProximaError::NotFound(format!("pipe `{name}`")));
            }
            Ok(self.build_status(name))
        })
    }

    fn snapshot_metrics<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<MetricsSnapshot, ProximaError>> + Send + 'lifetime>>
    {
        Box::pin(async move {
            let app = self.app.lock().await;
            let snapshot = app
                .metrics()
                .map(|metrics| metrics.snapshot())
                .unwrap_or_else(|| MetricsSnapshot {
                    counters: Vec::new(),
                    gauges: Vec::new(),
                    histograms: Vec::new(),
                });
            Ok(snapshot)
        })
    }

    fn start<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            let order = self.topological_order(name)?;
            for pipe_name in order {
                self.spawn_pipe(&pipe_name).await?;
            }
            Ok(self.build_status(name))
        })
    }

    fn stop<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            if !self.state.load().configs.contains_key(name) {
                return Err(ProximaError::NotFound(format!("pipe `{name}`")));
            }
            self.drop_pipe(name).await?;
            Ok(self.build_status(name))
        })
    }

    fn restart<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            self.drop_pipe(name).await?;
            self.bump_restart_count(name)?;
            let order = self.topological_order(name)?;
            for pipe_name in order {
                self.spawn_pipe(&pipe_name).await?;
            }
            Ok(self.build_status(name))
        })
    }

    fn logs<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
        max_lines: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            let buffer = self
                .log_buffers
                .get(name)
                .ok_or_else(|| ProximaError::NotFound(format!("logs for `{name}`")))?;
            Ok(buffer.snapshot(max_lines))
        })
    }

    fn apply<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
        spec: Value,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        Box::pin(async move {
            // Store the new config so subsequent inspection (list/status)
            // reflects the spec actually in use.
            self.state.rcu(|previous| {
                let mut next = (**previous).clone();
                let mut entry = next
                    .configs
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| PipeConfig {
                        name: name.to_string(),
                        spec: serde_json::Value::Null,
                        requires: Vec::new(),
                    });
                entry.spec = spec.clone();
                next.configs.insert(name.to_string(), entry);
                Arc::new(next)
            });
            // App::update_pipe rebuilds the pipe from the new spec and
            // atomically rewrites every mount that pointed at the old handle.
            // In-flight requests on the old handle complete; new requests hit
            // the new impl.
            let mut app = self.app.lock().await;
            app.update_pipe(name, Spec::Inline(spec)).await?;
            drop(app);
            Ok(self.build_status(name))
        })
    }
}

fn topological_walk(
    name: &str,
    configs: &HashMap<String, PipeConfig>,
    order: &mut Vec<String>,
    visiting: &mut HashSet<String>,
    visited: &mut HashSet<String>,
) -> Result<(), ProximaError> {
    if visited.contains(name) {
        return Ok(());
    }
    if visiting.contains(name) {
        return Err(ProximaError::Config(format!(
            "pipe dep cycle including `{name}`"
        )));
    }
    visiting.insert(name.to_string());
    let config = configs
        .get(name)
        .ok_or_else(|| ProximaError::NotFound(format!("pipe `{name}` not configured")))?;
    for dep in &config.requires {
        topological_walk(dep, configs, order, visiting, visited)?;
    }
    visiting.remove(name);
    visited.insert(name.to_string());
    order.push(name.to_string());
    Ok(())
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::SendPipe;
    use serde_json::json;

    fn fixture() -> DaemonControlPlane {
        let app = App::new().expect("app");
        let configs = vec![
            PipeConfig {
                name: "leaf".into(),
                spec: json!({"synth": {"status": 200, "body": "leaf"}}),
                requires: vec![],
            },
            PipeConfig {
                name: "mid".into(),
                spec: json!({"synth": {"status": 200, "body": "mid"}}),
                requires: vec!["leaf".into()],
            },
            PipeConfig {
                name: "root".into(),
                spec: json!({"synth": {"status": 200, "body": "root"}}),
                requires: vec!["mid".into()],
            },
        ];
        DaemonControlPlane::new(app, configs)
    }

    #[proxima::test]
    async fn list_pipes_returns_every_configured_with_stopped_state() {
        let plane = fixture();
        let listed = plane.list_pipes().await.expect("list");
        assert_eq!(listed.len(), 3);
        for status in &listed {
            assert_eq!(status.state, PipeState::Stopped);
        }
    }

    #[proxima::test]
    async fn status_unknown_name_returns_not_found() {
        let plane = fixture();
        let outcome = plane.status("does-not-exist").await;
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }

    #[proxima::test]
    async fn start_root_walks_dep_graph_in_topological_order() {
        let plane = fixture();
        let status = plane.start("root").await.expect("start root");
        assert_eq!(status.name, "root");
        assert_eq!(status.state, PipeState::Running);
        // every dep must now be running too.
        let listed = plane.list_pipes().await.expect("list");
        for pipe in listed {
            assert_eq!(
                pipe.state,
                PipeState::Running,
                "pipe `{}` should be running",
                pipe.name,
            );
        }
    }

    #[proxima::test]
    async fn stop_one_leaves_others_alone() {
        let plane = fixture();
        plane.start("root").await.expect("start root");
        plane.stop("mid").await.expect("stop mid");
        let mid = plane.status("mid").await.expect("mid status");
        assert_eq!(mid.state, PipeState::Stopped);
        let leaf = plane.status("leaf").await.expect("leaf status");
        assert_eq!(leaf.state, PipeState::Running);
    }

    #[proxima::test]
    async fn restart_bumps_count_and_keeps_running() {
        let plane = fixture();
        plane.start("leaf").await.expect("start leaf");
        let after_first = plane.restart("leaf").await.expect("restart");
        assert_eq!(after_first.state, PipeState::Running);
        assert_eq!(after_first.restart_count, 1);
        let after_second = plane.restart("leaf").await.expect("restart again");
        assert_eq!(after_second.restart_count, 2);
    }

    #[proxima::test]
    async fn dep_cycle_returns_typed_error() {
        let app = App::new().expect("app");
        let configs = vec![
            PipeConfig {
                name: "a".into(),
                spec: json!({"synth": {"status": 200, "body": "a"}}),
                requires: vec!["b".into()],
            },
            PipeConfig {
                name: "b".into(),
                spec: json!({"synth": {"status": 200, "body": "b"}}),
                requires: vec!["a".into()],
            },
        ];
        let plane = DaemonControlPlane::new(app, configs);
        let outcome = plane.start("a").await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn missing_dep_returns_not_found() {
        let app = App::new().expect("app");
        let configs = vec![PipeConfig {
            name: "needy".into(),
            spec: json!({"synth": {"status": 200, "body": "needy"}}),
            requires: vec!["does-not-exist".into()],
        }];
        let plane = DaemonControlPlane::new(app, configs);
        let outcome = plane.start("needy").await;
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }

    #[proxima::test]
    async fn apply_swaps_pipe_to_new_spec_and_router_reflects_new_body() {
        let plane = fixture();
        plane.start("leaf").await.expect("start leaf");
        let app_handle = plane.app.lock().await;
        app_handle
            .mount("/leaf", crate::MountTarget::Named("leaf".into()))
            .expect("mount leaf");
        drop(app_handle);

        // Pre-swap: routing to /leaf returns the original body.
        let initial = call_through_router(&plane, "/leaf").await;
        assert_eq!(initial, b"leaf");

        // Apply: hand the daemon a new spec for `leaf`.
        let new_spec = json!({"synth": {"status": 200, "body": "leaf-v2"}});
        let status = plane
            .apply("leaf", new_spec.clone())
            .await
            .expect("apply leaf");
        assert_eq!(status.name, "leaf");

        // Post-swap: the router (rebuilt by App::update_pipe) now hits
        // the new impl. This is the hot-swap acceptance test.
        let after = call_through_router(&plane, "/leaf").await;
        assert_eq!(after, b"leaf-v2");

        // And the stored config matches the new spec.
        let stored = plane
            .state
            .load()
            .configs
            .get("leaf")
            .expect("config")
            .spec
            .clone();
        assert_eq!(stored, new_spec);
    }

    async fn call_through_router(plane: &DaemonControlPlane, path: &str) -> Vec<u8> {
        let app_handle = plane.app.lock().await;
        let router = app_handle.router_handle();
        drop(app_handle);
        let request = crate::Request::builder()
            .method("GET")
            .path(path)
            .build()
            .expect("request");
        let response = SendPipe::call(&router, request).await.expect("router call");
        response.collect_body().await.expect("body").to_vec()
    }
}
