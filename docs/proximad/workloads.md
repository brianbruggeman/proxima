# proximad as the workload control plane

proximad is proxima's control plane: the k3s equivalent, multi-node by design.
Today it serves only the pipeline plane (`tools/proxima-cli/src/daemon.rs:165-168`):
run-to-completion child-process DAGs. The pieces of a workload plane exist and
are not connected. This plan connects them. Every element below names the
existing proxima primitive it reuses; nothing new is minted unless the
inventory showed no primitive covers it, and each such gap is listed with the
primitive it was checked against.

It implements the k3s-to-pipe mapping in `tools/proxima-vm/ROADMAP.md:618-626`.

## Element to primitive

| Element | Reuses | Where |
|---|---|---|
| workload control API (apply, start, stop, restart, status) | `ControlPlane` + `ControlPlanePipe` + `DaemonControlPlane` | `proxima-patterns/src/control_plane.rs:83-154,290-395`; `src/daemon_control_plane.rs:260-347` |
| pipelines beside workloads | `PipelineControlPlanePipe` mounted under `/pipelines` on the same router; the two traits stay separate | `src/pipelines/http_routes.rs:83-85` |
| lifecycle state on the wire | `PipeState` (Starting, Running, Stopping, Stopped, Failed) + `PipeStatus.restart_count`, `uptime_ms`, `last_message`; backoff reports as `Starting` with `last_message` | `control_plane.rs:36-56` |
| lock-free status read | `DaemonState` clone-and-swap behind `ArcSwap` | `daemon_control_plane.rs:30-39` |
| restart a workload that ended | the process supervisor loop: `RestartPolicy`, `max_restart_attempts`, exit published on a `watch` (`exit_tx`) | `src/upstreams/process.rs:366-427`, `:48-52` |
| restart delay | `Backoff::Exponential { initial, factor, max }` + `Jitter` replaces the supervisor's constant sleep | `proxima-primitives/src/pipe/resilience/backoff.rs:5-60` |
| crash loop | `CircuitBreaker` cooldown if a window bound is wanted | `proxima-primitives/src/pipe/resilience/circuit_breaker.rs:16-284` |
| "instance ended" signal | the same `watch` the process supervisor publishes; the resident vm pump publishes on its existing live-end branch | `process.rs:384-422`; `src/upstreams/vm_resident.rs:226-241` |
| cancellation tree (node -> instances) | `Signal` + `child()`, as `ProducerLifecycle::with_parent_signal` | `proxima-core/src/signal.rs:118-165`; `lifecycle.rs:79` |
| hot swap that listeners see | `SwappablePipe` / `SwapRegistry`: mounts hold the swappable handle, a swap or restart replaces what it points at; no router rewrite | `swap_registry.rs:31-128` |
| runtime listener bind / unbind | `App::build_listener` -> `ListenerHandle` (`bind_addr`, `stop`); graceful via `Shutdown::drain`; handles kept in `DaemonState` | `src/app.rs:1135-1147,1227-1265,1613`; `proxima-listen/src/handle.rs:26-50` |
| single owner of `App` instead of `TokioMutex` | `sync::mpsc` of requests + `oneshot` replies to one driver task | `proxima-primitives/src/sync/mpsc.rs`, `sync/oneshot.rs`; removes `daemon_control_plane.rs:7,45` |
| desired state persisted | `StateStore` put/list for pipes and listeners; change notify with `sync::watch` beside `put_*` / `remove_*` | `proxima-config/src/store/mod.rs:52-143`; `proxima-primitives/src/sync/watch.rs:76-241` |
| time, deterministic tests | `proxima_core::time::{now, interval}`, `Clock`, `MockClock` / `RecordingClock` | `proxima-core/src/time/mod.rs:41-215`; `proxima-primitives/src/pipe/clock.rs:20-60` |
| k8s manifests as input | `ConfigFormatRegistry::parse_sniff` (yaml) -> `serde_json::Value`; lowering is a function `Value -> Vec<Value>` that emits the `[[pipe]]` / `[[listen]]` tables `App::load_full` already consumes; vm and balancer pipes are ordinary pipe specs, no new factory | `proxima-config/src/lib.rs:85-255`; `app.rs:1152-1225` |
| Service over replicas | `UpstreamRef` per replica under a `Selection` (`RoundRobin` etc.), `OutlierPolicy` ejection | `proxima-patterns/src/balancer/selection.rs:230`; `upstream_ref.rs:8-130` |
| status watch | a `watch::Receiver` mapped to a stream, mirroring `PipelineControlPlane::subscribe_events` | `src/pipelines/control_plane.rs:77-130` |
| remote node | that node's `ControlPlanePipe` reached over UDS or `proximad serve --stdio`; the actuator calls it like the local one | `src/pipelines/control_plane.rs:77-79`; `daemon.rs:311-320` |
| placement, node registry | none found (greps for cluster, membership, gossip, raft, lease empty); records gain `node: Option<String>`, the member list is config, placement is a pipe over it | gap |

## Gaps the inventory proved

- `DaemonControlPlane::apply` returns 404 for a new name and leaves a config
  entry (`app.rs:639-643`, `daemon_control_plane.rs:323-337`).
- the resident vm upstream has no supervisor: its pump logs the guest's end and
  drops it (`vm_resident.rs:226-241`); the process supervisor exists but is
  process-only and sleeps a constant delay (`process.rs:408-410`).
- `App` keeps no per-listener router or handle registry (`app.rs:1152-1225,1245`),
  so a swap reaches `self.router` only.
- `StateStore` has no consumers and no change notification.
- no node identity or placement.

## Slices, each green

1. `apply` creates a missing pipe; config recorded only after the build succeeds.
2. resident vm upstream supervised by the process supervisor's policy (extract
   the loop so both upstream kinds drive it), delay from `Backoff`, end
   published on the `watch`, restart counted in `PipeStatus`.
3. mounts hold `SwappablePipe` handles; listener handles live in `DaemonState`;
   `apply_listener` / `remove_listener` over `build_listener`.
4. proximad serves `/pipelines` and the workload plane on one router; the
   driver task owns `App` through `mpsc` + `oneshot`.
5. `apply` persists through `StateStore`; startup loads it; `watch` notifies.
6. records carry `node`; member list in config; placement pipe; actuator calls
   the remote node's `ControlPlanePipe`.
7. k8s Deployment / Service / Ingress lowering through `ConfigFormatRegistry`.

## Proof

- e2e over the UDS control API: apply whoami (resident vm, OCI image) and a
  listener, GET 200, end the guest from inside, status shows
  `restart_count == 1` and `Running`, GET 200. Control: `RestartPolicy::Never`
  ends `Stopped` and the GET fails.
- two proximad processes on one host: apply with `node: B` on A serves on B.
- bench: apply-to-200 and end-to-replacement latency against the k3s baseline,
  with `/proc/loadavg`.
