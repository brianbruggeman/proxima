# proximad as the workload control plane

proximad is proxima's control plane: the k3s equivalent, multi-node by design.
Today it serves only the pipeline plane (`tools/proxima-cli/src/daemon.rs:165-168`):
run-to-completion child-process DAGs, restart hard-coded to `Never`
(`src/pipelines/executor.rs:293`). The pieces of a workload plane exist but
are not connected: `DaemonControlPlane` (`src/daemon_control_plane.rs`, built
only in tests and `benches/perf_audit.rs:90`), `StateStore`
(`proxima-config/src/store/mod.rs:52`, no consumers), the balancer, the
resident `vm` upstream with OCI pull (`src/upstreams/vm_resident.rs:143-152`).

This document is the plan for connecting them. It implements the k3s-to-pipe
mapping in `tools/proxima-vm/ROADMAP.md:618-626`; it does not replace it.

## Rules

- every IO is a pipe; the reconcile logic is a sans-IO FSM; no new type that a
  pipe expresses, and none that lets a caller do nothing new
- multi-node from the first slice: records carry `node`, observed state is
  keyed by `(node, name)`, a remote node is reached through the same control API
- k8s manifests are accepted as a config format; no k8s code is ported

## Slice 1: long-running workloads behind a listener (0 new types)

- proximad serves one router: `/pipelines/{*}` -> `PipelineControlPlanePipe`,
  `/{*}` -> `ControlPlanePipe` over `DaemonControlPlane`. The two traits stay
  separate; merging them gives a caller nothing new.
- `apply` creates or replaces. Today an unknown name returns 404 from
  `App::update_pipe` (`src/app.rs:639-643`) after `configs` was already written
  (`src/daemon_control_plane.rs:323-337`). Config is recorded only after the
  build succeeds.
- listeners are desired state: `apply_listener` / `remove_listener` on
  `ControlPlane` with `NotFound` defaults; `POST /listeners/{name}/apply`,
  `GET /listeners/{name}` reports the bound address.
- the mount-rewrite loop in `update_pipe` (`src/app.rs:670-689`) becomes a free
  function over any `ArcSwap<Router>`, so a swap or restart reaches every
  listener router that mounts the pipe.

Commits: apply creates a missing pipe atomically; lift the mount rewrite;
listener apply/remove; proximad serves both planes; e2e apply whoami (resident
vm, OCI image) behind a listener returns 200.

## Slice 2: desired state, nodes, k8s manifests (-1 type)

- `PipeConfig` is replaced by the store's `PipeRecord` (same fields).
- `apply` persists through `StateStore::put_pipe` / `put_listener` before it
  acknowledges; startup loads `list_pipes` / `list_listeners`. The store is a
  source pipe of records.
- `PipeRecord` / `ListenerRecord` gain `node: Option<String>`; `--node`
  (default hostname) and a `nodes = [{name, control}]` member list. Placement is
  a transform pipe over the member list and runs even with one member.
- k8s lowering is a transform pipe `Bytes -> Vec<Record>` over
  `serde_json::Value`, no mirror structs:
  - `Deployment` -> `replicas` resident vm pipes `{name}-{ordinal}`, `image`
    from the container, `guest_port` from `containerPort`, `restartPolicy` ->
    `RestartPolicy`, `nodeName` / hostname selector -> `node`
  - `Service` -> a balancer pipe over the ordinals; `NodePort` /
    `LoadBalancer` also a listener
  - `Ingress` -> mounts on the ingress listener; `rules[].host` is rejected with
    a typed error until host routing exists

## Slice 3: reconcile (3 enums)

A transform pipe `In = Input, Out = Action` in `proxima-patterns` (no_std +
alloc). State is `BTreeMap<(node, name), Instance>`; time arrives as `now_ms`.

```
Instance = Absent
         | Starting  { generation, attempt }
         | Running   { generation, attempt, started_ms }
         | Backoff   { generation, attempt, retry_at_ms }
         | Stopping  { generation, next: Option<u64> }
         | Completed { generation, exit_code: Option<i32> }
Input    = Desired { generation, policy } | Withdrawn
         | Started { generation } | StartFailed { generation }
         | Ended { generation, exit_code } | Stopped { generation }
         | Tick { now_ms }
Action   = Start { node, name, generation } | Stop { node, name, generation } | None
```

An input whose generation differs from the instance's is a no-op, so a stopped
instance's late `Ended` never restarts it. Restart follows `RestartPolicy` with
`restart_delay_ms * 2^attempt` backoff up to `max_restart_attempts`.

IO enters only as pipes: the control API, the store loader, a timer and the
actuator are sources of `Input`; the resident pump (`vm_resident.rs:226-241`)
and process exit fire a per-instance end signal that becomes `Ended`. The
actuator is a sink routed by `node`: local `App` calls, or the remote node's
`ControlPlanePipe`. The driver task owns `App`, removing
`Arc<TokioMutex<App>>`; status reads use a lock-free snapshot; request dispatch
stays `ArcSwap<Router>`.

## Proof

- FSM worked-example cases: happy path, restart under `Always`, stale
  generation ignored, `OnFailure` with exit 0 completes, attempts exhausted,
  withdraw during backoff, new generation while running.
- e2e: apply whoami (resident vm) and a listener over the UDS control API,
  watch to `Running`, GET 200, end the guest from inside, watch to
  `restart_count == 1`, GET 200. Control: `RestartPolicy::Never` must end
  `Completed` with the GET failing.
- two proximad processes on one host: apply with `node: B` on A serves on B.
- bench: apply-to-200 and end-to-replacement latency against the k3s baseline,
  with `/proc/loadavg`.
