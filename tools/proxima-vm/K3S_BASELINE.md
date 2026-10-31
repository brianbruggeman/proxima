# k3s baseline and proxima's current equivalent

Date: 2026-10-31. Host: `versailles`, nproc 8, 15 GiB RAM, Linux 7.2.11-zen1, `/dev/kvm` present, shared with other agents (loadavg is recorded per row and was 3 to 35 during the runs). Worktree `bench/k3s-baseline` at main `a93ab20c`. k3s v1.32.5+k3s1 (go1.23.8), default bundle (traefik, servicelb, coredns, local-path, metrics-server). oha 1.14.0. All proxima timings use `--release`.

Status of every number below: measurement. Mechanisms are stated where a trace or read supports them; otherwise the row says unexplained.

## Commands

proxima build (`CARGO_TARGET_DIR=/home/bix/.cache/cargo-k3s-baseline`):

```
cargo build --release --target x86_64-unknown-none --manifest-path tools/proxima-vm/guests/lambda/Cargo.toml --bin uppercase
cargo build --release --target x86_64-unknown-linux-musl --manifest-path tools/proxima-vm/guests/linux-init/Cargo.toml --bin uppercase-init
cargo build --release -p proxima-cli --features vm
```

proxima run: `examples/lambda/proxima.toml` and the two guest binaries copied to `/home/bix/.cache/k3s-baseline-tmp/lambda/`, started as a transient unit (no shell backgrounding):

```
sudo -n systemd-run -q --unit proxima-bench --uid=bix --collect $T/release/proxima serve --config $B/lambda/proxima.toml
```

k3s final command line (the unit was started with this for first boot and each warm restart):

```
sudo -n systemd-run -q --unit k3s-bench --collect /usr/local/bin/k3s server \
  --config $D/empty.yaml --private-registry $D/empty.yaml \
  --data-dir $D/data --write-kubeconfig $D/kubeconfig.yaml --write-kubeconfig-mode 644 \
  --node-name bench --kubelet-arg=root-dir=$D/kubelet \
  '--kubelet-arg=eviction-hard=nodefs.available<1Gi,imagefs.available<1Gi' \
  --kubelet-arg=image-gc-high-threshold=100 --kubelet-arg=image-gc-low-threshold=99
```

`D=/home/bix/.cache/k3s-bench` (removed at the end). Deviations from the stock invocation and why:

- `--config $D/empty.yaml`: the default `/etc/rancher/k3s/config.yaml` on this box holds `server: https://192.168.1.62:6443` plus a token (this box was an agent of another cluster). The first attempt without it made one outbound `GET /cacerts` to that address and died with `no route to host` (fatal, nothing joined). Every later run overrode the file.
- `--private-registry $D/empty.yaml`: isolates from `/etc/rancher/k3s/registries.yaml` (mirrors for private registries only).
- eviction/image-gc kubelet args: the second attempt (without them) hit DiskPressure because `/home` has 11 to 12 GiB free against the kubelet default threshold of 44 GiB; kubelet evicted both `helm-install-traefik` pods and tainted the node `disk-pressure:NoSchedule`, so traefik never started. That attempt was discarded and its data dir deleted; K1 first-boot below is the third attempt on a fresh data dir.

Measurement method (all inline shell, no script files): wall clock `date +%s%N` taken immediately before `systemd-run`; readiness polled with `kubectl` (k3s) or `curl` with `sleep 0.005` (proxima). Poll granularity and `kubectl`/`curl` process spawn are inside every latency number (tens of ms for kubectl, a few ms for curl). Percentiles for n runs: p50 = middle value, p99 = max when n < 100. RSS = `VmRSS` from `/proc/<pid>/status`; CPU = delta of utime+stime (fields 14,15 of `/proc/<pid>/stat`, 100 ticks/s). oha: `oha --no-tui ... --output-format json | jq`.

## Tables

### K1 / P1: cold start (exec to first ready)

| What | n | values (ms) | p50 | max | loadavg 1m at start |
|---|---|---|---|---|---|
| k3s first boot (fresh data dir, includes image pulls): node Ready | 1 | 8034 | 8034 | 8034 | 4.41 |
| k3s first boot: coredns Ready | 1 | 22941 | | | |
| k3s first boot: traefik Ready | 1 | 47909 | | | |
| k3s warm restart, same data dir: node Ready | 3 | 11866, 7756, 7719 | 7756 | 11866 | 12.27, 12.51, 12.19 |
| k3s warm restart: coredns Ready | 3 | 13838, 9924, 41493 | 13838 | 41493 | same |
| k3s warm restart: traefik Ready | 3 | 24559, 10773, 13918 | 13918 | 24559 | same |
| proxima exec to first 200 on /elf | 5 | 65, 53, 55, 53, 55 | 55 | 65 | 9.81, 9.34 x4 |
| proxima exec to first 200 on /linux (separate runs, /linux polled first) | 5 | 419, 411, 406, 413, 368 | 411 | 419 | 9.34 x3, 8.91 x2 |

Warm-restart readiness required the container `startedAt` (coredns, traefik) or node heartbeat to be at or after the restart second, because pod status in the datastore is stale-true across a restart. k3s first boot n=1 because it is a one-time event per data dir.

### K2 / P2: idle footprint

| What | value | notes |
|---|---|---|
| k3s RSS, k3s-server + containerd + 5 shims | 561,684 kB (server 453,868; containerd 56,968; shims 9,784 + 9,756 + 9,736 + 10,764 + 10,808) | 60 s settled after traefik Ready; loadavg 3.90 |
| k3s RSS, plus pod processes (pause x5, metrics-server 34,536, local-path 14,512, coredns 23,572, traefik 36,200, svclb entry x2) | 670,920 kB total | |
| k3s idle CPU over 30 s | 108 ticks total = 3.6% of one core (server 72, containerd 10, shims 5, metrics-server 12, coredns 7, traefik 2) | |
| k3s data dir on disk | 796 MiB (`du -sh`), kubelet dir 460 KiB | after first-boot image pulls |
| k3s binary | 72,138,904 B | |
| proxima RSS | 144,712 kB (30 threads) | 60 s after first /elf response (one request served); 144,680 kB 30 s later; 174,008 kB after the P3 load runs |
| proxima idle CPU over 30 s | 19 ticks = 0.63% of one core | loadavg 4.80 to 3.31 |
| proxima binary | 16,072,992 B (`release/proxima`, `--features vm`) | guest images: `uppercase` 14,144 B, `uppercase-init` 381,624 B; `/boot/vmlinuz-linux` referenced by path |

### K3 / P3: workload start

| What | n | p50 | p99 | max | loadavg |
|---|---|---|---|---|---|
| k3s pod create (kubectl apply) to phase Running, busybox cached | 20 | 860 ms | 914 ms | 914 ms | 4.20 |
| k3s pod create to first log line | 20 | 934 ms | 983 ms | 983 ms | 4.20 to 4.72 |
| k3s deployment scale 0 to 10, all Ready | 3 | 2048 ms (1207, 2048, 2240) | | 2240 | 4.71, 4.71, 6.65 |
| k3s deployment scale 0 to 50, all Ready | 3 | 6798 ms (5434, 6798, 6870) | | 6870 | 6.65, 9.97, 11.18 |
| proxima per-request guest, /elf, oha -c 1 n=500 x3 | 500 each | 3.018, 2.949, 3.368 ms | 19.87, 21.63, 15.60 ms | 26.54, 23.89, 21.87 ms | 3.05, 3.05, 3.28 |
| proxima per-request guest, /linux, oha -c 1 n=50 x3 | 50 each | 309.2, 309.3, 312.9 ms | 360.2, 336.6, 328.6 ms | 360.2, 336.6, 328.6 ms | 3.28, 3.15, 3.47 |
| proxima /elf throughput, oha -c 64 15 s x3 | 9123, 8889, 8137 | rps 612, 596, 546; latency p50 89.4, 84.1, 104.0 ms | 310.1, 440.1, 350.9 ms | 409.4, 660.3, 525.5 ms | 3.66, 5.97, 8.05 |

The oha error `aborted due to deadline` (63 to 64 per -c 64 run) is the in-flight request count when the 15 s deadline hit; all completed responses were 200.

### K4 / P4: request path to the same whoami backend (traefik/whoami:v1.10.2, 2 replicas, from the host, oha, all responses 200)

| Target | c=1 n=2000: rps / p50 / p99 / max (ms), 3 runs | c=64 15 s: rps / p50 / p99 / max (ms), 3 runs | loadavg |
|---|---|---|---|
| (a) traefik ingress, `http://192.168.1.76:80/` (servicelb node IP) | 4755, 5215, 5220 / 0.188, 0.183, 0.183 / 0.544, 0.327, 0.317 / 3.929, 2.149, 1.470 | 22945, 20506, 17014 / 1.727, 1.817, 2.085 / 13.638, 17.555, 23.924 / 38.5, 58.4, 93.8 | 7.39, 7.39, 14.85, 18.83 |
| (b1) NodePort, `http://127.0.0.1:31471/` | 10870, 10826, 10982 / 0.082 x3 / 0.212, 0.194, 0.204 / 0.553, 0.309, 0.381 | 44382, 44249, 33704 / 0.553, 0.562, 0.738 / 8.902, 8.786, 12.877 / 33.2, 30.6, 64.4 | 20.99 to 30.23 |
| (b2) ClusterIP, `http://10.43.5.110:80/` | 9779, 10893, 10748 / 0.085, 0.082, 0.083 / 0.236, 0.211, 0.226 / 2.706, 0.385, 1.103 | 40215, 38773, 31111 / 0.601, 0.646, 0.802 / 10.064, 10.014, 12.811 / 30.7, 35.7, 42.9 | 31.10 to 35.88 |
| P4 proxima reverse proxy, `http://127.0.0.1:8090/` to NodePort 127.0.0.1:31471 | 7082, 7365, 7358 / 0.123 x3 / 0.257, 0.246, 0.247 / 3.761, 0.480, 0.508 | 10124, 9501, 7207 / 0.186, 0.187, 0.277 / 0.390, 0.673, 28.985 / 11140.8, 5176.4, 14797.9 | 28.41, 26.38, 21.35, 17.42 |

P4 config (local, not in the repo): `$B/ingress.toml` = `name = "ingress"`, `[[upstreams]] name = "whoami"`, `http = "http://127.0.0.1:31471"`; run `proxima serve --config $B/ingress.toml --mount '/{*path}' --addr 127.0.0.1:8090` (form from `examples/config/nginx-style/proxima.toml:26`, upstream key `http` from `proxima-config/src/sugar/mod.rs:95`). proxima RSS during P4: 124,028 kB.

P4 c=64 distribution is bimodal. A fourth run (loadavg 14.92) gave 7286 rps, p50 0.272 ms, p90 1.98 ms, p99 38.3 ms, p99.9 894 ms, p99.99 3.9 s, histogram buckets reaching 6.0 s with 109,254 of 109,408 requests under 0.6 s. The cause of the multi-second tail is unexplained (not traced: no per-connection data was collected).

### K5: per-pod memory (50 idle `busybox sleep 3600` pods vs 0, 30 s settled)

| Run | 0 pods, total kB | 50 pods, total kB | delta kB | per pod kB | of which k3s-server+containerd+shims (cgroup) | of which container processes | host `free` used MB (0 to 50) |
|---|---|---|---|---|---|---|---|
| 1 | 882,188 | 1,925,024 | 1,042,836 | 20,857 | 773,372 to 1,686,808 (+913,436) | 108,816 to 238,216 (+129,400) | 8359 to 8739 |
| 2 | 891,648 | 1,282,260 | 390,612 | 7,812 | 788,328 to 1,089,824 (+301,496) | 103,320 to 192,436 (+89,116) | 8502 to 9301 |

loadavg: 17.29 / 13.65 / 11.10 / 7.92 at the four samples. The two runs disagree by 2.7x; the difference sits in the k3s-server/containerd/shim component (+913 MB vs +301 MB), whose 0-pod baseline (773 MB, 788 MB) is itself higher than the K2 idle value (562 MB) taken before the scale tests, so the server RSS depends on prior churn. Why the server RSS differs between run 1 and run 2 is unexplained (no heap profile taken).

## Function inventory (k3s function vs proxima today)

"absent" rows give the grep and its result; every grep was `grep -rIil -E '<pattern>' --include=*.rs --include=*.toml --exclude-dir=target --exclude-dir=.git .` from the worktree root and printed 0 files unless stated.

| k3s function | proxima today |
|---|---|
| Declarative desired state + API | partial: `StateStore` persists `pipes/<name>.toml` and `listeners/<name>.toml` (`proxima-config/src/store/mod.rs:1-8`, `:52`, `put_pipe :115`, `put_listener :134`); the `proximad` UDS/SSH-stdio daemon (`tools/proxima-cli/src/daemon.rs:136,188`) with `ControlPlane` start/stop/restart/apply (`src/daemon_control_plane.rs:260,273,286,315`). Unit is a pipe, not a pod/Deployment. No Kubernetes API object model: `grep 'ReplicaSet|desired_replicas|replica_count'` -> 0 files |
| Reconcile / controller loop | partial: store doc states "reconciled against it on startup and on every mutation" (`proxima-config/src/store/mod.rs:5-6`); `PipeConfig.requires` dependency order on start (`src/daemon_control_plane.rs:25-30`, test `:432`); a worker-count reconcile exists for the runtime thread pool, not for workloads (`proxima-runtime/src/concurrency/controller.rs:152`). No continuous watch-and-converge loop found in `src/daemon_control_plane.rs` by `grep -i reconcil` (matches only the three files above plus unrelated model-interop) |
| Scheduler (node placement) | absent: `grep 'schedule_pod|node_selector|bin_pack|pod_placement|fn schedule\('` -> 3 files, none placement (`proxima-patterns/src/alert/scheduled_trigger.rs:312`, `examples/backoff/main.rs:66`, `omega/tests/gemma4_rows_support/mod.rs:114`) |
| Replicas | absent: `grep 'ReplicaSet|desired_replicas|replica_count'` -> 0 files (`replicas` hits are Kafka metadata, `proxima-kafka/src/wire.rs:456`) |
| Health probes + restart | partial: `RestartPolicy {Always, Never, OnFailure}` and `ReadyProbe {None, StdoutLine, Tcp}` for child processes (`src/upstreams/process.rs:48,62`); `restart_count` in `PipeStatus` (`proxima-patterns/src/control_plane.rs:53`); process-RPC respawn on death (`src/upstreams/process_rpc.rs:418`). No periodic liveness/readiness probe: `grep 'readiness_probe|liveness_probe|probe_interval|HealthCheck|health_check'` -> 0 files. Passive outlier ejection: `OutlierPolicy {eject_after_failures: 5, recovery_after_successes: 2}` (`proxima-patterns/src/balancer/upstream_ref.rs:74-87`) |
| Rolling update | absent: `grep 'rolling_update|max_surge|RollingUpdate'` -> 0 files. In-place swap exists: `apply` swaps a pipe to a new spec (`src/daemon_control_plane.rs:315`, test `:505`); `reload` (`src/server.rs:213`) |
| Service discovery / DNS | partial: DNS client + listener and RFC 1035 codec (`proxima-dns/src/lib.rs:1`), DNS client upstream (`src/upstreams/dns.rs:1`). No cluster service registry that serves service names to workloads (no grep established one; not searched beyond the files above) |
| Load balancing across replicas | present: `Selection` impls `RoundRobin` (`proxima-patterns/src/balancer/selection.rs:230`), `WeightedRoundRobin` (`:279`), `WeightedLeastConn` (`:345`), `LeastConn` (`:395`), `Fallthrough` (`:111`); `UpstreamRef` per-upstream weight and metrics (`upstream_ref.rs:98`). Source of endpoints is static config, not discovered |
| Ingress | partial: HTTP listener with path mounts (`examples/lambda/proxima.toml:12-22`); reverse proxy to an `http` upstream measured in P4. Host-header routing: `grep -E 'host_match|virtual_host|host_route|sni_route|by_host' --include=*.rs` -> only AMQP `virtual_host` (`proxima-amqp/src/method.rs:103`), not HTTP routing. No Ingress resource reconciliation |
| Persistent storage | absent: `grep 'PersistentVolume|volume_claim|VolumeClaim|csi_driver'` -> 0 files. (`proxima-config` StateStore persists config only) |
| Secrets / config | partial: layered config via conflaguration with `#[setting(sensitive)]` (example use `proxima-centauri/src/config.rs:13`); auth tokens in `proxima-auth/src/token.rs`. No Secret/ConfigMap object: `grep 'SecretStore|secret_ref|from_secret|SecretRef|sealed_secret'` -> 0 files |
| Multi-node join | absent: `grep 'join_token|node_join|ClusterMember|gossip|memberlist|openraft|raft_log'` -> 0 files. Remote control exists as `proximad serve --stdio` over SSH (`tools/proxima-cli/src/pipeline.rs:2-4`), one daemon per host |
| Metrics | partial: `proxima-telemetry` counters/gauges/histograms (benches `proxima-telemetry/benches/bench_counter.rs`, `bench_histogram.rs`), `MetricsSnapshot` per pipe (`proxima-patterns/src/control_plane.rs:30`). `prometheus` is a dependency (`proxima-telemetry/Cargo.toml:69`) but `grep -rn -i prometheus --include=*.rs proxima-telemetry/src` printed nothing, so no exporter found there. No metrics-server equivalent API |
| OCI image pull / run | absent: `grep 'oci-spec|ociregistry|image_pull|registry-1\.docker|docker\.io|oci_runtime|OciImage'` -> 0 files. Guest units are a bare ELF or a Linux kernel + static `/init` booted under KVM per request (`src/upstreams/vm.rs:323,501`; `examples/lambda/proxima.toml`); host child processes via `src/upstreams/process.rs` and `proxima-process` |

## Not comparable (what each row measures differently)

- **K3 workload start vs P3 per-request guest.** k3s rows time creating a long-lived pod (scheduler, kubelet, containerd, pause + app container, then Running). proxima rows time one HTTP request that boots a fresh KVM guest (bare ELF for /elf, kernel + init for /linux), runs it, and returns; the guest does not persist after the response. A pod's steady-state request cost is K4, not K3. The proxima row closest to a pod's steady-state is P4, where proxima forwards to an existing backend.
- **K1 vs P1.** k3s timings end at a Kubernetes Ready condition on an API object; proxima timings end at the first 200 from an HTTP listener (the `/elf` run also boots one guest; the `/linux` run boots a kernel). The k3s first-boot row includes image pulls over the network; no proxima row pulls anything. k3s rows carry the `kubectl` poll spawn cost; proxima rows carry the `curl` spawn and `systemd-run` start. loadavg during k3s warm runs (12) is higher than during proxima runs (9 to 10), and both are shared-box values.
- **K2 vs P2.** The k3s figure sums a control plane, runtime, and five system pods; the proxima figure is one process with no workloads and one request served. proxima idle RSS is also sampled after a guest boot (mapped guest memory is not separately attributed).
- **K4(a) vs P4.** Traefik runs in a pod and routes to pod endpoints; proxima runs on the host and was pointed at the NodePort (kube-proxy DNAT to a replica), so the backend hop differs (service-endpoint dial vs NodePort DNAT). Server count per client request is the same (one hop), but the hop paths are not identical. The traefik and proxima loads ran at different loadavg (7 to 19 vs 17 to 28) because other agents shared the box; the NodePort and ClusterIP rows ran at 21 to 36.
- **K4(a) latency p99 at c=64 vs P4.** The P4 c=64 rows have multi-second max latency with sub-millisecond p50/p99 in two of three runs; oha records that as one distribution but the cause is unexplained, so the c=64 max column is not interpretable.
- **K5 vs P3.** Pod memory is a per-resident-pod cost; proxima has no resident per-workload unit, so there is no proxima row for K5.
- **First-boot k3s (K1, n=1)** is a one-shot including DNS and registry latency to the internet; it is not repeatable on the same data dir and has no p99.
- **Function table.** "present/partial/absent" is a code-read inventory against k3s's feature list, not a behavioral test of either side.

## Run hygiene

- Disk: `/home` free 12 GiB before, 11 GiB at the end; k3s data dir peaked at 796 MiB plus the whoami image.
- k3s reused the pre-existing bundled binary directory `/var/lib/rancher/k3s/data/f10c229b8445.../bin` read-only (containerd shim path in `ps` output); `find /var/lib/rancher /etc/rancher/node -newermt '2026-10-31 17:40'` printed nothing after the runs.
- k3s created a symlink `/etc/rancher/k3s/k3s.yaml` pointing into the bench dir; removed.
