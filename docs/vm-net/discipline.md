# vm-net discipline log

The 13-point gate (see `.claude/skills/disciplined-component/SKILL.md`):
build / tests / clippy / micro-bench / compare-bench / E2E / opt-sweep /
SIMD-SM-noBox / O(1) / Cfg-API / home-turf / Δ.

Scope: the host side of a resident Linux guest's network path —
`proxima-net/src/frame_stack.rs` (sans-IO `TcpStack` driven over three frame
pipes) as used by `tools/proxima-vm/src/kvm/host_wire.rs` and
`src/upstreams/vm_resident.rs`. Feature gate: `proxima-net/frame-stack`
(default-off). Host: `versailles`, nproc 8, 15 GiB, shared with other agents;
loadavg is recorded per row. Every number below is a measurement; a mechanism
is stated only where a counterfactual or trace supports it.

## Two workloads, not one

These must not share a headline:

| Workload | proxima role | k3s role | latest c=64 |
|---|---|---|---|
| D1 ingress (`tools/proxima-vm/K3S_BASELINE.md` "D1") | host reverse proxy to the whoami pod endpoints | traefik to the same endpoints | proxima 20,093 / 24,301 / 31,044 rps vs traefik 11,821 / 20,499 / 19,653 (loadavg 39-41, 2026-10-31) |
| resident workload (this log) | runs whoami itself in one KVM guest, 1 vCPU, 128 MiB | whoami x2 pods behind traefik | proxima 2,954 rps vs traefik 27,022 / 27,273 / 27,323 rps (2026-11-01) |

## C1 — frame-stack pump: one Rx consumer, per-connection wakes

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pending | pending | pending | pending | pending | baseline only | pending | pending | pending | n/a: no new tunable | pending | — | principles 1, 3, 11, 20, 21; slice 1 in worktree `proxima-frame-pump` |

**Incumbent design points:**
- wake-all `Waiters` fan-out (`frame_stack.rs` at 54588cc0: `pump` pushes the caller's waker every call, `impl ArcWake for Waiters` wakes every parked waker after any burst that moved a frame). Built for: one or a few connections, where waking everyone is cheap and no task can be left stale.
- host-kernel TCP into traefik + whoami x2 pods (k3s). Built for: many concurrent keep-alive connections on a multi-core host stack. This is the home-turf arm at c=64.

**Frequency bands:**
- 80% (per frame, per request): Rx frame ingest + wake of the addressed reader; read/write of a connection's bytes.
- warm: dial (ARP + SYN), accept.
- cold: ARP/SYN retransmit, `SynUnanswered`, resident guest ending.

**Defect being fixed (incumbent arm, measured):** CPU per request grows with
open connections. Release binary (`lto = "fat"`, `codegen-units = 1`), `proxima
serve` with `$M/resident.toml` (resident `traefik/whoami:v1.10.2`), oha 8 s per
arm, process CPU = delta of utime+stime of the proxima pid:

| arm | rps | ok | p50 ms | CPU us/request | loadavg 1m | design-favors |
|---|---|---|---|---|---|---|
| incumbent wake-all, c=1 | 4,567 | 36,610 | 0.212 | 266 | 3.98 | neutral |
| incumbent wake-all, c=8 | 5,665 | 45,314 | 1.390 | 325 | 4.14 | neutral |
| incumbent wake-all, c=64 | 3,238 | 25,900 | 3.258 | 2,346 | 4.99 | incumbent (traefik) |

Raw: `/mnt/data2/proxima-k3s/measure/herd-scaling.txt`. Per-thread CPU over a
10 s c=64 run (`thr-before.txt` / `thr-after.txt` there): `proxima-core-0..7`
807-919 ticks each (about 6.97 cores), `proxima-vm-resident-vcpu` 641 ticks,
3,010 rps. The vCPU thread is not the saturated one.

Status of the mechanism ("every frame wakes every parked connection, each
takes the driver lock and pumps"): **plausible** — read from the code and
consistent with the 8.8x CPU/request growth from c=1 to c=64, not yet proven
by a wake count. The slice-1 counting test is the proof, with its control run
against 54588cc0.

**Home-turf arm (traefik/whoami x2 on k3s, same session):**

| arm | rps (3 runs) | p50 ms | p99 ms | max ms | loadavg 1m |
|---|---|---|---|---|---|
| traefik c=64 15 s | 27,022 / 27,273 / 27,323 | 1.377 / 1.378 / 1.360 | 11.87 / 11.82 / 11.73 | 83.4 / 43.5 / 65.8 | 3.03 / 12.11 / 15.87 |
| traefik c=1 n=2000 | 1,431 / 1,929 / 2,074 | 0.465 / 0.231 / 0.223 | 1.857 / 1.556 / 1.543 | 20.2 / 14.5 / 3.9 | 0.83 |
| proxima c=1 n=2000 (incumbent pump) | 4,539 / 4,672 / 4,554 | 0.208 / 0.207 / 0.214 | 0.344 / 0.350 / 0.305 | 7.4 / 1.5 / 1.5 | 0.83 |

Unexplained: traefik c=1 here (1.4-2.1k rps) is below its own 2026-10-31 value
on the same box (4,755-5,220 rps, K4 row a). Not traced.

**Opt-sweep findings:** pending (slice 1).
**SIMD/SM/no-Box pass:** pending.
**O(1):** target — per-frame work independent of open-connection count; the
counting test asserts it.

### Changelog
| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-11-01 | baseline: incumbent wake-all pump through `proxima serve`, c=1/8/64 | baseline | 1 run per arm (CoV not yet computed) | loadavg 4-5; my own builds idle at the time |

## Per-piece decomposition (criterion, one bench file per piece)

The end-to-end c=1 cost is 266 us of proxima CPU per request. These pieces are
benched in isolation first, each against a named incumbent, sizes taken from
the frames one whoami GET actually moves (`net tx queue drained ... lengths=[54]`
ACK, `[402]` response, plus 1514 B full frame).

| Component | Bench file | Per-request op | Incumbent arm (design-favors: incumbent) | State |
|---|---|---|---|---|
| C2 TcpStack segment cycle | `proxima-net/benches/bench_tcp_stack.rs` | inbound segment → stack → outbound ACK/data; `write` + `read` of one request | host-kernel TCP over loopback, one keep-alive request/response round trip | pending |
| C3 frame channel | `tools/proxima-vm/benches/bench_frame_pipe.rs` | one frame send + recv across threads (plain and waking) | `crossbeam_queue::ArrayQueue` + `AtomicWaker`; `std::sync::mpsc::sync_channel` | pending |
| C4 virtio-net queue | `tools/proxima-vm/benches/bench_virtio_net_queue.rs` | guest TX chain drain → frame; frame → guest RX chain fill | no comparable micro surface (the incumbent is the host kernel's virtio backend); floor arm: `memcpy` of the same bytes | pending |
| C1 frame-stack pump | `proxima-net/benches/bench_frame_stack_pump.rs` | one frame to one of K open connections, K = 1 / 8 / 64 | incumbent wake-all at 54588cc0 (`--save-baseline wake-all`) | slice 1 in flight |

## Related defects found while measuring (not C1)

| Defect | Evidence | State |
|---|---|---|
| resident guest killed 10 s after boot by the per-request watchdog; every later request 502 | idle control: GET at 5 s → 200, at 12 s → 502; c=64 runs: 2,712,411 / 3,614,190 / 3,702,256 502s | fixed in d89a5acc (resident lane does not arm it); `resident_whoami` with a 1 ms watchdog fails on the old code (`watchdog forced exit`) and passes after |
| guest end was silent (only `frame source failed: Closed` downstream) | journal of the 502 runs | fixed in b6217db1 (error event with failure, end, exits) |
| guest RAM ~fully resident | live smaps: 129,792 kB of 131,072 at 128 MiB; 158,972 of 262,144 at 256 MiB; `init_on_alloc=0` 123,100; pagemap: one present run pages 192-28082 (108 MiB) + top 18 MiB | cause unmeasured; next: guest kernel `Memory:` line at loglevel 6 |
| host heap ~150 MB beside the guest | `[anon:mimalloc]` Rss 136,736 (guest already dead) / 156,172 / 173,124 kB | cause unmeasured |
| start to first 200: proxima p50 ≈ 2,084 ms vs k3s pod create to first 200 p50 ≈ 1,010 ms | `$M/p-start.txt`, `$M/k-start.txt`, n=10 each, loadavg 1.2-1.8 | cause unmeasured |
| `TcpStack` has no timer entry point: our lost data segment is never resent | `proxima-net/src/tcp_stack.rs` public API (`:121-370`) has no tick/timeout | candidate for the 8.25 s c=64 max; unproven |
