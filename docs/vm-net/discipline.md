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

**Wake-count proof (counting test + control):**
`frame_stack::tests::a_frame_for_one_connection_wakes_no_other_idle_connection`
(8 established connections, each parked in `poll_read` with its own counting
waker; one `ping` written to connection 3; the pump polled once).
- control on the incumbent (c43fca60, same test against the old API): wakes per
  connection `[1, 1, 1, 2, 1, 1, 1, 1]`, assertion `left: 7 right: 0` fails —
  every bystander woken by one frame.
- after (feat/frame-pump): bystander wakes 0, connection 3 woken exactly once.
Status of the mechanism: **measured** (the count follows the toggle).

### Changelog
| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-11-01 | baseline: incumbent wake-all pump through `proxima serve`, c=1/8/64 | baseline | 1 run per arm (CoV not yet computed) | loadavg 4-5; my own builds idle at the time |
| 2026-11-01 | single pump future + per-ConnId wakers (feat/frame-pump 472b3d01..94a0afe9) | bystander wakes per frame 7 → 0 at K=8; e2e CPU/request pending | counting test, deterministic | n/a |
| 2026-11-01 | e2e after C1 + C2 landed (release e32ad0ec, `proxima serve`, same config and method as the baseline) | CPU/request c=1 266 → 218 us, c=8 325 → 184 us, c=64 2,346 → 190 us; rps c=1 4,567 → 5,302, c=8 5,665 → 6,645, c=64 (8 s) 3,238 → 6,577; c=64 15 s ×3: 6,450 / 6,450 / 6,484 rps, p50 10.0-10.2 ms, max 25.9-32.3 ms (was 8,252 ms); all 200 | 1 run per CPU row; 3 runs at c=64 (range 0.5%) | loadavg 1.98-9.80 |

**After C1 + C2, against the home-turf arm (same session, alternating, c=64 15 s):**
traefik + whoami ×2: 27,155 / 27,486 / 27,486 rps, p50 1.36-1.39 ms, max 30.9-45.7 ms.
proxima resident whoami (one guest, one vCPU): 6,450-6,484 rps, p50 ~10 ms.
proxima is at about 0.24× traefik's throughput at c=64. The p50 follows from the
ceiling (64 in flight / ~6,500 rps ≈ 9.8 ms). The whole proxima process used
999 ticks over 8 s at c=64 (about 1.25 cores), so host CPU is no longer the
limit. Hypothesis, unmeasured: the ceiling is the single guest vCPU running
whoami plus the guest kernel's network stack, against k3s's two replicas on the
host kernel. Per-thread CPU at c=64 is the next measurement.

**Per-thread CPU at c=64 (10 s, 6,540 rps):** `proxima-vm-resident-vcpu` 1001
ticks (one core fully busy); the 8 `proxima-core` threads 13-34 ticks each.
Measured: the ceiling is the guest's single vCPU thread.
**Split of that vCPU core** (`/proc/<tid>/stat` fields 14, 15, 43; 10 s, 6,503
rps): utime 787, stime 208, guest_time 659. Guest work 66%, host device
emulation on the vCPU thread 13% (787 − 659), host kernel exit handling 21%.

**Two resident guests behind `select = { algorithm = "round_robin" }`** (one
composed pipe with two inline `type = "vm"` upstreams, `$M/resident-x2.toml`),
c=64 15 s ×3 alternating with traefik:

| arm | rps | p50 ms | p99 ms | max ms | loadavg 1m | design-favors |
|---|---|---|---|---|---|---|
| proxima, 2 guests × 1 vCPU | 12,668 / 12,824 / 12,665 | 1.51 / 1.66 / 1.38 | 18.6 / 18.3 / 18.7 | 43.9 / 233.8 / 21.4 | 1.08 / 7.08 / 16.41 | neutral |
| traefik + whoami ×2 | 26,977 / 27,370 / 27,181 | 1.38 / 1.37 / 1.37 | 12.3 / 11.9 / 12.2 | 41.3 / 30.4 / 48.2 | 1.36 / 6.26 / 13.42 | incumbent |

Both vCPU threads ran at about 100% of a core (1232+260 and 1191+300 ticks over
15 s). VmRSS 431 MB with two guests (290 MB with one).

**CPU per request, both sides, c=64 15 s** (utime+stime deltas, 100 ticks/s):

| arm | rps | ok | ticks | CPU us/request | cores used | loadavg 1m |
|---|---|---|---|---|---|---|
| traefik 3303 + whoami 2328 + whoami 2313 + k3s-server 29 + containerd/shims 6 | 27,394 | 410,903 | 7,979 | 194.2 (excludes kernel softirq: iptables, conntrack, host net stack) | ~5.3 | 4.60 |
| proxima, whole process, one resident guest (guest time included) | 6,646 | 99,647 | 1,866 | 187.3 | ~1.24 | 6.81 |

Measured: CPU per request is about equal; the traefik figure omits kernel
network time, so its true cost per request is higher than 194 us. The
throughput gap is the number of cores each side can use (one guest vCPU vs
~5.3 cores), and two guests doubled proxima's rps (6.5k to 12.7k).

**Exits per request** (1,000 requests at c=1, all 200, `RUST_LOG=proxima_vm=debug`,
counts from the journal after a cursor taken post-startup,
`$M/ex-journal.txt`): mmio interrupt-path access 5.64, net rx queue filled
2.06, rx serviced on doorbell 2.00, queue notify 1.06, net tx queue drained
1.00. With the vCPU split above (13% host emulation + 21% host kernel per
vCPU core), the exits are the candidate for the third of the core not spent in
the guest; attribution of those 34% to these specific exits is unmeasured.

**Implication:** two levers. (1) Let a workload use more cores: replicas or
guest vCPUs. Today, N resident upstreams in one composed pipe with
round-robin is expressible in config. (2) Fewer exits per request: virtio
notification suppression / event index and interrupt acknowledgement.

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

## C2 — TcpStack byte path

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pending | pending | pending | `bench_tcp_stack.rs` (8f6df058) | baseline only | pending | pending | pending | **fails: per-byte BTreeMap** | n/a | kernel loopback arm present | — | principles 1, 11, hot-path zero-alloc |

**Incumbent design point:** host-kernel TCP (loopback, `TCP_NODELAY`, one
keep-alive request/response per iteration across two threads) — built for
bulk byte streams; its cost is per-call (syscalls, wakeups), nearly flat in
payload size.

**Baseline (first run, loadavg 15.31, no same-code control yet):**

| arm | 54 B | 402 B | 1514 B | design-favors |
|---|---|---|---|---|
| `on_inbound_to` + `read` | 1.468 us | 19.63 us | 89.77 us | neutral |
| `write` + inbound response + `read` | 1.551 us | 20.06 us | 90.69 us | neutral |
| kernel loopback round trip (two threads, 2 syscalls each way) | 13.01 us | 13.07 us | 14.55 us | incumbent |

proxima's cost grows ~49 ns per payload byte (54 → 402 B adds 18.2 us for 348
bytes); the kernel's grows ~1.1 ns per byte (54 → 1514 B).

**Mechanism (read from code, status plausible until the fixed bench lands):**
`proxima-net/src/tcp_stack.rs:538-541` inserts every payload byte into
`pending: BTreeMap<u32, u8>`; `:554-567` removes each byte and `push_back`s it
into `recv: VecDeque<u8>`; `read` (`:322-328`) pops one byte per iteration.
Send side: `send_buf` is never trimmed (`:589-590` comment, `:599`
`to_vec` per segment), so a keep-alive connection retains every byte it ever
sent, and `sent: u32` indexes it.

**Baseline saved:** criterion `--save-baseline per-byte`, 3 runs, worktree
`proxima-tcp-stack` (branch `perf/tcp-stack-bytes` from 5fcf47e2).

**Per-byte baseline, 3 runs (loadavg 4.3-8.5):** `on_inbound_to`+`read` 54 B
1.449-1.463 us, 402 B 19.49-20.47 us, 1514 B 90.59-94.64 us;
`write`+response+`read` 54 B 1.533-1.682 us, 402 B 19.79-23.00 us, 1514 B
90.23-98.29 us; kernel loopback 12.83-14.89 us.

### Changelog
| Date | Change | Δ vs per-byte (criterion `--baseline per-byte`, default bench profile) | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-11-01 | bulk `read` (5d375627) + per-segment receive hold, bulk window cap, acked send bytes released, `DataPath::snd_una` (f28b40ff); landed as 53428277 | `on_inbound_to`+`read`: 54 B 63.2 ns (−95.7%), 402 B 71.8 ns (−99.6%), 1514 B 82.8 ns (−99.9%); `write`+response+`read`: 54 B 119.3 ns (−92.2%), 402 B 139.3 ns (−99.3%), 1514 B 185.3 ns (−99.8%); same-code control (kernel loopback, unchanged code): −3.2% / −5.6% / +1.1% (p = 0.46) | 1 run; the same-code arm sets the noise floor at about ±6% | loadavg 34.9 at build start, 12.76 at run end; other sessions active |

Mechanism of the delta: the cost was per payload byte (one `BTreeMap` insert
and remove plus two `VecDeque` operations per byte); after the change the
slope is ~0.01 ns per byte (402 → 1514 B adds 11 ns), i.e. bulk copies. Tests:
`cargo val -p proxima-net --features frame-stack,xdp` 74/74, including the six
new cases (gap fill, duplicate/overlap, held-segment overlap, bounded send
buffer across many acked writes, receive and send across u32 sequence wrap).

**Scope of the incumbent arm:** the kernel arm is a cross-thread loopback round
trip through syscalls; the proxima arms are the in-process sans-IO stack with
no thread hop. They are not the same scope, so the gap between them is not a
like-for-like verdict. The comparable number is the end-to-end CPU per request
through `proxima serve` (C1 table above), re-measured after both C1 and C2 land.

## Related defects found while measuring (not C1)

| Defect | Evidence | State |
|---|---|---|
| resident guest killed 10 s after boot by the per-request watchdog; every later request 502 | idle control: GET at 5 s → 200, at 12 s → 502; c=64 runs: 2,712,411 / 3,614,190 / 3,702,256 502s | fixed in d89a5acc (resident lane does not arm it); `resident_whoami` with a 1 ms watchdog fails on the old code (`watchdog forced exit`) and passes after |
| guest end was silent (only `frame source failed: Closed` downstream) | journal of the 502 runs | fixed in b6217db1 (error event with failure, end, exits) |
| guest RAM ~fully resident | live smaps: 129,792 kB of 131,072 at 128 MiB; 158,972 of 262,144 at 256 MiB; `init_on_alloc=0` 123,100; pagemap: one present run pages 192-28082 (108 MiB) + top 18 MiB | cause unmeasured; next: guest kernel `Memory:` line at loglevel 6 |
| host heap ~150 MB beside the guest | `[anon:mimalloc]` Rss 136,736 (guest already dead) / 156,172 / 173,124 kB | cause unmeasured |
| start to first 200: proxima p50 ≈ 2,084 ms vs k3s pod create to first 200 p50 ≈ 1,010 ms | `$M/p-start.txt`, `$M/k-start.txt`, n=10 each, loadavg 1.2-1.8 | cause unmeasured |
| `TcpStack` has no timer entry point: our lost data segment is never resent | `proxima-net/src/tcp_stack.rs` public API (`:121-370`) has no tick/timeout | candidate for the 8.25 s c=64 max; unproven |
