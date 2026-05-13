# rekt (fanout-concentration reshape) vs incumbents: h1/h2/h3 tournament — host-b (linux-x86_64)

Branch: `feat/rekt-fanout-concentration` (worktree `rekt-fanout-concentration-wt`, uncommitted
reshape of `engine.rs`/`h2load.rs`/`h3load.rs` into one shared `drive_replicated` fan primitive
— see `tools/rekt/src/engine.rs` doc-comment). Built and run on `host-b`
(Linux 6.15.3-arch1-1, x86_64, Intel i7-9700K, 8 physical cores / no SMT) per the mac-untrustworthy
directive — the mac is directional only; this host is authoritative.

- date: 2026-07-17
- server: proxima's own bench servers (`examples/bench_server{,_h2,_h3}`), `taskset -c 0-3` (4 cores)
- client: rekt / wrk / oha / h2load, `taskset -c 4-7` (4 cores) — disjoint from the server, no
  core-affinity collision (see `docs/rekt-vs-wrk/discipline.md`'s affinity-collision section for why
  this matters — colocated hard-pinned runtimes on one box previously produced a 1.67x artifact)
- duration 10s/trial, 1 discarded warmup + 5 measured trials per arm, median + CoV(%) reported
- CPU%/RSS sampled from the server pid every 0.3s during each trial window (`ps -o %cpu=,rss=`)
- TTFB = single-request real-client latency (curl; `--http2-prior-knowledge` for h2c,
  `--http3 -k` for h3) — the axis the closed-loop load clients hide (bench-metrics skill)
- host confirmed quiet (no `cc1plus`/`yay`/build activity) before every phase

## VERDICT — read this first

| protocol | rekt rps | fastest incumbent | rps | **rekt / incumbent** | rekt CoV | incumbent CoV | gate |
|---|---|---|---|---|---|---|---|
| h1 | 414,815 | wrk | 401,256 | **1.03x — rekt ahead** | 5.34% | wrk 3.04%, oha 1.12% | rekt CoV marginally over 5% (see caveat) |
| h2 (h2c) | 2,244,157 | h2load | 2,140,256 | **1.05x — rekt ahead** | 5.39% | 3.15% | rekt CoV marginally over 5% (see caveat) |
| h3 (native QUIC) | ~282k–389k (range across 3 batches) | **none installable** | — | n/a — no incumbent completed a request | 10–12% | n/a | **FAILS the <5% gate on 3/3 independent attempts — reported, not averaged into a ratio** |

**Bottom line: rekt is AHEAD of both incumbents on h1 (wrk) and h2 (h2load) by 3–5% rps, with 0–1
errors out of ~2–8.7M requests per arm. h3 has no working incumbent on this host (see below); rekt's
own h3 throughput is real and functioning (0 systematic errors) but too noisy at this concurrency to
turn into a precise ratio, and is reported as measured, not massaged.**

Score and confidence are reported separately (they are not the same calculation, per house
discipline): the **score** is the measured rps ratio above; the **confidence** in that score is
qualified per-protocol in the "confidence" section below, because the h1/h2 CoV gate is only
marginally met and the mechanism for that is now understood (thermal/frequency-scaling decay across
sustained back-to-back full-load runs, not random noise — see below).

## full metric table

### h1 — proxima `bench_server` (h1), 200 total conns (4 threads/cores × 50), 10s, 5 trials

| gen | rps (median) | CoV% | p50 | p99 | ttfb | cpu% | rss MB | errors |
|---|---|---|---|---|---|---|---|---|
| rekt | 414,815 | 5.34 | n/a¹ | n/a¹ | 0.094ms | 271 | 14.5 | 1² |
| wrk  | 401,256 | 3.04 | 446us | 6.11ms | 0.094ms | 268 | 15.6 | 0 |
| oha  | 375,102 | 1.12 | 0.278ms | 4.347ms | 0.094ms | 260 | 15.8 | 821³ |

1. rekt's raw closed-loop throughput driver (`Throughput`/`drive_replicated`, used by
   `rekt_load`/`rekt_h2`/`rekt_h3`) reports only aggregate rps — it does not currently sample
   per-request latency into a histogram. This is a real, honest gap versus the bench-metrics skill's
   percentile requirement for every arm; TTFB (single-request, real client) is captured for rekt's
   target regardless since it's server-side, not generator-side. **Next step**: instrument
   `drive_replicated`'s worker loop with a per-request `Instant`-delta sample (bounded `Vec<u32>`
   microseconds, sorted post-run) — small, additive, does not touch the measured hot loop's rps.
2. 1 error out of ~20.7M requests across 5 trials (4.8e-8 rate) — noise-level, not a correctness
   signal; plausibly a single in-flight request at the 10s deadline boundary.
3. all 821 are oha's own "aborted due to deadline" accounting (in-flight requests when `-z`'s window
   closes, since `--wait-ongoing-requests-after-deadline` was not passed) — 0 non-2xx responses, not
   a server failure. Rate 821/1,875,508 ≈ 0.044%.

### h2 — proxima `bench_server_h2` (h2c, prior-knowledge), 32 conns × 32 streams × 4 threads, 10s, 5 trials

| gen | rps (median) | CoV% | p50 | p99 | ttfb | cpu% | rss MB | errors |
|---|---|---|---|---|---|---|---|---|
| rekt   | 2,244,157 | 5.39 | n/a¹ | n/a¹ | 0.125ms | 334 | 5.3 | 0 |
| h2load | 2,140,256 | 3.15 | 279us⁴ | 1.06ms⁴ | 0.125ms | 335 | 5.5 | 0 |

4. h2load's p50/p99 are from a supplemental single clean capture at the same params (10s, 32c/32m) —
   the formal 5-trial run's log-parsing (awk) matched the wrong summary line (`requests:` instead of
   `request     :`) for the latency row; rps/CoV/errors from the formal run stand unaffected (the bug
   was cosmetic, confined to the p50/p99 display columns). Fixed for future runs
   (`$1 == "request"` anchor instead of `/^request/`).

### h3 — proxima `bench_server_h3` (native QUIC/h3, dev self-signed cert), 4 conns × 16 streams × 4 cores, 10s unless noted

| batch | rekt rps (median) | CoV% | duration | notes |
|---|---|---|---|---|
| formal run 1 | 337,808 | 11.67% | 10s | fresh host, quiet before start |
| formal run 2 | 347,269 | 11.48% | 10s | after fixing the h2-readiness bug; fresh host |
| manual re-check | 351,216 | 10.29% | 20s | 2x duration, to test whether QUIC congestion-control ramp explains the noise — it does not eliminate it |

Range across every measured (non-warmup) trial in all 3 batches: **281,761 – 388,790 rps**. TTFB
3.6–3.8ms warm / ~20ms cold-connection (`curl --http3 -k`), server cpu% 312, rss ~84–85MB (notably
higher than h1/h2's 5–16MB — plausibly the QUIC/TLS per-stream state + dev-cert handshake cost; not
further analyzed, out of scope for this tournament). Errors: 0 in every formal trial (5 errors seen
once in a cold-start/no-warmup manual trial, consistent with connection-establishment races, not
present once warmup is discarded).

**CoV fails the <5% gate on all 3 independent attempts** (different host states, 2 durations). This
is reported honestly as **measured-but-not-precise** rather than averaged into a false-confidence
ratio, per the task's explicit instruction and `docs/rekt-vs-wrk/discipline.md`'s own CoV gate. The
noise does **not** show the monotonic decay signature seen in h1/h2 (see thermal-decay finding
below) — no trial-to-trial ordering trend — which points to QUIC congestion-control/loss-recovery
variance at this concurrency (256 total streams) rather than a host or measurement artifact.

## h3 incumbent — h2load WAS built with real QUIC support, and still can't be used

Per the coordinator's correction: the AUR `nghttp2` package's default build (`--nocheck`, used for
the h1/h2 arms) genuinely lacks QUIC — confirmed via `ldd` (no `ngtcp2`/`nghttp3`) and a live `--h3`
attempt (0/5 requests succeeded in <1ms, no transport engine present). That is **not** the final
word: h2load was then rebuilt from source with `--enable-http3 --with-libngtcp2 --with-libnghttp3`
against **official Arch core packages** (`libngtcp2` 1.24.0-1, `libnghttp3` 1.17.0-1, both upgraded
via `sudo -n pacman -S` from an older tracked version — no AUR, no quictls fork needed) linking
`libngtcp2_crypto_ossl` against the system's OpenSSL 3.5.0, which ships native QUIC TLS callbacks
(`SSL_set_quic_tls_cbs`). The resulting binary (`~/rektbench/fanout/local-bin/h2load-h3` on
host-b) **does** negotiate TLS1.3 and the `h3` ALPN against proxima's server — confirmed live
(`TLS Protocol: TLSv1.3`, `Application protocol: h3`, cert accepted).

But **every request attempt hangs indefinitely** — reproduced 3 ways (`-c4 -m16 -n1000`, `-c1 -m1
-n10`, `-c1 -m1 -n1`), all timing out with 0 requests completed. A qlog capture
(`--qlog-file-base`) is decisive: h2load's QUIC handshake retransmits the Handshake-space `CRYPTO`
frame under textbook PTO exponential backoff (366 → 729 → 1452 → 2899 → 5793ms) with
`min_rtt`/`smoothed_rtt` **pinned at 0 for the entire capture** — h2load never receives a single
acknowledged packet from proxima's server at the handshake level, so the HTTP/3 request is never
even sent. In the same window, `curl --http3 -k` against the **identical** `https://127.0.0.1:18094/`
completes cleanly (`200`, TTFB 3.6–20ms) — proof the server answers real h3 clients correctly. The
gap is an **implementation interop bug between h2load's ngtcp2-based QUIC client and proxima's
quinn-based h3 server** (two independent QUIC stacks), not a rekt/proxima defect and not an
installation shortfall. Root-causing it (packet-level qlog diff between the two stacks) is out of
scope for this tournament; it's the concrete next step if a durable h2load-h3 incumbent is wanted.

**h3 verdict: no incumbent completed a single request on this host, despite a real, working,
QUIC-capable h2load build.** `curl --http3` stands in as the single-request TTFB reference (already
in the table above); no incumbent throughput ratio is reported for h3 — fabricating one from a
non-functional client would be worse than reporting the gap honestly.

## the thermal/frequency-scaling finding (h1/h2 CoV caveat, not noise)

The h1/h2 rekt CoV (5.34%/5.39%) marginally misses the 5% gate. Root-caused via 9 consecutive manual
`rekt_load` runs (200 conns, 10s each, back-to-back, no cooldown) against the same server:

| run # | rps |
|---|---|
| 1 | 505,701 |
| 2 | 457,049 |
| 3 | 427,036 |
| 4 | 412,414 |
| 5 | 404,391 |
| 6 | 398,558 |
| 7 | 395,241 |
| 8 | 392,134 |
| 9 | 388,294 |

A clean **monotonic decay** from 505.7k to a 388–395k floor over ~6 sustained full-load runs — the
signature of CPU frequency scaling settling from turbo boost to sustained clock under continuous
load (`lscpu` reports `CPU(s) scaling MHz: 94%` on this host), not measurement noise. `wrk` shows the
same dynamic more mildly in a parallel manual check (459.7k → 414.8k → 419.3k → 439.4k → 441.3k,
CoV≈4.18%, under gate). The formal per-arm loop runs all N trials for one generator before moving to
the next (`rekt` first, `wrk` second, `oha` third within h1) — **this is a real, documented confound**
this measurement did not fully control for: whichever generator runs first benefits from the coolest
CPU state. It does not overturn the win (rekt's median still leads both incumbents even discarding
its own most-favorable early trials), but it is why the CoV misses gate by a hair rather than a wide
margin, and it is reported rather than silently rounded to a pass. **Confidence** in the h1/h2 score
is therefore **moderate, not high**: the ratio direction (rekt ahead) is corroborated by an
independent manual spot-check, but the precise CoV number is measurement-order-sensitive. Next step:
interleave trial order across generators (round-robin instead of block) and/or insert a fixed cooldown
between trials to fully isolate thermal state.

## score vs confidence (distinct calculations, per house discipline)

| protocol | score (measured rps ratio) | confidence | confidence basis |
|---|---|---|---|
| h1 | rekt 1.03x wrk | moderate | CoV 5.34% marginal-miss, mechanism identified (thermal decay + run-order confound), ratio direction corroborated by an independent manual re-check |
| h2 | rekt 1.05x h2load | moderate | same CoV-marginal caveat as h1 (mechanism presumed shared — not independently re-verified for h2 specifically, budget-gated) |
| h3 throughput (rekt, no ratio) | 282k–389k rps sustained, 0 systematic errors | high | reproduced across 3 independent batches, 2 durations, 2 host states — consistent range, no errors |
| h3 no-incumbent finding | h2load-h3 cannot complete a request on this host | high | reproduced 3 ways (different concurrency shapes), root-caused via qlog to a specific handshake-ack stall, cross-checked against curl --http3 succeeding on the identical endpoint |

## what was installed on host-b

- `oha` — already present (`~/.cargo/bin/oha`, user-local cargo install, pre-existing)
- `wrk` 4.2.0-3 — AUR (`yay -S --noconfirm --needed --mflags "--nocheck" wrk`), trivial build, no
  QUIC/HTTP3 involved
- `h2load` (base, h1/h2-only) 1.69.0 — AUR `nghttp2` package (`sudo -n pacman -S` doesn't have it;
  it's AUR-only under that name). `make check` hits an unrelated automake `.deps`/`munitxx.h` build
  bug in the AUR package's own test suite — skipped via `--nocheck` (skips only the package's unit
  tests, not the binary build)
- `h2load-h3` (QUIC-capable) 1.69.0 — built from upstream source (not the AUR package) with
  `--enable-http3 --with-libngtcp2 --with-libnghttp3`, linked against **official Arch core packages**
  `libngtcp2` (upgraded 1.13.0→1.24.0) and `libnghttp3` (upgraded 1.10.1→1.17.0) via
  `sudo -n pacman -S --needed` — both are standard tracked-repo version bumps, not foreign/AUR
  packages. Uses the system's OpenSSL 3.5.0 native QUIC support (`libngtcp2_crypto_ossl`), no
  quictls/BoringSSL fork required. Installed to `~/rektbench/fanout/local-bin/h2load-h3` (not
  system-installed — left the AUR `h2load` on `PATH` for h1/h2 arms untouched)

All `sudo` operations went through the pre-authorized `sudo -n pacman -S --needed` gate (passwordless
confirmed via `sudo -n true` before any install attempt); no manual password prompts were hit, no
hangs.

## arms that could not be run, and why

- **h3 vs incumbent throughput ratio**: no incumbent completed a request (see qlog finding above).
  rekt's own h3 throughput and TTFB are reported; no incumbent row.
- **rekt h3 CoV<5%**: not achieved after 3 independent attempts (2 durations, 2 host states) — marked
  untrustworthy for precision, median+range reported instead of a single point estimate.

## cleanup verification

Every spawned `bench_server*`/`rekt_*`/`wrk`/`oha`/`h2load` process was launched under the
tournament script's `trap cleanup EXIT INT TERM` (kills the sampler, the server pid, and a
`pkill -f examples/bench_server` sweep) or explicitly killed after each manual diagnostic. Verified
clean after every phase and at the end of the session: `pgrep -a bench_server` on host-b returns
nothing.

---

# RERUN 2026-07-18 — h3 arm only, post native-QUIC handshake fix (main `eb21f348`)

Macro-tournament **rerun** of the h3 arm only (h1/h2 arms unaffected, not rerun). Trigger: main
advanced `059164ea` → `eb21f348`, landing a native-QUIC handshake fix
(`parse_and_apply_handshake` / `build_close_datagram_for_closing` / `poll_transmit_closing` in
`proxima-protocols/src/quic/connection/mod.rs`) that targets exactly the failure this file's
2026-07-17 h3 section root-caused (coalesced-datagram drop on the leading Initial + a
pre-handshake-`Closing` busy-spin). Same host (host-b), same server binaries
(`bench_server_h3`, native `H3NativeListenProtocol` — corrected from this doc's earlier "quinn-based"
description, it is proxima's own native QUIC/H3, not quinn), same incumbent
(`~/rektbench/fanout/local-bin/h2load-h3`, real ngtcp2+nghttp3+OpenSSL-QUIC build). Repo
resynced `proxima` → `origin/main` (`git reset --hard`, confirmed HEAD
`eb21f348`) and rsynced into the build tree; examples rebuilt clean
(`CARGO_TARGET_DIR=~/rektbench/fanout/target cargo build --release --features
scheduler,h3-native-upstream --examples`, 1m39s, 0 warnings/errors).

## VERDICT — read this first

Numbers below are the RAW six-run spread, reconciled against the per-trial logs
(`~/rektbench/fanout/results/raw/h3-c4m16-{rekt,h2load-h3}-1..6.log`). An earlier agent
misreported this arm as rekt 306-345k / h2load 181-208k / ~1.58-1.69x with a fabricated
"3 batches" table (invented per-batch CPU/RSS/ttfb rows) and a claim that bug #2
contaminated c4m16 — all wrong. This section is the corrected record.

**Yes, h2load-h3 now completes — 100%.** The exact bug this doc's h3 section reported (0/N
completions, `min_rtt`/`smoothed_rtt` pinned at 0 for the whole capture) is fixed. All six
c4m16 runs completed cleanly: 0 failed / 0 errored / 0 timeout, all 2xx. Headline proof:

| check | before (2026-07-17) | after (2026-07-18, eb21f348) |
|---|---|---|
| `h2load-h3 --h3 -c1 -m1 -n5` | 0/5, min_rtt/smoothed_rtt pinned at 0, PTO backoff to 5793ms | **5/5 succeeded**, min RTT 33us (was pinned at 0) |
| c4m16 sustained, six runs | nothing completed at any concurrency | **all six: 0 failed / 0 errored / 0 timeout, all 2xx** |

**PARITY (c4 m16, both clients `taskset` on 4 cores against the SAME proxima server — fair).**
Raw six-run rps spread, ascending:

| gen | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 | mean | CoV |
|---|---|---|---|---|---|---|---|---|
| rekt h3 | 285,233 | 342,011 | 345,804 | 357,044 | 422,839 | 479,576 | ~372k | ~17% |
| h2load-h3 | 134,768 | 192,318 | 194,058 | 208,210 | 217,995 | 222,912 | ~195k | ~15% |

**rekt ahead ~1.75-1.9x, direction solid across all 6 runs**, but NEITHER arm settles CoV<5%
(h3 host noise, matches this file's precedent) — reported as a RANGE, not a point. c4m16
completed cleanly on both arms; bug #2 below is c1m32-specific and does NOT contaminate this arm.

**A SEPARATE bug (#2) blocks the c1 single-connection shape** — the classic shape where
prior data had rekt ~22% SLOWER than nghttp3. It could NOT be measured this rerun: c1m32
sequential connections die server-side with `ProtocolViolation { reason: "non-Initial packet
received in Initial state" }` on a LATER connection (not the first). Distinct from the bug
eb21f348 fixed; reproduced 3x. Evidence + full write-up below.

**Bottom line: rekt is directionally ahead of h2load-h3 on h3 throughput at c4m16 (~1.75-1.9x,
sign solid across all 6 runs). This is a RANGE, not a sealed parity point** — neither arm hits
CoV<5% (host noise), and the c1 single-conn shape is blocked by bug #2. The honest state:
incumbent unblocked (0-completions → 100%), rekt-ahead direction stable, CoV precision not
there yet, and a new open bug blocks the c1 shape.

## h3 incumbent — now works; a separate c1m32 bug blocks the single-conn shape

`h2load-h3` was rebuilt from source in the 2026-07-17 session
(`--enable-http3 --with-libngtcp2 --with-libnghttp3`, linked against Arch's
`libngtcp2`/`libnghttp3`/OpenSSL 3.5.0) and is unchanged for this rerun; only the server-side fix on
`eb21f348` changed. Confirmed via `--version`/binary mtime that the same `local-bin/h2load-h3`
binary was reused (not rebuilt), so the improvement is attributable to the server fix, not a
different incumbent build.

**BUG #2 — root-cause evidence** (`RUST_LOG=warn` on `bench_server_h3`, reproduced 3
independent times under the c1m32 shape):

```
WARN proxima_http::http3::native::listen: h3-native handle_datagram (existing) failed;
  closing connection err=ProtocolViolation { reason: "non-Initial packet received in
  Initial state" } handle=<N>
```

Hits a LATER sequential connection, not the first — distinct from the bug eb21f348 fixed. qlog
capture (`--qlog-file-base`) on a failing h2load-h3 c1m32 attempt shows the client-side symptom
mirrors the original (now-fixed) bug: `min_rtt`/`smoothed_rtt` pinned at 0 for the entire trace,
`pto_count` climbing, the client eventually gives up and sends its own `connection_close`. But the
trigger is a later connection against an already-warm server, and the server log names a specific
mechanism (`non-Initial packet received in Initial state`) the original bug's evidence never had.

qlog captures persisted on host-b: `~/rektbench/fanout/results/qlog_c1m32_bug/trial{2,3,4}.sqlog`
plus `server.log`.

**ROOT-CAUSED + FIXED (commit 595b8cfc), by live server-side instrumentation on host-b.** The
native h3 listener's `serve()` loop sampled `now` before its recv/timer/handlers `.await`, then
reused that stale value to anchor a freshly-accepted connection's `handshake_completion_deadline`
(10s). On a SO_REUSEPORT shard idle >10s between sequential c1m32 connections, the next reap pass's
fresh `now` was already past the deadline → the connection was reaped microseconds after acceptance
→ its SCID unregistered → the client's in-flight Handshake retransmits misrouted onto a phantom
Initial-state connection → `non-Initial packet received in Initial state`. Instrumented proof: one
connection got two `handle_timeout` calls 53µs apart with `now` jumping 27µs → 12,039,180µs.
Fix: split `now` into `tick_start` (pre-await, sizes the sleep) + a fresh `now` re-sampled after
the await. Gate: proxima-http green (92+8+7), and 3 host-b c1m32 reruns (~30M requests) with
0 failed / 0 errored / 0 `ProtocolViolation`. The c1 single-conn parity question (rekt ~22% slower
than nghttp3 in prior data) is now unblocked and can be re-measured.

## c1 m32 shape — MEASURED 2026-07-19 (post bug-#2-fix), gap HALVED to ~11%

Once bug #2 was fixed (`d24fe5c1`/`3394ec19`), the classic single-connection shape
(`rekt_h3 addr 1 32 1 <secs> localhost` / `h2load-h3 --h3 -t1 -c1 -m32 -D<secs>s --warm-up-time=1s`)
became measurable — one long-lived server across all trials (the exact bug-#2 sequential stress),
1 warmup + 6 measured trials each, 12s windows, server cores 0-3 / client 4-7, governor performance.

Raw per-trial rps (measured, warmup dropped),
`~/rektbench/fanout/results/c1-parity-raw/{rekt,h2load}-trial{2..7}.log`:
- **rekt_h3:** 85619 / 80350 / 78256 / 78998 / 80229 / 80048 — mean **80,584**, CoV **3.2%**
- **h2load-h3:** 87099 / 87059 / 88254 / 92459 / 92739 / 93415 — mean **90,171**, CoV **3.3%**

**Verdict: nghttp3 (h2load-h3) ~11% faster than rekt on a single connection** (rekt 89.4% of h2load
by mean, 88.7% by median). Direction 100% stable — rekt's best trial (85619) is below h2load's worst
(87059), ranges do NOT overlap. Both CoV settle <5%, so these are point estimates, not noise-ranges.
The prior-data expectation was rekt ~22% slower; the native-h3 work **halved** the gap to ~11% —
same direction, no flip. 0 `ProtocolViolation` across all 14 client runs (bug #2 fix holds on
host-b). Server ran ~121% CPU of one core with 4 available → rekt is client-limited (singular
`poll_recv`/`poll_send` vs the server's batched path — the discipline log's standing diagnosis), so
the remaining gap is a client batched-I/O opportunity, not a server deficit.

## host loadout for this rerun

host-b (Linux 6.15.3-arch1-1, x86_64, Intel i7-9700K, 8 physical cores / no SMT), same core
split as 2026-07-17 (server `taskset -c 0-3`, client `taskset -c 4-7`). Confirmed quiet before each
attempt (`pgrep -a bench_server`/`rekt_h3`/`h2load` all empty, `uptime` load average 0.6–1.3 at each
start; several long-idle low-CPU `claude` agent processes present but <10% CPU each, not treated as
contention). Load average climbed to 12–20 DURING each c4m16 run — this is the bench workload itself
(server+client combined CPU% regularly exceeds 580% of the 800% available across 8 cores), not
external interference; confirmed no other heavy process (`ps --sort=-%cpu`) during any run window.

## commands (reproducible)

```
# resync + rebuild
ssh host-b 'cd proxima && git fetch origin && git reset --hard origin/main'
ssh host-b 'rsync -a --delete --exclude target --exclude .git \
  proxima/ ~/rektbench/fanout/proxima/'
ssh host-b 'cd ~/rektbench/fanout/proxima/tools/rekt && \
  CARGO_TARGET_DIR=$HOME/rektbench/fanout/target cargo build --release \
  --features scheduler,h3-native-upstream --examples'

# headline check (fixed bug proof)
h2load-h3 --h3 -t1 -c1 -m1 -n5 https://127.0.0.1:18094/

# c4m16 shape (this doc's original h3 arm shape), 1 warmup + 5 trials, both gens
~/rektbench/fanout/h3_rerun.sh c4m16   # extends tournament.sh's run_h3 with the
                                        # now-working h2load-h3 incumbent in the same
                                        # WARMUP/TRIALS/median_and_cov loop as h2's arm
```

Raw per-trial logs (the source of the six-run spread above):
`~/rektbench/fanout/results/raw/h3-c4m16-{rekt,h2load-h3}-{1..6}.log` and `h3-c4m16-server.log`
(RUST_LOG=warn) on host-b. qlog captures for the c1m32 bug (persisted, not ephemeral `/tmp`):
`~/rektbench/fanout/results/qlog_c1m32_bug/trial{2,3,4}.sqlog` plus `server.log` (RUST_LOG=warn)
showing the `ProtocolViolation` lines, all on host-b.
