---
name: bench-metrics
description: The full set of signals a performance benchmark MUST capture — throughput (rps), latency (percentiles + single-request with a real client), CPU%, memory (RSS), error rate, and variance (CoV) — and how to read them together. Use when designing or reviewing a benchmark, when a bench reports only throughput/rps, when deciding what to measure, when a "fast" number hides a problem, or when the user says "what should we capture", "bench metrics", "are we measuring latency", "is our bench complete", "what's missing from this bench". Codifies the hard lesson that throughput alone hides latency, and that CPU% is what distinguishes compute-bound from latency-bound.
---

# bench-metrics

A benchmark that reports one number lies by omission. The canonical
failure: a server shows good **throughput** while every individual
request is **slow** — and you never see it, because the load client
pipelines deep enough to hide the per-request latency. This skill is
the checklist of signals every perf bench must capture so that class
of blindness can't happen.

> Motivating case (2026-06-29, proxima h3 vs quiche): the bench
> reported only rps. proxima showed 42k req/s — "fine, 1.13x behind
> quiche". A single GET with a real client (curl) took **57 ms TTFB vs
> quiche's 4.6 ms** — 12x worse, totally hidden by the deep-pipelining
> load client. The root cause (a 25 ms ACK delay) then turned out to
> *also* cap throughput: fixing it took rps 42k → 82k (1.67x quiche).
> One missing axis (latency) hid both a latency bug and a throughput
> ceiling.

## The six axes — capture ALL of them, every run

| axis | what | why it's load-bearing |
|------|------|------------------------|
| **throughput** | requests/sec (or bytes/sec, ops/sec) at a given offered load | the headline, but ALONE it is a trap (below) |
| **latency** | per-request, as **percentiles** (p50/p90/p99/p99.9) AND a **single-request number from a real client** | the thing throughput hides; tails reveal stalls/retries |
| **CPU%** | server CPU utilization (per pinned core) during the run | distinguishes **compute-bound** (pegged ~100%) from **latency-bound** (idle, waiting) — the single most diagnostic axis |
| **memory** | RSS (peak + steady-state), and allocations/op if relevant | resource ceiling, leaks, the 500MB/10k-conn-class budget |
| **errors** | error count / rate (0 is the gate) | a fast number with errors is not a result |
| **variance** | CoV across N runs (≥3, ideally 5) + the host loadout | a single run is a sample, not a result; high CoV = unstable, not fast |

A bench missing any of these is incomplete. A row that reports
throughput with no latency, or latency with no CPU%, cannot be
interpreted.

## Why throughput ALONE is a trap

Throughput and latency are not the same measurement and do not imply
each other:

- A **closed-loop** load client (send request, await response, repeat)
  with deep pipelining maximizes throughput by keeping many requests in
  flight — which **amortizes and hides** per-request latency. 42k rps
  at 57 ms/request just means ~2400 requests are in flight. The rps
  looks great; every user waits 57 ms.
- `throughput ≈ concurrency / latency` for closed-loop load. So you can
  raise the throughput number by raising concurrency even while latency
  is terrible. The bench rewards the wrong thing.
- **Always also measure single-request latency with a real,
  non-pipelining client** (curl, a one-shot client) — that's the number
  a latency-sensitive user actually feels, and it's the one the load
  client erases.

## CPU% is the most diagnostic axis — capture it always

CPU% tells you *which kind* of problem you have, which tells you *what
to fix*:

- **~100% CPU, throughput plateaued** → **compute-bound**. The work per
  op is the wall (memcpy, alloc, crypto, parse). Fix: cut per-op cost.
- **< ~90% CPU, throughput low** → **latency-bound / idle-waiting**. The
  server is *waiting*, not working (a timer, an ACK delay, a lock, a
  blocking hop, an under-fed loop). Fix: find what it waits on. A server
  producing low rps at 60% CPU is leaving the core idle — that's a
  latency bug, not a throughput bug.
- CPU% that **rises with concurrency** (e.g. 65% → 95%) while
  throughput ramps is the signature of a latency-bound server that only
  fills the pipe at high concurrency — it needs N connections to do what
  a low-latency server does with 1.

Without CPU%, you cannot tell "slow because the work is expensive" from
"slow because it's sitting idle" — and those have opposite fixes.

## Latency: percentiles, not the mean

- Report **p50 / p90 / p99 / p99.9** (and max). The **mean lies** — a
  bimodal distribution (most requests fast, a tail hitting a retry/PTO
  timer) has a deceptive mean and a screaming p99.
- Latency **tails expose timer-driven stalls**: a p99 that's a clean
  multiple of a protocol timer (25 ms ack-delay, a 1 s PTO, exponential
  PTO backoff) is a retransmit/timeout storm, not load.
- Capture **single-request latency separately** (real client, cold and
  warm) — the load-client percentiles still ride the pipeline.

## Variance and the host

- **Never trust one run.** Run ≥3 (5 preferred); report the **range or
  CoV**, not a point estimate. CoV > ~5% means the number is noisy —
  re-run on a quiet box or report the range honestly.
- **Pin and isolate**: server on its own core(s), load client on
  disjoint cores, so core-sharing doesn't contaminate the comparison.
  Record the host loadout (load average, other tenants) with the row.
- A high-CoV "win" is not a win; it's an unstable measurement.

## The capture checklist (per bench point)

1. Offered load (connections × streams, request shape/size).
2. **throughput** (median over N runs + CoV).
3. **latency** percentiles (p50/p99/p99.9) from the load client, AND a
   single-request TTFB from a real one-shot client.
4. **CPU%** of the server (per pinned core).
5. **memory** RSS (peak + steady).
6. **errors** (must be 0 to count; otherwise the row is rejected).
7. **variance**: ≥3 runs, CoV, host loadout.
8. A **frequency-weighted read**: which axis is the wall at the offered
   load, and is the server compute-bound or latency-bound there.

## Interpreting the combination (the honest read)

State, for each bench point, the *bound type* and the *bill-mover*:

- compute-bound + good latency → optimize per-op cost (the headline is
  throughput).
- latency-bound + idle CPU → find the wait (timer, lock, blocking hop);
  throughput will follow when latency drops (the motivating case: ack
  delay capped *both*).
- good throughput + bad single-request latency → the load client is
  hiding a latency bug; do not ship the throughput number as the verdict.
- any axis with errors > 0 or CoV > 5% → not a result yet.

## What NOT to do

- Don't report throughput as "the number" without latency + CPU% beside
  it — that's the exact omission that hides latency bugs.
- Don't report a mean latency — report percentiles.
- Don't measure latency only through the deep-pipelining load client —
  add a real single-request client.
- Don't trust one run, or a run on a loaded box, or a high-CoV number.
- Don't conclude "compute-bound, needs optimization" without checking
  CPU% — a 60%-CPU server is idle-waiting, and the fix is latency, not
  throughput.
