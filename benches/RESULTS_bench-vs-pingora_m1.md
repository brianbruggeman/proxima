# bench-vs-pingora — m1

proxima h2 native vs pingora on pingora's home turf: h2 edge reverse-proxy
with multi-connection scaling and tail latency.

Tail percentiles are decomposed by phase: **warmup** (first 10% of iterations per criterion sample, caches cold), **steady** (middle 80%, latency ≤ 5× median), **spike** (middle 80% samples > 5× median — transient excursions), **spindown** (last 10%). A single combined histogram hides which phase the p999 came from; this decomposition surfaces whether tail-latency complaints are real steady-state behavior or measurement artifact. Configurable via `HDR_WARMUP_PCT` / `HDR_SPINDOWN_PCT` / `HDR_SPIKE_K` env vars.

> M1 numbers below are single-run; 5-trial-median is the doc-quality bar —
> re-run with 5 trials and --save-baseline before publishing.

Hardware: Apple M1 (or equivalent Apple Silicon), macOS.
Server pinned to its own runtime; bench driver on a separate default-tokio runtime.
Loopback TCP, plain HTTP/2 (no TLS, h2c prior-knowledge).

**Reference (Linux i7-9700K 5-trial medians from RESULTS_linux.md):**

| connections | proxima_native per_core | proxima_native default_tokio | hyper | pingora |
|-------------|------------------------|------------------------------|-------|---------|
| 1  | 37,346 RPS | 33,202 RPS | 23,467 RPS | 23,353 RPS |
| 4  | 96,501 RPS | 91,463 RPS | 66,428 RPS | 68,097 RPS |
| 16 | 246,170 RPS | 142,848 RPS | 174,420 RPS | 164,121 RPS |
| 64 | 286,447 RPS | 189,219 RPS | 188,670 RPS | 175,207 RPS |

---

## h1 warm GET (h1_vs_pingora)

| arm | mean latency | CoV | RPS (est) | p50 | p99 | bench file |
|-----|-------------|-----|-----------|-----|-----|------------|
| proxima h1 (loopback) | pending | pending | pending | pending | pending | h1_vs_pingora |
| hyper h1 (loopback) | pending | pending | pending | pending | pending | h1_vs_pingora |
| pingora h1 (loopback) | pending | pending | pending | pending | pending | h1_vs_pingora |

---

## h2 warm GET (h2_vs_pingora)

Single warm h2 connection, per-request cost.

| arm | m1 mean | linux mean | notes |
|-----|---------|------------|-------|
| proxima::serve_h2_connection | pending | 28.85 µs | proxima native |
| hyper::http2::Builder | pending | 38.01 µs | baseline |
| pingora::http::v2::HttpSession | pending | 39.89 µs | pingora baseline |

---

## multi-conn sweep — p50 latency (h2_tail_multi_conn)

Each row is the criterion-reported p50 latency per arm per connection count.
p99 and p999 are recorded separately (see next section).

| arm | conn=1 p50 | conn=4 p50 | conn=16 p50 | conn=64 p50 |
|-----|-----------|-----------|------------|------------|
| proxima_native_default_tokio | pending | pending | pending | pending |
| proxima_native_per_core      | pending | pending | pending | pending |
| hyper_default_tokio          | pending | pending | pending | pending |
| pingora_default_tokio        | pending | pending | pending | pending |

---

## tail latency at conn=64 — p99 / p999

| arm | p99 @ conn=64 | p999 @ conn=64 |
|-----|--------------|----------------|
| proxima_native_default_tokio | pending | pending |
| proxima_native_per_core      | pending | pending |
| hyper_default_tokio          | pending | pending |
| pingora_default_tokio        | pending | pending |

**Linux reference (RESULTS_linux.md single-run snapshot):**

| arm | p99 @ conn=64 |
|-----|--------------|
| proxima_native_default_tokio | 1.20 ms |
| proxima_native_per_core      | 840 µs  |
| hyper                        | 1.26 ms |
| pingora                      | 1.46 ms |

---

## h2_tail_multi_conn phased tail (conn=1 and conn=64)

Per-arm phase decomposition at the connection counts most likely to show warmup
artifact (conn=1) and steady-state tail (conn=64).

| arm | phase | p50 | p90 | p99 | p999 | max | count |
|-----|-------|-----|-----|-----|------|-----|-------|
| proxima_native_default_tokio/conn=1 | warmup | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=1 | steady | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=1 | spike | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=1 | spindown | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=64 | warmup | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=64 | steady | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=64 | spike | pending | pending | pending | pending | pending | pending |
| proxima_native_default_tokio/conn=64 | spindown | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=1 | warmup | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=1 | steady | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=1 | spike | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=1 | spindown | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=64 | warmup | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=64 | steady | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=64 | spike | pending | pending | pending | pending | pending | pending |
| proxima_native_per_core/conn=64 | spindown | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=1 | warmup | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=1 | steady | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=1 | spike | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=1 | spindown | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=64 | warmup | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=64 | steady | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=64 | spike | pending | pending | pending | pending | pending | pending |
| hyper_default_tokio/conn=64 | spindown | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=1 | warmup | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=1 | steady | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=1 | spike | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=1 | spindown | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=64 | warmup | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=64 | steady | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=64 | spike | pending | pending | pending | pending | pending | pending |
| pingora_default_tokio/conn=64 | spindown | pending | pending | pending | pending | pending | pending |

---

## notes

- Darwin `core_affinity` is best-effort; the kernel can ignore CPU pinning hints, so
  the per-core advantage seen on Linux (+64% RPS at conn=64) may not appear on M1.
- Run-to-run variance at conn=16+ is high on single-run; the Linux 5-trial median is
  the current doc-quality reference. Re-run this bench with 5 trials before publishing M1 claims.
- bench script: `scripts/bench-vs-pingora.sh`
- discipline log: `docs/bench-suite/discipline-bench-vs-pingora.md`
