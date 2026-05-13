# bench-vs-hyper results — m1

proxima h1/h2 native server vs hyper on hyper's home turf (general h1/h2
building-block server). Three arms: proxima(prime), proxima(per-core tokio),
hyper(tokio multi_thread).

Tail percentiles are decomposed by phase: **warmup** (first 10% of iterations per criterion sample, caches cold), **steady** (middle 80%, latency ≤ 5× median), **spike** (middle 80% samples > 5× median — transient excursions), **spindown** (last 10%). A single combined histogram hides which phase the p999 came from; this decomposition surfaces whether tail-latency complaints are real steady-state behavior or measurement artifact. Configurable via `HDR_WARMUP_PCT` / `HDR_SPINDOWN_PCT` / `HDR_SPIKE_K` env vars.

---

## h1 warm GET

Criterion bench: `h1_vs_hyper`. Each arm measures one complete request/response
cycle over a tokio::io::duplex transport (no TCP syscalls). Mean is per-request
latency in nanoseconds.

| arm | M1 mean | M1 p50/p99/p999 | Linux mean | Linux p50/p99/p999 | unit |
|-----|---------|-----------------|------------|--------------------|------|
| hyper::server::conn::http1 (duplex transport) | pending | pending | pending | pending | ns |
| proxima::Connection (in-process) | pending | pending | pending | pending | ns |
| proxima::Connection (duplex transport, async server task) | pending | pending | pending | pending | ns |

---

## h2 warm GET

Criterion bench: `h2_vs_hyper`. Both arms use a warm h2 connection over loopback
TCP (handshake done once per bench group, not per iteration). Mean is per-request
latency.

| arm | M1 mean | M1 p50/p99/p999 | Linux mean | Linux p50/p99/p999 | unit |
|-----|---------|-----------------|------------|--------------------|------|
| proxima::serve_h2_connection (warm) | pending | pending | pending | pending | ns |
| hyper::server::conn::http2 (warm) | pending | pending | pending | pending | ns |

---

## h1 streaming

Criterion bench: `h1_streaming`, group `h1_streaming_vs_hyper`. Both arms serve
16 × 4 KiB chunks (64 KiB total) over tokio::io::duplex with chunked
transfer-encoding. Per-iter latency recorded in hdrhistogram; p50/p90/p99/p999
printed to stdout during the run.

| arm | M1 mean | M1 p50/p99/p999 | Linux mean | Linux p50/p99/p999 | unit |
|-----|---------|-----------------|------------|--------------------|------|
| hyper::server::conn::http1 streaming (duplex) | pending | pending | pending | pending | ns |
| proxima::Connection streaming (duplex) | pending | pending | pending | pending | ns |

### h1 streaming — phased tail

| arm | phase | p50 | p90 | p99 | p999 | max | count |
|-----|-------|-----|-----|-----|------|-----|-------|
| hyper::http1 streaming | warmup | pending | pending | pending | pending | pending | pending |
| hyper::http1 streaming | steady | pending | pending | pending | pending | pending | pending |
| hyper::http1 streaming | spike | pending | pending | pending | pending | pending | pending |
| hyper::http1 streaming | spindown | pending | pending | pending | pending | pending | pending |
| proxima::Connection streaming | warmup | pending | pending | pending | pending | pending | pending |
| proxima::Connection streaming | steady | pending | pending | pending | pending | pending | pending |
| proxima::Connection streaming | spike | pending | pending | pending | pending | pending | pending |
| proxima::Connection streaming | spindown | pending | pending | pending | pending | pending | pending |

---

## h2 streaming responses

Standalone binary bench: `h2_streaming_responses`. All arms use a warm h2 client
over loopback TCP, 3 s measurement window. 32 × 2 KiB chunks (64 KiB total per
request). Latency recorded in hdrhistogram; p50/p90/p99/p999/max printed to
stdout.

| arm | M1 mean | M1 p50/p99/p999 | Linux mean | Linux p50/p99/p999 | unit |
|-----|---------|-----------------|------------|--------------------|------|
| proxima_native (default tokio) | pending | pending | pending | pending | ns |
| hyper (default tokio) | pending | pending | pending | pending | ns |
| pingora (default tokio) | pending | pending | pending | pending | ns |

### h2 streaming responses — phased tail

| arm | phase | p50 | p90 | p99 | p999 | max | count |
|-----|-------|-----|-----|-----|------|-----|-------|
| proxima_native (default tokio) | warmup | pending | pending | pending | pending | pending | pending |
| proxima_native (default tokio) | steady | pending | pending | pending | pending | pending | pending |
| proxima_native (default tokio) | spike | pending | pending | pending | pending | pending | pending |
| proxima_native (default tokio) | spindown | pending | pending | pending | pending | pending | pending |
| hyper (default tokio) | warmup | pending | pending | pending | pending | pending | pending |
| hyper (default tokio) | steady | pending | pending | pending | pending | pending | pending |
| hyper (default tokio) | spike | pending | pending | pending | pending | pending | pending |
| hyper (default tokio) | spindown | pending | pending | pending | pending | pending | pending |
| pingora (default tokio) | warmup | pending | pending | pending | pending | pending | pending |
| pingora (default tokio) | steady | pending | pending | pending | pending | pending | pending |
| pingora (default tokio) | spike | pending | pending | pending | pending | pending | pending |
| pingora (default tokio) | spindown | pending | pending | pending | pending | pending | pending |

---

*Populate by running `scripts/bench-vs-hyper.sh` on each platform.*
