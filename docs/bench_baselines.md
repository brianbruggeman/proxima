# Bench baselines

Substrate-wide benchmark numbers, two platforms. Each bench was run
once via `cargo bench --bench <name> -- --quick`; `--quick`
samples to a tight CI of variance and reports mean ± p95 within a
few seconds per group. Numbers are middle-of-three (criterion's
"point estimate"). For the CI gate (a follow-up), re-run without
`--quick` to get stable p99s.

**Platforms**:

- **macOS**: M-series Apple silicon, Darwin 24.x.
- **Linux (host-b)**: Intel i7-9700K @ 3.60 GHz (8 cores), kernel 6.15.

## H/1 hot path

| Bench | macOS | Linux |
|---|---|---|
| `h1_parse_head` small GET, 5 headers | 145 ns | 251 ns |
| `Connection` round-trip, no body (GET → 200) | 222 ns | 298 ns |
| `Connection` round-trip, POST + 5-byte body | 211 ns | 293 ns |

**Streaming**:

| Bench | Buffered | Streaming |
|---|---|---|
| `h1_streaming` cl_256 (macOS) | 209 ns | 259 ns |
| `h1_streaming` cl_256 (Linux) | 295 ns | 327 ns |
| `h1_streaming` cl_64KiB (macOS) | 1.26 µs | 4.7 µs |
| `h1_streaming` cl_64KiB (Linux) | 1.4 µs | 3.0 µs |
| `h1_streaming` chunked 16×4KiB (macOS) | 1.32 µs | 9.6 µs |
| `h1_streaming` chunked 16×4KiB (Linux) | 1.79 µs | 4.0 µs |

Streaming overhead is the per-chunk `Bytes::copy_from_slice`. The
buffered path keeps body as one `&[u8]` slice into the read buffer.
Linux memcpy is faster than macOS at these sizes, which is why the
gap is wider on macOS.

**vs hyper** (same duplex transport, identical async server shape):

| | hyper | proxima (duplex) | proxima in-process | proxima win |
|---|---|---|---|---|
| macOS | 15.4 µs | 14.6 µs | 574 ns | 5% faster |
| Linux | 2.08 µs | 1.49 µs | 436 ns | 28% faster |

In-process is the Connection state machine cost without an async
transport — what proxima can do before the I/O loop enters. The
duplex number is apples-to-apples.

**vs hyper AND pingora** (same loopback TCP transport, identical
small GET → 200 cycle, accept-once-per-bench, fresh client connect
per iter — pingora's `Stream` doesn't accept tokio duplex so this
is the fair head-to-head):

| | proxima | hyper | pingora | proxima vs hyper | proxima vs pingora |
|---|---|---|---|---|---|
| macOS | 67.4 µs | 70.0 µs | 74.9 µs | 4% faster | 11% faster |
| Linux | 62.0 µs | 68.2 µs | 76.4 µs | 10% faster | 23% faster |

Most of the absolute 60-75µs is socket bind + accept + connect +
EOF detect (kernel syscalls), not the connection-layer machinery.
The relative gap is the real signal — proxima's hot path is leanest
across both platforms. Pingora is the slowest variant because
`HttpSession` carries production session machinery (`UniqueID`,
`GetTimingDigest`, `GetProxyDigest`, `GetSocketDigest`, `Peek`,
`Ssl`, `Shutdown`) the bench doesn't use but every iteration
pays for. Proxima doesn't surface most of those yet — gaps
intentionally tracked. Trade is "leanest hot path today" vs
"more session-state APIs later"; both are realistic deltas as the
prod feature set matures.

## Substrate dispatch

`substrate_dispatch` — `Pipe::call` through composed middleware
chains. No I/O, no kernel.

| Chain depth | macOS | Linux |
|---|---|---|
| 1 pipe | 570 ns | 322 ns |
| 2 pipes | 660 ns | 405 ns |
| 4 pipes | 830 ns | 535 ns |
| 8 pipes | — | 647 ns |
| 16 pipes | — | 1.10 µs |

Linux is faster across the board — multi-thread tokio + uncontended
worker scheduling tuned for hot-path dispatch.

## Lock-free read primitives

`per_core_vs_arcswap` — direct comparison of thread-local read vs
`ArcSwap` read under writer contention. This is the substrate's
core architectural claim: hot-path reads stay cache-local.

| | macOS | Linux |
|---|---|---|
| thread-local read | 10 ns | 2.1 ns |
| ArcSwap read (uncontended) | 53 ns | 12 ns |
| ArcSwap read (under writer contention) | 47 ns | 71 ns |
| ArcSwap read (sustained writes) | — | 87 ns |
| ArcSwap read (many readers) | — | 103 ns |

Thread-local reads are 5–10× faster than ArcSwap on both platforms.
Under writer contention ArcSwap stays bounded (no lock; readers
still complete in ~50–100 ns).

## Swap latency

`swap_under_load` — `SwappablePipe` swap + read costs.

| | macOS | Linux |
|---|---|---|
| `current()` load (uncontended) | 16.6 ns | 29.2 ns |
| `swap()` (uncontended) | 101 ns | 91.7 ns |
| dispatch through Swappable, writer storm | 822 ns | 405 ns |

The dispatch number includes the full Pipe::call cycle plus the
ArcSwap load and a writer-storm storming the same swappable on
another thread. Sub-µs round-trip on Linux is fine for hot-swap
under traffic.

## Capture / drain

`capture_drain` — per-call recording sidecar cost.

| Workload | macOS | Linux |
|---|---|---|
| drain only (no attach) | 9.4 ns | 12.1 ns |
| attach 1 field + drain | 124 ns | 136 ns |
| attach 8 fields + drain | 1.01 µs | 769 ns |

Empty drain is essentially free. Single-attach (the common case,
typically just `trace_id`) is sub-200 ns. Pathological 8-field
attach scales linearly.

## Causal record + explain

`causal_record` — graph index for capture chains.

| | macOS | Linux |
|---|---|---|
| record single edge | **62 ns** | **85 ns** |
| explain chain of depth 8 | 3.26 µs | 1.27 µs |

**Stage 3b switched from `ArcSwap<Vec<CausalEdge>>` (CoW, O(N) per
append) to `Mutex<Vec<CausalEdge>>` (O(1) append) after the pattern-
matched `causal_record_primitives` bench showed Mutex<Vec> beats
both ArcSwap-CoW and DashMap<u64, Edge> at every concurrency level
CausalIndex actually hits.** Prior numbers were 817 µs macOS /
543 µs Linux because the index accumulated during the bench and
each append cloned the full Vec.

`explain` is bounded by chain depth, not index size, so it stays
stable across the conversion.

`causal_record_primitives.rs` is the comparison bench (3 primitives
× 3 concurrency regimes); `causal_record.rs` is the production
primitive's microbench.

## Tee / fan-out

`tee_backpressure` — body fan-out for the Selection fall-through
path. Two scenarios: drain primary only (selection success), and
drain primary + replay (selection fall-through).

| | macOS | Linux |
|---|---|---|
| wrap + drain primary (256 B) | 343 ns | 272 ns |
| wrap + drain primary (4 KiB) | 334 ns | 183 ns |
| wrap + drain primary (64 KiB) | 330 ns | 186 ns |
| wrap + replay (256 B) | 546 ns | 298 ns |
| wrap + replay (4 KiB) | 531 ns | 298 ns |

Sizes 4 KiB and 64 KiB land at similar latencies because
`Tee::wrap` uses a single `Bytes` buffer for content-known bodies —
the per-call cost is the wrap/drain ceremony, not the bytes.

## Telemetry / histogram

`histogram_record` — per-record cost for the telemetry histogram.

| | macOS | Linux |
|---|---|---|
| single-thread record | 472 ns | 242 ns |
| 8-worker shared histogram | 18 µs (total) | 10 µs |

Single-thread sub-500ns is the inner-loop cost. Multi-worker
includes the contention overhead — fine for the typical histogram
update rate (a few per request, not per byte).

## Network throughput

`network_throughput` — TCP request/response round-trip via a
proxima listener bound on localhost.

| | macOS | Linux |
|---|---|---|
| sustained small-request loop | 49 µs | 316 µs |
| large-body loop | 69 µs | 314 µs |

Linux throughput is single-core saturated. Multi-core scaling is in
`listener_throughput.rs` (the load-test example), not in this
microbench.

## simd_json vs serde_json

Decode cost for a typical config-sized JSON payload.

| | macOS | Linux |
|---|---|---|
| serde_json::from_slice | 7.88 µs | 8.30 µs |
| simd_json::from_slice | 6.94 µs | 6.47 µs |

simd_json ~12% faster on macOS, ~22% faster on Linux. Used in the
hot config-reload path.

## Bench inventory

| File | Coverage |
|---|---|
| `h1_dispatch.rs` | parse_head, Connection round-trip (no body, with body) |
| `h1_streaming.rs` | buffered vs streaming, three body shapes |
| `h1_vs_hyper.rs` | head-to-head vs hyper on shared duplex transport |
| `h1_vs_pingora.rs` | head-to-head vs hyper AND pingora on shared loopback TCP transport |
| `histogram_record.rs` | telemetry histogram inner-loop cost |
| `network_throughput.rs` | end-to-end TCP loop |
| `per_core_vs_arcswap.rs` | thread-local vs ArcSwap reads (the architectural claim) |
| `perf_audit.rs` | audit-style multi-primitive pass |
| `request_path.rs` | substrate-level Request → Response through synth, cache-hit |
| `simd_json_decode.rs` | simd_json vs serde_json |
| `substrate_dispatch.rs` | `Pipe::call` through 1–16-deep chains |
| `swap_under_load.rs` | SwappablePipe swap + read + dispatch-under-storm |
| `capture_drain.rs` | CaptureContext attach + drain |
| `causal_record.rs` | CausalIndex record + explain |
| `causal_record_primitives.rs` | head-to-head: ArcSwap<Vec> vs Mutex<Vec> vs DashMap<u64,Edge> under 0/1/4/16 concurrent recorders |
| `recording_sink_primitives.rs` | head-to-head: Mutex<File> vs O_APPEND vs SegQueue+writer-task for recording sink fan-out |
| `tee_sink_primitives.rs` | head-to-head: tokio::mpsc vs crossbeam ArrayQueue vs SegQueue for tee per-sink fan-out |
| `tee_backpressure.rs` | Tee::wrap drain + replay for selection fall-through |

## Reproduce

```sh
# single bench
cargo bench --bench h1_streaming -- --quick

# all benches
for b in h1_dispatch h1_streaming h1_vs_hyper histogram_record \
         network_throughput per_core_vs_arcswap perf_audit \
         request_path simd_json_decode substrate_dispatch \
         swap_under_load capture_drain causal_record tee_backpressure; do
  echo "$b"
  cargo bench --bench $b -- --quick 2>&1 | grep "time:"
done
```
