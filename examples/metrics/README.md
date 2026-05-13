# metrics

Counter, Gauge, Histogram — a metric is the `observe` form, specialized to aggregation.

## Builds on

[transform](../transform/README.md) — metrics are the observe form: watch, don't change, aggregate.

## What it demonstrates

`transform` named `observe` as `Pipe<In = T, Out = T>` — a call kept only for
its side effect, the value passed through unchanged (there, the observe counted
its own calls by hand). A metric pushes that one step further: the call doesn't even
return the value, it folds it into a running aggregate and the only way to
see it again is to read the instrument back.

`proxima_telemetry::metric` gives three instruments and a macro per
instrument:

| instrument | shape | macro |
|---|---|---|
| `Counter` | monotonic accumulator, `add(delta)` / `get()` | `counter!(INSTRUMENT, delta)` |
| `Gauge` | last-write-wins, `set_u64`/`set_f64`/`set_i64` / `get_u64`/`get_f64`/`get_i64` | `gauge!(INSTRUMENT, value)` |
| `Histogram<V>` | branchless base-2 bucketed distribution, `record(value)` / `count()` / `sum()` / `bucket_snapshot()` | `histogram!(INSTRUMENT, value)` |

The example declares one static of each (`Counter::new`, `Gauge::new`,
`Histogram::<f64>::new` — all `const fn`, so no lazy-init machinery needed),
drives five simulated requests through a `handle_request` hot path that
observes each instrument via its macro, then reads every instrument back:
`REQUESTS_TOTAL.get()`, `QUEUE_DEPTH.get_u64()`,
`REQUEST_LATENCY_MS.count()`/`.sum()`. Each read-back is asserted against the
value the aggregation rule predicts — not against any one call's return,
because none of these calls return a value that matters.

## Run

```
cargo run --example metrics
```

No feature flag needed: `Counter`/`Gauge` are always available, and
`histogram` is on by default in this workspace's `default` feature set. (The
`instrument-metrics` feature is a different seam — it wires a span's duration
into a histogram automatically; it isn't required for recording metrics by
hand, which is all this example does.)

## What you'll see

```
--- hot path: five requests, three instruments watching ---
  request: payload=2 -> latency=1ms, queue_depth=1
  request: payload=4 -> latency=2ms, queue_depth=2
  request: payload=6 -> latency=3ms, queue_depth=3
  request: payload=8 -> latency=4ms, queue_depth=2
  request: payload=10 -> latency=5ms, queue_depth=0
--- read back: the aggregate, not any one call's return value ---
counter   requests_total:     5
gauge     queue_depth:        0
histogram request_latency_ms: count=5 sum=15ms
all three instruments proved their own aggregation semantics
```

`requests_total` is `5` — one `add(1)` per call, not the payload size —
proof a counter aggregates the *number of observations*, not their values.
`queue_depth` ends at `0`, the last value set, even though it was `1, 2, 3,
2, 0` along the way — proof a gauge keeps only the most recent observation
and throws the rest away. `request_latency_ms` has `count=5` (every call
recorded) and `sum=15` (`1+2+3+4+5`) — proof a histogram, unlike a gauge,
keeps every observation in its running aggregate.
