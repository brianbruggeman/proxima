# cache — fallthrough + write-back

A cache in front of an origin: the first request misses and populates the cache via write-back; every later request hits the cache and the origin is never called again.

## Builds on

[filter](../filter/README.md) — the miss decision (`Fallthrough` skipping a `no_data` upstream) is the same short-circuit-before-the-inner-pipe shape `Filter`'s `Decide` teaches, just applied to upstream selection instead of a single pipe. [transform](../transform/README.md) — `CountingOrigin` and `CachedOriginDispatch` are both plain `Pipe`s, `In -> Out`.

## What it demonstrates

There is no single `Cache` primitive in proxima. What `scenarios/cached/scenario.toml` calls "a cache in front of an origin" is three real primitives wired together, the same way `src/load.rs::build_composed` wires any multi-upstream pipe:

- two upstreams behind one `Selection`: a `kv:cache` lookup (`KvUpstream` over `KvCache`) and a `synth` origin
- `Fallthrough` selection with `miss_on = [no_data]` — try the cache first; a cache miss (`ProximaError::NoData`) falls through to the origin
- `WriteBack` wrapping the whole dispatch — after any response, capture the body and `put` it into the cache backend, so a miss this request becomes a hit next request

`CachedOriginDispatch` is the same shape as `DispatchPipe` in `src/load.rs` (private to that module, so this example reproduces its two-line body): a `Pipe` whose entire job is `Selection::dispatch` over a fixed `[cache, origin]` upstream list (each upstream held as a labeled handle the dispatch iterates over in order). `WriteBack::single` wraps that dispatch; nothing about the cache backend knows it's "in front of" anything — the miss/hit behavior comes entirely from `Fallthrough` trying upstreams in order, and the "populate on miss" behavior comes entirely from `WriteBack` capturing every 2xx response, cache-served or not, into the same backend it's guarding. That 2xx restriction is `WriteBack`'s default `WriteBackConditions` (`only_on_success = true`, `200..=299`), not a special case wired for this example — a 4xx/5xx origin response would flow through untouched, never written to the cache.

One gap found while building this: the bundled `scenarios/cached/scenario.toml` drives its workload with `method = "POST"`. `KvUpstream::call`'s `Post`/`Put` arm always stores-and-succeeds — it never returns `no_data` — so a POST-driven run never actually exercises the fallthrough-to-origin path; the cache upstream answers every request itself. The scenario's own `success_rate_ge`/`write_back.writes_total >= 1` expectations pass either way, so this doesn't fail CI, but it isn't proving what the doc comment above it claims. The scenario test that DOES exercise the documented miss-then-hit path (`cached_scenario_drives_cache_hits_after_first_miss` in `tests/units/scenario_smoke.rs`) uses `GET`. This example follows `GET`, the semantically correct verb for a cache lookup and the one the passing in-code test actually uses.

## Run

```
cargo run --example cache
```

## What you'll see

```
cache: fallthrough + write-back (cache in front of an origin)

-- request 1: cache empty, falls through to origin, write-back populates cache --
  status=200 cache-header=None body={"id":"chatcmpl-fake","object":"chat.completion","choices":[]}
  origin calls so far: 1

-- requests 2..6: cache hits, origin never called again --
  request 2: status=200 cache-header=Some("HIT")
  request 3: status=200 cache-header=Some("HIT")
  request 4: status=200 cache-header=Some("HIT")
  request 5: status=200 cache-header=Some("HIT")
  request 6: status=200 cache-header=Some("HIT")

origin called 1 time(s) across 6 requests; the remaining 5 were served straight from the cache
```

Request 1 has no `x-proxima-cache` header — that header is only set by `KvUpstream`'s hit path (`entry_to_response`), so its absence proves the response came from the origin, not the cache. `origin_calls`, an `AtomicUsize` bumped inside `CountingOrigin::call`, reads `1` after all six requests: real proof the origin was invoked exactly once, not just that the printed output looks right. Requests 2 through 6 all carry `x-proxima-cache: HIT` and the exact body the origin produced on the miss — `WriteBack`'s capture wrote that body into the cache after request 1, and every later `Fallthrough::dispatch` call resolves at the cache upstream (index 0) without ever reaching the origin.
