# Build a caching reverse proxy

**Prerequisites:** [Foundations](./00-foundations.md); the [Gateway's forward proxy](./build-an-api-gateway.md) (§1) for the "an origin is just an upstream" idea.
**You will:** put a cache in front of an origin so a repeated request is served from the cache and the origin is hit once. There is no single `Cache` primitive — you compose three.
**New concepts (in order):** cache backend (`KvCache`/`KvUpstream`) · upstream selection (`Selection` + `Fallthrough`) · write-back (`WriteBack`).
**Answer key:** [`examples/cache/main.rs`](../../examples/cache/main.rs) — `cargo run --example cache`.

The example says it plainly: *"There is no single `Cache` primitive in proxima. A cache in front of an origin is three real primitives wired together."*

## 1. Two upstreams: the cache and the origin

An *upstream* is anything that answers a request — a `Pipe` behind an `UpstreamRef`. Here, a KV cache lookup and an origin (`cache/main.rs:44-60`):

```rust
let cache_backend = KvCache::new("cache", None, KvCaps::entries(1024))
    .expect("kv cache backend");
let cache_upstream = into_handle(KvUpstream::new(cache_backend.clone()));

// in a real reverse proxy this is the Gateway's ForwardPipe to the upstream;
// the example uses a SynthUpstream so it has a deterministic body to assert on.
let origin_upstream = into_handle(/* ForwardPipe { client } | SynthUpstream::new(...) */);

let upstreams = Arc::new(vec![
    UpstreamRef::new(cache_upstream, "cache", 1),
    UpstreamRef::new(origin_upstream, "origin", 1),
]);
```

`KvCache::new` takes a label, a default TTL (`None` — no automatic time-based expiry; only the capacity cap below evicts), and a capacity cap (`KvCaps::entries(1024)` — hold at most 1024 entries, evicting the least-recently-used one once full). It returns a `Result` wrapping an `Arc<KvCache>` — `.expect(...)` above unwraps it (this example's `main` isn't fallible, so a construction failure panics rather than propagating), and `Arc` is a reference-counted pointer, so every `.clone()` below shares the same underlying store. That `Arc<KvCache>` also implements the `KvHandle` trait; it is that trait, not the concrete `KvCache` type, that §3's write-back step stores into.

`KvUpstream` turns that `KvCache` into an upstream that answers from stored entries — and returns `ProximaError::NoData` on a miss. That miss is the signal the next piece keys on.

The trailing `1` in `UpstreamRef::new(handle, "cache", 1)` is a selection *weight*. `Fallthrough` (§2) ignores it and always tries upstreams in list order; other `Selection` strategies (round-robin, least-connections) use it to split traffic proportionally. `UpstreamRef::new` always takes one, even when — as here — nothing reads it.

## 2. Fallthrough: try the cache, fall through to the origin on a miss

`Selection` picks which upstream answers; `Fallthrough` tries them in order and moves on when one *misses*. `miss_on_no_data()` defines a miss as `ProximaError::NoData` — exactly what the empty cache returns. `CachedOriginDispatch` below isn't a proxima primitive — it's a small `Pipe` type the example authors itself, the same way Foundations' `HelloPipe` is example code, just to hold the two pieces `Selection::dispatch` needs (`cache/main.rs:61-64`):

```rust
let dispatch = CachedOriginDispatch {
    upstreams,
    selection: Arc::new(Fallthrough::miss_on_no_data()),
};
```

The dispatch is a thin `Pipe` whose whole job is `Selection::dispatch` over the fixed upstream list (`cache/main.rs:165-181`):

```rust
impl SendPipe for CachedOriginDispatch {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, request: Request<Bytes>)
        -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let upstreams = self.upstreams.clone();
        let selection = self.selection.clone();
        async move {
            let outcome = Selection::dispatch(selection.as_ref(), &upstreams, request).await?;
            Ok(outcome.response)
        }
    }
}
```

`Selection::dispatch` returns a `DispatchOutcome` whose `.response` field is the chosen upstream's reply; `call` above just unwraps it and hands it back.

That is the whole type — there is no second `impl Pipe for CachedOriginDispatch {}` to write, and no `name()` override to opt into. `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (Foundations §13), so once `CachedOriginDispatch` satisfies that `SendPipe` signature it is already mountable — "there is no second trait to opt into and nothing more to write."

A hit answers from the cache; a miss falls through to the origin. There is no "cache combinator" — **this wiring is the composition.**

## 3. Write-back: a miss now is a hit next time

`WriteBack` wraps the whole dispatch. After *any* response it captures the body and `put`s it into the cache backend — so the origin's answer to a miss populates the cache for next time (`cache/main.rs:66-67`):

```rust
let write_back_target: Arc<dyn KvHandle> = cache_backend.clone();
let cached_origin = WriteBack::single(into_handle(dispatch), write_back_target);
```

`cache_backend` is the same `Arc<KvCache>` from §1; retyping the clone as `Arc<dyn KvHandle>` just switches which face of it `WriteBack` sees — the `KvHandle` trait, not the concrete `KvCache` type — so `WriteBack` can store into any cache backend that implements it, not only this one.

`WriteBack` is an observe-and-store wrapper around the dispatch — the same "return the response, act on the side" shape as the **observe** role from Foundations, specialized to write into a cache.

## 4. Watch it: origin hit once, five cache hits

The example calls the composed pipe six times (`cache/main.rs:77-139`): request 1 misses → falls through to the origin → write-back stores the body; requests 2–6 hit the cache (`x-proxima-cache: HIT`) and the origin is never called again. It asserts the origin was called exactly once across all six.

```
cargo run --example cache
```

## What you built

A cache in front of an origin, from three composed primitives — no cache combinator:

- **cache upstream** — `KvUpstream` over a `KvCache`, returning `NoData` on a miss.
- **fallthrough selection** — `Fallthrough::miss_on_no_data()` tries cache, then origin.
- **write-back** — `WriteBack` stores each origin answer so the next request hits.

To make it a true reverse proxy, swap the origin for the [Gateway's `ForwardPipe`](./build-an-api-gateway.md) and serve via `App` as in Foundations.

**Going further** — wrap the origin in a `CircuitBreaker` ([`examples/circuit_breaker`](../../examples/circuit_breaker)) so repeated origin failures trip open and short-circuit: another `Pipe` around the origin, exactly as the cache wraps the dispatch.
