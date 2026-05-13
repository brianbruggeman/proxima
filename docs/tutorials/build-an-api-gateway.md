# Build an API gateway

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md).
**You will:** put a policy chain in front of an upstream — forward (proxy) → route by path → rate-limit per upstream → require auth — each policy an ordinary `Pipe`, composed outside-in, so a rejected request never reaches the next stage.
**New concepts (in order):** forward/proxy (`Client`) · routing (`RoutingPipe`) · rate-limit gate (`RateLimit`) · auth (`Auth`, the short-circuit filter).
**Answer key:** [`examples/gateway/main.rs`](../../examples/gateway/main.rs) — `cargo run --example gateway`.

The example states the whole idea in its own header: *"A gateway is the proxy example's forward pipe with a policy chain composed in front of it. No new primitive family — `Auth` and `RoutingPipe` are ordinary `Pipe`s, `RateLimit` is the token-bucket gate. A request rejected by one policy never reaches the next."* We build it one policy at a time.

## 1. Forward — the proxy is one call

The base unit of a gateway is forwarding a request to an upstream. proxima's `Client` is *itself* a `SendPipe`, so forwarding **is** calling it (`gateway/main.rs:318-335`):

```rust
#[derive(Clone)]
struct ForwardPipe { client: Client }

impl SendPipe for ForwardPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, request: Request<Bytes>)
        -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let client = self.client.clone();
        async move { SendPipe::call(&client, request).await }
    }
}
```

That is the whole type — there is no second `impl Pipe for ForwardPipe {}` to write. `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (Foundations §13), and `ForwardPipe` already satisfies that signature, so it is already mountable as-is.

`Client::http(format!("http://{origin_api_bind}"))?` builds the client (`gateway/main.rs:71`). Inside `call`, the request goes out via `SendPipe::call(&client, request)` rather than the more familiar `client.call(request)` — `Client` also has its own inherent `call(method, path)` builder method, and Rust always picks an inherent method over a trait method of the same name, so writing `SendPipe::call(&client, request)` (the trait's `call`, with `client` passed explicitly as the first argument) is how you force the right one to run. That is the entire proxy — the standalone [`examples/proxy`](../../examples/proxy) is exactly this pipe. Everything below just wraps it.

## 2. Route — pick an upstream by path

Two upstreams (an `api` origin and a `web` origin), each behind its own `ForwardPipe`. `RoutingPipe` picks one by path prefix (`gateway/main.rs:97-99`). `api_forward` and `web_forward` are each a `PipeHandle` (built in section 3 below) that wraps a rate-limited `ForwardPipe` for that upstream — for now, just read them as "the api handle" and "the web handle":

```rust
let routed = RoutingPipe::new("gateway-router")
    .route("/api/{*rest}", api_forward)   // "/api/..." -> api upstream (built in §3 below)
    .fallback(web_forward);               // everything else -> web upstream (built in §3 below)
```

`RoutingPipe` is a `Pipe`: its `call` matches the request path against each route's pattern and delegates to the chosen `PipeHandle`. Routing is composition, not a network detail.

## 3. Rate-limit — a gate in front of each upstream

Wrap each `ForwardPipe` in `RateLimit`, a token-bucket **gate**. Once a per-upstream budget is spent, the gate answers `429` (with a `retry-after`) and the forward never runs (`gateway/main.rs:72-81`):

```rust
let api_forward: PipeHandle = into_handle(RateLimit::with_clock(
    ForwardPipe { client: api_client },
    TokenBucketConfig { capacity: 2, refill_per_sec: 0 },
    KeyExtractor::ConstantKey("api-upstream".into()),
    RateLimitCaps::default(),
    clock,
));
```

`TokenBucketConfig` is the budget: `capacity` tokens, refilled at `refill_per_sec` (here `0`, so the bucket never refills and the boundary is reached purely by call count). `KeyExtractor` picks the rate-limit bucket key per request; `ConstantKey("api-upstream".into())` means every request to this upstream draws from the same one bucket, so the whole `api` origin shares a single budget rather than one bucket per caller. `RateLimitCaps::default()` is a light cap on the bucket map itself (how many distinct keys it will track, how long an idle key is kept, how often it is swept) — irrelevant here since `ConstantKey` only ever produces one key, but required by the gate's signature.

Note the `Clock` seam: rate limiting is measured against an **injected** clock, never a real `sleep`, so it is deterministic and testable. The example injects a `FakeClock` whose time only advances when told (`gateway/main.rs:341-356`); [`examples/rate_limit`](../../examples/rate_limit) drives the same seam with an advancing clock. This is the `gate` idiom — SHED (reject now) / WAIT / BALANCE readiness composed as a pipe, instead of a bespoke `poll_ready`. See [`examples/gate`](../../examples/gate).

## 4. Auth — short-circuit before anything runs

Wrap the whole router in `Auth`, a **filter** that rejects a missing/wrong bearer token with `401` before routing or rate-limiting is ever reached (`gateway/main.rs:103-110`):

```rust
let gateway_pipe = Auth {
    inner: into_handle(routed),
    header: "authorization".to_string(),
    allow: BTreeSet::from([VALID_TOKEN.to_string()]),
    realm: Arc::from(b"gateway".as_slice()),
    on_unauthorized_status: 401,
    strip_prefix: Some("Bearer ".to_string()),
};
```

`realm` is just a byte-string label echoed back in the `401` response's challenge; wrapping it in `Arc` (shared ownership — many owners can point at the same bytes without copying them) is only there because `Auth` gets cloned across cores, not because the value itself needs anything special.

`Auth<Inner>` is the short-circuit `filter` idiom (see [`examples/filter`](../../examples/filter)): a pure decision admits or rejects the request before the inner pipe runs. Because `Auth` wraps the router, an unauthorized request never touches routing or the rate limiter — **the composition order is the policy order.**

## 5. Compose outside-in and serve

The chain, outermost first: `Auth` → `RoutingPipe` → `RateLimit<ForwardPipe>` → upstream. Mount and build the listener directly — `mount` accepts a handle directly (Foundations §12) — with a wildcard mount so every path reaches the gateway (`gateway/main.rs:114-123`):

```rust
let gateway_app = App::builder()
    .with_runtime_cores(1)
    .with_defaults()?
    .build()?;
gateway_app.mount("/{*rest}", into_handle(gateway_pipe))?;
let gateway_listener = gateway_app.build_listener(ListenerSpec::http(gateway_bind))?;
```

`with_runtime_cores(1)` sets the worker-thread count explicitly on the builder before `.with_defaults()?.build()?` finishes constructing the app — one core is enough for one listener answering one request at a time. `build_listener` is synchronous: it blocks the calling thread only until the accept lane has acked ready, no `futures::executor::block_on` and no `run_until_signal` — the example's own `main` is `#[proxima::main(runtime = "tokio")]`, so it already has an async context to `.await` on for the later `ShutdownBarrier` drain, but nothing here needs to `.await` the serving itself.

Run it and watch the six scenarios the example asserts end-to-end (`gateway/main.rs:159-266`, called from `main` at line 126): a missing/wrong token is `401` and the origin is never hit; `/api/...` reaches the api origin and `/web/...` the web; the third call to a capacity-2 bucket is `429` and never forwards.

```
cargo run --example gateway
```

## What you built, and the one idea

Four policies, each an ordinary `Pipe`, composed outside-in:

- **forward** — `Client` is a `SendPipe`; proxying is one call.
- **route** — `RoutingPipe` picks an upstream by path.
- **rate-limit** — `RateLimit` gates each upstream against an injected `Clock`.
- **auth** — `Auth` short-circuits unauthorized requests before any of the above.

No new primitive family, no hand-copied bytes: **a request rejected by one policy never reaches the next, because composition order is policy order.**

**Going further** — add resilience wrappers around the forward, each just another `Pipe` wrapping it: `Retry<Inner>` ([`examples/retry`](../../examples/retry)), `Fallback<P, S>` ([`examples/fallback`](../../examples/fallback)), or mirror traffic with `FanOut` ([`examples/fan_out`](../../examples/fan_out)). The [caching reverse proxy](./build-a-caching-reverse-proxy.md) and [load balancer](./build-a-load-balancer.md) reuse this same forward pipe as their base.
