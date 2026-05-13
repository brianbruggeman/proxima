# gateway — proxy + policy (auth · route · rate-limit)

A gateway is `proxy`'s forward pipe with a policy chain composed in front of it: reject bad callers, pick an upstream, throttle over a budget — then, and only then, forward.

## Builds on

[proxy](../proxy/README.md) — the forward itself: `SendPipe::call(&client, request)`, unchanged.
[gate](../gate/README.md) — admission-as-composition; `RateLimit` is the rate-shaped instance of the same idea.
[filter](../filter/README.md) — reject-before-the-inner-pipe; `Auth` is the bearer-token instance of the same idea.

## What it demonstrates

Nothing here is a new primitive family. A gateway is three existing `Pipe`s nested around a forward:

```
Auth<RoutingPipe<RateLimit<ForwardPipe>>>
```

- **AUTH** (`proxima::Auth<Inner>`) — reads the `authorization` header, strips `Bearer `, checks it against an allow-list. No match (missing or wrong) short-circuits with 401 before `Inner` is ever called — the same reject-before-the-inner-pipe shape `filter`'s `Decide` taught, just with the predicate and the rejection response both built in.
- **ROUTE** (`proxima::RoutingPipe<Handle>`) — a pattern-routed dispatcher: `/api/{*rest}` goes to one upstream, `.fallback(...)` sends everything else to another. The routing decision is which `Handle` gets the request, not a rewrite of it.
- **RATE-LIMIT** (`proxima::RateLimit<Inner, Extractor, Clk>`) — a per-key token bucket in front of each upstream. A token available admits; an empty bucket answers 429 with `retry-after` and never calls `Inner`. Each upstream gets its own bucket, because `RateLimit` wraps the upstream's own forward pipe, not the router.

The chain order is the enforcement order: a request that fails AUTH never reaches ROUTE or RATE-LIMIT; a request that RATE-LIMIT throttles never reaches the origin. Two origins (`api`, `web`), each with their own call counter, prove both the routing decision and the throttle boundary — not by reading code, by an `AtomicUsize` that does or doesn't move.

## Policy chain

| Policy | Primitive | Reject code | What's checked |
|---|---|---|---|
| AUTH | `Auth<Inner>` | 401 | `authorization: Bearer <token>` against an allow-list |
| ROUTE | `RoutingPipe<Handle>` | — (dispatch, not a reject) | path prefix (`/api/{*rest}` vs. fallback) |
| RATE-LIMIT | `RateLimit<Inner, Extractor, Clk>` | 429 | per-upstream token bucket, keyed by a constant key |

## Run

```
cargo run --example gateway
```

## What you'll see

```
auth: missing token never reaches route or the origin
HTTP/1.1 401 Unauthorized
www-authenticate: Bearer realm="gateway"
...
route: authorized "/api/..." forwards to the api upstream
HTTP/1.1 200 OK
x-upstream: api
...
route: authorized "/web/..." falls through to the web upstream
HTTP/1.1 200 OK
x-upstream: web
...
rate-limit: a third call exceeds the budget (429), origin never hit
HTTP/1.1 429 Too Many Requests
retry-after: 1
...
PASS: auth rejects before route, route sends each prefix to its own upstream, rate-limit
throttles per upstream before the forward — three composed policies, no bytes copied by hand.
```

Six requests, six assertions: no token and a wrong token both 401 with the origin call counters still at zero; an authorized `/api/...` request reaches the api origin (`x-upstream: api`); an authorized `/web/...` request falls through to the web origin (`x-upstream: web`); a second `/api/...` call still fits the capacity-2 budget; a third exceeds it and gets 429 with the api counter unchanged — the throttle fired before the forward, not after. The budget is exhausted purely by call count (`refill_per_sec: 0`) over an injected `FakeClock` that never advances on its own — no sleeps, no wall-clock race.
