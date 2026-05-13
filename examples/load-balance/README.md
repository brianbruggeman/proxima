# load-balance

Distribute across N healthy backends.

**Builds on:** proxy, fan-in

## Builds on

[proxy](../proxy/README.md) — the forward. [fan-in](../fan_in/README.md) — the "many
sources, pick among them" shape, mirrored: fan-in merges many sources into one stream;
load-balance picks one backend out of many per request.

## What it demonstrates

A load balancer is not new machinery — it's `proxy`'s forward, aimed at whichever backend a
selection policy picks. `LoadBalancerPipe` holds a pool of `Backend`s (each a `Client` plus a
health flag) and a rotation cursor; `call` picks a backend, then does exactly what `ProxyPipe`
does:

```rust
fn select_backend(&self) -> Option<Client> {
    let backend_count = self.backends.len();
    for _ in 0..backend_count {
        let index = self.cursor.fetch_add(1, Ordering::SeqCst) % backend_count;
        let backend = &self.backends[index];
        if backend.healthy {
            return Some(backend.client.clone());
        }
    }
    None
}
```

The cursor walks the *whole* pool, not a pre-filtered healthy subset — an unhealthy backend is
skipped every time it comes up in rotation, and a backend that later recovers resumes its place
for free, with no separate re-registration step. No backend ever falls out of the algebra; it's
just never selected while unhealthy.

| policy | health handling |
|---|---|
| round-robin (`cursor` mod pool size) | skips unhealthy, resumes rotation once it's healthy again |

The example stands up three origins (`origin-a`, `origin-b`, `origin-c`), marks `origin-b`
unhealthy via a plain `bool` on its `Backend` entry — the listener still runs, the load balancer
just never routes to it — and drives 12 requests through the load balancer. Each origin counts
its own hits in an `AtomicU32`; that counter, not the load balancer's internal state, is what the
assertions check.

## Run

```
cargo run --example load-balance --features "runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http-prime-deps"
```

Same feature set as `proxy` — `http-prime-deps` for `Client::http`'s prime backend, the
`runtime-prime-*` quartet for the listeners.

## What you'll see

```
origin-a listening on 127.0.0.1:8091 (healthy)
origin-b listening on 127.0.0.1:8092 (marked unhealthy)
origin-c listening on 127.0.0.1:8093 (healthy)
load balancer listening on 127.0.0.1:8090, pool: origin-a(healthy), origin-b(unhealthy), origin-c(healthy)

request  1: served by origin-a
request  2: served by origin-c
...
request 12: served by origin-c

per-backend counts: origin-a=6 origin-b=0 origin-c=6
PASS: distributed across healthy backends only, unhealthy backend saw zero requests.
```

12 requests over 2 healthy backends split exactly 6/6 — round-robin over a fixed rotation, no
randomness. `origin-b` answers zero of them: the unhealthy flag is checked, not bypassed. The
per-request line comes from parsing the `x-backend-id` header each origin stamps on its own
response — proof of which backend actually answered, not an assumption from the load balancer's
side.
