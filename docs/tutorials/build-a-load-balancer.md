# Build a load balancer

**Prerequisites:** [Foundations](./00-foundations.md); the [Gateway's forward proxy](./build-an-api-gateway.md) (§1).
**You will:** distribute requests across a pool of healthy backends, skipping unhealthy ones — proxy's forward plus a selection policy over N clients.
**New concepts (in order):** a backend pool (a `Client` per backend + a health flag) · round-robin selection over the healthy subset · forwarding to the picked backend.
**Answer key:** [`examples/load-balance/main.rs`](../../examples/load-balance/main.rs) — `cargo run --example load-balance`.

The example frames it: *"A load balancer is `proxy`'s forward composed with a selection policy over a pool. `fan-in` taught 'many sources, one merged stream, pull only the ready'; this is the mirror shape — one request in, one backend picked out of many, skipping whichever aren't ready (here: not healthy)."*

## 1. A pool of backends

Instead of one `Client` bound to one upstream (the gateway's forward), N clients bound to N backends, each with a health flag (`load-balance/main.rs:65-69, 147-151`):

```rust
struct Backend { label: &'static str, healthy: bool, client: Client }

let backends = vec![
    build_backend(origin_a_bind, "origin-a", true)?,   // healthy
    build_backend(origin_b_bind, "origin-b", false)?,  // unhealthy — never selected
    build_backend(origin_c_bind, "origin-c", true)?,   // healthy
];
```

`build_backend` (`load-balance/main.rs:213-224`, not shown) builds one `Backend` from an origin address, a label, and a health flag.

## 2. Round-robin over the healthy subset

`LoadBalancerPipe` holds the pool and a cursor. `select_backend` walks the full pool from the cursor and returns the first healthy client; none healthy is the pool being *down* — a typed error, not a papered-over fallback (`load-balance/main.rs:92-102`):

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
    None   // pool down
}
```

`fetch_add` bumps and reads the counter in one atomic step so parallel requests don't collide — you don't need more than that here.

The cursor walks the *full* pool (not a healthy-only subset), so a backend that later recovers resumes its place in rotation with no re-registration step.

## 3. Forward to the picked backend

The pipe's `call` is exactly proxy's forward, aimed at whichever `Client` selection picked (`load-balance/main.rs:105-124`):

```rust
impl SendPipe for LoadBalancerPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, request: Request<Bytes>)
        -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let selected = self.select_backend();
        async move {
            match selected {
                Some(client) => SendPipe::call(&client, request).await,   // forward
                None => Err(/* no healthy backend available */),
            }
        }
    }
}
```

That `None` arm is elided pseudocode — `Err(/* ... */)` doesn't compile as written. The real code returns a typed error: `Err(ProximaError::Io(std::io::Error::other("load balancer: no healthy backend available")))` (`load-balance/main.rs:118-120`).

That is the whole type — there is no second `impl Pipe for LoadBalancerPipe {}` marker to write. `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (Foundations §13), so `LoadBalancerPipe` is already mountable the moment its `impl SendPipe` above compiles — "there is no second trait to opt into and nothing more to write."

Mount and serve as in Foundations, through `App::builder().with_defaults()?.build()?` + `app.mount("/", pipe)?` + `app.build_listener(ListenerSpec::http(bind))?` (`load-balance/main.rs:243-256`). The example drives 12 requests and asserts they split evenly across the two healthy backends while the unhealthy one sees zero (`load-balance/main.rs:261-297`).

## What you built

- **pool** — a `Client` per backend plus a health flag.
- **selection** — round-robin over the healthy subset; an all-unhealthy pool is a typed failure, not a hidden fallback.
- **forward** — the picked backend gets proxy's one-call forward.

A load balancer is proxy plus a selection policy — the mirror of fan-in (one out of many, skipping the unready). This example **hand-rolls** round-robin to show the shape plainly; the [caching reverse proxy](./build-a-caching-reverse-proxy.md)'s `Selection`/`Fallthrough` is the same idea as a reusable primitive. Swap `select_backend` for a `Selection` strategy when you want weights, least-connections, or health probes — without rewriting the forward.
