# Build a CRUD origin service

**Prerequisites:** [Foundations](./00-foundations.md) — the **transform** role and serving.
**You will:** make proxima the *origin* — a small REST service where four handlers, each a transform, are mounted by method + path over one shared store.
**New concepts (in order):** a shared store (Arc-cloned into handlers) · one handler per verb (transform) · routing by method + path (`mount_with_methods` + `MethodFilter`) · path params (`{id}`).
**Answer key:** [`examples/crud/main.rs`](../../examples/crud/main.rs) — `cargo run --example crud`.

The example frames it: *"proxima IS the origin. Every earlier example puts proxima in front of something else; here proxima answers directly."*

## 1. A shared store, cloned into each handler

The store is plain shared state; every handler holds an `Arc` clone, so cloning a pipe (which the router does once per mount) shares state instead of forking it (`crud/main.rs:33-40`):

```rust
#[derive(Clone)]
struct Store { items: Arc<Mutex<BTreeMap<u64, Bytes>>>, next_id: Arc<AtomicU64> }
```

`items` is a shared, lock-guarded map from item id to its stored bytes: `BTreeMap<u64, Bytes>` is the map itself, `Mutex` makes it safe to mutate from whichever handler is running, and `Arc` is what makes that `Mutex` shareable across the cloned handlers.

## 2. One handler per verb — each a transform

Each CRUD verb is its own `SendPipe`, `Request<Bytes> -> Response<Bytes>` — the same **transform** shape from Foundations, holding a `Store` clone. There is no second `impl Pipe for X {}` to write: `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (Foundations §13), so the moment a handler's `impl SendPipe` compiles against that signature, it is already mountable (`crud/main.rs:67-90`, CREATE):

```rust
struct CreateItem { store: Store }
impl SendPipe for CreateItem {
    type In = Request<Bytes>; type Out = Response<Bytes>; type Err = ProximaError;
    fn call(&self, request: Request<Bytes>)
        -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let (_, body) = request.body_bytes().await?;
            let new_id = store.next_id.fetch_add(1, Ordering::Relaxed);
            store.lock_items().insert(new_id, body.clone());
            Ok(Response::new(201)
                .with_header("location", format!("/items/{new_id}"))
                .with_body(body))
        }
    }
}
```

Two calls worth naming:

- `request.body_bytes()` reads the request body into a `Bytes` buffer and returns `(request, body)`; the discarded `_` is the request itself (its headers and other metadata) handed back in case a handler still needs it — dropped here because CREATE only needs the body.
- `store.lock_items()` is a small helper on `Store` (defined next to `Store::new` in the full example) that locks the shared map and hands back the guard, so every handler reaches `items` through one place.

Read / Update / Delete are the same shape with the right semantics — READ is 200 or 404; PUT updates *only* (404 on a missing id, not a silent upsert); DELETE is 204 or 404 (`crud/main.rs:95-180`).

## 3. Route by method + path

Mount one handler per (path, method) with `mount_with_methods` + `MethodFilter`: `mount_with_methods` dispatches a request only when both the path *and* the method match, and `MethodFilter::only([...])` is the list of methods that are allowed through — the method+path `Decide` (the **filter** idiom) wired into the router. Each handler still goes through `into_handle` the way Foundations taught, just called inline instead of stored in a `let` first. Path params like `{id}` are extracted by the router *before* the handler runs (`crud/main.rs:186-222, 57-63`):

```rust
app.mount_with_methods("/items",      into_handle(CreateItem { store: store.clone() }), MethodFilter::only(["POST".into()]))?;
app.mount_with_methods("/items/{id}", into_handle(ReadItem   { store: store.clone() }), MethodFilter::only(["GET".into()]))?;
app.mount_with_methods("/items/{id}", into_handle(UpdateItem { store: store.clone() }), MethodFilter::only(["PUT".into()]))?;
app.mount_with_methods("/items/{id}", into_handle(DeleteItem { store }),                MethodFilter::only(["DELETE".into()]))?;

// inside a handler: request.context.path_params.get("id")
```

An unmatched request gets a 404 from the router itself — no fallback pipe required.

## 4. Serve and drive the lifecycle

`main` builds the app via the builder, mounts the four routes, then calls `build_listener` directly — no `App::new()`/`serve`/`run_until_signal` here, since this example drives a fixed request sequence and exits rather than serving until a signal (`crud/main.rs:227-253`):

```rust
let app = App::builder().with_defaults()?.build()?;
mount_routes(&app, Store::new())?;
let listener = app.build_listener(ListenerSpec::http(bind))?;

run_crud_flow(bind);

listener.shutdown();
```

`build_listener` blocks only until the accept lane has acked ready, the same synchronous listener setup the [multi-runtime](./build-a-multi-runtime-service.md) and [gateway](./build-an-api-gateway.md) tutorials use. `run_crud_flow` then drives the full CRUD lifecycle over loopback HTTP/1 plus the sad paths (update/delete on a deleted item → 404), asserting each, before the listener is explicitly shut down (`crud/main.rs:258-317`):

```
cargo run --example crud
```

## What you built

- **shared store** — `Arc`-cloned into each handler; cloning the pipe shares state, not forks it.
- **handlers** — one `SendPipe` per verb, each a transform `Request -> Response`.
- **routing** — `mount_with_methods` + `MethodFilter` dispatch by method + path; `{id}` params are extracted before the handler runs.

Here proxima is the origin, not a proxy — but the pieces are the same: transforms, a filter (method+path) wired into the router, and shared state. Nothing new, aimed inward.
