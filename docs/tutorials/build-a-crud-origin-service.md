# Build a CRUD origin service

**Prerequisites:** [Foundations](./00-foundations.md) — the **transform** role and `Handler`/`into_handle`.
**You will:** make proxima the *origin* — a small REST service where four handlers, each a transform, are mounted by method + path over one shared store.
**New concepts (in order):** a shared store (`Arc`-cloned into handlers) · one handler per verb (transform) · the `Handler` blanket impl · routing by method + path (`mount_with_methods` + `MethodFilter`) · path params (`{id}`).
**Answer key:** [`examples/crud/main.rs`](../../examples/crud/main.rs) — `cargo run --example crud --features http1-native`.

The example frames it: *"proxima IS the origin. Every earlier example puts proxima in front of something else; here proxima answers directly."*

## 1. A shared store, cloned into each handler

The store is plain shared state; every handler holds an `Arc` clone, so cloning a pipe (which the router does once per mount) shares state instead of forking it (`crud/main.rs:35-39`):

```rust
use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, MutexGuard, PoisonError};

#[derive(Clone)]
struct Store {
    items: Arc<Mutex<BTreeMap<u64, Bytes>>>,
    next_id: Arc<AtomicU64>,
}

impl Store {
    fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(BTreeMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn lock_items(&self) -> MutexGuard<'_, BTreeMap<u64, Bytes>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
```

`items` is a shared, lock-guarded map from item id to its stored bytes: `BTreeMap<u64, Bytes>` is the map itself, `Mutex` makes it safe to mutate from whichever handler is running, and `Arc` is what makes that `Mutex` shareable across the cloned handlers. `lock_items` recovers a poisoned lock instead of propagating the poison — one failed request shouldn't take every later request down with it.

## 2. One handler per verb — each a transform

Each CRUD verb is its own `SendPipe`, `Request<Bytes> -> Response<Bytes>` — the same **transform** shape from Foundations, holding a `Store` clone. There is no second `impl Pipe for X {}` to write: `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (`proxima-primitives/src/pipe/handler.rs:97-100`), so the moment a handler's `impl SendPipe` compiles against that signature, it is already mountable. CREATE assigns the next id, stores the request body verbatim, and hands the id back as a `Location` header (`crud/main.rs:66-89`); READ is the one worth showing alongside it, since it is where the "200 or 404" behavior every other verb needs actually lives — `{id}` is a path param, a fragment the router extracted from the URL before this pipe ever ran, read back out of `request.context.path_params` (`crud/main.rs:56-62, 93-117`; `Store` repeated here so the block stands on its own):

```rust
use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, MutexGuard, PoisonError};

#[derive(Clone)]
struct Store {
    items: Arc<Mutex<BTreeMap<u64, Bytes>>>,
    next_id: Arc<AtomicU64>,
}

impl Store {
    fn new() -> Self {
        Self {
            items: Arc::new(Mutex::new(BTreeMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    fn lock_items(&self) -> MutexGuard<'_, BTreeMap<u64, Bytes>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

struct CreateItem {
    store: Store,
}

impl SendPipe for CreateItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
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

fn item_id(request: &Request<Bytes>) -> Option<u64> {
    request
        .context
        .path_params
        .get("id")
        .and_then(|raw| raw.parse().ok())
}

struct ReadItem {
    store: Store,
}

impl SendPipe for ReadItem {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let store = self.store.clone();
        async move {
            let Some(resource_id) = item_id(&request) else {
                return Ok(Response::not_found());
            };
            match store.lock_items().get(&resource_id) {
                Some(value) => Ok(Response::ok(value.clone())),
                None => Ok(Response::not_found()),
            }
        }
    }
}
```

Two calls worth naming from CREATE:

- `request.body_bytes()` reads the request body into a `Bytes` buffer and returns `(request, body)` (`proxima-primitives/src/pipe/request.rs:317`); the discarded `_` is the request itself (its headers and other metadata) handed back in case a handler still needs it — dropped here because CREATE only needs the body.
- `store.lock_items()` locks the shared map and hands back the guard, so every handler reaches `items` through one place.

From READ: a missing or unparseable `{id}` and a well-formed `{id}` that isn't in the store both fall through to the same `Response::not_found()` — the router only guarantees a *syntactic* match on `/items/{id}`, never that the id exists.

UPDATE and DELETE are the identical shape again, `item_id` plus a `lock_items()` call, with one semantic difference each worth stating plainly rather than inferring from the code: PUT *updates only* — a missing id is 404, not a silent upsert (`crud/main.rs:125-148`); DELETE is 204 on success, 404 on an id that was never there or already gone (`crud/main.rs:156-176`).

## 3. Route by method + path

Mount one handler per (path, method) with `mount_with_methods` + `MethodFilter` (`src/app.rs:743`, `proxima-primitives/src/pipe/routing.rs:74-85`): `mount_with_methods` dispatches a request only when both the path *and* the method match, and `MethodFilter::only([...])` is the list of methods that are allowed through — the method+path `Decide` (the **filter** idiom) wired into the router. Each handler still goes through `into_handle` the way Foundations taught. The real file factors this into its own `mount_routes` function (`crud/main.rs:181-217`); the same shape, mounting the two handlers built above:

```rust
fn mount_routes(app: &App, store: Store) -> Result<(), ProximaError> {
    app.mount_with_methods(
        "/items",
        into_handle(CreateItem {
            store: store.clone(),
        }),
        MethodFilter::only(["POST".to_string()]),
    )?;
    app.mount_with_methods(
        "/items/{id}",
        into_handle(ReadItem { store }),
        MethodFilter::only(["GET".to_string()]),
    )?;
    Ok(())
}
```

UPDATE and DELETE mount identically, just swapping in `UpdateItem`/`DeleteItem` and `"PUT"`/`"DELETE"` (`crud/main.rs:200-214`). An unmatched request gets a 404 from the router itself — no fallback pipe required.

## 4. Serve and drive the lifecycle

`main` builds the app via the builder, mounts the routes, then calls `build_listener` directly — no `App::new()`/`serve`/`run_until_signal` here, since this example drives a fixed request sequence and exits rather than serving until a signal (`crud/main.rs:222-251`):

```rust
async fn start(bind: SocketAddr) -> Result<(), ProximaError> {
    let app = App::builder().with_defaults()?.build()?;
    let store = Store::new();

    app.mount_with_methods(
        "/items",
        into_handle(CreateItem {
            store: store.clone(),
        }),
        MethodFilter::only(["POST".to_string()]),
    )?;
    app.mount_with_methods(
        "/items/{id}",
        into_handle(ReadItem { store }),
        MethodFilter::only(["GET".to_string()]),
    )?;

    let listener = app.build_listener(ListenerSpec::http(bind))?;

    listener.shutdown();
    let runtime = app
        .runtime()
        .ok_or_else(|| ProximaError::Config("app has no runtime installed".into()))?;
    let report = ShutdownBarrier::new(runtime).broadcast_drop().await;
    println!(
        "drained: cores_acked={} hooks_drained={}",
        report.cores_acked, report.hooks_drained
    );

    Ok(())
}
```

`build_listener` (`src/app.rs:1114`) blocks only until the accept lane has acked ready, the same synchronous listener setup the [multi-runtime](./build-a-multi-runtime-service.md) and [gateway](./build-an-api-gateway.md) tutorials use — no polling, no sleeping. `listener.shutdown()` (`proxima-listen/src/handle.rs:602`) consumes the handle and stops accepting; `ShutdownBarrier::broadcast_drop()` (`proxima-primitives/src/sync/shutdown.rs:157`) then drains every core and background hook the runtime is holding, reporting how many of each it caught. The real file's `main` runs this exact sequence, plus a `run_crud_flow` step in between that drives the full CRUD lifecycle over loopback HTTP/1 — create, read, update, re-read, delete, re-read — plus the sad paths (update/delete on a deleted item → 404), asserting each (`crud/main.rs:256-315`):

```
cargo run --example crud --features http1-native
```

## What you built

- **shared store** — `Arc`-cloned into each handler; cloning the pipe shares state, not forks it.
- **handlers** — one `SendPipe` per verb, each a transform `Request -> Response`, each already a `Handler` via the blanket impl.
- **routing** — `mount_with_methods` + `MethodFilter` dispatch by method + path; `{id}` params are extracted before the handler runs, and a missing-vs-not-found `{id}` both resolve to the same 404.

Here proxima is the origin, not a proxy — but the pieces are the same: transforms, a filter (method+path) wired into the router, and shared state. Nothing new, aimed inward.
