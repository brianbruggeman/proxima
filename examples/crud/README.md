# crud

proxima *is* the origin: a small REST service.

## Builds on

[transform](../transform/README.md) — each handler is the same `Pipe` shape:
`Request<Bytes> -> Response<Bytes>`.
[filter](../filter/README.md) — routing is the same method+path `Decide` idea,
wired into the router instead of a standalone gate.

## What it demonstrates

Every earlier example in this curriculum puts proxima *in front of*
something else — a transform, an upstream, a filter. `crud` is proxima
answering directly: no upstream, no proxying, just handlers and state.

A REST service is two things composed:

1. **routing** — dispatch by method + path, so `POST /items` reaches a
   different handler than `GET /items/{id}`.
2. **handlers** — small pipes that read and mutate shared state.

Routing is `App::mount_with_methods`, called once per (path, method) pair.
Each mount is a `Mount { path: PathPattern, methods: MethodFilter, .. }`;
the router tries them in registration order and calls the first one whose
path *and* method both match. No route matching is a 404 from the router
itself — no fallback pipe required.

| Method | Path         | Handler       | Behavior                                  |
|--------|--------------|---------------|--------------------------------------------|
| POST   | `/items`     | `CreateItem`  | assigns the next id, stores the body, 201 with `Location` |
| GET    | `/items/{id}`| `ReadItem`    | 200 + body, or 404                         |
| PUT    | `/items/{id}`| `UpdateItem`  | replaces an existing value, or 404 (never upserts) |
| DELETE | `/items/{id}`| `DeleteItem`  | removes an existing value, 204, or 404     |

`{id}` is a `PathPattern` param: the router extracts it from the URL before
the handler ever runs, and hands it over as
`request.context.path_params["id"]`.

Each handler holds a `Store` — one `Arc<Mutex<BTreeMap<u64, Bytes>>>` plus an
`Arc<AtomicU64>` id counter. `App` clones a pipe once per mount, and cloning
`Store` is an `Arc` refcount bump, so all four handlers share one map instead
of forking their own. That's the whole "shared state" story: no database, no
extra primitive, just interior mutability behind the same clone every other
example already relies on.

`UpdateItem` and `DeleteItem` both check existence before mutating and return
404 on a miss — a `PUT` on a missing id is not silently treated as a create,
and a `DELETE` on a missing id is not a silent no-op. That asymmetry (`POST`
always creates, `PUT`/`DELETE` require the resource to already exist) is
exercised directly in `main`'s sad-path assertions, not left to the reader's
imagination.

## Run

```
cargo run --example crud
```

## What you'll see

```
listening on 127.0.0.1:8080 (prime runtime, 1 core)
POST /items ->
201 banana bread

GET /items/1 ->
200 banana bread

PUT /items/1 ->
200 banana bread, toasted

DELETE /items/1 ->
204 

GET /items/1 (after delete) ->
404 

drained: cores_acked=1 hooks_drained=0
```

The client is a plain blocking `TcpStream`, same as `hello`'s — proof that
`127.0.0.1:8080` is a real HTTP/1 server, not an in-process call. `main`
drives the full lifecycle (create, read, update, re-read, delete, read-after-
delete) and then the two sad paths (`PUT`/`DELETE` on the now-deleted id),
asserting on status and body at every step — a routing or state-sharing
regression fails the example run, not just the eyeball check.
