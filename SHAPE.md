# shape

Foundational design for the proxima host-language bindings (Python, TypeScript, and any future peer). This is the contract that every binding must satisfy, and the rule book for what they are allowed to expose.

For the Rust feature inventory see [`FEATURES.md`](FEATURES.md); for the runtime model see [`README.md`](README.md). This doc is one level above both — it decides *what shape* the bindings take and *why that shape is non-negotiable*.

## table of contents

- [premise](#premise)
- [three layers, one vocabulary](#three-layers-one-vocabulary)
- [one api, multiple transports](#one-api-multiple-transports)
- [primitives — the shared vocabulary](#primitives--the-shared-vocabulary)
- [python — fastapi-shaped sugar](#python--fastapi-shaped-sugar)
- [typescript — equivalent shape](#typescript--equivalent-shape)
- [the middleware discipline](#the-middleware-discipline)
- [the composability test](#the-composability-test)
- [what this rules out](#what-this-rules-out)
- [open questions](#open-questions)

---

## premise

`Pipe` is universal. Every binding exposes it as a first-class trait. A Python coroutine and a Rust `HttpUpstream` are peer citizens — both implement `Pipe`, both are `PipeHandle`s, both compose into specs by name, both hot-swap, both record, both `check_determinism`, both surface in `explain`.

If a TOML spec can say `http = "..."`, it must equally be able to say `pipe = "scoring"` and have that resolve to a Python class without anyone noticing.

Anything weaker than this — callbacks, second-class handlers, "advanced" surfaces — breaks the recursive composition story that proxima is built on.

---

## three layers, one vocabulary

```
                       ┌────────────────────────────────────────┐
                       │   Layer 2: host-language Pipe          │  per-request boundary cost
                       │   (ML model, stateful handler, tests)  │  same API as Rust pipes
                       ├────────────────────────────────────────┤
host-language API ──▶  │   Layer 1: orchestration (App, Spec,   │  one-time boundary cost
                       │   Mount, apply, record, explain)       │  request path is pure Rust
                       ├────────────────────────────────────────┤
                       │   Layer 0: substrate (runtime,         │  always Rust
                       │   listeners, registries, recording,    │  invariant across bindings
                       │   causal, determinism, hot-swap)       │
                       └────────────────────────────────────────┘
```

| layer | who runs it | who pays | example |
| --- | --- | --- | --- |
| 0 — substrate | Rust, always | nobody (substrate cost) | per-core runtime, listeners, `ArcSwap` registries, `Causal`, `Tee`, `Isolate` |
| 1 — orchestration | host calls into Rust | setup-time boundary cost | `App.open`, `app.pipe`, `app.mount`, `app.apply`, `app.record`, `app.metrics` |
| 2 — peer pipe | host implements `Pipe` | per-request boundary cost | Python ML model, TS handler with closure-state, integration test fixture |

The primitives (`Pipe`, `Request`, `Response`, `Body`, `Spec`, `Mount`, `PipeHandle`, `ProximaError`) span Layers 1 and 2 unchanged. That is what makes Layer 2 a peer instead of a callback.

---

## one api, multiple transports

There is no "daemon client SDK" separate from the "embedded SDK". There is one `App`, and it has transports:

| transport | constructor | use |
| --- | --- | --- |
| in-process | `App.local()` | the embedding host runs the substrate itself |
| local daemon | `App.open("ipc:///run/proxima.sock")` | controller talks to a long-running daemon over UDS |
| remote daemon | `App.open("mcp+tcp://host:port")` | controller talks to a remote daemon over MCP/TCP/TLS |

The methods on `App` are identical across transports. The wire is a deployment detail; the API surface is not.

---

## primitives — the shared vocabulary

Every binding exposes exactly this set, with identical names. No language-specific synonyms.

| primitive | role |
| --- | --- |
| `Pipe` | request → response, the unit of composition |
| `PipeHandle` | type-erased reference to a Pipe; what factories produce |
| `Request` / `Response` | typed I/O envelopes |
| `Body` | streaming bytes-or-buffer with explicit backpressure |
| `RequestContext` | telemetry, deadline, trace_id, cancel token, path params, capture sidecar |
| `Spec` | declarative pipe description: `Spec.file(path)`, `Spec.inline(value)`, `Spec.handle(svc)` |
| `Mount` / `MountTarget` | route → pipe binding with method filter |
| `App` / `AppBuilder` | lifecycle, registries, run, shutdown |
| `RunConfig` | `Http(addr)`, `Https(addr, cert, key)`, `Unix(path)`, `Mcp(...)`, etc. |
| `Shutdown` | graceful drain handle |
| `ProximaError` | structured error type with the same variants as Rust |
| middleware values | `Auth.*`, `RateLimit.*`, `Retry.*`, `Transform.*`, `WriteBack.*`, `Tee`, `Diff`, `Isolate` |

Generated, not hand-written: JSON Schema is the source of truth for `Spec`. Pydantic models (Python), `zod` schemas (TS), and `TypedDict`/TS types fall out of the schema in CI. Drift is impossible by construction.

---

## python — fastapi-shaped sugar

The decorator façade is a strict projection of the foundation. Every decorator produces a real `Pipe` and a real `Mount`; nothing parallel exists underneath.

### canonical form

```python
import asyncio
from proxima import App, Mount, Https, Spec

async def main():
    app = await App.local()
    backend = await app.pipe("backend", Spec.file("backend.toml"))
    app.mount("/{*path}", Mount.handle(backend))
    shutdown = await app.run(Https("0.0.0.0:443", cert="cert.pem", key="key.pem"))
    await shutdown.until_signal()

asyncio.run(main())
```

The request path is pure Rust. Python is a controller.

### fastapi-shaped form

```python
from proxima import App, Depends
from proxima.middleware import RateLimit, Retry, Auth
from pydantic import BaseModel

app = App()

class ScoreIn(BaseModel):
    text: str
    weights: dict[str, float] = {}

class ScoreOut(BaseModel):
    score: float

@app.post("/score", response_model=ScoreOut)
@app.use(RateLimit.per_ip(rps=100))         # rust middleware, configured from python
@app.use(Retry.exponential(max_attempts=3)) # never implemented in python
async def score(req: ScoreIn, db=Depends(get_db)) -> ScoreOut:
    return ScoreOut(score=compute(req.text))

@app.get("/health")
async def health() -> dict:
    return {"ok": True}

if __name__ == "__main__":
    app.run("0.0.0.0:443", tls=("cert.pem", "key.pem"))
```

Each piece of sugar desugars to a substrate construct:

| sugar | desugars to |
| --- | --- |
| `@app.post("/score")` | wrap fn in a `Pipe`, register as factory `"score"`, `Mount.handle("score")` at `POST /score` |
| `@app.use(RateLimit.per_ip(...))` | wrap the resulting handle with the Rust `RateLimit` middleware |
| `Depends(get_db)` | per-request DI, resolved before `Pipe.call` invokes the user function |
| `response_model=ScoreOut` | pydantic encode at the boundary; failure → `ProximaError::Internal` |
| `app.run(addr, tls=...)` | `App.local()` + `Https(addr, ...)` + `until_signal()` |

---

## typescript — equivalent shape

Structurally identical. The dialect choice (decorators vs builder) is the one open call out — both forms must satisfy the same primitives and composability rules.

```typescript
import { App, Mount, Https, Spec } from "proxima";

const app = await App.local();
const backend = await app.pipe("backend", Spec.file("backend.toml"));
app.mount("/{*path}", Mount.handle(backend));
const shutdown = await app.run(Https("0.0.0.0:443", { cert: "cert.pem", key: "key.pem" }));
await shutdown.untilSignal();
```

Builder-style Layer 2 (zod-validated):

```typescript
import { App, RateLimit, Retry } from "proxima";
import { z } from "zod";

const ScoreIn  = z.object({ text: z.string(), weights: z.record(z.number()).default({}) });
const ScoreOut = z.object({ score: z.number() });

const app = new App();

app.post("/score", {
  input: ScoreIn,
  output: ScoreOut,
  middleware: [RateLimit.perIp({ rps: 100 }), Retry.exponential({ maxAttempts: 3 })],
  handler: async (req) => ({ score: compute(req.text) }),
});

await app.run("0.0.0.0:443", { tls: { cert: "cert.pem", key: "key.pem" } });
```

---

## the middleware discipline

**Middleware is configured from host code, never implemented in it.**

| concern | runs in | why |
| --- | --- | --- |
| auth, retry, rate limit, transform, write-back, tee, diff, isolate | Rust | substrate speed (ns–µs), lock-free, observable through `proxima.*` metrics |
| business logic, model inference, integration with host-language libraries | Python / TS | crossing the boundary is the cost of using host code |

If you want a Python middleware, you actually want a Python *handler* that calls an upstream — that is Layer 2 and it is fine. The line is hard: cross-cutting concerns run in Rust at substrate speed; business logic runs in host language at boundary cost. No middle ground.

This is the single rule that keeps the bindings from drifting into "yet another web framework."

---

## the composability test

A binding is correct only if both of these work without special-casing.

### (1) a decorated app composes into a larger proxima app

```python
api = App()

@api.post("/score")
async def score(req: ScoreIn) -> ScoreOut: ...

outer = await App.local()
await outer.factories.register("scoring_api", api)
outer.mount("/v1/{*path}", Mount.handle("scoring_api"))
```

### (2) a toml spec elsewhere references the host-language app by name

```toml
# scored-cached.toml
name = "scored-cached"

[[upstreams]]
pipe = "scoring_api"   # registered above; language is invisible

[[upstreams]]
kv = "cache"
ttl = "1h"

[select]
algorithm = "fallthrough"
write_back = [[1, 0]]     # cache the python handler the same as any rust upstream
```

If either fails, the sugar has become a parallel universe and the binding has stopped being foundational. `proxima describe` must walk through host-language pipes the same way it walks through Rust ones — same tree, same node labels, no carve-outs.

---

## what this rules out

- **Fluent host-language spec DSLs** (`proxima.cache().http(...).ttl("1h")`). Spec is the architecture. A parallel builder drifts the moment the Rust types evolve. Bindings expose Pipe-authoring and Spec-loading; not Spec-construction.
- **Callback-shaped Layer 2.** Host code reaches the substrate as `Pipe`, not as a closure leashed inside someone else's pipe. The `CallbackUpstream` registry exists for in-process Rust test fixtures; it is not the binding entry point.
- **Separate "daemon client" and "embedded" SDKs.** One `App`, multiple transports. Splitting the API would force users to learn two surfaces and would let the easier one dictate the shape.
- **Python or TS implementations of substrate middleware.** Anything in [the middleware discipline](#the-middleware-discipline) table's Rust row is off-limits to host code.
- **WSGI/ASGI compatibility shims as a core feature.** Proxima has its own protocol. An ASGI adapter is a legitimate Layer 2 Pipe someone can write; it is not part of the binding surface.
- **ORM, templating, session helpers.** Out of scope. Use whatever the host ecosystem provides.

---

## open questions

These are genuinely undecided and worth deliberation before the first binding ships.

| question | tension |
| --- | --- |
| streaming / SSE shape | async-generator return as the sugar; backpressure must surface as `await` per yield, never as unbounded buffering |
| background tasks | `Pipe::background_tasks()` host surface — `@app.background` decorator? explicit lifecycle hooks? |
| middleware order | top-down (FastAPI) vs bottom-up (Rust composition order). Pick one and document it. |
| validation error surface | typed exception subclass vs auto-coerced `422`. FastAPI does both; we should pick the default. |
| openapi generation | free since spec JSON Schema already exists. Feature, not foundation. Worth shipping with v1? |
| worker model | `num_cores()` workers default. Sub-interpreters (CPython 3.13+) vs single GIL vs free-threaded build — explicit knob, not implicit. |
| typescript dialect | decorators (TC39 stage 3) vs builder. Mirror Python or lean TS-native (zod + builder)? Decide once, hold the line. |

None of these change the layering. They decide the exact feel of the surface.
