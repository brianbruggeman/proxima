# principles

What proxima believes, and what it does about it. The design discipline that everything else in this codebase descends from.

For the API contract see [SHAPE.md](../SHAPE.md). For the feature inventory see [FEATURES.md](../FEATURES.md). This doc lives one level above both — it is the *why* the other two answer to.

---

## reliability is adaptive capacity

- Reliability is not the absence of failure. It is the system's capacity to keep functioning while reality refuses to hold still.
- Failure handling is degrade-gracefully, not catch-and-retry. Failure prevention is encode-the-invariant, not document-it-on-a-wiki.
- The control plane is not a set of features; it is the substance of reliability. Hot-swap, drain, status, recording, and `explain` exist so operators have leverage during incidents — and so the same machinery can be used preemptively in tests.
- `app.apply` is rollback. `app.status` surfaces blast radius. Quiesce + drain protect in-flight work. `Isolate` decides absorb-vs-amplify on a per-call basis. Every one of these is a primitive for adaptive capacity.

## operational lore must be machine-enforced

- The half-life of tribal knowledge at any growing team is shorter than the next on-call rotation. If an invariant lives only in someone's head, you have a future incident already scheduled.
- Encode invariants in something the machine reads: types, schemas, validators, tests. The medium is negotiable. The discipline is not.
- The spec system is proxima's institutional memory. JSON Schema generated from Rust types is the single source of truth, projected into every host language as Pydantic / zod / TS types — generated, never hand-written. Drift across languages is impossible by construction.

## dangerous machinery is contained at typed boundaries

- The right question is never "is this pure?" It is "where is the impurity, and how much of the codebase is allowed to know about it?"
- Mutation, `unsafe`, FFI, and async-cancellation hazards are tolerable when fenced behind narrow typed APIs. The mechanism that enforces the boundary must be visible at the boundary — not buried in a comment three modules away.
- HTTP/1's typestate read handles (`HeadReadyHandle`, `BufferedRequestHandle`, `ExpectGate`, `BodyChunkHandle`, `BodyEndHandle`) and `ResponseWriter` are the boundary. Internal buffer bookkeeping is arbitrarily complex; the external surface refuses to compile invalid sequences.
- `Isolate` middleware is the panic/exception boundary for any call into untrusted code, including host-language Pipes. A Python exception cannot crash the worker; it returns `ProximaError::Internal` with the boundary intact.

## the spec is the only door

- Pipe composition has exactly one path: the spec system (`Spec.file`, `Spec.inline`, `Spec.handle`). There is no parallel fluent DSL in any binding. There is no "advanced mode" that bypasses validation. There is no second way.
- Singular paths beat documented warnings every time. "There is only one door" outranks "use the right door" in every team larger than one.
- The FastAPI-shaped binding sugar is a *projection* of the spec system, never a separate construction API. A decorated route resolves to the same `PipeHandle` a TOML upstream would. `proxima describe` walks through host-language pipes with no carve-outs.

## proxima is workflow execution at request granularity

- Recording = event history. `check_determinism` = replay determinism check. Pipes = activities. Spec composition = workflow definition. Hot-swap = workflow versioning. The substrate is the same one durable-execution platforms provide, applied at request granularity instead of process granularity.
- The replacement target is fragile chains of `if retry < 3 then sleep else 500`, ad-hoc database-backed state machines, and cron jobs reading workflow state out of a table. Replace them with `Retry.exponential` and `WriteBack` configured on a middleware-wrapped Pipe, with recording for replay and `explain` for diagnosis.
- Determinism is a property of every Pipe, not a special workflow construct. The same `check_determinism` runs against any handle in the registry. Host-language Pipes that fail determinism fail loudly at the harness, not silently in production.

## the domain is transport-neutral

- A Pipe must be invokable from `proxima call`, an HTTP listener, an MCP listener, a QUIC listener, or another spec — and behave identically modulo transport translation. Anything less means the domain has leaked into the transport.
- `ProximaError` carries zero transport-specific variants. The variants are domain (`BadInput`, `UpstreamUnavailable`, `Cancelled`, `Internal`, ...). Listeners translate. The domain layer never carries an HTTP code or a QUIC stream ID.
- Error-to-transport mapping is centralized per listener, in one file per transport, testable in isolation. Drift between mappings is a code review concern, not a runtime concern.
- The same discipline applies to the binding boundary: `ProximaError` surfaces as a typed exception subclass in Python, a discriminated union in TS — never as a string, never as an HTTP code in host-language code.

## types serve the team, not the compiler

- Encode invariants whose violation produces *silent* corruption: framing order, transaction-then-publish atomicity, write-back-only-after-success. The feedback loop for silent failures is too long to rely on human diligence.
- Use runtime checks for invariants that fail *loudly*: spec validation, config schema, request body decoding. A clear error message at a known boundary is enough.
- Resist modeling the entire domain in types. The domain is messy; the type system wants crispness; honest engineering lives between.
- Encapsulate complex machinery behind small public APIs. `h2` is a hairy state machine; its surface is `serve_h2_connection`. HPACK is a zero-unsafe Huffman tree plus integer codec; its surface is two functions. Bindings do the same: typestate stays in Rust; host languages see `async def call`.

## every IO is observable by construction

- A library that cannot be instrumented from outside is disqualified from production use. If consumers cannot add tracing, retries, timeouts, or fault injection without forking, the library is opaque and proxima will not use it. The same standard applies in reverse — proxima will not be that library to its consumers.
- Every upstream is a `PipeHandle`, wrappable by any middleware. Every middleware is itself a `Pipe`. There is no concrete top-level `make_http_request` function; there is an `HttpUpstream` factory whose call site is interceptable.
- Middleware composes as a monoid. Each middleware is an endomorphism on `PipeHandle`; composition is associative; the identity is pass-through. Fifteen engineers can each author one middleware in isolation and the result composes for free. `Middleware::compose([..])` is the canonical stacking API.
- OpenTelemetry is the lingua franca for spans and traces. Proxima emits OTel natively across listeners, middleware, and upstreams. The `Telemetry` trait stays pluggable for custom sinks; OTel is the default.
- Libraries never log to stderr. Proxima emits structured events through `tracing`; the application decides where they go. Bindings inherit this rule: host-language code in Layer 2 does not `print`.

## compromises are named, fenced, and revisited

- Every system has parts the contract layer does not cover. The discipline is naming them, explaining why, and revisiting periodically. Hidden compromises become incidents.
- Every `Mutex` / `RwLock` carries the *WHY here / WHY NOT removable / WHY this is right* triple, with a bench citation or structural argument.
- Every deferred optimization is recorded with a measured win and a trigger condition. We do not gold-plate today's working code; we record the path back when the trigger fires.
- The `unsafe` surface is documented honestly. `h2` is zero-unsafe; transitive crates (rustls, hyper-util, tokio_uring) are not. The boundary is named at the trust line, not hidden.
- ".Internal" is a feature, not a leak. `proxima::internal::*` exposes substrate primitives (`BufferPool`, `ShardedHistogram`, per-thread sharded slot patterns) for advanced consumers under an explicit instability warning. Better than users forking the crate.

## "if it compiles, it works" is a lie for IO

- Type safety is necessary, not sufficient. Integration tests catch what types cannot: leap-day timing windows, partner APIs that interpret omitted-vs-null differently, retry logic that doubles-charges under one race.
- The proxima test floor is 510+ lib tests, 17+ e2e, property tests for concurrent edge graphs, and criterion benches for every hot path. Bindings ship at or above this density. Empty-test releases are not releases.
- Determinism is a runtime check, not a type check. `check_determinism` runs a Pipe N times against the same request and asserts byte-identical output. Non-negotiable for any Pipe, including host-language ones.

## pragmatism is the culture

- Idealism unchecked is a production liability. The engineer who rewrites the substrate in a novel type-level encoding is not shipping. Power tools are not religions; abstraction maximalism is not engineering.
- Mechanical refactoring is leveraged ruthlessly. Change a trait; the compiler hands you every call site; work through the list methodically; done. Refactor velocity is a survival trait at any non-trivial codebase size and we protect it deliberately.
- Ship the foundational shape, bench it, refine. [SHAPE.md](../SHAPE.md) decides what is correct; the discipline logs decide what is deferred; criterion decides what is measured. The three together are how proxima moves.

---

## the through-line

Strip every example and there are three claims this codebase answers to:

1. **Reliability is adaptive capacity, not absence of failure.** Hot-swap, observability, recording, and explicit boundaries are how you buy it.
2. **Operational lore must be encoded in something the machine enforces.** The spec system, the type system, schemas, and the test suite carry institutional knowledge across hires, departures, and reorganizations.
3. **Cross-cutting concerns compose by construction, not by convention.** Monoid composition of middleware is the universal pattern; per-language encoding is a detail.

Every principle above is one of these three, applied to a specific surface of the system. If a proposed change does not serve at least one of them, it does not belong.
