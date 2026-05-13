# Algebraic + sans-IO `Pipe` — design sketch

Status: **SUPERSEDED BY WHAT ACTUALLY LANDED (2026-07).** This sketch
(2026-05-24) is history, not current teaching material: `proxima-pipe`
and `proxima-process` no longer exist as standalone crates (both folded
into `proxima-primitives`/`proxima-recording`), and the `Pipe` trait
this doc analyzes and proposes changing is NOT today's trait. The
CURRENT trait, verified at `proxima-primitives/src/pipe/primitives.rs`:

```rust
pub trait Pipe {
    type In;
    type Out;
    type Err: core::fmt::Debug + 'static;
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
    // + a default `and_then` combinator
}
```

No `Send`/`Sync` bound on `Pipe` itself or its returned future — `Pipe`
is the local, `!Send`-friendly root. `Send` is the separate additive
`SendPipe: Send + Sync + 'static` trait (its own `type Err: Debug + Send
+ 'static`, future `+ Send`), not a bound tacked onto `Pipe`. This
matches this sketch's own §2.2 "Path A" recommendation in spirit
(type-parametric/associated In/Out) but the actual landed shape is
associated types, not the generic-parameter form §2.1-2.3 sketch below
propose — see `docs/algebraic-pipe/discipline.md` C-GP3/C-GP4/C-GP5 for
the full history of how it actually converged. Read everything below as
the historical design questions that were asked, not as a description of
the trait as it exists today.

Status: **DRAFT** — single-pass sketch written 2026-05-24 under
`unattended`. The follow-on disciplined-component plan executes against
this; nothing here is binding yet.

This document is **Phase C.3** of
[~/.claude/plans/we-want-the-dependency-drifting-bear.md](../../../../.claude/plans/we-want-the-dependency-drifting-bear.md).
It captures the design questions, the proposed shape, and the explicit
unknowns that the follow-on plan must resolve before a single line of
the refactor lands.

The reference for the pattern is
[proxima-process](../../proxima-process). That crate already runs the
ground+operator+marker algebra against a typed `GenericPipe<In, Out>`
trait; this sketch's job is to determine how (or whether) to lift the
same pattern into `proxima-pipe` itself.

---

## 1. Current shapes (as of 2026-05-24 — see the status note above for what's true now)

### `proxima-pipe::Pipe`

[proxima-pipe/src/pipe.rs:267-278](../../proxima-pipe/src/pipe.rs#L267-L278)
(`proxima-pipe` no longer exists — see status note above):

```rust
pub trait Pipe: Send + Sync + 'static {
    fn call(&self, request: Request)
        -> impl Future<Output = Result<Response, ProximaError>> + Send;
    fn name(&self) -> &str { "anonymous" }
    fn background_tasks(&self) -> Vec<BackgroundTask> { Vec::new() }
}

pub trait DynPipe: Send + Sync + 'static {
    fn call_dyn(&self, request: Request)
        -> Pin<Box<dyn Future<Output = Result<Response, ProximaError>> + Send + '_>>;
    fn name_dyn(&self) -> &str;
    fn background_tasks_dyn(&self) -> Vec<BackgroundTask>;
}
```

Three observations:

1. **Hardcoded `Request → Response` types.** The trait can only model
   one shape of transform — the one proxima's HTTP-flavored
   `Request`/`Response` envelopes encode. Anything that wants
   `bytes → typed`, `typed → bytes`, or `Foo → Bar` either coerces into
   `Request`/`Response` (lossy, allocating) or uses a parallel trait
   (proxima-process's `GenericPipe`).
2. **`name()` + `background_tasks()` are runtime concerns on the
   abstraction.** They're proxima-runtime-side (scheduling, logging) but
   sit on the algebra. Forces every algebraic operator to forward them.
3. **`DynPipe` exists for `Arc<dyn DynPipe>` callers.** The dyn-trait
   path requires `Pin<Box<Future>>`, which requires `alloc`. The
   marker-erasure cliff lives here: an `Arc<dyn DynPipe>` cannot carry
   `NoStd`/`AllocFree`/`WithoutNetwork` bounds.

### `proxima-process::GenericPipe`

[proxima-process/src/dispatch.rs:70-85](../../proxima-process/src/dispatch.rs#L70-L85):

```rust
pub trait GenericPipe {
    type In: Send;
    type Out: Send;
    fn call(&self, input: Self::In)
        -> impl Future<Output = Result<Self::Out, ProximaError>> + Send;
    fn name(&self) -> &'static str;
}
```

[proxima-process/src/operators.rs:51-95](../../proxima-process/src/operators.rs#L51-L95):

```rust
pub struct Series<A, B> { first: A, second: B }

impl<A, B> GenericPipe for Series<A, B>
where
    A: GenericPipe + Sync,
    B: GenericPipe<In = A::Out> + Sync,
{
    type In = A::In;
    type Out = B::Out;
    fn call(&self, input: Self::In) -> impl Future<...> + Send { ... }
    fn name(&self) -> &'static str { "series" }
}

// Marker propagation — Series<A, B> is M iff both A and B are M.
impl<A: NoStd, B: NoStd> NoStd for Series<A, B> {}
impl<A: AllocFree, B: AllocFree> AllocFree for Series<A, B> {}
impl<A: IsPure, B: IsPure> IsPure for Series<A, B> {}
impl<A: WithoutFilesystem, B: WithoutFilesystem> WithoutFilesystem for Series<A, B> {}
// ...and 8 more
```

This is the worked example. Type-parametric trait + typed operator +
AND-semantic blanket impls on each marker. Composes through Series at
compile time; the `where _: NoStd + WithoutNetwork` bound at a use site
fails immediately if any leaf in the chain regresses.

---

## 2. Target shape

The goal is to make `proxima-pipe::Pipe` the **same shape** as
`GenericPipe` — type-parametric, marker-friendly — without forcing
every existing `impl Pipe for X` to be rewritten.

### 2.1 Two paths

**Path A: Promote `GenericPipe` and rename `Pipe` to its specialization.**

```rust
// proxima-pipe (or new proxima-algebra) — the algebra:
pub trait Pipe<In = Request, Out = Response>: Send + Sync + 'static {
    fn call(&self, input: In)
        -> impl Future<Output = Result<Out, ProximaError>> + Send;
}

// In proxima-pipe (compatibility): existing `Pipe` impls compile if
// they used the default In = Request, Out = Response.
```

Existing `impl Pipe for EchoPipe` continues to work because the type
parameters default to `Request, Response`. The signature change from
`fn call(&self, request: Request)` to `fn call(&self, input: In)` is
visible only in the trait declaration; impls can still write
`fn call(&self, request: Request)` because the type parameter is
already `Request`.

**Path B: Keep `Pipe` as-is, add a separate `GenericPipe` to proxima-pipe.**

Less disruption. Two traits coexist. Operators live on `GenericPipe`,
marker propagation lives on `GenericPipe`, and the existing `Pipe` is
the `GenericPipe<In = Request, Out = Response>` specialization with
blanket impl bridging in one direction.

Penalty: long-term, two traits where one would do; users have to learn
which is which.

### 2.2 Recommendation: Path A, with `Pipe<Request, Response>` as the legacy alias

```rust
// proxima-pipe/src/pipe.rs — post-refactor sketch
pub trait Pipe<In = Request, Out = Response>: Send + Sync + 'static {
    fn call(&self, input: In)
        -> impl Future<Output = Result<Out, ProximaError>> + Send;
}

// Optional helper: legacy callers that named the trait without
// type arguments keep working. They were already `Pipe`, they
// stay `Pipe`. Default type parameters Just Work.
```

Open question worth the follow-on plan answering with a prototype: does
Rust resolve `impl Pipe for X { fn call(&self, request: Request) ... }`
the same way it would resolve
`impl Pipe<Request, Response> for X { fn call(&self, input: Request) ... }`
when the parameter is named `request` vs `input`? The trait method's
parameter name is not part of the signature; only the type matters. So
yes — existing impls compile. But this needs a prototype.

`name()` and `background_tasks()` move OFF `Pipe`:

```rust
pub trait Named { fn name(&self) -> &str; }
pub trait HasBackgroundTasks { fn background_tasks(&self) -> Vec<BackgroundTask>; }
```

Existing types that need both pick them up with two extra `impl` blocks.
The algebra (`Pipe`, operators, markers) stays minimal.

### 2.3 Operators

Same pattern as proxima-process, lifted to `Pipe<In, Out>`:

```rust
pub struct Series<A, B> { first: A, second: B }

impl<A, B, In, Mid, Out> Pipe<In, Out> for Series<A, B>
where
    A: Pipe<In, Mid> + Sync,
    B: Pipe<Mid, Out> + Sync,
{
    fn call(&self, input: In) -> impl Future<Output = Result<Out, ProximaError>> + Send {
        async move {
            let intermediate = self.first.call(input).await?;
            self.second.call(intermediate).await
        }
    }
}

impl<A: NoStd, B: NoStd> NoStd for Series<A, B> {}
impl<A: AllocFree, B: AllocFree> AllocFree for Series<A, B> {}
// ...the 10 other markers, AND-semantically
```

`Tee<A, B>`, `Race<A, B>`, `Quorum<A, B, ...>`, `Match` follow the same
pattern. The marker propagation is mechanical; each operator has the
same 12 blanket impls.

### 2.4 Marker source

Comes from Phase C.1 of the cliff plan: the 12 marker traits move from
`proxima-process-protocol` to `proxima-core`, and `proxima-pipe` adds
the marker-propagating blanket impls on its operators.

---

## 3. Backward-compat strategy (sketch)

### 3.1 Existing `Pipe` impls

Inventory (rough, from grep): ~60 `impl Pipe for X` sites across the
workspace today. Of those:

- Most are leaves: upstreams (HttpUpstream, SynthUpstream, KvCache,
  Process*, Pty*), middleware (Auth, RateLimit, Retry, Transform).
- Most use `Request`/`Response` directly — they would compile unchanged
  under Path A.
- A few wrap inner pipes generically (Tee, Diff, Isolate, Swap,
  Causal, SwappablePipe, WriteBack). These may need an explicit
  `Pipe<Request, Response>` annotation to disambiguate during the
  transition.

### 3.2 `DynPipe` and `Arc<dyn DynPipe>`

The `DynPipe` trait stays for the same reason it exists today: runtime
polymorphism (registry lookups, hot-swap, config-driven chains).
**Markers don't propagate across the dyn boundary** — that's a fact of
Rust's object safety, not a flaw. The proxima-process crate already
documents this as the "marker-erasure cliff" via `DynDispatchChain`.

Pattern: typed `Pipe<In, Out>` carries markers; the moment a caller
hands it to a registry (`Arc<dyn DynPipe>`), the marker proofs vanish.
Anything that needs markers stays typed all the way down. Anything that
needs runtime polymorphism (e.g. `proxima-control-plane` hot-swap)
accepts the marker-erasure cost.

The follow-on plan must decide:

- Does `DynPipe` become `DynPipe<In, Out>` parallel to `Pipe<In, Out>`?
- Or does `DynPipe` stay shaped as `Arc<dyn DynPipe<In = Request, Out =
  Response>>` (default-typed)?
- Or do we have one `DynPipe` per common shape (DynPipe for HTTP,
  DynBytePipe for `bytes → bytes`, ...)?

Recommendation worth prototyping: keep `DynPipe` HTTP-shaped (`In =
Request, Out = Response`), introduce `DynBytePipe` for sans-IO
byte→byte work if/when it earns its keep. Don't proliferate dyn
variants.

### 3.3 Composition primitives (`Tee`, `Diff`, `Isolate`, `Swap`)

Currently in `proxima-compose`. Use `Arc<dyn DynPipe>` internally,
spawn tokio tasks for fan-out.

Two options after the refactor:

1. **Typed versions in `proxima-pipe`/operators:** `Tee<A, B>` as a
   typed struct with blanket impls. Marker propagation works.
   Compose-time fan-out is type-driven.
2. **Keep current `proxima-compose` impls as the dyn-tier path:** they
   stay `Arc<dyn DynPipe>`-based, lose markers, but retain runtime
   polymorphism. Useful for config-driven chains (`Tee(pipe = "auth",
   pipe = "log")`).

Likely answer: both. Typed operators in the algebra crate for compile-
time use; dyn operators in `proxima-compose` for runtime use. Same
shape, different tier — same as the
`GenericPipe`/`GenericPipeDyn` split proxima-process already has.

---

## 4. Open design questions (the follow-on plan must answer)

These are explicit decision points, not aspirational. Each blocks
actual code until resolved.

### Q1. `Future`-returning under no_std

`core::future::Future` exists in `core`. So `impl Future + Send` in a
trait method works under `#![no_std]`. But:

- **`Send` bound under no_std + no-alloc-free.** `Send` is meaningful
  only when there's a runtime that moves values between threads.
  Bare-metal single-core targets don't need `Send`. Options: (a) keep
  `Send` unconditionally (cheap; embeds the std assumption); (b) drop
  `Send` via a feature flag; (c) two traits (`Pipe` with `Send`,
  `LocalPipe` without). Today proxima-pipe already has
  `ThreadLocalPipe` — that's the precedent for option (c).
- **`Pin<Box<dyn Future>>` requires alloc.** Confirms: `DynPipe`
  requires alloc; typed `Pipe<In, Out>` doesn't.

### Q2. Where does the algebra crate live?

Three options:

| option | host crate | trade-off |
|---|---|---|
| A | proxima-pipe | obvious; tight coupling of algebra to one canonical In/Out |
| B | new proxima-algebra | clean separation; both proxima-pipe + proxima-process consume it; one more crate |
| C | proxima-core | already on the cliff; markers are already moving there in C.1 |

Recommendation: B. `proxima-algebra` carries `Pipe<In, Out>`, operators
(`Series`, `Tee`, `Race`, `Match`), and the marker blanket impls. Both
`proxima-pipe` (HTTP-flavored specialization) and `proxima-process`
(byte-flavored specialization) depend on it. proxima-process's
`GenericPipe` either renames to `proxima_algebra::Pipe` or stays as a
local alias.

Counter: C avoids adding a crate. The markers ARE already going to
proxima-core. The algebra (Pipe trait + operators) could co-locate.
The downside: proxima-core would gain a trait surface beyond errors,
which expands its public API more than the C.1 move alone.

### Q3. Effect markers — where exactly?

Three approaches:

| approach | shape |
|---|---|
| 1. On the `Pipe` impl | `impl NoStd for MyPipe {}` next to `impl Pipe for MyPipe` |
| 2. As phantom on Pipe's generics | `Pipe<In, Out, Effects = (NoStd, WithoutNetwork)>` — heavy |
| 3. Sibling trait, blanket-propagated | `trait WithoutNetwork {}` + propagation on operators (today's proxima-process pattern) |

Recommendation: 3. Already proven by proxima-process. Adds zero
runtime cost. Reads naturally at use sites (`where P: Pipe + NoStd +
WithoutNetwork`). The 12 markers move to proxima-core in C.1; operators
in proxima-algebra (or proxima-pipe) carry the blanket impls.

### Q4. Marker derivation — manual `impl` or via `#[derive]`?

Manual `impl WithoutFilesystem for MyParser {}` is what proxima-process
does. It's explicit; the developer must consciously assert the
property. A derive macro (`#[derive(WithoutFilesystem)]` or
`#[proxima_marker(WithoutFilesystem)]`) would be ergonomic but risks
hiding the assertion. **Manual.** The marker IS the assertion; making
it derive-generated weakens the audit.

### Q5. `Send + Sync + 'static` — keep, relax, or stratify?

Current `Pipe` requires all three. For bare-metal single-core targets,
`Send` and `Sync` are spurious. Stratify with a parallel trait
(`LocalPipe`) when the no-alloc work makes single-core a real target;
until then, keep `Send + Sync + 'static` on the algebraic Pipe.
proxima-pipe already has `ThreadLocalPipe` as the precedent.

### Q6. `name()` and `background_tasks()`?

Move off Pipe into `Named` + `HasBackgroundTasks` traits. Existing
impls add two extra `impl` blocks; the algebra (Pipe + operators)
stays minimal. Operators don't have to forward what they don't model.

### Q7. What about `RequestContext` / capture / telemetry?

Today's `Pipe::call(Request)` has the `RequestContext` baked into
`Request`. A generic `Pipe<In, Out>` shouldn't assume Request contains
context. Two paths:

- Pass context as a second arg: `fn call(&self, input: In, ctx:
  &Context) -> ...`. Heavier signature.
- Wrap inputs in a `Contextual<In>`: `fn call(&self, input:
  Contextual<In>) -> Future<Output = Contextual<Out>>`. Composable but
  adds wrapping.
- Leave context to each domain's specialization: `Pipe<Request,
  Response>` (the proxima-pipe canonical form) keeps `Request`
  containing context; `Pipe<Bytes, Bytes>` for sans-IO doesn't care
  about context.

Recommendation: the third. The algebra doesn't impose a context model;
domains layer it onto their In/Out types.

### Q8. `ProximaError` — error type parameter?

Today `Pipe::call(...) -> Future<Output = Result<Response,
ProximaError>>`. The error type is fixed. For an algebra that may host
domain-specific errors (e.g. proxima-process's grounds may want a
narrower error), do we parameterize?

| approach | shape |
|---|---|
| Fixed | `-> Future<Output = Result<Out, ProximaError>>` (today) |
| Parameterized | `Pipe<In, Out, Err = ProximaError>` |

Recommendation: keep fixed at first. ProximaError is the workspace's
shared error vocabulary. Adding `Err = ProximaError` as a default type
parameter is mechanically safe and can be done in a later iteration
without breaking call sites.

### Q9. Public surface promise

If `proxima-algebra` (or `proxima-pipe`) exposes `Pipe<In, Out>` as a
public trait, every In/Out used at public boundary is part of the API.
The discipline: only standardize specializations that have a clear
home turf (HTTP, bytes, protobuf-shaped types). Domain-specific In/Out
types stay in the domain crate; the algebra is open over them.

---

## 5. Connection to proxima-process

proxima-process is the worked example. The pattern works there because:

- It's a small surface (~10 grounds, 1 operator, 12 markers, 3 effect
  domains)
- The grounds are simple enough that hand-asserting markers is
  obvious from the code
- The composition is sequential (Series) for now; Race / Tee / Quorum
  follow when needed

If we extract `proxima-algebra`, proxima-process's `GenericPipe` either:

1. Renames to `proxima_algebra::Pipe` — same trait, different home.
2. Stays as a local alias: `pub use proxima_algebra::Pipe as
   GenericPipe;` — protects the existing module path.

Recommendation: option 2 during transition, option 1 after the follow-
on plan stabilizes.

The proxima-process discipline log
[proxima-process tests/type_system_guarantees.rs](../../proxima-process/tests/type_system_guarantees.rs)
should be the template for `proxima-algebra/tests/type_system_guarantees.rs`
— same 23 trait-bound + wire-format checks, lifted to the new home.

---

## 6. What this sketch does NOT decide

- Whether to break `Pipe`'s current method signature (decided: no —
  Path A keeps it via default type parameters)
- Whether all current `impl Pipe for X` sites need rewrites (decided:
  most don't; a small typed-operator subset does)
- How marker derivation handles generics (Q4: manual, no derive)
- The actual API of `Tee`/`Race`/`Quorum`/`Match` (Q4 from the
  knob-inventory says implement when needed; proxima-process has only
  Series + dispatch_match today)
- Bench targets (the follow-on disciplined-component plan picks them)
- The eventual `proxima-algebra` vs. proxima-pipe-internal placement
  (Q2 still open)

---

## 7. Validation criteria for the follow-on plan

The follow-on plan's "definition of done" must include:

1. A prototype impl of `Pipe<In, Out>` that compiles existing `impl
   Pipe for X { fn call(&self, request: Request) ... }` sites
   **unchanged**.
2. `Series<A, B>` + the 12 marker blanket impls landed.
3. A `tests/type_system_guarantees.rs` with at minimum one positive
   and one negative compile-fail case per marker (12 markers × 2 = 24
   trybuild cases).
4. A bench arm comparing the typed `Series<A, B>` against the existing
   `proxima-compose::Tee` (or whichever comp prim is the closest
   incumbent) — `design-favors: incumbent` arm required per
   disciplined-component gate point 13.
5. A worked example in proxima-process where a marker bound (`P: Pipe
   + WithoutNetwork`) fails to compile when a network-touching ground
   is added — proof the markers actually catch what they claim to.
6. The marker promotion from Phase C.1 of the cliff plan is the
   prerequisite; algebra crate (or proxima-pipe extension) is one
   layer above.

---

## 8. Glossary

| term | meaning |
|---|---|
| Pipe | The unit of composition. `Pipe<In, Out>` after the refactor; `Pipe` (fixed In=Request, Out=Response) before. |
| ground | A leaf Pipe — implements the trait directly, doesn't compose other pipes. Proxima-process term. |
| operator | A typed wrapper that composes pipes (Series, Tee, Race, Match). |
| marker | An empty trait (NoStd, AllocFree, WithoutNetwork, ...) asserting a property of the type. Zero runtime cost. |
| marker propagation | Blanket impls on operators: `Series<A,B>: NoStd` iff `A: NoStd AND B: NoStd`. AND-semantic. |
| effect-absence | Negative form for effects: `WithoutNetwork` etc. AND-propagates through operators; OR would require specialization (unstable). |
| determinism hierarchy | Markers from weakest to strongest: `Deterministic` ⊃ `Reproducible`; `IdempotentSideEffectFree`; `Commutative`. |
| marker-erasure cliff | When a typed `Pipe<In, Out>` is wrapped in `Arc<dyn DynPipe>`, marker proofs vanish — runtime polymorphism cost. |
