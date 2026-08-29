# boundary — one config-selected slot to integrate, then replace, a service

Stand an existing service (`theirs`) next to your replacement (`ours`).
Pick ONE control point that both traffic strategies pass through, set it
once, and walk it forward via config as the migration proceeds — never a
recompile, never a scatter of `if replacing_theirs` checks through the
codebase.

## Builds on

- [record](../record/README.md) — the `RecordUpstream` + cassette sink stack
  this example's `Record` arm reuses verbatim.
- [replay](../replay/README.md) — `ReplayUpstream` and the typed
  `ProximaError::ReplayMiss` this example's `Replay` arm reuses verbatim.
- [fallback](../fallback/README.md) — read this one AFTER `boundary`, not
  before. Its `Fallback` combinator is the natural instinct for a fifth
  boundary strategy ("front theirs, degrade to the cassette when it's
  down") and it's a real, working pipe combinator — just not one that fits
  *this* slot. See "The `Pipe` vs `SendPipe` gotcha" below.

## The one concept: a `PipeHandle` slot IS the boundary

`PipeHandle` (`proxima::pipe::PipeHandle`, `proxima-primitives/src/pipe/handler.rs:83`)
is an erased, shareable, `Request<Bytes> -> Response<Bytes>` pipe — a type
alias for `Arc<dyn SendDynPipe<Request<Bytes>, Response<Bytes>>>`. Whatever
pipe you erase into that slot with `into_handle` **is** the boundary's
behaviour. There is no separate "boundary" type to configure — the slot and
the pipe algebra are the same thing.

A `BoundaryMode` config enum picks which BUILT-IN pipe fills the slot. Every
arm below is an EXISTING pipe from the library, unchanged, `Request<Bytes> ->
Response<Bytes>`:

| mode | pipe | what it proves |
|---|---|---|
| `Off` | the inner handle, untouched | zero-cost pass-through — you've cut over to `ours`, or are still fronting `theirs` with no boundary machinery at all |
| `Record` | `RecordUpstream` | tees every `(request, response)` to a cassette as it flows — capture `theirs` in production as the golden oracle |
| `Replay` | `ReplayUpstream` | serves the cassette back byte-identical, no upstream call — feed recorded traffic through `ours` in CI; an uncaptured request is a typed `ProximaError::ReplayMiss`, never a silent wrong answer |
| `Shadow` | `Diff` | fans out to both `theirs` and `ours`, reports divergence (`200` identical / `409` diverged) — run `ours` live against real traffic without trusting it yet |

Section 1–4 of `main.rs` drive all four arms through the same `wire_boundary`
match, one request each, and assert on what each mode actually returns.

## No `Observe` mode, on purpose

The four-arm table has a conspicuous gap: no `Observe`. That's deliberate.
Observation in proxima is `#[proxima::instrument]` — a function attribute,
orthogonal to the boundary, left on in every mode already. A boundary
"observe pipe" would just be `#[instrument]` wearing a costume: the same
telemetry, reachable a worse way, through a slot that has nothing to do with
observability. Naming a fifth arm for it would be adding a type the algebra
already has a better answer for.

## No `trait BoundaryStrategy`

There's also no trait to implement. A strategy **is** a
`SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>`,
full stop — `Handler` in proxima's vocabulary
(`proxima-primitives/src/pipe/handler.rs:73`), blanket-implemented for every
qualifying `SendPipe`. Minting a `trait BoundaryStrategy { fn handle(&self,
Request) -> Response; }` here would just be `SendPipe` under a new name —
the exact "blanket-impl-under-a-new-name" pattern this codebase's Rust rules
rule out. Extending the boundary is never "implement this trait"; it's
"write, or compose, a pipe."

## Two doors, neither one a new abstraction

**Door (a): add a named arm.** `BoundaryMode` is a closed menu — an enum has
to enumerate its values — but adding a strategy that deserves a name and a
`PROXIMA_BOUNDARY_MODE` value is one arm in the enum plus one arm in
`wire_boundary`'s match. Still zero new types: the arm's body is always
`into_handle(some_existing_pipe(...))`.

**Door (b): hand the slot a pipe directly.** For a strategy that ISN'T on
the menu — a one-off, a per-deployment tweak — don't touch the enum at all.
`PipeHandle` is just a slot; anything shaped like a `SendPipe` goes in it.
Section 5 proves this door by composing a fifth strategy — tagging every
response `theirs` serves with a migration-marker header, so an ops
dashboard can watch the strangler-fig progress live — entirely from
existing pieces, with no `BoundaryMode::Tagged` arm and no new struct:

```rust
let tagged = into_handle(
    Transform::new(theirs()).with_response_op(ResponseOp::SetHeader {
        name: "x-served-by".into(),
        value: "theirs (boundary canary)".into(),
    }),
);
```

`Transform` (`proxima::Transform`, backed by
`proxima-primitives/src/pipe/transform.rs`) is the request/response op
pipeline middleware — unrelated to the `examples/transform` lesson on the
pipe algebra's four forms, despite the shared name. It wraps any
`PipeHandle` and applies a list of `RequestOp`/`ResponseOp`s on the way in
and out. Wrapping `theirs()` in it and re-erasing with `into_handle` is
composition, not a new type — the same `into_handle` call every built-in arm
already makes.

## The `Pipe` vs `SendPipe` gotcha

The instinctive fifth strategy is different from the one above: "front
`theirs`, and when it's down, serve the last-known-good answer off the
cassette instead" — which reads exactly like
[`Fallback`](../fallback/README.md):

```rust
Fallback { primary: theirs(), secondary: replay }
```

This does **not** compile as a boundary strategy, and the reason is worth
understanding before you compose your own pipes here. Proxima's pipe
algebra has two relevant tiers (`proxima-primitives/src/pipe/primitives.rs`):

- `Pipe` — the root form. Borrow-shaped, `!Send`, no `'static` bound. Every
  pipe implements at least this.
- `SendPipe` — the additive cross-core form. `Send + Sync + 'static`, and
  its returned future is `Send` too, so it can be dispatched across cores
  and erased into a shareable handle.

`Fallback<P, S>` (`proxima-primitives/src/pipe/resilience/fallback.rs:17`)
implements only `Pipe`. There is **no blanket bridge** from `Pipe` to
`SendPipe` — the module doc explains why: writing `impl<P: Pipe + Send>
SendPipe for P` would require bounding `P::call`'s returned RPITIT future,
which needs return-type notation, an unstable language feature
(rust#109417). So every additive tier costs a full standalone trait, and a
combinator author has to opt a type into `SendPipe` explicitly — `Fallback`
never was.

`into_handle` (`proxima-primitives/src/pipe/handler.rs:86`), the only door
into a `PipeHandle`, demands `SendPipe` exactly:

```rust
pub fn into_handle<Implementor>(pipe: Implementor) -> PipeHandle
where
    Implementor: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError> + 'static,
```

So `Fallback { primary: theirs(), secondary: replay }` cannot fill this
slot — full stop, not a workaround-able limitation. Hand-rolling a one-off
`SendPipe` impl just to force `Fallback` through this example would BE the
"new type" this example spends five sections proving you don't need; that's
why section 5 reaches for `Transform` (which already implements `SendPipe`
over a `PipeHandle` inner) instead. `Fallback` is real, useful, and exactly
the right tool the moment you're composing pipes on the `!Send`, per-core
side of the algebra — see the [fallback example](../fallback/README.md) for
that. It's just not the tool for *this* slot.

## Set the boundary once; conflaguration turns it on/off

`BoundaryConfig` follows the house pattern
(`proxima/src/cassette_config.rs`): `#[derive(Builder, Deserialize,
Serialize, Settings)]`, resolved by `BoundaryConfig::from_env()` from
`PROXIMA_BOUNDARY_MODE` (or a config file conflaguration is pointed at).
Nothing in the four-arm match changes when you flip it — `main` reads the
config once, prints what it resolved to, and then still runs all five
sections in order so the example is deterministic to read. In a real
service, the config's `mode` is what feeds `wire_boundary` at startup
instead of a hardcoded `BoundaryMode` literal:

```sh
PROXIMA_BOUNDARY_MODE=shadow cargo run --example boundary
```

## Run

```sh
cargo run --example boundary
```

## What you'll see

```
boundary: one config-selected slot to integrate-then-replace a service

configured boundary mode (PROXIMA_BOUNDARY_MODE): Off

--- 1. Off: pass-through to ours, zero boundary machinery ---
  served: "served by ours"

--- 2. Record: serve theirs, tee the interaction to a cassette ---
  served (by theirs, the oracle): "served by theirs"
  cassette events: 5 (terminal captured: true)

--- 3. Replay: serve the cassette back byte-identical, theirs never called ---
  served (off disk): "served by theirs"
  unrecorded request correctly missed: GET /v1/never-recorded?

--- 4. Shadow: run ours against theirs, report where they disagree ---
  agree   -> status 200
  diverge -> status 409

--- 5. Extend: tag theirs' responses, no new type, no new enum arm ---
  theirs, tagged -> "served by theirs" (x-served-by: Some("theirs (boundary canary)"))

PASS: one PipeHandle slot, config-selected; every strategy an existing pipe.
```

- **Off** serves `"served by ours"` — the boundary added nothing, so what
  comes back is exactly what `ours` (a canned `SynthUpstream`) returns.
- **Record** serves `"served by theirs"` (the oracle, not `ours`) and, after
  `terminal.drained().await` — a completion signal, not a poll loop — the
  cassette holds 5 events: `Started`, `RequestEnded` (the request body is
  small enough to be buffered, not chunked, so no `RequestChunk` appears),
  `ResponseStarted`, one `ResponseChunk` carrying `"served by theirs"`, and
  the terminal `Ended` — confirmed captured.
- **Replay** serves the same `"served by theirs"` bytes straight off disk —
  no call to `theirs` happens this time, proven by an unrelated request
  (`GET /v1/never-recorded`, never in the cassette) resolving to a typed
  `ProximaError::ReplayMiss` instead of a wrong-but-200 answer. The match key
  is method + path + sorted query only by default
  (`proxima-recording/src/replay/keying.rs`), so the identical `chat_request()`
  used in Record matches its own recording exactly.
- **Shadow** returns `200` when `theirs` and `ours` agree byte-for-byte, and
  `409` (with a JSON divergence report as the body) the moment they don't —
  proven by running the pipe twice with different canned answers.
- **Extend** shows the `x-served-by` header landing on the response even
  though no `BoundaryMode` arm or new type was written for it — `Transform`
  composed around `theirs()` and dropped straight into the same
  `into_handle` boundary the first four sections use.

## In algebra terms

- The boundary is not a type — it's a `PipeHandle` slot. Filling it is the
  entire integration surface.
- Every built-in strategy is `Request<Bytes> -> Response<Bytes>` unchanged;
  none of them touch the shape, they only vary the behaviour behind it.
  Observation is orthogonal (`#[proxima::instrument]`), not a fifth arm.
  There is no `trait BoundaryStrategy` because `SendPipe` already is one.
- Extension has exactly two doors — a named enum arm, or composing a pipe
  straight into the slot — and neither one is "add a type."
- `Pipe` (root, `!Send`) and `SendPipe` (additive, cross-core, erasable) are
  different tiers with no blanket bridge between them (RTN is unstable);
  `into_handle` only accepts `SendPipe`. Not every real, useful combinator
  in the algebra fits every slot — knowing which tier a combinator lives on
  is part of composing correctly, not a limitation to route around.
