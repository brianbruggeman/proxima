# Where the algebraic+marker Pipe pattern might apply

> **Crate consolidation / path note (2026-07).** Several crate names in
> this survey are stale. `proxima-compose` and `proxima-recording-pipe`
> existed at write time and have since folded into `proxima-primitives`
> and `proxima-recording` respectively. `proxima-process-protocol`'s
> markers moved to `proxima-core::markers`. `proxima-vm` and
> `proxima-middleware` (candidates 3 and 4 below) were never built —
> this survey's "lowest-friction adopter" framing for `proxima-vm` did
> not turn into shipped work. The `Pipe` trait itself also changed shape
> since this survey — see the status note in
> [design-sketch.md](./design-sketch.md) for the current
> `proxima-primitives::pipe::Pipe` (associated types, no `Send` bound).
> Treat this whole document as a historical motivation snapshot, not a
> current adoption roadmap.

Status: **DRAFT** — single-pass survey written 2026-05-24 under
`unattended`. This is **Phase D** of
[~/.claude/plans/we-want-the-dependency-drifting-bear.md](../../../../.claude/plans/we-want-the-dependency-drifting-bear.md).

This is a **light survey** — six candidates with one-line rationale +
load-bearing marker. **Motivation, not commitment.** Full per-crate
analysis happens in each candidate's own follow-on plan, once the
algebra (Phase C / next plan) is shaped against a real API. Surveying
in detail before the API exists would anchor design to current shapes
instead of vice versa — exactly what the disciplined-component skill
warns against.

The pattern reference is
[proxima-process](../../proxima-process) (worked example;
ground+operator+marker algebra) and
[docs/algebraic-pipe/design-sketch.md](./design-sketch.md) (the
draft API shape Phase C.3 sketches).

---

## Candidates

### 1. `proxima-recording`

**Why the pattern might fit.** Recording sinks compose naturally:
`Tee<RecordA, RecordB>` for dual-destination, `Series<Filter, Sink>`
for capture-then-write, branched fan-out for replay-record splits.
Today, [proxima-recording-pipe](../../proxima-recording-pipe) wraps
pipes for recording but the composition is implicit — `LiveCaptureContext`
attaches via the existing `proxima-pipe::Pipe::call(Request)` path.
With typed operators, the recording topology becomes type-checked.

**Load-bearing markers.**

- `WithoutNetwork` — proves a recording chain is fs-only (the common
  case: write JSONL/binary records to disk, no remote sink). A wire
  protocol assertion: "this recording can't be exfiltrating data."
- `Deterministic` — required for any sink in a `Causal` replay chain.
  Today it's a runtime invariant; with markers it's a bound.

**Shape sketch.** `Series<Capture, BufferedRingSink>`, `Tee<FsSink, NoopSink>`,
where `Capture: Pipe<Request, RecordedEvent>` and sinks are
`Pipe<RecordedEvent, ()>`. Marker propagation: `Series` is `Deterministic`
iff both arms are.

**Out of scope for this plan.** Recording-core does file I/O via
`tokio::fs` — std-bound. The marker work applies to the typed wrappers
in `proxima-recording-pipe`, not the IO sink itself.

---

### 2. `proxima-control-plane`

**Why the pattern might fit.** Control verbs are sequenced:
`Series<RegisterPipe, StartListener>` or `Series<LoadConfig,
ApplyPipeline, StartServe>`. Each verb is a step in a larger orchestration
state machine that today lives as imperative code in
[proxima-control-plane](../../proxima-control-plane). With typed
operators, the orchestration becomes algebraically describable —
which gives rollback semantics for free (each verb's `Reproducible`
marker tells the rollback layer whether to retry vs. cleanup-then-retry).

**Load-bearing markers.**

- `Reproducible` — for verbs in a replay-able control flow. A
  `Reproducible` verb can be re-run after a crash and the system
  converges to the same state.
- `IdempotentSideEffectFree` — for retry-safe verbs (idempotent
  register, idempotent start-if-not-running). The retry layer can be
  trait-bound to require this.

**Shape sketch.** A control-plane operation is `Pipe<VerbInput,
VerbOutput>`. `Series<RegisterPipe, StartListener>` describes the
sequence. Markers gate which verbs can appear in which positions —
e.g. a verb in a "must be retry-safe" slot has the
`IdempotentSideEffectFree` bound.

**Out of scope for this plan.** Control-plane state machine is std-bound;
this is a typed-API refinement, not a portability move.

---

### 3. `proxima-vm` (lowest-friction adopter)

**Why the pattern might fit.** Proxima-vm **already uses the
proxima-process markers** for contained discovery — a recent design note
records `VmDispatchHandler` mirroring
`GenericPipe<In = ChildRequest, Out = ChildResponse>`. Extending the
same algebra to the VM's discovery surface unifies the contained-
discovery type system with the process-shim one. End-to-end, a
"this VM cannot escape its sandbox" claim is a single compile-time
bound that traces from the shim through the VM through every
discovery hook.

**Load-bearing markers.**

- `WithoutFilesystem`, `WithoutNetwork`, `WithoutSpawn` — the three
  security primitives. The proxima-vm boundary's value proposition
  IS these markers. Without them, contained discovery has to be
  audited by hand.

**Shape sketch.** The dispatch handler is already a `GenericPipe`-shape;
the VM-internal discovery hooks (filesystem grant resolver, network
grant resolver, etc.) become typed operators. A
`Series<UntrustedGuest, MarkerErasureBoundary, TrustedHostHandler>`
captures the trust transition explicitly.

**Why it's the lowest-friction adopter.** The markers are already there.
The grounds (in proxima-process) are already typed. Lifting the algebra
into proxima-vm is mechanical, not architectural.

---

### 4. `proxima-middleware`

**Why the pattern might fit.** Auth, RateLimit, Retry, Transform, ...
middleware composes. Today it's a chain via
[proxima-compose](../../proxima-compose)'s
`Arc<dyn DynPipe>`-based machinery. With typed operators:
`Series<Auth, Series<RateLimit, Series<Retry, Inner>>>`. The marker
propagation gives compile-time guarantees about the chain — e.g.
"this middleware stack contains a rate limiter" becomes a trait bound.

**Load-bearing markers.**

- `IsPure` — for pure transforms (e.g. payload rewrites). A transform
  middleware that isn't `IsPure` may be touching network/fs and the
  chain caller can refuse it.
- `WithoutNetwork` — for in-process rate-limiting (vs. distributed
  rate-limiters that need network). Lets the caller assert "this
  middleware runs without remote coordination."

**Shape sketch.** Each middleware is `Pipe<In, Out>` (often `In = Out =
Request → Response` or some transform of that). Operators compose
them; markers stratify them.

**Out of scope for this plan.** Existing `proxima-compose` operators
stay as the runtime-poly path; algebraic operators are the compile-
time-typed path. Both live side by side (see design sketch §3.3).

---

### 5. consumer-crate 5-path retrieval

**Why the pattern might fit.** A downstream consumer's retrieval pipeline
already composes algebraically — five paths
combine via `min(bmx, chain, mini, ngram, tower)` (per that consumer's own notes). That's
already a `Quorum`-shaped operator hand-written. With typed `Pipe<In,
Out>`, the composition becomes `Quorum5<Bmx, Chain, Mini, NGram, Tower>`
and the marker propagation enforces "no LLM in this path"
(`IsPure`) at compile time.

**Load-bearing markers.**

- `IsPure` — the entire consumer-crate 5-path is graph-structure-only, no LLM.
  Marker enforces this; if someone adds an LLM-dependent path, the
  bound breaks and the build fails. The "no LLM" claim becomes
  compile-time-checkable instead of code-review-checkable.
- `Deterministic` — required for the BMX-determinism work (that consumer's
  notes: "HashMap drain order caused tied-score docs to swap ranks between
  runs" — fixed). The marker locks the property going forward.

**Shape sketch.** Each retrieval path is `Pipe<Query, RankedResults>`.
A `Quorum5<...>` operator takes 5 paths and emits the min-rank fusion.
The consumer crate gets typed operators replacing the
hand-coded min().

**Out of scope for this plan.** The consumer is a separate workspace; the
algebra crate (or proxima-pipe) becomes a workspace dep there. The
marker promotion (proxima-core, Phase C.1) makes the markers
accessible without dragging proxima-process-protocol into that consumer.

---

### 6. `proxima-runtime` selection

**Why the pattern might fit.** Runtime backend choice IS an algebra —
prime vs. tokio. Today the choice is via Cargo features
(`proxima-runtime-prime` vs. `proxima-runtime-tokio`). With markers,
"this code path is alloc-free, must select prime" becomes a compile-
time bound at the *use* site, not just at the *crate-feature* level.

**Load-bearing markers.**

- `AllocFree` — the alloc-free path can't pick tokio (tokio requires
  alloc). Marker on the runtime impl tells the bound at the consumer.
- `NoStd` — bare-metal target must pick prime.

**Shape sketch.** The `proxima-runtime::Runtime` trait gains marker
impls on the typed `RuntimeKind<PrimeBackend>` vs.
`RuntimeKind<TokioBackend>`. Consumers `where R: Runtime + AllocFree`
get a compile error if they configured tokio.

**Out of scope for this plan.** The runtime trait is already on the
cliff; this is a typed-API refinement that benefits from marker
promotion (Phase C.1) being done first. The work is small but spans the
runtime-prime + runtime-tokio adapters.

---

## Summary table

| # | Candidate | Load-bearing marker(s) | Friction |
|---|---|---|---|
| 1 | proxima-recording | `WithoutNetwork`, `Deterministic` | medium — needs typed wrapper refactor |
| 2 | proxima-control-plane | `Reproducible`, `IdempotentSideEffectFree` | medium — orchestration refactor |
| 3 | **proxima-vm** | `WithoutFilesystem`, `WithoutNetwork`, `WithoutSpawn` | **low — already uses markers** |
| 4 | proxima-middleware | `IsPure`, `WithoutNetwork` | medium — coexists with `proxima-compose` |
| 5 | consumer-crate 5-path retrieval | `IsPure`, `Deterministic` | low — algebra already exists as hand-coded `min()` |
| 6 | proxima-runtime selection | `AllocFree`, `NoStd` | low — adapter-level work |

**Lowest-friction first.** If this survey turns into action plans,
proxima-vm and the consumer crate are the obvious early adopters because the algebra
is already there in spirit; the work is making it type-checked.

---

## What this survey does NOT do

- It doesn't commit to porting any of these.
- It doesn't enumerate every workspace crate that *might* benefit —
  only the six the design-sketch level of confidence justifies.
- It doesn't specify which marker propagation rules each operator
  needs (that's the per-candidate plan's job, after the algebra is
  shaped).
- It doesn't bench any current vs. proposed implementation (premature
  — needs the algebra to exist first).

## What this survey IS for

Two things:

1. **Evidence the algebra generalizes.** If the pattern fits ≥3
   distinct workspace areas, it earns its place as a workspace
   primitive (proxima-pipe extension or proxima-algebra crate).
   Six candidates clears that bar.
2. **A starting list for follow-on plans.** When the algebra lands
   and the next pass of work asks "what do we port first?", this
   document is the entry point — pick the lowest-friction adopter
   (proxima-vm or the consumer crate) and write a focused porting plan from there.

---

## Consumer-repo adopters

Everything above stops at proxima's own crate boundary — candidates
1-6 all live inside this workspace. That's a survey gap, not a claim
that the pattern is proxima-internal only: candidate 5 (consumer-crate 5-path
retrieval) already gestures across the boundary, and a follow-on audit
of that consumer repo (2026-07-11) found several more hand-rolled dataflows that
match a `proxima-pipe` combinator closely enough to be adoption
candidates in their own right.

The full audit — table of target, current hand-roll, combinator, and
file:line — lives in the consumer repo, not here (per the
`ai_docs/README.md` convention: architecture docs for the consumer's code live
in that consumer repo, not in proxima's docs tree).

**Motivation, not commitment** — same caveat as the six candidates
above. The three highest-value adopters from that audit:

1. **Consumer-crate retrieval scorers → `ScatterGather`.** Its
   `score_bmx_with_chunks`/`search_bmx` and specificity scorer both
   hand-roll the same per-token scatter→gather→merge→rank shape
   (the consumer crate's `bmx.rs:103`, `:341`); `ScatterGather` is that
   shape already typed, and one adoption covers two near-duplicate
   call sites.
2. **Consumer-crate `LocalEventBus` → `FanOut`/`KeyedLiveFilter`.** The
   pub/sub broadcast + recv/publish worker loops
   (the consumer crate's `async_runtime.rs:35`) are a hand-rolled
   `broadcast::Sender` fan-out — exactly what `FanOut` +
   `KeyedLiveFilter` typed-compose, with marker propagation replacing
   the manual channel plumbing.
3. **Consumer-crate stage-nests → `forms::Pipe` `Series`.** Three separate
   files (`pipeline.rs:61`, `stream.rs:60`, `async_pipeline.rs:40`)
   independently hand-roll the same Option-tuple stage-nesting match —
   the strongest duplication signal in that audit, and a single
   `Series` adoption collapses all three.

## Pointers

- Pattern reference: [design-sketch.md](./design-sketch.md)
- Marker source: [proxima-process-protocol/src/markers.rs](../../proxima-process-protocol/src/markers.rs)
- Operator pattern: [proxima-process/src/operators.rs:51-95](../../proxima-process/src/operators.rs#L51-L95)
- Guiding principles: `/guiding-principles` skill
