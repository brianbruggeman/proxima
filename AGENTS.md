# proxima agent instructions

Binding rules for any agent — Claude Code, Codex, or otherwise — working in this
repository. Read this file first. It is the entry point; the detail lives in
`.claude/skills/`, which every reader can open directly.

## Where the rules live

| File | What it holds |
|---|---|
| `.claude/skills/guiding-principles/SKILL.md` | The 21 workspace principles + the proxima-quic / proxima-h3 overlay axioms. **Binding, not advisory.** |
| `.claude/skills/model-calibration/SKILL.md` | What to do differently per model generation (Opus 5 vs 4.8 and earlier): verification, narration, delegation, scope, cost lever. |
| `.claude/skills/load-proxima/SKILL.md` | The pipe algebra bootstrap — the `Pipe` trait and where every other answer lives. |
| `.claude/skills/proxima-log/SKILL.md` | proxima's own telemetry: `#[proxima::instrument]`, the log macros, the metric instruments. Not the `tracing` crate. |
| `.claude/skills/conflag/SKILL.md` | `conflaguration` config across the tier matrix — the `Settings`+`Builder` house pattern and the build-time `sized` constants. |
| `.claude/skills/disciplined-component/SKILL.md` | Bench-driven greenfield primitives: default-off flag, baselines, discipline log. |
| `.claude/skills/bench-metrics/SKILL.md` | The signals a benchmark must capture. Throughput alone is not a benchmark. |
| `.claude/agents/` | The specialist workers (architect, critic, judge, debugger, integrator, migrator, security, test-writer, bencher, concentrator, teacher). |

Claude Code loads `.claude/agents/` and `.claude/skills/` automatically. Other
harnesses should read the files above by relative path.

## Non-negotiable

- **Do not guess.** When a signature or a line matters, read it and cite
  `file:line`. An inferred claim is wrong regardless of how confident it reads
  (principle 6).
- **You are not done until you can validate that the code works.** Build it, run
  the tests, show the output.
- **A causal or quantitative claim is unbacked until a measurement artifact is
  in the same breath** (principle 18). No bench number, no perf claim.
- **Do the correct thing** — no `TODO`, no `#[ignore]`, no "defer to v2" in place
  of the right shape (principle 15). A prerequisite that surfaces is part of the
  same body of work.
- **The incumbent's behaviour is the oracle** for any behaviour-preserving change
  (principle 14).
- **A subagent's report is not a read.** Before stating a signature, a trait
  relationship, or the absence of something as fact, open the file yourself and
  cite `file:line`. A report that reads complete is the failure mode — it drops
  exactly the detail that would have changed the claim.
- **A bounded fix you can name is a fix you do.** If you can state it in a
  sentence and it is under ~50 lines, do it in the same turn. "Flagging, not
  building" and "want me to?" are for work that is large, contested, or
  destructive. Surfacing a defect and leaving it is not discipline.

## Before adding a type

Answer one binary question: **can I express this with a pipe?** If yes, you do
not get a type. Not a threshold, not a judgment call, not "it's only 30 lines" —
a threshold is something you can argue past, and that is exactly how these land.
Answer it by *writing the pipe*, not by reasoning about whether it is worth it.

Second binary question, for plumbing that honestly answers "no" to the first —
erasure wrappers, coercion hosts, newtypes that exist to carry an impl:
**what can a caller do that they could not do before?** If the answer is nothing,
it is not a type, it is a relocation — delete it. Answer it by writing the call
site both ways. If the two lines are identical, you have your answer.

A plausible justification for a new type is the danger, not the safety. When you
find yourself writing a paragraph defending a type's existence, that paragraph is
the finding.

`examples/` is exempt — an example implementing a form locally is what examples
are for. **Nothing else is.** `tools/` is not exempt: a load generator, a CLI, a
bench harness is a *consumer* of the algebra, and "it's not the library" is the
exact sentence that mints a redundant type. If a consumer needs a type the
algebra cannot express, that is a finding about the algebra — report it.

**No blanket impls.** `impl<T: Bound> OurTrait for T` adapts an open set of
foreign types invisibly. If you cannot delete one without minting a type to host
the impl instead, stop — that newtype is the blanket impl under a new name, and
the defect is the trait that needed one. Report it; do not compensate for it.

## Source rules

- edition 2024, never 2021. clippy pedantic, deny warnings.
- never edit `Cargo.toml` dependencies by hand — `cargo add`. Workspace, profile,
  feature, and metadata edits are fine.
- no `unwrap()` outside tests and derive macros; no `panic!` / `todo!` /
  `unimplemented!` in production paths.
- no `allow(...)`. If genuinely unavoidable, a one-line comment states why.
- no `ref` if it can be avoided. No `cfg(debug_assertions)` scaffolding in source.
- all imports at module top; never inline `std::path::PathBuf` in a body; no `use`
  inside a function.
- no variable names of 2 characters or less (`_` for discards is fine).
- `&str` over `String` in parameters; references over ownership unless ownership
  genuinely makes more sense.
- `thiserror` for libraries and internals, `anyhow` only at the CLI edge.
- `#[must_use]` on functions returning `Result`. Derive `Debug`, `Clone`,
  `PartialEq` where appropriate.
- `Result`/`Option` chaining with `?`; iterators over explicit loops; pattern
  matching over if-else chains. Encode invariants in the type system.
- **box-free by default**, whole workspace: no `Box<dyn Trait>`, `Box<dyn Future>`,
  `Box::pin`, or `#[async_trait]`. Reach for a discriminated enum + match,
  typestate, generic params, or a state-machine future. A legitimate `Box` (open
  dyn set, recursion, a *measured* large enum variant) carries a one-line why.
- **RPITIT for trait-async**: `fn f(&self) -> impl Future<Output = ..>` or
  `async fn` in trait. Never `#[async_trait]`, never `Pin<Box<dyn Future>>`.
  `poll_*(&self, cx) -> Poll<..>` is the other box-free surface, preferred for
  reactor-driven and cancellable code.
- 1:1 file to struct + impl where practical; trait impls and small related types
  may colocate when semantically coherent.

## Comments

Never, with rare exceptions. A comment must jog a reader out of reading mode to
explain something esoteric, novel, or a gotcha. Lowercase, terse, and it states
**why** — never what. Docstrings on struct fields (clap, config) where they earn
it; library field docs carry semantic meaning and an example of real data; config
fields with defaults carry semantic meaning only. Module and function docs very
sparingly, public interfaces only, and only when name + signature aren't obvious.

No emojis, anywhere.

## Hot-path requirements

For query, scoring, codec, and inner-loop traversal paths — choose the strictest
tier that actually applies; a benchmark goal is not a semantic invariant.

- 500MB RAM cap; ≥55MB/s sustained; <1ms query latency.
- zero-copy, and **no heap allocation** on query / scoring / inner-loop paths.
  Use mmap, LSM, or prebuilt storage.
- concurrent read paths lock-free.
- **lock discipline**: lock-free first. If serialization is unavoidable in async
  code, yield via an async gate (waker-based async lock or single-consumer pipe)
  rather than parking a thread, so it works at every tier including bare metal.
  A synchronous blocking mutex is last resort, outside async only (sync FFI
  boundary, dedicated blocking worker), and must be `proxima_lock::Mutex` — never
  a bare `parking_lot::Mutex` or `std::sync::Mutex`. Spinlocks are ruled out. A
  std-gated mutex must be an optional, std-gated dependency so no_std builds
  never compile it.
- async streaming for ingestion and payload flow.
- no O(n²) in query, scoring, or traversal — inverted index, bloom filter, or
  sparse matrix instead.
- treat zero-cost / zero-copy / zero-alloc / lock-free / O(1) claims as **unproven**
  until instrumentation or a bench shows them.

## Telemetry

proxima's own surface, never `tracing` directly:

- `proxima::telemetry::{trace!, debug!, info!, warn!, error!}` — not `tracing::*`
- `#[proxima::instrument]` — not `#[tracing::instrument]`; one attribute yields
  trace + metric + log
- export is never OTLP-only: wire a console sink (`Exporter::stdout()` /
  `stderr()` / `StdSplit`) and a file sink (`Exporter::file(path)`) alongside any
  OTLP exporter
- never hand-roll an env-gated file dump for forensics — emit a structured event
  and point a file-sink `Exporter` at it

Levels: `trace` noisy inner-loop, assume disabled. `debug` dev narrative — state
transitions, decision points, non-critical validation failures. `info` sparse and
business-meaningful — major workflow transitions only. `warn` degraded but
self-healing. `error` contract violations and user-visible failures. **A
production incident must be explainable from `info` and `error` alone.**

Messages are lowercase, terse, causal — they say *why*; the fields carry *what*.

## Testing

- `cargo nextest run` is the default runner. Fall back to `cargo test` only for
  doctests — **nextest does not run them** — or when nextest is unavailable.
- **The landing gate is not just nextest.** `cargo check --all-targets` compiles
  examples but never *runs* them, and nextest never runs doctests. Both classes
  have shipped broken to public main. A landing gate must add `cargo test --doc`
  with the right features, and actually run the examples.
- **`cargo test --doc` exits 0 when it matched nothing.** Zero-passed is
  indistinguishable from all-passed by exit code alone, so a gate that only
  checks the exit status proves nothing. **Assert a nonzero test count.** A
  ` ```ignore ` fence is how a doctest that does not even compile hides from
  this.
- `#[proxima::test]` instead of `#[rstest]` / `#[tokio::test]` / bare `#[test]`
  for async or parameterized tests — it subsumes rstest cases plus cassette
  record/replay and drives the body on prime, with a tokio fallback. Plain sync
  unit tests with no cases may stay `#[test]`.
- `case::terse_semantically_unique_desc(...)` for parameterized cases.
- arrange / act / assert. Happy and sad paths both.
- `expect()` with a message explaining the failure, over bare `unwrap()`.
- **no sleeps in any test, ever.** Tests are deterministic.
- unit tests inside the source file (`#[cfg(test)] mod tests`) over separate
  integration files; integration tests only for genuinely cross-module behaviour.
- filesystem tests use `tempfile`/`tempdir` and create their directories in setup.
  `temp_env::with_vars` to scope env mutations.
- real-world data, not `b"AAAA"` stubs — except in negative "rejects garbage"
  tests (principle 9). Tests are documentation (principle 17): the name states
  the contract, the body reads as a usage example.
- prefer faking and stubbing over heavy mock frameworks.

## Build

Prefer debug builds for speed. Release only for CLI testing or timing.
