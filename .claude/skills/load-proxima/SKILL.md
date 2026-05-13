---
name: load-proxima
description: Point an agent at proxima's source of truth — the `Pipe` trait, and where every other answer lives. Holds no concept but the pipe; everything else is a pointer. Use when you (or an agent) are about to build, change, or explain something in the proxima workspace, or before hand-rolling any dataflow (loop/channel) in it. Triggers on "load proxima", "/load-proxima", "teach me proxima", "how do proxima pipes work", "get up to speed on proxima".
---

# load-proxima

This skill is a **map, not territory**. It holds one concept — the pipe — and pointers to where every other answer lives. Read the pointer for whatever you will actually touch.

**Source is the only oracle.** Do not teach or assert an API shape you have not read (guiding-principles principle 6). If a doc and the code disagree, the code is right and the doc is the bug.

**Never add a summary to this file.** Not the algebra, not the tiers, not the runtime, not the combinator vocabulary — however small, however true today. A summary is a second copy; the copy goes stale the moment the code moves, and it keeps reading as confidently as the day it was written. Add a pointer instead.

## The one idea

**Everything is a `Pipe`, and big things are small pipes composed.** A pipe is one async step: `In -> Result<Out, Err>`.

Read it at `proxima-primitives/src/pipe/primitives.rs` — the trait, and the composition method on it. `call` is RPITIT: never boxed, never `#[async_trait]`. Composition is a default method on the trait itself, so composing pipes is part of what a pipe *is*.

Before adding a type, prove no existing primitive could be extended (principle 1, RISC reuse-first).

That is the whole of what this skill asserts. For anything else, read the pointer.

## Where the answers live

| you need | read |
| --- | --- |
| the pipe, authoritative | `proxima-primitives/src/pipe/primitives.rs` |
| everything else in the pipe module | `proxima-primitives/src/pipe/` |
| which combinator for which job | `ai_docs/examples-index.jsonl` |
| agent bootstrap / task routing | `ai_docs/AGENT.md` → `ai_docs/index.jsonl` → `ai_docs/task-routes.jsonl` |
| learn it as a human, from zero | `docs/tutorials/README.md` → `docs/tutorials/00-foundations.md` |
| runnable, compile-tested code | `examples/hello/main.rs`; `examples/README.md` for the curriculum |
| the binding rules in full | `~/.claude/skills/guiding-principles/SKILL.md`, `~/.claude/rules/rust.md` |

**Don't hand-roll dataflow.** Before writing a loop or channel for filtering, fan-out, merging, backpressure, retry, or rate-limiting, check `ai_docs/examples-index.jsonl` — the combinator probably exists, typed.

## After loading

State, in one line, which primitives your task will compose (principle 2, teaching surface). If a specialized `proxima-*` agent fits better — architect for design, teacher for docs, debugger for hard bugs, security for crypto, integrator/migrator for cross-crate change, test-writer, bencher — hand off. Otherwise proceed, grounded in the files above, never in inference.
