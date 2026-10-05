# fsm techniques: task cards

TASKS.md lists slices. Each slice is cut into cards in `tasks/NN-<slice>.md`. A card is the
unit an executor agent runs.

## executor

Every card is sized and written for a low-cost executor: "Luna-class", meaning a GPT-5.6 Luna
or a Claude Haiku/Sonnet agent.

The executor has:
- the repo;
- this file;
- SPEC.md;
- the one card.

It has no other context, and it makes no design decision.

If a card needs a choice, the choice is already written in the card:
- type names, function signatures and field names;
- which file a function goes in;
- test names and the exact assertions;
- the expected test count.

An executor that finds the card's premise false stops and reports. It does not improvise.
Examples of a false premise: a named symbol is absent, a count differs, or an anchor moved
semantically.

## size rule (each card about 20 minutes of executor wall time, build and test included)

A card fails the size rule, and must be split, if any of these holds:
- it touches more than 3 source files, not counting `Cargo.toml` feature lines and the test
  file;
- it adds more than about 120 lines of non-test code;
- it adds more than one public item that another card consumes;
- its validation needs more than one GPU or model-loading run;
- it contains the words "and also", or two unrelated verbs.

A worked-example card (hand derivation into `worked-examples.md`) holds one example.

Admitted exceptions (2026-10-04). Each card is over the file count, but its extra edits are
mechanical one-line arms that cannot be split without breaking green:
- FT3.4 and FT3.6: a new cfg-gated `BoundOpKind` variant forces one copied arm in every
  exhaustive match.
- FT2.20: moving the decode.rs guard into `apply_serving_config` touches 4 files; splitting it
  would leave the knob silently ignored.
- FT2.1 and FT2.2: a new `ServingConfig` field must reach every exhaustive destructure (the
  prompt-cache key and the resident-plan identity) in the same commit. 3 of the 5 files are
  one-line edits.
- FT9.12: prefilling a loaded chunk through the blend program spans 5 files. Its host
  leaf-row functions have no caller outside the decode hook, so any split leaves dead code.
- FT4.1: replacing `truncate` with the fallible `try_truncate` changes one signature, which
  forces both of its callers in the same commit (5 files).
- FT5.9: one new file, one struct, three methods; it cannot be split without leaving an
  unused type.
- FT7.3: a new `ServingConfig.decode` field must reach the settings literal and both exhaustive
  destructures (the prompt-cache key and the resident-plan identity) in the same commit, as for
  FT2.1 and FT2.2; the other three edits are one line each.
No other card is exempt.

## green rule

Every card ends green, so it is a commit point:
- its own test command passes at the stated count;
- `cargo clippy -p <crate> --features <card's features> --all-targets` is clean;
- `cargo check -p proxima-core --no-default-features --features alloc` passes whenever the card
  touches proxima-core.

Owner, 2026-10-04: "every card should be coherent commitable." A card is exactly one commit,
and that commit is coherent on its own:
- one logical change, nameable in one conventional-commit subject: `feat:`, `fix:`,
  `refactor:`, `docs:`, `test:`, `chore:` or `perf:`; lowercase, imperative, no trailing period,
  under 72 characters;
- it carries its own test and the reason it exists, and reads sensibly in `git log` without
  the card;
- it leaves no dead code. Every new item is used or public and is exercised by a test in the
  same card. A card that only adds an item another card will use is not coherent: fold the
  first use into it, or make the test its use.
  - "Make the test its use" works only for items reachable from the crate's public API.
  - A `pub(crate)`, `pub(super)` or private item used only under `#[cfg(test)]` still trips
    `dead_code` in the non-test clippy build, and that build denies warnings.
  - So such an item needs its first production caller in the same card;
- it is revertable alone. Reverting it never breaks an earlier card's tests;
- the card lists exactly what it stages, and `git diff --cached --stat` must equal that list.

## English only in anything that is committed

Owner, 2026-10-04: "never reference stage or ac or phase etc. those will be gone the moment we
commit and start working in another session. we must use english."

The rule covers everything a card produces:
- commit messages;
- code, doc comments and test names;
- error messages and log or example output;
- `worked-examples.md`;
- fixture READMEs.

None of these may contain:
- a slice, stage or phase number;
- an AC, R, W, FT, D or O id;
- a card id;
- "spec item N", or any other pointer into these spec files.

Describe the behaviour in English instead:
- a test is `rewind_into_a_sealed_block_is_refused`, not `ac3_rewind`;
- a doc comment says "the block size every tier shares", not "R4's block";
- a worked example is titled by what it computes ("top-fraction selection over 12 scores"),
  never "W7".

The cards themselves may use ids, because that is where the index lives. Whatever a card
tells the executor to write must be in English.

A card never leaves a broken intermediate state for a later card to repair. When a refactor
must span two cards, the first card adds the new form beside the old, and the second removes
the old form.

## workspace-wide rules every card inherits (2026-10-04)

These hold for every card. A card does not restate them, and an auditor does not refuse a
card for leaving them implicit.

- **Where the spec lives.** Before the first card executes, this directory is committed to main
  at `proxima-tensor/specs/fsm-techniques/`. That covers SPEC.md, TASKS.md, CARDS.md,
  `tasks-recut/` and `worked-examples.md`.
  - Every card path is relative to the root of the checkout holding main
    (`/Users/brianbruggeman/repos/slot-0/proxima-windows`).
  - A card that appends to `worked-examples.md` stages
    `proxima-tensor/specs/fsm-techniques/worked-examples.md`.
- **Test modules and clippy.** The workspace denies `clippy::unwrap_used` and
  `clippy::expect_used`. Every new `#[cfg(test)]` module or test file carries
  `#[allow(clippy::unwrap_used, clippy::expect_used)]`, as existing test modules do
  (`serving.rs`, `prompt_cache_settings.rs`, `rope_scaling.rs`).
- **No pointers into plans.** Committed comments and docs never say "this step", "a later
  step", "this plan" or "this card". When a card touches a file whose existing comments say
  that, it rewrites those comments in plain English in the same commit.
- **No intra-doc link to an item that does not exist yet at that commit.**
- **Test models.** gemma4 E2B is the dense model. gemma4 26B and granite3.1-moe 1B are the
  MoE models. No qwen of any kind.
- **Oracles are recorded once and replayed.** The gemma4 26B parity fixture on main uses
  chat-templated prompts where llama is confident (main a7c08c4c), so the 26B has a live
  llama oracle.
- **The size rule counts public types.** A new method on an existing type that a later card
  calls is allowed alongside one new public type.

## anchors

- Line numbers drift: other sessions edit the same files. A card names every location as
  `path::symbol` (function, type, field or test name), with the line only as a hint:
  `path::symbol (~line N at <sha>)`.
- The executor re-locates by symbol before it edits.
- A card states the commit its anchors were read at.

## card template

```
### <slice>.<n> <imperative title>

- id: FT<slice>.<n>
- needs: FT<a>.<b>, ... (cards that must be done first; "none" if none)
- budget: 20 min
- crate(s): <crate> (features: <list>)
- read first: <path::symbol>, ... (at most 4 anchors, each with a one-line reason)
- change:
  1. <file>: <exact edit: names, signatures, body intent in one or two lines>
  2. ...
- test: add `<test_name>` in `<path>` asserting <exact assertion, with the worked value or
  the oracle file>.
- validate: `<one command>`
- expect: `<N> passed` (and, where relevant, the named tests that must appear)
- also green: clippy line, plus the no_std alloc check if proxima-core is touched
- stage: <exact list of paths to `git add`>
- commit: `<type>(<scope>): <subject>` (the exact message; no body unless a non-obvious
  why needs one)
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the
  stage list, and the commit landed with that message
- do not: <files and areas that are out of bounds for this card>
- gpu: none | one run, waiting for a quiet box (the peer-gate check in SPEC "machine
  safety")
```

## machine safety (applies to every card)

- Model-loading tests run with `-j 1` only, and one model-loading process at a time.
- Before any Metal or model run:
  - check `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"`,
    and wait while a match exists;
  - this machine has kernel-panicked twice from concurrent model loads.
- Each card uses its own `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_<card>`, and removes it
  when done.
- Logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/<card>/`, never
  under /tmp.
- Never run rustfmt or `cargo fmt` on a src file (they follow `mod` declarations). A new
  standalone test file may be formatted by path.
- Never `git checkout`/`stash`/`reset`/`worktree`. Every card ends in exactly one commit. It stages only its `stage:` list, uses its `commit:`
  message, and carries no attribution trailer.

## the end state: sans-IO correctness

The last slice (17) is a sans-IO conformance suite. It runs every FSM transition, under every
place's config-selected pipe, against a deterministic scripted backend, with:
- no GPU;
- no file or network IO;
- no model weights.

The checks are on traces and properties. When it passes:
- every technique's control flow is proven correct independently of kernels and models;
- the oracle ACs (llama parity) prove the numerics separately.
