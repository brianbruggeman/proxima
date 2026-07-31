---
name: unattended
description: When working on a plan or bug or task, the user needs to step away for a while and leave the assistant unattended. While working through, place a note in the journal to capture the state, context, questions, data, decisions and next steps, so that when they return, they can pick up where they left off without losing momentum. Captured evidence climbs the measurement→result→conclusion→decision ladder — a bare number without its proven deep why is a measurement, never a result, and is never promoted past what is understood. Pulls in `/disciplined-component`, `/guiding-principles`, and slot-0/AGENTS.md automatically; engages `/research-rigor`, `/algorithm-development`, `/algorithm-rigor`, `/discovery-loop`, `/security-review` per principle 13 as needed.
---

# unattended

The user is stepping away. Continue the work autonomously and capture
state continuously into the slot-0 Obsidian journal so they can resume
cold without losing momentum.

## What `/unattended` binds — every time

When the user invokes this skill, the following bind without further
prompting:

### Skills pulled in automatically

- `/guiding-principles` — workspace principles plus proxima-quic /
  proxima-h3 overlay axioms. Every decision tested against the
  relevant principles before execution. Two principles bind
  especially hard while unattended:
  - **Principle 14:** if you do not have absolute parity on output
    with the incumbent, you've probably screwed up correctness; the
    burden is on you to prove the incumbent is the one with the bug,
    beyond any shadow of a doubt.
  - **Principle 15:** do the correct thing, do not defer or punt. We
    are building foundation. No `TODO` / `FIXME` / `#[ignore]` /
    "deferred to v2" in lieu of the correct code. Equally forbidden:
    *principled-stopping* — invoking principles 13/14/15 to halt at
    "I named the prerequisite" when no external constraint blocks
    closure. Greenfield has zero legitimate-deferral surface; a
    discovered skill-spend (`/research-rigor` /
    `/algorithm-development` / `/algorithm-rigor` / `/discovery-loop` /
    `/security-review`) is the FIRST step
    of the same work, not a reason to stop. Run the skill, land the
    prerequisite, land the target — one body of work. The temptation
    to punt-while-sounding-principled to save time before the user
    returns is exactly what this rules out.
  - **Principle 16:** execution must not outrun proof. At the
    velocity unattended sessions invite, the discipline log fills
    faster than the substrate that PROVES each row's claim can keep
    up. A row marked DONE without CI / saved bench baselines /
    vendored parity vectors / snapshot harness re-verifying it on
    every commit is a hypothesis, not a contract. Before sealing any
    row in an unattended session, ask: can the CI re-prove this
    claim from scratch, right now, without my memory? If no, the
    proof substrate is the next step — not the next row.
- `/disciplined-component` — if the unattended work is a multi-
  component bench-driven build, the discipline log IS the unattended
  journal. Do NOT spin up a parallel journal note that duplicates
  state; extend the existing discipline log. The unattended Obsidian
  note becomes a thin pointer to the discipline log with the current
  row + next-step delta; the discipline log carries the substance.
- `/research-rigor` — invoke when a contested design decision shows
  up mid-work (multiple plausible answers + wrong answer is
  expensive). Resolution lands in `edges.md` and is cited from the
  journal.
- `/algorithm-development` — invoke when a non-mechanical algorithm
  needs landing (RFC pseudocode port, integer-arithmetic algorithm,
  novel state machine). Paper-derive the worked example FIRST, then
  implement. The worked example becomes a test.
- `/algorithm-rigor` — escalate from `/algorithm-development` when
  that algorithm is ALSO contested (multiple plausible formulations,
  wrong rule is expensive). Tournaments competing worked-example
  bundles; the winner's worked example becomes the locked test.
- `/discovery-loop` — invoke when the work has no oracle: a new
  capability / architecture / heuristic whose correct output is
  unknown (success is a held-out metric, not a target output).
  Pre-register the hypothesis + kill criterion; a stable win then
  engages `/algorithm-development` to lock it as a test.
- `/security-review` — invoke for crypto material, authentication
  code, key derivation, AEAD composition, anti-replay, address
  validation. Finding goes in the component's discipline-log row.

Use them when the work calls for them — principle 13 governs the
tagging. Don't ask the user mid-session; the green light is the
`/unattended` invocation itself.

### Project context references

- `slot-0/AGENTS.md` — hot-path invariants (500MB RAM, ≥55MB/s,
  sub-1ms p99, zero-copy inner loop, no heap alloc inner loop,
  lock-free reads, no O(N²) on query paths). A component that breaks
  these does not land regardless of how clean the micro-bench looks.
- When the work touches QUIC or H3, the overlay axioms from
  `/guiding-principles` (sections "proxima-quic axioms" and
  "proxima-h3 axioms") apply on top of the workspace principles.

### Git discipline (auto-applied to every commit)

- **Conventional commits.** `feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `chore:`, `perf:`, `ci:`. Imperative, lowercase, no period,
  under 72 chars.
- **No co-author trailer.** Never add `Co-Authored-By: Claude` or any
  cosign. The user's git config is the source of truth for authorship.
- **No `--no-gpg-sign` / `--no-verify` unless explicitly requested.**
  If a hook blocks, investigate; if 1Password blocks signing on a
  specific machine, the user has flagged that case and you may use
  `--no-gpg-sign` — but only there.
- **One logical change per commit.** Rebase, don't merge. Linear
  history.

## The evidence ladder — measurement → result → conclusion → decision

The journal exists so the user can resume cold; a journal full of numbers
they cannot make sense of defeats its purpose (*"I can't even make any
sense of your data without the deep why."*). Every checkpoint climbs the
four-rung ladder — no rung skipped, no lower rung relabelled as higher:

- **Measurement** — a bare number/observation. Explains nothing.
- **Result** — a measurement plus the why, *supported all the way down by
  data* (every link backed, not one asserted mechanism).
- **Conclusion** — drawn only across enough results to understand the
  mechanism.
- **Decision** — taken only on conclusions.

The full doctrine — why the shallow why is worse than an open question,
why a mechanism's own output / partial data / a compaction summary do not
count as grounding, delegating the per-case dig to the forensic agent, and
discipline-theater — is **principle 19**, which `/unattended` already
binds. Read it; do not restate it here. What is unattended-specific:

- A measurement you cannot yet explain is logged as an **open question**,
  never promoted. "measured X, don't yet know why" is honest, resumable
  state and is often the right place to stop — far better than a
  hand-waved why.
- Hold the grounding data on disk (journal / discipline log / scratchpad),
  cited, so a compaction sends you back to the data, not a summary. The
  journal IS the durable substrate P19 requires.
- Distinct from principle 16: P16 asks *can CI re-prove this number*; the
  ladder asks *do we understand why*. A row can be CI-reproducible and
  still be a mere measurement — reproducibly unexplained.

## When to use

- The user says "step away", "be back in N", "babysit this", "work
  unattended", "go do yard work", "go to the garage", "/unattended",
  or anything that signals they're leaving the keyboard.
- Long-running work (multi-step plan, debug loop, bench sweep) where
  mid-session state would be lost without a journal entry.

## Setup

1. **If this is a disciplined-component task:** the discipline log
   under `docs/<initiative>/discipline.md` already exists or is being
   created — that file is the substantive journal. Invoke `/doc` to
   create a thin pointer note under
   `10 - Journals/12 - Notes/<year>/Week <WW>/<YYYY-MM-DD>/<slug>-unattended.md`
   that just points at the discipline log + the current row +
   next-step delta. Don't duplicate state across both files; the
   discipline log is authoritative.
2. **If this is not disciplined-component work:** invoke `/doc` to
   create the full unattended note. Slug is task-specific (e.g.
   `chain-test-rebench-unattended`, `proxima-stage-7-unattended`).
   The `doc` skill creates today's daily if missing and links the new
   note from it.
3. **Note body**: use the unattended template at
   `20 - Assets / 20 - Templates / unattended.md` filled in with:
   current state, context, open questions, data captured so far,
   decisions made, next steps. Cross-link the discipline log row if
   one exists.

## Loop

1. Do the work. Test every decision against the relevant
   `/guiding-principles` rules. Engage `/research-rigor`,
   `/algorithm-development`, `/algorithm-rigor`, `/discovery-loop`,
   `/security-review` per principle 13 when
   the work shape calls for them. Do not skip steps to make the
   journal look cleaner — the journal must reflect reality, including
   dead ends and rolled-back tweaks.
2. After each meaningful checkpoint (new finding, decision, rolled-
   back attempt, blocker, principle conflict), update the
   authoritative log (discipline log row OR unattended note). State /
   Decisions / Next steps must always be current; questions and data
   grow append-only. Place each captured number on the evidence ladder:
   a measurement with no proven why goes under questions, not under
   Decisions; a Decisions entry must trace to a conclusion that traces
   to explained results. If the checkpoint is a discipline-log row, the
   row gets a Changelog entry (date, change, Δ vs prior, CoV / runs,
   host loadout) — and the row's claim carries the mechanism, not just
   the delta.
3. If a checkpoint produces a durable artifact distinct from the
   resumption state (a perf win, a recipe, a design fragment, a
   resolved edge), invoke `/doc` again with a separate slug for that
   artifact. The unattended note stays as the resumption anchor; the
   new note captures the standalone finding. Both end up linked from
   today's daily.
4. Commit work in semantic, focused commits as you go. No cosign.
   Don't push unless the user told you to before stepping away.
5. On the final pass before idling, re-read the resumption pointer
   cold and ask: "could the user resume from this alone — pointer +
   discipline log row + AGENTS.md — without me re-explaining?" If not,
   fix it.

## What NOT to do unattended

- Don't push, force-push, merge, or open PRs unless the user
  explicitly authorised it before stepping away.
- Don't take destructive actions (`git reset --hard`, `rm -rf`,
  branch deletion, dependency removal) without explicit prior
  authorisation. The unattended green light is for continuing
  declared work, not new destructive scope.
- Don't bypass `/guiding-principles` for momentum. A principle-
  violating "good enough" change while the user is away is the
  canonical class of work the user will roll back when they return.
  If a principle conflict shows up, log it in the journal + stop on
  that thread + move to a parallel thread or pause for return.
- Don't add Claude as co-author. Never.
- Don't claim a perf win without a number from the bench. Vibes
  don't pass; numbers do (per `/disciplined-component`).
- Don't log a measurement as a result, a shallow one-liner why as a
  result, or decide off a bare measurement (principle 19). Unexplained →
  open question, not a dressed-up result.
- Don't seal a row on the *intention* to prove — "proof running, holding
  for it", "a result in waiting", or mirroring the correction back, is
  discipline-theater (P19). A launched-but-unfinished dig is an open
  question; the row becomes a result only when the delta is in it.
