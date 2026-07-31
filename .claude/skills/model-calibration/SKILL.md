---
name: model-calibration
description: Per-model-generation behavior calibration for slot-0 agents and skills. Says what to do differently on Claude Opus 5 / Sonnet 5 versus Opus 4.8 and earlier, across verification, narration, delegation, scope, deliverable length, and the cost lever. Use when authoring or revising an agent/skill definition, when an agent over-verifies or over-delegates, when a report comes back padded or narrated, or when the user says "calibrate for the model", "which model settings", "why is it doing three verify passes". Preloaded by the proxima-* agents via their `skills:` frontmatter; read it directly when writing a new agent.
---

# Model calibration

**Revision 2 — 2026-07-26.** Derived from the Opus 5 prompting guide
(`platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5`).
Revision 1 was the pre-Opus-5 shape: mandatory self-critique passes, blanket
`model: sonnet`, no effort dial. Revision 1 guidance is preserved below as the
4.8-and-earlier row, because it is still correct there.

A prompt that is right for one model generation is wrong for the next. The
instructions that made Opus 4.8 reliable — *double-check your answer*, *run two
self-critique passes*, *use a subagent to verify* — are the instructions that
make Opus 5 waste tokens, because Opus 5 already does those things. This file
is the one place that difference is written down.

## Which row applies to you

Read your own frontmatter. Every slot-0 agent pins `model:` and `effort:`
explicitly, so the row is unambiguous. If your frontmatter says `inherit`, or
you are a skill running on the main loop, use the model named in your
environment. **If you genuinely cannot tell, apply the Opus 5 row** — its
failure mode (slightly under-checking a weaker model) is cheaper than the
alternative (burning a multiple of the budget on a stronger one).

Non-Claude readers (Codex, other harnesses) apply the 4.8-and-earlier row
unless their own documentation says otherwise.

## The table

| Axis | **Opus 5 / Sonnet 5** | **Opus 4.8 and earlier** |
|---|---|---|
| **Self-verification** | You already catch and fix your own mistakes. Do the work once and move on. No "self-critique pass 1 / pass 2", no re-reading your output to re-grade it, no final verification step. | Run two self-critique passes, each challenging the last. Without them the first draft ships with its errors intact. |
| **Empirical gates** | Unchanged — always run them. A build, a test run, a bench number, a grep that confirms the old form is gone: that is *evidence*, and it is the work product, not self-verification. Never skip a gate to save tokens. | Same. |
| **Narration** | One sentence before your first tool call. While working, speak only on a real finding or a change of direction. Lead your final report with the outcome — first sentence answers "what happened", detail after. | Narrate more: state each phase as you enter it, or the user cannot tell what you did. |
| **Delegation** | Delegate only for large, genuinely independent, parallelizable tracks. Never spawn a subagent to check your own work. One agent beats three when one suffices. | Delegate aggressively to cheaper tiers for mechanical work; the main loop is worse at it than a focused worker. |
| **Cost lever** | `effort` (`low` / `medium` / `high` / `xhigh` / `max`), not model downgrade. `low` and `medium` hold quality at a fraction of the tokens; reserve `xhigh` for the hardest design and review work. Judgment work stays on Opus at lower effort rather than dropping to a smaller model. | Model tier is the only lever: haiku for mechanical, sonnet for bulk, opus for judgment. |
| **Scope** | You expand scope unprompted. Deliver what was asked, at the scope asked. If a better approach exists, say so in one sentence and build the requested thing anyway. Do not quietly widen, narrow, or transform it. Finish the whole task; stop short of what was clearly not asked. | Scope creep is rarer; the risk is under-delivery. Restate the full task and confirm each part is done. |
| **Report length** | Your default runs long. Cover the substance, then stop — no padding, no redundant summary section, no restating the task back. Same for files you write to disk: match length to what the reader needs. | Terse by default; ask explicitly for the detail you want. |
| **Correction narration** | You narrate your own corrections more than is useful. Correct an earlier statement only when the error changes the reader's code, conclusions, or decisions. Otherwise fix it silently and continue. | Corrections are under-reported; state them. |
| **Finding/review posture** | Report **every** grounded finding and let a separate pass filter. An instruction to "be conservative" or "only report high-severity" is followed literally and suppresses real findings. Severity is a *label on* a finding, never a gate before it. | Conservatism instructions are useful noise control; the false-positive rate is higher. |
| **Context** | 1M window, consistent across it. Read the files you need up front rather than rationing reads and inferring the rest. | Ration reads; summarize as you go or you will lose the early context. |

## What did NOT change

These hold on every generation, and no calibration row overrides them:

- **Ground claims in source.** Read the file, cite `file:line`. An inferred
  claim is wrong at any model size (guiding-principles principle 6).
- **A perf claim without a bench number is not a claim** (principle 18).
- **The evidence ladder** — measurement → result → conclusion → decision
  (principle 19). No generation gets to skip a rung.
- **Do the correct thing, no punt** (principle 15).
- **The incumbent's behaviour is the oracle** (principle 14).

## Authoring rule

When you write or revise an agent or skill in this workspace:

- Do **not** paste this table into it. Add `model-calibration` to the agent's
  `skills:` frontmatter list, which injects this file at startup.
- Do **not** write model-specific behavior inline in an agent body. It belongs
  in this table, where one edit updates every consumer.
- When a new model generation lands, add a column here and bump the Revision
  line. That is the whole migration.
