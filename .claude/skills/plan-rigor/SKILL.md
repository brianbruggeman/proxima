---
name: plan-rigor
description: Self-play tournament for vetting an implementation plan — incumbent Plan A from the Plan subagent, parallel plan-critique, alternative Plan B (independent), synthesis AB, judged by a fresh 3-agent Borda panel on plan-specific axes (risk surface, ordering, dependencies, rollback, blast radius, missing steps). Use when the plan is high-stakes, contentious, has hidden coupling, or a single linear pass is unlikely to spot the holes — architecture moves, migrations, rewrites, multi-system refactors, anything where a wrong order is expensive. Triggers on "vet this plan", "plan-rigor", "rigorous plan", "Mac's pipeline for planning", "tournament the plan", "judge this plan", "best of N plans". NOT for trivial sequencing, mechanical refactors, or one-file changes (use `Plan` directly).
---

# Plan Rigor

Sibling of `research-rigor`, specialized for vetting an implementation plan. Same tournament shape; different subagent (`Plan`), different role briefs, plan-specific judge axes.

## When to use

- the plan touches many systems, has ordering risk, or affects production
- a wrong sequence (migration applied before schema; refactor before test net) is expensive
- the user wants the plan stress-tested, not just produced
- explicit invocation

Do NOT use for:
- a single file change with obvious sequencing — use `Plan` directly
- evidence gathering — use `research-flow`
- contested non-plan answers (designs, ideas, written arguments) — use `research-rigor`

## Cost warning and role dial

Same as `research-rigor`: 6–8x baseline. Plan agents are heavier than general-purpose. Confirm before invoking.

Set `model` and `effort` per role — that is the cost lever, not agent count:

| Role | model | effort |
|---|---|---|
| Plan A / Plan B / synthesizer | `opus` | `high` |
| critique | `opus` | `high` |
| judge ×3 | `opus` | `medium` |

Lower judge *effort*, never judge *model* — a judge that cannot hold the seven axes produces noise, and noise in the Borda count is worse than a smaller panel.

**Do not add agents beyond the roles listed.** No verifier, no second critique, no meta-judge. Current models check their own work; a verification agent on top re-derives what the panel already decides.

Give each author the **complete task brief up front**. A plan authored against a brief you meant to correct mid-flight is worse than one authored against the whole thing at the start.

## Loop

### Round 0 — produce incumbent Plan A

Spawn one `Agent` with `subagent_type=Plan`, fresh context, full task brief. Returned plan is incumbent **A**.

### Round k — tournament

**Phase 1 (parallel) — critique and alternative plan:**

In a single message:

- `critique_agent` — fresh `general-purpose`. Input: task + Plan A. Output: written critique of A. Score these axes explicitly:
  - **risk surface** — what could break, with what blast radius
  - **ordering** — wrong-order steps, missed dependencies
  - **rollback** — what's reversible vs. point-of-no-return
  - **missing steps** — gaps between stated steps
  - **hidden coupling** — files/systems touched but not named
  - **observability** — what proves each step worked
  - **scope discipline** — steps that aren't strictly required by the task
  Critique only — do not propose fixes.
- `author_B` — fresh `Plan`. Input: task ONLY (blind to A). Output: independent plan **B**.

Blindness of B is mandatory — if B sees A, you get edits, not alternatives.

**Phase 2 — synthesis:**

Spawn `synthesizer_agent` — fresh `Plan`. Input: task, A, B, critique. Output: merged plan **synthesis_AB** that takes A's and B's strengths and addresses each axis from the critique.

**Phase 3 (parallel) — judge panel:**

Three judges in one message (fresh `general-purpose`, prompts distinct enough to defeat anchoring). Each judge sees:

- the task
- A, B, synthesis_AB (labelled, randomized order per judge)
- the critique (auxiliary, not a candidate)

Each judge ranks `[first, second, third]` and scores each plan on the seven axes above with one-line justification per axis. The ranking is what feeds Borda; the per-axis scores feed the trail.

**Aggregate (Borda count):**

- 1st → 2, 2nd → 1, 3rd → 0
- Sum across 3 judges per candidate
- Winner = highest total. Tie-break by first-place vote count, then by sum of per-axis scores across judges.

Winner becomes new incumbent A.

### Convergence

- Same candidate wins two consecutive rounds → stop. Emit it.
- Hard cap: 4 rounds. After that, emit final winner + "no convergence" flag.

## Output

- **Plan**: the final incumbent A
- **Provenance**: round produced, whether stable
- **Risk register**: best per-axis scores rolled up from the winning round's judges
- **Trail** (optional): per-round critique + Borda totals + per-axis judge scores

## Rules

- author A and B and synthesizer all use `subagent_type=Plan` — that subagent is what makes this skill distinct from `research-rigor`
- B must be blind to A — non-negotiable
- judges run in parallel, single message
- randomize candidate order per judge
- judges are `general-purpose`, not `Plan` — judging is evaluation, not authoring; using `Plan` for judging biases toward production over evaluation
- Borda ties broken by first-place count, then by per-axis sum; never by orchestrator preference
- if a Plan agent returns prose without a stepwise plan, reject and respawn — do not accept a non-plan as a plan
- cap at 4 rounds

## Common failure modes

- **Synthesis is mush** — synth_agent concatenates A and B instead of merging. Mitigation: synthesizer brief must explicitly require choosing between conflicting steps with a one-line justification per choice.
- **B reinvents A** — when the task is small enough that there's one natural plan. Symptom: B looks ~95% like A in every round. The skill is wrong for this task; abort and use `Plan` directly.
- **Judges all favor A** — incumbent bias. Mitigation: judges see A/B/synth in randomized order with neutral labels (Plan-1 / Plan-2 / Plan-3), not "incumbent" / "revision" / "synthesis".
- **Per-axis scores ignored** — Borda only uses ranking. If a candidate wins on overall ranking but loses on rollback axis specifically, surface that in the output even when it doesn't change the winner.
- **Scope creep through synthesis** — synth_agent adds steps that neither A nor B had. Mitigation: synthesizer brief requires every step in the merge to trace to A or B (or to a specific critique point being addressed).
