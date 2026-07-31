---
name: research-rigor
description: Self-play tournament for hard answers — incumbent A, parallel critique, alternative author B, synthesis AB, judged by a fresh 3-agent Borda panel; winner becomes the next incumbent until A wins two rounds in a row. Use when the task is hard, contentious, or a single pass is unlikely to be right — design decisions, hard architectural calls, ambiguous specs, contested rewrites, "best of N" answers. Triggers on "Mac's pipeline", "self-play", "tournament", "research-rigor", "be rigorous", "judge panel", "do this carefully", "fan out and critique". NOT for evidence gathering (use research-flow), simple lookups, or tasks with one correct answer.
---

# Research Rigor

Self-play tournament for refining an answer. Each round: produce competing variants, have a fresh panel judge them by Borda count, promote the winner to incumbent. Converges when the same incumbent wins two rounds in a row.

Adapted from Mac Wilkinson's multi-author / critique / judge pattern. Distinct from `research-flow`, which gathers evidence in parallel — this skill *refines a single answer* through competition.

## When to use

- the question is contentious, design-shaped, or has multiple plausible answers
- a single linear pass is unlikely to land the right answer
- the cost of being wrong is higher than the cost of 5–8x token spend
- explicit invocation by the user

Do NOT use for:
- evidence gathering or codebase surveys — use `research-flow`
- factual lookups with one correct answer — use `Read`/`Grep` directly
- mechanical refactors or trivial fixes
- anything where there is no real second-author angle to take

## Cost warning and role dial

This skill spends 6–8x baseline tokens per task (3 author/critique agents + 3 judges per round, up to 4 rounds). Confirm with the user before invoking.

**Set `model` and `effort` per role — that is the cost lever, not agent count.** Authors and critics carry the judgment; judges are evaluating against a fixed rubric and hold quality at lower effort.

| Role | model | effort | why |
|---|---|---|---|
| author A / author B | `opus` | `high` | the candidate's quality is the whole tournament |
| critique | `opus` | `high` | a shallow critique produces a shallow synthesis |
| synthesizer | `opus` | `high` | this is the round's actual output |
| judge ×3 | `opus` | `medium` | ranking against a stated rubric is cheaper than authoring, and accuracy holds |

Do **not** downgrade judges to a smaller model to save money — a judge that cannot evaluate the rubric produces noise, and noise in the Borda count is worse than a smaller panel. Lower the effort instead.

**Do not add agents beyond the roles listed.** No verifier agent, no second critique, no meta-judge. Current models already check their own work; a verification agent stacked on top spends real tokens to re-derive what the judge panel already decides.

## Loop

### Round 0 — produce incumbent A

Spawn a single `Agent` (subagent_type=`general-purpose`, fresh context) with the **complete task specification up front**. Current models do their best work given the whole brief and left to run; a brief you intend to correct mid-flight produces a worse candidate than one stated fully at the start. The returned answer is incumbent **A**.

For a proxima design decision, use `proxima-architect` instead of `general-purpose` — it loads the binding principles and the pipe algebra, so it designs in-bounds rather than being told the rules in the prompt.

Keep the main session uncluttered — let the agent do the authoring, not the orchestrator.

### Round k — tournament

Each round runs in three phases.

**Phase 1 (parallel) — critique and alternative author:**

In a single message, spawn two agents:

- `critique_agent` — fresh `general-purpose`. Input: task + current A. Output: a written critique of A (weaknesses, missed requirements, wrong assumptions, edge cases). Do NOT have it propose fixes — critique only.
- `author_B` — fresh `general-purpose`. Input: task ONLY (blind to A). Output: an independent revision **B**.

Blindness of B is the point — independence is the source of value. Do not show B's author the incumbent.

**Phase 2 — synthesis:**

Spawn `synthesizer_agent` — fresh `general-purpose`. Input: task, A, B, critique. Output: a merge **synthesis_AB** that takes the strengths of both and addresses the critique.

**Phase 3 (parallel) — judge panel:**

In a single message, spawn three judges (fresh `general-purpose`, distinct prompts so they don't anchor on each other). Each judge sees:

- the task
- A, B, synthesis_AB (labelled, in randomized order per judge to defeat position bias)
- the critique (as auxiliary context, not as a candidate)

Each judge returns a ranked triple `[first, second, third]` with one-line reasoning per slot.

**Aggregate (Borda count):**

- 1st place → 2 points, 2nd → 1 point, 3rd → 0 points
- Sum across the 3 judges per candidate
- Winner = highest total. Ties broken by count of first-place votes, then by judge-of-record (judge 1).

The winner becomes the new incumbent A for the next round.

### Convergence

- If the same candidate wins two consecutive rounds → stop. Emit that incumbent.
- Hard cap: 4 rounds (round 0 + 3 tournament rounds). If no convergence by round 3, emit the final round's winner and flag "no convergence — final answer is best-of-tournament, not stable."

## Output

- **Answer**: the final incumbent A
- **Provenance**: which round produced it (round 0 incumbent / round k synthesis / round k revision)
- **Convergence**: stable (won 2 in a row) or capped (no convergence)
- **Trail** (optional, if user asks): per-round critique + Borda totals

## Rules

- set `model` and `effort` per role (see the table above); never add a role that is not in the loop
- fresh context for every agent in every round — never reuse an agent across roles
- author B must be blind to A — the entire point is an independent take
- judges must run in parallel (single message, 3 `Agent` calls) — sequential judging anchors
- randomize candidate order per judge to defeat position bias
- never let the main orchestrator vote — only fresh judges count
- Borda with 3 candidates and 3 judges has rare exact ties; break by first-place-vote count, never by orchestrator preference
- if a judge returns malformed output (no clear ranking), spawn one replacement judge; do not infer the ranking
- cap at 4 rounds total — past that, returns diminish faster than cost grows
- **for a proxima design decision, use the specialist roles**: `proxima-architect` as author A / author B / synthesizer, `proxima-critic` as the critique, `proxima-judge` ×3 as the panel. They carry the binding principles as preloaded context, so the tournament argues about the design instead of about the rules

## Common failure modes

- **B is not blind** — orchestrator leaks A into B's prompt "for context". Kills the independence value; degrades to "edit A twice."
- **Judges anchor on order** — first candidate wins disproportionately. Mitigation: randomize per judge.
- **Critique used as a candidate** — critique is *about* A, not a competing answer. Judges should see it as auxiliary, never rank it.
- **No real B-angle exists** — task has one shape of right answer. The skill is wrong for this task; abort and answer directly.
- **Convergence theater** — A wins twice because B is consistently weak. Inspect: if every B loses by a wide margin, the prompt is underspecifying author_B's brief. Strengthen B's prompt, not the loop.
