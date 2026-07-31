---
name: retro
description: Mine a session transcript for recurring failure MODES, root-cause each to the mechanism that should have prevented it, and emit approved diffs back into agents/skills/rules/hooks so the same failure cannot recur.
---

# retro — turn failures into enforcement

You are auditing your own failures to make them structurally impossible, not to apologize.
Output is **diffs to durable files**, gated by the owner. A retro that ends in prose is a failed retro.

## Inputs

- Transcript: `~/.claude/projects/<project-slug>/<session-id>.jsonl` (argument, or the current session).
- Existing enforcement surfaces, read BEFORE proposing anything:
  - **repo instructions: `AGENTS.md` (root → cwd)** — the binding rules for this
    codebase, and the file Codex reads. Fixes usually belong here.
  - **repo agents: `.claude/agents/*.md`** — the specialist workers
  - **repo skills: `.claude/skills/*/SKILL.md`** — the methodologies, including
    `model-calibration` for anything that is a per-model-generation behaviour
  - memory: `~/.claude/projects/<project-slug>/memory/` (+ `MEMORY.md`)
  - machine-level rules: `~/.claude/CLAUDE.md`, `~/.claude/rules/*.md`
  - machine-level agents/skills: `~/.claude/agents/*.md`,
    `~/.claude/skills/*/SKILL.md`
  - hooks: `~/.claude/hooks/`

  **Prefer the repo surface over the machine surface.** A fix written into
  `~/.claude` is invisible to every other reader of this codebase — a teammate,
  CI, Codex, or a fresh clone. Land it in the repo unless the lesson is
  genuinely about this machine or this operator's personal workflow.

  **A behaviour that differs by model generation is not an agent fix.** It goes
  in `model-calibration`'s table, where one edit reaches every consumer, rather
  than being written inline into the agent that happened to surface it.

## Step 1 — mine the ground truth (do not infer, extract)

The owner's corrections are labeled data. Extract every instance of:

- an interrupt / `[Request interrupted by user]`
- rejection words: "no no no", "wrong", "stop", "don't", "should not exist", "why does/do", "wtf", profanity
- a redo instruction ("fix it", "again", "I already told you")
- a tool-call rejection
- an owner-supplied correction of fact

For each: capture the **verbatim quote**, the turn, and **what I had just done**. No paraphrase — the quote is the evidence.

## Step 2 — modes, not incidents

Cluster into failure MODES. A mode qualifies only if:
- it occurred **≥2 times**, OR
- it occurred once and was **severe** (data loss, unauthorized commit/push, a false claim the owner relied on).

Name each mode as a behavior, not a vibe: "claimed absence from a single grep" — not "insufficient rigor".
Count the instances. Report the count.

## Step 3 — root-cause to a MECHANISM

For each mode, answer exactly:

1. **Did a rule/memory for this already exist?** Search the surfaces. Quote it if found.
2. **Why didn't it fire?** (never loaded / loaded but ignored / too vague to be checkable / no trigger / wrong surface)
3. **Which surface should own the fix?**

This is the crux:

| if… | then the fix is… |
|---|---|
| nothing existed | write it at the weakest sufficient level |
| it existed and was **ignored** | **escalate one level** — do NOT restate it |
| it existed but was **unfalsifiable** | rewrite as trigger → action → check |
| it's a per-task procedure | skill |
| it's mechanically detectable | **hook** (the only real gate) |

**Escalation ladder:** memory < rule file < agent system prompt < skill < hook.
A rule that has already been violated has proven its level insufficient. Move it up. Restating it is theater.

## Step 4 — write fixes as checkable triggers

Every fix MUST be:
- **trigger → action**: "Before claiming X does not exist → grep both the public and unrestricted forms, and `git log -S`."
- **falsifiable**: a reader can tell whether it was followed.
- **placed**: name the exact file and the exact insertion point.

Reject any fix that is: "be more careful", "remember to", "try to", or a restatement of an existing rule.

## Step 5 — the replay test (mandatory gate)

For each proposed fix, **replay it against the transcript moment it targets**:

> At turn N I did X. Would this fix have fired and stopped X? Quote the trigger and the action.

If it would not have fired, **discard the fix and go back to Step 3**. This is the single highest-value step — it kills feel-good rules. Report the replay result per fix.

## Step 6 — emit gated diffs

For each surviving fix output, in this order:

1. **mode** + instance count
2. **evidence** — 1-2 verbatim owner quotes
3. **prior art** — the existing rule that failed, or "none"
4. **mechanism** — the surface + why this level (name the escalation if any)
5. **the diff** — exact file, exact text to add/replace
6. **replay** — proof it would have fired

Then STOP and let the owner approve per item. Do not apply anything unasked. Never commit, never push.

## Step 7 — close the loop

After approval:
- apply the approved diffs
- append applied fixes to `~/.claude/projects/<project-slug>/memory/retro_applied.md` (mode → fix → date → surface) so the next retro DEDUPES instead of re-proposing
- if a mode recurs after a fix was applied, that is a **level failure**: escalate the surface, and say so explicitly

## Discipline

- Cite, don't characterize. Every claim carries a quote or a file:line.
- Report the modes you found even if unflattering; the owner cannot re-derive them.
- Do not propose fixes to the owner's behavior. Only to yours and to the tooling.
- No new failure mode may be invented to look thorough. ≥2 instances or severe, or it isn't a mode.
