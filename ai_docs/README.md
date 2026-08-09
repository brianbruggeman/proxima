# Proxima AI Docs

This tree is the agent-facing **project memory** for Proxima. It records
decisions, failures, open questions, and the evidence behind them, in a
shape an agent can query before touching code.

**If you are a person new to proxima, this is not your entry point** — these
are terse, evidence-bearing routing records for an agent that already knows
the vocabulary (`Pipe`, sans-IO, tier, and so on), not teaching material.
Start at [`docs/tutorials/00-foundations.md`](../docs/tutorials/00-foundations.md)
instead, which defines every term before using it. Come back here once you
know the algebra and want the agent-facing index.

Start with `AGENT.md`.

## What belongs here, and what does not

The line is **standing rule versus recorded event** — not topic.

| | Lives in | Shape | Changes |
|---|---|---|---|
| **Standing rules** | `AGENTS.md`, `.claude/skills/` | prose, hand-curated, read start-to-finish | when policy changes |
| **Recorded events** | this tree | JSONL, evidence-bearing, read by task-route | every landing |

"Box-free by default" is a standing rule — it lives in `AGENTS.md`.
"The pgwire codec stays tier-3 on `thumbv7em-none-eabihf`, decode
borrowed" is a decision with evidence — it lives here. "On the ARM64
Hypervisor.framework HVC path, resume at the PC reported after the
exit" is a recorded failure — it lives here, and **nothing else can
hold it**. That is the justification for this tree.

**Do not restate a rule here.** Rules were duplicated into
`invariants.jsonl` as `proxima.rule.*` and `proxima.guiding.*` records
and drifted from the prose; 21 such records were removed on 2026-07-27.
Point at `proxima.AGENTS.hot_path` or
`proxima.guiding_principles.skill` instead — those index entries
resolve to the prose that owns the rule.

## Files

- `AGENT.md` - immediate bootstrap instructions for agents.
- `query.sh` - task query helper for agents.
- `index.jsonl` - top-level routing index.
- `task-routes.jsonl` - task-specific read plans and done criteria.
- `invariants.jsonl` - decisions, failures, and open questions, with
  their evidence. Not rules.
- `examples-index.jsonl` - the combinator vocabulary and where each
  example lives.
- `projections/` - long-form readable documents for a named audience
  (e.g. an operator wiring OTLP), not summaries of the JSONL.

## Record Shape

Records intentionally mirror the local memory taxonomy:

- `kind=3` concept
- `kind=5` decision or rule
- `kind=7` failure to avoid repeating

Relations use the shared index vocabulary:

- `idx=3` depends_on
- `idx=6` resolves
- `idx=7` grounded_in

Every record should include concrete evidence or the evidence required
to accept an exception. A rule without instrumentation requirements is
not an invariant; it is just advice.
