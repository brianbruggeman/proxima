# Proxima AI Docs

This tree is the agent-facing memory surface for Proxima. It records
rules, decisions, failures, evidence, and projections in a shape that an
agent can read before touching code.

JSONL files are the source of truth. Markdown files under
`projections/` are readable views of those records.

Start with `AGENT.md`.

## Files

- `AGENT.md` - immediate bootstrap instructions for agents.
- `query.sh` - task query helper for agents.
- `index.jsonl` - top-level routing index.
- `task-routes.jsonl` - task-specific read plans and done criteria.
- `invariants.jsonl` - rules that bind implementation and review.
- `projections/performance-discipline.md` - readable projection of the
  current performance and allocation discipline.

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
