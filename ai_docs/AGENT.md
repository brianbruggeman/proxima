# Proxima Agent Bootstrap

Read this file before broad code exploration in `proxima/`.

The purpose of `ai_docs` is immediate agent use: start from indexed
project memory, then inspect code only where the index says the evidence
lives.

## Startup

1. Read `ai_docs/index.jsonl`.
2. Read `ai_docs/task-routes.jsonl`.
3. Read `ai_docs/invariants.jsonl` records that match the task.
4. Follow the `source_paths` from the selected records before using
   broad search.
5. If the index is missing required structure or evidence, add records
   to `ai_docs` instead of bypassing the structure.

## Useful Queries

```bash
ai_docs/query.sh disciplined-component
ai_docs/query.sh sans-io
jq -c . ai_docs/index.jsonl
jq -c 'select(any(.applies_to[]?; . == "sans-io"))' ai_docs/invariants.jsonl
jq -c 'select(.task == "disciplined-component")' ai_docs/task-routes.jsonl
jq -c 'select(.task == "runtime-prime")' ai_docs/task-routes.jsonl
```

## How to Decide What to Read

- Performance-sensitive component work: start with
  `task-routes.jsonl` task `disciplined-component`.
- Parser or codec work: start with `task-routes.jsonl` task `sans-io`.
- Workspace split or crate ownership: start with
  `task-routes.jsonl` task `decomposition`.
- Runtime or prime work: start with `task-routes.jsonl` task
  `runtime-prime`.
- Bench claims: start with `task-routes.jsonl` task `bench-evidence`.

## Update Rule

When you learn a durable fact, add or update a JSONL record:

- `kind=5` for decisions and rules.
- `kind=7` for failures and negative bench results.
- Use `relations.idx=7` to point at grounding evidence.

Do not encode a guess as a durable fact. If evidence is missing, record
the missing evidence as the next experiment.
