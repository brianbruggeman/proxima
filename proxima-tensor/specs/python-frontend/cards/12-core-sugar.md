# Card 12: core Python sugar

**Owner:** Luna
**Branch:** `codex/python-frontend-card-12-core-sugar`
**Worktree:** `/private/tmp/proxima-python-card-12`
**Base:** current `origin/main` after Card 11 lands
**Dependency:** Cards 01, 05, 06, and 08
**Commit:** `feat(python): add core tensor activation and reduction sugar`

## Goal

Add the core activation, probability, and reduction expressions named in the prompt where Card 01 proves the existing Op representation can express their semantics.

## Changes

- Implement `.gelu()`, `.softmax(dim)`, `.sum(dim, keepdims=...)`, and `.argmax(dim)` with exact semantics using the existing Op graph and Rust bind/evaluation path. If Card 01 finds an expression has no exact path, revise and re-audit this card before it starts; do not silently omit a named operation.
- Preserve the existing single-program construction and Rust-owned bind/evaluation path.
- Compare every expression's full `Vec<Op>` sequence, shape/dtype, and materialized payload against an independent literal oracle and the incumbent Rust constructor/evaluator through the same comparator.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC12.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-12 -- uv run --project python -m pytest python/tests/test_core_sugar_graphs.py -q` | 8 named cases cover GELU, softmax, sum, and argmax graph/shape semantics; complete Op sequences match both independent literal and incumbent Rust constructor controls; 8 passed, 0 failed |
| AC12.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-12 -- uv run --project python -m pytest python/tests/test_core_sugar_payloads.py -q` | 8 complete payloads match independent literal and incumbent Rust evaluator outputs through the same comparator; 8 passed, 0 failed |
| AC12.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-12/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-12 --repo-root /private/tmp/proxima-python-card-12 --card-id 12` | `card=12 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC12.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-12 --repo-root /private/tmp/proxima-python-card-12` | `core_sugar_cases=8 exact_op_matches=8 unsupported_errors=0 literal_oracle_matches=8 rust_incumbent_outputs_pass_same_comparator=8 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

All four named operations have exact graph, shape, and payload evidence; no Python operation-by-operation execution is introduced.
