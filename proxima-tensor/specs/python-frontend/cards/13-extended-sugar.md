# Card 13: extended Python sugar

**Owner:** Luna
**Branch:** `codex/python-frontend-card-13-extended-sugar`
**Worktree:** `/private/tmp/proxima-python-card-13`
**Base:** current `origin/main` after Card 12 lands
**Dependency:** Cards 01, 05, 06, and 12
**Commit:** `feat(python): add indexing and scalar tensor sugar`

## Goal

Add common indexing, transpose, and scalar-expression conveniences where their behavior can be represented exactly by Proxima's existing operations.

## Changes

- Implement the supported `tensor.T`, basic slicing, scalar arithmetic, and augmented-assignment syntax after Card 01 identifies the exact Op mappings.
- Define `x *= scalar` as rebinding `x` to a new symbolic value; it must not mutate aliases to the prior value.
- Keep advanced indexing and unsupported stride/layout cases as named errors rather than materializing hidden Python-side copies.
- Compare graph sequences and full payloads with independent literal and incumbent Rust controls.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC13.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-13 -- uv run --project python -m pytest python/tests/test_extended_sugar_graphs.py -q` | 6 named cases cover transpose, basic indexing, scalar arithmetic, augmented assignment, and alias preservation; exact Op sequences match independent literal and incumbent Rust constructor controls; 6 passed, 0 failed |
| AC13.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-13 -- uv run --project python -m pytest python/tests/test_extended_sugar_payloads.py -q` | 6 complete outputs match independent literal and incumbent Rust evaluator results through the same comparator; 6 passed, 0 failed |
| AC13.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-13 -- uv run --project python -m pytest python/tests/test_unsupported_indexing.py -q` | 1 unsupported advanced-index case returns its named error; 1 passed, 0 failed |
| AC13.4 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-13/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-13 --repo-root /private/tmp/proxima-python-card-13 --card-id 13` | `card=13 functional_commands=3 accepted_commands=3 hashes_valid=1 generated_repo_artifacts=0` |
| AC13.5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-13 --repo-root /private/tmp/proxima-python-card-13` | `extended_sugar_cases=6 exact_op_matches=6 unsupported_index_errors=1 literal_oracle_matches=6 rust_incumbent_outputs_pass_same_comparator=6 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

Supported syntax has exact graph, alias, shape, and payload evidence, and unsupported indexing forms fail with a named error without hidden eager evaluation.
