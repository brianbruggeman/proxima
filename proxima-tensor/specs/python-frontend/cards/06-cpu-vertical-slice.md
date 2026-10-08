# Card 06: complete CPU vertical slice

**Owner:** Luna
**Branch:** `codex/python-frontend-card-06-cpu-vertical-slice`
**Worktree:** `/private/tmp/proxima-python-card-06`
**Base:** current `origin/main` after Card 05 lands
**Dependency:** Cards 01 through 05
**Commit:** `feat(python): evaluate symbolic tensor programs on cpu`

## Goal

Prove the end-to-end architecture: Python syntax builds one existing Proxima program, a single Rust call binds and evaluates the whole program, and Python receives a materialized result.

## Changes

- Add a whole-program `.run(inputs)` or equivalent entry point that calls existing bind/evaluation code once for the complete graph.
- Return a distinct materialized result object with shape, dtype, and explicit NumPy conversion ownership/copy semantics.
- Add one small worked fixture using `x.shape == (2, 4)`, `w.shape == (4, 3)`, and `(x @ w).silu()` with fixed inputs and an independent literal output oracle.
- Keep Python out of the operation execution loop.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC06.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-06 -- uv run --project python -m pytest python/tests/test_cpu_vertical_slice.py -q` | Op sequence, exact BoundOp kind/shape record, and complete output payload matches both an independently written literal and the incumbent Rust evaluator control; 1 invalid-input control returns a typed Proxima error; 2 passed, 0 failed |
| AC06.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-06 -- uv run --project python -m pytest python/tests/test_cpu_single_boundary_call.py -q` | 1 passed; Python-to-Rust program submission calls = 1; per-operation execution calls = 0 |
| AC06.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-06/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-06 --repo-root /private/tmp/proxima-python-card-06 --card-id 06` | `card=06 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC06.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-06 --repo-root /private/tmp/proxima-python-card-06` | `op_records=1 boundop_records=1 output_payloads=1 invalid_controls=1 boundary_calls=1 per_op_calls=0 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The example returns the expected full output payload through the Rust bind and CPU evaluator, and the separately materialized result's copy behavior is stated from observed ownership.
