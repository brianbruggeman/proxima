# Card 05: Python operators and symbolic inspection

**Owner:** Luna
**Branch:** `codex/python-frontend-card-05-python-syntax`
**Worktree:** `/private/tmp/proxima-python-card-05`
**Base:** current `origin/main` after Card 04 lands
**Dependency:** Cards 03 and 04
**Commit:** `feat(python): lower tensor syntax into proxima ops`

## Goal

Make basic composition natural in Python while each expression appends to the same Rust-owned program representation.

## Changes

- Implement `+`, `*`, `@`, and `.silu()` for symbolic values, plus top-level forms only where the report shows they improve API consistency.
- Expose shape/dtype and a graph view sourced from the stored Op sequence.
- Derive shape errors from existing Proxima inference/validation.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC05.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05 -- uv run --project python -m pytest python/tests/test_symbolic_ops.py -q` | 5 expressions match complete Rust Op sequence oracles; 5 passed, 0 failed |
| AC05.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05 -- uv run --project python -m pytest python/tests/test_symbolic_errors.py -q` | 1 incompatible-shape error matches the Rust error class/message; 1 passed, 0 failed |
| AC05.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05 -- uv run --project python -m pytest python/tests/test_symbolic_no_execution.py -q` | 1 passed; bind calls = 0, evaluator calls = 0, device calls = 0 |
| AC05.4 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-05 --repo-root /private/tmp/proxima-python-card-05 --card-id 05` | `card=05 functional_commands=3 accepted_commands=3 hashes_valid=1 generated_repo_artifacts=0` |
| AC05.5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-05 --repo-root /private/tmp/proxima-python-card-05` | `graph_payloads=5 shape_errors=1 bind_calls=0 evaluate_calls=0 device_calls=0 literal_oracle_matches=5 rust_incumbent_outputs_pass_same_comparator=5 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The operator results and inspection output match the Rust-authored graph payloads, and graph construction does not bind or execute.
