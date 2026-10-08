# Card 08: first-class program, bind, and explain

**Owner:** Luna
**Branch:** `codex/python-frontend-card-08-program-inspection`
**Worktree:** `/private/tmp/proxima-python-card-08`
**Base:** current `origin/main` after Card 07 lands
**Dependency:** Cards 03, 05, 06, and 07
**Commit:** `feat(python): expose bound program inspection`

## Goal

Give a Python-authored computation a first-class program handle and show binding decisions from Proxima's existing inference and bind outputs.

## Changes

- Add explicit program/output ownership and `bind`/`explain` entry points where they map directly to existing Rust interfaces.
- Include graph, inferred shape, BoundOp kinds, and existing fold/fusion details only when those details are present in source outputs.
- Ensure TOML-loaded and Python-authored programs report from the same underlying representation.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC08.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-08 -- uv run --project python -m pytest python/tests/test_program_inspection.py -q` | 3 programs (2 Python-authored, 1 TOML-loaded) match complete expected bound-operation records; 3 passed, 0 failed |
| AC08.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-08 -- uv run --project python -m pytest python/tests/test_explain_control.py -q` | 1 altered-output control changes the bound record; 1 passed, 0 failed |
| AC08.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-08/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-08 --repo-root /private/tmp/proxima-python-card-08 --card-id 08` | `card=08 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC08.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-08 --repo-root /private/tmp/proxima-python-card-08` | `bound_records=3 altered_output_controls=1 literal_oracle_matches=3 rust_incumbent_outputs_pass_same_comparator=3 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

Python and TOML programs use the same bind path, and every explain field can be traced to an existing Proxima output or is omitted.
