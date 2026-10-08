# Card 07: TOML load and supported graph round trip

**Owner:** Luna
**Branch:** `codex/python-frontend-card-07-toml-roundtrip`
**Worktree:** `/private/tmp/proxima-python-card-07`
**Base:** current `origin/main` after Card 06 lands
**Dependency:** Cards 01, 03, and 06
**Commit:** `feat(tensor): round trip supported python programs through toml`

## Goal

Connect Python-authored programs to Proxima's existing TOML representation without silent semantic loss or a parallel Python-only graph.

## Changes

- Add TOML load/save for the supported subset using the existing `ProgramSpec` and `Vec<Op>` path.
- Explicitly reject Op forms that cannot be represented or recovered without loss.
- Preserve input/output names and the complete supported operation sequence.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC07.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-07 -- uv run --project python -m pytest python/tests/test_toml_roundtrip.py -q` | 3 serialize-load cases preserve complete `Vec<Op>` equality and named outputs; 3 passed, 0 failed |
| AC07.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-07 -- uv run --project python -m pytest python/tests/test_toml_unsupported.py -q` | 2 unsupported forms return distinct named errors; 2 passed, 0 failed |
| AC07.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-07/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-07 --repo-root /private/tmp/proxima-python-card-07 --card-id 07` | `card=07 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC07.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-07 --repo-root /private/tmp/proxima-python-card-07` | `roundtrips=3 unsupported_errors=2 op_equalities=3 literal_oracle_matches=3 rust_incumbent_outputs_pass_same_comparator=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The supported subset round-trips exactly and every unsupported form fails explicitly with its input preserved in the external evidence bundle.
