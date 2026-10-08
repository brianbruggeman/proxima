# Card 11: NumPy ownership and DLPack

**Owner:** Luna
**Branch:** `codex/python-frontend-card-11-interchange`
**Worktree:** `/private/tmp/proxima-python-card-11`
**Base:** current `origin/main` after Card 10 lands
**Dependency:** Cards 01, 06, and 10
**Commit:** `feat(python): define tensor interchange ownership`

## Goal

Finish the supported host interchange contract and decide from observed ownership whether any DLPack path can be implemented safely.

## Changes

- Exercise NumPy input and output across contiguous and strided arrays, dtype conversion, mutability, and lifetime.
- State the exact owner of every exported/imported buffer and count copies at the boundary.
- Run each of the three DLPack cases against the exact capability decision in the architecture report. Every case must either pass producer/consumer lifetime checks or return the named unsupported-capability error; no case may be unresolved or described as zero-copy without owner evidence.

Capture the full command log, NumPy payloads, DLPack outcome records, ownership records, and checker output under `/private/tmp/proxima-python-frontend/evidence/card-11/`. The evidence checker opens all five NumPy payload/copy-count records and all three DLPack outcome/owner records; these files remain outside the worktree.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC11.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-11 -- uv run --project python -m pytest python/tests/test_interchange.py -q` | 5 ownership cases have complete input/output payloads and owners; 5 passed, 0 failed |
| AC11.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-11 -- uv run --project python -m pytest python/tests/test_dlpack.py -q` | 3 passed, 0 failed; each case records either a valid producer/consumer lifetime or the specified capability error; unresolved outcomes = 0 |
| AC11.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-11/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-11 --repo-root /private/tmp/proxima-python-card-11 --card-id 11` | `card=11 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC11.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-11 --repo-root /private/tmp/proxima-python-card-11` | `numpy_payloads=5 dlpack_cases=3 dlpack_outcome_records=3 owner_records=8 numpy_copy_records=5 unresolved_outcomes=0 literal_oracle_matches=5 rust_incumbent_outputs_pass_same_comparator=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

Every interchange statement follows from the opened payload and owner records. DLPack remains unadvertised if those records do not support its lifetime contract.
