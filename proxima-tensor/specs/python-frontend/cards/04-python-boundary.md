# Card 04: importable PyO3 boundary

**Owner:** Luna
**Branch:** `codex/python-frontend-card-04-python-boundary`
**Worktree:** `/private/tmp/proxima-python-card-04`
**Base:** current `origin/main` after Card 03 lands
**Dependency:** Cards 01 and 03
**Commit:** `feat(python): add proxima tensor module`

## Goal

Ship a narrow, installable `proxima` Python module over the existing `Vec<Op>`/`NodeId` construction surface, adding only the exact missing Rust functions Card 01 documented and Card 03 supplied. Python objects may hold symbolic Rust handles; the boundary does not own bind policy or execution scheduling.

## Changes

- Add the minimal PyO3 crate/package and build metadata selected by Card 01; use `cargo add` for dependencies rather than editing Cargo manifests by hand.
- Export named input and constant constructors plus symbolic `shape`, `dtype`, and `graph` inspection.
- Define exception conversion and ownership rules. Keep the module feature/build impact explicit and default-off if that is what the architecture report admits.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC04.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-04 -- uv run --project python -m pytest python/tests/test_import.py -q` | 4 passed, 0 failed; imported module origin is the built Proxima extension |
| AC04.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-04 -- cargo check -p proxima-python --all-targets` | 1 package checked, 0 compiler errors |
| AC04.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-04 -- cargo metadata --no-deps --format-version 1` | target census artifact contains 1 library target and 1 test target for `proxima-python` |
| AC04.4 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-04 -- uv run --project python -m pytest python/tests/test_symbolic_handle.py -q` | 4 passed, 0 failed; construction/inspection execution counter = 0 |
| AC04.5 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-04/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-04 --repo-root /private/tmp/proxima-python-card-04 --card-id 04` | `card=04 functional_commands=4 accepted_commands=4 hashes_valid=1 generated_repo_artifacts=0` |
| AC04.6 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-04 --repo-root /private/tmp/proxima-python-card-04` | `import_cases=4 symbolic_handle_cases=4 execution_calls=0 library_targets=1 test_targets=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The package imports from the built extension in the intended supported Python environment and the checked evidence shows only symbolic construction and inspection.
