# Card 09: Omega device execution

**Owner:** Luna
**Branch:** `codex/python-frontend-card-09-omega-session`
**Worktree:** `/private/tmp/proxima-python-card-09`
**Base:** current `origin/main` after Card 08 lands
**Dependency:** Cards 06 and 08
**Commit:** `feat(python): execute bound programs through omega`

## Goal

Expose the existing Omega plan/execution path without adding Python backend policy or scheduling.

## Changes

- Add a device/session request that passes through existing Proxima/Omega validation, placement, and execution.
- Keep reusable plans and backend objects Rust-owned.
- Run this card on a Metal-capable host. A missing device capability is a failed preflight for this card, not a skip that permits closure.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC09.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-09 -- uv run --project python -m pytest python/tests/test_omega_session_cpu.py -q` | 1 complete CPU output payload matches the oracle; 1 plan and 2 executions; 0 Python scheduling decisions; 1 passed, 0 failed |
| AC09.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-09 -- uv run --project python -m pytest python/tests/test_omega_session_metal.py -q` | On Metal-capable host: 1 Metal plan and 2 full execution payloads match both an independently written literal and the CPU oracle; 1 passed, 0 failed, 0 skips |
| AC09.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-09/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-09 --repo-root /private/tmp/proxima-python-card-09 --card-id 09` | `card=09 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC09.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-09 --repo-root /private/tmp/proxima-python-card-09` | `metal_supported=1 metal_plans=1 metal_payloads=2 cpu_oracle_payloads=1 literal_oracle_matches=2 rust_incumbent_outputs_pass_same_comparator=2 skips=0 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The device option reaches Rust-owned Omega execution and two Metal result payloads match both an independently written literal and the CPU oracle; the full external log records the Metal capability probe and contains no skipped execution.
