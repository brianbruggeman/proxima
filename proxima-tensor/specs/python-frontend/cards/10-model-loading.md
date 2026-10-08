# Card 10: model loading and bound session

**Owner:** Luna
**Branch:** `codex/python-frontend-card-10-model-loading`
**Worktree:** `/private/tmp/proxima-python-card-10`
**Base:** current `origin/main` after Card 09 lands
**Dependency:** Cards 01, 07, 08, and 09
**Commit:** `feat(python): load and bind tensor models`

## Goal

Expose a thin Python production path for an existing TOML model and safetensors weights through Proxima's Rust model loading and binding owners.

## Changes

- Add `px.load(model_toml, weights=...)` and an explicit `bind(device=...)` or equivalent. If Card 01 finds the required Rust loader/session capability is not public, extend the existing Rust-owned model path narrowly in this card; do not drop the requested loading behavior.
- Keep weight ownership, validation, binding, placement, and execution in Rust. The Python object holds a Rust-owned model/session handle.
- Preserve one complete model forward operation through the same program/bind/execution path verified by earlier cards.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC10.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10 -- uv run --project python -m pytest python/tests/test_model_load.py -q` | 2 valid fixture loads passed, 0 failed; 1 malformed descriptor returns the named Rust validation error |
| AC10.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10 -- uv run --project python -m pytest python/tests/test_model_session.py -q` | 1 full forward payload matches the independently authored literal output fixture and incumbent Rust output through the same comparator; model and session remain Rust-owned; Python per-op execution calls = 0 |
| AC10.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-10 --repo-root /private/tmp/proxima-python-card-10 --card-id 10` | `card=10 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC10.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-10 --repo-root /private/tmp/proxima-python-card-10` | `load_payloads=3 forward_payloads=1 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The named TOML/safetensors fixture is loaded, bound, and run through existing Rust owners, with the complete forward payload and error payloads retained outside the repository.
