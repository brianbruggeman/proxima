# Card 02: external validation evidence and cassette root

**Owner:** Luna
**Branch:** `codex/python-frontend-card-02-evidence-cassette`
**Worktree:** `/private/tmp/proxima-python-card-02`
**Base:** current `origin/main` after Card 01 lands
**Dependency:** Cards 00 and 01
**Commit:** `test: capture python frontend evidence outside repo`

## Goal

Extend the Proxima cassette test harness with an external cassette-root setting while preserving its existing default path. Reuse Card 00's capture/check tools for every test and git command.

## Changes

- Add a `PROXIMA_CASSETTE_DIR` override to the cassette path resolution in `proxima-test`, retaining manifest-relative `tests/cassettes` when unset.
- Exercise the cassette-supported `Handler` boundary with an external path. Keep pure tensor computation payload checks outside cassette claims.
- Compare record and replay payloads with a fixed literal event fixture. Run the incumbent manifest-relative cassette path through the same payload comparator and require it to pass; a corrupted external cassette is the degenerate rejection control.
- Use Card 00's external Rust target/cache directories and generated-artifact inventory. Reject worktree-local `target`, `build`, `.venv`, `.pytest_cache`, `__pycache__`, cassette, or evidence artifacts.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC02.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-02 -- cargo test -p proxima-test cassette_external_root:: -- --nocapture` | 4 path/mode tests passed, 0 failed |
| AC02.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-02 -- cargo test -p proxima-test cassette_record_replay_external:: -- --nocapture` | record/replay payload matches: 1; literal event-fixture matches: 1; incumbent default-root output passes the same comparator: 1; corrupted-cassette rejection: 1; failures: 0 |
| AC02.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-02/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-02 --repo-root /private/tmp/proxima-python-card-02 --card-id 02` | `card=02 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC02.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-02 --repo-root /private/tmp/proxima-python-card-02` | `record_events=1 replay_events=1 payload_matches=1 literal_event_matches=1 incumbent_default_root_output_passes_same_comparator=1 corrupted_rejections=1 cassette_path_external=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

Record and replay both use the external cassette root, the corrupted control fails as specified, and the complete evidence bundle is checked after push.
