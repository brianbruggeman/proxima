# Card 00: external validation capture and checker

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00-evidence-tools`
**Worktree:** `/private/tmp/proxima-python-card-00`
**Base:** fetched `origin/main` at card start
**Dependency:** spec-auditor ADMIT on this card set
**Commit:** `test: capture python frontend validation evidence`

## Goal

Provide reusable tooling that records and checks complete validation evidence outside the repository before any Python implementation begins.

## Changes

- Add no-dependency `capture_validation.py` and `check_evidence.py` under the frontend spec's `scripts/` directory.
- Capture each command's exact argument vector, working directory, start/end timestamps, relevant environment overrides, complete stdout and stderr, exit status, and SHA-256 hashes for payloads and cassettes. Retain failed attempts in order. Under each evidence root, route `CARGO_TARGET_DIR`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `PYTHONPYCACHEPREFIX`, `TMPDIR`, `XDG_CACHE_HOME`, and the default `PROXIMA_CASSETTE_DIR` to external subdirectories.
- Check actual files and payload records, validate hashes/counts, reject missing/truncated records and repository-local artifacts, implement `--precommit` without git-event requirements, and implement `--finalize-plan` to open every card bundle plus each worktree/commit/rebase/push record and require all 14 cards to be present and checked.
- Add a self-check that captures two synthetic commands, validates their streams and exit records, corrupts one copied artifact, and proves the checker rejects it.
- Before writing these scripts, run the configured `spec-auditor` agent on `SPEC.md`, `TASKS.md`, and every card. Record its ADMIT artifact under this card's external evidence directory. REFUSE means revise and rerun the audit before coding.

Before the first Card 00 inspection or edit, start `script -q /private/tmp/proxima-python-card-00-bootstrap.log /bin/zsh`; preserve the raw transcript and exact launcher argv. In that recorded shell, create `/private/tmp/proxima-python-frontend/evidence/card-00/` and write the launcher argv into the bootstrap manifest. After the shell exits, run `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00/session -- cp /private/tmp/proxima-python-card-00-bootstrap.log /private/tmp/proxima-python-frontend/evidence/card-00/bootstrap.log` to copy the completed transcript; the capture log records the copy command. Then set `CARGO_TARGET_DIR`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `PYTHONPYCACHEPREFIX`, `TMPDIR`, and `XDG_CACHE_HOME` to card-specific directories under `/private/tmp/proxima-python-frontend/`. The checker inventories `.venv`, `.pytest_cache`, `__pycache__`, `target`, `build`, `dist`, `*.egg-info`, cassette, and evidence artifacts under the worktree. Source changes remain in the worktree; validation outputs do not.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check /private/tmp/proxima-python-frontend/evidence/card-00/tool-self-check` | `captured_commands=2 stdout_records=2 stderr_records=2 exit_records=2 corrupted_artifact_rejected=1 repository_artifacts=0` |
| AC00.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00/session -- python3 --version` | one complete command event; stdout, stderr, exit code, and file hashes are present |
| AC00.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00 --repo-root /private/tmp/proxima-python-card-00 --card-id 00` | `card=00 functional_commands=2 accepted_commands=2 hashes_valid=1 generated_repo_artifacts=0` |
| AC00.4 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00 --repo-root /private/tmp/proxima-python-card-00` | after commit/rebase/push: `bootstrap_records=1 validation_events>=2 commit_events=1 rebase_events>=1 push_events=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 literal_oracle_matches=0 rust_incumbent_controls=0 correctness_controls_not_applicable=1` |

The first recorded Card 00 command is the `script -q .../bootstrap.log /bin/zsh` launcher specified in the spec. Before that session exits, run the configured spec auditor and all Card 00 preparation through the transcript. After it exits, copy the full transcript through the capture tool. Route every subsequent command through the capture tool, including acceptance, commit, rebase, fast-forward, push, and the transcript-copy command. The plan finalizer also checks that each bundle contains both a literal oracle and incumbent Rust control where its card performs correctness comparisons. Before commit run `--precommit`; after push run `--finalize`, which opens the complete log, manifest, payloads, cassette files, review artifacts, hashes, and generated-artifact inventory. Do not assert a fixed event count because failures and retries are retained in the full log.

## Complete when

The self-check's corrupted artifact is rejected, an actual full session is captured, and the post-push finalizer validates the complete external bundle with zero generated repository artifacts.
