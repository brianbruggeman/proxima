# Card 00j: generated repository artifact inventory

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00j-repo-inventory`
**Worktree:** `/private/tmp/proxima-python-card-00j`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00i`
**Commit:** `test: establish repo inventory`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Detect generated artifacts in working and committed card changes, including postpush trees.

## Changes

Compare changed committed paths against the card base and reject a forbidden generated path even when origin/main equals HEAD.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00j`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00j.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00j/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-repo-inventory /private/tmp/proxima-python-frontend/evidence/card-00j/fixture` | inventory_commands=4 committed_artifact_detected=1 stale_rejected=1 generated_repo_artifacts=0 |
| AC00j.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00j/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00j --repo-root /private/tmp/proxima-python-card-00j --card-id 00j` | `card=00j functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC00j.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00j/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00j --repo-root /private/tmp/proxima-python-card-00j --card-id 00j` | `card=00j functional=1 full_log=1 generated_repo_artifacts=0 committed_artifact_control=1 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
