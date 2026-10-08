# Card 00d: spawn and signal outcomes

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00d-process-outcomes`
**Worktree:** `/private/tmp/proxima-python-card-00d`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00c`
**Commit:** `test: establish process outcomes`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Record a child spawn error and a child terminated by SIGTERM as complete terminal outcomes.

## Changes

Implement start-before-spawn plus spawn-error and signal receipts; reject missing terminal records.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00d`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00d.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-process-outcomes /private/tmp/proxima-python-frontend/evidence/card-00d/fixture` | events=2 starts=2 terminals=2 spawn_errors=1 signals=1 signal_number=15 hashes_valid=1 |
| AC00d.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00d --repo-root /private/tmp/proxima-python-card-00d --card-id 00d` | `card=00d functional=1 accepted=1 hashes_valid=1 spawn_error=1 signal_number=15` |
| AC00d.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00d --repo-root /private/tmp/proxima-python-card-00d --card-id 00d` | `card=00d functional=1 full_log=1 spawn_error=1 signal_number=15 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
