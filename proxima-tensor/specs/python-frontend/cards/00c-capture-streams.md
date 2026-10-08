# Card 00c: basic command recorder streams

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00c-capture-streams`
**Worktree:** `/private/tmp/proxima-python-card-00c`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00b`
**Commit:** `test: establish capture streams`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Record a successful child and a child exiting with status 7, with full independent stdout/stderr streams and monotonic start/finish events.

## Changes

Implement normal process capture and immutable stdout/stderr/exit files; prove the exit-7 stream is retained and hashes match. Before this recorder exists, invoke Card 00b’s landed `check_plan_bootstrap.py --launch-card-session` from the parent integration worktree. That launcher captures fetch, worktree setup, edits, and the recorder’s first test into one sealed external segment; copy that segment and hash-link its manifest, complete transcript, and command payloads into this card session. The new recorder captures its own later acceptance events.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00c`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00c.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00c/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-capture-streams /private/tmp/proxima-python-frontend/evidence/card-00c/fixture` | events=2 starts=2 terminals=2 stdout=2 stderr=2 failed=1 exit7=1 hashes_valid=1 |
| AC00c.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00c/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00c --repo-root /private/tmp/proxima-python-card-00c --card-id 00c` | `card=00c functional=1 accepted=1 hashes_valid=1 stream_receipts=2` |
| AC00c.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00c/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00c --repo-root /private/tmp/proxima-python-card-00c --card-id 00c` | `card=00c functional=1 full_log=1 stream_receipts=2 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
