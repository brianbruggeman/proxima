# Card 00h: five-root historical archive chain

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00h-history-chain`
**Worktree:** `/private/tmp/proxima-python-card-00h`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00g`
**Commit:** `test: establish history chain`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Build an append-only hash chain over five retained pre-split roots without modifying or counting the unfinished active attempt.

## Changes

Open and hash each raw log/catalog/command payload; verify duplicate and exact-prefix relationships, and reject a changed historical catalog.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00h`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00h.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00h/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-history /private/tmp/proxima-python-frontend/evidence/card-00h --sidecar /private/tmp/proxima-python-frontend/evidence/pre-split-card-00/sidecar-manifest.json` | source_roots=5 unique_events=168 unique_failures=14 duplicate_roots=1 prefix_snapshots=1 source_index_sha256=7ae94687864ed583c29bd9a4224c29cf1629702cf7eb9cbf0534e8dae14ba2fb hashes_valid=1 active_attempt_counted=0 catalog_tamper_rejected=1 |
| AC00h.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00h/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00h --repo-root /private/tmp/proxima-python-card-00h --card-id 00h` | `card=00h functional=1 accepted=1 hashes_valid=1 source_roots=5 unique_events=168 unique_failures=14` |
| AC00h.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00h/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00h --repo-root /private/tmp/proxima-python-card-00h --card-id 00h` | `card=00h functional=1 full_log=1 source_roots=5 unique_events=168 unique_failures=14 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
