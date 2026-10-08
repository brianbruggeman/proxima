# Card 00e: external evidence and cassette containment

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00e-external-roots`
**Worktree:** `/private/tmp/proxima-python-card-00e`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00d`
**Commit:** `test: establish external roots`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Require evidence roots and cassette directories to resolve outside every Proxima worktree. Snapshot every cassette file, including JSONL.

## Changes

Reject repository and symlink-redirected roots; prove a JSONL cassette is copied and hashed.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00e`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00e.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00e/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-external-roots /private/tmp/proxima-python-frontend/evidence/card-00e/fixture` | repo_roots_rejected=1 inherited_repo_cassettes_rejected=1 repo_symlinks_rejected=1 external_symlinks_rejected=1 jsonl_cassettes=1 snapshots=1 |
| AC00e.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00e/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00e --repo-root /private/tmp/proxima-python-card-00e --card-id 00e` | `card=00e functional=1 accepted=1 hashes_valid=1 external_roots=1 cassette_hashes=1` |
| AC00e.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00e/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00e --repo-root /private/tmp/proxima-python-card-00e --card-id 00e` | `card=00e functional=1 full_log=1 external_roots=1 cassette_hashes=1 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
