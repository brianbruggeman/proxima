# Card 00g: typed artifacts and executable comparisons

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00g-typed-artifacts`
**Worktree:** `/private/tmp/proxima-python-card-00g`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00f`
**Commit:** `test: establish typed artifacts`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Add required typed artifact rows and executable byte/text/hash comparators.

## Changes

Open every referenced artifact and recompute each comparator; reject a removed cassette row and a mismatching exact-bytes pair.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00g`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00g.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00g/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-typed-artifacts /private/tmp/proxima-python-frontend/evidence/card-00g/fixture` | typed_rows_checked=2 comparator_matches=1 comparator_mismatch_rejected=1 missing_typed_row_rejected=1 |
| AC00g.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00g/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00g --repo-root /private/tmp/proxima-python-card-00g --card-id 00g` | `card=00g functional=1 accepted=1 hashes_valid=1 typed_rows=2 comparators=1` |
| AC00g.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00g/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00g --repo-root /private/tmp/proxima-python-card-00g --card-id 00g` | `card=00g functional=1 full_log=1 typed_rows=2 comparators=1 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
