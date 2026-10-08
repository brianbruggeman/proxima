# Card 00i: functional allowlist and precommit gate

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00i-precommit-allowlist`
**Worktree:** `/private/tmp/proxima-python-card-00i`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00h`
**Commit:** `test: establish precommit allowlist`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Count only exact successful functional argv entries for the selected card and require all declared functional commands.

## Changes

Reject missing, extra, or failed expected commands while retaining failed retries; verify precommit inventory event ordering.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00i`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00i.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00i/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-precommit-allowlist /private/tmp/proxima-python-frontend/evidence/card-00i/fixture` | functional=2 accepted=2 inventory_commands=4 stale_rejected=1 repo_artifacts=0 failed_retry_retained=1 |
| AC00i.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00i/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00i --repo-root /private/tmp/proxima-python-card-00i --card-id 00i` | `card=00i functional=1 accepted=1 hashes_valid=1 accepted_commands=2 stale_inventory_rejected=1` |
| AC00i.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00i/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00i --repo-root /private/tmp/proxima-python-card-00i --card-id 00i` | `card=00i functional=1 full_log=1 accepted_commands=2 stale_inventory_rejected=1 hashes_valid=1 captured_landing_records>=5 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log. This card closes only its installed capability and checks that the literal commit, rebase, integration fast-forward, push, and remote-query argv/stdout/stderr records were captured; Card 00l/AC10 later reopens every bundle and validates git state after Card 00k installs those readers.
