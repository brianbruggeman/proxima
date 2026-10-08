# Card 00l: plan-wide reader dispatch and forged-plan rejection

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00l-plan-finalizer`
**Worktree:** `/private/tmp/proxima-python-card-00l`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00k`
**Commit:** `test: establish plan finalizer`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Dispatch literal IDs `00a`–`00l` and `01a`–`13d`; fail closed for unregistered IDs and independently rerun registered readers read-only.

## Changes

Add read-only `--verify-closed-card` and `--verify-closed-family` checks that reopen sealed artifacts without rewriting them. Inspect actual named worktrees and heads; reject forged summaries and missing worktrees without rewriting result files. Verify the captured plan-session start before its first invocation and verify the sealed finalizer output afterward.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00l`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary. Its closed-card/family reader reopens the sealed result without changing it.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00l.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00l/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-plan /private/tmp/proxima-python-frontend/evidence/card-00l/plan-fixture` | bundles=2 verified=2 forged_summary_rejected=1 missing_worktree_rejected=1 result_rewrites=0 |
| AC00l.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00l/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00l --repo-root /private/tmp/proxima-python-card-00l --card-id 00l` | `card=00l functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC00l.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00l/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00l --repo-root /private/tmp/proxima-python-card-00l --card-id 00l` | `card=00l functional=1 full_log=1 hashes_valid=1 readers_rerun=2 forged_summary_rejected=1 missing_worktree_rejected=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
