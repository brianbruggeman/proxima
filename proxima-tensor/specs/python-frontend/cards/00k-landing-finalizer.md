# Card 00k: card landing and postpush finalizer

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00k-landing-finalizer`
**Worktree:** `/private/tmp/proxima-python-card-00k`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00j`
**Commit:** `test: establish landing finalizer`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Validate the ordered card commit, rebase, integration-worktree fast-forward, push, and remote-main proof against actual git state.

## Changes

Bind each event to the correct worktree and exact argv; reject wrong worktrees and refs.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00k`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00k.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00k/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --self-check-landing-finalizer /private/tmp/proxima-python-frontend/evidence/card-00k/fixture` | commit=1 rebase>=1 integration_ff=1 push=1 card_head=integration_head=remote_main=1 wrong_worktree_rejected=1 wrong_ref_rejected=1 |
| AC00k.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00k/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00k --repo-root /private/tmp/proxima-python-card-00k --card-id 00k` | `card=00k functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC00k.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-00k/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00k --repo-root /private/tmp/proxima-python-card-00k --card-id 00k` | `card=00k functional=1 full_log=1 hashes_valid=1 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
