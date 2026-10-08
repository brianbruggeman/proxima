# Card 00a2: sealed closure and attempt-chain verifier

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00a2-closure`
**Worktree:** `/private/tmp/proxima-python-card-00a2-closure`
**Base:** freshly fetched `origin/main`
**Dependency:** `00a1`
**Commit:** `feat: validate sealed bootstrap sessions`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Evidence:** `/private/tmp/proxima-python-frontend/evidence/card-00a2/session`
**Raw transcript:** `/private/tmp/proxima-python-card-00a2-closure.log`

## Goal

Extend the bootstrap verifier to validate completed logs, lifecycle records, and immutable attempt history after sealing.

## Changes

Extend the landed `check_plan_bootstrap.py` to open the sealed raw/copied transcript, ordered command streams and hashes, AC1/AC2 hash links, receipt-derived timer, lifecycle rows, git command records, and historical-invalid segment index. Reject stale receipt hashes, duplicate precommits, deleted tails, mismatched AC links, and unsealed terminals. Report success only after the outer terminal event is sealed and reopened. The landed seed helper captures and seals the finalizer invocation.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a2.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-closure /private/tmp/proxima-python-frontend/evidence/card-00a2/fixture --source /private/tmp/proxima-python-frontend/evidence/card-00a1/session` | `valid_closures=1 stale_attempts_rejected=1 duplicate_precommits_rejected=1 deleted_tails_rejected=1 mismatched_ac_links_rejected=1 unsealed_terminals_rejected=1` |
| AC00a2.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00a2/session --card-id 00a2 --raw /private/tmp/proxima-python-card-00a2-closure.log --repo-root /private/tmp/proxima-python-card-00a2-closure` | `card=00a2 functional=1 controls=5 active_hash_links=1 hashes_valid=1` |
| AC00a2.3 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --capture-postpush-closure /private/tmp/proxima-python-frontend/evidence/card-00a2/session --card-id 00a2 --raw /private/tmp/proxima-python-card-00a2-closure.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a2/session/full.log --repo-root /private/tmp/proxima-python-card-00a2-closure --finalizer-log /private/tmp/proxima-python-frontend/evidence/card-00a2/session/finalizer.json --finalize-argv 'python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00a2/session --card-id 00a2 --raw /private/tmp/proxima-python-card-00a2-closure.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a2/session/full.log --repo-root /private/tmp/proxima-python-card-00a2-closure'` | `card=00a2 full_log=1 captured_landing_records>=5 sealed_event=1 elapsed_seconds<=1800` |

## Complete when

All four adversarial controls reject for their recorded causes; AC1 and AC2 inputs are hash-linked to the current attempt; AC3 reopens its sealed outer event and checks elapsed time. Card 00b consumes the landed 00a hook, 00a1 verifier, and 00a2 closure reader.
