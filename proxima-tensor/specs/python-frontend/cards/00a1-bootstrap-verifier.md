# Card 00a1: bootstrap receipt and ledger verifier

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00a1-checker`
**Worktree:** `/private/tmp/proxima-python-card-00a1-checker`
**Base:** freshly fetched `origin/main`
**Dependency:** `00a`
**Commit:** `feat: verify bootstrap receipt and events`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Evidence:** `/private/tmp/proxima-python-frontend/evidence/card-00a1/session`
**Raw transcript:** `/private/tmp/proxima-python-card-00a1-checker.log`

## Goal

Build a strict verifier for actual bootstrap receipts, shell ledger records, and transcript byte spans.

## Changes

Create `check_plan_bootstrap.py` from scratch in this card's worktree. Read the landed 00a receipt, hook ledger, raw transcript, and fixture payloads. Verify exact command bytes against monotonic offsets and marker records. Add malformed receipt/event and forbidden pre-PTY helper controls that require exact rejection reasons. Bind every accepted result to the current source, receipt, and raw hashes. Keep the incomplete Card00a history index immutable and count it only as a historical-invalid sidecar. The already-landed `seed_bootstrap.py` owns post-push stage closure.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a1.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-host-bootstrap /private/tmp/proxima-python-frontend/evidence/card-00a1/fixture --source /private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-07` | `valid_segments=1 byte_spans_checked>=3 malformed_receipts_rejected=1 malformed_events_rejected=1 forbidden_prepty_helpers_rejected=1 wrong_reason_acceptances=0` |
| AC00a1.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00a1/session --card-id 00a1 --raw /private/tmp/proxima-python-card-00a1-checker.log --repo-root /private/tmp/proxima-python-card-00a1-checker` | `card=00a1 functional=1 controls=3 active_hash_links=1 hashes_valid=1` |
| AC00a1.3 | `python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --capture-postpush-closure /private/tmp/proxima-python-frontend/evidence/card-00a1/session --card-id 00a1 --raw /private/tmp/proxima-python-card-00a1-checker.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a1/session/full.log --repo-root /private/tmp/proxima-python-card-00a1-checker --finalizer-log /private/tmp/proxima-python-frontend/evidence/card-00a1/session/finalizer.json --finalize-argv 'python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00a1/session --card-id 00a1 --raw /private/tmp/proxima-python-card-00a1-checker.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a1/session/full.log --repo-root /private/tmp/proxima-python-card-00a1-checker'` | `card=00a1 full_log=1 captured_landing_records>=5 sealed_event=1 elapsed_seconds<=1800` |

## Complete when

The verifier opens real source artifacts and rejects each malformed control for its named reason. AC3 uses only the landed seed closure, captures its own full streams, and reads the sealed terminal before reporting elapsed time. Card 00a2 extends closure and history verification.
