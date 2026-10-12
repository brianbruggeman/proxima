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

Build a strict verifier for actual bootstrap receipts, shell ledger records, transcript event-marker spans, and the exact first repository-command echo.

## Changes

Create `check_plan_bootstrap.py` from scratch in this card's worktree. Read the landed 00a receipt, hook ledger, raw transcript, and fixture payloads. Verify every event's raw start/end marker pair against its ledger ID and validate the complete echoed argv for the first repository command. Add malformed receipt/event, forbidden pre-PTY helper, and added-argv echo controls that require exact rejection reasons. Bind every accepted result to the current source, receipt, and raw hashes. Keep the incomplete Card00a history index immutable and count it only as a historical-invalid sidecar. The already-landed `seed_bootstrap.py` owns post-push stage closure.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a1.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-host-bootstrap /private/tmp/proxima-python-frontend/evidence/card-00a1/fixture --source /private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-10` | `valid_segments=1 event_marker_pairs=104 first_command_echo=1 malformed_receipts_rejected=1 malformed_events_rejected=1 forbidden_prepty_helpers_rejected=1 echo_mutation_rejected=1 wrong_reason_acceptances=0` |
| AC00a1.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00a1/session --card-id 00a1 --raw /private/tmp/proxima-python-card-00a1-checker.log --repo-root /private/tmp/proxima-python-card-00a1-checker` | `card=00a1 functional=1 controls=4 active_hash_links=1 hashes_valid=1` |
| AC00a1.3 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --run-closure /private/tmp/proxima-python-frontend/evidence/card-00a1/session --raw /private/tmp/proxima-python-card-00a1-checker.log --repo-root /private/tmp/proxima-python-card-00a1-checker --seed-script proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py` | `card=00a1 primary_raw_equals_full=1 closure_segment_sealed=1 captured_landing_records=5 elapsed_seconds<=1800` |

## Complete when

The verifier opens real source artifacts and rejects each malformed control for its named reason. AC3 runs only after the primary PTY exits. It preserves the primary raw transcript byte-for-byte as `full.log`, captures landing operations in a separate waited closure PTY segment, and writes a hash-linked receipt for both segments and the child terminal streams. The child invokes the seed `--finalize` reader. It verifies the primary/full equality and both segment hashes before reporting elapsed time. Card 00a2 extends closure and history verification.
