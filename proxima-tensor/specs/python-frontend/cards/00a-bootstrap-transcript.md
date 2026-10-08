# Card 00a: bootstrap transcript and shell event ledger

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00a-bootstrap-transcript`
**Worktree:** `/private/tmp/proxima-python-card-00a`
**Base:** freshly fetched `origin/main` after the plan revision is committed
**Dependency:** spec-auditor ADMIT for the plan revision
**Commit:** `feat: capture bootstrap transcript and events`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Bootstrap argv:** `script -q /private/tmp/proxima-python-card-00a-bootstrap.log /bin/zsh`
**Bootstrap environment:** `ZDOTDIR=/private/tmp/proxima-python-frontend/bootstrap-zdotdir`
**Bootstrap event ledger:** `/private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl`
**Pre-PTY launcher receipt:** `/private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json`
**Bootstrap hook:** `/private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-hook.zsh`
**Bootstrap launcher:** `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --start-bootstrap-session`
**Finalizer log:** `/private/tmp/proxima-python-card-00a-finalizer.log`

## Goal

Capture Card 00a itself from before repository inspection with one continuous PTY transcript, a shell-command event ledger, and a pre-PTY launcher receipt.

## Changes

Before opening the PTY, create the external evidence directory and task-specific empty `ZDOTDIR` files, then write a launcher receipt containing the exact host argv/source, cwd, monotonic start/end, environment overrides, stdout/stderr, and output hashes. The `--start-bootstrap-session` mode records setup, installs the shell hook at the declared external path, and opens the exact `script` argv above before fetch or repository inspection. Later cards use this bootstrap entry mode to capture their own setup before their card-specific reusable launcher exists. Inside the PTY, install the shell hook before the first repository command; it records every submitted argv, cwd, start/end timestamps, exit status, environment overrides, and transcript byte offsets in the event ledger. Keep this PTY open through fetch/worktree creation, edits, checks, commit/rebase/integration fast-forward/push, and precommit. AC1 checks a deterministic bootstrap fixture and opens its receipt, events, and transcript prefix. After push, close the PTY; AC3 runs a separate post-push reader, waits for its transcript to seal, then checks that complete terminal event and computes this card's elapsed interval. This card does not implement the reusable `--launch-card-session` command; Card 00b adds that launcher on top of the hook and transcript format. The separate plan revision owns and verifies the card-ID/schema checker; its artifacts and timestamps do not count toward this card's logs, outputs, or timer.

- Keep all validation evidence and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00a`; never store them in the repository.
- Preserve the complete raw transcript, failed attempts, and retries. The prefix snapshots used before the PTY closes must remain linked to the final sealed transcript.
- The post-seal closure gate opens the receipt, full event ledger, raw and copied transcripts, finalizer segment, and their hashes before reporting elapsed time.
- Commit one bootstrap transcript/ledger change in this card's dedicated worktree, then use the specified rebase, integration fast-forward, and push workflow.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-bootstrap /private/tmp/proxima-python-frontend/evidence/card-00a/fixture --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --transcript /private/tmp/proxima-python-card-00a-bootstrap.log --prefix-snapshot 1` | bootstrap_events=3 launcher_receipt=1 transcript_prefix=1 argv_records=3 hook_installed=1 output_hashes_valid=1 plan_artifacts_counted=0 |
| AC00a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --transcript /private/tmp/proxima-python-card-00a-bootstrap.log --prefix-snapshot 1` | `card=00a functional=1 accepted=1 hashes_valid=1 launcher_receipt=1 bootstrap_transcript_prefix=1` |
| AC00a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --capture-postpush-closure --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --raw-transcript /private/tmp/proxima-python-card-00a-bootstrap.log --transcript /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap.log --finalizer-log /private/tmp/proxima-python-card-00a-finalizer.log --finalizer-argv 'python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --finalize --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --transcript /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap.log --raw-transcript /private/tmp/proxima-python-card-00a-bootstrap.log'` | `card=00a functional=1 full_log=1 bootstrap_receipt=1 bootstrap_events=3 transcript_hashes_valid=1 postpush_stage=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in this card's external chronological log. The plan-revision work remains a separate prior artifact and is excluded from the card's elapsed interval and acceptance counts. Card 00b consumes this shell hook to implement the reusable session launcher.
