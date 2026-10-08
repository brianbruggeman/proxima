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
**Bootstrap argv:** `script -q /private/tmp/proxima-python-card-00a-bootstrap.log /bin/zsh -f`
**Bootstrap cwd:** `/Users/brianbruggeman/repos/slot-0/proxima`
**Bootstrap environment:** `zsh -f`; first PTY command creates `/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00a` with `.zshenv` and `.zshrc`, records both hashes, then exports `ZDOTDIR`
**Bootstrap event ledger:** `/private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl`
**Bootstrap receipt:** `/private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json`
**Bootstrap hook:** `/private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-hook.zsh`
**Bootstrap entry point:** host `script` plus `/bin/zsh -f`; no repository helper runs before the PTY
**Finalizer log:** `/private/tmp/proxima-python-card-00a-finalizer.log`

## Goal

Capture Card 00a itself from before repository inspection with one continuous PTY transcript, a shell-command event ledger, and a bootstrap receipt created by the first PTY command.

## Changes

Start the declared host `script` command directly from the existing checkout before fetch, worktree creation, or repository inspection; it must not call a repository helper. `zsh -f` skips startup files. The first command typed into the captured PTY uses shell built-ins and Python standard library to create external evidence and `ZDOTDIR` directories, write the bootstrap receipt, create the hook at its declared external path, and source it before any repository command. The receipt records the exact host argv, starting cwd, monotonic start, environment overrides, and output hashes. The hook records later argv, cwd, start/end timestamps, exit status, environment overrides, and transcript offsets. AC1 runs this host bootstrap with the repository helper absent and proves the first PTY command creates the receipt and installs the hook before repository inspection.

- Keep all validation evidence and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00a`; never store them in the repository.
- Preserve the complete raw transcript, failed attempts, and retries. The prefix snapshots used before the PTY closes must remain linked to the final sealed transcript.
- The post-seal closure gate opens the receipt, full event ledger, raw and copied transcripts, finalizer segment, and their hashes before reporting elapsed time.
- Commit one bootstrap transcript/ledger change in this card's dedicated worktree, then use the specified rebase, integration fast-forward, and push workflow.

The host-negative-07 artifact is limited preimplementation feasibility evidence: it records the declared checkout cwd and a host-only PTY entry, but does not discharge AC00a.1. The AC fields below are future implementation requirements; no prefix-hash or malformed-control success is claimed for that fixture.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-host-bootstrap /private/tmp/proxima-python-frontend/evidence/card-00a/fixture --repo-helper-absent --cwd /Users/brianbruggeman/repos/slot-0/proxima --zdotdir /private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00a --check-zdotdir-files --check-receipt-process-fields --check-hook-event --check-forbidden-helper-control --host-script-argv 'script -q /private/tmp/proxima-python-card-00a-bootstrap.log /bin/zsh -f'` | repo_helper_present=0 cwd_checked=1 host_script_started=1 first_command_created_evidence=1 receipt_schema_valid=1 hook_schema_valid=1 zdotdir_created=1 zdotdir_files_checked=2 receipt_process_fields=1 hook_event_checked=1 malformed_receipt_rejected=1 malformed_hook_rejected=1 malformed_zdotdir_rejected=1 forbidden_prepty_helper_rejected=1 repo_inspection_before_hook=0 transcript_prefix=1 output_hashes_valid=1 |
| AC00a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --transcript /private/tmp/proxima-python-card-00a-bootstrap.log --prefix-snapshot 1` | `card=00a functional=1 accepted=1 hashes_valid=1 launcher_receipt=1 bootstrap_transcript_prefix=1` |
| AC00a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --capture-postpush-closure --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --raw-transcript /private/tmp/proxima-python-card-00a-bootstrap.log --transcript /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap.log --finalizer-log /private/tmp/proxima-python-card-00a-finalizer.log --finalizer-argv 'python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --finalize --repo-root /private/tmp/proxima-python-card-00a --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00a/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap-events.jsonl --transcript /private/tmp/proxima-python-frontend/evidence/card-00a/bootstrap.log --raw-transcript /private/tmp/proxima-python-card-00a-bootstrap.log'` | `card=00a functional=1 full_log=1 bootstrap_receipt=1 bootstrap_events=3 transcript_hashes_valid=1 postpush_stage=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in this card's external chronological log. The plan-revision work remains a separate prior artifact and is excluded from the card's elapsed interval and acceptance counts. Card 00b consumes this shell hook to implement the reusable session launcher.
