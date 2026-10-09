# Card 00b: reusable card-session launcher

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00b-session-launcher`
**Worktree:** `/private/tmp/proxima-python-card-00b`
**Base:** freshly fetched `origin/main` after Card 00a2 lands
**Dependency:** `00a2`
**Commit:** `feat: add reusable card session launcher`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Pre-launch raw transcript:** `/private/tmp/proxima-python-card-00b-bootstrap.log`
**Pre-launch receipt:** `/private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-receipt.json`
**Pre-launch event ledger:** `/private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-events.jsonl`
**Pre-launch hook:** `/private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-07/bootstrap-hook.zsh`
**Pre-launch ZDOTDIR:** `/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00b`

## Goal

Add a reusable `--launch-card-session` command that starts an external session under the Card 00a hook and returns a sealed, hash-verifiable source segment. It depends on the landed Card 00a capture hook, Card 00a1 receipt/ledger verifier, and Card 00a2 closure reader.

## Changes

Before the reusable launcher exists, start with host argv `script -q /private/tmp/proxima-python-card-00b-bootstrap.log /bin/zsh -f`. Its first captured PTY command creates Card 00b external evidence and `/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00b` with `.zshenv` and `.zshrc`, writes and hashes the bootstrap receipt, exports `ZDOTDIR=/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00b`, sets the event-ledger path, then sources the already-landed Card 00a hook before any fetch, worktree creation, inspection, or edit. It invokes no repository helper before the PTY starts. The hook records the remaining commands through finalization. The fixture opens and hash-checks Card 00a's retained host transcript/ledger segment, Card 00a1's verifier result, and Card 00a2's sealed closure result; these are distinct inputs and remain immutable.

- The functional check runs the launcher in a disposable fixture, opens its receipt/transcript/event files, and prints exactly the expected count fields below from those artifacts. Its input is the separately retained Card 00a host transcript/ledger segment.
- Keep all validation evidence and review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-00b`; never store them in the repository.
- Capture every command through inspection, edits, checks, commit, rebase, integration fast-forward, push, and finalization. Preserve failures and retries in one chronological log.
- Start the monotonic timer before fetch/worktree setup and close it only after the post-seal reader checks the terminal event.
- Commit one reusable launcher implementation in this card's dedicated worktree. Card 00c consumes it for ordinary command-stream capture.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00b.1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --self-check-launch-card-session /private/tmp/proxima-python-frontend/evidence/card-00b/fixture --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/events.jsonl --transcript /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/session/full.log` | launcher_events=3 fixture_sessions=1 fixture_receipt_checked=1 fixture_full_log_checked=1 fixture_segment_hash=1 prior_card_artifacts_modified=0 |
| AC00b.2 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --precommit --repo-root /private/tmp/proxima-python-card-00b --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/events.jsonl --transcript /private/tmp/proxima-python-card-00b-bootstrap.log --prefix-snapshot 1 --bootstrap-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-events.jsonl --bootstrap-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-receipt.json --bootstrap-log /private/tmp/proxima-python-card-00b-bootstrap.log --bootstrap-hook /private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-07/bootstrap-hook.zsh --fixture-transcript /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/session/full.log` | `card=00b functional=1 accepted=1 hashes_valid=1 fixture_receipt=1 fixture_segment_hash=1 bootstrap_receipt=1 bootstrap_ledger_checked=1 bootstrap_ledger_hash=1 bootstrap_segment_hash=1` |
| AC00b.3 | `python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --capture-postpush-closure --repo-root /private/tmp/proxima-python-card-00b --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/events.jsonl --raw-transcript /private/tmp/proxima-python-card-00b-bootstrap.log --transcript /private/tmp/proxima-python-frontend/evidence/card-00b/session/full.log --bootstrap-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-events.jsonl --sealed-bootstrap-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/session/bootstrap-events.jsonl --bootstrap-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-receipt.json --bootstrap-hook /private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-07/bootstrap-hook.zsh --fixture-transcript /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/session/full.log --finalizer-log /private/tmp/proxima-python-card-00b-finalizer.log --launcher-fixture-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/launcher-receipt.json --launcher-fixture-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/events.jsonl --launcher-fixture-transcript /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/session/full.log --finalizer-argv 'python3 proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py --finalize --repo-root /private/tmp/proxima-python-card-00b --launcher-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/launcher-receipt.json --manifest /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/events.jsonl --transcript /private/tmp/proxima-python-frontend/evidence/card-00b/session/full.log --bootstrap-receipt /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-receipt.json --bootstrap-log /private/tmp/proxima-python-card-00b-bootstrap.log --bootstrap-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/bootstrap-events.jsonl --sealed-bootstrap-manifest /private/tmp/proxima-python-frontend/evidence/card-00b/session/bootstrap-events.jsonl --bootstrap-hook /private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-07/bootstrap-hook.zsh --fixture-transcript /private/tmp/proxima-python-frontend/evidence/card-00b/fixture/session/full.log'` | `card=00b functional=1 full_log=1 fixture_receipt=1 fixture_segment_hash=1 bootstrap_receipt=1 bootstrap_ledger_checked=1 bootstrap_ledger_hash=1 sealed_bootstrap_ledger_hash=1 bootstrap_segment_hash=1 source_segment_hashes=2 postpush_stage=1 elapsed_seconds<=1800` |

## Complete when

The pre-launch bootstrap log/ledger/receipt and the fixture's separately rooted full command log, receipt, and sealed source segment are opened and hash-checked; the post-push closure also opens the distinct sealed bootstrap-ledger copy and checks it against the live ledger hash; both source hashes remain unchanged after closure, and the distinct paths prevent one segment from overwriting the other. The three acceptance commands produce the counted outputs above. After the final command is sealed, the post-seal closure record measures this card's own elapsed interval. The plan revision and Card 00a evidence remain outside that interval and count.
