# Card 00a: host capture and shell event ledger

**Owner:** Luna
**Branch:** `codex/python-frontend-card-00a-capture`
**Worktree:** `/private/tmp/proxima-python-card-00a-capture`
**Base:** freshly fetched `origin/main` after the plan amendment is committed
**Dependency:** spec-auditor ADMIT for the amended plan
**Commit:** `feat: capture bootstrap shell transcript`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Bootstrap argv:** `script -q /private/tmp/proxima-python-card-00a-capture.log /bin/zsh -f`
**Bootstrap cwd:** `/Users/brianbruggeman/repos/slot-0/proxima`
**Bootstrap evidence:** `/private/tmp/proxima-python-frontend/evidence/card-00a/session-active`
**Historical-invalid attempt index:** `/private/tmp/proxima-python-frontend/evidence/plan-amend-00a-split/card-00a-historical-invalid.jsonl` (SHA-256 `077c952a480504d96b5329dcf0dccd70140abe391095e56351779b79bc83151b`)

## Goal

Capture the first bootstrap and every later command in one continuous raw PTY transcript with a shell-command ledger and process receipt.

## Changes

Start host `script` directly before fetch or repository inspection; no repository helper runs before the PTY. The first PTY command uses shell built-ins and Python standard library to create the external evidence directory, receipt, event ledger, hook, and isolated `ZDOTDIR`, then sources the hook. Only after the hook is active may commands inspect the repository or write `seed_bootstrap.py`. That helper validates the active entry, captures precommit state, and runs the post-seal closure path needed by the first card. The helper is created and invoked inside the already captured PTY.

The incomplete attempt at `/private/tmp/proxima-python-frontend/evidence/card-00a/` is an immutable `historical_invalid` sidecar indexed by the path above. Its hashes and review artifacts remain external. It contributes zero successful AC counts and no elapsed time to this card.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --verify-entry /private/tmp/proxima-python-frontend/evidence/card-00a/session-active --raw /private/tmp/proxima-python-card-00a-capture.log --repo-root /private/tmp/proxima-python-card-00a-capture` | `host_entries=1 receipt_hashes_valid=1 hook_events>=2 first_repo_after_hook=1 omitted_command_rejected=1` |
| AC00a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --precommit /private/tmp/proxima-python-frontend/evidence/card-00a/session-active --raw /private/tmp/proxima-python-card-00a-capture.log --repo-root /private/tmp/proxima-python-card-00a-capture` | `card=00a functional=1 completed_functional_commands=1 pending_precommit_starts=1 prefix_matches=1 hashes_valid=1` |
| AC00a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --capture-postpush-closure /private/tmp/proxima-python-frontend/evidence/card-00a/session-active --card-id 00a --raw /private/tmp/proxima-python-card-00a-capture.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a/session-active/full.log --repo-root /private/tmp/proxima-python-card-00a-capture --finalizer-log /private/tmp/proxima-python-frontend/evidence/card-00a/session-active/finalizer.json --finalize-argv 'python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --finalize /private/tmp/proxima-python-frontend/evidence/card-00a/session-active --card-id 00a --raw /private/tmp/proxima-python-card-00a-capture.log --sealed-transcript /private/tmp/proxima-python-frontend/evidence/card-00a/session-active/full.log --repo-root /private/tmp/proxima-python-card-00a-capture'` | `card=00a full_log=1 captured_landing_records>=5 sealed_precommit_events=1 sealed_postpush_events=1 elapsed_seconds<=1800` |

## Complete when

All three exact commands and the finalizer's own sealed terminal are retained in the external chronological transcript. The historical-invalid attempt index is opened and hash-checked but excluded from this card's AC counts and timer. Card 00a1 can then build a strict verifier from scratch against the landed receipt and ledger.
