# Card 10b: bound model session and forward run

**Owner:** Luna
**Branch:** `codex/python-frontend-card-10b-model-session`
**Worktree:** `/private/tmp/proxima-python-card-10b`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `10a`
**Commit:** `feat(python): bound model session and forward run`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Bind a loaded model and run one full forward operation through existing Rust owners.

## Changes

Compare complete output to literal and incumbent Rust results; count Python per-op calls.

- The functional test prints exactly one line in the literal form `PROXIMA_AC_RESULT key=value ...`, derived from its observed fixtures/payload artifacts; the captured stdout includes it because pytest runs with `-s`. The reader reopens those artifacts and recomputes the reported fields.
- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-10b`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC10b.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10b/session -- uv run --project python -m pytest python/tests/test_model_session.py -q -s` | forward_payloads=1 literal_matches=1 rust_incumbent_matches=1 same_comparator_incumbent_passes=1 rust_owned_model_session=1 per_operation_calls=0 comparator=exact_bytes |
| AC10b.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10b/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-10b --repo-root /private/tmp/proxima-python-card-10b --card-id 10b` | `card=10b functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC10b.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10b/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-10b --repo-root /private/tmp/proxima-python-card-10b --card-id 10b` | `card=10b functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
