# Card 10a: TOML model and safetensors loading

**Owner:** Luna
**Branch:** `codex/python-frontend-card-10a-model-load`
**Worktree:** `/private/tmp/proxima-python-card-10a`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `09b`
**Commit:** `feat(python): TOML model and safetensors loading`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Expose `px.load` for one valid TOML descriptor and safetensors fixture, with a typed malformed-descriptor error.

## Changes

Keep parsing, weights ownership, and validation in Rust; preserve complete source and weight fixtures externally.

- The functional test prints exactly one line in the literal form `PROXIMA_AC_RESULT key=value ...`, derived from its observed fixtures/payload artifacts; the captured stdout includes it because pytest runs with `-s`. The reader reopens those artifacts and recomputes the reported fields.
- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-10a`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC10a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10a/session -- uv run --project python -m pytest python/tests/test_model_load.py -q -s` | valid_loads=2 malformed_errors=1 rust_validation_errors=1 literal_descriptor_matches=2 rust_loader_matches=2 same_comparator_incumbent_passes=2 comparator=exact_bytes |
| AC10a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-10a --repo-root /private/tmp/proxima-python-card-10a --card-id 10a` | `card=10a functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC10a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-10a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-10a --repo-root /private/tmp/proxima-python-card-10a --card-id 10a` | `card=10a functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
