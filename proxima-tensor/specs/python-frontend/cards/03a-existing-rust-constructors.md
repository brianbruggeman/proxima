# Card 03a: Rust construction behavior inventory

**Owner:** Luna
**Branch:** `codex/python-frontend-card-03a-existing-rust-constructors`
**Worktree:** `/private/tmp/proxima-python-card-03a`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `02b`
**Commit:** `feat(python): Rust construction behavior inventory`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Verify the existing Vec<Op>/NodeId constructors already cover Python-required construction before adding a Rust seam.

## Changes

Add named tests for the exact existing operation sequences and shape errors; do not add a new abstraction.

- The Rust test helper prints exactly one `PROXIMA_AC_RESULT key=value ...` line under `--nocapture`; it derives the fields from observed Op/payload/cassette records. The reader reopens those records and recomputes the values.
- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-03a`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC03a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03a/session -- cargo test -p proxima-tensor python_program::existing -- --nocapture` | constructor_cases=3 exact_op_sequences=3 shape_errors=1 evaluator_calls=0 literal_oracle_matches=3 rust_incumbent_outputs_pass_same_comparator=3 same_comparator_incumbent_passes=3 comparator=exact_bytes |
| AC03a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-03a --repo-root /private/tmp/proxima-python-card-03a --card-id 03a` | `card=03a functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC03a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-03a --repo-root /private/tmp/proxima-python-card-03a --card-id 03a` | `card=03a functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
