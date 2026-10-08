# Card 05b: symbolic contraction activation composition

**Owner:** Luna
**Branch:** `codex/python-frontend-card-05b-contraction-activation`
**Worktree:** `/private/tmp/proxima-python-card-05b`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `05a`
**Commit:** `feat(python): compose contraction and activation`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Map `@` and `.silu()` to one composed symbolic expression, and verify its exact graph and inferred shape against literal and Rust constructor controls.

## Changes

Exercise `(x @ w).silu()` and one incompatible contraction shape; compare the complete composed Op sequence and inferred shape with literal and Rust constructor controls. This card proves graph construction only; Card 06b first executes the composition and compares payloads. SiLU is required there; Card 12a adds GELU only.

- The functional test prints exactly one line in the literal form `PROXIMA_AC_RESULT key=value ...`, derived from its observed graph fixtures; the captured stdout includes it because pytest runs with `-s`. The reader reopens those artifacts and recomputes the reported fields.
- Keep all validation evidence, graph fixtures, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-05b`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC05b.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05b/session -- uv run --project python -m pytest python/tests/test_symbolic_matmul.py python/tests/test_silu_graph.py -q -s` | composed_graphs=1 exact_op_matches=1 shape_errors=1 bind_calls=0 evaluate_calls=0 literal_graph_matches=1 rust_incumbent_graph_matches=1 same_comparator_incumbent_passes=1 comparator=exact_bytes |
| AC05b.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05b/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-05b --repo-root /private/tmp/proxima-python-card-05b --card-id 05b` | `card=05b functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC05b.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-05b/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-05b --repo-root /private/tmp/proxima-python-card-05b --card-id 05b` | `card=05b functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
