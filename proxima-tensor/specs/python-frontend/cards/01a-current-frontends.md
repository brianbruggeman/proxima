# Card 01a: current frontends and declarative models

**Owner:** Luna
**Branch:** `codex/python-frontend-card-01a-current-frontends`
**Worktree:** `/private/tmp/proxima-python-card-01a`
**Base:** freshly fetched `origin/main` after the declared dependencies land
**Dependency:** `00l`
**Commit:** `feat(python): current frontends and declarative models`
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event

## Goal

Report only the repository-facing frontends and declarative representation from original deliverable 1. Verify Rust builders, TOML/config, CLI, ModelDescriptor, Settings, bon builders, and block specifications from this tree.

## Changes

Inspect frontend entry points and declarative model sources; mark observed fact, proposal, and unknown separately. Do not claim the later pipeline paths in this slice.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-01a`; never store them in the repository.
- Capture commands from inspection through edits, tests, commit, rebase, integration fast-forward, push, and finalization in this card's chronological session log. Preserve failures and retries.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and records `elapsed_seconds<=1800` only after checking the sealed event.
- Extend the registered evidence reader for this exact ID; the finalizer opens source artifacts and derives its output rather than trusting a submitted summary.
- Make one coherent change and one commit in this card's dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC01a.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_architecture.py proxima-tensor/specs/python-frontend/ARCHITECTURE.md --findings 1` | findings=1 requirement_ids=1 cited_findings=1 literal_requirement_matches=1 incumbent_report_controls=1 same_comparator_incumbent_passes=1 stale_citations=0 unresolved_findings=0 comparator=exact_text |
| AC01a.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-01a --repo-root /private/tmp/proxima-python-card-01a --card-id 01a` | `card=01a functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC01a.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01a/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-01a --repo-root /private/tmp/proxima-python-card-01a --card-id 01a` | `card=01a functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
