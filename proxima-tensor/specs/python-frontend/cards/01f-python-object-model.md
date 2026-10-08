# Card 01f: Python object model and API design

**Owner:** Luna
**Base:** freshly fetched `origin/main` after declared dependencies land
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Branch:** `codex/python-frontend-card-01f-python-object-model`
**Worktree:** `/private/tmp/proxima-python-card-01f`
**Dependency:** `01e`
**Commit:** `docs(python): define Python object model and API`

## Goal

Propose symbolic/materialized/program objects, a minimal human-facing API, implementation phases, and semantic risks; assemble the complete 17-item architecture report.

## Changes

Cite every proposal to preceding source investigations or current Rust behavior. The reader checks findings 14–17 and the complete 17-finding inventory, citations, literal requirements, incumbent report, and stale-citation negative control.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-01f`; never store them in the repository.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and appends `elapsed_seconds<=1800` only after checking the sealed event.
- The card-specific reader opens source artifacts and derives its output; it does not trust a submitted summary.
- Make one coherent change and one commit in this dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC01f.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01f/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_architecture.py proxima-tensor/specs/python-frontend/ARCHITECTURE.md --findings 14,15,16,17 --require-complete-report` | findings=4 requirement_ids=14,15,16,17 cited_findings=4 full_report_findings=17 full_report_citations=17 literal_requirement_matches=17 incumbent_report_controls=1 incumbent_report_passes=1 same_comparator_incumbent_passes=17 comparator=exact_text stale_citations=0 unresolved_findings=0 stale_citation_controls_rejected=1 |
| AC01f.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01f/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-01f --repo-root /private/tmp/proxima-python-card-01f --card-id 01f` | `card=01f functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC01f.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01f/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-01f --repo-root /private/tmp/proxima-python-card-01f --card-id 01f` | `card=01f functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
