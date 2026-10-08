# Card 01d: TOML representation and round-trip constraints

**Owner:** Luna
**Base:** freshly fetched `origin/main` after declared dependencies land
**Slice budget:** one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps
**Scope units:** 1
**Timer window:** starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event
**Branch:** `codex/python-frontend-card-01d-toml-representation`
**Worktree:** `/private/tmp/proxima-python-card-01d`
**Dependency:** `01c`
**Commit:** `docs(python): document program representation and round trips`

## Goal

Determine whether Python- and TOML-authored programs converge on the same existing representation and whether that representation can round-trip.

## Changes

Cite the exact Rust/TOML path, produce a literal requirement inventory for findings 9–10, and demonstrate a stale-source citation control.

- Keep all validation evidence, payloads, cassettes, and complete review request/response artifacts under `/private/tmp/proxima-python-frontend/evidence/card-01d`; never store them in the repository.
- Start the monotonic timer before fetch/worktree setup. Capture every command and failure with full argv, cwd, timestamps, stdout/stderr, and environment.
- AC3 captures the post-push reader. The outer capture wrapper seals that event, then its post-seal closure gate recomputes elapsed time from immutable start and terminal records and appends `elapsed_seconds<=1800` only after checking the sealed event.
- The card-specific reader opens source artifacts and derives its output; it does not trust a submitted summary.
- Make one coherent change and one commit in this dedicated worktree, then fast-forward through the integration worktree and push `main` to `origin`.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC01d.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_architecture.py proxima-tensor/specs/python-frontend/ARCHITECTURE.md --findings 9,10` | findings=2 requirement_ids=9,10 cited_findings=2 literal_requirement_matches=2 rust_incumbent_controls=1 same_comparator_incumbent_passes=2 roundtrip_claims=1 stale_citations=0 stale_citation_controls_rejected=1 comparator=exact_text |
| AC01d.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-01d --repo-root /private/tmp/proxima-python-card-01d --card-id 01d` | `card=01d functional=1 accepted=1 hashes_valid=1 generated_repo_artifacts=0` |
| AC01d.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01d/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-01d --repo-root /private/tmp/proxima-python-card-01d --card-id 01d` | `card=01d functional=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit=1 rebase>=1 integration_ff=1 push=1 remote_main_match=1 elapsed_seconds<=1800` |

## Complete when

The three counted outputs and post-seal elapsed closure record are in the external chronological log; the post-push reader opened and checked the referenced artifacts and git state.
