# Card 01: current architecture report

**Owner:** Luna, with Sol architecture review
**Branch:** `codex/python-frontend-card-01-architecture`
**Worktree:** `/private/tmp/proxima-python-card-01`
**Base:** fetched `origin/main` at card start
**Dependency:** Card 00 and spec-auditor ADMIT
**Commit:** `docs(tensor): map python frontend architecture`

## Goal

Produce the required evidence-backed design report before any Python implementation. Refresh every citation against this card's clean worktree and current `origin/main`. Do not copy the stale paths or behavior claims from the initiating prompt without verification.

## Changes

- Add `proxima-tensor/specs/python-frontend/ARCHITECTURE.md` with exactly 17 numbered findings matching the prompt's 17 required deliverables. Finding 1 covers current frontends and the overall declarative representation; finding 9 covers Python/TOML convergence and the TOML-to-Op path.
- For each finding, cite current `file:line` evidence, separate observed fact from proposal, and mark anything not established as unknown.
- Add `scripts/check_architecture.py` to verify that all 17 findings have content, each has a current `file:line` citation, all cited files and line numbers resolve in this worktree, and no finding remains unmarked as fact, proposal, or unknown.
- Use a literal 17-item investigation oracle derived from the required deliverable list, and include a source-cited incumbent Rust architecture report fixture. Run both through the same report checker: the literal inventory and incumbent report must pass; a deliberately stale citation control must fail.
- Include a concrete vertical-slice path and the actual Op sequence for its matmul/activation example.
- Record the capability route: Sol for boundary review, Luna for source inventory and implementation cards. Explain any later model change with the observed evidence.
- Update `SPEC.md` architecture/context and downstream cards if the evidence changes the starting assumptions.
- Verify and retain exactly one `python-frontend` route in `ai_docs/task-routes.jsonl` and one index entry in `ai_docs/index.jsonl`, pointing future sessions to the spec, card set, and architecture report.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC01.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01 -- python3 proxima-tensor/specs/python-frontend/scripts/check_architecture.py proxima-tensor/specs/python-frontend/ARCHITECTURE.md` | `findings=17 cited_findings=17 stale_citations=0 unresolved_findings=0 literal_requirement_matches=17 incumbent_report_controls=1 incumbent_report_passes=1 stale_citation_controls_rejected=1` |
| AC01.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01 -- jq -s 'reduce .[] as $item (0; if $item.task == "python-frontend" then . + 1 else . end)' ai_docs/task-routes.jsonl` | `1` |
| AC01.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01 -- jq -s 'reduce .[] as $item (0; if $item.id == "proxima.python_frontend" then . + 1 else . end)' ai_docs/index.jsonl` | `1` |
| AC01.4 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-01/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-01 --repo-root /private/tmp/proxima-python-card-01 --card-id 01` | `card=01 functional_commands=3 accepted_commands=3 hashes_valid=1 generated_repo_artifacts=0` |
| AC01.5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-01 --repo-root /private/tmp/proxima-python-card-01` | after push: `findings=17 cited_findings=17 stale_citations=0 unresolved_findings=0 route_records=1 index_records=1 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

All commands, including failures and the report checker, are captured with Card 00's external evidence tool. Sol's review is retained as a complete review artifact and checked for a disposition on every finding. The card's external bundle contains the unabridged session, structured command manifest, report, source-citation check, Sol review, and hashes; no generated log, payload, or review capture is committed.

## Complete when

The architecture report is committed with current citations, Sol's boundary review is recorded, its evidence bundle passes the checker, and the branch is rebased, fast-forwarded to main, and pushed to origin.
