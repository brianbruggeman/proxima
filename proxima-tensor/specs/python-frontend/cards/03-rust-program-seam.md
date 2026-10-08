# Card 03: Rust symbolic program seam

**Owner:** Luna
**Branch:** `codex/python-frontend-card-03-rust-program-seam`
**Worktree:** `/private/tmp/proxima-python-card-03`
**Base:** current `origin/main` after Card 02 lands
**Dependency:** Cards 01 and 02
**Commit:** `feat(tensor): expose missing python construction capability`

## Goal

Close only a concrete source-cited construction gap that Card 01 found in the existing `Vec<Op>` and `NodeId` APIs. This card does not add PyO3 or execute a program.

## Changes

- Card 01 must identify an exact existing API gap, call site, and why the missing capability belongs in `proxima-tensor` rather than in the thin PyO3 wrapper. If no such gap is evidenced, make this a test-only card asserting the existing constructors' complete graph behavior; do not add a Rust type or API.
- If a gap is evidenced, add only the narrow function(s) needed, using the existing `Vec<Op>` and `NodeId`; do not create a builder abstraction or typed graph handles without a separately cited ownership need.
- Map each supported operation to exact existing Op variants. Build matmul from the existing multiply/reduce representation only if current source requires it.
- Use existing shape inference and errors; do not create another IR, inference pass, or optimizer.
- Keep construction and graph inspection free of evaluator/device calls.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC03.1 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03 -- cargo test -p proxima-tensor python_program:: -- --nocapture` | 5 named construction cases passed; each asserts the complete expected `Vec<Op>` sequence and compares an independently written literal plus incumbent Rust constructor output through the same comparator |
| AC03.2 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03 -- cargo test -p proxima-tensor python_program::invalid -- --nocapture` | 2 typed construction/shape errors observed, 0 panics |
| AC03.3 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03 -- cargo test -p proxima-tensor python_program::construction_does_not_evaluate -- --nocapture` | 1 passed; evaluator/device-call counter = 0 |
| AC03.4 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/card-03/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --precommit /private/tmp/proxima-python-frontend/evidence/card-03 --repo-root /private/tmp/proxima-python-card-03 --card-id 03` | `card=03 functional_commands=3 accepted_commands=3 hashes_valid=1 generated_repo_artifacts=0` |
| AC03.5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-03 --repo-root /private/tmp/proxima-python-card-03` | `constructor_cases=5 typed_errors=2 evaluator_calls=0 literal_oracle_matches=5 rust_incumbent_outputs_pass_same_comparator=5 full_log=1 hashes_valid=1 generated_repo_artifacts=0 commit_events=1 rebase_events>=1 push_events=1` |

## Complete when

The complete operation sequence and shape/error payloads are in the checked external evidence bundle; any new public function is limited to the exact gap and ownership need cited in the architecture report.
