# Proxima Tensor Python frontend

status: draft
owner: brian
created: 2026-10-08

## problem

This work adds a Python frontend for tensor developers and coding agents to author, inspect, bind, and execute Proxima programs through the existing Rust `Vec<Op>` and bind/Omega owners, measured by the counted graph, bound-operation, payload, and no-per-operation-call criteria across all 14 cards.

## refutation condition

The design is wrong if the vertical-slice Python expression path executes or schedules each operation through repeated Python-to-Rust calls, or if it creates a second graph representation that cannot be compared with the existing `Vec<Op>` program.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | The architecture report answers all 17 requested investigations using current source citations and names unknowns. | yes |
| R2 | Python-authored expressions lower to Proxima's existing operation representation; no Python scheduler or second execution IR is introduced. | yes |
| R3 | Symbolic graph construction and materialized results have distinct, documented ownership and lifetime semantics. | yes |
| R4 | The first vertical slice imports `proxima`, builds a symbolic program, inspects shape and graph, binds and evaluates the complete graph, and returns a Python-readable result. | yes |
| R5 | Python-authored and TOML-authored programs converge on the same representation; serialization is implemented only for forms the representation can preserve. | yes |
| R6 | NumPy copies, views, strides, dtype, mutability, and lifetime behavior are measured and stated; DLPack is not called zero-copy without ownership evidence. | yes |
| R7 | Omega and existing Rust code retain validation, binding, folding, lowering, placement, and execution policy. | yes |
| R8 | Every implementation card has one dedicated worktree, one coherent commit, counted acceptance criteria, a checked evidence bundle outside the repository, and a rebase/fast-forward/push path. | yes |
| R9 | Validation can be rerun from retained full command transcripts and Proxima cassette replay where the tested boundary performs cassette-supported I/O. | yes |
| R10 | The documented Python surface covers the named core sugar that the inspected Op model can express: activations, softmax, reductions, transpose/indexing, and scalar/augmented assignment semantics. Unsupported forms fail explicitly. | yes |
| R11 | The plan distinguishes the first CPU vertical slice from completion of the ergonomic surface and the production/device lifecycle. | yes |

## architecture

The frontend owns Python syntax and symbolic handles. A narrow Rust boundary owns graph construction and hands the existing `Vec<Op>` plus output `NodeId` values to Proxima's existing shape, bind, and evaluation APIs. The bound program remains Rust-owned. A materialized result owns or borrows storage according to the selected evaluator and exposes only conversions whose lifetime and copy behavior are explicit.

The architecture report is the first product card, after Card 00 installs the evidence tooling. It must update this section and later cards if current-main evidence changes these boundaries. The current source inspection provides a starting hypothesis, not permission to skip that report.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| frontend representation | existing `Vec<Op>` and `NodeId` | `proxima-tensor/src/lib.rs` describes the program as a plain `Vec<Op>`; a Python-only graph would create a second semantic source. |
| model descriptor and TOML convergence | converge at `Vec<Op>` unless the architecture card proves a better shared owner | `ModelDescriptor` builds a model-specific program, while TOML `ProgramSpec` separately converts to `Vec<Op>`; they are not the same descriptor type. |
| eager evaluation | one whole-program Rust bind/evaluate call | per-operation calls would leave execution orchestration in Python. |
| cassette use | use Proxima cassette record/replay for cassette-supported I/O boundaries; use deterministic payload fixtures for pure local tensor evaluation | the existing cassette captures Handler traffic, not arbitrary function calls or tensor math. Claiming it replays computation would misstate the artifact. |
| evidence storage | full validation logs, payloads, cassettes, and check outputs live under `/private/tmp/proxima-python-frontend/evidence/` | generated validation artifacts must not become repository fixtures or be reduced to a terse summary. |
| model route | Sol reviews the architecture boundary; Luna executes bounded cards after that boundary is fixed | this task changes a public frontend and crosses the Rust/Python crate boundary; the route follows `pm/model-routing-cost-first-escalation.md`. |

### card workflow contract

Cards execute in order, one at a time. Before starting each card, fetch `origin/main` and create that card's named branch and worktree from the fetched `origin/main`. Do not use the dirty checkout that initiated this plan. Each card makes one coherent behavior or evidence change and one commit. Update that card's status and the resume lines in `TASKS.md` in the same commit after its functional checks and precommit evidence check pass.

Before Card 00 starts, land this spec/card package and its `ai_docs` route/index records as one docs-only commit on `main`; that commit is the source Card 00 finds in its `origin/main` worktree. Then run each card sequentially from freshly fetched main. For Card 00, start the external terminal transcript before its first worktree inspection with `script -q /private/tmp/proxima-python-card-00-bootstrap.log /bin/zsh`; retain the exact launcher argv and terminal transcript as the bootstrap record. Inside that transcript, create `/private/tmp/proxima-python-frontend/evidence/card-00/` and write the launcher argv into the session manifest. After the shell exits, copy the completed transcript into the evidence bundle using `capture_validation.py`, then continue with captured commands. For every card, capture every executed command from exploration through edits, builds, tests, retries, commit, rebase, fast-forward, push, and finalization: exact commands and arguments, working directory, start/end timestamps, exit status, full stdout and stderr, relevant environment overrides, and hashes of every payload and cassette. Make source and fixture changes through recorded commands so their full inputs and outputs are present. Save complete Sol/spec-auditor review requests and responses as external event records, not summaries. Keep those artifacts outside the repository at `/private/tmp/proxima-python-frontend/evidence/card-NN/`. A checker must open the files and verify required counts and hashes; a summary written by the agent is not the artifact. Preserve failed attempts in the same chronological log. Never replace a failed run with a reconstructed transcript. The terminal finalizer is the only command whose own invocation/output is appended by the capture tool after it checks the preceding log; its complete output is still retained in the external bundle.

At card start, run `git fetch origin main`, then use the branch and worktree from its card header with `git worktree add -b <branch> <worktree> origin/main`. Keep a separate clean integration worktree checked out detached at `origin/main`; after card validation and commit, fetch and rebase the card branch, fast-forward that integration worktree to the card head, and push `HEAD:main` to `origin`. Record those git commands in the card's full log as well.

Before commit, run all functional acceptance commands and `check_evidence.py --precommit`; this gate verifies the required behavior records and repository artifact inventory without requiring git commit/push events. Then commit the card, rebase the card branch onto the latest `origin/main`, fast-forward the dedicated integration worktree, and push the resulting main head to `origin`. Run the card's `--finalize` acceptance command after push; this separate postpush gate checks the full log, hashes, artifact inventory, and commit/rebase/push records. If the remote advances, fetch and rebase again. The request authorizes this per-card merge and push workflow. Do not force-push. Retain each worktree and evidence bundle through its review; remove a worktree only after its card is merged and its evidence is checked.

For cassette-backed cases, record and replay using Proxima's cassette handler and an external cassette root. Assert that replay returns the same event payloads and that a missing or corrupted cassette fails. Pure computation fixtures must compare their complete result payloads with an independent literal oracle; they must not be presented as cassette replay.

## acceptance criteria

Each card below owns its acceptance criteria and evidence bundle. These spec-level checks verify that the card set is executable before implementation begins.

| id | discharges | command | expected |
|---|---|---|---|
| AC00 | R8, R11 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend` | `cards=14 unique_worktrees=14 unique_branches=14 cards_with_acceptance=14 acceptance_rows=62 precommit_checks=14 postpush_finalizers=14 missing_fields=0` |
| AC01 | R8, R9 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend --check-evidence-plan` | `cards=14 captured_commands=48 precommit_checks=14 postpush_finalizers=14 external_evidence_paths=14 repository_evidence_paths=0 missing_fields=0` |
| AC02 | R8, R11 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend --check-card-order` | `card_order=14_ordered_cards architecture_before_implementation=1 cpu_slice_before_omega=1 model_load_after_omega=1 core_sugar_before_extended_sugar=1` |
| AC03 | R1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-01 --repo-root /private/tmp/proxima-python-card-01` | `findings=17 cited_findings=17 stale_citations=0 unresolved_findings=0 literal_requirement_matches=17 incumbent_report_controls=1 incumbent_report_passes=1 stale_citation_controls_rejected=1` |
| AC04 | R2, R4, R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-06 --repo-root /private/tmp/proxima-python-card-06` | `op_records=1 boundop_records=1 output_payloads=1 invalid_controls=1 boundary_calls=1 per_op_calls=0 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1` |
| AC05 | R3, R6 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-11 --repo-root /private/tmp/proxima-python-card-11` | `numpy_payloads=5 dlpack_cases=3 dlpack_outcome_records=3 owner_records=8 numpy_copy_records=5 unresolved_outcomes=0 literal_oracle_matches=5 rust_incumbent_outputs_pass_same_comparator=1` |
| AC06 | R5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-07 --repo-root /private/tmp/proxima-python-card-07` | `roundtrips=3 unsupported_errors=2 op_equalities=3 literal_oracle_matches=3 rust_incumbent_outputs_pass_same_comparator=1` |
| AC07 | R5, R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-10 --repo-root /private/tmp/proxima-python-card-10` | `load_payloads=3 forward_payloads=1 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1` |
| AC08 | R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-09 --repo-root /private/tmp/proxima-python-card-09` | `metal_supported=1 metal_plans=1 metal_payloads=2 cpu_oracle_payloads=1 skips=0 literal_oracle_matches=2 rust_incumbent_outputs_pass_same_comparator=2` |
| AC09 | R9 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-02 --repo-root /private/tmp/proxima-python-card-02` | `record_events=1 replay_events=1 payload_matches=1 literal_event_matches=1 incumbent_default_root_output_passes_same_comparator=1 corrupted_rejections=1 cassette_path_external=1` |
| AC10 | R8, R9 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize-plan /private/tmp/proxima-python-frontend/evidence/plan --cards proxima-tensor/specs/python-frontend` | `cards=14 worktrees=14 commits=14 rebases>=14 pushes=14 checked_bundles=14 missing_full_logs=0 repository_artifacts=0` |
| AC11 | R10 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-12 --repo-root /private/tmp/proxima-python-card-12` | `core_sugar_cases=8 exact_op_matches=8 unsupported_errors=0 literal_oracle_matches=8 rust_incumbent_outputs_pass_same_comparator=8` |
| AC12 | R10 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize /private/tmp/proxima-python-frontend/evidence/card-13 --repo-root /private/tmp/proxima-python-card-13` | `extended_sugar_cases=6 exact_op_matches=6 unsupported_index_errors=1 literal_oracle_matches=6 rust_incumbent_outputs_pass_same_comparator=6` |

The `check_cards.py` criteria validate the card plan's structure only. Runtime criteria are discharged by post-push finalizers, which open actual payload, source citation, operation/bound-operation, ownership, cassette, git-event, and full-log artifacts. Tensor behavior correctness criteria check an independent literal oracle and an incumbent Rust path through the same comparison; tooling-only and architecture-report criteria record these fields as not applicable. A finalizer that cannot inspect required records fails; its printed counts are not substitutes for the artifacts. Cards 00-11 prove the first vertical slice, program lifecycle, device/model paths, and ownership. Cards 12-13 extend that slice across the named core and extended sugar; the vertical slice alone is not the ergonomic-surface completion criterion.

## out of scope

- Reproducing PyTorch's full API surface.
- Training/autograd APIs, unless existing Proxima semantics make them part of the first report's supported scope.
- Claiming zero-copy NumPy or DLPack behavior before a measured ownership and lifetime proof.
- Adding a Python scheduler, backend policy layer, optimizer, or parallel tensor IR.
- Requiring a cassette for pure local arithmetic that has no cassette-supported I/O boundary.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| current main has moved beyond the inspected checkout | high | cards target stale paths or APIs | Card 01 refreshes every source citation from its clean worktree based on fetched `origin/main`. |
| existing IR does not preserve all Python-authored constructs in TOML | medium | serialization would silently lose semantics | Card 07 proves round trips on the supported subset and returns named errors for unsupported forms. |
| a materialized evaluator owns copied host outputs | high | misleading zero-copy API promises | Cards 06 and 11 record copies; Card 11 treats DLPack as a separate ownership decision. |
| cassette captures only transport events | high | tests could claim a replay guarantee they do not exercise | Card 02 proves record/replay on the actual cassette seam and keeps tensor payload assertions separate. |
| Metal availability differs by runner | medium | device AC cannot run on every host | Card 09 records an explicit hardware capability preflight and requires actual execution. |

## context

Source inspection and the Sol review of the current `codex/preserve-main-work-20261003` checkout found:

- `proxima-tensor/src/lib.rs:1-20` describes the stored program as `Vec<Op>` and says there is no separate graph arena.
- `proxima-tensor/src/spec/descriptor.rs:594-615` builds a model-specific forward program from `ModelDescriptor`.
- `proxima-tensor/src/spec/primitives.rs:9-13,59-110,331-447` defines TOML `ProgramSpec`/`NodeSpec` and converts the program into `Vec<Op>`; `program_spec.rs` is a stale prompt path.
- `proxima-tensor/src/op.rs:150-163,191-281` defines the current reduction and operation variants. `Reduce::keep` is not the general home for argmax/scatter/contraction; verify each requested mapping against current source.
- `proxima-tensor/src/spec/tests.rs:850-932` represents matrix multiplication as elementwise multiplication followed by an additive reduction and compares TOML-built operations with Rust-built operations. It does not prove serialization back to TOML.
- `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:16-23,73-98` exposes the existing bind path.
- `proxima-tensor/src/bind/types_layout_boundop.rs:126-146` defines `BoundOp` and its kind.
- `proxima-tensor/src/cpu/arena.rs:20-24,74-85` shows `Evaluated` owning `Vec<f32>` output buffers and exposing borrowed slices.
- `omega/src/metal/execute_and_hazards.rs:111-129` sends execution through the existing Omega plan path.
- `proxima-test/src/harness.rs:35-57` selects cassette mode and currently derives its cassette path under the package manifest's `tests/cassettes` directory. The card must add an external-root option before generated cassettes can satisfy this spec's artifact-location rule.
- `Cargo.toml` and tensor manifests had no PyO3 dependency in the inspected checkout. Card 00 must recheck the fetched main tree, target matrix, feature policy, and Python packaging toolchain.

Memory MCP tools were not exposed to the planning session. Repository-resident `ai_docs/AGENT.md`, `index.jsonl`, `task-routes.jsonl`, and matching invariants were consulted. This card set adds the Python frontend route/index entry; Card 01 verifies exactly one record of each and refreshes them if the architecture investigation changes the path.
