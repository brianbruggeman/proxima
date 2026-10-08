# Proxima Tensor Python frontend

status: draft
owner: brian
created: 2026-10-08

## problem

This work adds a Python frontend for tensor developers and coding agents to author, inspect, bind, and execute Proxima programs through the existing Rust `Vec<Op>` and bind/Omega owners, measured by the counted graph, bound-operation, payload, and no-per-operation-call criteria across all 47 cards.

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
| R12 | One concise, copyable Python authoring journey demonstrates the final public surface from symbolic construction through graph inspection and whole-program execution to Python-readable output; its expression path submits one program, not individual operations. | yes |

### developer-facing API target

This is the usability bar for the final public example, not a commitment to names before Card 01f verifies the Rust boundary:

```python
import proxima as px

values = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
weights = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
x = px.input("x", shape=(2, 4))
w = px.constant(weights)
y = (x @ w).silu().softmax(-1)
print(y.shape)
print(y.graph())
result = y.run({"x": values})
print(result.numpy())
```

The published walkthrough should read naturally when pasted into a REPL, and the same snippet should be executable from the package. Card 01f settles the exact names and supported semantics; Card 13d proves the complete journey without replacing it with a hand-authored descriptor or per-operation execution calls.

## architecture

The frontend owns Python syntax and symbolic handles. A narrow Rust boundary owns graph construction and hands the existing `Vec<Op>` plus output `NodeId` values to Proxima's existing shape, bind, and evaluation APIs. The bound program remains Rust-owned. A materialized result owns or borrows storage according to the selected evaluator and exposes only conversions whose lifetime and copy behavior are explicit.

The six architecture-report slices 01a–01f are the first frontend work, after the separate plan revision commits and Cards 00a–00l install and validate the evidence tooling. They must update this section and later cards if current-main evidence changes these boundaries. The current source inspection provides a starting hypothesis, not permission to skip that report.

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

Cards execute in order, one at a time. Before starting each card, fetch `origin/main`, verify the spec/card package exists there, and create that card's named branch and worktree from the fetched `origin/main`. Do not use the dirty checkout that initiated this plan. Each card makes one coherent behavior or evidence change and one commit. Update that card's precommit status and the resume line in `TASKS.md` in the same commit after its functional checks and precommit evidence check pass. That status records precommit validation only. The card AC3 stage reader checks the capability installed so far and records the literal landing command transcripts; Card 00l/AC10 later validates actual git state for all closed bundles after Card 00k installs the git reader.

Cards 00a–00l install the evidence workflow in dependency order, with one invariant and one negative control per card. Each post-push reader checks only capabilities already landed: 00a checks the bootstrap transcript and shell event ledger; 00b the reusable session launcher; 00c normal command streams; 00d spawn/signal outcomes; 00e containment; 00f bundle completeness; 00g typed artifacts; 00h the read-only five-root history chain; 00i allowlists/precommit; 00j repository inventory; 00k git landing; 00l plan-wide dispatch. Their staged AC3 readers validate only the capability landed so far and open the complete captured literal commit, rebase, integration-fast-forward, push, and remote-query argv/stdout/stderr records; they do not call these captured strings proof that landing succeeded. Card 00l and AC10 reopen every bundle after Card 00k adds actual git-state readers, then validate each successful landing event against worktree HEAD and origin/main. The capture wrapper seals each AC3 event before a post-seal closure gate checks and records the timer interval; the in-flight finalizer cannot attest to its own unsealed terminal event. Their bounded responsibilities are: 00a bootstrap transcript and shell event ledger; 00b reusable card-session launcher; 00c normal command streams; 00d spawn/signal outcomes; 00e external roots and cassette containment; 00f immutable bundle hashes and tail detection; 00g typed artifacts and executable comparators; 00h the read-only five-root history chain; 00i functional allowlists and precommit; 00j repository artifact inventory; 00k commit/rebase/integration fast-forward/push proof; and 00l card-ID reader dispatch and forged-plan rejection. The reader registry accepts `00a`–`00l`, `01a`–`01f`, and `02a`–`13d` as literal IDs and fails closed until a card registers its reader. Each of the 47 cards declares one concern, starts its timer before fetch/worktree creation, and records actual elapsed time through the post-seal closure gate against a 30-minute cap, one dedicated worktree, one commit, one captured precommit, and one captured post-push stage reader. Card 00a starts with one continuous external bootstrap transcript and a command-level timestamped event ledger; Card 00b adds the reusable card-session launcher; Card 00h pins and validates the read-only pre-split Card00 sidecar index SHA-256 `7ae94687864ed583c29bd9a4224c29cf1629702cf7eb9cbf0534e8dae14ba2fb`; cards 00c–00l build the recorder, integrity checks, precommit, inventory, landing, and plan reader in narrow slices. Keep `/private/tmp/proxima-python-frontend/evidence/card-00/` read-only: it is historical discovery evidence and never contributes events to a new card acceptance count. Card 00h verifies its five source roots without mutation and counts only the four historical roles for the unique totals. After Card 00l, execute frontend slices 01a–13d sequentially from freshly fetched main. For every card, capture every executed command from exploration through edits, builds, tests, retries, commit, rebase, fast-forward, push, and finalization: exact commands and arguments, working directory, start/end timestamps, exit status, full stdout and stderr, relevant environment overrides, and hashes of every payload and cassette. Make source and fixture changes through recorded commands so their full inputs and outputs are present. Save complete Sol/spec-auditor review requests and responses as external event records, not summaries. Keep those artifacts outside the repository at `/private/tmp/proxima-python-frontend/evidence/card-<exact-card-id>/`. A checker must open the files and verify required counts and hashes; a summary written by the agent is not the artifact. Preserve failed attempts in the same chronological log. Never replace a failed run with a reconstructed transcript. Every post-push stage-reader invocation is captured and its own terminal record is checked after sealing; full stdout/stderr and all prior failures remain in the external chronological log.

At card start, run `git fetch origin main`, then use the branch and worktree from its card header with `git worktree add -b <branch> <worktree> origin/main`. Keep a separate clean integration worktree checked out detached at `origin/main`; after card validation and commit, fetch and rebase the card branch, fast-forward that integration worktree to the card head, and push `HEAD:main` to `origin`. Record those git commands in the card's full log as well.

Before commit, run all functional acceptance commands and the captured `check_evidence.py --precommit`; it checks the evidence capabilities registered by cards completed so far and does not claim a later-stage inventory check prematurely. Card 00j adds complete generated-artifact inventory; card 00k adds ordered landing proof. The captured post-push reader checks only capabilities registered at its point in the sequence, and card 00l supplies read-only closed-card/family reopening and the plan-wide reader reruns each card reader after all capabilities exist. Then commit the card, rebase the card branch onto the latest `origin/main`, fast-forward the dedicated integration worktree, and push the resulting main head to `origin`. Run the card's captured AC3 stage-reader command after push. Cards 00c–00j validate only their installed capability and require captured literal landing command transcripts; they defer interpreting git state. Cards 00k onward use the registered landing reader. Card 00l/AC10 reopens all bundles and validates ordered commit/rebase/fast-forward/push/remote records against worktree HEAD and origin/main. If the remote advances, fetch and rebase again. The request authorizes this per-card merge and push workflow. Do not force-push. Retain each worktree and evidence bundle through its review; remove a worktree only after its card is merged and its evidence is checked.

For cassette-backed cases, record and replay using Proxima's cassette handler and an external cassette root. Assert that replay returns the same event payloads and that a missing or corrupted cassette fails. Pure computation fixtures must compare their complete result payloads with an independent literal oracle; they must not be presented as cassette replay.

## acceptance criteria

Each card below owns its acceptance criteria and evidence bundle. Spec-level AC00–AC02 verify the 47-card set, evidence coverage, and execution order before implementation begins. AC03–AC09 and AC11–AC12 reopen the corresponding frontend cards' sealed finalizer artifacts through the read-only closed-card/family verifier; AC10 is the sole plan-wide finalizer run after Card 13d and reads the separate external plan bundle.

| id | discharges | command | expected |
|---|---|---|---|
| AC00 | R8, R11 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend` | `cards=47 unique_worktrees=47 unique_branches=47 cards_with_acceptance=47 card_acceptance_rows=141 total_acceptance_rows=142 precommit_checks=47 postpush_stage_checks=47 plan_finalizers=1 missing_fields=0` |
| AC01 | R8, R9 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend --check-evidence-plan` | `cards=47 captured_commands=142 precommit_checks=47 postpush_stage_checks=47 plan_finalizers=1 external_evidence_paths=47 repository_evidence_paths=0 missing_fields=0` |
| AC02 | R8, R11 | `python3 proxima-tensor/specs/python-frontend/scripts/check_cards.py proxima-tensor/specs/python-frontend --check-card-order` | `card_order=47_ordered_cards evidence_tools_before_frontend=1 architecture_before_implementation=1 cpu_slice_before_omega=1 model_load_after_omega=1 core_sugar_before_extended_sugar=1` |
| AC03 | R1 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-01a /private/tmp/proxima-python-frontend/evidence/card-01b /private/tmp/proxima-python-frontend/evidence/card-01c /private/tmp/proxima-python-frontend/evidence/card-01d /private/tmp/proxima-python-frontend/evidence/card-01e /private/tmp/proxima-python-frontend/evidence/card-01f` | `findings=17 cited_findings=17 stale_citations=0 unresolved_findings=0 literal_requirement_matches=17 incumbent_report_controls=1 incumbent_report_passes=1 stale_citation_controls_rejected=1` |
| AC04 | R2, R4, R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-06a /private/tmp/proxima-python-frontend/evidence/card-06b` | `op_records=1 boundop_records=1 output_payloads=1 invalid_controls=1 boundary_calls=1 per_op_calls=0 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1` |
| AC05 | R3, R6 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-11a /private/tmp/proxima-python-frontend/evidence/card-11b /private/tmp/proxima-python-frontend/evidence/card-11c` | `numpy_payloads=5 dlpack_cases=3 dlpack_outcome_records=3 owner_records=8 numpy_copy_records=5 unresolved_outcomes=0 literal_oracle_matches=5 rust_incumbent_outputs_pass_same_comparator=1` |
| AC06 | R5 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-07a /private/tmp/proxima-python-frontend/evidence/card-07b` | `roundtrips=3 unsupported_errors=2 op_equalities=3 literal_oracle_matches=3 rust_incumbent_outputs_pass_same_comparator=1` |
| AC07 | R5, R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-10a /private/tmp/proxima-python-frontend/evidence/card-10b` | `load_payloads=3 forward_payloads=1 literal_oracle_matches=1 rust_incumbent_outputs_pass_same_comparator=1` |
| AC08 | R7 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-09a /private/tmp/proxima-python-frontend/evidence/card-09b` | `metal_supported=1 metal_plans=1 metal_payloads=2 cpu_oracle_payloads=1 skips=0 literal_oracle_matches=2 rust_incumbent_outputs_pass_same_comparator=2` |
| AC09 | R9 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-02a /private/tmp/proxima-python-frontend/evidence/card-02b` | `record_events=1 replay_events=1 payload_matches=1 literal_event_matches=1 incumbent_default_root_output_passes_same_comparator=1 corrupted_rejections=1 cassette_path_external=1` |
| AC10 | R8, R9 | `python3 proxima-tensor/specs/python-frontend/scripts/capture_validation.py --root /private/tmp/proxima-python-frontend/evidence/plan/session -- python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --finalize-plan /private/tmp/proxima-python-frontend/evidence/plan --cards proxima-tensor/specs/python-frontend` | `cards=47 worktrees=47 commits=47 rebases>=47 pushes=47 checked_bundles=47 missing_full_logs=0 repository_artifacts=0` |
| AC11 | R10 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-12a /private/tmp/proxima-python-frontend/evidence/card-12b /private/tmp/proxima-python-frontend/evidence/card-12c /private/tmp/proxima-python-frontend/evidence/card-12d` | `core_sugar_cases=8 exact_op_matches=8 unsupported_errors=0 literal_oracle_matches=8 rust_incumbent_outputs_pass_same_comparator=8` |
| AC12 | R10, R12 | `python3 proxima-tensor/specs/python-frontend/scripts/check_evidence.py --verify-closed-family /private/tmp/proxima-python-frontend/evidence/card-13a /private/tmp/proxima-python-frontend/evidence/card-13b /private/tmp/proxima-python-frontend/evidence/card-13c /private/tmp/proxima-python-frontend/evidence/card-13d` | `extended_sugar_cases=6 exact_op_matches=6 unsupported_index_errors=1 authoring_journey_examples=1 end_to_end_graph_matches=1 end_to_end_payload_matches=1 whole_program_calls=1 per_operation_calls=0 literal_oracle_matches=7 rust_incumbent_outputs_pass_same_comparator=7` |

The `check_cards.py` criteria validate the exact 47-card ID set and dependency order, one declared scope unit per card, unique worktrees and branches, 141 card acceptance rows plus one plan-wide criterion, precommit/finalizer coverage, external evidence paths, and execution order. Runtime criteria are discharged by post-push stage readers, which open the capability artifacts; Card 00l/AC10 later opens actual git state and the complete bundle, source citation, operation/bound-operation, ownership, cassette, git-event, and full-log artifacts. Tensor behavior correctness criteria check an independent literal oracle and an incumbent Rust path through the same comparison; tooling-only and architecture-report criteria record these fields as not applicable. A finalizer that cannot inspect required records fails; its printed counts are not substitutes for the artifacts. Cards 00a–00l establish and validate the evidence workflow; Frontend slices 01a–11c prove architecture, construction, the vertical slice, program lifecycle, device/model paths, and ownership. Slices 12a–13d add one named sugar family or authoring example each. The vertical slice alone is not the ergonomic-surface completion criterion.

## out of scope

- Reproducing PyTorch's full API surface.
- Training/autograd APIs, unless existing Proxima semantics make them part of the first report's supported scope.
- Claiming zero-copy NumPy or DLPack behavior before a measured ownership and lifetime proof.
- Adding a Python scheduler, backend policy layer, optimizer, or parallel tensor IR.
- Requiring a cassette for pure local arithmetic that has no cassette-supported I/O boundary.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| current main has moved beyond the inspected checkout | high | cards target stale paths or APIs | Cards 01a–01f refresh every source citation from its clean worktree based on fetched `origin/main`. |
| existing IR does not preserve all Python-authored constructs in TOML | medium | serialization would silently lose semantics | Card 07b proves round trips on the supported subset and returns named errors for unsupported forms. |
| a materialized evaluator owns copied host outputs | high | misleading zero-copy API promises | Cards 06a, 06b, 11a, 11b, and 11c record copies and ownership; Card 11c treats DLPack as a separate capability decision. |
| cassette captures only transport events | high | tests could claim a replay guarantee they do not exercise | Cards 02a and 02b prove path selection and record/replay on the actual cassette seam and keeps tensor payload assertions separate. |
| Metal availability differs by runner | medium | device AC cannot run on every host | Card 09b records an explicit hardware capability preflight and requires actual execution. |

## context

Source inspection and the Sol review of the current `codex/preserve-main-work-20261003` checkout found:

- `proxima-tensor/src/lib.rs:1-20` describes the stored program as `Vec<Op>` and says there is no separate graph arena.
- `proxima-tensor/src/spec/descriptor.rs:594-615` builds a model-specific forward program from `ModelDescriptor`.
- `proxima-tensor/src/spec/program_spec.rs:9-18,68-115,184-202` defines TOML `ProgramSpec`/`NodeSpec` and converts the program into `Vec<Op>`.
- `proxima-tensor/src/op.rs:150-163,191-281` defines the current reduction and operation variants. `Reduce::keep` is not the general home for argmax/scatter/contraction; verify each requested mapping against current source.
- `proxima-tensor/src/spec/tests.rs:850-932` represents matrix multiplication as elementwise multiplication followed by an additive reduction and compares TOML-built operations with Rust-built operations. It does not prove serialization back to TOML.
- `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:16-23,73-98` exposes the existing bind path.
- `proxima-tensor/src/bind/types_layout_boundop.rs:126-146` defines `BoundOp` and its kind.
- `proxima-tensor/src/cpu/arena.rs:20-24,74-85` shows `Evaluated` owning `Vec<f32>` output buffers and exposing borrowed slices.
- `omega/src/metal/execute_and_hazards.rs:111-129` sends execution through the existing Omega plan path.
- `proxima-test/src/harness.rs:35-57` selects cassette mode and derives its cassette path relative to the package manifest. The card must add an external-root option before generated cassettes can satisfy this spec's artifact-location rule.
- `Cargo.toml` and tensor manifests had no PyO3 dependency in the inspected checkout. Cards 01a and 01e must recheck the fetched main tree, target matrix, feature policy, and Python packaging toolchain.

Memory MCP tools were not exposed to the planning session. Repository-resident `ai_docs/AGENT.md`, `ai_docs/index.jsonl`, `ai_docs/task-routes.jsonl`, and `ai_docs/invariants.jsonl` were consulted. This card set adds the Python frontend route/index entry; Card 01f verifies exactly one record of each and refreshes them if the architecture investigation changes the path.
