# shared recurrent forward graph

status: draft
owner: Sol implementation; root integration; Luna admission
created: 2026-10-03

## problem

Extend the existing descriptor-driven TwoRange graph to express repeated physical
layers with separate logical KV state and explicit normalization/readout choices:
the two-physical-layer, two-iteration fixture must emit four logical cache owners
and two physical weight sets; three frozen ordinary-model Op/root payloads must
remain byte-identical; irrational literal output error is bounded by the existing
absolute-plus-relative 0.000001 tolerance. Both routes use the current helpers.

## refutation condition

A separate transformer body, changed ordinary-model Op graph, shared recurrent KV
by accident, extra normalization, incorrect guidance boundary, or a failing literal
negative control refutes the affected requirement.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| G1 | Existing recurrence-disabled descriptors emit the same Op vectors and roots as their current builders. | yes |
| G2 | Recurrence uses physical weight identity and separate logical cache/routing identity; checked dimensions reject invalid schedules before construction. | yes |
| G3 | Existing Q/K norms remain the default for their current builders; an explicit absent norm passes raw projected Q/K to existing RoPE. | yes |
| G4 | Each configured loop boundary applies the shared final norm exactly once and feeds its normalized state into the next pass. | yes |
| G5 | Fixed/adaptive logit and fixed-hidden readouts use the existing paper arithmetic and distinct representation boundaries. | yes |
| G6 | The same helpers evaluate full-prefix and cached continuation with complete literal cache/output payloads. | yes |

## architecture

The authoritative data and helper contracts are WORK-PACKET.md sections 2, 3 and
5, with these bindings:

- Add `RecurrenceConfig { iterations: u32, normalize_each_iteration: bool }`,
  `LoopCdMode`, and `LoopCdConfig` to the existing descriptor surface. Defaults
  are no recurrence and disabled guidance with zero strength/reference one.
- Add `qk_norm: bool` to LayerAttentionConfig and preserve every current caller's
  actual prior behavior. Make projected norm weights optional in current helpers.
- `ForwardBuild` contains `forward: BuildForwardProgram` and
  `readouts: Option<RecurrentReadoutRoots>`; retain build_forward's public tuple
  return by projecting from build_forward_with_readouts. Do not duplicate bodies.
- Reuse the TwoRange builder. Its positional entry remains a compatibility
  wrapper; one descriptor-based implementation emits all nodes.
- Physical schedule count stays `block_count`. Logical count is checked
  `block_count * iterations`. A logical visit's physical index is remainder;
  history and routing identities use the logical index. SharedFromLayer donors
  rebase within the current iteration and keep their existing explicit meaning.
- Reuse Input NodeIds for identical physical weight leaves on the recurrent
  path, checking declared shape/dtype equality. No weight-value copies or one
  mapped device object per recurrence merely due to duplicate input nodes.
- Keep cold graph construction alloc-based; no new runtime, cache or Op variant.
- Before production edits, capture complete Op vectors and every root from the
  current builder for three tiny ordinary TwoRange descriptors: dense, routed
  MoE, and Gemma branch norms with an explicit shared donor. Store these immutable
  text payloads under `results/baseline/` with the capture command and source
  diff identity. The disabled-descriptor case compares to those frozen payloads,
  not solely to the compatibility wrapper. A mutated root or Op must fail that
  comparison. Baseline capture is test support, not a production implementation.
  Native capture command: `CARGO_TARGET_DIR=/private/tmp/proxima-ouro-mainline-target cargo run -p proxima-tensor --example recurrent_graph_baseline --features 'std config' -- proxima-tensor/specs/ouro-mainline/results/baseline`.
  Exactly three artifacts: `dense-two-range.json`, `routed-two-range.json`,
  `gemma-shared-two-range.json`. Each includes all Ops, all tuple roots, fixture
  identity and the descriptor. `create_new` prevents later overwrites.
- Expose raw/normalized reference/terminal roots and optional head roots exactly
  as in the packet. Disabled guidance executes no reference head. Hidden mode
  modifies final readout only, not already-built cache roots.
- The packet's fixed-hidden literal is `[-3/sqrt(65),11/sqrt(65)]`.
- C2 diagnostic capture also exposes optional `guided_raw` and
  `adaptive_strength` roots in the existing `RecurrentReadoutRoots` record.
  These are the already-emitted fixed-hidden contrast and per-row adaptive
  strength nodes. No additional Op or generation readback is introduced. The
  existing remap translates them with the other diagnostic roots; the lab
  requests and reads them instead of substituting CPU-derived intermediates.
  Existing hidden/adaptive cases assert their literal payloads, retaining the
  10-case selection and all frozen ordinary graph expectations.
- Reject zero depth, logical-count overflow, invalid reference iteration,
  nonfinite/negative strength, unused disabled strength, unsupported recurrent
  ShortConv and recurrent PLE before emitting a partially usable graph.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| Base implementation | Existing TwoRange builder and attention/FFN helpers | Copying the lab graph would repeat the original split. |
| Norm absence | Optional norm operand | Gamma one still normalizes the input. |
| Experimental switch | Recurrence absent by default; existing kernel compile gates remain unchanged | This adapts graph composition and adds no new performance primitive/kernel. |
| Layer identity | Explicit logical and physical indices passed to existing helpers | Weight aliasing must not alias KV or routed-site state. |
| Adaptive top two | Existing reduction algebra with equal maxima accounted for | Filtering distinct values gives the wrong margin on ties. |

## acceptance criteria

Use native commands, no scripts. Runs are delegated; preserve raw logs and actual
vectors. New sync literal tests may use plain #[test], consistent with the file.
Use absolute+relative 0.000001 for irrational tiny fixtures, inherited from
loopcd-panel/ORACLE-SPEC.md; integer fixtures are exact. No tolerance widening.

| id | discharges | command | expected |
|---|---|---|---|
| GAC1 | G1-G6 | `cargo nextest run -p proxima-tensor --features std --lib -E 'test(recurrent_shared_)' --no-capture` | Exactly 10 named cases from WORK-PACKET.md section 6, all run, zero skipped; every literal vector retained. |
| GAC2 | G1,G3 | `cargo nextest run -p proxima-tensor --lib -E 'test(build_forward_matches_direct_builder_call_at_real_mistral_dims) \| test(build_forward_matches_direct_builder_call_at_real_qwen2_dims)' --no-capture` | Exactly 2 existing graph-comparison cases run; compare actual Op vectors and root identities. |
| GAC3 | G1,G2 | `cargo check -p proxima-tensor --no-default-features --features alloc --message-format=json` | Exactly 1 proxima_tensor library artifact, 0 compiler error diagnostics, build-finished success true. |
| GAC4 | G1-G6 | `cargo clippy -p proxima-tensor --lib --all-targets` | 0 compiler/clippy errors; retain the complete log and actual built targets. |

The disabled-descriptor case requires all three complete frozen graph/root
payload comparisons and a deliberately mutated graph control. Capture precedes
product mutation; an empty or regenerated candidate baseline is inadmissible.

The ten names are recurrent_shared_disabled_descriptor_same_graph,
recurrent_shared_schedule_two_indices, recurrent_shared_qk_norm_none_direct_projection,
recurrent_shared_sandwich_norm_boundaries, recurrent_shared_norm_each_iteration_once,
recurrent_shared_distinct_kv_literal, recurrent_shared_cached_vs_prefill_literal,
recurrent_shared_loopcd_logits_hidden_literal, recurrent_shared_adaptive_ties_literal,
and recurrent_shared_logical_moe_and_shared_donors. Their individual payloads and
negative controls are prescribed in WORK-PACKET.md sections 5 and 6.

## out of scope

- This component does not implement another executor, sampler, checkpoint reader
  or backend kernel. The integration task consumes it through the existing loader.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| Adding config fields breaks external literals | certain | Compile failures | Update every existing initializer to its previous behavior; compile affected callers. |
| Shared KeySource change reaches sibling builders | certain | Semantic drift | Some preserves the exact current operation chain; assert graph equality. |
| Hidden/readout roots get confused | concrete | Different model | Separate raw/normalized roots and wrong-boundary fixture. |
| Recurrent routed sites/donors use physical ids | concrete | Wrong routing/cache | Literal logical site/donor assertions. |

## context

- WORK-PACKET.md sections 2, 3, 5 and 6.
- Existing `spec/{descriptor,attention_forward,lfm2_single_range_cached}.rs`.
- `loopcd-panel/{ORACLE-SPEC,PAPER-METHODS-SPEC}.md` for registered controls.
- Source ownership: Sol owns tensor spec code/tests and initializer adjustments
  necessary for these config additions. Root owns the umbrella spec and later
  integration; interop implementation begins after these signatures compile.
