# speculative-draft-mtp

status: draft
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)

## problem

proxima-model-interop has zero support for llama.cpp's `draft-mtp` speculation type
(`common_speculative_impl_draft_mtp`, `common/speculative.cpp:1331`) -- no NextN/MTP tensor names in
its GGUF binds, no hidden-state readout analogous to `llama_get_embeddings_nextn`
(`src/llama-ext.h:105`), no descriptor for the assistant sidecar family (the GGUF whose
`general.architecture` value is the `LLM_ARCH_GEMMA4_ASSISTANT` name, `src/llama-arch.cpp:60`), and
no local GGUF fixture carries an MTP head (the gemma4-E2B blob used by the parent spec, the batiai
gemma4-26b blob and the qwen3.6-35b-a3b blob all show 0 occurrences of `nextn.eh_proj` in their
tensor directories), so this sub-spec makes `draft-mtp` drafting available as ServingConfig data --
MTP head roots as extra readouts on the target's `ModelDescriptor` (`decode-as-data` R5) for the
single-GGUF family (qwen35, qwen35moe), and a second `ModelDescriptor` driving its own `ServingState`
(`decode-as-data` R6) for the two-GGUF sidecar family -- with the drafted tokens token-identical to
llama.cpp `f1ea20621`'s own `--spec-type draft-mtp` on the same GGUF(s), discharging parent spec R11a
in full and adding no technique-named type (`decode-as-data` AC9, R11).

## refutation condition

Written before evidence: if, after slice 1 obtains real MTP-head GGUFs, the qwen35 MTP forward graph
(`src/models/qwen35.cpp:485-644`) or the sidecar graph (`src/models/gemma4-assis*.cpp:84-196`)
cannot be reproduced token-identically from `ModelDescriptor` fields, a readout list and `Op::Input`
leaves -- i.e. either needs a graph concept (a new `Op`, or a family-keyed branch in the loop) that
the descriptor cannot carry as data -- this sub-spec's architecture section is wrong and must be
redone against the actual graph shape, not patched.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `mtp_head_present(&ParsedGguf) -> bool` reproduces `common_speculative_types_from_gguf`'s non-dflash branch (`common/speculative.cpp:2273-2301`): true iff the `general.architecture` metadata value is not `dflash` and tensor `blk.<block_count-1>.nextn.eh_proj.weight` exists. It positively identifies the single-GGUF trailing-layer family (qwen35, qwen35moe, step35, deepseek, ...) and does NOT positively identify the sidecar family (R9 names why) | yes |
| R2 | the qwen35 profile (`proxima-model-interop/src/profiles/qwen35.toml`) declares the NextN block as data, and `build_forward` lowers it as extra head readout roots, token-identical to `llama_model_qwen35::graph_mtp` (`src/models/qwen35.cpp:485-644`): hnorm(h_prev) concat enorm(embed(token)) -> eh_proj -> one attention+FFN block (QK-norm, sigmoid-gated attention output, RoPE-multi, SwiGLU FFN) -> head_norm -> shared head or fallback lm_head. `h_prev` enters as an `Op::Input` leaf next to the token id | yes |
| R3 | the qwen35moe profile declares the same NextN block with a routed-expert FFN, token-identical to its own `graph_mtp` in `src/models/qwen35moe.cpp` (read from source before implementation, not inferred from R2's shape) | yes |
| R4 | the propose pipe reproduces `common_speculative_impl_draft_mtp::draft` (`common/speculative.cpp:1584-1731`) for the non-chained, non-shared-memory case (`is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` at `:1420` is false for qwen35/qwen35moe, because `ctx_other` is only propagated into `cparams` for the GEMMA4_ASSISTANT/EAGLE3/DFLASH archs, `src/llama-context.cpp:146-162`): top-k=10 draft sampler, per-step `p_min` confidence early exit, growing-KV token append at `pos0 + i + 1`, cross-call `pending_h` carryover for the first token of the next `process()` batch; the accept stage overwrites `pending_h` from `verify_h[min(n_accepted, n_rows-1)]` (`common/speculative.cpp:1733-1747`), composing with the parent spec's R3a/R3b KV rollback (`decode-as-data` commit pipe, prefix-truncate case); and the catch-up decode (`common/speculative.cpp:1475-1582`, guarded by `if (!is_mem_shared)` at `:1508`) pairs each prompt/accepted token with the target's `h_nextn` tap row shifted right by one position, decoded through the NextN readout roots once per prefill batch, caching the last row in `pending_h` for the next propose call | yes |
| R7 | `ServingConfig`'s speculative section (parent R9) accepts the llama type name `draft-mtp` as configuration data (the existing `SpeculativeTypeName::DraftMtp`, `proxima-model-interop/src/speculative_settings.rs:36`) and routes it by data: R1's detection selects the readout-on-target path, and an explicit sidecar-path field selects the second-descriptor path | yes |
| R8 | for the same target (and sidecar, when one is configured), prompt and seed, proxima's generated token ids with `draft-mtp` enabled equal the ids llama.cpp `f1ea20621`'s `llama-speculative-simple` generated and the slice vendored under `proxima-model-interop/tests/fixtures/llama-mtp/`, run with the inference-time flag `--spec-type draft-mtp` (`common/arg.cpp:4243-4249`, registered for `LLAMA_EXAMPLE_SPECULATIVE`/`SERVER`/`CLI`; NOT `--mtp`, which is download-only at `common/arg.cpp:3081-3086`), with `--model-draft`/`-md` (`common/arg.cpp:532-539`) pointing at the sidecar GGUF for the sidecar case and omitted for qwen35/qwen35moe (the `else if (spec_mtp)` branch reuses `model_tgt` for `ctx_dft` when no `-md` is given, `common/speculative.cpp:2557-2564`), for 3 seeds per family; every vendored oracle run must itself have drafted (its recorded `n_drafted` >= 1, which `llama-speculative-simple` prints at `examples/speculative-simple/speculative-simple.cpp:355`), so an oracle that silently drafted nothing cannot pass; parity is measured the same sample-and-match way as parent R1 | yes |
| R9 | the sidecar family is a second `ModelDescriptor` built from GGUF-key reads (the `general.architecture` value equal to the name at `src/llama-arch.cpp:60`, `block_count`, `n_layer_nextn`), bound from the sidecar GGUF: tensors `nextn.pre_projection`/`nextn.post_projection` (global, not per-block, `src/models/gemma4-assis*.cpp:39-56`) plus `n_layer_nextn` full transformer layers with their own norm/attention/FFN weights (`:56-95`), width-validated the way the ctor does (`n_embd == llama_model_n_embd_out(ctx_tgt)`, `common/speculative.cpp:1376-1378`, mirrored by the load-time throw at `src/models/gemma4-assis*.cpp:29-31`) with a typed error, not a panic. No Rust type carries the family's name | yes |
| R10 | the sidecar descriptor's forward is token-identical to `llama_model_gemma4_assistant::graph::graph` (`src/models/gemma4-assis*.cpp:84-196`): it reads the target's own `tok_embd` (`GGML_ASSERT(cparams.ctx_other != nullptr); model_other->tok_embd`, `:107-110`) to embed the drafted token, concatenates it with the incoming hidden state, projects through `nextn_proj_pre`, runs `n_layer_nextn` attention+FFN blocks whose attention reads the target's shared KV (`build_attn_inp_kv_iswa()`, `mem_other` wiring at `src/llama-context.cpp:393`), and projects through `nextn_proj_post` to `t_h_nextn`. The cross-descriptor weight and state sharing is expressed as descriptor data (embedding source and KV source name the target's), resolved when the draft `ServingState` is constructed -- a capability the descriptor and `ServingState` have not needed before, named here rather than assumed solved | yes |
| R11 | the sidecar's propose pipe reproduces the `is_mem_shared` branches: the catch-up-decode skip (`common/speculative.cpp:1507-1508`, "if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode") and the same-position-for-every-draft-token append (`:1697-1706`, "with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens") | yes |
| R12 | the oracle comparator is itself validated: the comparator fed llama's own fixture on both sides passes, and fed one flipped id returns a mismatch | yes |
| R13 | a `draft-mtp` configuration that selects both the readout-on-target path and a sidecar path, or neither, is rejected with a typed validation error rather than silently picking one | yes |

## architecture

Nothing here is a new type. MTP is a readout list on a descriptor (single-GGUF family) or a second
descriptor with its own `ServingState` (sidecar family), driven through the four pipes of
`decode-as-data` (propose, shape, accept, commit). Its row in that spec's table is "extra readout
roots on the target program (nextn block) | chain | sample-and-match | prefix".

- **detection (R1)** is a free function over `&ParsedGguf`, not a method on a type: detection needs
  no bound weights. It returns `true` only for the single-GGUF family. Sidecar identification (R9) is
  a direct read of the `general.architecture` metadata value, because the sidecar's own tensors are
  named `nextn.pre_projection`/`nextn.post_projection` (global), never `blk.N.nextn.eh_proj.weight`
  (per-block). Two data reads, not one generalized one: conflating them would silently misclassify
  either. It lives beside the existing tensor-name lookup (`proxima-model-interop/src/bind.rs:298`
  `find_tensor`, `proxima-gguf/src/parser.rs`); slice 6 decides which crate by reading their
  neighbors first.
- **NextN readout (R2/R3)**: the family profile TOML declares the NextN block (one layer at index
  `n_layer`, one past the trunk's last real layer, `src/models/qwen35.cpp:496-497`, plus `eh_proj`,
  `enorm`, `hnorm`, head norm, shared head). `build_forward` (`proxima-tensor/src/spec/descriptor.rs:863`)
  lowers it as head roots in the readout list (`decode-as-data` R5). The graph takes the previous
  hidden state as an `Op::Input` leaf (`proxima-tensor/src/op.rs:201`), matching
  `llm_graph_input_embd_h` (`src/models/qwen35.cpp:504-514`). No new `Op` variant.
- **hidden-state tap (R4)**: the target's `h_nextn` row per prefill position is a hidden-state tap
  readout at the final layer (`decode-as-data` R5, "hidden-state taps at named layers"), the
  analogue of `llama_get_embeddings_nextn_ith` (`src/llama-ext.h:108`). Main exposes only
  `BoundProgram::hidden_root` and `residual_roots` (`proxima-model-interop/src/lowering.rs:137,141`),
  last-row consumers; the all-positions tap is the readout list, `decode-as-data` slice 3, a
  prerequisite named in TASKS slice 0.
- **sidecar descriptor (R9/R10)** is a second `ModelDescriptor` (`proxima-tensor/src/spec/descriptor.rs:130`)
  loaded by the same path as the existing draft-model support (`proxima-model-interop/src/generate/draft_model.rs:174`,
  `load_draft`) and driven by its own `ServingState` instance. What is new is only that its
  embedding table and its attention KV are the target's: descriptor fields naming the source,
  resolved at construction into a borrowed read-only handle to the target state. The shape (field
  names, how the draft `ServingState` aliases the target's cache layers) is the design slice's
  output (TASKS slice 5), a signature there, not a settled design here.
- **draft loop (R4 single-GGUF, R11 sidecar)** is the propose pipe configured by the
  `draft-mtp` config value; accept is sample-and-match; commit is prefix truncate. The sidecar's
  `is_mem_shared` behaviour (same KV position for every draft token, no catch-up decode) is a branch
  on the descriptor's KV-source field in the propose pipe, not a second pipe, because the accept and
  rollback shape (R4) is identical across both.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| both families in one sub-spec | single-GGUF (R2-R4) and sidecar (R9-R11) both required here | parent R11a requires an audited draft-mtp sub-spec and gemma4 is proxima's primary family, so a sub-spec that produces different drafts than llama.cpp for gemma4 fails R11a's own test |
| MTP as readouts plus descriptor | NextN roots in the readout list; sidecar as a second descriptor | `decode-as-data` R5/R6/R11 supersede a per-family capability method; a name check (`general.architecture` equality in code) is the forbidden shape, a key read feeding a profile is the allowed one |
| detection is two data reads, not one | R1 tensor-name read for single-GGUF, direct `general.architecture` value read for sidecar | the sidecar's tensors are never named `nextn.eh_proj`; one detector covering both would miss the sidecar or bolt a string special case onto a shape check |
| sidecar sharing is descriptor data | embedding source and KV source fields on the draft descriptor | a bespoke program type holding a target handle would be a per-family type (`decode-as-data` R6); the sharing is a property of the descriptor, resolved when the `ServingState` is built (principle 15: a hidden prerequisite found mid-spec is part of the same work, not a footnote) |

## acceptance criteria

`$QWEN` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/qwen-mtp.gguf`, the single-GGUF MTP fixture (qwen35 or qwen35moe checkpoint); slice 1a places the file (or a symlink) at exactly this path
`$GEMMA_MTP` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf`, the sidecar GGUF size-matched to `$GEMMA_TGT` (E2B); slice 1b places the file at exactly this path
`$GEMMA_TGT` = `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd` (proxima's local gemma4-E2B target, already verified to load)
`$LLAMA_SPEC_SIMPLE` = `/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-speculative-simple`, an absolute binary path (no `cd`); it registers `--spec-type` and `--model-draft` for `LLAMA_EXAMPLE_SPECULATIVE` (`common/arg.cpp:4246`) and prints `n_drafted = N` and `n_accept = N` (`examples/speculative-simple/speculative-simple.cpp:355-356`)

| id | discharges | command | expected |
|---|---|---|---|
| AC1a | R1 | `strings -a "$QWEN" \| grep -c 'nextn.eh_proj'` | count >= 1 |
| AC1b | R9 | `strings -a "$GEMMA_MTP" \| grep -c 'nextn.pre_projection'` | count >= 1 (the sidecar's own tensor name, not `eh_proj`) |
| AC2a | R8 | `$LLAMA_SPEC_SIMPLE --model "$QWEN" --spec-type draft-mtp -p "test" -n 8 2>&1 \| grep -cE 'n_drafted = [1-9]'` | 1 (one `n_drafted = N` line with N >= 1; 0 is RED) |
| AC2b | R8 | `$LLAMA_SPEC_SIMPLE --model "$GEMMA_TGT" --model-draft "$GEMMA_MTP" --spec-type draft-mtp -p "test" -n 8 2>&1 \| grep -cE 'n_drafted = [1-9]'` | 1 |
| AC3 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/mtp_head_present_matches_llama_gguf_detection/)'` | 1 passed; one synthetic GGUF with `blk.N.nextn.eh_proj.weight` (written by `proxima-gguf`'s own writer) asserts `true`, one without asserts `false`; `$QWEN`'s real header prints `real_fixture_detected = true`; `$GEMMA_MTP` asserts `false` |
| AC4 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/qwen35_mtp_readout_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50 (per-position hidden-state to logits rows, fixture generated from llama.cpp's own `h_nextn` tensors) |
| AC5 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/qwen35moe_mtp_readout_matches_llama_fixture/)'` | 1 passed; `cases` >= 50; oracle: the per-position `h_nextn` rows and logits llama.cpp `f1ea20621` computes for the qwen35moe graph (`src/models/qwen35moe.cpp`), dumped by slice 3's method |
| AC6 | R4 (consistency: speculation on vs off in the same engine, parent R1; the llama.cpp oracle for the same requirements is AC11) | `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity -- --drafter draft-mtp "$QWEN" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40 > parity_qwen.txt; grep -c 'identical = true' parity_qwen.txt; grep -cE 'speculative_verify_steps = [1-9]' parity_qwen.txt` | two lines printed: `2` then `2` (greedy and sampled blocks both identical, both with at least one verify step) |
| AC7 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_accepts_draft_mtp/)'` | 2 passed (single-GGUF variant routes to the readout-on-target path, sidecar variant routes to the second-descriptor path) |
| AC15 | R13 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_rejects_ambiguous_draft_mtp/)'` | 2 passed (both selected is a typed error, neither selected is a typed error) |
| AC8 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_sidecar_descriptor_binds/)'` | 2 passed (binds against `$GEMMA_MTP`; width mismatch against a different-width target returns the typed error) |
| AC9 | R10 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_sidecar_readout_matches_llama_fixture/)'` | 1 passed; `cases` >= 50; oracle: the per-position `t_h_nextn` rows llama.cpp `f1ea20621` computes for the sidecar graph (`src/models/gemma4-assis*.cpp:195`), dumped by slice 3's method |
| AC10 | R11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/draft_mtp_is_mem_shared_matches_llama/)'` | 1 passed; oracle: the draft positions and ids llama.cpp `f1ea20621` records for a shared-memory run (`common/speculative.cpp:1697-1706`), vendored by slice 22b |
| AC11 | R4,R8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_llama_parity_qwen_seed_/)'` | 3 passed (seeds 1, 2, 3); each prints `ids_compared = N` with N >= 8, and compares proxima's ids with the ids vendored from `$LLAMA_SPEC_SIMPLE --model "$QWEN" --spec-type draft-mtp --seed <seed> -n 40` (oracle: llama.cpp `f1ea20621`) |
| AC12 | R8,R9,R10,R11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_llama_parity_gemma_seed_/)'` | 3 passed (seeds 1, 2, 3); each prints `ids_compared = N` with N >= 8, against ids vendored from `$LLAMA_SPEC_SIMPLE --model "$GEMMA_TGT" --model-draft "$GEMMA_MTP" --spec-type draft-mtp --seed <seed> -n 40` (oracle: llama.cpp `f1ea20621`) |
| AC17 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/draft_mtp_matches_llama_fixture/) or test(/draft_mtp_process_catchup_matches_llama/)'` | 2 passed; oracle: draft token ids, `pending_h` rows and catch-up pairings dumped from llama.cpp `f1ea20621`'s `common_speculative_impl_draft_mtp` by slice 3's method |
| AC16 | R8 | `for family in qwen gemma; do jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-mtp/$family.json; done` | two lines printed: `3` then `3` (every vendored oracle run drafted, so an oracle that drafted nothing cannot back AC11/AC12) |
| AC13 | R12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_oracle_control_/)'` | 2 passed: `mtp_oracle_control_self_` (the comparator fed llama's fixture ids on both sides passes) and `mtp_oracle_control_perturbed_` (one flipped id returns a mismatch, and the test asserts that rejection) |
| AC14 | R9,R10,R11 (consistency: speculation on vs off in the same engine; the llama.cpp oracle is AC12) | `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity -- --drafter draft-mtp --draft-model "$GEMMA_MTP" "$GEMMA_TGT" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40 > parity_gemma.txt; grep -c 'identical = true' parity_gemma.txt; grep -cE 'speculative_verify_steps = [1-9]' parity_gemma.txt` | two lines printed: `2` then `2` |

## out of scope

- MTP heads on families proxima has never bound at all: glm-dsa, deepseek2/32/4, minimax_01, nemotron-h-moe, qwen3next, gemma3n -- binding those base families is a proxima-model-interop coverage prerequisite outside speculative decode's scope
- `chain_heads` mode (step3.5's `n_mtp_layers > 1` on the non-shared-memory path, `common/speculative.cpp:1421,1608-1616`) -- qwen35 asserts `n_layer_nextn == 1` (`src/models/qwen35.cpp:489`), so chaining never triggers for the two single-GGUF families; the sidecar's multi-layer nextn block (R10) is one graph pass over `n_layer_nextn` layers under shared memory, a different mechanism, and is in scope
- MTP sidecar download/auto-resolution (`opts.download_mtp`, `common/arg.cpp:295-419`, the `--mtp` planner flag) -- proxima's fixtures are local paths; slices 1a/1b name the repos to fetch by hand
- other sidecar sizes (E4B, 26B-A4B, 31B) -- E2B is chosen for size-parity with proxima's local gemma4-E2B target; the other sizes are a fixture swap, not new requirements
- the readout list itself (`decode-as-data` R5, slice 3) and the descriptor-level family profiles -- owned by those specs; this sub-spec consumes them

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no public GGUF conversion preserves `nextn.*` for qwen35/qwen35moe (quantizers drop MTP heads as unused auxiliary weights) | high | slice 1a takes longer than a normal fixture slice | AC1a is the check that catches a silently stripped conversion before any other work starts |
| qwen35moe's MoE FFN inside the NextN block (R3) is not a straight swap of the dense FFN (R2): shared experts, routing or a different `eh_proj` shape | medium | R3 needs its own read-first pass | slice 4 records every divergence from `src/models/qwen35moe.cpp` before R3's implementation slice |
| the readout list has no second-input shape (token id plus `h_prev`) on main | medium | R2/R3 may need an `Op::Input` plumbed through the lowered inputs, a `decode-as-data`-adjacent surface | slice 8 decides it before any qwen-specific graph slice |
| cross-descriptor weight and KV sharing (R10) has no analogue in the current descriptor/`ServingState` pair | high | R9-R11 could need a cache-aliasing capability that ripples into the parent spec's R3a/R3b rollback | named, with a dedicated design slice (TASKS slice 5) before any sidecar implementation slice |
| llama.cpp's stock tools may not expose a per-position `h_nextn` dump for AC4/AC5/AC9 | medium | fixture generation needs a small instrumented llama.cpp build | `cb(cur, "h_nextn", -1)` (`src/models/qwen35.cpp:630`, `src/models/gemma4-assis*.cpp:195`) is already a named tensor on both graphs; slice 3 verifies reachability before assuming a patch |
| the named sidecar GGUFs are third-party conversions (AtomicChat), not an official release | medium | AC11/AC12 would measure the conversion's fidelity | AC2b runs llama.cpp itself against `$GEMMA_MTP` first; if llama.cpp produces garbage the fixture is bad regardless of proxima |

## context

- proxima: `proxima-tensor/src/spec/descriptor.rs:130` (`ModelDescriptor`), `:863` (`build_forward`), `proxima-tensor/src/op.rs:201` (`Op::Input`), `proxima-model-interop/src/lowering.rs:118-141` (`BoundProgram`, `hidden_root`, `residual_roots`), `proxima-model-interop/src/profiles/qwen35.toml`, `proxima-model-interop/src/profiles/qwen35moe.toml`, `proxima-model-interop/src/profiles/gemma4.toml`, `proxima-model-interop/src/generate/draft_model.rs:174` (`load_draft`), `proxima-model-interop/src/generate/drafter.rs`, `proxima-model-interop/src/speculative_settings.rs:36` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205` (`SpeculativeType`), `proxima-model-interop/src/bind.rs:298` (`find_tensor`), `proxima-gguf/src/parser.rs`, `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp:1331-1441` (ctor, `is_mem_shared`/`chain_heads` derivation at `:1420-1421`), `:1457-1473` (`begin`), `:1475-1582` (`process`, catch-up decode, shared-memory skip at `:1507-1508`), `:1584-1731` (`draft`, shared-memory same-position branch at `:1697-1706`), `:1733-1747` (`accept`), `:2273-2301` (`common_speculative_types_from_gguf`), `:2515-2564` (`common_speculative_init_result`), `common/arg.cpp:4243-4249` (`--spec-type`), `common/arg.cpp:532-539` (`--model-draft`), `common/arg.cpp:3081-3086` (`--mtp`, download only), `common/arg.cpp:295-419` (sidecar auto-download), `common/speculative.cpp:33-45` (type name map), `src/llama-ext.h:96-117`, `src/llama-context.cpp:144-162`, `src/llama-context.cpp:393`, `src/llama-hparams.h:67` (`n_layer_nextn`), `src/llama-model.cpp:1325-1326`, `src/llama-arch.cpp:60`, `src/llama-arch.cpp:582-589,969-979` (NextN tensor names), `src/models/qwen35.cpp:485-644` (`graph_mtp`), `src/models/qwen35moe.cpp`, `src/models/gemma4-assis*.cpp:1-60` (tensor loading, width-check throws), `:84-196` (`graph::graph`)
- real MTP-head GGUFs to obtain (named via web search 2026-09-28, not yet downloaded or byte-checked): `google/gemma-4-E2B-it-assistant` (Google's safetensors release); `AtomicChat/gemma-4-E2B-it-assistant-GGUF` on Hugging Face (third-party GGUF, E2B-sized); a Qwen3.5/3.6 GGUF quantization that preserves `nextn.*` (official Qwen release or a bartowski/unsloth static quant, re-checked -- slice 1a's own task)
- local fixtures checked and found without an MTP head (0 occurrences of `nextn.eh_proj` and of `nextn.pre_projection` in each tensor directory): `~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd` (gemma4-E2B, the target), `~/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129` (batiai/gemma4-26b), `~/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` (library/qwen3.6:35b-a3b, `qwen35moe` per its GGUF metadata, without the trained MTP head)
