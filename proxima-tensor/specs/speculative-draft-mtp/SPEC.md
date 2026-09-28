# speculative-draft-mtp

status: audited
owner: brian bruggeman
created: 2026-09-28

## problem

proxima-model-interop has zero support for llama.cpp's `draft-mtp` speculation type
(`common_speculative_impl_draft_mtp`, `common/speculative.cpp:1331`) -- no NextN/MTP
tensor names in its GGUF binds (`git grep -c nextn proxima-gguf/src proxima-model-interop/src`
= 0), no hidden-state capability method analogous to `llama_get_embeddings_nextn`
(`src/llama-ext.h:105`), no registered `Architecture` for the GGUF arch string
`gemma4-assistant` (`src/llama-arch.cpp:60`), and no local GGUF fixture carries an MTP
head (checked: the gemma4-E2B blob used by the parent spec, the batiai gemma4-26b blob,
and the qwen3.6-35b-a3b blob all show 0 occurrences of `nextn.eh_proj` in their tensor
directories) -- so this sub-spec makes `draft-mtp` drafting available, token-identical to
llama.cpp's own `--spec-type draft-mtp` on the same GGUF(s), for every architecture
proxima currently binds that ships or pairs with an MTP head: qwen35, qwen35moe (single
GGUF, trailing trained layers), and gemma4 via its `gemma4-assistant` sidecar (two GGUFs,
shared KV and shared embedding weights) -- discharging parent spec R11a in full, not a
qwen-only slice of it.

## refutation condition

Written before evidence: if, after slice 1 obtains real MTP-head GGUFs, the qwen35 MTP
forward graph (`src/models/qwen35.cpp:485-644`) or the gemma4-assistant graph
(`src/models/gemma4-assistant.cpp:84-196`) cannot be reproduced token-identically from
proxima's existing binds within a bounded, nameable set of new capability methods (i.e.
each needs a materially different runtime concept proxima has no hook for and cannot be
given one), this sub-spec's architecture section is wrong and must be redone against the
actual graph shape, not patched.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `mtp_head_present(&ParsedGguf) -> bool` reproduces `common_speculative_types_from_gguf`'s non-dflash branch (`common/speculative.cpp:2273-2301`): true iff `general.architecture != "dflash"` and tensor `blk.<block_count-1>.nextn.eh_proj.weight` exists -- positively identifies the single-GGUF trailing-layer family (qwen35, qwen35moe, step35, deepseek, ...); does NOT positively identify a `gemma4-assistant` sidecar (R9 names why) | yes |
| R2 | qwen35 gains `Architecture::speculative_draft_mtp_program`, building the single-block MTP graph token-identical to `llama_model_qwen35::graph_mtp::graph_mtp` (`src/models/qwen35.cpp:485-644`): hnorm(h_prev) concat enorm(embed(token)) -> eh_proj -> one attention+FFN block (QK-norm, sigmoid-gated attention output, RoPE-multi, SwiGLU FFN) -> head_norm -> shared_head_head or fallback lm_head | yes |
| R3 | qwen35moe gains the same capability, token-identical to its own `graph_mtp` equivalent in `src/models/qwen35moe.cpp` (MoE FFN in place of qwen35's dense FFN; read from source before implementation, not inferred from R2's shape) | yes |
| R4 | draft loop reproduces `common_speculative_impl_draft_mtp::draft` (`common/speculative.cpp:1584-1731`) for the non-chained, non-shared-memory case (qwen35/qwen35moe are neither `chain_heads` nor `is_mem_shared`, per `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` at `:1420` evaluating false for them -- `ctx_other` is only ever propagated into `cparams` for `LLM_ARCH_GEMMA4_ASSISTANT`/`LLM_ARCH_EAGLE3`/`LLM_ARCH_DFLASH`, `src/llama-context.cpp:146-162`): top-k=10 draft sampler, per-step `p_min` confidence early exit, growing-KV token append at `pos0 + i + 1`, cross-call `pending_h` carryover for the first token of the next `process()` batch | yes |
| R5 | `accept()` hidden-state rollback reproduces `common/speculative.cpp:1733-1747`: `pending_h` is overwritten from `verify_h[min(n_accepted, n_rows-1)]`, composing with the parent spec's R3a/R3b KV rollback | yes |
| R6 | `process()` reproduces the catch-up decode at `common/speculative.cpp:1475-1582` for the non-shared case (guarded by `if (!is_mem_shared)` at `:1508`): pairs each prompt/accepted token with the target's `h_nextn` row shifted right by one position, decodes through the MTP block once per prefill batch, and caches the last row in `pending_h` for the next `draft()` call | yes |
| R7 | `ServingConfig`'s speculative section (parent R9) accepts `draft-mtp` as a `Drafter` enum variant that routes to R2/R3 (single-GGUF) or R9/R10 (sidecar) via R1's detection plus an explicit sidecar-path config field, with no ambiguity about which family a given config selects | yes |
| R8 | oracle: llama.cpp `f1ea20621` run with the real inference-time flag `--spec-type draft-mtp` (`common/arg.cpp:4243-4249`, registered for `LLAMA_EXAMPLE_SPECULATIVE`/`SERVER`/`CLI` -- **not** `--mtp`, which is `.set_examples({LLAMA_EXAMPLE_DOWNLOAD})`-only at `common/arg.cpp:3081-3086` and exists solely to tell the model-download planner to also fetch an MTP sidecar, never selecting the type at inference), with `--model-draft`/`-md` (`common/arg.cpp:532-539`) pointing at the sidecar GGUF for the gemma4-assistant case and omitted for qwen35/qwen35moe (`common_speculative_init_result`'s `else if (spec_mtp)` branch reuses `model_tgt` for `ctx_dft` when no `-md` is given, `common/speculative.cpp:2557-2564`), is byte-for-byte the reference; parity is measured the same sample-and-match way as parent R1 |
| R9 | proxima registers a new `Architecture`, `name() == "gemma4-assistant"` matching the GGUF `general.architecture` string (`src/llama-arch.cpp:60`), bound from a *second* GGUF file (the sidecar): tensors `nextn.pre_projection`/`nextn.post_projection` (global, not per-block -- `src/models/gemma4-assistant.cpp:39-56`) plus `n_layer_nextn` full transformer layers with their own `attn_norm`/`wq`/`wo`/ffn weights (`:56-95`, distinct from qwen35's single eh_proj block), width-validated the way the ctor does (`n_embd == llama_model_n_embd_out(ctx_tgt)`, `common/speculative.cpp:1376-1378`, mirrored by the GGUF-load-time throw at `src/models/gemma4-assistant.cpp:29-31`) | yes |
| R10 | gemma4-assistant's forward graph is token-identical to `llama_model_gemma4_assistant::graph::graph` (`src/models/gemma4-assistant.cpp:84-196`): reads the **target's own** `tok_embd` tensor directly (`GGML_ASSERT(cparams.ctx_other != nullptr); model_other->tok_embd`, `:107-110`) to embed the drafted token, concatenates it with the incoming hidden state, projects through `nextn_proj_pre`, runs `n_layer_nextn` attention+FFN blocks whose attention reads the **target's shared KV memory** (`build_attn_inp_kv_iswa()`, `mem_other` wiring at `src/llama-context.cpp:393`), and projects through `nextn_proj_post` to `t_h_nextn` -- this is cross-program weight AND state sharing, a capability proxima's `Architecture`/`BoundProgram`/`LayerCache` trio has never needed before (named explicitly, not assumed solved) | yes |
| R11 | gemma4-assistant's draft loop reproduces the `is_mem_shared` branches: `process()`'s catch-up-decode skip (`:1507-1508`, "if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode"), and `draft()`'s same-position-for-every-draft-token append (`common_speculative_impl_draft_mtp::draft`'s `else if (is_mem_shared)` branch, `:1697-1706`, "with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens") | yes |

## architecture

Two families sharing one draft-loop shape and one capability-method pattern (the parent
spec's precedent, `proxima-model-interop/src/architecture.rs:292-309`):

- **detection** (R1) lives in `proxima-model-interop/src/architecture.rs` or
  `proxima-gguf` (whichever already owns tensor-name lookup by convention -- read
  `ParsedGguf::tensor_data_range`'s neighbors before choosing) as a free function over
  `&ParsedGguf`, not a method on `Architecture` -- detection needs no bound weights. It
  only ever returns `true` for the single-GGUF family; gemma4-assistant identification
  (R9) is a direct `general.architecture == "gemma4-assistant"` string match instead,
  because its own tensors are named `nextn.pre_projection`/`nextn.post_projection`
  (global), never `blk.N.nextn.eh_proj.weight` (per-block) -- the two families need two
  distinct detection paths, not one generalized one, and conflating them would silently
  misclassify either.
- **single-GGUF MTP forward program** (R2/R3) is a new `Architecture` capability method,
  `speculative_draft_mtp_program`, mirroring `speculative_verify_program`'s shape:
  default `Ok(None)`, overridden by qwen35 and qwen35moe. Its `BoundProgram` differs from
  the normal decode program in inputs (takes the *previous hidden state* `h_prev` as an
  extra tensor input alongside the token id, matching `llm_graph_input_embd_h` at
  `src/models/qwen35.cpp:504-514`) and in weights (reads `layer.nextn.*` instead of the
  trunk's per-layer weights, at block index `n_layer()`, one past the trunk's last real
  layer -- `src/models/qwen35.cpp:496-497`).
- **hidden-state exposure**: R6's catch-up decode needs the *target* model's `h_nextn`
  row per prefill position, the proxima analogue of `llama_get_embeddings_nextn_ith`
  (`src/llama-ext.h:108`). This does not exist on proxima's `BoundProgram` today (grep
  confirms zero `logits_root`-adjacent hidden-state output); it is this sub-spec's own
  parse/bind gap, named here rather than assumed solved by the parent spec.
- **sidecar MTP program** (R9/R10) is a second, structurally different capability: a
  *second, independently loaded* GGUF whose `Architecture` impl needs read access to two
  things the first program owns -- the target's `tok_embd` weight tensor and the target's
  live per-layer KV state. Proxima's `Architecture`/`BoundProgram` trait pair has no
  concept of "a program bound against another program's already-loaded tensors and
  cache" -- every existing capability method (`bind`, `speculative_verify_program`)
  builds a self-contained `BoundProgram` from its own file's weights. This is the gap R9
  and R10 name rather than paper over: the shape is likely a new field on `BoundProgram`
  (a borrowed handle to the target's `LayerCache` plus a borrowed reference to its
  `tok_embd` tensor) threaded through a `speculative_draft_mtp_sidecar_program(&self,
  parsed, file_bytes, target: &BoundProgram)` -- a signature, not a settled design; the
  settled design is TASKS.md slice work, not this spec.
- **draft loop** (R4/R5/R6 for single-GGUF; R11 for sidecar) is a new `Drafter` enum
  variant, `DraftMtp`, added to the parent spec's closed `enum Drafter` -- box-free,
  matches the parent's dispatch pattern. The sidecar case's `is_mem_shared` behaviour
  (same KV position for every draft token, no catch-up decode) is a match arm inside the
  same variant's implementation, gated on which forward program (R2/R3 vs R9/R10) the
  loaded config selected -- not a second `Drafter` variant, because the accept/rollback
  shape (R5) is identical across both.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| both families in one sub-spec | qwen35/qwen35moe (R2-R6) and gemma4-assistant (R9-R11) are both required here | the parent spec's R11a says an audited sub-spec exists for draft-mtp and draft correctness is that sub-spec's requirement -- gemma4 is proxima's primary architecture, so a sub-spec that produces different drafts than llama.cpp for gemma4 specifically fails R11a's own test, not just this sub-spec's |
| MTP program is a capability method, not a name check | `speculative_draft_mtp_program` (single-GGUF), a distinct sidecar-shaped method (R9/R10) | parent spec's own precedent (`architecture.rs:292-309`'s doc comment explicitly rejects `name() == "gemma4"` checks) |
| detection is two paths, not one | R1 (tensor-name sniff) for single-GGUF, direct arch-string match for sidecar | gemma4-assistant's tensors are never named `nextn.eh_proj`; a single detector that tried to cover both would either miss gemma4-assistant or need a arch-string special case bolted onto a tensor-shape check, which is what a second path already is, done honestly |
| sidecar's target-tensor/target-KV access is a named new capability, not assumed away | `BoundProgram` gains a borrowed-handle shape (exact type TBD in TASKS.md) | principle 15: a hidden prerequisite discovered mid-spec is part of the same body of work, not a footnote; principle 1 (RISC reuse) says look for an existing primitive first, but grep confirms none exists -- this is a genuine gap, not an unexamined one |

## acceptance criteria

`$QWEN` = the single-GGUF MTP fixture slice 1 obtains (qwen35 or qwen35moe checkpoint)
`$GEMMA_TGT` = `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd` (proxima's existing local gemma4-E2B target, already verified to load)
`$GEMMA_MTP` = the gemma4-assistant sidecar GGUF slice 1b obtains, size-matched to `$GEMMA_TGT` (E2B)
`$LLAMA_SPEC` = `cd /Users/brianbruggeman/repos/others/llama.cpp && ./build/bin/llama-cli --spec-type draft-mtp -no-cnv` (model/draft
flags and `-n` are appended before the trailing `-p <prompt>`, never after it -- `-p`/`--prompt`
takes exactly one value argument, `common/arg.cpp:1767`, so anything placed after `-p` in a
command line is consumed as, or trails, the prompt text rather than being parsed as its own flag)

| id | discharges | command | expected |
|---|---|---|---|
| AC1a | R1 | `strings -a "$QWEN" \| grep -c 'nextn.eh_proj'` | count ≥ 1 |
| AC1b | R9 | `strings -a "$GEMMA_MTP" \| grep -c 'nextn.pre_projection'` | count ≥ 1 (gemma4-assistant's own tensor name, not `eh_proj` -- R1's architecture note) |
| AC2a | R8 | `$LLAMA_SPEC --model "$QWEN" -p "test" -n 8` | exit 0; stderr's speculative-decode summary reports `n_drafted > 0` |
| AC2b | R8 | `$LLAMA_SPEC --model "$GEMMA_TGT" --model-draft "$GEMMA_MTP" -p "test" -n 8` | exit 0; stderr's speculative-decode summary reports `n_drafted > 0` |
| AC3 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/mtp_head_present_matches_llama_gguf_detection/)'` | 1 passed; test constructs one synthetic GGUF with `blk.N.nextn.eh_proj.weight` (via `proxima-gguf`'s own writer, per principle 9's canonical-encoder allowance) and one without, asserts `true`/`false` respectively, and asserts against `$QWEN`'s real header too: prints `real_fixture_detected = true`; also asserts `false` against `$GEMMA_MTP` (R1's documented non-detection of the sidecar family) |
| AC4 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/qwen35_mtp_program_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N ≥ 50 (per-position hidden-state -> logits parity rows, fixture generated from `$LLAMA_SPEC` internal state dump) |
| AC5 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/qwen35moe_mtp_program_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 50 |
| AC6 | R4,R5,R6 | `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity -- --drafter draft-mtp "$QWEN" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40` | exit 0; both config blocks (greedy; sampled) print `identical = true`, `speculative_verify_steps` ≥ 1 |
| AC7 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_accepts_draft_mtp/)'` | 2 passed (single-GGUF variant, sidecar variant) |
| AC8 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/gemma4_assistant_registers_and_binds/)'` | 1 passed |
| AC9 | R10 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/gemma4_assistant_mtp_program_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 50 |
| AC10 | R11 | `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity -- --drafter draft-mtp --draft-model "$GEMMA_MTP" "$GEMMA_TGT" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40` | exit 0; both config blocks `identical = true`, `speculative_verify_steps` ≥ 1 |
| AC11 | R8 | `for seed in 1 2 3; do $LLAMA_SPEC --model "$QWEN" --seed $seed -n 40 -p "Repeat exactly five times: the quick brown fox jumps over the lazy dog." > /tmp/llama_qwen_$seed.txt; cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity -- --drafter draft-mtp --seed $seed "$QWEN" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40 > /tmp/proxima_qwen_$seed.txt; diff /tmp/llama_qwen_$seed.txt /tmp/proxima_qwen_$seed.txt; done` | 3 runs, each `diff` prints nothing (0 lines) |
| AC12 | R8 | same as AC11 with `--model "$GEMMA_TGT" --model-draft "$GEMMA_MTP"` (llama side) and `--draft-model "$GEMMA_MTP" "$GEMMA_TGT"` (proxima side) | 3 runs, each `diff` prints nothing (0 lines) |

## out of scope

- MTP heads on architectures proxima has never bound at all: glm-dsa, deepseek2/32/4,
  minimax_01, nemotron-h-moe, qwen3next, gemma3n -- binding these base architectures is a
  proxima-model-interop coverage prerequisite entirely outside speculative decode's scope
- `chain_heads` mode (step3.5's `n_mtp_layers > 1` on the *non*-shared-memory path,
  `common/speculative.cpp:1421,1608-1616`) -- qwen35 asserts `n_layer_nextn == 1`
  (`src/models/qwen35.cpp:489`), so chaining never triggers for the two single-GGUF
  architectures this sub-spec covers; gemma4-assistant's own multi-layer nextn block
  (R10) is a different mechanism (one graph pass over `n_layer_nextn` layers under
  shared memory, not per-step re-dispatch across separate heads) and is in scope
- MTP draft-model download/auto-resolution (`opts.download_mtp`, `common/arg.cpp:295-419`,
  the `--mtp` download-planner flag itself) -- proxima's fixtures are local paths (parent
  spec's own pattern: `$E2B` is a local blob path, never an HF URL); slice 1/1b name the
  exact repos to fetch by hand instead
- other Gemma4-assistant sizes (E4B, 26B-A4B, 31B) -- E2B is chosen for exact size-parity
  with proxima's existing local gemma4-E2B target fixture; the other sizes are the same
  architecture and would be a fixture-swap, not new requirements, if ever needed

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no public GGUF conversion pipeline preserves `nextn.*` tensors by default for qwen35/qwen35moe (many quantizers drop MTP heads as "unused" auxiliary weights) | high | slice 1 takes longer than a normal fixture-acquisition slice | slice 1's validation command (AC1a) is exactly the check that would catch a silently-stripped conversion before any other work starts |
| qwen35moe's MoE FFN inside the MTP block (R3) is not a straight swap of qwen35's dense FFN (R2) -- shared experts, routing, or a different eh_proj shape | medium | R3 needs its own read-first pass, cannot be copy-pasted from R2 | R3's requirement text already says "read from source before implementation, not inferred from R2's shape" |
| proxima's `BoundProgram` has no second-tensor-input shape (token id + hidden state) anywhere today | medium | R2/R3 may need a `BoundProgram` field addition, which is parent-spec-adjacent surface, not sub-spec-local | flagged in architecture's "hidden-state exposure" paragraph; resolved in slice ordering (TASKS.md puts the `BoundProgram` shape question before the qwen35-specific graph) |
| gemma4-assistant's cross-program tensor/KV sharing (R10) has no analogue anywhere in proxima's current architecture -- this is the largest unknown in the whole sub-spec | high | R9/R10/R11 could need a `BoundProgram`/`LayerCache` capability that ripples into the parent spec's own R3a/R3b rollback machinery | named explicitly rather than estimated away; TASKS.md puts a dedicated read-and-design slice (no code) before any gemma4-assistant implementation slice, mirroring how the qwen35moe divergence-read slice precedes its implementation |
| llama.cpp's own oracle path may not expose a clean per-position hidden-state dump for AC4/AC5/AC9's fixture generators | medium | fixture generation needs a small instrumented llama.cpp build (a debug print or a `GGML_TENSOR_FLAG_OUTPUT` on `h_nextn`) rather than stock `llama-cli` | `cb(cur, "h_nextn", -1)` (`src/models/qwen35.cpp:630`, `src/models/gemma4-assistant.cpp:195`) is already a named, `cb`-tagged tensor on both graphs -- llama.cpp's existing `--verbose-prompt`/tensor-dump instrumentation can likely reach it without a source patch; verify before assuming a patched build is needed |
| the two named HF repos for gemma4-assistant (below) are third-party GGUF conversions (AtomicChat), not an official Google GGUF release, and could diverge from the safetensors reference `google/gemma-4-E2B-it-assistant` in quantization or conversion bugs | medium | oracle parity (AC11/AC12) would then be measuring the third-party conversion's fidelity, not the architecture's | AC2b's end-to-end llama.cpp run is the check: if llama.cpp itself produces garbage output against `$GEMMA_MTP`, the fixture is bad regardless of what proxima does, and slice ordering catches this before any proxima work starts |

## context

- proxima (worktree `feat/speculative-decode`): `proxima-model-interop/src/architecture.rs:266-309` (`Architecture` trait, `speculative_verify_program`), `proxima-model-interop/src/architecture.rs:526-535` (registry: qwen35, qwen35moe, gemma4, dense -- no `gemma4-assistant` entry), `proxima-model-interop/src/gemma4/bind.rs:1105-1115` (the one existing capability-method override, reference shape), `proxima-gguf/src/pipe.rs:25-32` (`ParsedGguf`), `proxima-gguf/src/parser.rs:80,449-479` (tensor-name dedup, the place to add MTP tensor lookups)
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp:1331-1441` (ctor, `is_mem_shared`/`chain_heads` derivation at `:1420-1421`), `:1457-1473` (`begin`), `:1475-1582` (`process`, catch-up decode, shared-memory skip at `:1507-1508`), `:1584-1731` (`draft`, shared-memory same-position branch at `:1697-1706`), `:1733-1747` (`accept`), `:2273-2301` (`common_speculative_types_from_gguf`), `:2515-2564` (`common_speculative_init_result` ctor -- self-contained vs sidecar `ctx_dft` construction), `common/arg.cpp:4243-4249` (`--spec-type`, the real inference-time flag), `common/arg.cpp:532-539` (`--model-draft`/`-md`/`--spec-draft-model`), `common/arg.cpp:3081-3086` (`--mtp`, download-planner only -- **not** the inference flag), `common/arg.cpp:295-419` (sidecar auto-download, out of scope), `common/speculative.cpp:33-45` (`common_speculative_type_from_name_map`, confirms the string token is `"draft-mtp"`), `src/llama-ext.h:96-117` (`llama_set/get_embeddings_nextn*`, `llama_set_nextn_layer_offset`, `llama_get_ctx_other` -- none in the public `include/llama.h`), `src/llama-context.cpp:144-162` (`ctx_other` is only ever propagated for `GEMMA4_ASSISTANT`/`EAGLE3`/`DFLASH` -- proof `is_mem_shared` is false for qwen35/qwen35moe), `src/llama-context.cpp:393` (`mem_other` wiring into the memory module), `src/llama-hparams.h:67` (`n_layer_nextn`), `src/llama-model.cpp:1325-1326` (`LLM_KV_NEXTN_PREDICT_LAYERS` read), `src/llama-arch.cpp:60` (`LLM_ARCH_GEMMA4_ASSISTANT` -> `"gemma4-assistant"`), `src/llama-arch.cpp:582-589,969-979` (NextN tensor name table), `src/models/qwen35.cpp:485-644` (`graph_mtp`, the full forward graph), `src/models/qwen35moe.cpp` (its own `graph_mtp`, not yet read -- R3's own prerequisite), `src/models/gemma4-assistant.cpp:1-60` (tensor loading, width-check throws), `:84-196` (`graph::graph`, cross-program `tok_embd`/KV reads)
- real MTP-head GGUFs to obtain (verified via web search 2026-09-28, not yet downloaded or byte-checked): `google/gemma-4-E2B-it-assistant` (Google's own safetensors release, per Google's MTP documentation and the Gemma 4 technical report, arXiv:2607.02770); `AtomicChat/gemma-4-E2B-it-assistant-GGUF` on Hugging Face (a third-party GGUF conversion, E2B-sized to match proxima's existing local target); a Qwen3.5/3.6 GGUF quantization that preserves `nextn.*` (candidate source: the official Qwen GGUF release or a bartowski/unsloth static quant, re-checked -- unverified as of this spec, named as slice 1's own task, not assumed to exist)
- local fixtures checked and found without an MTP head (0 occurrences of `nextn.eh_proj` in each tensor directory, and separately confirmed 0 occurrences of `nextn.pre_projection`): `~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd` (gemma4-E2B, the parent spec's `$E2B`, this sub-spec's `$GEMMA_TGT` -- it is the *target*, correctly headless of its own MTP tensors since those live in the separate assistant sidecar), `~/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129` (batiai/gemma4-26b), `~/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` (library/qwen3.6:35b-a3b, architecture confirmed `qwen35moe` via its own GGUF metadata -- the right architecture family, just a checkpoint without the trained MTP head)
