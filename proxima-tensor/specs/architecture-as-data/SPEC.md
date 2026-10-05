---
status: admitted
---

# architecture as data

## problem

Owner, 2026-10-04: "gemma4 should not have special logic. it should be a configuration that
is lowered." Owner, 2026-09-25: "allow for any architecture without requiring any specific or
special logic."

Problem: on main 9dd9deef the AC4 name search over the four crates' non-test source prints
1214 (run by the spec auditor, 2026-10-04), and this spec drives it to 0 outside the
family-profile file while llama.cpp token parity holds on every real checkpoint.

## refutation condition

This is the wrong thing to build if either of these is observed:

- a slice cannot keep a real checkpoint's token ids equal to llama.cpp's without
  reintroducing a name-keyed branch. That would make the difference behaviour, not data.
- a value that differs between two architectures cannot be expressed by any descriptor or
  profile field.

Either one becomes a recorded irreducible (see below) or retracts the invariant.

## context

- Descriptor and lowering: `proxima-tensor/src/spec/descriptor.rs` (`ModelDescriptor`,
  `CacheStrategy`, `build_forward` :538).
- Architecture trait and registry: `proxima-model-interop/src/architecture.rs` (trait :282,
  `with_builtin` :612).
- Per-architecture modules:
  - `proxima-model-interop/src/gemma4/{bind,hparams,program}.rs`
  - `proxima-model-interop/src/qwen35.rs`
  - `proxima-model-interop/src/qwen35moe/*.rs`
  - `proxima-model-interop/src/dense.rs`
  - `proxima-model-interop/src/lfm2.rs`
- Dense weight binding: `proxima-model-interop/src/bind.rs`.
- Decode loop: `proxima-model-interop/src/generate/decode.rs`.
- Serving config: `proxima-model-interop/src/serving.rs`.
- Tokenizer dispatch: `proxima-tokenizer/src/gguf.rs`.
- Task classifier: `proxima-model-interop/src/task.rs`.
- Existing llama oracle data:
  - `proxima-model-interop/examples/data/gemma4_ring_llama_ids.txt` (66 ids, llama-server
    f1ea20621);
  - `proxima-tokenizer/tests/fixtures/llama-gemma4-tokenize/` (15 fixtures).

## evidence (read-only audit of main 9dd9deef, 2026-10-04; rows marked * re-read at the line)

| # | site | violation |
|---|---|---|
| V1 | `proxima-model-interop/src/qwen35.rs:600`, `proxima-model-interop/src/qwen35moe/program.rs:218`, `proxima-model-interop/src/lfm2.rs:655` | bespoke whole-model builders; `spec::LayerKind` has no Gdn/Ssm variant |
| V2 | `proxima-tensor/src/spec/descriptor.rs:538` | three cache engines selected by code |
| V3 | `proxima-model-interop/src/gemma4/bind.rs:233`, `qwen35.rs:281`, `qwen35moe/bind.rs:122`, `lfm2.rs:367`, `bind.rs` | five per-architecture weight-name tables |
| V4* | `proxima-model-interop/src/gemma4/bind.rs:1176-1195` | SWA RoPE table hard-coded `1.0e4, 256` |
| V5 | `proxima-model-interop/src/generate/decode.rs:2018,2020,3321` | `ExpertResidency<40, 256>` in the generic loop |
| V6* | `proxima-model-interop/src/gemma4/bind.rs:1146-1162` | verify program for gemma4 only; trait default `Ok(None)` |
| V7 | `proxima-tensor/src/spec/descriptor.rs:104`, `proxima-model-interop/src/architecture.rs:345` | window and KV ring reachable only via gemma4 |
| V8* | `proxima-model-interop/src/bind.rs:619` | `force_split_half_rope: architecture == "qwen2"` |
| V9 | `proxima-model-interop/src/gemma4/bind.rs:717-878` | activation, scales, value_norm, norm shift, gating as Rust constants |
| V10 | `proxima-model-interop/src/architecture.rs:282-511` | capability methods return per-arch constants |
| V11 | `proxima-model-interop/src/serving.rs:914-953` | 7 `qwen35moe_*` fields + `gdn_prefill_backend` |
| V12 | `proxima-tensor/src/spec/descriptor.rs:122-466` | test-only `gemma4_descriptor`/`mistral_descriptor` in the library |
| V13* | `proxima-tokenizer/src/gguf.rs:91` | `"gpt2" \| "gemma4"` arm |
| V14 | `proxima-tokenizer/src/gguf.rs:17` | `tokenizer.ggml.pre` never read |
| V15 | `proxima-model-interop/src/task.rs:124-132` | substring name list decides `CausalGeneration` |

Irreducible by construction: the GDN recurrence, MoE routing and the pre-gather residency
protocol are op-level primitives. The descriptor names them; it does not encode their math.
Profile values GGUF does not carry are data in the profile file.

## requirements

- R1. Every architecture-dependent numeric or enum value in a lowered program comes from the
  descriptor (GGUF metadata over the family profile).
- R2. There is no `Architecture` concept: no trait, impl, registry or per-family type. Owner: "fsm makes it composable. architecture means that we've fucked up and we just hide it." A per-family type is a place to hide special logic; composition through the FSM and config leaves nowhere to hide it. Owner, 2026-10-04: "I did not
  authorize a trait here ... it needs to be fsm based w/ a conflaguration driving it."
  - The model is a conflaguration config: `ModelDescriptor` derives `Settings` and `Validate`.
    Its layers are, in order: family-profile TOML defaults, then GGUF metadata, then env.
  - It has a fluent builder that round-trips with the config.
  - Serving is the sans-IO `ServingState` FSM, today in `proxima-model-interop/src/serving_fsm.rs`
    and moving to `proxima-core/src/serving_state.rs` (generic over its entry) by
    fsm-techniques slice 1
    (Prefill, Decode, Verify, Accept, Rollback, Finish), driven by that config. It replaces the
    closure in `run_decode_loop_from_ids`.
  - Which states are reachable comes from config fields. For example, Verify is reachable only
    when the descriptor's layers can rewind and the speculative config is non-empty. It never
    comes from a type that implements a trait.
- R3. A speculative verify program is derived for every family whose layers can rewind.
- R4. A sliding window and KV ring apply to exactly the layers whose descriptor entry has a
  window.
- R5. Weights bind by walking the lowered program's `Op::Input` leaf names against the GGUF
  tensor directory.
- R6. No architecture name in non-test source outside the family-profile file and its loader.
- R7. The op graph at real dims is byte-identical to the incumbent at 9dd9deef wherever a
  slice claims no graph change.
- R8. Token ids on every real checkpoint equal llama.cpp f1ea20621's greedy ids.
- R10. Owner, 2026-10-04: "we still need that performance, so we may need to make changes to what
  we have to keep the performance we've already obtained."
  - Decode ms/token and prefill time on every measured model stay within run-to-run noise of
    the baseline captured at main 9f0647da, or improve.
  - Config cost lands at load time only. The lowered program, kernel choice and dispatch count
    must be what a hand-specialized path would produce.
  - Where config-driven lowering would cost speed, the change is to the lowering, never a
    per-model fast path. A specialization the config enables (a fused kernel, a placed
    single-range KV path) is admitted by the shape of the op, not by model name.
- R9. Owner, 2026-10-04: "I need our models to be 100% conflaguration driven. like the full graph,
  etc. before we lower into a hardware backend needs to be 100% programmable through
  conflaguration x fsm x sans-io."
  - Everything that determines the pre-lowering program is fields of one serializable config:
    - the layer schedule: which mixer per layer (attention, GDN, shortconv), its heads, window,
      KV sharing and RoPE table;
    - the FFN (dense, routed, shared expert, activation, gating);
    - norms and scales;
    - embedding and head;
    - the cache layout and its mask;
    - the verify shape;
    - the step inputs (the RoPE tables).
  - The config composes compiled primitives. A new model variant is a config file, with zero new
    Rust.
  - GGUF metadata and the family profile only populate that config; they are layers, not code
    paths.
  - Lowering is sans-IO and pure: config plus weight directory in, op graph out. No file or
    device IO.
  - The config type and lowering compile at proxima-tensor's no_std+alloc tier, with serde.
    conflaguration `Settings`/`Validate` and the layered loader sit at the std composition
    boundary, per guiding principle 4's layering caveat.
  - The `ServingState` FSM drives execution from that config (R2).

## acceptance criteria

Two kinds of check, labelled as such:
- **ORACLE**: llama.cpp f1ea20621 artifacts, sharing no code with proxima, all vendored by
  slice 0 under `proxima-model-interop/tests/fixtures/llama-parity/`:
  - greedy ids per checkpoint (llama-server, temperature 0, top_k 1);
  - `gguf_kv.txt` per checkpoint (llama.cpp's `gguf-dump` key/value listing);
  - `swa_layers.txt` per windowed checkpoint (the per-layer SWA flags llama.cpp logs at load).
- **CONSISTENCY**: the incumbent at 9dd9deef (op-graph digests, bound names and bytes). This
  shows a refactor slice did not change the program. It is never the correctness oracle.

Control: whether 9dd9deef passes the same check, stated per AC. A capability AC must FAIL at
9dd9deef; that proves it discriminates.

| AC | discharges | kind | command | expected | control at 9dd9deef |
|---|---|---|---|---|---|
| AC0 | R7 | consistency | `cargo nextest run -p proxima-model-interop --features std -E 'test(/arch_data_digest_/)'` | 8 passed (gemma4 26B 13314 ops, logits root NodeId(13313), gemma4 E2B, openchat, qwen2, qwen3, qwen35, qwen35moe, granite moe) | 8 passed |
| AC1 | R1 | oracle | `cargo nextest run -p proxima-model-interop --features std -E 'test(/swa_rope_from_metadata/)'` | 2 passed: E2B and 26B tables equal the `rope.freq_base_swa`/`rope.dimension_count_swa` in `gguf_kv.txt` | 2 passed only if the files hold 1e4/256 (the hard-coded values); slice 0 records which, and a file with other values makes the control FAIL |
| AC2 | R3, R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/generic_verify_llama_parity_/)'` | 3 passed: gemma4 E2B, openchat and qwen3 with speculation on equal their llama ids | 1 passed, 2 failed (openchat and qwen3 have no verify program) |
| AC3 | R5 | consistency | `cargo nextest run -p proxima-model-interop --features std -E 'test(/generic_binder_/)'` | 7 passed: bound names and tensor-byte digest equal the incumbent's | n/a: the generic binder does not exist at 9dd9deef. Slice 0 asserts the incumbent's capture is non-empty: 7 passed |
| AC4 | R6 | consistency | `git grep -nIiP '\b(gemma4\|qwen35moe\|qwen35\|qwen2\|lfm2\|mistral\|llama)\b' -- proxima-model-interop/src proxima-tensor/src omega/src proxima-tokenizer/src ':!*tests*' ':!*profiles*' \| wc -l` | 0 | 1214 |
| AC5 | R2 | consistency | `git grep -nP '\b(trait\|impl\|struct\|enum)\b[^;{]*Architecture' -- proxima-model-interop/src proxima-tensor/src \| wc -l`, then `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/descriptor_config_parity_\|serving_fsm_drives_/)'` | 0; then 9 passed: one descriptor config-vs-builder round trip per checkpoint (7), plus 2 FSM tests (a plain decode and a speculative verify-accept-rollback run that the live generate path routes through `ServingState`) | first command prints 17 (trait, registry, 4 family impls, the test fake, and the per-family `Architecture`/`*Architecture` hparams structs); second: tests absent |
| AC6 | R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/llama_parity_/)'` | 7 passed, one per checkpoint; 8 passed from slice 9 (lfm2 added) | 3 passed, 4 failed at ac4eb2c7 (measured 2026-10-04); 4 passed (gemma4 e2b, openchat, qwen2, qwen3) once ids are compared through llama's first EOG (owner stop-set policy). Still failing: D2 gemma4 26b diverges at index 0 on 2 of 3 prompts; O1 qwen35 + qwen35moe have no oracle (llama f1ea20621 rejects the blobs: rope.dimension_sections length 3, expects 4) |
| AC7 | R4 | oracle | `cargo nextest run -p proxima-model-interop --features std -E 'test(/window_ring_layers_/)'` | 2 passed: (a) gemma4 E2B ring layers equal `swa_layers.txt`; (b) a synthetic descriptor with a window on one dense layer gets a ring on exactly that layer | 1 passed, 1 failed ((b) fails: dense layers ignore the window) |
| AC8 | R8 | oracle | `cargo nextest run -p proxima-tokenizer --features gguf -E 'binary(gemma4_llama_oracle)'` | 20 passed, 0 failed | 20 passed |
| AC9 | R9 | consistency | `cargo nextest run -p proxima-model-interop --features std,conflaguration -j 1 -E 'test(/model_config_roundtrip_/)'` | 6 passed: for each checkpoint that lowers through the descriptor (gemma4 26B and E2B including the verify program, openchat, qwen2, qwen3, granite moe), GGUF -> config -> TOML text -> config -> lowered program has the same op count, op digest and logits root as AC0's. qwen35 and qwen35moe lower through bespoke builders until slice 9 makes them descriptors; they join then (8 passed) | tests absent; at HEAD the descriptor has no serde, so it cannot round-trip |
| AC10 | R9, R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal,conflaguration -j 1 -E 'test(/zero_rust_variant_/)'` | 2 passed: (a) the qwen2 0.5B model loaded from a hand-written TOML (in `tests/fixtures/model-configs/`, not derived from GGUF metadata) over its GGUF weights equals the llama ids in `llama-parity/qwen2`; (b) a TOML variant that changes the layer schedule (gemma4 E2B with every layer set full-attention) lowers and runs with no Rust change, and its op count differs from E2B's | tests absent |
| AC12 | R10 | oracle (incumbent speed: llama-server f1ea20621 and Ollama in the same interleaved run) plus a consistency baseline | `cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file <1k-token prompt> --processes 2 --runs 7 --arm base=<decode_gbps_baseline built at 9f0647da> --arm tip=<built at the slice> --llama-server <f1ea20621> --ollama gemma4:e2b-it-qat`, run alone on a quiet box (Ollama idle, no peer GPU or cargo jobs) | tip median ms/token <= base median + max(base MAD, 2%), outliers removed; the same for prefill ms; llama and Ollama arms printed alongside | base vs base: within the same bound (this proves the noise floor) |
| AC11 | R9 | consistency | `cargo check -p proxima-tensor --no-default-features --features alloc` and `git grep -nE 'std::(fs\|io\|net)\|File::' -- proxima-tensor/src/spec \| wc -l` | exit 0; 0, with `spec` compiled at the alloc tier (it sat behind `config`, so the check built none of it) | exit 0; 0 (measured at edd4163c), compiling zero lines of `spec` |

## out of scope

- New kernels; any omega kernel change.
- Quantized KV cache.

## decision (taken, owner may overrule)

- Collapsing V2 by folding SingleRange into TwoRange changes the op graph (masked vs
  never-masked cached block), so R7 cannot hold that way.
- Taken instead: a descriptor field `cache_mask` selects the existing algebra inside one
  engine, which keeps the graph byte-identical.

## findings from slice 0 (llama.cpp f1ea20621 oracle, ac4eb2c7)

- Tokenizer: proxima reproduces llama's prompt ids on 15 of 15 (checkpoint, prompt) pairs.
- D1 (not a defect, by owner policy 2026-09-17): the stop set is the caller's policy, and proxima
  stops only on `eos_token_id` (`proxima-model-interop/src/generate/residency_caches.rs:3433`).
  - llama.cpp stops on its EOG set (`src/llama-vocab.cpp:2869-3027`). For gemma4 E2B that set
    is {1, 50, 106}.
  - The parity test therefore compares ids up to and including llama's first EOG id.
- D2: gemma4 26B diverges from llama at generated index 0 ("The capital of France is") and at
  index 1 (river paragraph).
  - Both sides produce repetitive output on these raw prompts.
  - Root-caused by slot-0-7c (2026-10-04; evidence in
    `proxima-speculative-decode-evidence/bisect_26b/`): not a regression, and not shown to be a
    proxima numerics defect.
    - On raw prompts this Q3_K_M blob emits nonsense in proxima, llama-server and Ollama alike.
    - proxima's 26B ids are byte-identical at 2150fde9, 27d88bdf and f9fc44aa.
    - llama's own CPU and Metal backends share no id in their step-0 top-5.
    - proxima's per-layer drift from llama-Metal is smaller than llama-CPU's at every layer.
    - The one routing difference is an 8th-expert near-tie (router logits 1.81350 vs 1.80960).
  - Consequence: the 26B parity case compares noise. It needs chat-templated prompts where the
    model is confident, re-vendored from llama-server. Before that it was excluded from AC6's
    pass count, the same way O1 is.
  - Measured 2026-10-04 on chat-templated prompts (llama-server f1ea20621, greedy, 32 tokens
    or first end-of-generation id, `n_probs` 2). Prompts are rendered as
    `<|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`, which is what
    the GGUF's `tokenizer.chat_template` emits with thinking off; the BOS id is added by
    tokenization.
    - Selection was by llama alone, before proxima ran: nine candidates, kept in candidate
      order if llama's top-1 minus top-2 logprob is at least 1.0 nats at every generated step.
    - Smallest margin per candidate: "The capital of France is" 2.318, "What is 17 plus 25?"
      4.898, "Name the largest planet in our solar system." 10.403 (kept); `def fibonacci(n):`
      0.146, the river paragraph 0.031, a Python square function 0.642, "Translate 'good
      morning' into French." 3.842, "What is the chemical symbol for gold?" 3.479, Celsius to
      Fahrenheit 0.106. The first three that pass are the three kept.
    - Result: proxima's 26B ids equal llama's on all three kept prompts, and
      `llama_parity_gemma4_26b` passes; the 5-test `llama_parity_` run is 5 passed.
      The fixture records each prompt's minimum margin and the criterion.
    - The all-candidate recording is in `.long_ctx_backups/land/e2_26b_candidates.json`. On the
      the earlier first recording had `def fibonacci(n):` diverge at index 5 (llama " a",
      proxima " an", margin 0.146); that prompt fails the criterion and is not in the test.
  - Separate and untraced: proxima's CPU interpreter on 26B is off from llama at layer 0
    (`l_out-0` sum_rel_diff -8.06). slot-0-7c queued it.
- O1: qwen35 and qwen35moe have no oracle.
  - The Ollama blobs carry `rope.dimension_sections` of length 3, and llama f1ea20621 refuses
    to load them.
  - A header-patched copy, or a llama-convertible GGUF from the original HF weights, would
    restore the oracle. Making that copy needs owner approval: the patch command was denied
    by the permission check.
  - The two `llama_parity_qwen35` and `llama_parity_qwen35moe` cases were removed at 59cf72e6
    (owner excluded qwen from testing). The `llama_parity_` set lists 5: gemma4 26b, gemma4 e2b,
    openchat, qwen2, qwen3. Measured 2026-10-04: 5 passed, including gemma4 26b on confident chat prompts (D2 above).
