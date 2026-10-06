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
- Lowering from a family profile: `proxima-model-interop/src/lowering.rs` (`bind_checkpoint`,
  `header_descriptor`, `bind_speculative_verify`, `kv_layers`, `step_state`); the profile's `schedule_source`
  names the header reader, and `kv_cache_shape`, `ffn_routing` and `command_buffer_chunks` ride the same
  profile (`proxima-tensor/src/spec/descriptor.rs` `FamilyProfile`). The trait and registry this replaced
  were `proxima-model-interop/src/architecture.rs` at 9dd9deef (trait :282, `with_builtin` :612).
- Header readers, one per schedule source (no per-family type, no registry):
  - `proxima-model-interop/src/gemma4/{bind,hparams,program}.rs`
  - `proxima-model-interop/src/qwen35.rs`
  - `proxima-model-interop/src/qwen35moe/*.rs`
  - `proxima-model-interop/src/dense.rs`
  - `proxima-model-interop/src/lfm2.rs`
- Weight binding: `proxima-model-interop/src/bind_leaves.rs` walks a lowered program's `Op::Input` leaves;
  `proxima-model-interop/src/profiles/binding.rs` and `profiles/binding/*.toml` hold the names a
  tensor directory uses that a program does not; the per-tensor binders (`bind_dense_as`,
  `bind_matmul_weight_as`, `bind_moe_expert_weights`) stay in `proxima-model-interop/src/bind.rs`.
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
- R3. A speculative verify program is lowered for every family whose layers can rewind when the
  descriptor arms it (`speculative_verify`). The family profile carries the default, off until that
  family's verify step is measured to cost less than the drafts it checks (R10); a config layer
  overrides it.
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
| AC2 | R3, R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/generic_verify_llama_parity_/)'` | 5 passed: gemma4 E2B, openchat, qwen2, qwen3 and granite moe, each loaded from its header descriptor with `speculative_verify` set, drafts forced at widths 1 and 3, at least one verify step run, equal their llama ids | the same tests with `speculative_verify` unset: 0 passed, 2 failed ("the verify program never ran"; measured on gemma4 E2B and qwen2). At the previous commit the openchat and qwen3 verify programs did not exist |
| AC3 | R5 | consistency | `cargo nextest run -p proxima-model-interop --features std -E 'test(/generic_binder_/)'` | 8 passed (the seven checkpoints of the original count plus granite moe): bound names, codec, byte length and sha256 equal the incumbent's; the storage class of an f32 tensor (borrowed from the mapping or owned) is not compared, because the incumbent's gemma4 binder held its norms owned and its layer output scale borrowed, a split no property of the program decides; the f32 bytes and sha256 are compared | n/a: the generic binder does not exist at 9dd9deef. Slice 0 asserts the incumbent's capture is non-empty: 7 passed |
| AC4 | R6 | consistency | `git grep -nIiP '\b(gemma4\|qwen35moe\|qwen35\|qwen2\|lfm2\|mistral\|llama)\b' -- proxima-model-interop/src proxima-tensor/src omega/src proxima-tokenizer/src ':!*tests*' ':!*profiles*' \| wc -l` | 0 | 1214 |
| AC5 | R2 | consistency | `git grep -nP '\b(trait\|impl\|struct\|enum)\b[^;{]*Architecture' -- proxima-model-interop/src proxima-tensor/src \| wc -l`, then `cargo nextest run -p proxima-model-interop --features std,metal,conflaguration -E 'test(/model_config_roundtrip_/) or test(/serving_fsm_drives_/)'` | 0; then 10 passed: one descriptor config round trip per checkpoint that lowers through the descriptor (8: gemma4 26B and E2B, openchat, qwen2, qwen3, granite moe, qwen35, qwen35moe), plus 2 FSM tests (a plain decode and a speculative verify-accept-rollback run that the live generate path routes through `ServingState`) | first command prints 17 (trait, registry, 4 family impls, the test fake, and the per-family `Architecture`/`*Architecture` hparams structs); second: tests absent |
| AC6 | R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/llama_parity_/) and not test(/generic_verify_/)'` | 7 passed, one per checkpoint with an oracle: gemma4 26b, gemma4 e2b, granite moe, openchat, qwen2, qwen3 (6 at slice 8; qwen35 and qwen35moe have none, O1) and lfm2 from slice 9 | 3 passed, 4 failed at ac4eb2c7 (measured 2026-10-04); 4 passed (gemma4 e2b, openchat, qwen2, qwen3) once ids are compared through llama's first EOG (owner stop-set policy). Still failing: D2 gemma4 26b diverges at index 0 on 2 of 3 prompts; O1 qwen35 + qwen35moe have no oracle (llama f1ea20621 rejects the blobs: rope.dimension_sections length 3, expects 4) |
| AC7 | R4 | oracle | `cargo nextest run -p proxima-model-interop --features std -E 'test(/window_ring_layers_/)'` | 2 passed: (a) gemma4 E2B ring layers equal `swa_layers.txt`; (b) a synthetic descriptor with a window on one dense layer gets a ring on exactly that layer | 1 passed, 1 failed ((b) fails: dense layers ignore the window) |
| AC8 | R8 | oracle | `cargo nextest run -p proxima-tokenizer --features gguf -E 'binary(gemma4_llama_oracle)'` | 20 passed, 0 failed | 20 passed |
| AC9 | R9 | consistency | `cargo nextest run -p proxima-model-interop --features std,conflaguration -j 1 -E 'test(/model_config_roundtrip_/)'` | 6 passed: for each checkpoint that lowers through the descriptor (gemma4 26B and E2B including the verify program, openchat, qwen2, qwen3, granite moe), GGUF -> config -> TOML text -> config -> lowered program has the same op count, op digest and logits root as AC0's. qwen35 and qwen35moe joined at slice 9, when they became descriptors (8 passed) | tests absent; at HEAD the descriptor has no serde, so it cannot round-trip |
| AC10 | R9, R8 | oracle | `cargo nextest run -p proxima-model-interop --features std,metal,conflaguration -j 1 -E 'test(/zero_rust_variant_/)'` | 2 passed: (a) the qwen2 0.5B model loaded from a hand-written TOML (in `tests/fixtures/model-configs/`, not derived from GGUF metadata) over its GGUF weights equals the llama ids in `llama-parity/qwen2`; (b) a TOML variant that changes the layer schedule (gemma4 E2B with every layer set full-attention) lowers and runs with no Rust change, and its op count differs from E2B's | tests absent |
| AC12 | R10 | oracle (incumbent speed: llama-server f1ea20621 and Ollama in the same interleaved run) plus a consistency baseline | `cargo run --release -p proxima-model-interop --example decode_arms -- --prompt-file <1k-token prompt> --processes 2 --runs 7 --arm base=<decode_gbps_baseline built at 9f0647da> --arm tip=<built at the slice> --llama-server <f1ea20621> --ollama gemma4:e2b-it-qat`, run alone on a quiet box (Ollama idle, no peer GPU or cargo jobs) | tip median ms/token <= base median + max(base MAD, 2%), outliers removed; the same for prefill ms; llama and Ollama arms printed alongside | base vs base: within the same bound (this proves the noise floor) |
| AC11 | R9 | consistency | `cargo check -p proxima-tensor --no-default-features --features alloc` and `git grep -nE 'std::(fs\|io\|net)\|File::' -- proxima-tensor/src/spec \| wc -l` | exit 0; 0, with `spec` compiled at the alloc tier (it sat behind `config`, so the check built none of it) | exit 0; 0 (measured at edd4163c), compiling zero lines of `spec` |

## gate tiers (measured 2026-10-06, `/private/tmp/cargo_target_arch`, ollama stopped)

Real-checkpoint tests build with the `gate` profile (`Cargo.toml` `[profile.gate]`: release codegen,
no fat lto, 16 codegen units, `debug-assertions` and `overflow-checks` on, `panic = "unwind"`,
incremental). They serialize against each other through the `real-checkpoint` nextest group
(`.config/nextest.toml`), so no `-j 1` is needed. The per-slice tier is the `slice-gate` nextest
profile: every test in the package except the bind, decode, verify and cache tests of the large
checkpoints (gemma4 26b, openchat, qwen2, qwen3, qwen35, qwen35moe) and every lfm2 test. The
digests and descriptor round trips of every checkpoint stay in it. The weights hash in
`generic_binder_` is unchanged (every bound byte, same fixtures); it is fast because the dev
dependency `sha2` carries its `asm` feature (granite 4.2 s -> 0.77 s).

Per-slice gate, run from the checkout, in this order (about 2 minutes; measured breakdown below):

```
cargo clippy -p proxima-tensor -p proxima-model-interop --features proxima-model-interop/std,proxima-model-interop/metal --all-targets -- -D warnings
cargo check -p proxima-tensor --no-default-features --features alloc
cargo check -p proxima-model-interop --no-default-features
cargo nextest run -p proxima-tensor --cargo-profile gate
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate
```

Measured, warm cache: clippy 24.5 s (clippy of a one-file change is seconds; this figure is the
first run after a full-workspace change), alloc check 1.3 s, interop no-default check under 1 s,
tensor 779 passed in 6.7 s run, interop 707 passed in 86.4 s run (125 skipped: 105 ignored or
feature-gated, 20 end-of-run). The chain ran in 122 s end to end with 112 s of it in the last
command. A one-file edit in `proxima-tensor` rebuilds the gate profile in 10.4 s; the first build of the
profile is 81 s.

End-of-run gate, once, after the last slice (everything above, plus the 20 large-checkpoint tests):

```
cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate
```

Measured: 727 passed, 105 skipped, 607 s. The timing arms (AC12) are separate and run alone.

Where the time went before (test profile, same machine): `generic_binder_` gemma4 26B 606 s, openchat
182 s, E2B 167 s, granite 67 s, qwen35moe about 17 minutes. 99.9% of the granite and E2B tests was the
sha256 over every weight byte, at 320-340 MB/s because `sha2` was running its portable compression
function; `opt-level` did not move it. Attribution table:
`/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fast_gates/attribution.md`.

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

## findings from slice 6 (verify cost, 2026-10-05)

- Lowering a verify program for a family does not make speculation pay for it. With the default
  n-gram drafter (`ngram-simple`, up to 48 drafted rows) armed, measured on the 970 to 1000
  token prompt, 128 new tokens, release std+metal, one model process at a time, Ollama quit
  (`evidence/verify_cost/`):
  - granite moe 1b, `decode_arms` 8 processes x 7 runs, tip vs 0c: ms/token 20.9740 (MAD 0.0810)
    vs 14.6570 (MAD 0.0700), delta +6.3170 against limit 0.2931, within=false; peak footprint
    +19923008 against 12157222, within=false.
  - `generate_from_ids` with speculation on vs off, 3 alternating rounds each (ms/token on, off;
    verify steps, drafted rows, accepted rows): granite moe 21.841/15.198 (4, 189, 49); qwen2
    36.307/20.267 (7, 119, 16); qwen3 240.040/29.173 (7, 114, 69); openchat 279.439/24.329 (3,
    114, 114); gemma4 E2B 13.039/11.696 (1, 2, 2). Round 0 of each model; rounds 1 and 2 are in
    the evidence files.
  - openchat with the verify program armed and a forced draft width, step gap in ms (plain decode
    step 23): width 1 about 140, width 2 about 490, width 4 about 2050, width 8 about 4050, width
    16 to 38 about 3800 to 6200.
  - Not traced: why a multi-row step against a 1100-token cache costs this much on the dense
    two-range path. Its kernels are out of scope (see "out of scope"); the numbers are the
    reason the profile default is off, not a finding about the kernels.
- Consequence for R3: the verify program is lowered when `ModelDescriptor::speculative_verify` is
  set. The family profile supplies the default (`speculative_verify = true` only for gemma4,
  whose verify step was measured to pay), and a config layer overrides it. With the default, the
  granite moe decode program and its speed are the 0c baseline's.

## findings from slice 7 (one cache engine, 2026-10-05)

- The three `CacheStrategy` arms were two whole-model builders plus the cacheless one, and the
  two cached builders carried three textual copies of the same two-block score algebra: the
  dense layer (`append_mistral_cached_layer`), the MoE layer (`append_mistral_cached_moe_layer`)
  and the gemma4-shaped layer (`append_lfm2_two_range_cached_attention`). Compared line by line,
  the copies differ in three places only: the cached block's mask (`Option` in the dense and MoE
  layers, always present in the gemma4-shaped one), where the local block's `-inf` constant is
  emitted, and an instrumentation call in the dense layer.
- Landed: one score core, `two_block_attention.rs` (`group_queries`, `append_cached_block_scores`,
  `append_local_block_and_combine`), used by all three layers, with the mask as the one argument
  that selects an arm. `CacheStrategy` is `Cacheless | Cached`; `ModelDescriptor::cache_mask`
  (`CacheMask::Bounded | Padded`, serde default `Bounded`) names the cached block's mask. The
  rewrite is node for node: the 8 AC0 digests are unchanged.
- Not collapsed: `build_forward` still reaches two whole-model preludes, chosen by
  `(cache_strategy, cache_mask)`. The dense builder emits its leaves (`cos`/`sin`, masks,
  `cached_len`) in an order the AC0 digests pin and the gemma4-shaped builder emits another, so
  one prelude would change the op graph of one family (R7). The preludes also differ in what they
  support (fused QKV, biases and layer taps in one; shared KV, PLE and per-layer widths in the
  other), and those are fields the mask does not select. Reaching one prelude needs a graph
  change that re-captures the incumbent digests, which this slice does not do.

## findings from slice 8 (weights bind from the lowered program, 2026-10-06)

- Every `Op::Input` leaf of the 8 AC0 programs was listed against the incumbent's bound set before
  any code changed (leaf name, shape, GGUF dims and type, consuming ops;
  `evidence/bind_from_leaves/survey/survey_*.tsv`). Leaves, bound weights and leaves nothing
  binds: gemma4 26B 787, 688, 99; gemma4 E2B 595, 541, 54; granite moe 321, 243, 78; openchat 393,
  291, 102; qwen2 369, 291, 78; qwen3 513, 399, 114; qwen35 387, 321, 66; qwen35moe 839, 733, 106.
  The leaves nothing binds are the step inputs (ids, eps, the RoPE tables, `cached_len`, the
  `kv_cache.*` and `ssm_cache.*` leaves, `lm_head_row`) in all 8; the only bound tensor no leaf
  names is gemma4's `rope_freqs.weight`.
- Landed: `bind_program_leaves` (`proxima-model-interop/src/bind_leaves.rs`) walks the leaves and
  chooses the per-tensor binder from the program:
  - a leaf a `Multiply` feeds into an `Add` reduce is a matmul weight when its contracted axes lead
    its kept axes (the program declares `[in, out]`, the file stores `[out, in]`) and binds in file
    order otherwise (the conv kernel of the hybrid layers binds in file order, bytes equal to the
    incumbent's);
  - a rank-3 leaf a computed map indexes is an expert stack;
  - every other leaf binds as stored.
  A leaf the directory does not name resolves through `profiles/binding/*.toml`: `default.toml`
  (a tied output reads the embedding table, a fused `gate_up` expert stack splits by row into
  `gate` and `up`, the paired gate/up and fused qkv diagnostic programs join their sources) and one
  small file per family that needs more (gemma4 `extra`, qwen35 three renames, qwen35moe
  `decode_f32`, lfm2 a rename and three row parts).
- Deleted: `bind_all_weights`, `bind_gemma4_weights` with `gemma4_tensor_names` and the fused
  gate/up split, `bind_qwen35_weights`, `bind_qwen35moe_weights` with `qwen35moe_tensor_names`,
  `bind_lfm2_weights` with the `in_proj` split, the dead `GEMMA4_NORM_SHIFT` path, the
  `PROXIMA_HEAD_PRIVATE_COPY` knob (its only caller was `bind_all_weights`) and the error variants
  only those functions raised. `hf_bind.rs` stays: it maps safetensors names, a naming universe
  with no GGUF directory to walk, and V3 does not list it.
- Designs abandoned, with the reason:
  - read a leaf's orientation off its shape against the GGUF dims: a square matrix
    (`[1024, 16, 64]` over dims `[1024, 1024]`) reads both ways, so the consuming reduce decides;
  - one binder function per family, the shape the slice deletes;
  - an alias list kept in the Rust of each bespoke builder: the names are data, so they moved to
    `profiles/binding`.
- Findings that came out of the data, not the plan:
  - the incumbent held gemma4's norms as owned buffers and its layer output scale as a borrowed
    view, a split no property of the program decides, so AC3 does not compare the storage class of
    an f32 tensor (`without_f32_storage_class` in the test; bytes and sha256 still compare);
  - qwen35moe binds `ssm_alpha.weight` and `ssm_beta.weight` as owned f32 transposed while qwen35
    binds the same leaves packed. qwen35moe has no oracle (O1), so the incumbent's choice is
    carried as the `decode_f32` data in its binding file instead of being changed;
  - a binder keyed by the checkpoint's `general.architecture` string lost the qwen35 renames for a
    foreign architecture that delegates to the builtin one
    (`external_architecture_hybrid_cache`, 3 tests: `MissingStepInput { name: "blk.0.ssm_in.weight" }`).
    The hybrid and gemma4 binders key their profile by the architecture that lowers the program,
    the dense one by the checkpoint's family;
  - `BoundProgram::lowered_from` kept the weights the header program bound, so a config that adds a
    leaf (`paired_gate_up_reduce` adds `ffn_gate_up.weight`) left it unbound. It now binds the
    leaves the new program adds and leaves the rest as bound
    (`a_config_that_pairs_gate_and_up_binds_the_fused_operand_it_adds`; with the bind call disabled
    the same test fails at `layer 0 fused gate/up operand is bound`).
- Equality with the table binder on a checkpoint outside the 8: LFM2.5-8B-A1B Q4_K_M, 293 bound
  names on both sides, no name on one side only, bytes and codec equal on every name, measured with
  both binders compiled together before the table binder was deleted
  (`evidence/bind_from_leaves/gates/lfm2_compare.log`).
- A weight leaf the directory does not name is skipped like a step input and surfaces at the first
  step as an unbound input naming it. The 8 checkpoints are held to the opposite by
  `assert_unbound_leaves_are_step_inputs`: every leaf nothing binds is in the step-input vocabulary
  the test lists.

## findings from slice 9 (recurrent hybrids and lfm2 as descriptors, 2026-10-06)

- Landed:
  - `LayerKind::Gdn`; `ModelDescriptor` fields for the recurrence (`ssm_conv_kernel`,
    `ssm_state_size`, `ssm_group_count`, `ssm_time_step_rank`, `ssm_inner_size`, `ssm_epsilon`,
    `v_head_reordered`), the shared expert (`expert_shared_feed_forward`), the pinned prefill
    length (`prefill_width`) and the attention variant (`gated_attention`), each with a serde
    default so every earlier config still loads; `FfnCombination::RoutedWithSharedExpert`.
  - `build_forward` returns a named `ForwardProgram` instead of a seven-tuple, with one
    `Qwen35LayerRoots` per layer for every engine (the zip of schedule against cache roots that
    `rebuild_layer_roots` did in `proxima-model-interop` now runs inside the lowering, and its
    interop error variant is gone) and a `layer_diagnostics` table the routed hybrid fills.
  - `hybrid_forward.rs`: the layer loop that was `qwen35_forward_program_with_last_row`
    (`(layer + 1) % full_attention_interval` decided the mixer) and the loop of
    `qwen35moe/program.rs` now read `layers[i].kind` and the descriptor's fields. Two arms,
    chosen by `layers[i].ffn.combination` (dense SwiGLU, or routed with a shared expert),
    because their leaf order is pinned by the AC0 digests, the same reason slice 7 kept two
    preludes. `qwen35_forward_program*` stays as a positional wrapper that builds a descriptor
    and calls `build_forward`. `qwen35moe/program.rs` is now the map from the header and the
    family profile to a descriptor, `shared_expert.rs` moved into the tensor crate.
  - Profiles `qwen35`, `qwen35moe`, `lfm2` and `lfm2moe`; `qwen35::descriptor_from_architecture`,
    `qwen35moe::descriptor_from_architecture`. AC9 now covers both: 8 passed.
  - lfm2 loads through `DenseArch`: a header that declares zero KV heads on some layers is a
    hybrid, the tensor directory says which kind each such layer is
    (`LayerKind::from_tensor_names`), the FFN widths and the leading dense block count come off
    the header, and the descriptor is cacheless. `uniform_lfm2_schedule` and `run_lfm2_prefill`
    are deleted; `Lfm2Architecture` carries the schedule from the same descriptor.
- Designs abandoned, with the reason:
  - a `CacheStrategy` arm that forwards to the bespoke qwen35 builder with an interval argument:
    it keeps the bespoke op order by construction and a variant with another layer pattern
    would need Rust;
  - `LayerKind::Gdn(GdnConfig)` with the recurrence shape in the variant: both builders take one
    recurrence shape for the whole model, so flat model-global fields follow `l_cache`, `qk_norm`
    and the other flat knobs and keep `LayerKind` a `Copy` unit enum;
  - an `Lfm2Arch` registry entry: one more per-family type, where the dense builder already
    reads a per-layer KV array;
  - a `cache_strategy` field in the family profile: whether a schedule can be cached follows
    from its layer kinds (no cached lowering exists for a short-convolution layer), the rule
    `gemma4_descriptor_from_gguf` already used.
- Findings that came out of the data, not the plan:
  - the first routing rule (a schedule holding a `Gdn` layer) sent a qwen35 header with
    `full_attention_interval = 1` through the plain attention engine, and an all-recurrent prefix
    through an engine that wanted an attention layer to read the shape from: 3 model tests
    failed (`LeafShapeMismatch` on `blk.0.attn_q.weight`, 16 elements against 32, twice;
    `UnsupportedInBuilder` "a hybrid schedule with no attention layer" once). The gated
    attention variant lays `attn_q` out as `[Q | gate]` per head, which no layer kind says, so it
    is the field `gated_attention`; every layer carries the one attention shape, a recurrent layer
    unread.
  - lfm2 was recorded twice. The first recording (the three raw prompts of the other fixtures,
    32 tokens) gave 2 divergences of 3: proxima and llama split at generated index 12 and 13,
    where llama's top-1 minus top-2 logprob is 0.133 and 0.124 nats. The fixture was replaced
    under the gemma4 26B criterion (llama top-1 minus top-2 at least 1.0 nat at every compared
    step, chosen by llama alone in candidate order; 14 chat-templated candidates, 3 kept, compared
    through the longest such prefix of the 32 requested tokens: 21, 19 and 20 ids). The raw
    recording stays in `.long_ctx_backups/arch_data/slice_9/`.
  - the oracle tells the rope pairing: the lfm2 profile pairs split-half (llama.cpp's rope-type
    table puts LFM2 and LFM2MOE with the NEOX families) and the three prompts pass; the same
    config with interleaved pairing diverges on 1 of 3 (`model_config_edit_to_the_lfm2_rope_pairing_diverges_from_llama_on_at_least_one_prompt`).
- Not done, and why:
  - lfm2 runs cacheless: every generated token re-prefills the whole sequence, because no
    short-convolution state-cache engine exists (the cached engines lower attention layers only).
    Parity is measured; the speed of that path is not a goal of this slice and is not compared
    to anything.
  - `Qwen35Arch` and `Qwen35MoeArch` are still registered `Architecture` values (slice 10c);
    the qwen35moe runtime still reads `qwen35moe::hparams::Architecture` for the multi-axis
    position table and the pre-gather path.
  - qwen35 and qwen35moe have no oracle (O1); their parity here is AC0 (digests byte-identical
    through the descriptor lowering) and AC9.
- Gates at the slice tip, per-slice tier in the gate profile (cb155e98), logs in `evidence/recurrent_hybrids/gates/v_*.log`:
  clippy over both crates exit 0 (1.4 s, warm); tensor alloc check exit 0 (0.4 s); interop no-default check exit 0 (0.2 s);
  `nextest -p proxima-tensor --cargo-profile gate` 779 passed, 8 skipped, 6.8 s run;
  `nextest -p proxima-model-interop --features std,metal --cargo-profile gate --profile slice-gate` 707 passed, 125 skipped, 86.7 s run;
  `generic_binder_qwen35` and `generic_binder_qwen35moe` 2 passed, 830 skipped, 43.8 s run (1.6 s and 42.2 s).
  AC0 8 passed (3.6 s), AC9 8 passed (3.8 s), AC6 7 passed (200.2 s, lfm2 included).
- Performance arms (decode_arms, tip against the 0c binary, `evidence/recurrent_hybrids/perf_e2b` and `perf_granite`):
  36 of 36 bound lines within (18 per model, 0 outside). Tip against 0c, gemma4 E2B: ms/token 12.2310 vs 12.2150,
  prefill 2531.5 vs 2533.9 ms, peak RSS -4857856 B, footprint -3571744 B, GPU bytes -1179648 B. granite moe: ms/token
  14.8675 vs 14.8875, prefill 6522.97 vs 6524.01 ms, peak RSS -117661696 B (limit 54258074), footprint -1417248 B, GPU
  bytes 0.

## findings from slice 10b (the serving state machine drives the live decode loop, 2026-10-06)

- R2's FSM half is wired: `run_decode_loop_from_ids` holds one `ServingState<u32, usize>` (entry `u32`, cache the
  cache cursor), and `ServingState` lives in `proxima-core/src/serving_state.rs` (the `serving_fsm.rs` named in R2 no
  longer exists). `decode.rs` `run_decode_loop_from_ids`: `Prefill`/`Decode` supply the step's input ids, a drafted
  step enters `Verify`, and the verify readout calls `accept_rows` then `resume` (every draft matched) or `rollback`.
- Verify evaluates `last` plus the D drafts, D+1 rows. `Accept` resumes through `advance_decode` of the bonus token,
  because that row is the next single-row decode step and the verify batch already computed it; `Rollback` carries the
  cursor to truncate the attention caches to. A `Verify` snapshot is a `usize` copy, which is why the cache type is the
  cursor and not the per-layer `Vec<LayerCacheState>` that `generate/serving_backend.rs` sketches (cloning that for a
  snapshot would copy every KV layer per verify step).
- `ServingState::accept_rows` requires `row_tokens` only through row `accepted` (the verifier stops sampling at the
  first differing row, so later rows do not exist); `row_caches` still has one entry per draft.
- Closed by `f7f40179`: the `pending` queue and the evaluation closure are gone from `run_decode_loop_from_ids`.
  The serving loop (now `drive_serving_loop`) is a `'decode: while` over the state machine: one `'evaluate` block
  per step reads the state, runs the evaluation and settles it, leaving the step's tokens in `settled_tokens` (one
  token, or the accepted drafts plus the correction or bonus token after a verify); a `for` over `settled_tokens`
  delivers each through `deliver_token`, which is the body `decode_until_stop_or_budget` used to hold, now shared
  by both (its state is three caller-owned locals passed by reference, no carrier type). A step's tokens are
  drained inside the iteration that produced them, so no queue crosses steps. Budget, eos and `on_token`
  `Break` stop at the same token as before (AC5 FSM half asserts the step counts, AC6 and AC2 the ids).
  `run_decode_loop_from_ids` is now a wrapper over `drive_serving_loop` that runs the monolithic expert-source
  release on every exit, because the closure's old release sat between the loop and `decode_result?`, and the
  inline loop's `?` returns leave no such point.
- Still on `decode_until_stop_or_budget` with a closure: `run_decode_loop_placed_kv` (the single-range, device
  resident KV arm, `metal-output-placement` on macOS only, never taken for MoE, seeded or two-range calls). It holds
  `cached_len` and `next_ids` locals, not a `ServingState`; it is a second decode loop outside this slice's row.
- Behavior change: drafting needs `Decode`, so a seeded call whose new range is one token no longer drafts at its
  prefill step. Ids are unchanged (AC2, AC6).
- Gates at the slice tip, per-slice tier in the gate profile, logs in `evidence/serving_fsm/gates/`: clippy exit 0,
  tensor alloc check exit 0, interop no-default check exit 0, tensor 779 passed 8 skipped (6.9 s), interop slice-gate
  709 passed 125 skipped (100.2 s), 136 s for the chain; core `nextest -p proxima-core --features config` 138 passed;
  AC5 FSM half 2 passed, AC6 7 passed (154.9 s), AC2 5 passed (164.8 s).
- Performance arms (decode_arms, 2 processes x 3 runs, tip against the 0c binary, `evidence/serving_fsm/perf_e2b`
  and `perf_granite`): 12 of 12 bound lines within. gemma4 E2B ms/token 11.9500 vs 12.1605, prefill 2514.49 vs
  2494.48 ms; granite moe ms/token 14.8760 vs 14.8240, prefill 6519.96 vs 6515.96 ms. The full 8 x 7 arms run at the
  end of the run.
- Gates at `f7f40179` (tree `3b87f73af10c55ded0fc0fcbfa52529bc7b20cf1`), logs in `evidence/serving_fsm_inline/gates/`:
  clippy exit 0, tensor alloc check exit 0, interop no-default check exit 0, `--features std` all-targets check exit 0
  (the one build where `decode_until_stop_or_budget` has no non-test caller; it is `cfg(any(test, placed-kv))`),
  tensor 779 passed 8 skipped (6.7 s), interop slice-gate 709 passed 125 skipped (98.5 s), AC5 FSM half 2 passed
  (7.3 s), AC6 7 passed (148.2 s), AC2 5 passed (164.0 s).
- Performance arms for the inline loop (decode_arms, 2 processes x 3 runs, `evidence/serving_fsm_inline/perf_e2b`
  and `perf_granite`), run on a release `decode_gbps_baseline` built from the staged tree whose id is the commit's
  tree id above (sha256 `bdeb9143692378f74d440ff0063cf871641bd069bec70d0e822b46b7def0ffe8`, kept as
  `perf/decode_gbps_baseline_slice10b_fix`), against the 0c binary, Ollama stopped by SIGTERM (`osascript` quit
  returned "User canceled"; `/api/ps` refused the connection before each run; reopened after): 12 of 12 bound
  lines within. gemma4 E2B ms/token 12.0565 vs 12.1745 (limit 0.2435), prefill 2516.04 vs 2516.51 ms, peak RSS
  -16441344 B, footprint +7716928 B (limit 14120444), GPU bytes -1179648 B. granite moe ms/token 15.1290 vs
  15.1245 (limit 0.3025), prefill 6518.54 vs 6516.96 ms, peak RSS +31776768 B (limit 49721344), footprint
  -23101376 B, GPU bytes 0.

## findings from slice 10c (the architecture trait and registry are deleted, 2026-10-06)

- Landed as six commits: `95ad3fa8` (profile fields), `35494fe2` (the trait, its four impls and the registry),
  `9bdfa6c3` (header hparams type names), `bba7f356` (recurrent runtime type names), `292c2b96` (routed-expert
  serving knob names), `48c57e9b` (three broken intra-doc links in `proxima-tensor`).
- What replaced the trait. `FamilyProfile` (`proxima-tensor/src/spec/descriptor.rs`) carries four more data fields,
  all in the profile TOML keyed by `general.architecture`: `schedule_source` (`uniform`, `sliding_pattern`,
  `recurrent_interval`, `recurrent_routed_interval`: which compiled header reader fills the descriptor),
  `kv_cache_shape` (`uniform`, `custom`, `monolithic`), `ffn_routing` (`dense`, `routed`) and
  `command_buffer_chunks` (default 1, gemma4 8). `proxima-model-interop/src/lowering.rs` is a set of free functions
  (`bind_checkpoint`, `bind_checkpoint_with_kv_layout`, `bind_speculative_verify`, `header_descriptor`,
  `kv_layers`, `step_state`, `trained_context_length`, `rope_freq_factors`, `sliding_rope_inputs`); each header
  reader is one `match` on `schedule_source`, and the lowering, the weight bind and the `BoundProgram` roots are the
  one generic `bind_descriptor` for every family (a root an engine does not produce is the empty value of its type,
  so the digest records of all 8 checkpoints are unchanged).
- Designs ruled out while writing it. (1) A smaller trait or a `HeaderReader` registry: the schedule source is an
  enum in the profile, so adding a family whose header already reads as one of the four layouts is a profile file.
  (2) Deriving `kv_cache_shape` and `ffn_routing` from the descriptor ("every layer is attention with one
  window"): a gemma4 variant with every layer full attention would satisfy that predicate and enter the
  single-range builder, which has no per-layer widths, PLE or value norm; the profile states the shape instead.
  (3) Keeping the foreign `step_inputs` and `rope_freq_factors` hooks: both reduced to functions of data already on
  the bound model (`ModelHparams::sliding_rope`, and an owned `rope_freqs.weight` the binding profile names).
- Removed with the trait: `tests/external_architecture_{registry,step_inputs,hybrid_cache,single_position_prefill}.rs`
  (16 integration tests) and the registry's 10 unit tests, 26 together; they proved the foreign-registry seam, which
  no longer exists. Also removed: `InteropError::UnknownArchitecture`, `LoadedModel::load_with_registry`,
  `StepInputContext`, `bind_gemma4_all_positions_logits`. The one behavior those tests held that a config can still
  express is kept: `capability_matrix.rs` loads the synthetic checkpoint with `last_row_only = false` and expects
  `LogitsShapeMismatch`, with the header descriptor as the positive control. The two reduce flags
  (`load_with_paired_gate_up_reduce`, `load_with_fused_qkv_reduce`) are descriptor fields now: they load the
  header descriptor with the flag set when the schedule source is `uniform`, and load as `load` does otherwise.
- The digest fixtures lost one line each, `registry_entry=...`, which recorded the deleted registry's route name;
  every op count, op digest, root, and bound-weight line is unchanged (AC0 8 passed, AC3 8 passed).
- Renames: `ModelArchitecture` is `ModelHparams`; the four per-family header structs are `Gemma4Hparams`,
  `Qwen35MoeHparams`, `Qwen35Hparams`, `Lfm2Hparams`; `Qwen35LayerRoots` is `LayerCacheRoots`, `Qwen35SsmShape`
  is `SsmShape`, `Qwen35DenseAttention*` is `DenseAttention*`, `Qwen35Gdn*` is `Gdn*`,
  `Qwen35MoeLayerDiagnostics`, `Qwen35MoeExecutionMode`, `Qwen35MoePreGatherPlan`, `Qwen35MoeRouteHistory`,
  `Qwen35MoeLayerSegments` lose the model name; the seven serving knobs `qwen35moe_pre_gather`,
  `_persistent_cuts`, `_residency_budget_bytes`, `_expert_prefetch`, `_monolithic_all_low`, `_layer_window`,
  `_monolithic_high_mmap` are `moe_*` (the `examples/gguf_generate.rs` env spelling follows:
  `PROXIMA_MOE_PRE_GATHER`). Test function names and the `qwen35moe-*` cargo feature names keep the checkpoint
  name.
- Measured, AC5 first command: 17 at 91280669, 0 at `292c2b96` (`evidence/family_profile/acs/ac5_grep.txt` is
  empty). AC4 (the name search, not this row's AC): 1190 at 91280669, 1113 at `292c2b96`.
- Spec correction: the AC5 second command named `descriptor_config_parity_` tests that no commit ever held; the
  config round trips are `model_config_roundtrip_` (8 checkpoints since slice 9), and the FSM tests take 7 to 8 s
  with `metal` and 70 to 225 s on the CPU without it, so the command carries `metal`. 10 passed (8.2 s).
- Gates, per-slice tier in the gate profile at `292c2b96`, logs in `evidence/family_profile/gates/`: clippy exit 0,
  tensor alloc check exit 0, interop no-default check exit 0, interop all-targets check with `instrument`,
  `qwen35moe-linked-suffix`, `qwen35moe-expert-prefetch` exit 0, the root package's `gguf_generate`,
  `stream_generate`, `write_qwen35_sidecar` and `openai_serve_gguf` examples check exit 0, tensor 779 passed 8
  skipped (7.8 s), interop slice-gate 697 passed 125 skipped (105.6 s; 709 at the previous slice, plus the profile
  field test, minus 26 deleted tests, plus 11 `lowering` unit tests and 2 `capability_matrix` tests). At `292c2b96`
  (`evidence/family_profile/acs/`): AC0 8 passed (3.0 s), AC3 8 passed (71.7 s), AC6 7 passed (154.7 s), AC2 5 passed
  (164.9 s), AC5 second command 10 passed (8.2 s), AC10 2 passed, AC1 2 passed, AC7 2 passed.
- Performance, decode loop touched (the step inputs and the routing and shape reads moved from a trait object to
  fields): decode_arms 2 processes x 3 runs, tip (release `decode_gbps_baseline` built from `292c2b96`, sha256
  `e81e3601b2b28febbb3a53ed1b11ef1eefed2c09d470dc6d39454eeaac106a55`, kept as
  `perf/decode_gbps_baseline_slice10c`) against the 0c binary, Ollama stopped by SIGTERM (`osascript` quit returned
  "User canceled"; reopened after). The box was shared with another checkout's `cargo xwin check` runs, so each
  model was re-run until one run started with no `cargo`, `xwin` or `rustc` process present. Quiet runs
  (`perf_e2b`, `perf_granite`): 12 of 12 bound lines within. gemma4 E2B ms/token 12.0660 vs 12.3825 (limit
  0.3485), prefill 2520.9850 vs 2517.5075 ms (limit 50.3501), TTFT 2521 vs 2517.5, peak RSS +6455296 B, footprint
  -3096640 B, GPU bytes -1179648 B; granite moe ms/token 15.0555 vs 15.0500 (limit 0.3010), prefill 6519.0090 vs
  6516.9450 ms (limit 130.3389), peak RSS -44974080 B, footprint -9290176 B, GPU bytes 0. Contended runs, kept:
  `perf_e2b_run1` (load average 7.86) ms/token 12.4120 vs 12.1950 (limit 0.2439), 6 of 6 within;
  `perf_granite_run1` (an xwin check running) ms/token 21.8120 vs 21.7535 and peak footprint +12066848 B against a
  limit of 11620804 (within=false), the other 5 within; `perf_granite_run2` (xwin running at launch) base arm CoV
  18.79%, ms/token 15.0600 vs 18.3380, within only because the base MAD is 3.365. Neither contended granite run
  is a measurement of this change.
- Not done, and why it is outside the row: the decode loop still calls `qwen35moe_forward_program_at_width` through
  `LoadedModel::qwen35moe_hparams` to pin a prompt width. `ModelDescriptor::prefill_width` is the field that
  carries the same value, so the loop can rebuild that program from the descriptor in force and drop the family
  reader from the loop; AC4 counts that name and the `qwen35moe`, `gemma4` and `lfm2` module and profile-key
  literals still in non-test source.
