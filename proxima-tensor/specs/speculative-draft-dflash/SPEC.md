# speculative-draft-dflash

status: draft
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)

## problem

proxima has no representation of a block-diffusion draft model -- the second descriptor of
`decode-as-data` R6 has no masked-block input, no cross-model feature injection and no `dflash.*`
metadata reads (`git grep -ci dflash -- proxima-model-interop/src proxima-gguf/src proxima-tensor/src`
lists only `serving.rs` (5 matches) and `speculative_settings.rs` (6 matches), where only the llama type name
exists) -- so llama.cpp's `draft-dflash` speculation type (`common/speculative.cpp:923-1329`, upstream
`f1ea20621`), which drafts a whole block of tokens per step by injecting the target model's hidden
states into a second, smaller model's KV cache, has zero coverage on the proxima side; this sub-spec of
`proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md` (R11c) makes `draft-dflash` available
through `ServingConfig`'s speculative section as data on the `decode-as-data` frame -- hidden-state tap
readouts on the target (`decode-as-data` R5), a second `ModelDescriptor` with its own `ServingState`
(`decode-as-data` R6), and a bidirectional block shape (`decode-as-data` visibility inputs) -- with
proxima's generated token ids equal to llama.cpp's on a real target and draft GGUF pair, and adds no
technique-named type (`decode-as-data` AC9).

## refutation condition

Written before evidence: if no `draft-dflash`-typed GGUF pair (target and draft), or for R7 no selector-bearing (`dflash.selector_top_k` above 0) pair, can be obtained or
produced within slice 1 -- no pre-converted GGUF is published for a locally reachable model, and
`convert_hf_to_gguf.py` at `f1ea20621` has no registered converter for `general.architecture` equal to
`dflash` (`grep -c 'dflash' /Users/brianbruggeman/repos/others/llama.cpp/convert_hf_to_gguf.py` is the
check, slice 1) -- then the requirements that need a real model (R6, R7, R10) are struck down to
structural-only coverage (R1-R5, R8, R9, R11) and this sub-spec cannot be audited past `draft` for
those requirements until the model gap is closed by a separately scoped initiative.

Separately, and independent of file availability: if, with a pair in hand, proxima's drafted block
produces a different first drafted token id than llama.cpp's on 3 of the first 3 vendored prompts of
`dflash1.json` while the injected features match llama.cpp's to 1e-4 (AC6 passing), the claim that a
tap readout plus a bidirectional-block draft descriptor reproduces the DFlash draft is false for the
draft graph itself, and the descriptor design (R3, R6) must be redone, not tuned.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | the draft descriptor reads the metadata keys `dflash.block_size`, `dflash.conv_kernel_size`, `dflash.conv_group_size`, `dflash.selector_rank`, `dflash.selector_top_k` (mirroring `src/llama-hparams.h:250-254`) from a `dflash` `general.architecture` GGUF as descriptor data, without erroring on a checkpoint that has them | yes |
| R2 | the descriptor resolves the tensor names `blk.%d.attn_conv_base/proj`, `blk.%d.ffn_conv_base/proj`, `selector_predecessor`, `selector_successor`, `selector_hidden` (`src/llama-arch.cpp:702-708`) | yes |
| R3 | the draft program takes the masked-block input `[id_last, <mask> * (block_size-1)]` as the row layout the host shape pipe produces | yes |
| R4 | the target's lowered program emits the `target_layer_ids` layers' input hidden states for every new position as a tap readout (`decode-as-data` R5), gathered into the `[n_chunk, n_embd_enc]` shape the draft consumes (mirrors `llama_set_embeddings_layer_inp`/`llama_get_embeddings_layer_inp`, `src/llama-ext.h:111,115`, and the gather loop at `common/speculative.cpp:1153-1163`) | yes |
| R5 | the fused encoder path (post `662a0b012`): target features reach the draft's embd-input decode in one decode call per chunk, never through a separate encode-then-reupload step | yes |
| R6 | DFlash1 (non-selector) block decode: top-k=10 sampler per masked position, `p_min` early stop, greedy token ids equal llama.cpp's own draft output given the same injected features and RNG state (`common/speculative.cpp:1295-1318`) | yes |
| R7 | DFlash2 (selector) block decode: the output is read as a `[block, selector_top_k]` lattice, a greedy predecessor-argmax chain with softmax `p_min` truncation, deterministic given the lattice and without resampling (`common/speculative.cpp:1237-1265`, gated by `llama_model_dflash_selector_top_k(model) > 0`) | yes |
| R8 | the accept stage adds no dflash-specific rollback state: llama.cpp's `accept()` is a no-op (`common/speculative.cpp:1326-1328`), so only the parent spec's R3a attention-KV truncate runs | yes |
| R9 | `ServingConfig`'s speculative section (parent R9) carries the `draft-dflash` sub-config `block_size` (0 = read from GGUF metadata), `n_max`, `n_min`, `p_min`, `backend_sampling`, with defaults equal to llama.cpp's `common_params_speculative_draft` (`common/common.h:327-352`) and the `--spec-draft-n-max`/`--spec-draft-p-min`/`--spec-draft-backend-sampling` flags (`common/arg.cpp:4137,4194,4201`) | yes |
| R10 | for the same target, draft, prompt and seed, proxima's generated token ids with `draft-dflash` enabled equal the ids llama.cpp `f1ea20621`'s `llama-speculative-simple` generates with `--spec-type draft-dflash --model-draft`, for 3 seeds | yes, gated by the refutation condition |
| R11 | the oracle comparator is itself validated: fed llama's own fixture on both sides it passes, and fed one flipped id it returns a mismatch | yes |
| R12 | the vendored fixtures are produced by a C++ harness linking llama.cpp `f1ea20621` and calling the public API the DFlash implementation calls (`src/llama-ext.h:96-128`), never re-derived from reading the C++; each fixture records the upstream commit in its header | yes |
| R13 | attention over the block is non-causal (the bidirectional block shape of `decode-as-data`: one `seq_member` column of ones, constant `mask_pos`) unless `dflash.attention.causal` is set, in which case it is causal | yes |
| R14 | each layer's local convolution (attention and FFN conv base plus proj) equals llama.cpp's on the same inputs (`src/models/dflash.cpp`) | yes |
| R15 | every vendored oracle run itself drafted (its recorded `n_drafted` >= 1), so an oracle that silently drafted nothing cannot back R10 | yes |

## architecture

Nothing here is a new type. DFlash is a tap readout list on the target descriptor, a second
descriptor (the draft) with its own `ServingState`, and the bidirectional block shape, driven through
the four pipes of `decode-as-data` (propose, shape, accept, commit). Its row in that spec's table is
"masked block on a draft descriptor | block | sample-and-match on the target | prefix".

- **descriptor data (R1/R2/R3)**: the `dflash.*` keys and the conv/selector tensors become fields
  of the draft `ModelDescriptor` (`proxima-tensor/src/spec/descriptor.rs:130-296`), named by what they are
  (block size, local conv kernel and group size, candidate-selector rank and top-k), not by the
  technique; a family profile TOML for the `dflash` `general.architecture` value, beside
  `proxima-model-interop/src/profiles/*.toml`, supplies the layer shape. The `general.architecture`
  value is read at load as a key, never as a Rust type. `proxima-gguf` needs no architecture-name
  enum: `proxima-gguf/src/parser.rs` and `value.rs` offer generic key and tensor accessors, and name
  dispatch already lives in proxima-model-interop's profile selection.
- **masked block (R3)**: input `[id_last, <mask> * (block_size-1)]` is a row layout the host shape
  pipe produces (`decode-as-data` slice 2); non-causal attention is `seq_member` one column of ones
  and constant `mask_pos` (`decode-as-data` table of configurations, "bidirectional block"), selected
  by config. The convs are graph ops over existing `Elementwise`/`Reduce`; no new `Op` variant.
- **target tap and injection (R4/R5)**: the tap is the readout list (`decode-as-data` R5, slice 3), not
  a method on a type. Injection feeds the gathered features as the draft's embd-input batch after
  every target forward, mirroring `process()` (`common/speculative.cpp:1098-1181`), and reads the
  draft's block output before verify. The existing token-id-only n-gram drafters
  (`proxima-model-interop/src/generate/drafter.rs`) do not fit a float embd-input, which is why the
  draft-model path (`proxima-model-interop/src/generate/draft_model.rs:174-183`, `load_draft`) is the
  precedent this follows, not that file's enum.
- **block decode (R6/R7)**: DFlash1's top-k sampler and DFlash2's lattice chain are two propose
  configurations selected by `selector_top_k > 0` (`common/speculative.cpp:1237-1265` versus `:1295-1318`),
  two branches because they are mutually exclusive per model and produce tokens through different
  math (argmax over a lattice versus top-k sampling); one function behind a boolean would hide the
  branch.
- **fixture generator (R12)**: a C++ harness in the shape of the parent's n-gram generator
  (`proxima-tokenizer/tests/fixtures/llama-ngram/`), compiled against llama.cpp `f1ea20621`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| target for the parity fixture | `Qwen/Qwen3-4B` plus `z-lab/Qwen3-4B-DFlash` (`docs/speculative.md:63-72`, the only DFlash pairing named in this checkout's docs) | no gemma4-paired non-Markov DFlash model is named upstream; the gemma4-paired speculator (`RedHatAI/gemma-4-31B-it-speculator.dspark`, `docs/speculative.md:110`) is DSpark's model, not this sub-spec's |
| Qwen3 target support | not added here | a Qwen3 dense target profile is its own initiative; conflating it would make R11c's scope unbounded |
| structural versus end-to-end requirements | split explicitly (R1-R5, R8, R9, R11, R12 need no target parity; R6, R7, R10 do) | lets the sub-spec land its structural slices even if the model gap (refutation condition) takes longer than the draft-side work |
| DFlash2 selector math | its own branch, not folded into DFlash1's sampler | see the architecture bullet on block decode |
| dspark's anchor-first, Markov-head and confidence-truncation paths | out of scope | owned by `speculative-draft-dspark`, per the parent's R11a-d split |

## acceptance criteria

`$DFLASH_DFT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/qwen3-4b-dflash.gguf`, the draft GGUF slice 1 places (converted from `z-lab/Qwen3-4B-DFlash` with `--target-model-dir Qwen/Qwen3-4B`) at exactly this path
`$DFLASH_TGT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/qwen3-4b.gguf`, the Qwen3-4B target GGUF slice 1 places at exactly this path
`$LLAMA_SPEC_SIMPLE` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple`, built from llama.cpp `f1ea20621` by slice 0b (the checkout's prebuilt `build/bin` predates the source and registers no `--spec-type`); it prints `n_drafted = N` (`examples/speculative-simple/speculative-simple.cpp:355`)
`$DFLASH2_DFT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/selector-dflash.gguf`, a draft GGUF whose `dflash.selector_top_k` is above 0, which slice 1b places at exactly this path; `$DFLASH2_TGT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/selector-target.gguf`, the target that checkpoint names via `--target-model-dir`, placed by slice 1b
Fixtures live under `proxima-model-interop/tests/fixtures/llama-dflash/fixtures/`.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R10 (prerequisite) | `strings -a "$DFLASH_DFT" \| grep -c 'dflash.block_size'` | >= 1 (the file is a real `dflash` draft) |
| AC2 | R12 | `for name in dflash1 process block conv params gguf_dump; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/$name.json; done` | six lines printed, in order: >= 50, >= 10, >= 1, >= 50, >= 1, >= 1 |
| AC3 | R1 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_descriptor_reads_known_keys/)'` | 1 passed; prints `keys_read = 5`; oracle: the five `dflash.*` key values in `gguf_dump.json`, produced by llama.cpp `f1ea20621`'s `llama-gguf "$DFLASH_DFT" r n` |
| AC4 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_tensor_names_resolve/)'` | 1 passed; prints `tensors_resolved = N` with N >= 7; oracle: the tensor-name list in `gguf_dump.json` from the same llama.cpp dump |
| AC5 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_masked_block_shape/)'` | 1 passed; prints `block_size = N mask_positions = N-1` and asserts the block row layout equals `block.json`, the layout llama.cpp `f1ea20621` builds at `common/speculative.cpp:1205-1210` for `$DFLASH_DFT` |
| AC6 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_target_tap_matches_llama_fixture/)'` | 1 passed; prints `positions = N layers = L max_abs_diff < 1e-4` with N >= 10 and L >= 1 (L equals the `target_layer_ids` count in `process.json`); oracle: `process.json`, the rows llama.cpp `f1ea20621` returns from `llama_get_embeddings_layer_inp` |
| AC7 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_features_reach_draft_in_one_decode_call/)'` | 1 passed; prints `decode_calls_per_chunk = 1` over N >= 10 chunks; oracle: the per-chunk `llama_decode` call count (1) recorded in `process.json` by the harness at `common/speculative.cpp:1098-1181` |
| AC8 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash1_block_decode_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50; oracle: `dflash1.json`, drafted ids from llama.cpp `f1ea20621` (gated by the refutation condition) |
| AC9 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash2_selector_decode_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50; oracle: `dflash2.json`, drafted ids from llama.cpp `f1ea20621` run on `$DFLASH2_DFT` with `$DFLASH2_TGT` (gated by the refutation condition) |
| AC10 | R8 | `git grep -ciE 'DflashRollback\|dflash.*snapshot' -- proxima-model-interop/src proxima-tensor/src; git grep -c 'fn truncate' -- proxima-model-interop/src` | the first command prints no file with a match (0 matches); the control prints >= 1 (the attention-KV truncate `LayerCache::truncate` the rollback reuses, so the grep is searching the right tree) |
| AC11 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_draft_dflash_defaults_match_llama/)'` | 2 passed (builder and loader each produce a `draft-dflash` sub-config whose `n_max`, `n_min`, `p_min`, `backend_sampling` equal the defaults recorded in `params.json`); oracle: `params.json`, the `common_params_speculative_draft` defaults printed by llama.cpp `f1ea20621`'s harness |
| AC12 | R11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_oracle_control_/)'` | 12 passed: for each of the six comparators (ids, gguf keys and tensor names, block layout, tap tolerance, decode-call count, config defaults) one `dflash_oracle_control_self_<comparator>` test (llama's own fixture on both sides passes) and one `dflash_oracle_control_perturbed_<comparator>` test (one perturbed value is rejected, asserted) |
| AC13 | R10 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_llama_parity_seed_/)'` | 3 passed (seeds 1, 2, 3); each prints `ids_compared = N` with N >= 8; oracle: ids vendored in `ids.json` from `$LLAMA_SPEC_SIMPLE --model "$DFLASH_TGT" --model-draft "$DFLASH_DFT" --spec-type draft-dflash --seed <seed> -n 40` (gated by the refutation condition) |
| AC14 | R15 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/ids.json` | 3 (every vendored oracle run drafted, so an oracle that drafted nothing cannot back AC13) |
| AC15 | R7 (prerequisite) | `strings -a "$DFLASH2_DFT" \| grep -c 'dflash.selector_top_k'` | >= 1 (the file carries the selector key; the value above 0 is asserted by AC16's `selector_top_k` field) |
| AC16 | R12 | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/dflash2.json` | >= 50, and `jq '.selector_top_k' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/dflash2.json` prints a number >= 1 |
| AC17 | R13 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_attention_causality_switch/)'` | 2 passed (`dflash.attention.causal` unset gives the all-ones visibility; set gives the lower-triangular one); oracle: the `attn_causal` and mask records in `block.json` from llama.cpp `f1ea20621` for the same two settings |
| AC18 | R14 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_local_conv_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 50, `max_abs_diff < 1e-4`; oracle: `conv.json`, the per-layer conv outputs llama.cpp `f1ea20621` computes |

## out of scope

- draft-dspark (Markov head, anchor-first layout, confidence-head truncation) -- R11d, `speculative-draft-dspark`
- adding Qwen3 as a supported target profile in proxima-model-interop -- a separate initiative; this sub-spec needs it only for the end-to-end fixture (AC13), not for the structural ACs
- M-RoPE position handling for DFlash embd batches (`is_mrope`, the 4-position-row fix of `b10f9ca58`) -- no M-RoPE target exists in proxima yet; unblocked when a vision or M-RoPE target lands
- a new HF-to-GGUF converter for `general.architecture` equal to `dflash` if no pre-converted GGUF exists -- llama.cpp upstream work (PR #22105 per `docs/speculative.md:79`); slice 1 names the exact missing converter class and stops there rather than taking on an upstream C++ or Python contribution
- the readout list, visibility inputs and shape pipes themselves (`decode-as-data` slices 1-3) -- owned by that spec; this sub-spec consumes them

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no `draft-dflash` GGUF pair is obtainable (the refutation condition) | medium-high | R6, R7, R10 stay unmeasured; the sub-spec caps at structural-only | slice 1 spends its full budget on acquisition and reports the exact blocker (pre-converted GGUF search versus missing converter) before any implementation slice starts |
| DFlash's draft backbone is Qwen3-shaped only (confirmed for DSpark's Markov head at `docs/speculative.md:106-107`, not independently confirmed for plain DFlash) | medium | the draft profile may need Qwen3 dense-layer shapes even for the structural ACs | read `z-lab/Qwen3-4B-DFlash`'s `config.json` layer stack before writing the profile; if Qwen3-shaped, the profile reuses the Qwen3 layer shape plus the conv/selector tensors |
| the tap readout changes the lowered program's shape for every family, not just the target | low | wider blast radius than this sub-spec's scope | the readout list is opt-in per call (`decode-as-data` AC7's logits-only digest test) |
| the fused-encoder shape (`662a0b012`) drifts from a pre-fusion oracle if the harness is built against an older llama.cpp | low | parity tests pass against a stale oracle | the harness pins `f1ea20621`, recorded in each fixture header |

## context

- proxima: `proxima-tensor/src/spec/descriptor.rs:130-296` (`ModelDescriptor`), `proxima-model-interop/src/lowering.rs:118-189` (`BoundProgram`; `hidden_root` at `:137`, `residual_roots` at `:141`), `proxima-model-interop/src/profiles/qwen3.toml`, `proxima-model-interop/src/generate/draft_model.rs:174-183` (`load_draft`), `proxima-model-interop/src/generate/drafter.rs`, `proxima-model-interop/src/speculative_settings.rs:36-59` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205-217` (`SpeculativeType`), `proxima-gguf/src/parser.rs`, `proxima-gguf/src/value.rs`, `proxima-model-interop/src/generate/residency_caches.rs` (`LayerCache::truncate`), `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp:923-1329` (`common_speculative_impl_draft_dflash`: ctor `959-1064`, `process` `1098-1181`, `draft` `1183-1324`), `src/llama-arch.cpp:146,370-374,702-708` (`LLM_ARCH_DFLASH`, KV keys, tensor names), `src/llama-hparams.h:250-254`, `src/llama-ext.h:96-128`, `common/common.h:173-186,327-352`, `common/arg.cpp:4137,4194,4201,4238-4244,4245-4254`, `docs/speculative.md:55-98` (DFlash/DSpark usage, named models, `--spec-draft-n-max` clamped to the trained block size), `examples/speculative-simple/speculative-simple.cpp:355`, `examples/gguf/gguf.cpp`, `convert_hf_to_gguf.py` (`--target-model-dir` at line 165 is generic, not dflash-specific)
- upstream commits: `662a0b012` (encoder fused into KV-cache injection decode), `b10f9ca58` (DFlash2 local convolution and candidate selector)
