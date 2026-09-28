# speculative-draft-dflash

status: audited
owner: brian bruggeman
created: 2026-09-28

## problem

proxima-model-interop has no representation of a second, differently-shaped model at all
(`Architecture::bind` and `Architecture::speculative_verify_program`,
`proxima-model-interop/src/architecture.rs:277-309`, both take one `ParsedGguf` / one
`file_bytes` and return one `BoundProgram`), no cross-model KV injection, no block-diffusion
decode loop, and no `dflash`-arch tensor/metadata mapping, so llama.cpp's `draft-dflash`
speculation type (`common/speculative.cpp:923-1329`, upstream `f1ea20621`) -- which drafts a
whole block of tokens per step by injecting the target model's hidden states into a second,
smaller model's KV cache -- has zero coverage on the proxima side; this sub-spec of
`proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md` (R11c) makes `draft-dflash`
available through `ServingConfig`'s speculative section, with proxima's own drafted-and-
accepted tokens byte-identical to llama.cpp's, on real GGUFs on disk.

## refutation condition

Written before evidence: if no `draft-dflash`-typed GGUF pair (target + draft) can be
obtained or produced within slice 1 -- no pre-converted GGUF is published for a locally
reachable model, and `convert_hf_to_gguf.py` at `f1ea20621` has no registered HF-to-GGUF
converter for `general.architecture == "dflash"` (confirmed this session: zero matches for a
`dflash`-named `ModelBase` subclass, `convert_hf_to_gguf.py` grep, this session) -- then this
sub-spec's end-to-end parity requirements (R5, R6, R9) are struck down to structural-only
coverage (R1-R4, R7, R8) and the sub-spec cannot be audited past `draft` until the model gap
is closed by a separate, explicitly-scoped initiative.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | proxima-gguf/proxima-model-interop parses a `dflash`-arch GGUF's metadata keys (`dflash.block_size`, `dflash.conv_kernel_size`, `dflash.conv_group_size`, `dflash.selector_rank`, `dflash.selector_top_k`, mirroring `src/llama-hparams.h:250-254`) and tensor names (`blk.%d.attn_conv_base/proj`, `blk.%d.ffn_conv_base/proj`, `selector_predecessor/successor/hidden`, `src/llama-arch.cpp:702-708`) into a typed `DflashMetadata` without erroring on a checkpoint that has them | yes |
| R2 | a `DflashArch` (or equivalent) `Architecture` impl binds the draft model's forward program: masked-block input `[id_last, <mask> * (block_size-1)]`, non-causal attention unless `dflash.attention.causal` is set, local convolution (attn/ffn conv base+proj) per layer | yes |
| R3 | target-side layer-output extraction: the target's bound program can emit the `target_layer_ids` layers' input hidden states for every new position, gathered zero-copy into the `[n_chunk, n_embd_enc]` shape the draft's encoder consumes (mirrors `llama_set_embeddings_layer_inp`/`llama_get_embeddings_layer_inp`, `src/llama-ext.h:111,115`, and the gather loop at `common/speculative.cpp:1145-1156`) | yes |
| R4 | the fused encoder path (post `662a0b012`): target features are consumed directly by the draft's embd-input decode in one call, never round-tripped through a separate encode-then-reupload step | yes |
| R5 | DFlash1 (non-selector) block decode: top-k=10 sampler per masked position, `p_min` early-stop, greedy token ids byte-identical to llama.cpp's own draft output given the same injected features and RNG state (`common/speculative.cpp:1291-1316`) | yes |
| R6 | DFlash2 (selector) block decode: `h_nextn` read as a `[block, selector_top_k]` lattice, greedy predecessor-argmax chain with softmax `p_min` truncation, deterministic given the lattice -- no resampling (`common/speculative.cpp:1235-1259`, gated by `llama_model_dflash_selector_top_k(model) > 0`) | yes |
| R7 | draft-dflash's own `accept()` is a no-op (`common/speculative.cpp:1327-1328`) -- no dflash-specific rollback state exists beyond parent spec's R3a (attention KV truncate); this sub-spec adds zero new rollback machinery and states that explicitly | yes |
| R8 | `ServingConfig`'s speculative section (parent R9) grows a `draft_dflash` sub-config: `block_size` (0 = read from GGUF metadata), `n_max`, `n_min`, `p_min`, `backend_sampling`, mirroring `common_params_speculative_draft` (`common/common.h:327-334`) and the `--spec-draft-n-max`/`--spec-draft-p-min`/`--spec-draft-backend-sampling` flags (`common/arg.cpp:4137,4194,4201`) | yes |
| R9 | end-to-end: on the SAME target+draft GGUF pair, proxima's drafted tokens (pre-verify) are byte-identical to llama.cpp's own `draft-dflash` output for the same prompt and seed | yes, but gated by the refutation condition |

## architecture

Two new pure pieces plus one loop-seam extension, all sans-IO, composing with the parent
spec's existing `enum Drafter` (parent SPEC.md architecture section):

- **`DflashMetadata`** (`proxima-gguf`): a typed read of the `dflash.*` KV keys plus the
  `attn_conv_*`/`ffn_conv_*`/`selector_*` tensor name table, returned alongside `ParsedGguf`
  the same way any other architecture's hparams are read today -- no new crate, no new
  `enum` of "known architectures" in proxima-gguf (that dispatch already lives in
  proxima-model-interop, confirmed this session: `proxima-gguf/src/*.rs` has no
  architecture-name enum, only generic KV/tensor accessors).
- **`DflashArch`**: a new `Architecture` impl, added to the existing closed set proxima
  already matches on by name (same shape as `Gemma4Arch`, not a new dispatch mechanism).
  `bind` builds the draft's masked-block forward program; `speculative_verify_program` is
  unused here (dflash is a *draft* type, not a verify-capable target -- verify always runs
  on the target's own program, parent R1/R2).
- **Target-feature extraction**: a capability method on `Architecture`, default `Ok(None)`
  like `speculative_verify_program`, that a target's bound program can implement to emit
  named layers' input hidden states per new position. Gemma4 gains this override (it is
  proxima's only real target architecture today).
- **Cross-model KV injection**: the decode loop, when running `Drafter::Dflash(..)`, calls
  the target's extraction hook after every target forward, feeds the gathered features as
  the draft's embd-input batch (mirrors `process()`, `common/speculative.cpp:1098-1181`),
  and reads the draft's block output before verify -- one new loop branch in `decode.rs`,
  not a new `Drafter` trait shape (the existing `begin`/`draft`/`accept` signature over
  token ids does not fit dflash's float-embd input, so `Drafter::Dflash` is the one variant
  that also carries the injected-features step; documented as the one place the closed enum
  is not uniform, same discipline as `Drafter::DraftModel` already needing a second GGUF).

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| target architecture for the parity fixture | `Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash` (`docs/speculative.md:63-72`, the only DFlash pairing named anywhere in this checkout's docs) | no gemma4-paired `draft-dflash` (non-Markov) model is named upstream; only a gemma4-paired *DSpark* speculator exists (`RedHatAI/gemma-4-31B-it-speculator.dspark`, `docs/speculative.md:110`), and that is R11d's model, not this sub-spec's |
| Qwen3 target support | NOT added by this sub-spec | adding a Qwen3 dense target architecture to proxima-model-interop is its own initiative with its own spec; conflating it here would make R11c's scope unbounded, violating spec-first's "each requirement testable in isolation" |
| structural ACs vs end-to-end ACs | split explicitly (R1-R4, R7, R8 need no target-arch parity; R5, R6, R9 do) | lets the sub-spec be audited and slice-landed even if the model/target-arch gap (refutation condition) takes longer to close than the draft-side implementation |
| DFlash2 selector math | ported as its own branch, not folded into DFlash1's sampler loop | the two are mutually exclusive per-model (`is_dflash2` gates which loop runs, `common/speculative.cpp:1235` vs `:1291`) and produce tokens through entirely different math (argmax over a lattice vs top-k sampling); one function doing both would hide the branch behind a boolean, contrary to the parent spec's closed-enum-over-boolean-flags discipline |
| dspark's anchor-first / Markov-head / confidence-truncation paths | out of scope here | owned by R11d / `speculative-draft-dspark` sub-spec, per parent SPEC.md's own split (R11a-d) and its out-of-scope line "draft correctness for mtp/eagle3/dflash/dspark -- owned by their sub-specs" |

## acceptance criteria

`$EX` = `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity --`
`$DFLASH_GGUF` / `$QWEN3_4B_GGUF` = the draft/target GGUF paths recorded by slice 1's note in TASKS.md
(`z-lab/Qwen3-4B-DFlash`, converted with `--target-model-dir Qwen/Qwen3-4B`, or the locally-produced
equivalent if no pre-converted GGUF is found)

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/dflash_metadata_parses_known_keys/)'` | 1 passed; test prints `keys_read = N` with N = 5 (block_size, conv_kernel_size, conv_group_size, selector_rank, selector_top_k) |
| AC2 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/dflash_tensor_names_resolve/)'` | 1 passed; prints `tensors_resolved = N` with N ≥ 7 (attn_conv_base, attn_conv_proj, ffn_conv_base, ffn_conv_proj, selector_predecessor, selector_successor, selector_hidden) |
| AC3 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_masked_block_shape/)'` | 1 passed; prints `block_size = N mask_positions = N-1` matching the fixture GGUF's `dflash.block_size` |
| AC4 | R3,R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/target_layer_extraction_feeds_draft_embd_in_one_call/)'` | 1 passed; prints `llama_process_calls = 1` per chunk (no separate encode pass) |
| AC5 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash1_block_decode_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 50 (gated by refutation condition -- REQUIRES the model from slice 1) |
| AC6 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash2_selector_decode_matches_llama_fixture/)'` | 1 passed; `cases` ≥ 50 (gated by refutation condition) |
| AC7 | R7 | `git grep -c 'DflashRollback\|dflash.*snapshot' -- proxima-model-interop/src` | 0 matches (no dflash-specific rollback type exists) |
| AC8 | R8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_draft_dflash_builder_matches_loader/)'` | 1 passed |
| AC9 | R9 | `$EX --drafter draft-dflash --draft-model "$DFLASH_GGUF" "$QWEN3_4B_GGUF" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40` | exit 0, both config blocks `identical = true`, `speculative_verify_steps` ≥ 1 (gated by refutation condition) |

## out of scope

- draft-dspark (Markov head, anchor-first layout, confidence-head truncation) -- R11d,
  `speculative-draft-dspark` sub-spec
- adding Qwen3 as a supported target architecture in proxima-model-interop -- a separate,
  unscoped initiative; this sub-spec only requires it for the end-to-end parity fixture
  (AC9), not for the structural ACs (AC1-AC4, AC7, AC8)
- M-RoPE position handling for dflash embd batches (`is_mrope`, `b10f9ca58`'s 4-position-row
  fix) -- no M-RoPE target architecture exists in proxima yet; tracked as a follow-on once a
  vision/M-RoPE target lands
- producing a new HF-to-GGUF converter for `general.architecture == "dflash"` if no
  pre-converted GGUF can be found -- that is llama.cpp upstream work (tracked as PR #22105
  per `docs/speculative.md:79`), not proxima work; if it becomes this sub-spec's blocker,
  slice 1 names the exact missing converter class and stops there rather than silently
  taking on an upstream C++/Python contribution

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no `draft-dflash` GGUF pair is obtainable (refutation condition) | medium-high | R5, R6, R9 stay unmeasured indefinitely; sub-spec caps at "audited, structural-only" | slice 1 spends its full budget on acquisition and reports the exact blocker (pre-converted GGUF search vs missing converter) before any implementation slice starts |
| DFlash's own draft backbone is Qwen3-shaped only (confirmed for DSpark's Markov head at `docs/speculative.md:106-107`; NOT independently confirmed for plain DFlash this session) | medium | `DflashArch::bind` may need Qwen3 dense-layer shapes even for the structural ACs | read `z-lab/Qwen3-4B-DFlash`'s config.json layer stack before writing `DflashArch::bind`; if it is Qwen3-shaped, `DflashArch` binds Qwen3 dense layers plus the dflash-specific conv/selector tensors, not a from-scratch layer stack |
| target-feature extraction hook changes `BoundProgram`'s shape for every architecture, not just gemma4 | low | wider blast radius than this sub-spec's stated scope | default `Ok(None)` (same pattern as `speculative_verify_program`), gemma4-only override, zero change to every other architecture's `bind` |
| fused-encoder shape (`662a0b012`) drifts from a pre-fusion oracle if the fixture generator is built against an older llama.cpp commit | low | parity tests pass against a stale oracle | fixture generator pins `f1ea20621` (this sub-spec's tripwire commit), same discipline as the parent spec's fixture header |

## context

- proxima: `proxima-model-interop/src/architecture.rs:256-309` (`Architecture` trait,
  `bind`, `speculative_verify_program`), `proxima-model-interop/src/gemma4/bind.rs:1105-1116`
  (the one `speculative_verify_program` override), `proxima-model-interop/src/generate/residency_caches.rs:600-606`
  (`LayerCacheState::{Attention, DenseAttention, Ssm, <gemma4 shared-KV>}` -- none shaped for a dflash mask-token block),
  `proxima-gguf/src/{parser,value,restack,writer}.rs` (generic KV/tensor accessors, no
  architecture-name enum), parent spec's `decode.rs` draft/verify loop and `Drafter` enum
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`:
  `common/speculative.cpp:923-1329` (`common_speculative_impl_draft_dflash`, full struct:
  ctor `959-1064`, `process` `1098-1181`, `draft` `1183-1324`), `662a0b012` (encoder fused
  into KV-cache injection decode), `b10f9ca58` (DFlash2 local convolution + candidate
  selector), `src/llama-arch.cpp:146,370-374,702-708` (`LLM_ARCH_DFLASH`, KV keys, tensor
  names), `src/llama-hparams.h:250-254` (`dflash_block_size` etc.), `src/llama-ext.h:96-128`
  (`llama_set/get_embeddings_nextn`, `llama_set/get_embeddings_layer_inp`,
  `llama_model_target_layer_ids{,_n}`, `llama_model_dflash_selector_top_k`),
  `common/common.h:174-185,327-334` (`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`,
  `common_params_speculative_draft`), `common/arg.cpp:4137,4194,4201,4238-4246`
  (`--spec-draft-n-max`, `--spec-draft-p-min`, `--spec-draft-backend-sampling`,
  `--spec-draft-model`/`-md`, `--spec-type`), `docs/speculative.md:55-98` (DFlash/DSpark
  usage docs, named models, `--spec-draft-n-max` clamped to trained block size),
  `convert_hf_to_gguf.py` (no `dflash`-registered converter class found this session --
  `--target-model-dir` flag exists at line 165 but is generic, not dflash-specific)
