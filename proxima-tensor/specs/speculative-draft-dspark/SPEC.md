# speculative-draft-dspark

status: audited
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)

## problem

`speculative-decode-llama-parity`'s R11d requires an audited sub-spec for `draft-dspark`
(`COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`, `common/common.h:179`), and proxima has no DSpark support
(`git grep -ciE 'dspark|markov' -- proxima-model-interop/src proxima-gguf/src proxima-tensor/src` lists
only the llama type name in `serving.rs` and `speculative_settings.rs`); DSpark is dispatched through the
same llama.cpp struct as DFlash, `common_speculative_impl_draft_dflash`, constructed with
`type = COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` (`common/speculative.cpp:2654-2656`), so this sub-spec adds
DSpark's two deltas over `speculative-draft-dflash` -- a semi-autoregressive Markov head that conditions
each in-block token on the previous in-block token's identity, and an optional per-token confidence head
that truncates the draft -- as extra head readout roots on the DFlash draft descriptor
(`decode-as-data` R5, R6) plus two metadata reads, with proxima's generated token ids equal to llama.cpp
`f1ea20621`'s on a real DSpark target and draft pair, and adds no technique-named type
(`decode-as-data` AC9).

## refutation condition

Written before evidence: if, across a 200-block real-GGUF corpus, no drafted token ever differs between
"Markov bias and confidence computed" and "Markov bias and confidence skipped" (AC13 prints
`differing_blocks = 0`), then R3 and R4 are decoration, DSpark collapses into DFlash's sub-spec as one
metadata flag, and this sub-spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a GGUF whose `general.architecture` is `dflash` and which contains tensor `markov_w1.weight` classifies as the DSpark speculation type, and one without it as the DFlash type, equal to `common_speculative_types_from_gguf` (`common/speculative.cpp:2289-2303`) | yes |
| R2 | the draft descriptor resolves `markov_w1.weight`, `markov_w2.weight`, the optional `markov_w2.scale`, and the optional `conf_proj.weight`/`conf_proj.bias` (`src/llama-arch.cpp:699-701`, `gguf-py/gguf/constants.py:1204-1207,1989-1991`) | yes |
| R3 | the metadata keys `dflash.sample_from_anchor` and `dflash.has_confidence_head` (`common/speculative.cpp:987,1002`; written by `gguf_writer.py:1040,1043`) are read into the descriptor | yes |
| R4 | the Markov bias `markov_w2 @ markov_w1[prev_token]` (lora_mm form, with the optional int8 scale) is computed per block position and added to that position's raw logits before sampling, equal to `build_dspark_markov_head` (`src/models/dflash.cpp:295-406`) for the same block and previous-token sequence | yes |
| R5 | when `has_confidence_head` is true and `p_min > 0.0`, the per-position confidence `sigmoid(conf_proj . [feat(i); markov_w1[prev(i)]] + bias)` truncates the draft at the first position where it is below `p_min`, equal to the DSpark branch of the draft loop (`common/speculative.cpp:1267-1294`) | yes |
| R6 | when `p_min > 0.0` and `has_confidence_head` is false, drafting returns a typed error equivalent to llama.cpp's `std::runtime_error("DSpark draft has no confidence head...")` (`common/speculative.cpp:1005`) instead of panicking or sampling unconditionally | yes |
| R7 | the block layout follows `sample_from_anchor` read from GGUF metadata, never hardcoded: true samples a full `block_size` tokens from position 0 (anchor-first, `common/speculative.cpp:1016,1205,1271`); false samples `block_size - 1` tokens from positions 1..block_size-1, the bonus-anchor layout of commit `9cd719af2` | yes |
| R8 | the Markov bias (R4) and confidence gate (R5) are load-bearing: across a 200-block real-GGUF corpus at least one drafted token differs between "computed" and "skipped" -- the measurable form of the refutation condition | yes |
| R9 | for the same target, draft, prompt and seed, proxima's generated token ids with `draft-dspark` enabled equal the ids llama.cpp `f1ea20621`'s `llama-speculative-simple` generates with `--spec-type draft-dspark --model-draft`, for 3 seeds | yes |
| R10 | every vendored oracle run itself drafted (its recorded `n_drafted` >= 1), so an oracle that silently drafted nothing cannot back R9 | yes |
| R11 | the oracle comparators are themselves validated: for each comparator, llama's own fixture on both sides passes, and one perturbed value is rejected | yes |
| R12 | the vendored fixtures are produced by a C++ harness linking llama.cpp `f1ea20621` and calling the public API the DSpark branch calls, never re-derived from reading the C++; each fixture records the upstream commit in its header, and the bias fixture includes both a scaled and an unscaled checkpoint's cases | yes |
| R13 | no Rust type, trait, enum or impl is named for DSpark: the Markov head is readout roots and a propose-pipe branch on the DFlash path, with no duplicate of DFlash's block batching, feature extraction or injection | yes |

## architecture

Nothing here is a new type. DSpark is the DFlash draft descriptor (`speculative-draft-dflash`) plus, when
`markov_w1.weight` exists, extra head readout roots (`decode-as-data` R5, "extra head roots"), mirroring
llama.cpp's own `is_dspark` boolean on the shared struct (`common/speculative.cpp:948,963`) rather than a
second class. Its row in the `decode-as-data` table is the DFlash row: "masked block on a draft descriptor
| block | sample-and-match on the target | prefix".

- **descriptor (R2/R3)**: the Markov tensors and the two metadata keys are fields on the draft
  `ModelDescriptor` (`proxima-tensor/src/spec/descriptor.rs:130`) the DFlash sub-spec already extends,
  populated at load only when `markov_w1.weight` exists (`src/models/dflash.cpp:124-135`). The
  confidence head is optional (`TENSOR_NOT_REQUIRED`, `src/models/dflash.cpp:132-133`). Plain borrowed
  tensor views, box-free, no new trait.
- **Markov and confidence roots (R4/R5)**: lowered by `build_forward` as head roots in the readout list;
  `Op::Input` carries the previous in-block token. After the block's raw logits are produced, the
  propose pipe adds the bias (R4) and, when a confidence head exists and `p_min > 0.0`, applies the
  gate (R5) before returning the draft. `sample_from_anchor` (R7) selects the loop's start index and
  length and is read once at bind time, never per call.
- **classification (R1)** is a pure function over parsed metadata and tensor names, on the auto-detect
  path DFlash and MTP already share (`common_speculative_types_from_gguf`, `common/speculative.cpp:2280-2306`):
  dflash architecture without `markov_w1.weight` is DFlash, with it is DSpark.
- **fixture generator (R12)**: a C++ harness compiled against llama.cpp `f1ea20621`, calling the same
  graph builder and draft-loop branch.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| DSpark as descriptor data on DFlash's draft | extra head roots plus two metadata reads | a second program duplicates DFlash's KV injection and block batching (principle 1); llama.cpp itself dispatches DSpark through the identical struct (`common/speculative.cpp:2654-2656`) |
| confidence-gate failure mode | typed error | llama.cpp hard-fails with `runtime_error` (`common/speculative.cpp:1005`) rather than silently sampling unconditionally (principles 14, 15) |
| `sample_from_anchor` source | GGUF metadata only, read once at bind | llama.cpp never exposes it as a CLI override (`common/speculative.cpp:987-989`); a flag would let proxima diverge from the incumbent's per-model contract |
| oracle | llama.cpp `f1ea20621`'s `build_dspark_markov_head` and the DSpark draft-loop branch, through a compiled harness and `llama-speculative-simple` | re-deriving the formulas from reading the C++ is principle 14's "from memory" failure |
| unit versus end-to-end fixtures | both: harness values for R4/R5, vendored ids for R7/R9 | R4/R5 need internal values no end-to-end run exposes; R7/R9 need the real binary as the whole-system oracle |

## acceptance criteria

`$DSPARK_DFT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/dspark.gguf`, the DSpark draft GGUF slice 1 places at exactly this path
`$DSPARK_TGT` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/target.gguf`, the target compatible with its Markov-head vocabulary, placed by slice 1 at exactly this path
`$LLAMA_SPEC_SIMPLE` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple`, built from llama.cpp `f1ea20621` by slice 0b (the checkout's prebuilt `build/bin` predates the source and registers no `--spec-type`); it prints `n_drafted = N` (`examples/speculative-simple/speculative-simple.cpp:355`)
Fixtures live under `proxima-model-interop/tests/fixtures/llama-dspark/fixtures/`.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R9 (prerequisite) | `strings -a "$DSPARK_DFT" \| grep -c 'markov_w1.weight'; strings -a "$DSPARK_DFT" \| grep -c 'dflash.sample_from_anchor'` | two lines printed, each >= 1 (the file is a real DSpark draft, not a plain DFlash one) |
| AC2 | R12 | `for name in classify bias confidence layout errors gguf_dump; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/$name.json; done` | six lines printed, in order: >= 3, >= 200, >= 200, >= 2, >= 2, >= 1 |
| AC3 | R12 | `jq '[.cases[] \| select(.scaled == true)] \| length' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/bias.json; jq '[.cases[] \| select(.scaled == false)] \| length' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/bias.json` | two lines printed, each >= 1 |
| AC4 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/classifies_dspark_from_markov_tensor/)'` | 1 passed; prints `cases = N` with N >= 3; oracle: `classify.json`, the type llama.cpp `f1ea20621`'s `common_speculative_types_from_gguf` returns for `$DSPARK_DFT`, for the plain DFlash draft `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/qwen3-4b-dflash.gguf` (placed by `speculative-draft-dflash` slice 1), and for a synthetic dflash GGUF without `markov_w1.weight` |
| AC5 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_descriptor_resolves_markov_and_confidence_tensors/)'` | 1 passed; prints `tensors_found = N` on `$DSPARK_DFT` with N equal to how many of the five names `markov_w1.weight`, `markov_w2.weight`, `markov_w2.scale`, `conf_proj.weight`, `conf_proj.bias` appear in `gguf_dump.json` (2 <= N <= 5), and `tensors_found = 2` on a synthetic GGUF written with exactly `markov_w1.weight` and `markov_w2.weight`; oracle: the tensor-name list in `gguf_dump.json`, from llama.cpp `f1ea20621`'s `llama-gguf "$DSPARK_DFT" r n` |
| AC6 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_descriptor_reads_anchor_and_confidence_keys/)'` | 1 passed; prints `keys_read = 2`; oracle: the two key values in `gguf_dump.json` |
| AC7 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_markov_bias_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 200, `max_abs_diff < 1e-4`; oracle: `bias.json`, the bias vectors llama.cpp `f1ea20621` computes in `build_dspark_markov_head` |
| AC8 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_confidence_gate_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 200 and `truncated_cases >= 1`; oracle: `confidence.json`, the confidences and truncation points from llama.cpp `f1ea20621`'s DSpark branch |
| AC9 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_missing_confidence_head_is_typed_error/)'` | 2 passed (no confidence head with `p_min > 0.0` is the typed error; with `p_min = 0.0` drafting succeeds); oracle: `errors.json`, the two outcomes llama.cpp `f1ea20621` records for the same two configurations |
| AC10 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_sample_from_anchor_layout/)'` | 2 passed (anchor-first and bonus-anchor); oracle: `layout.json`, the start index and token count llama.cpp `f1ea20621` uses for each setting |
| AC11 | R13 | `git grep -niP '\b(trait\|impl\|struct\|enum)\b[^;{]*Dspark' -- proxima-model-interop/src proxima-tensor/src proxima-gguf/src \| wc -l; git grep -niP '\b(trait\|impl\|struct\|enum)\b[^;{]*Ngram' -- proxima-model-interop/src proxima-tensor/src proxima-gguf/src \| wc -l` | two lines printed: `0`, then a number >= 1 (the control: the same pattern shape finds the n-gram types, so the first 0 is a real absence) |
| AC12 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_llama_parity_seed_/)'` | 3 passed (seeds 1, 2, 3); each prints `ids_compared = N` with N >= 8; oracle: ids in `ids.json` from `$LLAMA_SPEC_SIMPLE --model "$DSPARK_TGT" --model-draft "$DSPARK_DFT" --spec-type draft-dspark --seed <seed> -n 40` (llama.cpp `f1ea20621`) |
| AC13 | R8 | `cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model "$DSPARK_DFT" "$DSPARK_TGT" --blocks 200 > ablation.txt; grep -cE 'differing_blocks=[1-9]' ablation.txt` | 1 (the line `differing_blocks=N blocks=200` with N >= 1; a printed `differing_blocks=0` matches 0 lines and invokes the refutation condition) |
| AC14 | R10 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/ids.json` | 3 |
| AC15 | R11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_oracle_control_/)'` | 14 passed: for each of the seven comparators (ids, classification, tensor names and keys, bias floats, confidence, layout, error outcome) one `dspark_oracle_control_self_<comparator>` test (llama's own fixture on both sides passes) and one `dspark_oracle_control_perturbed_<comparator>` test (one perturbed value is rejected, asserted) |
| AC16 | R12 | `for name in classify bias confidence layout errors ids gguf_dump; do jq -r '.header.upstream_commit' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/$name.json; done \| grep -c '^f1ea20621'` | 7 (every fixture's header records the llama.cpp commit it was generated from) |

## out of scope

- DFlash's own correctness (block layout input, tap, injection, conv, DFlash1 and DFlash2 decode) -- owned by `speculative-draft-dflash`; this sub-spec adds only the Markov and confidence delta on top of it
- non-vanilla Markov head types (`markov_head_type != "vanilla"`) -- every landed backbone (Gemma4, Qwen3.6, DeepSeekV4, LFM2, BailingMoE3, Nemotron3.5) ships vanilla; a non-vanilla head is a new upstream feature not in `f1ea20621`
- speculators-format checkpoint conversion (d2t remap, reduced draft vocab expansion) -- llama.cpp's `conversion/qwen.py` concern; proxima consumes the already-converted GGUF
- DFlash2's lattice selector (`is_dflash2`) -- a DFlash-only path DSpark never takes (`common/speculative.cpp:1237-1238` gates it on `is_dflash2`, mutually exclusive with the DSpark branch at `:1267`)
- the readout list (`decode-as-data` R5, slice 3) and the DFlash descriptor and loop -- owned by those specs; this sub-spec consumes them

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no real DSpark GGUF pair is obtainable | medium | AC7, AC8, AC10, AC12, AC13 block on slice 1; the classification, tensor-name and key reads still land against synthetic-but-real-shaped GGUFs written by `proxima-gguf`'s writer | slice 1 names the checkpoint llama.cpp commit `9cd719af2`'s message verified (`gemma4-26b-a4b-dspark`, "greedy outputs are byte-identical... acceptance 0.46, mean draft len 3.7"); if unobtainable, `satgeze/Qwen3.6-27B-DSpark` (`conversion/qwen.py:791`) is the fallback |
| `markov_w2.scale` (int8 lora-mm scale, loaded as `dspark_markov_w2_s`, commit `ca3d5a3e1`) is present on some checkpoints and absent on others | medium | R4 branches on its presence; a fixture covering one case under-tests | AC3 requires scaled and unscaled cases in `bias.json` (synthesized if only one real file is available) |
| `conf_proj` is `TENSOR_NOT_REQUIRED` (`src/models/dflash.cpp:132-133`); some real checkpoints have no confidence head | high ("vanilla-markov exports ship without it", commit `ca3d5a3e1`) | R5 and R6 both depend on both states being exercised | AC5 and AC9 use the real checkpoint and a minimal no-confidence synthetic GGUF |

## context

- proxima: `proxima-tensor/src/spec/descriptor.rs:130` (`ModelDescriptor`), `proxima-model-interop/src/lowering.rs:118-141` (`BoundProgram`), `proxima-model-interop/src/speculative_settings.rs:36` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205` (`SpeculativeType`), `proxima-gguf/src/parser.rs`, `proxima-gguf/src/value.rs`, `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-model-interop/examples/data/speculative_corpus.jsonl`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-draft-dflash/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp` (dispatch table `:39`, enum-to-name `:2236`, struct `:922-1330`, `is_dspark` field `:948`, ctor `:963`, confidence-head-required check `:999-1006`, `n_draft_max` `:1016`, block-token count `:1205`, DSpark draft-loop branch `:1267-1294`, GGUF auto-detect `:2280-2306`, block-draft output sizing `:2486-2498`, factory dispatch `:2629,2654-2656`), `common/common.h` (enum `:173-186`), `src/llama-arch.h:706-708`, `src/llama-arch.cpp:699-701,989-992`, `src/models/dflash.cpp` (tensor creation `:124-135`, `build_dspark_markov_head` `:295-406`, call sites `:846-847,1025-1026`), `gguf-py/gguf/constants.py:1204-1207,1989-1991,5263-5265`, `gguf-py/gguf/gguf_writer.py:1040,1043`, `examples/speculative-simple/speculative-simple.cpp:355`, `examples/gguf/gguf.cpp`
- upstream history, most to least load-bearing: `84075273c` (DSpark's original landing), `9cd719af2` (speculators-format checkpoints, `bonus_anchor` to `sample_from_anchor`, verified `gemma4-26b-a4b-dspark` acceptance numbers), `ca3d5a3e1` (Nemotron3.5, optional confidence head, `markov_w2_s` scale tensor), `0d0bfcd4f` (backend sampling for dflash and dspark), `f65e568fd` (GGUF metadata auto-detect), `633733d0a` (Gemma4 DSpark draft backbone), `2115b73d8`/`07822bddf`/`d646c9d15` (BailingMoE3/LFM2/DeepSeek-V4 backbone support, confirming DSpark is backbone-agnostic on the draft side)
