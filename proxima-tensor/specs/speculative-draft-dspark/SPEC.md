# speculative-draft-dspark

status: audited
owner: brian bruggeman
created: 2026-09-28

## problem

`speculative-decode-llama-parity`'s R11d requires an audited sub-spec for `draft-dspark`
(`COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`, `common/common.h:179`); proxima has zero DFlash or
DSpark support today (verified: no `dflash`/`dspark`/`markov`/`target_layer_ids` hits in
`proxima-model-interop/src` or `proxima-gguf/src`), and DSpark's own sub-spec (this one)
depends on DFlash's sub-spec (R11c, not yet written) rather than duplicating it -- DSpark is
dispatched through the *same* llama.cpp struct as DFlash, `common_speculative_impl_draft_dflash`,
constructed with `type = COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK` instead of `DRAFT_DFLASH`
(`common/speculative.cpp:2654-2656`), so this spec makes DSpark's two deltas over DFlash --
a semi-autoregressive Markov head that conditions each in-block token on the previous
in-block token's identity, and an optional per-token confidence head that self-truncates
the draft -- available on top of DFlash's all-positions program, token-identical to
llama.cpp `f1ea20621` on a real DSpark GGUF pair.

## refutation condition

Written before evidence: if R3 (Markov bias) and R4 (confidence gate) can be satisfied by
forwarding DFlash's own encoder-conditioned block logits unchanged -- i.e. no drafted token
across a 200-block real-GGUF corpus (AC10) ever differs between "bias/confidence computed"
and "bias/confidence skipped" -- then R3 and R4 are decoration, DSpark collapses into
DFlash's sub-spec as one metadata flag, and this sub-spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a GGUF whose `general.architecture` is `dflash` and which contains tensor `markov_w1.weight` classifies as `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`, not `DRAFT_DFLASH`, matching `common_speculative_types_from_gguf` (`common/speculative.cpp:2296-2306`) | yes |
| R2 | proxima-gguf parses `markov_w1.weight`, `markov_w2.weight`, the optional `markov_w2.scale` (`src/llama-arch.cpp:699-701`, `gguf-py/gguf/constants.py:1204-1207,1989-1991`), the optional `conf_proj.weight`/`conf_proj.bias`, and the two DSpark-owned metadata keys `dflash.sample_from_anchor` and `dflash.has_confidence_head` (`common/speculative.cpp:987,1002`; written by `gguf_writer.py:1040,1043`) | yes |
| R3 | the Markov bias `bias = markov_w2 @ markov_w1[prev_token]` (lora_mm form with optional int8 scale) is computed per block position and added to that position's raw logits before sampling, byte-identical to `build_dspark_markov_head` (`src/models/dflash.cpp:295-398`) for the same block and previous-token sequence | yes |
| R4 | when `has_confidence_head` is true and `p_min > 0.0`, per-position confidence `conf(i) = sigmoid(conf_proj . [feat(i); markov_w1[prev(i)]] + bias)` truncates the draft at the first position where `conf(i) < p_min`, matching `common_speculative_impl_draft_dflash::draft`'s dspark branch (`common/speculative.cpp:1267-1290`) | yes |
| R5 | when `p_min > 0.0` and `has_confidence_head` is false, drafting returns a typed error equivalent to llama.cpp's `std::runtime_error("DSpark draft has no confidence head...")` (`common/speculative.cpp:1005`) instead of panicking or silently sampling unconditionally | yes |
| R6 | block layout dispatches on `sample_from_anchor` read from GGUF metadata, never hardcoded: `true` samples a full `block_size` tokens from position 0 (anchor-first, `common/speculative.cpp:1016,1205,1271`); `false` samples `block_size - 1` tokens from positions 1..block_size-1, shape-identical to DFlash's own bonus-anchor layout (speculators-format exports, commit `9cd719af2`) | yes |
| R7 | DSpark reuses DFlash's all-positions bind, target-layer feature extraction, noise-block batching, and KV-injection (owned by R11c) unmodified -- this sub-spec adds only the Markov/confidence graph and the two dispatch reads above; no duplicate block-batching or feature-extraction function exists in the DSpark module | yes |
| R8 | a real DSpark draft GGUF and a compatible target GGUF are available locally for parity testing | yes |
| R9 | the Markov bias (R3) and confidence gate (R4) are load-bearing, not decoration: across a 200-block real-GGUF corpus, at least one drafted token differs between "bias/confidence computed" and "bias/confidence skipped" -- the measurable form of this sub-spec's own refutation condition | yes |

## architecture

DSpark is not a new `Architecture` impl: it is DFlash's draft program (R11c) plus an
optional Markov head bound when `markov_w1.weight` is present, mirroring llama.cpp's own
`is_dspark` boolean on the shared struct (`common/speculative.cpp:948,963`) rather than a
second class.

- `DflashDraftProgram` (owned by R11c) gains one field: `markov_head: Option<MarkovHead>`,
  populated by `Architecture::bind` on the draft GGUF only when `markov_w1.weight` exists
  (`src/models/dflash.cpp:124-135`).
- `MarkovHead { markov_w1: TensorView<'file>, markov_w2: TensorView<'file>, markov_w2_scale:
  Option<TensorView<'file>>, confidence: Option<ConfidenceHead> }`, `ConfidenceHead { proj:
  TensorView<'file>, bias: TensorView<'file> }` -- plain borrowed data, box-free, no new
  trait.
- the shared DFlash draft loop (R11c) gains one branch: after the block's raw logits are
  produced, `if let Some(markov) = &program.markov_head { .. }` computes the bias (R3) and,
  when `markov.confidence.is_some() && p_min > 0.0`, the confidence gate (R4) before
  returning the draft. `sample_from_anchor` (R6) selects the loop's start index and length;
  it is read once at bind time, never per-call.
- GGUF classification (R1) is a pure function over parsed metadata + tensor names, called
  from the same auto-detect path DFlash and MTP already share (`common_speculative_types_from_gguf`,
  `common/speculative.cpp:2280-2306`): dflash arch, no `markov_w1.weight` -> DFlash; dflash
  arch, `markov_w1.weight` present -> DSpark.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| DSpark as a DFlash field, not a second Architecture | `Option<MarkovHead>` on DFlash's program | a second impl duplicates DFlash's KV-injection and block-batching (principle 1, RISC reuse); llama.cpp itself dispatches DSpark through the identical struct (`common/speculative.cpp:2654-2656`) |
| confidence-gate failure mode | typed `InteropError` variant | llama.cpp hard-fails with `runtime_error` (`common/speculative.cpp:1005`) rather than silently sampling unconditionally -- principle 14 (incumbent is the correctness oracle) and P15 (no silent failure) |
| `sample_from_anchor` source | GGUF metadata only, read once at bind | llama.cpp never exposes this as a CLI override (`common/speculative.cpp:987-989`); a CLI flag would let proxima diverge from the incumbent's per-model contract |
| oracle | llama.cpp `f1ea20621` `build_dspark_markov_head` + the dspark draft-loop branch, exercised through the real binary and a compiled fixture harness | re-deriving the bias/confidence formulas from reading the C++ is principle 14's "from memory" failure |
| unit vs end-to-end fixtures | both: a compiled-harness fixture for R3/R4's exact bias/confidence values, plus the parity CLI for R6/R8's end-to-end token stream | R3/R4 need internal values no end-to-end run exposes; R6/R8 need the real binary as the whole-system oracle |

## acceptance criteria

`$EX` = `cargo run --release -p proxima-model-interop --features std --example speculative_decode_parity --`
`$DSPARK_DRAFT` = the file obtained in slice 1 (see AC9); path recorded in TASKS.md's resume block once known
`$DSPARK_TARGET` = the target GGUF compatible with `$DSPARK_DRAFT`'s markov head vocab, obtained alongside it

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-gguf -E 'test(/classifies_dspark_from_markov_tensor/)'` | 1 passed |
| AC2 | R2 | `cargo nextest run -p proxima-gguf -E 'test(/parses_dspark_markov_and_confidence_tensors/)'` | 1 passed; test prints `tensors_found = 4` (markov_w1, markov_w2, conf_proj weight, conf_proj bias) on a fixture GGUF with a confidence head, `tensors_found = 2` on one without |
| AC3 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_markov_bias_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 200 |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_confidence_gate_matches_llama_fixture/)'` | 1 passed; prints `cases = N` with N >= 200 and `truncated_cases >= 1` |
| AC5 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_missing_confidence_head_is_typed_error/)'` | 1 passed |
| AC6 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dspark_sample_from_anchor_layout/) or test(/dspark_bonus_anchor_layout/)'` | 2 passed |
| AC7 | R7 | `git grep -c 'fn inject_target_features\|fn build_noise_block\|fn extract_target_layers' -- proxima-model-interop/src/dspark.rs` then `git grep -c 'use .*dflash::' -- proxima-model-interop/src/dspark.rs` | 0; >= 1 |
| AC8 | R3,R4,R6 | `$EX --drafter draft-dspark "$DSPARK_DRAFT" "$DSPARK_TARGET" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40` | exit 0; both config blocks (greedy, sampled) print `identical = true`, `speculative_verify_steps` >= 1 |
| AC9 | R8 | `python3 /Users/brianbruggeman/repos/others/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py "$DSPARK_DRAFT" | grep -c 'markov_w1.weight\|dflash.sample_from_anchor'` | 2 (one tensor line, one metadata-key line) -- proves the obtained file is a real DSpark draft, not a plain DFlash one |
| AC10 | R9 | `cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model "$DSPARK_DRAFT" "$DSPARK_TARGET" --blocks 200` | prints `differing_blocks = N` out of `200`; N >= 1 admits R3/R4 as load-bearing, N = 0 invokes the refutation clause |

## out of scope

- DFlash's own correctness (all-positions bind, encoder-feature extraction, noise-block
  batching, KV-injection) -- owned by R11c; this sub-spec only adds the Markov/confidence
  delta on top of it
- non-vanilla Markov head types (`markov_head_type != "vanilla"`) -- every landed backbone
  (Gemma4, Qwen3.6, DeepSeekV4, LFM2, BailingMoE3, Nemotron3.5) ships vanilla; a non-vanilla
  head is a new upstream feature, not yet in `f1ea20621`
- speculators-format checkpoint conversion (d2t remap, reduced draft vocab expansion) --
  that is llama.cpp's `conversion/qwen.py` concern; proxima consumes the already-converted
  GGUF, it does not convert speculators-format checkpoints itself
- DFlash2 (`is_dflash2`, lattice-based backend selector) -- a DFlash-only path DSpark never
  takes (`common/speculative.cpp:1237-1238` gates it on `is_dflash2`, mutually exclusive
  with the `is_dspark` branch at `:1267`)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| no real DSpark GGUF pair is obtainable within this sub-spec's timeframe | medium | AC3, AC4, AC8, AC9 block on slice 1; R1/R2/R5/R7 can still land and test against a synthetic-but-real-shaped fixture (a small hand-built GGUF with the right tensor names, per principle 9's "as close to real as possible" exception) | slice 1 names the exact upstream-verified checkpoint (`gemma4-26b-a4b-dspark`, referenced in llama.cpp commit `9cd719af2`'s message: "Verified against gemma4-26b-a4b-dspark: greedy outputs are byte-identical... acceptance 0.46, mean draft len 3.7") as the target; if unobtainable, `satgeze/Qwen3.6-27B-DSpark` (`conversion/qwen.py:791`) is the fallback named example |
| `markov_w2_s` (int8 lora-mm scale, commit `ca3d5a3e1`) is present on some checkpoints and absent on others | medium | R3's bias computation branches on its presence; a fixture covering only one case would under-test | AC3's fixture generator (slice 2) must include cases from both a scaled and an unscaled checkpoint, or synthesize both if only one real file is available |
| `conf_proj` is `TENSOR_NOT_REQUIRED` (`src/models/dflash.cpp:132-133`) -- some real checkpoints have no confidence head at all | high (commit `ca3d5a3e1`'s own message: "vanilla-markov exports ship without it") | R4/R5 both depend on this being exercised in both states | the obtained checkpoint pair (AC9) is checked for `conf_proj.weight` presence explicitly; if the obtained model lacks it, a second minimal fixture (not a full model) proves R5's error path |

## context

- proxima: `proxima-model-interop/src/architecture.rs:292-309` (`speculative_verify_program`
  default), `proxima-gguf/src/` (no dflash/dspark parsing today -- verified empty), the
  parent spec's `proxima-model-interop/examples/speculative_decode_parity.rs`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`: `common/speculative.cpp` (dispatch
  table `:39`, enum-to-name `:2236`, struct `:922-1330`, `is_dspark` field `:948`, ctor
  `:963`, confidence-head-required check `:999-1006`, `n_draft_max` `:1016`, block-token
  count `:1205`, dspark draft-loop branch `:1267-1290`, GGUF auto-detect `:2280-2306`,
  block-draft output sizing `:2486-2498`, factory dispatch `:2629,2654-2656`),
  `common/common.h` (enum `:173-186`, `need_n_rs_seq` `:398`), `src/llama-arch.h:706-708`,
  `src/llama-arch.cpp:699-701,989-992`, `src/models/dflash.cpp` (tensor creation
  `:124-135`, `build_dspark_markov_head` `:295-398`, call sites `:846-847,1025-1026`),
  `gguf-py/gguf/constants.py:1204-1207,1989-1991,5263-5265`,
  `gguf-py/gguf/gguf_writer.py:1040,1043`
- upstream history (most-to-least directly load-bearing): `84075273c` (DSpark's original
  landing), `9cd719af2` (speculators-format / SpecForge checkpoints, `bonus_anchor` ->
  `sample_from_anchor`, verified `gemma4-26b-a4b-dspark` acceptance numbers),
  `ca3d5a3e1` (Nemotron3.5, optional confidence head, `markov_w2_s` scale tensor),
  `0d0bfcd4f` (backend sampling for dflash+dspark), `f65e568fd` (GGUF metadata
  auto-detect), `633733d0a` (Gemma4 DSpark draft backbone -- no example checkpoint
  registered yet), `2115b73d8`/`07822bddf`/`d646c9d15` (BailingMoE3/LFM2/DeepSeek-V4
  backbone support -- confirms DSpark is backbone-agnostic on the draft side)
- sibling sub-spec: `proxima-tensor/specs/speculative-draft-dflash/SPEC.md` (R11c, not yet
  audited at the time this spec was written) -- R3, R4, R6, R7 above all assume its
  `DflashDraftProgram` type and draft loop exist; if that type's name or shape changes,
  this spec's architecture section must be re-read against it before slice 3 starts
