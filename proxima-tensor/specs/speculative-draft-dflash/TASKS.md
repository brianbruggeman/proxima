# speculative-draft-dflash -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$DFLASH_DFT`, `$DFLASH_TGT`, `$LLAMA_SPEC_SIMPLE` as defined in SPEC.md acceptance criteria.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisite, owned by `decode-as-data` slice 3: the `build_forward` readout list. This sub-spec consumes it and does not re-implement it | AC6 (prerequisite) | `cargo nextest run -p proxima-tensor -E 'test(/readout_/)'` | 2 passed | [ ] | fails until `decode-as-data` slice 3 lands |
| 0b | build the oracle binaries from llama.cpp `f1ea20621` into `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621` (`cmake -S /Users/brianbruggeman/repos/others/llama.cpp -B <dir> -DGGML_METAL=ON`, then build `llama-speculative-simple` and `llama-gguf`); the checkout's prebuilt `build/bin` predates the source | AC13 (prerequisite) | `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple --help 2>&1 \| grep -c -- '--spec-type'` | 1 | [ ] | |
| 1 | acquire or produce the `draft-dflash` pair: search for a pre-converted `z-lab/Qwen3-4B-DFlash` GGUF; if none exists, read its HF `config.json` to confirm the backbone shape, then write (or port from a newer llama.cpp revision) the missing `convert_hf_to_gguf.py` model class for `general.architecture` equal to `dflash`; place the draft at `$DFLASH_DFT` and the Qwen3-4B target at `$DFLASH_TGT` | AC1 | `strings -a "$DFLASH_DFT" \| grep -c 'dflash.block_size'` | >= 1 | [ ] | run first; record the acquisition path taken (pre-converted found versus converter written) and the result of `grep -c 'dflash' convert_hf_to_gguf.py` in this note before any other slice starts |
| 1b | obtain a selector-bearing DFlash2 draft (`dflash.selector_top_k` above 0, the model class of llama.cpp commit `b10f9ca58`) and the target it names via `--target-model-dir`; place them at `$DFLASH2_DFT` and `$DFLASH2_TGT` | AC15 | `strings -a "$DFLASH2_DFT" \| grep -c 'dflash.selector_top_k'` | >= 1 | [ ] | if no such checkpoint can be obtained or converted, the note names the blocker and R7 is struck to structural-only per the refutation condition |
| 2 | fixture generator: a C++ harness compiled against llama.cpp `f1ea20621`'s DFlash implementation, emitting `dflash1.json`, `process.json`, `block.json`, `conv.json`, `params.json` from the slice-1 pair on real prompts, plus `gguf_dump.json` from `llama-gguf "$DFLASH_DFT" r n`; vendored under `proxima-model-interop/tests/fixtures/llama-dflash/fixtures/` with the upstream commit in each header | AC2 | `for name in dflash1 process block conv params gguf_dump; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/$name.json; done` | six lines printed, in order: >= 50, >= 10, >= 1, >= 50, >= 1, >= 1 | [ ] | struck to the structural fixtures if slice 1 produced no GGUF pair (refutation condition); |
| 2a | `dflash2.json` from `$DFLASH2_DFT` with `$DFLASH2_TGT`, same harness and header rules | AC16 | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/dflash2.json` | >= 50 | [ ] | |
| 2b | vendor the oracle ids: `$LLAMA_SPEC_SIMPLE --model "$DFLASH_TGT" --model-draft "$DFLASH_DFT" --spec-type draft-dflash --seed <seed> -n 40` for seeds 1, 2, 3, recorded with `n_drafted` and `n_accept` in `ids.json` | AC14 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/ids.json` | 3 | [ ] | |
| 3 | descriptor reads the five `dflash.*` keys (R1) | AC3 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_descriptor_reads_known_keys/)' 2>&1 \| grep -cE 'keys_read = 5'` | 1 (the test asserts the five key values before it prints `keys_read = 5`) | [ ] | |
| 3b | descriptor resolves the seven conv and selector tensor names (R2) | AC4 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_tensor_names_resolve/)' 2>&1 \| grep -cE 'tensors_resolved = [0-9]+'` | 1 (the test asserts `tensors_resolved` >= 7 before it prints) | [ ] | |
| 4 | masked-block input row layout (R3) | AC5 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_masked_block_shape/)' 2>&1 \| grep -cE 'block_size = [0-9]+ mask_positions = [0-9]+'` | 1 (the test asserts the layout and `mask_positions` == `block_size` - 1 before it prints) | [ ] | needs slice 1's GGUF for real tensor shapes; a synthetic fixture with the right metadata keys and zero-filled tensors suffices for this slice alone |
| 4b | non-causal block attention unless `dflash.attention.causal` is set (R13) | AC17 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_attention_causality_switch/)'` | 2 passed | [ ] | |
| 4c | local convolution per layer (R14) | AC18 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_local_conv_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+ max_abs_diff = '` | 1 (the test asserts `cases` >= 50 and `max_abs_diff` < 1e-4 before it prints) | [ ] | |
| 5 | target-layer tap through the readout list, gathered to `[n_chunk, n_embd_enc]` (R4) | AC6 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_matches_llama_fixture/)' 2>&1 \| grep -cE 'positions = [0-9]+ layers = [0-9]+ max_abs_diff = '` | 1 (the test asserts `positions` >= 10, `layers` >= 1 and `max_abs_diff` < 1e-4 before it prints) | [ ] | |
| 5b | one-call embd injection into the draft's decode (R5) | AC7 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_features_reach_draft_in_one_decode_call/)' 2>&1 \| grep -cE 'chunks = [0-9]+ decode_calls_per_chunk = 1'` | 1 (the test asserts `chunks` >= 10 and one decode call per chunk before it prints) | [ ] | |
| 6 | DFlash1 block decode: top-k=10 sampler, `p_min` early stop (R6) | AC8 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash1_block_decode_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+'` | 1 (the test asserts `cases` >= 50 and id equality before it prints) | [ ] | |
| 7 | DFlash2 selector-lattice decode: predecessor-argmax chain, softmax `p_min` truncation (R7) | AC9 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash2_selector_decode_matches_llama_fixture/)' 2>&1 \| grep -cE 'cases = [0-9]+'` | 1 (the test asserts `cases` >= 50 and id equality before it prints) | [ ] | |
| 8 | confirm no dflash-specific rollback type exists (R8 is a negative requirement; this slice is the grep, plus its control) | AC10 | `git grep -ciE 'DflashRollback\|dflash.*snapshot' -- proxima-model-interop/src proxima-tensor/src \| wc -l; git grep -c 'fn truncate' -- proxima-model-interop/src \| wc -l` | two lines printed: `0`, then >= 1 (the control counts files containing `fn truncate`) | [ ] | |
| 9 | `draft-dflash` sub-config with defaults equal to llama.cpp's, builder and loader parity (R9) | AC11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_draft_dflash_defaults_match_llama/)'` | 2 passed | [ ] | lands after parent slice 8 (the `ServingConfig` speculative section) |
| 10 | oracle comparator control (R11) | AC12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_oracle_control_/)'` | 12 passed | [ ] | run before slice 11 |
| 11 | llama oracle parity across 3 seeds (R10) | AC13 | `cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_llama_parity_seed_/)' 2>&1 \| grep -cE 'ids_compared = [0-9]+'` | 3 (one line per seed; each test asserts `ids_compared` >= 8 and id equality before it prints) | [ ] | requires slice 1's pair; struck if slice 1 never closes |

## resume

Last landed slice: none
Next action: slice 1 -- search for a pre-converted `z-lab/Qwen3-4B-DFlash` GGUF; if absent, read its HF
`config.json` and decide whether `convert_hf_to_gguf.py` needs a new `dflash` model class or can reuse
an existing Qwen3 one with the dflash tensors appended
Open question: whether plain DFlash (not DSpark) is backbone-restricted to Qwen3 the way DSpark's
Markov head is (`docs/speculative.md:106-107` states it only for DSpark); resolve while reading the
target model's config in slice 1

## struck

-
