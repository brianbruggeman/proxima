# speculative-draft-eagle3 -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$PAIR_TGT`, `$PAIR_DFT`, `$LLAMA_SPEC_SIMPLE` as defined in SPEC.md acceptance criteria.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisite, owned by `decode-as-data` slice 3: the `build_forward` readout list (logits rows, hidden-state taps at named layers, extra head roots). This sub-spec consumes it and does not re-implement it | AC4 (prerequisite) | `cargo nextest run -p proxima-tensor -E 'test(/readout_/)'` | 2 passed | [ ] | fails until `decode-as-data` slice 3 lands |
| 0b | build the oracle binaries from llama.cpp `f1ea20621` into the fixed directory (`cmake -S /Users/brianbruggeman/repos/others/llama.cpp -B /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621 -DGGML_METAL=ON` then `cmake --build` the `llama-speculative-simple` target); the checkout's prebuilt `build/bin` predates the source and registers no `--spec-type` | AC14, AC18 (prerequisite) | `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple --help 2>&1 \| grep -c -- '--spec-type'` | 1 | [ ] | |
| 1 | obtain the target and draft pair: download `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` (the smaller of the two named pairs) and its matching gemma-4-26B-A4B-it target, convert both with `convert_hf_to_gguf.py --target-model-dir` per `docs/speculative.md:23-32`, and place them at the `$PAIR_TGT` and `$PAIR_DFT` paths | AC1 | `ls -1 "$PAIR_TGT" "$PAIR_DFT" 2>/dev/null \| wc -l` | 2 | [ ] | record the HF repo ids, converted file sizes and the download-size comparison with the 31B pair in this note |
| 2 | fixture generator: a C++ harness linking llama.cpp `f1ea20621`, calling `llama_set_embeddings_layer_inp`/`llama_set_embeddings_nextn`/`llama_get_embeddings_nextn(_ith)` directly plus the real `common_speculative_impl_draft_eagle3` end to end on real prompts against the slice-1 pair; dumps `hidden_tap.json`, `encoder.json`, `decoder_draft.json` (each case records `stash_bytes` from `get_state`) to `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/` with the upstream commit in each header | AC2 | `for name in hidden_tap encoder decoder_draft; do jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/$name.json; done` | three lines printed, each >= 50 | [ ] | |
| 2b | vendor the oracle ids: run `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 --seed <seed> -n 40` for seeds 1, 2, 3; record ids, `n_drafted`, `n_accept` and the llama.cpp commit in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/ids.json` | AC15 | `jq '[.cases[] \| select(.n_drafted > 0)] \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/ids.json` | 3 | [ ] | |
| 2c | pairing fixture: write two synthetic drafts with `proxima-gguf`'s writer (`target_layer_ids_n = 2`; one id above the target's `n_layer`), load each through llama.cpp `f1ea20621` in the harness, record the accept/reject outcome in `proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json` | AC17 | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-eagle3/fixtures/pairing.json` | 2 | [ ] | |
| 3 | draft descriptor from GGUF-key reads and the `eagle3` family profile TOML (R1); no forward program yet | AC3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_descriptor_reads_target_layers_and_tensors/)'` | 1 passed | [ ] | |
| 4 | gemma4 all-positions hidden-state tap at the 3 caller-chosen layer indices, through the readout list (R2) | AC4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/gemma4_hidden_tap_matches_llama_fixture/)'` | 1 passed, `max_abs_diff < 1e-4` | [ ] | |
| 5 | encoder graph: 3-layer fuse -> optional RMSNorm -> `fc` -> g_embd (R3) | AC5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_encoder_matches_llama_fixture/)'` | 1 passed, `cases` >= 50 | [ ] | |
| 6 | decoder step graph: concat(embd_norm, g_norm) -> attention -> FFN -> output_norm -> lm_head -> optional `d2t` (R4) | AC6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_decoder_step_matches_llama_fixture/)'` | 1 passed, `cases` >= 50 | [ ] | |
| 7 | full draft loop: deferred-boundary cross-ubatch bridge, seed step, top-k=10 autoregressive sampling, `p_min` early stop, `n_min`/`n_max` bounds (R5) | AC7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_draft_loop_matches_llama_fixture/)'` | 1 passed, `cases` >= 50, `p_min_early_stops` >= 1, `n_max_stops` >= 1 | [ ] | |
| 8 | accept-time boundary rewind (R6) | AC8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_accept_rewinds_boundary/)'` | 1 passed | [ ] | |
| 8b | empty stash and no-op restore for an attention-only target (R7) | AC9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_state_stash_empty_for_attention_target/)'` | 1 passed | [ ] | |
| 9 | route the `draft-eagle3` config value to the propose pipe carrying taps and the draft descriptor, with builder/loader parity (R8) | AC10 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_selects_draft_eagle3/)'` | 2 passed | [ ] | |
| 9b | EAGLE3 pairing compatibility typed errors (R9) | AC11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_pairing_rejects_wrong_layer_count/)'` | 2 passed | [ ] | |
| 10 | oracle comparator control (R12) | AC13 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_oracle_control_/)'` | 2 passed | [ ] | run before slice 11 |
| 11 | llama oracle parity across 3 seeds (R13) | AC14 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/eagle3_llama_parity_seed_/)'` | 3 passed, `ids_compared` >= 8 each | [ ] | |
| 13 | `speculative_acceptance_corpus` gains `--drafter`/`--draft-model` flags; the oracle's acceptance (AC18) is recorded first, then the proxima measurement (AC12) | AC18, AC12 | `$LLAMA_SPEC_SIMPLE --model "$PAIR_TGT" --model-draft "$PAIR_DFT" --spec-type draft-eagle3 -p "Repeat exactly five times: the quick brown fox jumps over the lazy dog." -n 40 2>&1 \| grep -cE 'n_accept +=  *[1-9]'` then `cargo run --release -p proxima-model-interop --features std --example speculative_acceptance_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model "$PAIR_DFT" "$PAIR_TGT" > acceptance_eagle3.txt; grep -cE 'mean_accepted_per_step=([1-9][0-9]*\.[0-9]+\|0\.[0-9]*[1-9])' acceptance_eagle3.txt` | two lines printed: `1` then `1` | [ ] | a second `0` means stop and redesign R2 per the refutation condition |

## resume

Last landed slice: none
Next action: slice 1 -- resolve and record the exact HF repo ids for the gemma-4-26B-A4B-it target and
its eagle3 speculator, download, convert with `convert_hf_to_gguf.py --target-model-dir`
Open question: whether the 26B-A4B (MoE) pairing exercises gemma4's MoE lowering in a way that
complicates R2's hidden-state tap (the 31B dense pairing may be the simpler first target despite its
size); resolve during slice 1 by comparing download sizes and checking whether the gemma4 MoE profile
already yields per-layer taps of compatible shape through the readout list

## struck

-
