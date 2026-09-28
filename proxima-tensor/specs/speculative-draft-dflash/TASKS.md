# speculative-draft-dflash -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$DFLASH_GGUF` / `$QWEN3_4B_GGUF` as defined in SPEC.md acceptance criteria (slice 1's note
records the exact converted paths once run).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | acquire or produce a `draft-dflash`-typed GGUF pair: search for a pre-converted `z-lab/Qwen3-4B-DFlash` (or equivalent) GGUF; if none exists, read `z-lab/Qwen3-4B-DFlash`'s HF config to confirm its backbone shape, then write (or port from a newer llama.cpp revision) the missing `convert_hf_to_gguf.py` `ModelBase` subclass for `general.architecture == "dflash"` | none directly -- unblocks R5, R6, R9 | `python3 -c "import gguf; ctx=gguf.GGUFReader('$DFLASH_GGUF'); print(ctx.fields['general.architecture'].parts)"` | prints `dflash` | [ ] | run first; report exact acquisition path taken (pre-converted found vs converter written) before any other slice starts |
| 2 | `DflashMetadata` typed read in proxima-gguf: the 5 `dflash.*` KV keys + the 7 tensor names | AC1, AC2 | `cargo nextest run -p proxima-gguf -E 'test(/dflash_metadata_parses_known_keys/) or test(/dflash_tensor_names_resolve/)'` | 2 passed, keys_read=5, tensors_resolved≥7 | [ ] | |
| 3 | `DflashArch::bind`: masked-block input construction (`[id_last, <mask>*(block_size-1)]`), non-causal attention wiring, local conv (attn/ffn base+proj) per layer | AC3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash_masked_block_shape/)'` | 1 passed | [ ] | requires slice 1's GGUF for real tensor shapes; a synthetic fixture with the right metadata keys and zero-filled tensors is acceptable for this slice alone |
| 4 | gemma4 target-layer extraction override (default `Ok(None)` capability method on `Architecture`, gemma4 implements it) + one-call embd injection into the draft's decode (no separate encode pass) | AC4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/target_layer_extraction_feeds_draft_embd_in_one_call/)'` | 1 passed, llama_process_calls=1 | [ ] | |
| 5 | fixture generator: C++ harness compiled against llama.cpp `common/speculative.cpp`'s dflash struct, emitting JSON draft-token cases from the slice-1 GGUF pair on real prompts; vendored with upstream commit in the header | AC5, AC6 (inputs) | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dflash/fixtures/dflash1.json proxima-model-interop/tests/fixtures/llama-dflash/fixtures/dflash2.json` | 2 files, each ≥ 50 | [ ] | struck if slice 1 could not produce a GGUF pair (refutation condition) |
| 6 | DFlash1 block decode: top-k=10 sampler, `p_min` early-stop | AC5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash1_block_decode_matches_llama_fixture/)'` | 1 passed, cases≥50 | [ ] | |
| 7 | DFlash2 selector-lattice decode: predecessor-argmax chain, softmax `p_min` truncation | AC6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/dflash2_selector_decode_matches_llama_fixture/)'` | 1 passed, cases≥50 | [ ] | |
| 8 | confirm no dflash-specific rollback type was added (R7 is a negative requirement -- this slice is the grep, not new code, unless slice 3/4 accidentally added one) | AC7 | `git grep -c 'DflashRollback\|dflash.*snapshot' -- proxima-model-interop/src` | 0 | [ ] | |
| 9 | `ServingConfig` `draft_dflash` sub-config (block_size override, n_max, n_min, p_min, backend_sampling) + builder/loader parity | AC8 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_draft_dflash_builder_matches_loader/)'` | 1 passed | [ ] | composes with parent spec's slice 8 (ServingConfig speculative section) -- lands after it |
| 10 | `Drafter::Dflash` variant in the decode loop's closed enum; `--drafter draft-dflash --draft-model` wiring in the parity example | AC9 | `$EX --drafter draft-dflash --draft-model "$DFLASH_GGUF" "$QWEN3_4B_GGUF" "Repeat exactly five times: the quick brown fox jumps over the lazy dog." 40` | exit 0, both blocks identical=true | [ ] | requires slice 1's GGUF pair; struck to "manual smoke only, no AC" if slice 1 never closes |
| 11 | mark this sub-spec `status: audited` once slices 1-9 are green (slice 10 may remain open if the refutation condition fired) | AC1-AC8 | `grep -c '^status: audited' proxima-tensor/specs/speculative-draft-dflash/SPEC.md` | 1 | [ ] | parent spec's AC15 reads this line |

## resume

Last landed slice: none
Next action: slice 1 -- search for a pre-converted `z-lab/Qwen3-4B-DFlash` GGUF; if absent,
read its HF `config.json` and decide whether `convert_hf_to_gguf.py` needs a new `dflash`
`ModelBase` subclass or can reuse an existing Qwen3 one with dflash tensors appended
Open question, if any: whether plain DFlash (not DSpark) is backbone-restricted to Qwen3 the
same way DSpark's Markov head is (`docs/speculative.md:106-107` states it only for DSpark) --
unresolved this session, resolve while reading the target model's config in slice 1

## struck

- none yet -- this spec is `draft`, not yet audited
