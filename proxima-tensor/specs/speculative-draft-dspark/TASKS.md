# speculative-draft-dspark -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$EX`, `$DSPARK_DRAFT`, `$DSPARK_TARGET` as defined in SPEC.md acceptance criteria.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | obtain a real DSpark draft GGUF + compatible target GGUF (`gemma4-26b-a4b-dspark` per llama.cpp commit `9cd719af2`'s verification message; fallback `satgeze/Qwen3.6-27B-DSpark`, `conversion/qwen.py:791`); record both local paths in this file's resume block | AC9 | `python3 /Users/brianbruggeman/repos/others/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py "$DSPARK_DRAFT" \| grep -c 'markov_w1.weight\|dflash.sample_from_anchor'` | 2 | [ ] | if neither named checkpoint is obtainable, name the actual substitute here and re-state why it still exercises a vanilla Markov head (out-of-scope rules out non-vanilla heads) |
| 2 | fixture generator: C++ harness compiled against llama.cpp's `build_dspark_markov_head` (`src/models/dflash.cpp:295-398`) and the dspark draft-loop branch (`common/speculative.cpp:1267-1290`), emitting JSON cases (prev-token block, computed bias vector, confidence value where present, resulting draft ids) from `$DSPARK_DRAFT`; vendored at `proxima-model-interop/tests/fixtures/llama-dspark/` with the upstream commit in the header; includes at least one case with `markov_w2_s` present and one without, and one checkpoint/synthetic case with no `conf_proj` (per the risk row on optional confidence heads) | AC3, AC4 (inputs) | `jq '.cases \| length' proxima-model-interop/tests/fixtures/llama-dspark/fixtures/dspark_markov.json` | >= 200 | [ ] | |
| 3 | proxima-gguf: parse `markov_w1.weight`, `markov_w2.weight`, optional `markov_w2.scale`, optional `conf_proj.weight`/`conf_proj.bias`, and the `dflash.sample_from_anchor` / `dflash.has_confidence_head` metadata keys; classify DSpark vs DFlash by tensor presence | AC1, AC2 | AC1 then AC2 | 1 passed; 1 passed with `tensors_found` counts per AC2 | [ ] | depends on R11c's DFlash GGUF-side parsing existing first for the shared dflash-arch keys (`block_size`, `attention.causal`, `target_layer_ids`, `mask_token_id`) |
| 4 | `MarkovHead` field on `DflashDraftProgram`; bias computation `markov_w2 @ markov_w1[prev_token]` wired into the shared DFlash draft loop | AC3 | AC3 | 1 passed, cases >= 200 | [ ] | depends on R11c's `DflashDraftProgram` and draft loop existing |
| 5 | confidence-gate truncation (`sigmoid(conf_proj . [feat; markov_w1[prev]] + bias) < p_min`) plus the typed error when `p_min > 0.0` and no confidence head | AC4, AC5 | AC4 then AC5 | 1 passed, cases >= 200, truncated_cases >= 1; 1 passed | [ ] | |
| 6 | `sample_from_anchor` layout dispatch: anchor-first (full `block_size`) vs bonus-anchor (`block_size - 1`, positions 1..) | AC6 | AC6 | 2 passed | [ ] | |
| 7 | structural RISC-reuse check: assert the DSpark module defines no duplicate block-batching/feature-extraction functions and imports DFlash's | AC7 | AC7 | 0; >= 1 | [ ] | if this fails, the fix is deleting the duplicate in the dspark module, never adding an `allow` or a re-export shim |
| 8 | end-to-end parity: `--drafter draft-dspark` flag on the parity example, run against `$DSPARK_DRAFT`/`$DSPARK_TARGET` | AC8 | AC8 | exit 0, both blocks identical=true, steps >= 1 | [ ] | this is the whole-system oracle check; if it fails while AC1-AC7 pass, the bug is in wiring (loop start index, p_min plumbing), not in R3/R4's math |
| 9 | `dspark_markov_ablation` example: runs the draft loop twice per block over a 200-block corpus, once with the Markov bias/confidence gate active and once with them forced off, and counts blocks where the drafted tokens differ -- this is the refutation condition's own measurement | AC10 | AC10 | `differing_blocks` >= 1 out of 200 | [ ] | if 0, stop: per the refutation condition this sub-spec is struck and R3/R4 collapse into DFlash's sub-spec as a metadata flag -- do not proceed to promote this spec's status |

## resume

Last landed slice: 0
Next action: slice 1 -- obtain `gemma4-26b-a4b-dspark` (or the named fallback) and record its
local path here, then run the AC9 gguf_dump.py grep to confirm it is a real DSpark draft
Open question, if any: whether `gemma4-26b-a4b-dspark` is publicly downloadable or only
referenced in the upstream commit message as an internal verification checkpoint -- slice 1
must resolve this by attempting the fetch, not by assuming either answer

## struck

-
