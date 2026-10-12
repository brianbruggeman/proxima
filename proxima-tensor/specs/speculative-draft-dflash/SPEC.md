# speculative-draft-dflash

status: draft
owner: brian bruggeman
created: 2026-09-28
restated: 2026-10-12 against `decode-as-data` R5, R6, R11 (readouts plus a draft descriptor)
restated: 2026-10-12 against owner direction: correctness is lossless output over our own trained draft, not llama.cpp token parity

## problem

proxima cannot run `draft-dflash` speculation (a block-diffusion draft fed the target's hidden states) as data on the `decode-as-data` frame, and this spec adds that and measures it by one relation: with greedy acceptance, output equals the target's own plain greedy decode on all 51 corpus prompts for gemma4 E2B and granite 1b.

## refutation condition

Written before evidence: if, on gemma4 E2B over the 51-prompt corpus (`proxima-model-interop/examples/data/speculative_corpus.jsonl`), `draft-dflash` with a randomly initialized draft produces fewer than 51 prompts whose ids equal plain greedy, the tap-plus-bidirectional-block design (R3, R4a, R4b, R4c, R6) or the verify and rollback plumbing is wrong and must be redone, not tuned. Separately, if with the trained draft (`speculative-draft-training`) any of the 51 prompts has `accepted=0`, the injected features carry no signal about the target and R4a is wrong.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | the draft descriptor reads the metadata keys `dflash.block_size` and `dflash.target_layers` (both required; `target_layers` is a non-empty array of `i32` target layer ids, read from the key `target_layers` under the `dflash` prefix, `src/llama-arch.cpp:365`, `src/models/dflash.cpp:36-38`; an empty array is refused as `common/speculative.cpp:974` asserts `target_layer_ids_n > 0`, and the draft's input width is the array length times the target width, `common/speculative.cpp:978`; an array whose element type is not `i32` is refused), `dflash.conv_kernel_size`, `dflash.conv_group_size`, `dflash.selector_rank`, `dflash.selector_top_k` and `dflash.attention.causal` (all optional; an absent integer key reads as 0 and an absent `dflash.attention.causal` reads as false), key names mirroring `src/llama-arch.cpp:370-374` and `common/speculative.cpp:990-992`, from a `dflash` `general.architecture` GGUF as descriptor data; the descriptor classifies by the key alone: `selector_top_k` of 0 is the plain block draft (DFlash1) and above 0 is the selector draft (DFlash2) | yes |
| R2 | the descriptor resolves, for any draft and independent of classification, the tensor names `blk.%d.attn_conv_base`, `blk.%d.attn_conv_proj`, `blk.%d.ffn_conv_base`, `blk.%d.ffn_conv_proj`, `selector_predecessor`, `selector_successor`, `selector_hidden` (`src/llama-arch.cpp:702-708`) to the tensors the GGUF carries and reports the count resolved; resolution never changes the R1 classification | yes |
| R3 | the draft program takes the masked-block input `[id_last, <mask> * (block_size-1)]` as the row layout the host shape pipe produces, with positions `n .. n+block_size-1`; the drafted tokens are read from block positions `1 .. block_size-1` | yes |
| R4a | the target's lowered program emits the hidden states of exactly the layers named by the descriptor's `dflash.target_layers` (R1) for every new position as a tap readout (`decode-as-data` R5): one gathered row per new position per requested layer, gathered into the `[n_chunk, n_embd_enc]` shape the draft consumes, with the tapped layer ids equal to the descriptor's `target_layer_ids` | yes |
| R4b | each tapped row equals that layer's residual output read independently of the tap path | yes |
| R4c | the tap executed on Metal equals the same program evaluated on the CPU, with the tapped layer ids pinned to the descriptor's `target_layer_ids` | yes |
| R5 | target features reach the draft's embd-input decode in one decode call per chunk, never through a separate encode-then-reupload step | yes |
| R6 | DFlash1 block decode: at each masked position the highest-probability candidate is taken; the block read stops before pushing a token whose probability is below `p_min`; the result is cleared when fewer than `n_min` tokens were collected | yes |
| R7 | DFlash2 selector decode: the draft output is read per block row as `selector_top_k` candidate ids followed by `selector_top_k` x `selector_top_k` predecessor-by-successor scores (the scores for a predecessor start at offset `selector_top_k + predecessor * selector_top_k` within the row, `common/speculative.cpp:1243-1247`), a greedy predecessor-argmax chain whose first predecessor is 0 (`common/speculative.cpp:1241`), with softmax `p_min` truncation and the `n_min` clear of R6 (`:1261-1262`), deterministic given the lattice and without resampling, selected when `selector_top_k` > 0 | yes |
| R8 | the accept stage adds no dflash-specific rollback state: only the parent spec's R3a attention-KV truncate runs | yes |
| R9 | `ServingConfig`'s speculative section (parent R9) carries the `draft-dflash` sub-config `block_size` (0 = read from GGUF metadata), `n_max`, `n_min`, `p_min`, `backend_sampling`, with defaults `n_max` 3, `n_min` 0, `p_min` 0.0, `backend_sampling` true, the values of llama.cpp's `common_params_speculative_draft` (`common/common.h:327-352`); builder and loader produce the identical config | yes |
| R10 | attention over the block is non-causal (the bidirectional block shape of `decode-as-data`: one `seq_member` column of ones, constant `mask_pos`) unless `dflash.attention.causal` is set, in which case it is causal | yes |
| R11 | our trained E2B draft GGUF (`speculative-draft-training`) loads through R1 and R2 and pairs with the E2B target, and is refused with a typed width error against the granite target | yes |
| RL1 | on gemma4 E2B and on granite 1b, with greedy acceptance, the output of `draft-dflash` equals the target's own plain greedy decode token for token on every prompt of the 51-prompt corpus: the count of identical prompts equals the corpus size (51) | yes |
| RL2 | the draft is our trained draft for the target (`speculative-draft-training`); on that draft every prompt also has `n_drafted` > 0 and `accepted` > 0, so lossless is not satisfied by a method that never speculates or never agrees | yes |
| RL3 | until a trained draft exists, the method is exercised with a randomly initialized draft of the right shape (`--draft-random <seed>`, seeded and deterministic); that proves losslessness, plumbing and the rollback path (`rejected_total` > 0, pre-registered, status plausible); `n_drafted` > 0 on every prompt is still gated, and acceptance is reported, not gated | yes |
| RL4 | granite 1b is exercised only when the granite verify program exists and the draft shape fits granite (`speculative-draft-training` R10); otherwise the granite arm is recorded as refused with the typed `speculative_verify` error, never skipped silently | yes |
| RL5 | the comparator and the zero-work guard are validated by controls: plain output on both sides passes with identical = compared, one flipped token is rejected, and a run that drafted nothing is flagged | yes |

## architecture

Nothing here is a new type. DFlash is a tap readout list on the target descriptor, a second descriptor (the draft) with its own `ServingState`, and the bidirectional block shape, driven through the four pipes of `decode-as-data` (propose, shape, accept, commit). Its row in that spec's table is "masked block on a draft descriptor | block | sample-and-match on the target | prefix".

- **descriptor data (R1/R2/R3)**: the `dflash.*` keys and the optional conv/selector tensors become fields of the draft `ModelDescriptor` (`proxima-tensor/src/spec/descriptor.rs:130-296`), named by what they are (block size, local conv kernel and group size, candidate-selector rank and top-k), not by the technique; a family profile TOML for the `dflash` `general.architecture` value, beside `proxima-model-interop/src/profiles/*.toml`, supplies the layer shape. The `general.architecture` value is read at load as a key, never as a Rust type. `proxima-gguf` needs no architecture-name enum: `proxima-gguf/src/parser.rs` and `value.rs` offer generic key and tensor accessors, and name dispatch already lives in proxima-model-interop's profile selection.
- **masked block (R3)**: llama.cpp builds `n_draft + 1` rows, the first `id_last` and the rest the mask token, at `common/speculative.cpp:1205-1210`; here the row layout is produced by the host shape pipe (`decode-as-data` slice 2). Non-causal attention is `seq_member` one column of ones and constant `mask_pos` (`decode-as-data` table of configurations, "bidirectional block"), selected by config.
- **target tap and injection (R4a/R4b/R4c/R5)**: the tap is the readout list (`decode-as-data` R5, slice 3), not a method on a type. Injection feeds the gathered features as the draft's embd-input batch after every target forward, as llama.cpp's `process` does (`common/speculative.cpp:1098-1181`), and reads the draft's block output before verify. The existing token-id-only n-gram drafters (`proxima-model-interop/src/generate/drafter.rs`) do not fit a float embd-input, which is why the draft-model path (`proxima-model-interop/src/generate/draft_model.rs:174-183`, `load_draft`) is the precedent this follows, not that file's enum.
- **block decode (R6/R7)**: DFlash1's per-position read (llama.cpp `common/speculative.cpp:1295-1318`) and DFlash2's lattice chain (`common/speculative.cpp:1237-1265`) are two propose configurations selected by `selector_top_k > 0`, two branches because they are mutually exclusive per model and produce tokens through different math (probability threshold over a distribution versus argmax over a lattice); one function behind a boolean would hide the branch. llama.cpp's accept is a no-op (`common/speculative.cpp:1326-1328`), which is R8.
- **shared functions**: block batching (R3), feature extraction (R4a) and injection (R5) are the three functions `masked_block_rows`, `tap_feature_rows` and `inject_feature_rows`, each defined once; `speculative-draft-dspark` calls them through this propose path and its AC12 counts their definitions and call sites.
- **lossless harness (RL1-RL5)**: the `speculative_lossless_corpus` example is created by `speculative-draft-eagle3` slice 1 and consumed here unchanged with `--drafter draft-dflash`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| correctness oracle | the target's own plain greedy decode, over the 51-prompt corpus | llama.cpp parity needs a llama-format DFlash pair, and the only one named upstream is a Qwen3 pairing, which the owner removed from the test set; losslessness is the property speculation must hold |
| draft source | our trained draft (`speculative-draft-training`); a seeded random draft until it exists | a random draft proves losslessness and plumbing; only the trained draft can show the injected features carry signal, which is why RL2 gates `accepted` > 0 only there |
| trained form | DFlash1 (no conv, no selector) | DFlash2's conv and selector are extra heads whose only trained examples are llama-format third-party checkpoints; the DFlash2 decode (R7) stays specified and testable on a hand-built lattice, and a trained DFlash2 is a variant of `speculative-draft-training` |
| DFlash2 selector math | its own branch, not folded into DFlash1's read | see the architecture bullet on block decode |
| DSpark's anchor-first, Markov-head and confidence-truncation paths | out of scope | owned by `speculative-draft-dspark` |

## acceptance criteria

`$E2B` = `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`, the gemma4 E2B target
`$GRANITE` = `/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`, the granite 3.1 1b target
`$CORPUS` = `proxima-model-interop/examples/data/speculative_corpus.jsonl`, 51 prompts
`$DRAFT_E2B` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/e2b-dflash.gguf`, the trained E2B draft `speculative-draft-training` slice 11 places at exactly this path
`$DRAFT_GRANITE` = `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/granite-dflash.gguf`, the trained granite draft `speculative-draft-training` slice 12 places at exactly this path
`$OUT` = `target/spec-dflash-ac`, the directory the lossless examples write their transcripts to
`$LOSSLESS_ANY`, `$LOSSLESS_DRAFTED`, `$LOSSLESS_ACCEPTED`, `$SUMMARY_LINE` = anchored full-line patterns of the harness output defined by `speculative-draft-eagle3` (one `lossless id=... identical=... n_drafted=... accepted=... rejected=... first_divergence=...` line per prompt, then one `summary prompts=<n> identical=<n> ... rejected_total=<n> ...` line): any prompt line, an identical prompt line with `n_drafted` >= 1, an identical prompt line with `n_drafted` >= 1 and `accepted` >= 1, and the summary line with `prompts=51`, `identical=51` and `rejected_total` >= 1; all are exported in the `ac commands` section

Every count below is asserted by its test, which prints the count only after the assertion passes; the command greps the printed line.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | section `ac commands`, `### AC1` | 1 (the test writes a synthetic `dflash` GGUF with `proxima-gguf`'s writer carrying only `dflash.block_size = 8` and `dflash.target_layers = [3, 7, 11]`, reads it back, asserts `block_size` 8, `target_layers` `[3, 7, 11]`, the five absent keys reading as 0 or false, and that the descriptor classifies as DFlash1 by `selector_top_k` 0; the control writes a second GGUF with `dflash.selector_top_k = 4`, `dflash.attention.causal = true` and `dflash.target_layers = [3, 7, 12]` and asserts the descriptor differs and classifies as DFlash2; before the print, a GGUF with no `dflash.block_size` and a GGUF with no `dflash.target_layers` are each asserted to return the typed error, a GGUF whose `dflash.target_layers` is an empty array (wrong length) is asserted to return the typed error, and a GGUF whose `dflash.target_layers` is an `f32` array `[3.0, 7.0, 11.0]` (not `i32`) is asserted to return the typed error) |
| AC2 | R2 | section `ac commands`, `### AC2` | 1 (a synthetic 1-layer GGUF carrying all seven names and `dflash.selector_top_k` 0 resolves 7 and still classifies as DFlash1, so the tensors do not classify; the control, a GGUF carrying none, resolves 0) |
| AC3 | R3 | section `ac commands`, `### AC3` | 1 (the test prints the full line `masked_block block_size = 4 mask_positions = 3 positions = 100,101,102,103 read_positions = 1,2,3` after its assertions; with `id_last` 7, mask token 5 and first position 100, the rows are ids `[7, 5, 5, 5]` at positions 100..103 and the read positions are 1, 2, 3; the control builds `block_size` 5 and asserts `mask_positions` 4, so the assertion cannot be satisfied by a constant) |
| AC4a | R4a | section `ac commands`, `### AC4a` | 1 (the test takes the layer ids from a draft descriptor whose `dflash.target_layers` is `[2, 13, 25]`, runs the tap on E2B over a 64-token corpus prompt, and asserts `positions` = 64, `layers` = 3, exactly one gathered row per new position per requested layer (`rows_per_layer` = 64 for each of the 3 layers), the gathered width equal to 3 times the target `n_embd`, and the tapped layer ids equal to the descriptor's; the control compares the tapped ids with `[2, 13, 24]` and asserts the comparator rejects) |
| AC4b | R4b | section `ac commands`, `### AC4b` | 1 (oracle independent of the tap path: for each requested layer L of `[2, 13, 25]` the test evaluates the same E2B CPU forward with the stack truncated after block L, with no readout list, reads that program's final residual output, and asserts the tapped row at every position equals it with `independent_max_abs_diff` < 1e-3 (pre-registered, status plausible, not yet measured); controls inside the test: the tap compared with itself gives exactly 0.0; one element moved by 2e-3 is rejected by the same comparator; the layer-13 tapped rows compared against the layer-12 truncated reference are rejected by the same comparator, printed as `wrong_layer_rejected = true`) |
| AC4c | R4c | section `ac commands`, `### AC4c` | 1 (the layer ids come from the draft descriptor whose `dflash.target_layers` is `[2, 13, 25]` and are asserted equal to it; the tap runs on E2B over a 64-token corpus prompt on Metal and on CPU, with the CPU forward as the oracle (owner frame), and the test asserts `max_abs_diff` < 1e-3 (pre-registered, status plausible, not yet measured); controls: the CPU tap compared with itself gives exactly 0.0, and one element moved by 2e-3 is rejected by the same comparator) |
| AC5 | R5 | section `ac commands`, `### AC5` | 1 (the test prints the full line `injection chunks = 12 decode_calls_per_chunk = 1 split_path_calls_per_chunk = 2` after its assertions; a counting draft backend over 12 chunks; the real path is asserted at 1 call per chunk, and the control drives the encode-then-reupload shape through the same counter and asserts it reports 2, so the counter can tell the two apart) |
| AC6 | R6 | section `ac commands`, `### AC6` | 1 (the test prints the full line `block_decode_dflash1 cases = 4 p_min_stops = 3 n_min_clears = 1` after its assertions; a scripted block output with per-position top-1 `(token, probability)` of `(21, 0.90)`, `(22, 0.80)`, `(23, 0.30)` at block positions 1..3: four cases, with `n_min` 0 in cases 1, 2 and 4 and `n_min` 3 in case 3: (1) `p_min` 0.5 gives `[21, 22]`, stopped by `p_min` at position 3; (2) `p_min` 0.0 gives `[21, 22, 23]`, no stop; (3) `p_min` 0.5 stops at position 3 with 2 collected, then clears to `[]` because 2 < `n_min`; (4) `p_min` 0.95 stops at position 1 with 0 collected and gives `[]`, which is not a clear because 0 is not below `n_min` 0; so `cases = 4`, `p_min_stops` counts the cases that ended on `p_min` (cases 1, 3, 4, so 3) and `n_min_clears` counts the cases where a non-empty collected list was discarded because its length was below `n_min` (case 3 only, so 1); the `p_min` and `n_min` values are pre-registered, status plausible; the test asserts all four before it prints) |
| AC7 | R7 | section `ac commands`, `### AC7` | 1 (the test prints the full line `selector_decode cases = 4 truncated = 1 n_min_clears = 1 ablated_differs = true` after its assertions; `selector_top_k` 2; block rows `[101, 102, 0.2, 0.8, 0.5, 0.1]` and `[201, 202, 0.9, 0.1, 0.3, 0.7]`, each laid out as the K candidate ids then K x K predecessor-by-successor scores; row 1 scores `[0.2, 0.8]` pick predecessor 1 and token 102, row 2 reads scores at offset `1 x K` = `[0.3, 0.7]`, picks predecessor 1 and token 202; four cases, with `n_min` 0 in cases 1 to 3 and the chain starting from predecessor 0: (1) `p_min` 0.0 gives `[102, 202]`; (2) `p_min` 0.6 (pre-registered, status plausible) gives `[102]`, `truncated = 1`, because softmax at the argmax of `[0.3, 0.7]` is 0.5987 < 0.6 while for `[0.2, 0.8]` it is 0.6457; (3) the control forces predecessor 0 for row 2 and asserts the result becomes `[102, 201]`, which differs from case 1, so `ablated_differs = true`; (4) `n_min` 2 with `p_min` 0.6 collects `[102]`, one token below `n_min`, and clears to `[]`, counted by `n_min_clears = 1`; `cases = 4`) |
| AC8 | R8 | section `ac commands`, `### AC8` | four lines, measured in this worktree at HEAD f1f00666a, the base taken from main (an ancestor of the current origin/main): `151`, `72`, `0`, `1` (tracked files under `proxima-model-interop/src`, tracked files under `proxima-tensor/src`, files matching the forbidden names, files matching the forbidden names plus `fn truncate`; the fourth is `proxima-model-interop/src/generate/residency_caches.rs` with 4 matching lines, the attention-KV `LayerCache::truncate` the rollback reuses, so the grep reads the right tree). Gate: line 1 >= 1 and line 2 >= 1 (pre-registered floors, reason: tracked-file counts move with every landed commit, so the measured 151 and 72 are recorded and the gate is that both pathspecs hold tracked files, which a mistyped pathspec printing 0 would fail), line 3 exactly `0`, line 4 >= 1 (pre-registered floor, reason: the positive control must find at least the one truncate definition measured today; 0 is RED) |
| AC9 | R9 | section `ac commands`, `### AC9` | 1 (builder and loader each produce the sub-config with the defaults of `common_params_speculative_draft` (`common/common.h:327-352`), `block_size` 0 meaning read from GGUF metadata; the test asserts the two configs are equal field for field and prints `block_size`; the controls set `n_max` 5 through the loader and `block_size` 8 through the loader and assert each config differs from the default) |
| AC10 | R10 | section `ac commands`, `### AC10` | 1 (the test prints the full line `causality noncausal_ones = 16 causal_ones = 10` after its assertions; for a block of 4: with `dflash.attention.causal` unset the visibility is all ones, 16 ones; set, it is lower triangular, 10 ones; the test asserts both counts and that the two matrices differ) |
| AC11 | R11 | section `ac commands`, `### AC11` | 1 (the test prints the full line `trained_pair e2b_pair = ok granite_pair = width_mismatch block_size = N with N >= 1` after its assertions; loads `$DRAFT_E2B`; asserts the E2B pairing passes and the granite pairing returns the width error; blocked until `speculative-draft-training` slice 11) |
| AC12 | R4a, R4b | section `ac commands`, `### AC12` | 1 (the sibling spec's `readout_` tests, `2 tests run: 2 passed`, so the readout list the tap consumes exists; 0 tests run is RED; this is TASKS slice 0, a prerequisite owned by `decode-as-data` slice 3, discharged here as a runnable check) |
| ACL1 | RL1, RL3 | section `ac commands`, `### ACL1` | three lines printed: `51`, `51`, `1` (51 prompts processed; 51 of them identical to plain greedy with at least one drafted token; the summary line states prompts=51, identical=51 and a nonzero `rejected_total` (>= 1, pre-registered, status plausible), so the rollback path ran; acceptance is printed in `accepted_total` and `mean_accepted_per_step` and is not gated for a random draft) |
| ACL2 | RL1, RL2, RL3 | section `ac commands`, `### ACL2` | three lines printed: `51`, `51`, `1` (every prompt identical to plain greedy, with `n_drafted` and `accepted` both above 0 on every prompt; the summary line gates `rejected_total` >= 1 (pre-registered, plausible): lossless output with zero rollbacks over all 51 prompts means verify never disagreed with the draft, so the rollback path was never exercised by that draft; `rejected_total` is a corpus sum, so a trained draft that rejects nothing on some prompts still passes; ACL1's random draft proves rollback independent of the weights; blocked until `speculative-draft-training` slice 11 places `$DRAFT_E2B`) |
| ACL3 | RL1, RL3, RL4 | section `ac commands`, `### ACL3` | three lines printed: `51`, `51`, `1` (the summary line gates `rejected_total` >= 1, pre-registered, plausible, as in ACL1; blocked until the granite verify program exists: the arming check in the second block of `### ACL3` prints 0 today, and `speculative-draft-training` slice 9 makes it 1) |
| ACL4 | RL1, RL2, RL3, RL4 | section `ac commands`, `### ACL4` | three lines printed: `51`, `51`, `1` (the summary line gates `rejected_total` >= 1, pre-registered, plausible, with the ACL2 reason: zero rollbacks over 51 prompts means the rollback path was not exercised by that draft; blocked until `speculative-draft-training` slice 12 places `$DRAFT_GRANITE`; struck to a recorded shape refusal if that slice finds granite's shape cannot carry the draft) |
| ACL5 | RL5 | section `ac commands`, `### ACL5` | 3 (three tests, each asserting before it prints: `lossless_comparator_self_passes` prints `control=self compared=51 identical=51`; `lossless_comparator_rejects_flipped_token` flips one id of one prompt and prints `control=flip compared=51 identical=50`; `lossless_guard_rejects_zero_draft` feeds 51 prompts that drafted nothing and prints `control=zero_draft drafted_prompts=0 flagged=true`) |
| ACL6 | RL4 | section `ac commands`, `### ACL6` | 1 (three cases and a control, asserted before the print: (1) a granite descriptor with `speculative_verify` set false, built by the test and independent of whether `granitemoe.toml` has been armed yet, with `draft-dflash` selected returns the typed `InteropError` refusal whose message names `speculative_verify`, counted by `unarmed_refused` = 1; (2) the same granite descriptor with `speculative_verify` set true returns no refusal; (3) the gemma4 E2B descriptor, whose profile arms verify, returns no refusal; `armed_refused` counts refusals in cases (2) and (3) and is 0; the control feeds the matcher a different typed error (the missing draft path error of the config tests) and asserts the matcher rejects it, printed as `control_wrong_error_rejected = true`; the test asserts `cases` = 3 so a run that exercised no case cannot print; the grep verified on synthetic lines: `cases = 2`, `unarmed_refused = 0`, `armed_refused = 1`, `control_wrong_error_rejected = false` and empty output each print 0, the passing line prints 1) |

## out of scope

- llama.cpp parity as a gate: the only DFlash pair named upstream is Qwen3-4B with its published draft (`docs/speculative.md:63-73` in the llama.cpp checkout), and the owner removed Qwen3 and every llama fixture from the test set (2026-10-12); matching llama.cpp's draft zoo is not a goal
- the DFlash2 local convolution (`build_dflash2_conv`, `src/models/dflash.cpp:408` onward) as a numeric requirement: its only oracle is llama's output on a third-party checkpoint, and `speculative-draft-training` trains DFlash1; the conv tensors are still resolved by name (R2) and the selector decode is covered by hand-built lattices (R7)
- draft-dspark (Markov head, anchor-first layout, confidence-head truncation): `speculative-draft-dspark`
- M-RoPE position handling for DFlash embd batches (upstream commit `b10f9ca58`): no M-RoPE target exists in proxima yet; unblocked when a vision or M-RoPE target lands
- converting llama-format DFlash checkpoints from Hugging Face: proxima consumes GGUF it exports itself
- training the draft: `speculative-draft-training`
- the readout list, visibility inputs and shape pipes themselves (`decode-as-data` slices 1-3): owned by that spec; this sub-spec consumes them

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the tap readout changes the lowered program's shape for every family, not just the target | low | wider blast radius than this sub-spec's scope | the readout list is opt-in per call (`decode-as-data` AC7's logits-only digest test) |
| a random draft rejects nearly everything, so the accept path is exercised only by scripted blocks before training lands | high | the accept-heavy half runs only under the trained draft | AC6 and AC7 exercise the block read with scripted outputs, independent of any model |
| granite's profile does not arm verify, so ACL3 and ACL4 cannot run | certain today | granite arms stay blocked | `speculative-draft-training` slice 9 owns arming it |

## context

### background

- gap: proxima has no representation of a block-diffusion draft model -- the second descriptor of `decode-as-data` R6 has no masked-block input, no cross-model feature injection and no `dflash.*` metadata reads (a case-insensitive `dflash` search of the tracked files under `proxima-model-interop/src`, `proxima-gguf/src` and `proxima-tensor/src` lists only `serving.rs` and `speculative_settings.rs`, where only the llama type name exists)
- incumbent: llama.cpp's `draft-dflash` speculation type (`common/speculative.cpp:923-1329`, upstream `f1ea20621`) drafts a whole block of tokens per step by injecting the target model's hidden states into a second, smaller model's KV cache; it has zero coverage on the proxima side
- owner direction (2026-10-12): no DFlash draft exists for gemma4 E2B or granite 1b, and the owner's direction is to train our own drafts rather than match llama.cpp's draft zoo
- scope: this sub-spec of `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md` (R11c) makes `draft-dflash` available through `ServingConfig`'s speculative section as data on the `decode-as-data` frame -- hidden-state tap readouts on the target (`decode-as-data` R5), a second `ModelDescriptor` with its own `ServingState` (`decode-as-data` R6), and a bidirectional block shape (`decode-as-data` visibility inputs)
- correctness relation: speculative decoding with greedy acceptance is lossless, so its output must equal the target's own plain greedy decode token for token (the chain is incumbent ~ plain decode, an owner-stated premise this spec does not re-measure, then speculative == plain decode)
- it adds no technique-named type (`decode-as-data` AC9)
- thresholds in the ACs (`p_min`, `n_min` and tolerance values used as test parameters, `rejected_total` >= 1) are pre-registered, status plausible, not yet measured; `R9` defaults are derived from `common/common.h:327-352`

- proxima: `proxima-tensor/src/spec/descriptor.rs:130-296` (`ModelDescriptor`), `proxima-model-interop/src/lowering.rs:118-189` (`BoundProgram`; `hidden_root` at `:137`, `residual_roots` at `:141`), `proxima-model-interop/src/profiles/gemma4.toml`, `proxima-model-interop/src/generate/draft_model.rs:174-183` (`load_draft`), `proxima-model-interop/src/generate/drafter.rs`, `proxima-model-interop/src/speculative_settings.rs:36-59` (`SpeculativeTypeName`), `proxima-model-interop/src/serving.rs:205-217` (`SpeculativeType`), `proxima-gguf/src/parser.rs`, `proxima-gguf/src/value.rs`, `proxima-model-interop/src/generate/residency_caches.rs` (`LayerCache::truncate`), `proxima-model-interop/examples/speculative_decode_parity.rs`, `proxima-tensor/specs/decode-as-data/SPEC.md`, `proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md`, `proxima-tensor/specs/speculative-draft-eagle3/SPEC.md` (the lossless harness), `proxima-tensor/specs/speculative-draft-training/SPEC.md`
- llama.cpp `f1ea20621` at `~/repos/others/llama.cpp`, design references only (read with `git show f1ea20621:<path>`): `common/speculative.cpp:923-1329` (`common_speculative_impl_draft_dflash`), `:990-992` (`dflash.attention.causal` read), `:1098-1181` (`process`), `:1183-1324` (`draft`), `:1205-1210` (block rows), `:1237-1265` (selector lattice), `:1295-1318` (DFlash1 read, the else branch of the `is_dspark` test), `:1326-1328` (no-op `accept`), `src/llama-arch.cpp:370-374,702-708` (keys, tensor names), `src/llama-hparams.h:250-254`, `common/common.h:327-352`

## ac commands

Each command is literal shell; run it from the repository root of a checkout of this repository.

The variables defined in the acceptance criteria preamble are bound first:

```sh
export E2B=/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd
export GRANITE=/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80
export CORPUS=proxima-model-interop/examples/data/speculative_corpus.jsonl
export DRAFT_E2B=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/e2b-dflash.gguf
export DRAFT_GRANITE=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/granite-dflash.gguf
export OUT=target/spec-dflash-ac
export LOSSLESS_ANY='^lossless id=[^ ]+ identical=(true|false) n_drafted=[0-9]+ accepted=[0-9]+ rejected=[0-9]+ first_divergence=[^ ]+$'
export LOSSLESS_DRAFTED='^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[0-9]+ rejected=[0-9]+ first_divergence=[^ ]+$'
export LOSSLESS_ACCEPTED='^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* rejected=[0-9]+ first_divergence=[^ ]+$'
export SUMMARY_LINE='^summary prompts=51 identical=51( [a-z_]+=[0-9.]+)* rejected_total=[1-9][0-9]*( [a-z_]+=[0-9.]+)*$'
mkdir -p "$OUT"
```

### AC1

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_descriptor_reads_known_keys/)' 2>&1 | grep -cE '^dflash_keys keys_read = 7 block_size = 8 target_layers = \[3, 7, 11\] selector_top_k = 0 class = DFlash1 causal = false perturbed_differs = true perturbed_class = DFlash2 missing_block_size_rejected = true missing_target_layers_rejected = true wrong_length_rejected = true non_i32_rejected = true$'
```

### AC2

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_tensor_names_resolve/)' 2>&1 | grep -cE '^dflash_tensors tensors_resolved = 7 absent_resolved = 0 class_unchanged_by_tensors = true$'
```

### AC3

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_masked_block_shape/)' 2>&1 | grep -cE '^masked_block block_size = 4 mask_positions = 3 positions = 100,101,102,103 read_positions = 1,2,3$'
```

### AC4a

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_rows_per_layer/)' 2>&1 | grep -cE '^tap_rows positions = 64 layers = 3 rows_per_layer = 64 layer_ids = \[2, 13, 25\] gathered_cols_equal_layers_times_n_embd = true layer_ids_equal_descriptor = true control_other_ids_rejected = true$'
```

### AC4b

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_equals_independent_forward/)' 2>&1 | grep -cE '^tap_independent positions = 64 layers = 3 independent_max_abs_diff = 0\.[0-9]+ control_self = 0\.0 control_perturbed_rejected = true wrong_layer_rejected = true$'
```

### AC4c

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_metal_equals_cpu/)' 2>&1 | grep -cE '^tap_metal positions = 64 layers = 3 layer_ids = \[2, 13, 25\] max_abs_diff = 0\.[0-9]+ control_self = 0\.0 control_perturbed_rejected = true$'
```

### AC5

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_features_reach_draft_in_one_decode_call/)' 2>&1 | grep -cE '^injection chunks = 12 decode_calls_per_chunk = 1 split_path_calls_per_chunk = 2$'
```

### AC6

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash1_block_decode_reads_block/)' 2>&1 | grep -cE '^block_decode_dflash1 cases = 4 p_min_stops = 3 n_min_clears = 1$'
```

### AC7

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash2_selector_decode_lattice/)' 2>&1 | grep -cE '^selector_decode cases = 4 truncated = 1 n_min_clears = 1 ablated_differs = true$'
```

### AC8

```sh
git ls-files -- proxima-model-interop/src | wc -l
git ls-files -- proxima-tensor/src | wc -l
git grep -ciE 'DflashRollback|dflash.*snapshot' -- proxima-model-interop/src proxima-tensor/src | wc -l
git grep -ciE 'DflashRollback|dflash.*snapshot|fn truncate' -- proxima-model-interop/src proxima-tensor/src | wc -l
```

### AC9

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/speculative_config_draft_dflash_defaults/)' 2>&1 | grep -cE '^dflash_config block_size = 0 builder_equals_loader = true builder_n_max = 3 builder_n_min = 0 p_min = 0 backend_sampling = true override_differs = true block_size_override_differs = true$'
```

### AC10

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_attention_causality_switch/)' 2>&1 | grep -cE '^causality noncausal_ones = 16 causal_ones = 10$'
```

### AC11

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_trained_draft_loads_and_pairs/)' 2>&1 | grep -cE '^trained_pair e2b_pair = ok granite_pair = width_mismatch block_size = [1-9][0-9]*$'
```

### AC12

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -cE '^ +Summary \[ *[0-9.]+s\] 2 tests run: 2 passed, [0-9]+ skipped$'
```

### ACL1

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-random 17 --max-tokens 64 "$E2B" > "$OUT/lossless_dflash_e2b_random.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_e2b_random.txt"
grep -cE "$LOSSLESS_DRAFTED" "$OUT/lossless_dflash_e2b_random.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_e2b_random.txt"
```

### ACL2

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-model "$DRAFT_E2B" --max-tokens 64 "$E2B" > "$OUT/lossless_dflash_e2b_trained.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_e2b_trained.txt"
grep -cE "$LOSSLESS_ACCEPTED" "$OUT/lossless_dflash_e2b_trained.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_e2b_trained.txt"
```

### ACL3

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-random 17 --max-tokens 64 "$GRANITE" > "$OUT/lossless_dflash_granite_random.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_granite_random.txt"
grep -cE "$LOSSLESS_DRAFTED" "$OUT/lossless_dflash_granite_random.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_granite_random.txt"
```

The arming check named in ACL3, which prints 0 today and 1 once `speculative-draft-training` slice 9 lands:

```sh
grep -c '^speculative_verify = true$' proxima-model-interop/src/profiles/granitemoe.toml
```

### ACL4

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-model "$DRAFT_GRANITE" --max-tokens 64 "$GRANITE" > "$OUT/lossless_dflash_granite_trained.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_granite_trained.txt"
grep -cE "$LOSSLESS_ACCEPTED" "$OUT/lossless_dflash_granite_trained.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_granite_trained.txt"
```

### ACL5

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self compared=51 identical=51|flip compared=51 identical=50|zero_draft drafted_prompts=0 flagged=true)$'
```

### ACL6

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_granite_verify_arming_/)' 2>&1 | grep -cE '^arming method=dflash cases = 3 unarmed_refused = 1 error_names_speculative_verify = true armed_refused = 0 control_wrong_error_rejected = true$'
```
