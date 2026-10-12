# speculative-draft-dflash -- slices

Each slice: one commit, one behaviour change, one validation command block, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

Every validation command is literal and lives in the `## commands` section below, one fenced block per slice, so no table escape sits inside a command; the block at the top of that section exports every variable the blocks use, with the values defined in the acceptance criteria preamble of SPEC.md. Run each block from the repository root, line by line. Slice 0 gates on sibling-spec work and discharges AC12 (its first command) and ACL5 (its second).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisites owned by sibling specs: the `build_forward` readout list (`decode-as-data` slice 3) and the lossless harness (`speculative-draft-eagle3` slice 1). This sub-spec consumes both and does not re-implement them | AC12 (first command), ACL5 (second command) | commands, slice 0 | two lines: `1`, `3` | [ ] | fails until those slices land; the second line is the harness's comparator and guard tests run as a presence check, discharged by slice 11 |
| 1 | descriptor reads the seven `dflash.*` keys (including a non-empty `i32` `dflash.target_layers`) and classifies DFlash1 versus DFlash2 by `selector_top_k` (R1) | AC1 | commands, slice 1 | 1 | [ ] | |
| 2 | descriptor resolves the seven conv and selector tensor names when present, without changing the classification (R2) | AC2 | commands, slice 2 | 1 | [ ] | |
| 3 | masked-block input row layout (R3) | AC3 | commands, slice 3 | 1 | [ ] | the row layout is the function `masked_block_rows`, defined once (`speculative-draft-dspark` AC12) |
| 4 | non-causal block attention unless `dflash.attention.causal` is set (R10) | AC10 | commands, slice 4 | 1 | [ ] | |
| 5 | target-layer tap through the readout list: rows per layer and layer ids (R4a), equality with an independent forward (R4b), Metal equals CPU (R4c) | AC4a, AC4b, AC4c | commands, slice 5 | three lines: `1`, `1`, `1` | [ ] | needs slice 0; the extraction is the function `tap_feature_rows`, defined once (`speculative-draft-dspark` AC12) |
| 6 | one-call embd injection into the draft's decode (R5) | AC5 | commands, slice 6 | 1 | [ ] | the injection is the function `inject_feature_rows`, defined once (`speculative-draft-dspark` AC12) |
| 7 | DFlash1 block decode: per-position top-1, `p_min` stop, `n_min` clear (R6) | AC6 | commands, slice 7 | 1 | [ ] | |
| 8 | DFlash2 selector-lattice decode: predecessor-argmax chain from predecessor 0, softmax `p_min` truncation, `n_min` clear (R7) | AC7 | commands, slice 8 | 1 | [ ] | |
| 9 | confirm no dflash-specific rollback type exists (R8 is a negative requirement; this slice is the grep, plus its pathspec checks and positive control) | AC8 | commands, slice 9 | four lines: `151`, `72`, `0`, `1` measured at worktree HEAD f1f00666a (an ancestor of the current origin/main); gate: >= 1, >= 1 (pre-registered floors, reason: tracked-file counts move with every landed commit), exactly `0`, >= 1 (pre-registered floor, reason: the positive control must find the measured truncate definition; 0 is RED) | [ ] | |
| 10 | `draft-dflash` sub-config with defaults `n_max` 3, `n_min` 0, `p_min` 0.0, `backend_sampling` true, `block_size` 0, builder and loader parity (R9) | AC9 | commands, slice 10 | 1 | [ ] | lands after parent slice 8 (the `ServingConfig` speculative section) |
| 11 | lossless comparator and guard controls (RL5), then lossless over a random draft on E2B, 51 prompts (RL1, RL3) | ACL5, ACL1 | commands, slice 11 | four lines: `3`, `51`, `51`, `1` | [ ] | needs slice 0; a count below 51 on the third line invokes the refutation condition |
| 12 | lossless over the trained E2B draft, `accepted` > 0 on every prompt (RL1, RL2); also loads and pairs the trained draft (R11) | ACL2, AC11 | commands, slice 12 | four lines: `1`, `51`, `51`, `1` (AC11, 51 prompts, 51 identical with `accepted` > 0, `rejected_total` >= 1 pre-registered, plausible: zero rollbacks over 51 prompts means verify never disagreed with the draft) | [ ] | blocked on `speculative-draft-training` slice 11 |
| 13 | granite arm over a random draft (ACL3) | ACL3 | commands, slice 13 | three lines: `51`, `51`, `1` | [ ] | blocked on `speculative-draft-training` slice 9 (granite verify) |
| 14 | lossless over the trained granite draft, `accepted` > 0 on every prompt (RL1, RL2, RL4) | ACL4 | commands, slice 14 | three lines: `51`, `51`, `1` | [ ] | blocked on `speculative-draft-training` slice 12 (places the granite draft) and slice 9 (arms granite verify); struck to a recorded shape refusal if granite's shape cannot carry the draft |
| 15 | typed `speculative_verify` refusal on an unarmed profile, no refusal on an armed one (RL4) | ACL6 | commands, slice 15 | 1 | [ ] | runs today, independent of `speculative-draft-training` slice 9 |

## commands

Variables, exported once before any slice block:

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

Slice 0:

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -cE '^ +Summary \[ *[0-9.]+s\] 2 tests run: 2 passed, [0-9]+ skipped$'
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self compared=51 identical=51|flip compared=51 identical=50|zero_draft drafted_prompts=0 flagged=true)$'
```

Slice 1:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_descriptor_reads_known_keys/)' 2>&1 | grep -cE '^dflash_keys keys_read = 7 block_size = 8 target_layers = \[3, 7, 11\] selector_top_k = 0 class = DFlash1 causal = false perturbed_differs = true perturbed_class = DFlash2 missing_block_size_rejected = true missing_target_layers_rejected = true wrong_length_rejected = true non_i32_rejected = true$'
```

Slice 2:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_tensor_names_resolve/)' 2>&1 | grep -cE '^dflash_tensors tensors_resolved = 7 absent_resolved = 0 class_unchanged_by_tensors = true$'
```

Slice 3:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_masked_block_shape/)' 2>&1 | grep -cE '^masked_block block_size = 4 mask_positions = 3 positions = 100,101,102,103 read_positions = 1,2,3$'
```

Slice 4:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_attention_causality_switch/)' 2>&1 | grep -cE '^causality noncausal_ones = 16 causal_ones = 10$'
```

Slice 5:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_rows_per_layer/)' 2>&1 | grep -cE '^tap_rows positions = 64 layers = 3 rows_per_layer = 64 layer_ids = \[2, 13, 25\] gathered_cols_equal_layers_times_n_embd = true layer_ids_equal_descriptor = true control_other_ids_rejected = true$'
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_equals_independent_forward/)' 2>&1 | grep -cE '^tap_independent positions = 64 layers = 3 independent_max_abs_diff = 0\.[0-9]+ control_self = 0\.0 control_perturbed_rejected = true wrong_layer_rejected = true$'
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_target_tap_metal_equals_cpu/)' 2>&1 | grep -cE '^tap_metal positions = 64 layers = 3 layer_ids = \[2, 13, 25\] max_abs_diff = 0\.[0-9]+ control_self = 0\.0 control_perturbed_rejected = true$'
```

Slice 6:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_features_reach_draft_in_one_decode_call/)' 2>&1 | grep -cE '^injection chunks = 12 decode_calls_per_chunk = 1 split_path_calls_per_chunk = 2$'
```

Slice 7:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash1_block_decode_reads_block/)' 2>&1 | grep -cE '^block_decode_dflash1 cases = 4 p_min_stops = 3 n_min_clears = 1$'
```

Slice 8:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash2_selector_decode_lattice/)' 2>&1 | grep -cE '^selector_decode cases = 4 truncated = 1 n_min_clears = 1 ablated_differs = true$'
```

Slice 9:

```sh
git ls-files -- proxima-model-interop/src | wc -l
git ls-files -- proxima-tensor/src | wc -l
git grep -ciE 'DflashRollback|dflash.*snapshot' -- proxima-model-interop/src proxima-tensor/src | wc -l
git grep -ciE 'DflashRollback|dflash.*snapshot|fn truncate' -- proxima-model-interop/src proxima-tensor/src | wc -l
```

Slice 10:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/speculative_config_draft_dflash_defaults/)' 2>&1 | grep -cE '^dflash_config block_size = 0 builder_equals_loader = true builder_n_max = 3 builder_n_min = 0 p_min = 0 backend_sampling = true override_differs = true block_size_override_differs = true$'
```

Slice 11:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self compared=51 identical=51|flip compared=51 identical=50|zero_draft drafted_prompts=0 flagged=true)$'
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-random 17 --max-tokens 64 "$E2B" > "$OUT/lossless_dflash_e2b_random.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_e2b_random.txt"
grep -cE "$LOSSLESS_DRAFTED" "$OUT/lossless_dflash_e2b_random.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_e2b_random.txt"
```

Slice 12:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_trained_draft_loads_and_pairs/)' 2>&1 | grep -cE '^trained_pair e2b_pair = ok granite_pair = width_mismatch block_size = [1-9][0-9]*$'
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-model "$DRAFT_E2B" --max-tokens 64 "$E2B" > "$OUT/lossless_dflash_e2b_trained.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_e2b_trained.txt"
grep -cE "$LOSSLESS_ACCEPTED" "$OUT/lossless_dflash_e2b_trained.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_e2b_trained.txt"
```

Slice 13:

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-random 17 --max-tokens 64 "$GRANITE" > "$OUT/lossless_dflash_granite_random.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_granite_random.txt"
grep -cE "$LOSSLESS_DRAFTED" "$OUT/lossless_dflash_granite_random.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_granite_random.txt"
```

Slice 14:

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus "$CORPUS" --drafter draft-dflash --draft-model "$DRAFT_GRANITE" --max-tokens 64 "$GRANITE" > "$OUT/lossless_dflash_granite_trained.txt" 2>&1
grep -cE "$LOSSLESS_ANY" "$OUT/lossless_dflash_granite_trained.txt"
grep -cE "$LOSSLESS_ACCEPTED" "$OUT/lossless_dflash_granite_trained.txt"
grep -cE "$SUMMARY_LINE" "$OUT/lossless_dflash_granite_trained.txt"
```

Slice 15:

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash_granite_verify_arming_/)' 2>&1 | grep -cE '^arming method=dflash cases = 3 unarmed_refused = 1 error_names_speculative_verify = true armed_refused = 0 control_wrong_error_rejected = true$'
```

## resume

Last landed slice: none
Next action: slice 0 once `decode-as-data` slice 3 and `speculative-draft-eagle3` slice 1 land; slices 1 to 4 and 7 to 10 need neither
Open question: whether the plain block draft needs the DFlash2 conv tensors at all for the trained E2B draft; answered by `speculative-draft-training` R6

## struck

- 2026-10-12: slices 0b, 1, 1b, 2, 2a, 2b, 4c, 11 of the earlier cut (llama.cpp oracle build, Qwen3-4B DFlash pair acquisition and converter hunt, DFlash2 selector checkpoint acquisition, llama fixture generator, vendored oracle ids, local-convolution fixture comparison, llama seed parity) -- the owner removes llama.cpp token parity as a gate and removes the Qwen fixtures; the fixtures existed only to feed those gates
