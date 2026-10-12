# speculative-draft-dspark -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

Every validation command is literal, in the commands section below, one fenced block per slice; the paths are the absolute paths defined in the acceptance criteria preamble of SPEC.md.

| # | slice | discharges | expected | done | note |
|---|---|---|---|---|---|
| 0 | prerequisites owned by sibling specs: the `build_forward` readout list (`decode-as-data` slice 3), the lossless harness (`speculative-draft-eagle3` slice 1) and the DFlash block decode (`speculative-draft-dflash` slice 7). This sub-spec consumes them and does not re-implement them | prerequisite, no AC (the three commands check sibling slices: the second is the ACL5 test set, which belongs to ACL5's own slice in the lossless harness spec) | three lines: `1`, `3`, `1` | [ ] | fails until those slices land |
| 1 | classify DSpark versus DFlash from tensor presence (R1) | AC1 | 1 | [ ] | depends on the DFlash sub-spec's GGUF-side reads of the shared dflash keys existing first |
| 2 | descriptor resolves the Markov and confidence tensors (R2) | AC2 | 3 | [ ] | |
| 3 | descriptor reads `dflash.sample_from_anchor` and `dflash.has_confidence_head` (R3) | AC3 (R3, R3a) | 1 | [ ] | |
| 4 | Markov head readout roots on the DFlash draft descriptor; bias `markov_w2 @ markov_w1[prev_token]` in the shared draft loop (R4) | AC4 (R4) | 1 | [ ] | |
| 5 | confidence-gate truncation `sigmoid(conf_proj . [feat; markov_w1[prev]] + bias) < p_min` (R5) | AC5 | 1 | [ ] | |
| 6 | typed error when `p_min > 0.0` and no confidence head (R6) | AC6 (R6, R6a) | 1 | [ ] | |
| 7 | `sample_from_anchor` layout dispatch: anchor-first versus bonus-anchor (R7) | AC7 (R7, R7a) | 1 | [ ] | |
| 8 | structural check: no type named for DSpark exists, with its control (R9) | AC8 | two lines: `0`, then >= 1 | [ ] | if the first is above 0, the fix is deleting or folding the type into the DFlash descriptor, never an `allow` |
| 9 | lossless over a random draft on E2B, 51 prompts (RL1, RL3) | ACL1, ACL5 | four lines: `51`, `51`, `1`, `3` | [ ] | a count below 51 on the second line is a lossless failure |
| 10 | lossless over the trained E2B draft, `accepted` > 0 on every prompt (RL1, RL2); also loads and pairs the trained draft (R10) | ACL2, AC9, ACL5 | five lines: `51`, `51`, `1` (corpus-level `rejected_total` >= 1), `1` (AC9 load and pair), `3` (ACL5 controls) | [ ] | blocked on `speculative-draft-training` slice 11; the `rejected_total` gate reason is in SPEC.md ACL2 (threshold >= 1 pre-registered, plausible) |
| 11 | `dspark_markov_ablation` example: runs the draft loop per block over a 200-block run with `--p-min 0.5`, once with the Markov bias computed and once skipped, and once with the confidence gate active and once forced off; prints `markov_differing_blocks=N executed_blocks=B`, `conf_differing_blocks=N executed_blocks=B` and `p_min_stops=M executed_blocks=B`, where B is a counter incremented per executed block, not the `--blocks` argument; `--zero-markov` is the control | AC10, AC11 | two lines: `1`, `1` | [ ] | a printed `markov_differing_blocks=0` on the first command means stop: per the Markov refutation condition R4 is struck |
| 12 | granite arm over a random draft (ACL3) | ACL3 | three lines: `51`, `51`, `1` | [ ] | blocked on `speculative-draft-training` slice 9 (granite verify) |
| 13 | lossless over the trained granite draft, `accepted` > 0 on every prompt (RL1, RL2, RL4) | ACL4 | three lines: `51`, `51`, `1` | [ ] | blocked on `speculative-draft-training` slice 12 and slice 9's granite verify; the `rejected_total` >= 1 gate has the reason in SPEC.md ACL2; struck to a recorded shape refusal if granite's shape cannot carry the draft |
| 14 | typed `speculative_verify` refusal on an unarmed profile, no refusal on an armed one (RL4) | ACL6 | 1 | [ ] | runs today, independent of `speculative-draft-training` slice 9 |
| 15 | structural check: DSpark reuses DFlash's block batching, feature extraction and injection (R9a), with the planted control | AC12 | three lines: `3 3`, `0`, then >= 3 | [ ] | needs `speculative-draft-dflash` slices 3, 5 and 6 to have created the three functions; a first value other than `3 3` means DSpark reimplemented or bypassed the shared work: fold it into the shared function, never an `allow` |
| 16 | confidence gate fires and is load-bearing (R11): `p_min_stops` >= 1 and `conf_differing_blocks` >= 1 at `--p-min 0.5` | AC13 | two lines: `1`, `1` | [ ] | blocked on `speculative-draft-training` slice 11; `p_min_stops=0` or `conf_differing_blocks=0` strikes R5, independently of slice 11 |
| 17 | executed-block counter control: `--blocks 50` prints `executed_blocks=50` on all three lines | AC14 | 3 | [ ] | blocked on `speculative-draft-training` slice 11 |
| 18 | each shared function emits one `trace!` event with `function = "<its name>"` per call (R12): adds the one line to `masked_block_rows`, `tap_feature_rows` and `inject_feature_rows` | AC18 | 1 | [ ] | explicit dependency: `speculative-draft-dflash` slices 3, 5 and 6 create the functions first and do not themselves require the event; this slice edits those function bodies after they land |
| 19 | Markov head early return on an oversized block (R7b) | AC16 | 1 | [ ] | the early return leaves the logits raw and raises no error |
| 20 | confidence-gate control: `--p-min 0.0` prints `conf_differing_blocks=0` and `p_min_stops=0` (R11) | AC17 | two lines: `1`, `1` | [ ] | blocked on `speculative-draft-training` slice 11; a nonzero count here voids AC13 |
| 21 | positive call-count of the shared functions through the DSpark path (R9a) | AC15 | 1 | [ ] | needs slice 18 and `speculative-draft-dflash` slices 3, 5 and 6 |


## commands

Each block is the literal validation command of the slice with the same number; run it from the worktree root.

### slice 0

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -c 'Summary .* 2 tests run: 2 passed'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self|flip|zero_draft) '; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dflash1_block_decode_reads_block/)' 2>&1 | grep -cE 'cases = 4 p_min_stops = 3 n_min_clears = 1'
```

### slice 1

```sh
cargo nextest run -p proxima-gguf --no-capture -E 'test(/classifies_dspark_from_markov_tensor/)' 2>&1 | grep -cE 'cases = 4 dspark = 1 dflash = 1 neither = 2'
```

### slice 2

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_descriptor_resolves_markov_and_confidence_tensors/)' 2>&1 | grep -cE '^tensors_found = (0|2|4)$'
```

### slice 3

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_descriptor_reads_anchor_and_confidence_keys/)' 2>&1 | grep -cE 'keys_read = 2 absent_anchor = true absent_conf = true flipped_differs = true non_true_strings_false = true'
```

### slice 4

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_markov_bias_worked_example/)' 2>&1 | grep -cE 'positions = 3 chain = 2,2,2 max_abs_diff = [0-9.e-]+ skipped_chain = 0,0,0 raw_prev_chain = 2,2,0 tie_pick = 1 tie_control = 0'
```

### slice 5

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_confidence_gate_worked_example/)' 2>&1 | grep -cE 'cases = 4 kept = 1,2,0,0 truncated_cases = 3'
```

### slice 6

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_missing_confidence_head_is_typed_error/)' 2>&1 | grep -cE 'cases = 5 flag_false_p_min_positive_flag_error = true flag_false_p_min_zero_ok = true flag_absent_no_tensor_p_min_positive_tensor_error = true flag_true_p_min_positive_ok = true flag_absent_tensors_p_min_positive_ok = true'
```

### slice 7

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_sample_from_anchor_layout/)' 2>&1 | grep -cE 'anchor_first_rows = 4 anchor_first_read_from = 0 bonus_rows = 5 bonus_read_from = 1 clamp_anchor_first = 4 clamp_bonus = 3'
```

### slice 8

```sh
git grep -niP '\b(trait|impl|struct|enum)\b[^;{]*Dspark' -- proxima-model-interop/src proxima-tensor/src proxima-gguf/src | wc -l; git grep -niP '\b(trait|impl|struct|enum)\b[^;{]*Ngram' -- proxima-model-interop/src proxima-tensor/src proxima-gguf/src | wc -l
```

### slice 9

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-dspark --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_dspark_e2b_random.txt 2>&1; grep -c '^lossless ' lossless_dspark_e2b_random.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* ' lossless_dspark_e2b_random.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_dspark_e2b_random.txt
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self|flip|zero_draft) '
```

### slice 10

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-dspark --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_dspark_e2b_trained.txt 2>&1; grep -c '^lossless ' lossless_dspark_e2b_trained.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_dspark_e2b_trained.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_dspark_e2b_trained.txt; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_trained_draft_loads_and_pairs/)' 2>&1 | grep -cE 'class = dspark tensors_found = [2-4] e2b_pair = ok granite_pair = width_mismatch'
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self|flip|zero_draft) '
```

### slice 11

```sh
cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --p-min 0.5 --blocks 200 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > ablation.txt 2>&1; grep -cE '^markov_differing_blocks=[1-9][0-9]* executed_blocks=200$' ablation.txt; cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --p-min 0.5 --zero-markov --blocks 200 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > ablation_zero.txt 2>&1; grep -cE '^markov_differing_blocks=0 executed_blocks=200$' ablation_zero.txt
```

### slice 12

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-dspark --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_dspark_granite_random.txt 2>&1; grep -c '^lossless ' lossless_dspark_granite_random.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* ' lossless_dspark_granite_random.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_dspark_granite_random.txt
```

### slice 13

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-dspark --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/granite-dspark.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_dspark_granite_trained.txt 2>&1; grep -c '^lossless ' lossless_dspark_granite_trained.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_dspark_granite_trained.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_dspark_granite_trained.txt
```

### slice 14

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_granite_verify_arming_/)' 2>&1 | grep -cE '^arming method=dspark cases = 3 unarmed_refused = 1 error_names_speculative_verify = true armed_refused = 0 control_wrong_error_rejected = true$'
```

### slice 15

```sh
for fn in masked_block_rows tap_feature_rows inject_feature_rows; do defs=$(git grep -hP "fn $fn\b" -- proxima-model-interop/src proxima-tensor/src | wc -l); calls=$(git grep -hP "\b$fn\(" -- proxima-model-interop/src proxima-tensor/src | grep -vc "fn $fn"); echo "$fn $defs $calls"; done | awk '{ok += ($2==1 && $3>=1)} END{print NR, ok}'; git grep -hiP 'fn [a-z0-9_]*(dspark|markov)[a-z0-9_]*(block|tap|inject)' -- proxima-model-interop/src proxima-tensor/src | wc -l; git grep -hP 'fn [a-z0-9_]*(block|tap|inject)' -- proxima-model-interop/src proxima-tensor/src | wc -l
```

### slice 16

```sh
cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --p-min 0.5 --blocks 200 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > ablation_conf.txt 2>&1; grep -cE '^conf_differing_blocks=[1-9][0-9]* executed_blocks=200$' ablation_conf.txt; grep -cE '^p_min_stops=[1-9][0-9]* executed_blocks=200$' ablation_conf.txt
```

### slice 17

```sh
cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --p-min 0.5 --blocks 50 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > ablation_50.txt 2>&1; grep -cE ' executed_blocks=50$' ablation_50.txt
```

### slice 18

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/shared_functions_emit_named_trace_events/)' 2>&1 | grep -cE '^events = masked_block_rows,tap_feature_rows,inject_feature_rows each = 1 control_double_call = 2,1,1$'
```

### slice 19

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_markov_head_oversized_block_returns_early/)' 2>&1 | grep -cE '^block_size = 3 biased_len3_chain = 2,2,2 oversized_len4_max_abs_diff = 0 oversized_error = none control_block_size_4_biased = true$'
```

### slice 20

```sh
cargo run --release -p proxima-model-interop --features std --example dspark_markov_ablation -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf --p-min 0.0 --blocks 200 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > ablation_conf_off.txt 2>&1; grep -cE '^conf_differing_blocks=0 executed_blocks=200$' ablation_conf_off.txt; grep -cE '^p_min_stops=0 executed_blocks=200$' ablation_conf_off.txt
```

### slice 21

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/dspark_draft_goes_through_shared_functions/)' 2>&1 | grep -cE '^run_a blocks = 3 calls = 3,3,3 run_b blocks = 5 calls = 5,5,5$'
```

## resume

Last landed slice: none
Next action: slice 0 once the three prerequisite slices land; slices 1 to 8 need only the DFlash descriptor reads
Open question: none

## struck

- 2026-10-12: slices 0b, 1, 2, 2a, 2b, 2c, 8, 9 of the earlier cut (llama.cpp oracle build, gemma-26B or Qwen3.6 DSpark checkpoint acquisition, llama fixture generator, scaled/unscaled bias fixture, vendored oracle ids, fixture provenance headers, oracle comparator controls, llama seed parity) -- the owner removes llama.cpp token parity as a gate and removes the 26B and Qwen fixtures; the fixtures existed only to feed those gates
