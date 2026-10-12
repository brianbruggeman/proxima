# speculative-draft-eagle3 -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

`$E2B`, `$GRANITE`, `$CORPUS`, `$DRAFT_E2B`, `$DRAFT_GRANITE` are the absolute paths defined in the acceptance criteria preamble of SPEC.md; every validation command is literal and lives in the `commands` section below the table (a markdown table cell cannot carry a literal pipe).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisite, owned by `decode-as-data` slice 3: the `build_forward` readout list (logits rows, hidden-state taps at named layers, extra head roots). This sub-spec consumes it and does not re-implement it | AC13 (R2 prerequisite; the command asserts N = 2 `decode-as-data` readout tests passed, with a zero-match control) | see `commands` slice 0 | two lines: `1`, `0` | [ ] | fails until `decode-as-data` slice 3 lands |
| 1 | the lossless harness: `speculative_lossless_corpus` example (plain greedy versus speculative per prompt, `--draft-random <seed>`, `--draft-model <path>`, `--drafter <name>`), its pure comparator, a `summary prompts=<n> identical=<n> ... rejected_total=<n>` line that carries the corpus-sum `rejected_total`, and the three control tests (RL5); the example builds and runs a 1-prompt smoke (`--drafter none --limit 1`, flags `--corpus --drafter --draft-random --draft-model --max-tokens --limit`) whose summary carries a count | ACL5, ACL0 | see `commands` slice 1 | four lines: `3`; then `1`, `1`, `1` | [ ] | the other three method specs consume this example |
| 2 | draft descriptor from GGUF-key reads (12 required, 1 conditional, 4 optional tensors), the llama-written `$GRANITE` parser and writer oracle, and the `eagle3` family profile TOML `proxima-model-interop/src/profiles/eagle3.toml` resolved by `general.architecture = eagle3` (R1); no forward program yet | AC1, AC12 | see `commands` slice 2 | two lines: `1`, `1` | [ ] | |
| 3 | gemma4 all-positions hidden-state tap at 3 caller-chosen layer indices, through the readout list (R2) | AC2 | see `commands` slice 3 | 1 | [ ] | needs slice 0 |
| 4 | encoder graph: 3-layer fuse -> optional RMSNorm -> `fc` -> g_embd (R3) | AC3 | see `commands` slice 4 | 1 | [ ] | |
| 5 | decoder step graph: concat(embd_norm, g_norm) -> attention -> FFN -> output_norm -> lm_head -> optional `d2t`, the single-position attention case and the target lm_head / token_embd fallback (R4, R11) | AC4, AC11 | see `commands` slice 5 | two lines: `1`, `1` | [ ] | |
| 6 | draft loop: top-1 per step, `p_min` stop, `n_max` stop, `n_min` clear (R5) | AC5 | see `commands` slice 6 | 1 | [ ] | |
| 7 | accept-time boundary rewind (R6) | AC6 | see `commands` slice 7 | 1 | [ ] | |
| 8 | empty stash and no-op restore for an attention-only target (R7) | AC7 | see `commands` slice 8 | 1 | [ ] | |
| 9 | route the `draft-eagle3` config value to the propose pipe carrying taps and the draft descriptor, with builder/loader parity (R8; the five tests include the parity test) | AC8 | see `commands` slice 9 | 1 | [ ] | |
| 10 | pairing compatibility typed errors (R9) | AC9 | see `commands` slice 10 | 1 | [ ] | |
| 11 | lossless over a random draft on E2B, 51 prompts (RL1, RL3), and seeded determinism (same seed identical `accepted_total` and `proposed_hash`, other seed a different `proposed_hash`) | ACL1, ACL8 | see `commands` slice 11 | six lines: `51`, `51`, `1`; `1`, `0`, `4` | [ ] | a count below 51 on the second line invokes the refutation condition |
| 12 | lossless over the trained E2B draft, 51 prompts, `accepted` > 0 on every prompt (RL1, RL2); also loads and pairs the trained draft (R10) | ACL2, AC10 | see `commands` slice 12 | four lines: `51`, `51`, `1` (rejected_total >= 1, pre-registered, plausible: zero rollbacks over 51 prompts means the rollback path was never exercised by this draft), `1` (AC10) | [ ] | blocked on `speculative-draft-training` slice 11 |
| 13 | granite armed arms: lossless over a random draft (ACL3) and over the trained draft (ACL4); each must run | ACL3, ACL4 | see `commands` slice 13 | ten lines: `1`, `51`, `51`, `1`, `0` (ACL3: armed precondition, lossless, identical, summary with rejected_total >= 1, refusals) then `1`, `51`, `51`, `1`, `0` (ACL4; rejected_total >= 1, pre-registered, plausible) | [ ] | blocked on `speculative-draft-training` slice 9 (granite verify) for the random arm and slice 12 for the trained arm |
| 14 | typed `speculative_verify` refusal on an unarmed profile, no refusal on an armed one (RL4) | ACL6 | see `commands` slice 14 | 1 | [ ] | runs today, independent of `speculative-draft-training` slice 9 |
| 15 | granite refusal arms: unarmed, the harness refuses with the typed `speculative_verify` error, prints one `refused` line and no `lossless` or `summary` line (ACL3u, ACL4u); armed but shape-struck, the same with `draft_shape` (ACL4s) | ACL3u, ACL4u, ACL4s | see `commands` slice 15 | eight lines: `0`, `1`, `0`, `0` (ACL3u) then `0`, `1`, `0`, `0` (ACL4u); ACL4s only when training slice 12 strikes ACL4: `1`, `1`, `0`, `0` | [ ] | runs today while `granitemoe.toml` prints 0; replaced by slice 13 once training slice 9 arms it |
| 16 | `--tap-rows-from-prompt <k>` in the harness: true-row arm and replaced-row arm, per-arm summaries and the `tap_control` line carrying the corpus sums of `accepted` (ACL7), with the k = 0 degenerate control | ACL7 | see `commands` slice 16 | three lines: `2`, `1`, `1` | [ ] | needs slice 3 (tap), slice 11 and `speculative-draft-training` slice 11 (`$DRAFT_E2B`) |

## commands

Run each from the worktree root.

### slice 0

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -c 'Summary .* 2 tests run: 2 passed'; cargo nextest run -p proxima-tensor -E 'test(/readout_no_such_test_/)' 2>&1 | grep -c 'Summary .* 2 tests run: 2 passed'
```

expected: two lines: `1`, `0` (the second is the control: a filter that matches no test must not print 1)

### slice 1

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self|flip|zero_draft) '; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter none --limit 1 --max-tokens 16 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_smoke.txt 2>&1; grep -c '^lossless ' lossless_smoke.txt; grep -cE '^summary prompts=1 identical=1 .* accepted_total=0 rejected_total=0 ' lossless_smoke.txt; grep -cE 'drafted_prompts=0 flagged=true' lossless_smoke.txt
```

expected: four lines: `3`, then `1`, `1`, `1`

### slice 2

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_descriptor_reads_target_layers_and_tensors/)' 2>&1 | grep -cE 'target_layers = \[3, 7, 11\] target_hidden_size = 16 required_resolved = 12 conditional_resolved = 1 optional_resolved = 4 perturbed_differs = true missing_required_typed_error = true rms_epsilon_read = true norm_before_residual_read = true key_names_equal_llama = true llama_written_i32_array_len_matches_tokens = true llama_written_bool_key_read = true writer_array_bytes_equal_llama = true'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_profile_resolves_by_architecture/)' 2>&1 | grep -cE '^profile_resolution architecture = eagle3 resolved = eagle3 control_architecture = gemma4 control_resolved = gemma4 control_not_eagle3 = true unknown_architecture_typed_error = true$'
```

expected: two lines: `1`, `1`

### slice 3

```sh
cargo nextest run -p proxima-model-interop --features std,metal --no-capture -E 'test(/gemma4_hidden_tap_metal_equals_cpu/)' 2>&1 | grep -cE 'positions = 64 layers = 3 rows_per_layer = 64 layer_ids = \[2, 13, 25\] max_abs_diff = [0-9.e-]+ independent_max_abs_diff = [0-9.e-]+ control_self = 0\.0 control_perturbed_rejected = true wrong_layer_rejected = true pseudo_index_rows = 64'
```

expected: 1

### slice 4

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_encoder_worked_example/)' 2>&1 | grep -cE 'cases = 2 max_abs_diff = [0-9.e-]+ unnormed_equals_slice = true normed_differs = true ops_outside_allowed = 0'
```

expected: 1

### slice 5

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_decoder_stepwise_equals_batch/)' 2>&1 | grep -cE 'positions = 8 batch_vs_stepwise_max_abs_diff = [0-9.e-]+ g_embd_perturbed_diff = [0-9.e-]+'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_decoder_worked_example/)' 2>&1 | grep -cE '^decoder_worked_example n_embd = 2 draft_vocab = 3 target_vocab = 5 max_abs_diff = [0-9.e-]+ argmax_target_id = 4 unmapped_neg_inf = 2 wrong_residual_rejected = true norm_before_residual_max_abs_diff = [0-9.e-]+ norm_before_residual_distinct = true single_position_attention_max_abs_diff = [0-9.e-]+ single_position_perturbed_rejected = true single_position_wq_invariant = true no_d2t_max_abs_diff = [0-9.e-]+ no_d2t_neg_inf = 0 fallback_head_max_abs_diff = [0-9.e-]+ own_head_wins = true fallback_embd_max_abs_diff = [0-9.e-]+ own_embd_wins = true fallback_missing_typed_error = true ops_outside_allowed = 0$'
```

expected: two lines: `1`, `1`

### slice 6

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_draft_loop_stops/)' 2>&1 | grep -cE 'cases = 4 p_min_stops = 3 n_max_stops = 1 n_min_clears = 1'
```

expected: 1

### slice 7

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_accept_rewinds_boundary/)' 2>&1 | grep -cE 'cases = 5 wrong_row_rejected = true'
```

expected: 1

### slice 8

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_state_stash_empty_for_attention_target/)' 2>&1 | grep -cE 'stash_len = 0 restore_noop = true mutation_survives = true'
```

expected: 1

### slice 9

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/speculative_config_selects_draft_eagle3/)' 2>&1 | grep -c 'Summary .* 5 tests run: 5 passed'
```

expected: 1

### slice 10

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_pairing_/)' 2>&1 | grep -c 'Summary .* 5 tests run: 5 passed'
```

expected: 1

### slice 11

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_eagle3_e2b_random.txt 2>&1; grep -c '^lossless ' lossless_eagle3_e2b_random.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* ' lossless_eagle3_e2b_random.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_eagle3_e2b_random.txt; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > seed17_b.txt 2>&1; grep -c 'accepted_total=' seed17_b.txt; diff <(grep -oE 'accepted_total=[0-9]+|proposed_hash=[0-9a-f]+' lossless_eagle3_e2b_random.txt) <(grep -oE 'accepted_total=[0-9]+|proposed_hash=[0-9a-f]+' seed17_b.txt) | wc -l; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-random 18 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > seed18.txt 2>&1; diff <(grep -o 'proposed_hash=[0-9a-f]*' lossless_eagle3_e2b_random.txt) <(grep -o 'proposed_hash=[0-9a-f]*' seed18.txt) | wc -l
```

expected: six lines: `51`, `51`, `1` (ACL1); `1`, `0`, `4` (ACL8: same seed gives identical `accepted_total`, seed 18 differs)

### slice 12

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/e2b-eagle3.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_eagle3_e2b_trained.txt 2>&1; grep -c '^lossless ' lossless_eagle3_e2b_trained.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_eagle3_e2b_trained.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_eagle3_e2b_trained.txt; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_trained_draft_loads_and_pairs/)' 2>&1 | grep -cE 'e2b_pair = ok granite_pair = width_mismatch target_layers = \[[0-9]+, [0-9]+, [0-9]+\]'
```

expected: four lines: `51`, `51`, `1` (rejected_total >= 1, pre-registered, plausible: zero rollbacks over 51 prompts means the rollback path was never exercised by this draft), `1` (AC10)

### slice 13

```sh
grep -c '^speculative_verify = true' proxima-model-interop/src/profiles/granitemoe.toml; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_eagle3_granite_random.txt 2>&1; grep -c '^lossless ' lossless_eagle3_granite_random.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* ' lossless_eagle3_granite_random.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_eagle3_granite_random.txt; grep -c '^refused ' lossless_eagle3_granite_random.txt; grep -c '^speculative_verify = true' proxima-model-interop/src/profiles/granitemoe.toml; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/granite-eagle3.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_eagle3_granite_trained.txt 2>&1; grep -c '^lossless ' lossless_eagle3_granite_trained.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_eagle3_granite_trained.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_eagle3_granite_trained.txt; grep -c '^refused ' lossless_eagle3_granite_trained.txt
```

expected: ten lines: `1`, `51`, `51`, `1`, `0` (ACL3, armed random arm, rejected_total >= 1 proves rollback independent of the weights) then `1`, `51`, `51`, `1`, `0` (ACL4, armed trained arm; rejected_total >= 1, pre-registered, plausible: zero rollbacks over 51 prompts means the rollback path was never exercised by this draft); a first line of `0` means the profile is unarmed, this slice is not runnable, and slice 15 applies

### slice 14

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/eagle3_granite_verify_arming_/)' 2>&1 | grep -cE '^arming method=eagle3 cases = 3 unarmed_refused = 1 error_names_speculative_verify = true armed_refused = 0 control_wrong_error_rejected = true$'
```

expected: 1

### slice 15

```sh
grep -c '^speculative_verify = true' proxima-model-interop/src/profiles/granitemoe.toml; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_eagle3_granite_random_unarmed.txt 2>&1; grep -c '^refused method=eagle3 reason=speculative_verify ' lossless_eagle3_granite_random_unarmed.txt; grep -c '^lossless ' lossless_eagle3_granite_random_unarmed.txt; grep -c '^summary ' lossless_eagle3_granite_random_unarmed.txt; grep -c '^speculative_verify = true' proxima-model-interop/src/profiles/granitemoe.toml; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/granite-eagle3.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_eagle3_granite_trained_refused.txt 2>&1; grep -cE '^refused method=eagle3 reason=speculative_verify ' lossless_eagle3_granite_trained_refused.txt; grep -c '^lossless ' lossless_eagle3_granite_trained_refused.txt; grep -c '^summary ' lossless_eagle3_granite_trained_refused.txt
```

expected: eight lines: `0`, `1`, `0`, `0` (ACL3u) then `0`, `1`, `0`, `0` (ACL4u); the first line of each group is the profile precondition and must print 0, otherwise the arm is armed and slice 13 applies

ACL4s, run only if `speculative-draft-training` slice 12 strikes ACL4 for shape:

```sh
grep -c '^speculative_verify = true' proxima-model-interop/src/profiles/granitemoe.toml; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/granite-eagle3.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 > lossless_eagle3_granite_shape_refused.txt 2>&1; grep -c '^refused method=eagle3 reason=draft_shape ' lossless_eagle3_granite_shape_refused.txt; grep -c '^lossless ' lossless_eagle3_granite_shape_refused.txt; grep -c '^summary ' lossless_eagle3_granite_shape_refused.txt
```

expected: four lines: `1`, `1`, `0`, `0`

### slice 16

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/e2b-eagle3.gguf --tap-rows-from-prompt 1 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > tap_control_k1.txt 2>&1; grep -cE '^summary arm=(true|replaced) prompts=51 ' tap_control_k1.txt; grep -cE '^tap_control true_accepted_total=[0-9]+ replaced_accepted_total=[0-9]+ true_gt_replaced=true$' tap_control_k1.txt; cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-eagle3 --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/e2b-eagle3.gguf --tap-rows-from-prompt 0 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > tap_control_k0.txt 2>&1; grep -cE '^tap_control true_accepted_total=([0-9]+) replaced_accepted_total=\1 true_gt_replaced=false$' tap_control_k0.txt
```

expected: three lines: `2`, `1`, `1`; `true_gt_replaced=false` at k = 1 with AC2 passing refutes R2

## resume

Last landed slice: none
Next action: slice 1 -- the lossless harness and its three controls
Open question: whether gemma4's Padded engine can expose the 3 tap rows without a second lowering, answered by `decode-as-data` slice 3

## struck

- 2026-10-12: slices 0b, 1, 2, 2b, 2c, 11 and 13 of the earlier cut (llama.cpp oracle build, gemma 26B/31B pair download and conversion, llama fixture generator, vendored oracle ids, pairing fixture, llama seed parity, oracle acceptance comparison) -- the owner's direction removes llama.cpp token parity as a gate and removes the 26B/31B and other-family targets; the fixtures they produced existed only to feed those gates
