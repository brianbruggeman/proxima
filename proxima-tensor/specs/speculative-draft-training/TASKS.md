# speculative-draft-training -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

Every validation command is literal and lives in the `## commands` section, one fenced block per slice. `FIXTURES` stands for `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures`, the E2B target is `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd` and the granite target is `/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisites owned by sibling specs: the `build_forward` readout list (`decode-as-data` slice 3), the lossless harness (`speculative-draft-eagle3` slice 1), and the existing granite census of `fp6-qat-ablation` | AC30 (prerequisite, sibling specs) | commands section, slice 0 | two lines: `1`, `3` | [ ] | fails until those slices land |
| 1 | the `draft-train` feature and the `draft_corpus` example: corpus split with the disjointness counts (R1, R15) | AC1 | commands section, slice 1 | 1 | [ ] | the feature is added with `cargo add` and a features-table edit shaped like `qat` (`proxima-model-interop/Cargo.toml:327`) |
| 1b | the overlap detector control (R1) | AC2 | commands section, slice 1b | 1 | [ ] | |
| 2 | `draft_selfdistill` example: target greedy generations plus hidden-state taps for E2B, run into a fresh `mktemp` directory writing `ids_e2b.jsonl`, `taps_index_e2b.jsonl` and the taps shards, then copied under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/` only when all three gates read 1 (R2) | AC3 | commands section, slice 2 | `1 1 1`, then `2` (the two placed index files) | [ ] | the jq gates are the AC3 gates, verified on synthetic jsonl and a stub run (see SPEC.md AC3); a stale file in the fixtures directory cannot satisfy the gate because the example writes to a new directory every run |
| 2b | label check against a re-decode, with its perturbed control (R2) | AC4 | commands section, slice 2b | 1 | [ ] | compares the target's greedy decode to itself; slices 2c and 2d add the independent oracle |
| 2c | prerequisite of slice 2d: build the llama.cpp oracle binaries `llama-completion` and `llama-tokenize` at `f1ea20621` into the build directory of `speculative-draft-mtp` slice 0b (R2) | AC27 (prerequisite of AC25) | commands section, slice 2c | two lines: `cmake_exit=0`, `2` | [ ] | the build output is kept in `oracle_build.log` and the exit status is printed (verified with a stub `cmake`, see SPEC.md AC27); needs `speculative-draft-mtp` slice 0b (the cmake configure); then `cmake --build <that build directory> --target llama-completion llama-tokenize`; Ollama stays off |
| 2d | `draft_label_oracle` example: llama.cpp greedy decode against the recorded labels on 16 recorded prompts, with the flipped-label control (R2) | AC25 | commands section, slice 2d | 1 | [ ] | N=16 pre-registered, status plausible; the grep verified on synthetic lines: `records=15`, a text mismatch, id mismatches, a prompt mismatch, an undetected control and empty output each print 0 |
| 2e | tap shards equal an independent forward of the target, with the perturbed and wrong-layer controls (R2) | AC23 | commands section, slice 2e | 1 | [ ] | needs slice 2's shards; the tolerance 1e-3 and N=16 are pre-registered, status plausible, not yet measured |
| 3 | draft vocabulary reduction and `d2t` (R4) | AC5 | commands section, slice 3 | 1 | [ ] | a coverage below 0.90 fails the test; record the measured value in this note |
| 4 | `draft_census` example over the three draft training programs, plus the refusal control (R5) | AC6, AC7 | commands section, slice 4 | two lines: `3`, `1` | [ ] | a nonzero refusal is an adjoint rule to add (`fp6-qat-ablation` R1a precedent), not a reason to cut the op |
| 4b | one program per method, target frozen, AdamW appended, with the unfrozen-target control (R3) | AC21 | commands section, slice 4b | 3 | [ ] | grep verified on synthetic lines: `programs=2`, `target_grads=3`, `adamw_nodes=0`, a control with 0 target gradients and a control not rejected each lower the count from 3 |
| 5 | single-sequence overfit on the CPU path, per method (R6) | AC8 | commands section, slice 5 | 3 | [ ] | |
| 6 | one training step on CPU and on Metal agree (R7) | AC9 | commands section, slice 6 | 1 | [ ] | |
| 6b | save at step 32, resume, and match the uninterrupted loss curve, with the dropped-optimizer-state control (R8) | AC24 | commands section, slice 6b | 3 | [ ] | tolerance 1e-5 pre-registered, status plausible; grep verified on synthetic lines: `within=false`, `dropped_optimizer_rejected=false`, `resumed_at=0` and a missing method each lower the count from 3 |
| 7 | EAGLE3 training run on E2B: 1024 steps, held-out line every 128 (R8, R9) | AC10 | commands section, slice 7 | 1 | [ ] | the jq gate is verified on synthetic logs: a log with 1000 train lines, a null last held-out loss, a NaN or -Infinity last held-out loss, a NaN, Infinity or -Infinity train loss, an increasing held-out curve, a final held-out loss above 6.0, one held-out line, a misspelt `kind` and an empty file each print 0 (rerun this session, raw output in the scratch directory); AC13 is computed from this log in slice 8b |
| 8 | DFlash and DSpark training runs on E2B (R8, R9) | AC11, AC12 | commands section, slice 8 | `1`, `1` | [ ] | same jq gate as slice 7 |
| 8b | the random-initialization baseline read from all three logs (R9) | AC13 | commands section, slice 8b | 1 | [ ] | verified on synthetic input: a value of 7.2, two values, four values, a non-numeric value and 11.0 each print 0 |
| 9 | arm granite's verify program: `speculative_verify = true` in `granitemoe.toml` and the one-row-per-position test (R10) | AC14 | commands section, slice 9 | two lines: `1`, `1` | [ ] | the first prints 0 today; if the MoE verify program cannot keep one row per position, this slice's outcome is the recorded typed refusal and slice 12 and the granite arms of the method specs are struck with that evidence |
| 10 | export to the draft GGUF and the train/serve round trip, per method (R13) | AC15 | commands section, slice 10 | `3`, `3` | [ ] | |
| 11 | place the three E2B drafts and measure acceptance against the random draft, with the corpus-level rollback gate on the trained draft (R11, R12) | AC31, AC16, AC17, AC18 | commands section, slice 11 | `3`; then `1`, `1`, `1`; then `1`, `1`, `1`; then `51`, `51`, `1` for each of the three methods | [ ] | AC16 below 0.5 or AC17 above 0.05 invokes the refutation condition; the `rejected_total` >= 1 clause (pre-registered, status plausible) is the corpus-level proof the rollback path ran on the trained draft, and a per-prompt `rejected` > 0 gate is not used; greps and awks verified on synthetic files (raw output in the scratch directory) |
| 12 | granite: self-distill, train and export the three granite drafts, when slice 9 armed the verify program (R10, R13) | AC26 | commands section, slice 12 | `1`, `1`, `1`, `3` | [ ] | struck to the recorded refusal if slice 9 recorded one; AC14 is discharged in slice 9 |
| 13 | determinism: three runs into new directories, the same seed twice and a seed-18 control (R14) | AC19 | commands section, slice 13 | two lines: `1`, `1` | [ ] | the control requires both curves to have 9 held-out losses, so a crashed seed-18 run prints 0; verified on synthetic logs (see SPEC.md AC19) |
| 14 | cost record: wall time and maximum resident set size of the slice 7 run, recorded as measurements (R16) | AC20 | commands section, slice 14 | three lines: `1`, `1`, `1` | [ ] | recorded, not gated: copy the three values into this note; no threshold and no agreement between the clocks is claimed; greps verified on synthetic files with each line removed in turn |
| 15 | structural checks: no `.py` or `.sh` under the new paths, the `draft-train` feature defined like `qat` with the five examples requiring it, and no new `pub` type in a library `src/` since `383a005dc`, each with its control (R15) | AC22, AC28, AC29 | commands section, slice 15 | seven lines: `0 rs_ok`, `1`, `1`, `5`, `1`, `0`, then >= 5 | [ ] | run after slices 1, 2, 2d, 4 and 7 have added the five examples; the controls and the synthetic verification are in SPEC.md AC22, AC28 and AC29 |

## commands

Each block is the literal text of the slice's validation command; run its lines in order from the repository root.

### slice 0

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -cE 'Summary .* 2 tests run: 2 passed, [0-9]+ skipped$'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self compared=51 identical=51|flip compared=51 identical=50|zero_draft drafted_prompts=0 flagged=true)$'
```

### slice 1

```sh
cargo run --release -p proxima-model-interop --features draft-train --example draft_corpus 2>&1 | grep -c '^records train_prose=1024 train_code=1024 heldout_prose=64 heldout_code=64 eval_overlap=0 train_heldout_overlap=0$'
```

### slice 1b

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_corpus_overlap_detector/)' 2>&1 | grep -c '^overlap_detector planted_detected = true clean_detected = false$'
```

### slice 2

```sh
mkdir -p "/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train"; export OUT=$(mktemp -d "/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/selfdistill.XXXXXX"); cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_selfdistill -- --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --out "$OUT" > "$OUT/selfdistill.log" 2>&1; ran=$(grep -c '^selfdistill records=2176 ' "$OUT/selfdistill.log"); ids=$(jq -s -c '{records:length, train:([.[]|select(.split=="train")]|length), heldout:([.[]|select(.split=="heldout")]|length), len_ok:([.[]|((.generated_ids|length)>=16 and (.generated_ids|length)<=64)]|all), empty_prompt:([.[]|select((.prompt_ids|length)==0)]|length)}' "$OUT/ids_e2b.jsonl" | grep -c '^{"records":2176,"train":2048,"heldout":128,"len_ok":true,"empty_prompt":0}$'); taps=$(jq -s -c '{records:length, present:([.[]|(.tap_rows>=1)]|all), at_most_64:([.[]|(.tap_rows<=64)]|all)}' "$OUT/taps_index_e2b.jsonl" | grep -c '^{"records":2176,"present":true,"at_most_64":true}$'); echo "$ran $ids $taps"; [ "$ran $ids $taps" = "1 1 1" ] && cp -R "$OUT"/. /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/; ls -1 /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/ids_e2b.jsonl /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/taps_index_e2b.jsonl | wc -l
```

### slice 2b

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_selfdistill_labels_are_target_greedy/)' 2>&1 | grep -c '^labels records_checked = 64 label_mismatches = 0 perturbed_label_detected = true$'
```

### slice 2c

```sh
cmake --build /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621 --target llama-completion llama-tokenize > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/oracle_build.log 2>&1; echo "cmake_exit=$?"; ls -1 /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-completion /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-tokenize | wc -l
```

### slice 2d

```sh
cargo run --release -p proxima-model-interop --features draft-train --example draft_label_oracle -- --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --ids /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/ids_e2b.jsonl --prose 8 --code 8 --llama-bin /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin 2>&1 | grep -c '^oracle records=16 text_mismatches=0 id_mismatches=0 prompt_id_mismatches=0 perturbed_label_detected=true$'
```

### slice 2e

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_taps_equal_independent_forward/)' 2>&1 | grep -cE '^taps records_checked = 16 rows_checked = [1-9][0-9]* layers = 3 max_abs_diff = [0-9]+(\.[0-9]+)?(e-[0-9]+)? perturbed_rejected = true wrong_layer_rejected = true$'
```

### slice 3

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_vocab_reduction/)' 2>&1 | grep -cE '^draft_vocab_reduction draft_vocab = 32768 d2t_len = 32768 d2t_roundtrip = true logits_cols = 32768 masked_label_loss = 0 masked_label_grad = 0 masked_control_nonzero = true coverage = 0\.9[0-9]+$'
```

### slice 4

```sh
cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_census 2>&1 | grep -cE '^census method=(eagle3|dflash|dspark) refusals=0 nodes=[1-9][0-9]*$'; cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_census_reports_a_refusal/)' 2>&1 | grep -c '^census_control refusals = 1$'
```

### slice 4b

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_step_program_is_one_frozen_target_adamw_/)' 2>&1 | grep -cE '^step_program method=(eagle3|dflash|dspark) programs=1 target_grads=0 adamw_nodes=[1-9][0-9]* unfrozen_control_target_grads=[1-9][0-9]* unfrozen_control_rejected=true$'
```

### slice 5

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_overfits_one_sequence_/)' 2>&1 | grep -cE '^overfit method=(eagle3|dflash|dspark) first_loss=[0-9]+\.[0-9]+ last_loss=0\.0[0-9]+$'
```

### slice 6

```sh
cargo nextest run -p proxima-model-interop --features draft-train,metal --no-capture -E 'test(/draft_step_cpu_equals_metal/)' 2>&1 | grep -cE '^step_equivalence cpu_loss = [1-9][0-9]*\.[0-9]+ metal_loss = [1-9][0-9]*\.[0-9]+ abs_diff = 0\.000[0-9]* perturbed_rejected = true$'
```

### slice 6b

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_resume_equals_uninterrupted_/)' 2>&1 | grep -cE '^resume method=(eagle3|dflash|dspark) steps=64 resumed_at=32 max_abs_diff=[0-9]+(\.[0-9]+)?(e-[0-9]+)? within=true dropped_optimizer_max_abs_diff=[0-9]+(\.[0-9]+)?(e-[0-9]+)? dropped_optimizer_rejected=true$'
```

### slice 7

```sh
cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method eagle3 --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --steps 1024 --seed 17 --out /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-eagle3 > train_e2b_eagle3.jsonl 2> train_e2b_eagle3.err; jq -s -c 'def fin: type=="number" and (isnan|not) and (isinfinite|not); [.[]|select(.kind=="train")] as $t | [.[]|select(.kind=="heldout")] as $h | {train_n:($t|length), heldout_n:($h|length), all_finite:([.[]|select(.kind=="train" or .kind=="heldout")|.loss|fin]|all), heldout_decreased:(($h|length)>1 and ($h[-1].loss < $h[0].loss)), below_cap:(($h|length)>0 and ($h[-1].loss <= 6.0))}' train_e2b_eagle3.jsonl | grep -c '^{"train_n":1024,"heldout_n":9,"all_finite":true,"heldout_decreased":true,"below_cap":true}$'
```

### slice 8

```sh
cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method dflash --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --steps 1024 --seed 17 --out /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-dflash > train_e2b_dflash.jsonl 2> train_e2b_dflash.err; jq -s -c 'def fin: type=="number" and (isnan|not) and (isinfinite|not); [.[]|select(.kind=="train")] as $t | [.[]|select(.kind=="heldout")] as $h | {train_n:($t|length), heldout_n:($h|length), all_finite:([.[]|select(.kind=="train" or .kind=="heldout")|.loss|fin]|all), heldout_decreased:(($h|length)>1 and ($h[-1].loss < $h[0].loss)), below_cap:(($h|length)>0 and ($h[-1].loss <= 6.0))}' train_e2b_dflash.jsonl | grep -c '^{"train_n":1024,"heldout_n":9,"all_finite":true,"heldout_decreased":true,"below_cap":true}$'; cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method dspark --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --steps 1024 --seed 17 --out /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-dspark > train_e2b_dspark.jsonl 2> train_e2b_dspark.err; jq -s -c 'def fin: type=="number" and (isnan|not) and (isinfinite|not); [.[]|select(.kind=="train")] as $t | [.[]|select(.kind=="heldout")] as $h | {train_n:($t|length), heldout_n:($h|length), all_finite:([.[]|select(.kind=="train" or .kind=="heldout")|.loss|fin]|all), heldout_decreased:(($h|length)>1 and ($h[-1].loss < $h[0].loss)), below_cap:(($h|length)>0 and ($h[-1].loss <= 6.0))}' train_e2b_dspark.jsonl | grep -c '^{"train_n":1024,"heldout_n":9,"all_finite":true,"heldout_decreased":true,"below_cap":true}$'
```

### slice 8b

```sh
for f in train_e2b_eagle3 train_e2b_dflash train_e2b_dspark; do jq -s '[.[]|select(.kind=="heldout")][0].loss' $f.jsonl; done | awk '{n++; if ($1>=9.897 && $1<=10.897) good++} END{print (n==3 && good==3)?1:0}'
```

### slice 9

```sh
grep -c '^speculative_verify = true$' proxima-model-interop/src/profiles/granitemoe.toml; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/granitemoe_verify_program_keeps_one_logits_row_per_position/)' 2>&1 | grep -c '^granitemoe_verify rows_per_position = 1 positions = 8$'
```

### slice 10

```sh
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_export_roundtrip_/)' 2>&1 | grep -cE '^export method=(eagle3|dflash|dspark) max_abs_diff = [0-9]+(\.[0-9]+)?(e-[0-9]+)? perturbed_rejected = true$'
cargo nextest run -p proxima-model-interop --features draft-train --no-capture -E 'test(/draft_export_hand_vector_/)' 2>&1 | grep -cE '^export_hand_vector method=(eagle3 max_abs_diff = [0-9]+(\.[0-9]+)?(e-[0-9]+)? expected = -inf,0\.848528,-inf,1\.131371,1\.979899|dflash max_abs_diff = [0-9]+(\.[0-9]+)?(e-[0-9]+)? expected = 1\.568929,0\.588348|dspark max_abs_diff = [0-9]+(\.[0-9]+)?(e-[0-9]+)? expected = 3,7,11) perturbed_rejected = true$'
```

### slice 11

```sh
for method in eagle3 dflash dspark; do mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method && cp /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-$method/draft.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method/e2b-$method.gguf; done; ls -1 /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/e2b-eagle3.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/e2b-dflash.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/e2b-dspark.gguf | wc -l
for method in eagle3 dflash dspark; do cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-$method --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method/e2b-$method.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > acc_trained_$method.txt 2>&1; grep -E '^summary prompts=51 ' acc_trained_$method.txt | grep -oE 'mean_accepted_per_step=[0-9]+\.[0-9]+' | cut -d= -f2 | awk '{ok=($1>=0.5)} END{print (NR==1 && ok)?1:0}'; done
for method in eagle3 dflash dspark; do cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-$method --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > acc_random_$method.txt 2>&1; grep -E '^summary prompts=51 ' acc_random_$method.txt | grep -oE 'mean_accepted_per_step=[0-9]+\.[0-9]+' | cut -d= -f2 | awk '{ok=($1<=0.05)} END{print (NR==1 && ok)?1:0}'; done
for method in eagle3 dflash dspark; do cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-$method --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method/e2b-$method.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_$method.txt 2>&1; grep -c '^lossless ' lossless_$method.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_$method.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_$method.txt; done
```

### slice 12

```sh
for method in eagle3 dflash dspark; do cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method $method --target /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 --steps 1024 --seed 17 --out /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/granite-$method > train_granite_$method.jsonl 2> train_granite_$method.err; jq -s -c 'def fin: type=="number" and (isnan|not) and (isinfinite|not); [.[]|select(.kind=="train")] as $t | [.[]|select(.kind=="heldout")] as $h | {train_n:($t|length), heldout_n:($h|length), all_finite:([.[]|select(.kind=="train" or .kind=="heldout")|.loss|fin]|all), heldout_decreased:(($h|length)>1 and ($h[-1].loss < $h[0].loss)), below_cap:(($h|length)>0 and ($h[-1].loss <= 6.0))}' train_granite_$method.jsonl | grep -c '^{"train_n":1024,"heldout_n":9,"all_finite":true,"heldout_decreased":true,"below_cap":true}$'; mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method && cp /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/granite-$method/draft.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-$method/granite-$method.gguf; done; ls -1 /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-eagle3/granite-eagle3.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dflash/granite-dflash.gguf /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-dspark/granite-dspark.gguf | wc -l
```

### slice 13

```sh
for run in a:17 b:17 c:18; do cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method eagle3 --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --steps 1024 --seed ${run##*:} --out "/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-eagle3-det-${run%%:*}" > det_${run%%:*}.jsonl 2> det_${run%%:*}.err; done; for pair in b:true c:false; do jq -n -c --slurpfile a det_a.jsonl --slurpfile b det_${pair%%:*}.jsonl 'def fin: type=="number" and (isnan|not) and (isinfinite|not); ($a|map(select(.kind=="heldout")|.loss)) as $x | ($b|map(select(.kind=="heldout")|.loss)) as $y | {n:($x|length), m:($y|length), finite:(($x|length)>0 and ([$x[],$y[]]|map(fin)|all)), within:(($x|length)>0 and (($x|length)==($y|length)) and ([range(0;$x|length)|(($x[.]-$y[.])|fabs)]|max) <= 0.001)}' | grep -c "^{\"n\":9,\"m\":9,\"finite\":true,\"within\":${pair##*:}}\$"; done
```

### slice 14

```sh
/usr/bin/time -l cargo run --release -p proxima-model-interop --features draft-train,metal --example draft_train -- --method eagle3 --target /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --steps 1024 --seed 17 --out /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-train/e2b-eagle3-timed > time_e2b_eagle3.txt 2>&1; grep -cE 'wall_seconds=[1-9][0-9]*\.[0-9]+ tokens_per_second=[1-9][0-9]*\.[0-9]+' time_e2b_eagle3.txt; grep -cE '^ +[1-9][0-9]*\.[0-9]+ +real ' time_e2b_eagle3.txt; grep -cE '^ +[1-9][0-9]* +maximum resident set size$' time_e2b_eagle3.txt
```

### slice 15

```sh
git ls-files -co --exclude-standard -- 'proxima-model-interop/examples/draft_*' 'proxima-tensor/specs/speculative-draft-training' | awk '/\.(py|sh)$/{scripts++} /\.rs$/{rs++} END{print (scripts+0) " " ((rs>=5)?"rs_ok":"rs_missing")}'; grep -cE '^draft-train = \["std", "proxima-autograd"\]$' proxima-model-interop/Cargo.toml; grep -cE '^qat = \["std", "metal", "proxima-autograd"\]$' proxima-model-interop/Cargo.toml; grep -cE '^required-features = \[.*"draft-train".*\]$' proxima-model-interop/Cargo.toml; grep -cE '^required-features = \[.*"qat".*\]$' proxima-model-interop/Cargo.toml; git diff 383a005dc..HEAD -- proxima-model-interop/src proxima-autograd/src omega/src proxima-gguf/src | grep -cE '^\+[[:space:]]*pub(\([^)]*\))? (unsafe )?(struct|enum|trait|union|type) '; git diff --name-only 383a005dc..HEAD -- 'proxima-model-interop/examples/draft_*' | grep -c '\.rs$'
```

## resume

Last landed slice: none
Next action: slice 1 (the feature and the corpus split) needs nothing from a sibling spec; slice 0 gates slice 2 onward; slice 2c needs `speculative-draft-mtp` slice 0b
Open question: whether the gemma4 Padded engine's tap rows cover every prompt and generated position at the three layer ids chosen for EAGLE3, answered by `decode-as-data` slice 3

## struck

-
