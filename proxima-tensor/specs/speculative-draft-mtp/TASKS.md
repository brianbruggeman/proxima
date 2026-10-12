# speculative-draft-mtp -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

Every validation command below is literal; the paths are the absolute paths defined in the acceptance criteria preamble of SPEC.md.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | prerequisites owned by sibling specs: the `build_forward` readout list (`decode-as-data` slice 3) and the lossless harness (`speculative-draft-eagle3` slice 1). This sub-spec consumes both and does not re-implement them | AC16 (first command), ACL5 (second command) | section `slice commands`, `### slice 0` | two lines: `1`, `3` | [ ] | fails until those slices land |
| 0b | build the oracle binaries from llama.cpp `f1ea20621` into `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621` (the cmake configure and build commands are the first block of `### slice 0b`); the checkout's prebuilt `build/bin` predates the source and registers no `--spec-type` | AC14 | section `slice commands`, `### slice 0b` | 1 | [ ] | |
| 1 | obtain a published GGUF of `google/gemma-4-E2B-it-assistant`, E2B-sized to pair with `$E2B`, by searching Hugging Face for `gemma-4-E2B-it-assistant` and downloading the GGUF file as published; no conversion step exists in this spec (no Python, and a Rust safetensors-to-GGUF export is a separate spec), so a GGUF that fails AC1 is a failed slice, not a reason to convert | AC1 | section `slice commands`, `### slice 1` | two lines: `1`, `0` (the AC1 control on `$E2B`) | [ ] | slice 2 runs AC2, the check that the downloaded GGUF is a working sidecar |
| 2 | verify llama.cpp drives `--spec-type draft-mtp --model-draft` end to end on the E2B pair (proves the sidecar oracle path before any proxima code), then build the patched oracle tool that dumps the sidecar run's internals: export `f1ea20621` into `llama-f1ea20621-dump/src`, apply the committed patch `proxima-model-interop/tests/fixtures/llama-mtp/mtp-dump.patch` (writes one JSON line per event to the file named by `LLAMA_MTP_DUMP`: `kind=draft` with `ids` and `positions`, `kind=h_nextn` with `pos` and the `n_embd`-float `h_row`, `kind=catchup` with `token`, `pos`, `h_row`), build `llama-speculative-simple` there, run it with `--spec-type draft-mtp --seed 1 -n 40`, and vendor the dump as `proxima-model-interop/tests/fixtures/llama-mtp/gemma-internals.jsonl` (the export, patch, build, run and vendor commands are the first block of `### slice 2`); llama.cpp stays an external oracle tool and is never a dependency of proxima | AC2 (first command), AC17 (second command) | section `slice commands`, `### slice 2` | five lines: `1`, then `>= 1` (draft records with equal positions), `0` (draft records with differing positions), `0` (catch-up records), `>= 50` (non-empty `h_nextn` rows, the `cases` floor slice 7 asserts); the two `>=` bounds are pre-registered, status plausible: the record counts depend on how many draft calls the dump run makes, which exists only after the run, and `>= 1` and `>= 50` are the non-vacuity floors | [ ] | needs slices 0b and 1; if the `h_nextn` count is below 50, raise the dump run's `-n` until it is (the dump run's `-n` is independent of AC11's `-n 40`); the jq was verified on synthetic files: a good file prints `1 0 0 50`, differing positions `1 1 0 50`, a catch-up record `1 0 1 50`, no `h_nextn` rows `1 0 0 0`, 49 rows `1 0 0 49`, empty `h_row` arrays `1 0 0 0`, a single-position draft `0 0 0 50`, no records `0 0 0 0` |
| 3 | design only: read `src/models/gemma4-assistant.cpp:84-200` line by line against `ModelDescriptor` and `ServingState` (`proxima-core/src/serving_state.rs:21`) and the draft-model loader `load_draft` (`proxima-model-interop/src/generate/draft_model.rs:174-183`) and write `proxima-tensor/specs/speculative-draft-mtp/SIDECAR_SHAPE.md` with three `###` sections: descriptor fields naming the target's embedding and KV sources, the draft `ServingState`'s read-only alias of the target's cache layers, and the readout that yields the next hidden row | AC15 | section `slice commands`, `### slice 3` | 3 | [ ] | reviewed before slice 6 starts |
| 4 | classify the sidecar form, the unsupported single-GGUF form and non-MTP GGUFs (R1); in `proxima-gguf` beside `proxima-gguf/src/parser.rs` (AC3 fixes `-p proxima-gguf`) | AC3 | section `slice commands`, `### slice 4` | 1 | [ ] | |
| 5 | sidecar descriptor: GGUF-key reads, tensor bind, width-compatibility typed errors, against slice 1's fixture (R2) | AC4 | section `slice commands`, `### slice 5` | 1 | [ ] | |
| 6 | land the sidecar sharing shape from slice 3: descriptor source fields and the read-only cache alias on the draft `ServingState` (R3) | AC5 | section `slice commands`, `### slice 6` | 1 | [ ] | the slice most likely to run long; split once slice 3's design is in hand |
| 7 | sidecar forward (target `tok_embd` read, shared-KV attention, pre/post projections) against slice 2's `h_nextn` rows, with the comparator controls inside the test (R3) | AC6 | section `slice commands`, `### slice 7` | 1 | [ ] | |
| 8 | propose and accept stages: top-1 per step, `p_min` stop, `n_max` stop, `n_min` clear, carried-row rollback (R4) | AC7 | section `slice commands`, `### slice 8` | two lines: `1`, `1` | [ ] | |
| 9 | `is_mem_shared` branches: catch-up-decode skip and same-position append, with the non-shared control (R5) | AC8, AC9 | section `slice commands`, `### slice 9` | two lines: `1`, `1` | [ ] | |
| 10 | route the `draft-mtp` config value in `ServingConfig`'s speculative section by data, with the two typed-error cases (R6) | AC10 | section `slice commands`, `### slice 10` | 1 | [ ] | |
| 11 | oracle comparator controls for ids, float rows and positions (R8) | AC13 | section `slice commands`, `### slice 11` | 1 | [ ] | run before slice 14 |
| 12 | lossless over a random sidecar-shaped draft on E2B, 51 prompts (RL1, RL3) | ACL1 | section `slice commands`, `### slice 12` | three lines: `51`, `51`, `1` | [ ] | a count below 51 on the second line is a verify or rollback bug, not the refutation condition (lossless holds for any draft) |
| 13 | lossless over the published assistant on E2B, `accepted` > 0 on every prompt (RL1, RL2) | ACL2 | section `slice commands`, `### slice 13` | three lines: `51`, `51`, `1` | [ ] | the third grep is a corpus-level `rejected_total` >= 1 (pre-registered, plausible): zero rollbacks over 51 prompts means this draft never exercised rollback; the random arm in slice 12 proves rollback independent of the weights |
| 14 | secondary llama oracle: vendor the ids for seeds 1, 2, 3 by running the oracle loop in the first block of `### slice 14` and recording ids, `n_drafted`, `n_accept` and the llama.cpp commit in `proxima-model-interop/tests/fixtures/llama-mtp/gemma.json`, then run the parity tests | AC12, AC11 | section `slice commands`, `### slice 14` | three lines: `3`, `3`, `3` | [ ] | |

## resume

Last landed slice: none
Next action: slices 0b and 1 are independent; slice 1 downloads the published GGUF of `google/gemma-4-E2B-it-assistant` found via web search 2026-09-28, not yet downloaded or byte-checked. Confirm with `strings -a /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf | grep -c nextn.pre_projection` before downloading anything larger than a HEAD-checkable manifest.
Open question: slice 3 (the sidecar sharing design) is the prerequisite for slices 6, 7 and 9 and should not start until slice 2's dump-reachability check has run, since the fixture method chosen there may constrain what the sharing shape must expose for testing. Slices 4, 5, 8, 10, 11 do not depend on slice 3.

## struck

- 2026-10-12: slices 1a, 2a, 3 (Qwen reachability), 3b, 4, 7, 8, 10, 11, 12, 13, 14, 19b, 21, 23 of the earlier cut -- the single-GGUF Qwen3.5/3.6 family (Qwen MTP-head GGUF acquisition, qwen35 and qwen35moe readouts, `h_prev` plumbing, propose/accept and catch-up against Qwen internals, Qwen end-to-end and llama seed parity); the owner removed the Qwen fixtures and the Qwen-only requirements (2026-10-12)

## slice commands

Each command is literal shell; run it from the repository root.

### slice 0

```sh
cargo nextest run -p proxima-tensor -E 'test(/readout_/)' 2>&1 | grep -c 'Summary .* 2 tests run: 2 passed'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/lossless_comparator_|lossless_guard_/)' 2>&1 | grep -cE '^control=(self compared=51 identical=51|flip compared=51 identical=50|zero_draft drafted_prompts=0 flagged=true)$'
```

### slice 0b

```sh
cmake -S /Users/brianbruggeman/repos/others/llama.cpp -B /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621 -DGGML_METAL=ON && cmake --build /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621 --target llama-speculative-simple llama-eval-callback
```

The check:

```sh
/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple --help 2>&1 | grep -c -- '--spec-type'
```

### slice 1

```sh
strings -a /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf | grep -c 'nextn.pre_projection'; strings -a /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd | grep -c 'nextn.pre_projection'
```

### slice 2

```sh
mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/src && git -C /Users/brianbruggeman/repos/others/llama.cpp archive f1ea20621 | tar -x -C /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/src && patch -p1 -d /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/src < proxima-model-interop/tests/fixtures/llama-mtp/mtp-dump.patch && cmake -S /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/src -B /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/build -DGGML_METAL=ON && cmake --build /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/build --target llama-speculative-simple && LLAMA_MTP_DUMP=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/gemma-internals.jsonl /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/build/bin/llama-speculative-simple --model /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --model-draft /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf --spec-type draft-mtp --seed 1 -n 40 -p "test" && cp /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621-dump/gemma-internals.jsonl proxima-model-interop/tests/fixtures/llama-mtp/gemma-internals.jsonl
```

The checks:

```sh
/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple --model /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --model-draft /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf --spec-type draft-mtp -p "test" -n 8 2>&1 | grep -cE 'n_drafted = [1-9]'; jq -s '([.[] | select(.kind == "draft" and (.positions | length) > 1 and (.positions | unique | length) == 1)] | length), ([.[] | select(.kind == "draft" and (.positions | unique | length) > 1)] | length), ([.[] | select(.kind == "catchup")] | length), ([.[] | select(.kind == "h_nextn" and ((.h_row | length) > 0))] | length)' proxima-model-interop/tests/fixtures/llama-mtp/gemma-internals.jsonl
```

### slice 3

```sh
grep -c '^### ' proxima-tensor/specs/speculative-draft-mtp/SIDECAR_SHAPE.md
```

### slice 4

```sh
cargo nextest run -p proxima-gguf --no-capture -E 'test(/mtp_form_classification/)' 2>&1 | grep -cE 'cases = 4 sidecar = 1 single_gguf_unsupported = 1 not_mtp = 2'
```

### slice 5

```sh
cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_sidecar_descriptor_binds/)' 2>&1 | grep -c 'Summary .* 3 tests run: 3 passed'
```

### slice 6

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/mtp_sidecar_alias_equals_copy/)' 2>&1 | grep -cE 'rows = ([89]|[1-9][0-9]+) alias_vs_copy_max_abs_diff = [0-9.e-]+ target_mutated_alias_differs = true'
```

### slice 7

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/mtp_sidecar_readout_matches_llama_fixture/)' 2>&1 | grep -cE 'cases = ([5-9][0-9]|[1-9][0-9]{2,}) max_abs_diff = [0-9.e-]+ control_self = 0\.0 control_perturbed_rejected = true'
```

### slice 8

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/mtp_draft_loop_stops/)' 2>&1 | grep -cE 'cases = 4 p_min_stops = 3 n_max_stops = 1 n_min_clears = 1'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/mtp_accept_selects_carried_row/)' 2>&1 | grep -cE 'cases = 5 accept_rows = 0,2,3,3,keep wrong_row_rejected = true'
```

### slice 9

```sh
cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/draft_mtp_is_mem_shared_positions/)' 2>&1 | grep -cE 'draft_positions = 5,5,5 catchup_decodes = 0 non_shared_positions = 5,6,7 non_shared_catchup_decodes = 1'; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/draft_mtp_positions_match_llama_fixture/)' 2>&1 | grep -cE 'records = [1-9][0-9]* control_self = true control_perturbed_rejected = true'
```

### slice 10

```sh
cargo nextest run -p proxima-model-interop --features std -E 'test(/speculative_config_draft_mtp_/)' 2>&1 | grep -c 'Summary .* 3 tests run: 3 passed'
```

### slice 11

```sh
cargo nextest run -p proxima-model-interop --features std -E 'test(/mtp_oracle_control_/)' 2>&1 | grep -c 'Summary .* 6 tests run: 6 passed'
```

### slice 12

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-mtp --draft-random 17 --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_mtp_e2b_random.txt 2>&1; grep -c '^lossless ' lossless_mtp_e2b_random.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* ' lossless_mtp_e2b_random.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_mtp_e2b_random.txt
```

### slice 13

```sh
cargo run --release -p proxima-model-interop --features std --example speculative_lossless_corpus -- --corpus proxima-model-interop/examples/data/speculative_corpus.jsonl --drafter draft-mtp --draft-model /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf --max-tokens 64 /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd > lossless_mtp_e2b_published.txt 2>&1; grep -c '^lossless ' lossless_mtp_e2b_published.txt; grep -cE '^lossless id=[^ ]+ identical=true n_drafted=[1-9][0-9]* accepted=[1-9][0-9]* ' lossless_mtp_e2b_published.txt; grep -cE '^summary prompts=51 identical=51 .* rejected_total=[1-9][0-9]* ' lossless_mtp_e2b_published.txt
```

### slice 14

```sh
for seed in 1 2 3; do /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/llama-f1ea20621/bin/llama-speculative-simple --model /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --model-draft /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fixtures/draft-mtp/gemma4-e2b-assistant.gguf --spec-type draft-mtp --seed "$seed" -n 40 > "llama_mtp_gemma_seed_$seed.txt" 2>&1; done
```

The checks:

```sh
jq '([.cases[] | select(.n_drafted > 0)] | length), (.cases | length)' proxima-model-interop/tests/fixtures/llama-mtp/gemma.json; cargo nextest run -p proxima-model-interop --features std --no-capture -E 'test(/mtp_llama_parity_gemma_seed_/)' 2>&1 | grep -cE 'ids_compared = ([89]|[1-9][0-9]+)$'
```
