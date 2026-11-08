# serve-flint-through-omega -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

| # | slice | depends on | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|---|
| 1 | Capture flint fixtures: header dump, 779-tensor directory, llama.cpp greedy ids (temp 0), tokenizer ids, and the oracle-control test that feeds those ids through the comparator. | none | AC1, AC12, AC21 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_fixture_header_) or test(flint_fixture_control_) or test(flint_oracle_control_)'` | 3 passed, 0 failed | [x] | fixtures from the real GGUF (sha256 63c2ecc4…) by llama.cpp b12110/cd959bae9: `gguf_kv_loader.txt` (52 kv, loader dump), `tensor_dir.txt` (779 rows), `llama_ids.json` (prompt 5 ids, 16 greedy ids), `tokenizer_ids.json` (22 ids), `layer_patterns.txt` (both 49-entry arrays, captured by `tools/gguf-kv-extract`, which shares no code with proxima-gguf; llama.cpp corroborates the length and leading entries). Tests in `proxima-model-interop/src/oracle_fixtures.rs`; AC1 1 passed, AC12 1 passed, AC21 1 passed |
| 2 | Make the sliding-pattern reader tolerate absent `key_length_swa`/`value_length_swa`/`rope.dimension_count`; assert literal 128 values from the flint header and gemma4's own literals, plus the control that a mutated header fails. | 1 | AC2, AC23 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(sliding_pattern_header_) or test(control_rejects_mutated_sliding_pattern_header)'` | 3 passed, 0 failed | [ ] | |
| 3 | Read `leading_dense_block_count=2`, `expert_shared_count=2`, `expert_shared_feed_forward_length=1024`, `expert_gating_func`, and sink presence generically, plus the control that a mutated header fails. | 1 | AC3, AC24 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_reader_leading_dense_and_shared_) or test(control_flint_reader_leading_dense_)'` | 2 passed, 0 failed | [ ] | |
| 4 | Consume `attention.rope_pattern` per layer; assert the flint descriptor equals the committed 49-entry array, plus the control that a mutated array fails. | 2, 3 | AC4, AC25 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_reader_rope_pattern_) or test(control_rejects_mutated_rope_pattern)'` | 2 passed, 0 failed | [ ] | |
| 5 | Lower `FfnCombination::RoutedWithSharedExpert` in the scheduled engine; replace the refusal at `attention_forward.rs:1301`; assert a hand-worked literal output and that the same comparison rejects a perturbed one. | 3 | AC5, AC18 | `cargo nextest run -p proxima-tensor --lib -E 'test(routed_with_shared_expert_literal_) or test(control_routed_with_shared_expert_)'` | 2 passed, 0 failed | [ ] | |
| 6 | Add the sink descriptor knob and program leaf; default off keeps node sets unchanged, plus the control that a planted node is rejected. | 1 | AC6, AC26 | `cargo nextest run -p proxima-tensor --lib -E 'test(attention_sink_descriptor_) or test(attention_sink_program_) or test(control_rejects_planted_sink_node)'` | 3 passed, 0 failed | [ ] | |
| 7 | Add the CPU evaluator for the sink term against one hand-worked literal, plus the control that the same comparison rejects a perturbed literal. | 6 | AC7, AC19 | `cargo nextest run -p proxima-tensor --lib -E 'test(attention_sink_cpu_literal_) or test(control_attention_sink_cpu_)'` | 2 passed, 0 failed | [ ] | |
| 8 | Emit and dispatch the sink term in omega MSL; assert the same hand-worked literal on Metal, its rejection of a perturbed literal, and unchanged gemma4 parity. | 6, 7 | AC8, AC20 | `cargo nextest run -p proxima-omega --lib -E 'test(attention_sink_metal_parity_literal_) or test(control_attention_sink_metal_parity_)'` | 2 passed, 0 failed | [ ] | |
| 9 | Bind the full flint tensor set (dense 0–1, routed+shared 2–48, sinks, Q/K norms, embd/output), plus the control that a directory one tensor short reports it. | 5, 6, 8 | AC9, AC27 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_bind_all_tensors_) or test(control_rejects_missing_tensor)'` | 2 passed, 0 failed | [ ] | |
| 10 | Add `proxima-model-interop/src/profiles/flint.toml` as data; assert no literal `flint` outside profiles/tests, plus the control that a planted literal is found. | 2, 3, 4, 5 | AC10, AC28 | `cargo nextest run -p proxima-model-interop --lib --features std -E 'test(flint_profile_is_data_) or test(control_finds_planted_literal)'` | 2 passed, 0 failed | [ ] | |
| 11 | Map `tokenizer.ggml.pre = "flint"` through the existing pre-type table; assert encode equals the captured ids, decode round-trips, and the captured ids pass the same comparison as their own control. | 1 | AC11, AC17 | `cargo nextest run -p proxima-tokenizer --lib --features gguf -E 'test(flint_pre_) or test(control_flint_tokenizer_)'` | 2 passed, 0 failed | [ ] | |
| 12 | End-to-end greedy parity: `LoadedModel::load` + generate on flint against the committed oracle, CPU and Metal arms. | 9, 10, 11 | AC13 | `cargo nextest run -p proxima-model-interop --lib --features std,metal -E 'test(flint_greedy_parity_)'` | 2 passed, 0 failed | [ ] | |
| 13 | RAGD: pin the proxima rev carrying slice 12, build the daemon, and run its provider-resolution tests. | 12 | AC14, AC22 | `bash -c 'CARGO_TARGET_DIR=/private/tmp/flint-ragd-target cargo build --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_daemon_memory >/dev/null && ls /private/tmp/flint-ragd-target/debug/ragd \| wc -l'`; `cargo nextest run --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_llm -E 'test(providers::proxima::tests)'` | `1`; `9 passed, 1 skipped, 0 failed` | [ ] | RAGD-80 |
| 14 | RAGD: daemon `provider = proxima` with the flint GGUF; run AC15 and AC16 with ollama stopped, plus the serving controls. | 13 | AC15, AC16, AC29, AC30 | `bash -c '! curl -sf http://127.0.0.1:11434/api/tags >/dev/null && curl -sS http://127.0.0.1:9091/v1/llm \| jq -r "if all(.[]; .config.provider==\"proxima\") then length else \"mixed\" end"'`; `bash -c '! curl -sf http://127.0.0.1:11434/api/tags >/dev/null && curl -sS -X POST http://127.0.0.1:9091/v1/chat/send -H "Content-Type: application/json" -d "{\"message\":\"Reply with the single word: flint.\"}" \| jq -r ".data.turn \| \"\(.parts\|length) \(.route.tag)\""'`; `cargo nextest run --manifest-path /Users/brianbruggeman/repos/slot-0/ragd-wt-final/Cargo.toml -p ragd_daemon_memory -E 'test(control_proxima_serving_)'` | `3`; `1 llm-only`; `2 passed, 0 failed` | [ ] | RAGD-81 |
## resume

Last landed slice: 1 (fixtures + oracle control, card PROXIMAOMEGA-4); spec admitted by spec-auditor 2026-11-07 (R 12/12, AC 30/0 orphaned, slices 14/0 unvalidated, paths 37/0 dangling)
Next action: slice 2 — make `sliding_pattern::from_metadata` tolerate absent `attention.key_length_swa` / `value_length_swa` / `rope.dimension_count`, then run AC2; note `from_metadata` still requires `attention.sliding_window_pattern` and `attention.head_count_kv`, both present in the captured header.
Open question: none

Workflow for a fresh session (absolute paths):

1. `/Users/brianbruggeman/.claude/agents/spec-auditor.md` — ADMIT/REFUSE before any code.
2. Per slice: `/Users/brianbruggeman/.agents/skills/validate-slice/SKILL.md` — run the AC, assert the count, fan out review on that diff.
3. Slice close: `/Users/brianbruggeman/.agents/skills/coherent-cards/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/coherent-commit/SKILL.md`, then
   `/Users/brianbruggeman/.agents/skills/postmortem/SKILL.md`.
4. Design and measurement: `/Users/brianbruggeman/.agents/skills/guiding-principles/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/algorithm-development/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/bench-metrics/SKILL.md`,
   `/Users/brianbruggeman/.agents/skills/instrument/SKILL.md`.
5. Domain agents: `/Users/brianbruggeman/.claude/agents/proxima-architect.md`,
   `proxima-critic.md`, `proxima-test-writer.md`, `card-worker-lean.md`, `postmortem.md`.

## struck

-
