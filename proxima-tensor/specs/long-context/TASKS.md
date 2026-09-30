# long-context -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

Cargo and Metal slices wait until slot-0-3f reports its timing window closed.

Every command runs from `/Users/brianbruggeman/repos/slot-0/proxima` after
`source proxima-tensor/specs/long-context/env.sh`. That file sets `CARGO_TARGET_DIR`,
`$GEMMA4`, `$QWEN36`, `$QWEN3_8B` (absolute blob paths) and `$LONG_CTX_SPEC`, and defines
`niah`.

Slices 0a and 0c measure HEAD `df3766dd`, not the working tree, because slices 1-5 are
already written uncommitted. Slice 00 creates the export without a worktree. Slice 0d
removes it.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 00 | export HEAD | AC12, AC23 setup | `rm -rf $HOME/repos/slot-0/.long_ctx_backups/tree_head && mkdir -p $HOME/repos/slot-0/.long_ctx_backups/tree_head && git archive df3766dd \| tar -x -C $HOME/repos/slot-0/.long_ctx_backups/tree_head && test -f $HOME/repos/slot-0/.long_ctx_backups/tree_head/Cargo.toml && echo exported` | `exported` | [x] | ran 2026-09-29: `exported`, 53M in $HOME/repos/slot-0/.long_ctx_backups/tree_head |
| 0a | passing-test baseline at HEAD | AC23 | `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/passed_tests.sh --self-test \| wc -l` then `(cd $HOME/repos/slot-0/.long_ctx_backups/tree_head && CARGO_TARGET_DIR=/private/tmp/cargo_target_long_ctx_head /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/passed_tests.sh) > /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/before.txt && wc -l < /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/before.txt` | `2`, then a count > 0, written here | [x] | ran 2026-09-29 in the df3766dd export with std+metal: `1384 tests run: 1384 passed (6 slow), 81 skipped`, exit 0; before.txt = 1384 lines |
| 0b | load qwen3-8b on the dense path | AC21 precondition | `cargo run -p proxima-model-interop --example bench_local --features std -- "$QWEN3_8B" cpu "The capital of France is" 8` | 8 generated tokens printed, no error | [ ] | |
| 0c1 | write `examples/gemma4_ring_parity.rs` in full-cache mode only, using HEAD's public API (`--record FILE` writes 256 greedy token ids; `--expect FILE` compares and prints `full vs head: N/256`; `--ollama-facts` sends the four `gemma4_correctness_gate.rs` prompts raw to Ollama `/api/generate` at temperature 0 and prints `ollama answered_correctly=<bool>` per check) | AC12 oracle | `cargo check -p proxima-model-interop --example gemma4_ring_parity --features std,metal 2>&1 \| grep -c '^error'` | `0` | [ ] | |
| 0c | ~~pin HEAD gemma4 tokens~~ -> pin BASE tokens at ddd0a542 | AC12 oracle | see amendment below | `256` | [x] | 2026-09-29: at df3766dd the run failed with `arena peak_bytes=723518320 exceeds arena_transient_cap=172812125 at query_rows=1`. That is the prefill cap defect fixed by 4ce5103d and 54ba74fc on feat/speculative-decode. Re-run on a git-archive export of that branch tip ddd0a542: `recorded 256`, wc 256. Saved as `gemma4_base_tokens.txt` |
| 0c-old | pin HEAD gemma4 tokens | AC12 oracle | `cp proxima-model-interop/examples/gemma4_ring_parity.rs $HOME/repos/slot-0/.long_ctx_backups/tree_head/proxima-model-interop/examples/ && (cd $HOME/repos/slot-0/.long_ctx_backups/tree_head && CARGO_TARGET_DIR=/private/tmp/cargo_target_long_ctx_head cargo run -p proxima-model-interop --example gemma4_ring_parity --features std,metal -- "$GEMMA4" --record /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/gemma4_head_tokens.txt) && wc -w < /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/gemma4_head_tokens.txt` | `256` | [ ] | |
| 0d | remove the export and its target | AC12, AC23 (teardown of their setup) | `rm -rf $HOME/repos/slot-0/.long_ctx_backups/tree_head /private/tmp/cargo_target_long_ctx_head && ls /private/tmp \| grep -c long_ctx_head` | `0` | [ ] | |
| 1 | per-layer KV row bytes in `MemoryBudget::derive`; gemma4 charges own layers only | AC5a | `cargo nextest run -p proxima-model-interop --features std memory_budget_gemma4_own_layers` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code wired: `MemoryBudget::derive` and `fit_context_length` now take the per-layer slice only (the uniform forms are deleted, so an old caller cannot compile); `apply_memory_fit_gate` (`decode.rs`) fits and budgets `LoadedModel::kv_layers`, which drops each sliding window until slice 6 lands the ring (`stored_kv_layers`, `load_model.rs`) |
| 1a | gemma4 sliding-window cap in the budget | AC5b | `cargo nextest run -p proxima-model-interop --features std memory_budget_gemma4_window_cap` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code written |
| 1b | the same for qwen35moe | AC6 | `cargo nextest run -p proxima-model-interop --features std memory_budget_qwen35moe` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code written |
| 2 | default config admissible, plus a rejection control | AC7 | `cargo nextest run -p proxima-model-interop --features std serving_default_admission` | 2 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "2 passed"; code written |
| 3 | `trained_context_length` from GGUF | AC1 | `cargo nextest run -p proxima-model-interop --features std trained_context_read` | 3 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "3 passed"; code written |
| 3b | `RopeScaling` parse from GGUF | AC8 | `cargo nextest run -p proxima-model-interop --features std rope_scaling_from_gguf` | 4 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "4 passed"; code written |
| 3c | `RopeScaling` per-call override | AC9 | `cargo nextest run -p proxima-model-interop --features std rope_scaling_override` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code written |
| 4 | `context_length: Option<u32>` resolves to L | AC2 | `cargo nextest run -p proxima-model-interop --features std context_default_resolves` | 2 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "2 passed"; code wired: `ServingConfig.context_length` is `Option<u32>` (default `None`); every reader goes through `LoadedModel::serving_context_length`, and `run_decode_loop_observed_seeded` and `forward_node_values_on_backend` write the resolved value back before use. A checkpoint with no `{arch}.context_length` resolves `None` to `u32::MAX`, leaving the memory fit as the only bound |
| 4b | fit clamp after L | AC3 | `cargo nextest run -p proxima-model-interop --features std context_fit_clamps` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code written |
| 4c | `ContextExceedsTrained` and `ContextLength::Extrapolate` | AC4 | `cargo nextest run -p proxima-model-interop --features std context_over_limit` | 3 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "3 passed"; code written |
| 5 | YaRN inverse frequencies | AC10a | `cargo nextest run -p proxima-model-interop --features std yarn_inv_freq` | 2 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "2 passed"; code written; paper walk done 2026-09-29 |
| 5b | YaRN attention factor | AC10b | `cargo nextest run -p proxima-model-interop --features std yarn_attention_factor` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code written |
| 5c | scaled cos/sin in `build_position_inputs` | AC10c | `cargo nextest run -p proxima-model-interop --features std yarn_scaled_rope_table` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed"; code wired: `build_position_inputs` takes a required `RopeScaling` (the unscaled overload is deleted); its three production callers in `decode.rs` pass `LoadedModel::effective_rope_scaling`. The test tolerance is `\|theta\| x 2^-22 x af + 1e-6` |
| 6 | gemma4 ring allocation | AC11 | `cargo nextest run -p proxima-model-interop --features std gemma4_ring_rows` | 1 passed | [x] | ran 2026-09-29 (`--features std`): nextest printed "1 passed" for `generate::kv_ring::tests::gemma4_ring_rows`. From the E2B header (`test_support::gemma4_e2b_header`), the 15 own-KV layers hold 12 x 512 (the header's `attention.sliding_window`, read not assumed) and 3 x 2048 rows after 2048 positions are appended through `attention_cache`, the constructor the decode loop uses |
| 6b | ring write/read mapping; the parity example gains the ring arm and `--ring-offset` | AC12, AC13 | `cargo run -p proxima-model-interop --example gemma4_ring_parity --features std,metal -- "$GEMMA4" --expect /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/gemma4_head_tokens.txt`, then again with `--ring-offset 1` | `full vs head: 256/256 ring vs head: 256/256`; then `full vs head: 256/256` and `ring vs head:` below 256 | [x] | ran 2026-09-29, GPU. `OMEGA_ARENA_TRANSIENT_CAP=1000000000` was set at build time (see amendment): ring on, `full vs head: 256/256` and `ring vs head: 256/256`; with `--ring-offset 1`, `full vs head: 256/256` and `ring vs head: 15/256`. Without that build knob the first arm stops with `arena peak_bytes=723518320 exceeds arena_transient_cap=172812125 at query_rows=1 (device_limit=51539607552)`, the df3766dd cap defect, before any ring code runs |
| 6c | ring against external oracles (runs after 7f, once the harness exists) | AC12x | the three commands of SPEC AC12x (a), (b), (c) | `4`; `4`; Y >= 9 and X >= Y | [ ] | |
| 7 | niah haystack: pinned source | AC13b | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_haystack_sha` | 1 passed | [x] | 2026-09-29: 1 passed |
| 7a | niah haystack: trim | AC13c | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_haystack_trim` | 1 passed | [x] | 2026-09-29: 1 passed (real gemma4 tokenizer, 3.2 s) |
| 7b | niah needle depths | AC13d | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_needle_depths` | 1 passed | [x] | 2026-09-29: 1 passed |
| 7c | Ollama request: same prompt | AC13e | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_ollama_request_prompt` | 1 passed | [x] | 2026-09-29: 1 passed |
| 7d | Ollama request: options | AC13f | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_ollama_request_options` | 1 passed | [x] | 2026-09-29: 1 passed |
| 7e | shared scoring | AC13g | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_scoring_shared` | 3 passed | [x] | 2026-09-29: 3 passed |
| 7f | end-to-end controls | AC14, AC15 | `niah --model "$GEMMA4" --ctx 8192 --needles 10 --control`, then without `--control` | `0/10` both arms; then Y >= 9, X >= Y | [ ] | |
| 8 | F16 KV storage plus kernel half load | AC16 | `niah --model "$GEMMA4" --ctx 32768 --needles 10 --kv f32,f16` | B == A, B >= Y, Y > 0 | [ ] | |
| 8b | default KV F16; admission control re-run | AC17, AC7 | `cargo nextest run -p proxima-model-interop --features std default_kv_is_f16 serving_default_admission` | 3 passed | [ ] | |
| 9 | Q8_0 quantize-on-write plus in-kernel dequant; admission control re-run | AC18, AC7 | `niah --model "$GEMMA4" --ctx 32768 --needles 10 --kv f16,q8_0` then `cargo nextest run -p proxima-model-interop --features std serving_default_admission` | C >= Y, Y > 0; then 2 passed | [ ] | |
| 10 | gemma4 @ 131072 | AC19 | `niah --model "$GEMMA4" --ctx 131072 --needles 10 --kv f16` | `kv_bytes=811597824`, P > 811597824, X >= Y, X > 0 | [ ] | |
| 11 | qwen3.6 @ 262144 | AC20 | `niah --model "$QWEN36" --ctx 262144 --needles 10 --kv f16` | `kv_bytes=5368709120`, X >= Y, X > 0 | [ ] | |
| 12 | qwen3-8b yarn vs none | AC21 | `niah --model "$QWEN3_8B" --ctx 131072 --needles 10 --kv q8_0 --rope-scaling none,yarn:4:32768 --allow-extrapolation` | `kv_bytes=10267656192`, yarn > none, yarn >= Y, yarn > 0 | [ ] | |
| 13 | decode-cost bench, interleaved arms | AC22 | `cargo run -p proxima-model-interop --release --example kv_dtype_decode_bench --features std,metal -- --model "$GEMMA4" --ctx 65536 --iterations 5` | 4 arms; proxima q8/f16 <= ollama q8/f16 | [ ] | |
| 14 | regression; admission control re-run | AC23, AC7 | `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/passed_tests.sh > $HOME/repos/slot-0/.long_ctx_backups/tree_after.txt && comm -23 /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/before.txt $HOME/repos/slot-0/.long_ctx_backups/tree_after.txt \| wc -l` then `cargo nextest run -p proxima-model-interop --features std serving_default_admission` | `0`; then 2 passed | [ ] | |

## resume

Last landed slice: none. Written 2026-09-29: the spec, the YaRN worked example, `env.sh`
and `passed_tests.sh` (its self-test prints 2 lines for a 2-PASS sample). Audits 1-5
refused; revised five times. AC12's HEAD oracle is kept against audit 5, per
`AGENTS.md:35`.
Slices 1-5 code is being written uncommitted, with no cargo, by a dispatched agent.
Spec ADMITTED on audit pass 8 (2026-09-29).
Next action: When slot-0-3f reports "done", run 00, 0a, 0c1, 0c, 0d, then 0b.
Shared checkout (confirmed with slot-0-ea 2026-09-29):
- Every modified omega file is slot-0-ea's unlanded 7-commit tiled-GEMM series, waiting on
  owner authorization. That includes `cached_attention_render.rs:205,542,604`, which are
  one-argument `preamble(..., bool)` call-site changes.
- Slices 8 and 9 edit on top of that working-tree version; the hunks are disjoint.
- `serving.rs:355-363` is slot-0-ea's doc hunk (`exact_activations` default true). It is
  never staged in this spec's commits; stage with `git add -p`.
- slot-0-ea also owns `generate/decode.rs`, `qwen35moe/execution.rs`,
  `tests/real_gemma4_registry_probe.rs` and
  `tests/gemma4_tiled_gemm_defaults_full_logit_vector_diff.rs`.

Amendments after the slices 1-5 agent report (2026-09-29):
- Every nextest command gained `--features std`. The crate default is `[]`, and most of
  these tests are std-gated; without the flag they run 0 tests.
- `passed_tests.sh` must run with the same features, or the baseline misses the std tests.
- `MemoryBudget` takes `&[(kv_heads, head_dim, Option<window>)]`, not `&[u64]`, because
  a flat byte slice cannot carry the R5b cap.
- `trained_context_length` takes `parsed`, because the trait is stateless.
- `RopeScaling` is plain `Copy` data without serde, because `serving.rs:17-25` bars serde
  from the crate after a measured 31% forward regression.
- The linear limit is `trained x factor`.
- AC7's default also had to change `gpu_layers` (`serving.rs:729`, non-metal builds) and
  `reasoning_budget` (`:764`).
- Wiring written (still unvalidated, no cargo): the `Option<u32>` context flip, the
  per-layer fit and the scaled RoPE call sites in `generate/decode.rs`, `load_model.rs`,
  `pregather.rs`, `residency_caches.rs`, `serving.rs` and `memory_fit.rs`. `LoadedModel`
  gained `trained_context_length`, `rope_scaling` and (metal) `kv_layers`, filled at all
  three load paths. slot-0-ea's lint hunk in `decode.rs` (old range 3887-3939) is
  byte-identical.
- `passed_tests.sh` passes `--features proxima-model-interop/std` (plus
  `proxima-model-interop/metal` on Darwin) and exits 1 on zero passing tests; omega and
  proxima-tensor already default `std`.

Validation pass 2026-09-29 (slices 1-5):
- Building `proxima-model-interop --features std` failed at HEAD `df3766dd` with 58 errors
  (`cannot find macro trace/debug/warn`). `generate/mod.rs` imported
  `proxima_telemetry::{debug, trace, warn}` only under `metal` or `instrument`, while
  `decode.rs` and `pregather.rs` call them ungated. Fixed: the import is gated on `std`, and
  the `std` feature enables `dep:proxima-telemetry`, `proxima-telemetry/std` and
  `proxima-telemetry/emit` (the set `metal` already used). The 13 ACs ran with this fix.
- `test_support::parsed_header` moved above `mod tests` (clippy `items_after_test_module`).
- Full nextest, `--features std --no-fail-fast`: 248 run, 248 passed, 32 skipped.
- Doc tests: 0 exist in the crate (only `text` fences at `hf_bind.rs:236`, `bind.rs:6693`).
- Workspace clippy (`warnings = deny`, `--all-targets --no-deps`) exits 0 for `std` and
  `std,metal`. `-W clippy::pedantic` is not in the workspace lints; it reports 509 errors
  crate-wide, 21 of them in the new `rope_scaling.rs` (cast precision, doc backticks, float `==`).

Git state (2026-09-29): slot-0-ea detaches the root checkout at df3766dd so slot-0-3f can
fast-forward main with feat/speculative-decode. This spec's commits then need a rebase or
cherry-pick onto the new main. Check the overlap of the speculative-decode diffstat with
decode.rs, residency_caches.rs, serving.rs and memory_fit.rs first.
Preview from the `feat/speculative-decode` tip (read 2026-09-29): it can fast-forward from
df3766dd, and overall touches 52 files, +30637/-478.
- **Overlapping files:** decode.rs +453, residency_caches.rs +101, serving.rs +336;
  memory_fit.rs is untouched.
- **Additive overlaps:** in serving.rs, new `SpeculativeConfig`/`SpeculativeType*` types,
  a `speculative` field on ServingConfig and a default line. In residency_caches.rs, a new
  `SpeculativeDecodeStats`.
- **The one semantic collision:** decode.rs has
  `DrafterSet::build(&serving_config.speculative, serving_config.context_length as usize)`.
  After the rebase it must use the resolved context (`serving_context_length`), because
  `context_length` becomes `Option<u32>`. Otherwise it fails to compile, which is the
  intended behaviour of the flip.

Incident 2026-09-29: an agent ran `cargo fmt -p proxima-model-interop`.
- slot-0-ea checked against its series tree b6848f27. `qwen35moe/execution.rs` is
  identical, and its decode.rs let-chains are verbatim.
- Two test files differed in formatting only: `tests/gemma4_tiled_gemm_defaults_full_logit_vector_diff.rs`
  and `tests/real_gemma4_registry_probe.rs`. slot-0-ea restores them itself; this spec
  never touches them.
- Briefs for this checkout forbid crate-wide fmt (memory:
  `feedback_no_crate_fmt_in_shared_checkout`).

Amendment 2026-09-29, base commit moved from df3766dd to ddd0a542.
- **Why:** df3766dd cannot run the 2,048-token gemma4 case because of the arena prefill cap.
  feat/speculative-decode (tip ddd0a542, landing on main, owner-authorized) fixes it, and
  this spec rebases onto it anyway.
- **AC12/AC13 oracle file:** `gemma4_base_tokens.txt`, recorded at ddd0a542. It replaces
  `gemma4_head_tokens.txt`, and the example's `--expect` label stays `full vs head`.
- **AC23 baseline:** `before.txt` is re-recorded at ddd0a542. The df3766dd list is kept as
  `before_df3766dd.txt` (1384 lines).
- **Cleanup:** the df3766dd export was removed (0d done: `ls /private/tmp | grep -c
  long_ctx_head` = 0). The ddd0a542 export stays at `$HOME/repos/slot-0/.long_ctx_backups/tree_sd` until its
  baseline is recorded.
- **AC23 baseline re-recorded at ddd0a542:** `1410 tests run: 1410 passed (6 slow), 85
  skipped`, exit 0; before.txt = 1410 lines. The export was removed.
- **Open question:** does the arena cap also need to scale with cached context length?
  54ba74fc scales it by bind-time rows. AC19/AC20 at 131072 and 262144 will show it.
- **0b is done:** qwen3-8b loads on the dense path. On the CPU it printed
  " Paris. The capital of Germany is Berlin" (8 tokens, TTFT 212 s, TTNT 39.8 s), after
  the std telemetry-import fix.

Slices 7-7e (2026-09-29): the harness is written. Under `--features std,metal --example
long_context_niah`, the six filters print 1/1/1/1/1/3 passed, and 12/12 run in the whole
example. Haystack is War and Peace (PG #2600), sha256 d7a1d4c8…b7b5, about 750k tokens.
The rows' checkboxes were left unticked; that is fixed in the same commit as this note.

Harness gaps, each owned by the slice named:
- **Slice 8:** `kv_bytes` comes from the harness's own per-layer layout x element size,
  not the runtime's `MemoryBudget`, which is private and prices f32 only. Slice 8 makes
  `MemoryBudget` public and dtype-aware, and the harness then reads it.
- **Slice 11, qwen3.6 @ 262144:** `prefill_chunk_positions` is ignored unless the
  one-evaluation prefill path is on. That path is off by default and documented as
  numerically defective (`decode.rs:~2980-3001`). If 262144 prefill overruns memory,
  that path is the lever, and its defect becomes in-scope.
- **Slices 10-11:** `peak_metal_bytes` samples `current_allocated_size()` at load, the
  prefill boundary and each token, so a transient peak inside prefill can be missed. It
  is enough for the AC19 `P > kv_bytes` gate. It is not proof of peak memory.

Amendment 2026-09-29, slices 6 and 6b (gemma4 ring KV).
- **The spec named the wrong loop.** gemma4 never reaches `run_decode_loop_placed_kv`:
  `pregather.rs:2607` builds the single-range program only for `KvCacheShape::Uniform`, and
  gemma4 is `Custom`. Its KV is the host `LayerCache` of `run_decode_loop_observed_seeded`,
  copied into a scratch and re-bound as named blocks each step, so the ring is a host cache
  with a per-layer extent, and `omega/src/msl/cached_attention_render.rs` is unchanged (the
  omega tree is byte-identical to slot-0-ea's). SPEC R11/R12 architecture note is rewritten.
  R13/R15 and AC19 (`kv_bytes=811597824` at 131072) describe placed-KV buffers for gemma4 that
  do not exist; slices 8 and 9 rest on the same premise and need re-reading against
  `LayerCache`/`KvPadScratch` before they start.
- **Mechanism.** The program has no modulo op, so `p % window` is not expressible in the
  graph. `LayerCache::append_at` writes position p to ring row `(p + write_offset) % capacity`
  and `unroll_live_rows` copies the newest `min(cached_len, window)` rows oldest-first into
  the scratch. The two-range builder takes `sliding_kv_ring`: windowed layers' `kv_cache.*`
  leaves are bounded by `SLIDING_KV_SYMBOL` (slot 2, `symbols::SLIDING_KV_BOUND`,
  `FIRST_FREE` is now 3) and their mask reads `cached_len_swa`. Exactness: distance from
  query to key is unchanged by dropping the evicted prefix, and proxima-tensor's CPU test
  `two_range_cached_gemma4_sliding_ring_matches_full_cache_logits_across_prefill_and_decode`
  measures a max logit difference of 0 over a 3-token prefill, 1- and 2-token steps and a
  sliding layer sharing its donor's ring; the control (window one row stale) measures 0.395.
- **Fusion.** `cached_attention_candidates` binds a layer's ninth operand to whichever of
  `cached_len` / `cached_len_swa` its mask reads.
- **Layout is a load parameter.** `Architecture::bind` is unchanged (full cache), so every
  external `GEMMA4.bind` caller keeps its program. `LoadedModel::load` binds
  `KvLayout::SlidingRing`; `load_with_kv_layout(.., KvLayout::Full)` (hidden from docs) is the
  control arm, and `kv_layers_for_layout` strips windows only for that arm, so the memory fit
  prices what each layout allocates. `stored_kv_layers` is gone.
- **Speculative decode.** A verify step writes drafts before it knows how many survive, so a
  ring holds `window + draft_len` rows when `PROXIMA_SPECULATIVE_DECODE` is set at allocation,
  and speculation stays off for a call whose carried-in rings have no slack.
- **Ring-offset hook.** `LoadedModel::with_ring_write_offset_for_parity_control` (hidden from
  docs): the example is a separate crate, so `pub(crate)` cannot reach it. Not in `ServingConfig`.
- **Tests moved with the program.** The 8 real-checkpoint `symbol_dependency_kv_bucket_crossing`
  tests inferred at `[1, kv]`; they now bind slot 2 and mask both KV slots.
- **Not measured:** whether any speculative verify step fired in the GPU run with
  `PROXIMA_SPECULATIVE_DECODE=1` (both arms printed 256/256); ring behaviour past 2,048
  positions on the GPU.

Slice 7f run (2026-09-29, gemma4, ctx 8192, OMEGA_ARENA_TRANSIENT_CAP=1e9 build knob):
- **Ollama:** `--control` found 0/10 and the positive run found 10/10, so the harness
  controls hold (Y=10 >= 9).
- **proxima:** both arms failed. `arena peak_bytes=7436824048` (control, 7,732 prompt
  tokens) and `7734859792` (7,898 tokens) exceeded the 1e9 cap at query_rows=1. At 2,048
  tokens the peak was 723,518,320.
- **Reading:** peak grows about 10x for 3.8x length, near-quadratic. This is the gemma4
  PREFILL blocker. It must be root-caused before 131072 can run, because an [L,L]
  attention-scores materialization would be about 550 GB per layer at 131072.
- **Status:** root-cause dispatched. 7f stays unchecked.

main landed 2026-09-29: df3766dd became 3d0696d2 (checked with rev-parse).
- **Rebased hashes:** the series was rebased before landing, so ddd0a542 is NOT an
  ancestor of main. The arena prefill-cap fix is `554ebb1e` on main (was 54ba74fc).
- **Speculative default:** main turns speculative decoding on by default
  (`6028c267`). The niah harness and the parity example must pin
  `speculative: SpeculativeConfig::none()`, or record it per arm.
- **Base tokens:** `gemma4_base_tokens.txt` was recorded at ddd0a542, not at main.
  After the rebase, AC12's `full vs head` re-establishes whether main matches it.
- **Backup:** this spec's uncommitted work is saved as `long_ctx_tracked.patch` (173 KB)
  and `long_ctx_untracked.tar` in the session scratchpad.
- **Rebase needs commits:** moving onto main needs commits, and commits need the
  owner's authorization (slot-0 AGENTS.md: never commit without asking).

Port onto main 3d0696d2 (2026-09-29), in the export $HOME/repos/slot-0/.long_ctx_backups/tree_main, with
nothing committed.
- **Gates:** `283 passed, 0 failed` (std). All 14 slice filters and all 6 niah filters
  print their exact counts, and clippy std,metal exits 0.
- **Metal regression (AC23 comm):** NOT yet run, because the GPU was busy.
- **Patch:** `long_ctx_on_main.patch`, 3,960,844 B across 45 files, in the scratchpad.
- **Speculation:** main's per-call `SpeculativeConfig` sizes the ring slack
  (`speculative_draft_limit`). The default ngram-simple size_m is 48.
- **Drafter bounds:** each drafter's emit bound is cited in code (`proxima-tokenizer/src/draft`:
  `ngram_simple.rs:174,179`, `ngram_map.rs:449-450,527-528`, `ngram_mod.rs:282-301`,
  `ngram_cache.rs:577`).
- **Budget:** it now prices ring slack via `MemoryBudget::derive(.., draft_slack)`.
  gemma4 @ 131072 f32 = 1,624,375,296 B at default slack and 1,623,195,648 B with
  speculation off (my own earlier 1,624,178,688 was an arithmetic error that implied a
  slack of 40; the code-derived value is asserted).
- **AC19 amendment still pending:** it must name the speculative config. The harness pins
  `SpeculativeConfig::none()`, so its `kv_bytes` excludes the slack.

Order change, 2026-09-29:
- **Prefill memory before anything else.** It is the gating blocker for gemma4 above
  about 8K (see the `design-kv-residency.md` header, where v1 is REJECTED).
- **Slices 8/8b/9 are frozen** until prefill is root-caused and KV residency v2 is written.
  v2 must answer three things: device-handle ownership vs `ServingCache` clones,
  speculative rewind, and verify-vs-decode numerics under narrow storage.

Prefill root cause, PROVEN 2026-09-29 by a debugger dump of the arena live-set at the peak.
The live set sums exactly to peak_bytes at 4 sizes.
- **Mechanism.** gemma4 prefill is ONE evaluation of all L rows. `decode.rs` batch_count
  takes the `else { 1 }` arm, because gemma4 has `single_position_step: false`
  (`gemma4/bind.rs:1043`), so `split_prefill` and `one_evaluation_prefill` are both false.
  Its attention is unfused multiply-then-reduce on the two-range program. Three f32
  `[L, keys, 8]` score tensors plus two materialized `exp` buffers are live at the peak.
  Score bytes are `4*8*(2*L*ceil32(L) + L^2)`, which equals 5,988,325,920 exactly at
  L=7895, 77.5% of the peak.
- **Fit.** peak = 107.99*L^2 + 126,654*L B. Measured: 759 tokens 175,597,144;
  1782 tokens 568,615,696; 3831 tokens 2,069,936,152; 7895 tokens 7,730,974,360.
  Decode peak is 1,092,632 B, so decode is not the problem.
- **Refuted.** Logits (last-row gather, 1 MiB). Per-layer-input embeddings (linear in L).
  FFN (linear in L).
- **Fusion is not available.** `metal-fuse-attn-decode` declines prefill with
  `local_window_not_vacuous` (`dead_code_cached_attention.rs:1149`) and
  `softmax_weights_decode_only` (`:1225`). The kernel itself handles query_rows > 1
  (`cached_attention_render.rs:346-351`).
- **`plan_query_rows` = 1 is an artifact.** It is the fallback when no CachedAttention op
  exists.
- **Doc/code mismatch (fix it):** `serving.rs:495-499` documents
  `prefill_one_evaluation` default true; `serving.rs:687` sets false.

New slices, which go before 8.x and run in the main-based port tree $HOME/repos/slot-0/.long_ctx_backups/tree_main:
- **P1 chunked prefill for two-range architectures.** Reuse the existing two-range
  program with `ubatch_size`-wide chunks, advancing cached_len (llama.cpp's `-ub`
  meaning). No new program and no new config field. Gate: arena peak at fixed chunk C
  grows linearly in L, and greedy output is token-identical to the one-evaluation prefill
  at L where both fit.
- **P2 fused chunked prefill.** Relax the recognizer so the kernel takes multi-row
  windowed attention. That makes peak O(C*head_dim), which is required for 131072:
  at C=512, P1 alone is about 2.1 GB per score buffer.

main moved again 2026-09-29: 3d0696d2 became 73fd1cbd (slot-0-ea's 9-commit tiled-GEMM
series; 42 files, +8811/-760). Those commits also contain every root-checkout hunk that
belongs to slot-0-ea.
- **Dry run:** `patch --dry-run` of the 3d0696d2-based port onto a 73fd1cbd export
  ($HOME/repos/slot-0/.long_ctx_backups/tree_main2) fails 1 hunk of 21, in serving.rs. Everything else applies.
  Re-port after P1 lands in $HOME/repos/slot-0/.long_ctx_backups/tree_main.
- **Backups:** kept in `<session>/long_ctx_backups/`, NOT the scratchpad. An agent's
  scratch cleanup deleted the earlier scratchpad backups.
  - `long_ctx_on_3d0696d2_<HHMM>.patch`: the port, including any P1 work in progress at
    snapshot time.
  - `root_tracked_df3766dd.patch` and `root_untracked.tar`: the root tree's work.

Re-ported onto 73fd1cbd (2026-09-29) at $HOME/repos/slot-0/.long_ctx_backups/tree_main2.
- 20 of 21 hunks applied. The one reject was slot-0-ea's `exact_activations` doc, which
  main already has at `serving.rs:732`, so it was dropped.
- CPU gates: `cargo check --features std,metal --all-targets` 0 errors and 0 warnings;
  nextest std `287 passed, 34 skipped, 0 failed`; clippy exit 0.
- Next: apply P1's final diff (from $HOME/repos/slot-0/.long_ctx_backups/tree_main) on top, then retire
  long_ctx_main. long_ctx_main2 is the working tree from then on.

CORRECTNESS FINDING from slot-0-3f (2026-09-29, root-caused by them; the fix is theirs,
split with slot-0-ea, in omega).
- **Defect:** omega dispatches with more than 2^32 threads are silently truncated
  (`uint gid`, threads mod 2^32).
- **gemma4 node 17** (per-layer-embedding projection, [rows,1536]x[1536,8960], 256 lanes)
  overflows at 1873 rows. My check: 1873*8960*256 = 4,296,212,480 > 4,294,967,296,
  while 1872*8960*256 = 4,293,918,720 fits. The tail outputs read zero, with no error.
- **Consequence: every one-evaluation gemma4 prefill above 1872 tokens was computed wrong.**
- **`gemma4_base_tokens.txt` is INVALID as an oracle.** It came from a 2048-token
  one-evaluation prefill. AC12/AC13 parity results recorded against it are
  determinism-only. Re-record it after the omega fix lands on main.
  The file is deleted. It is re-recorded (`--record`, length as produced, EOS at 53 tokens
  on the corrected run) after the truncation fix lands; llama.cpp is the oracle for that recording.
- **P1 is not the fix.** At ubatch 512, chunked prefill keeps node 17 under the limit, but
  score ops overflow at about 4090 rows x 32 lanes, so large cached ranges must still be
  checked per dispatch. P1 sidesteps the bug below that limit; it does not fix it.
- **Unaffected:** the 8K niah run failed before decoding. The Ollama arm is unaffected.

P1 chunked prefill DONE in $HOME/repos/slot-0/.long_ctx_backups/tree_main (3d0696d2 base), 2026-09-29.
- **Mechanism:** non-single-position architectures chunk prefill by `ubatch_size` rows
  through the same two-range program (`ubatch_prefill_chunks`). No new config field.
- **Unit tests:** 4 passed. The measured relative diff vs one-evaluation is 1e-6 to 2e-6,
  against a Higham-derived bound of 3.74e-4, and a control exceeds the bound.
- **nextest std:** 287 passed, 0 failed. Correctness gate at ubatch 4: 4/4 answers.
  Clippy exit 0.
- **niah, gemma4, ubatch 512, seed 20260929:**

| ctx | arena peak, largest chunk (B) | found (proxima vs ollama) |
|---|---|---|
| 2048 | 122,159,424 | 10/10 vs 10/10 |
| 4096 | 193,462,592 | 10/10 vs 10/10 |
| 8192 | 336,068,928 | 10/10 vs 10/10 |
| 16384 | 621,281,600 | 10/10 vs 10/10 |

  Peak is linear: 34,811.67*L + 60,462,739 B, residuals within +-0.23%. At L=7895
  one-evaluation needed 7.73 GB; chunked needs 336 MB (23x less).
- **Thread census** (real program through `omega::emit`, no device):
  - One-evaluation exceeds 2^32 from about 1,365 rows. The first offender is the FFN
    gate/up reduce, rows x 3,145,728. Relayed to slot-0-3f.
  - Chunked at 512 stays under 2^32 through 16,118 keys (max 4.16e9, 96.9%). Past about
    16K keys a 512-row chunk overflows. Until slot-0-3f's 64-bit grid fix lands, chunk
    width must satisfy C * keys * (lanes factor) < 2^32; for 131072 keys that means C <= ~64.
- **Default ubatch_size is 32** (the owner invocation's `-ub 32`), so default-config
  prefill now chunks at 32 rows. The TTFT effect is UNMEASURED; it must be measured
  against Ollama before landing.
- **peak_metal_bytes limitation:** the harness samples after prefill, so it misses the
  prefill arena. The arena peak comes from the debug event at
  `arena_encode_dispatch_finish.rs:218`.
- **Unexplained:** the per-key increment of 34,816 B is not decomposed.
- **Parity vs `gemma4_base_tokens.txt`:** ubatch 512 gave 0/256 (a coherent 53-token
  summary). ubatch 0 gave 256/256 (verbatim copy). The base is corrupt (overflow), so
  both are measurements only.

Owner challenge, 2026-09-29: "isn't that just pushing the can down the road? what about
the frontier 1T models?" Accepted.
- **What P1 does not fix:** it bounds attention memory to O(C*L), so C must shrink as L
  grows (C <= ~64 at 131072). That is a stopgap.
- **P2 is now the primary mechanism:** fused, tiled online-softmax attention for prefill
  and decode, with scores never materialized. That is the FlashAttention shape, which
  Ollama/llama.cpp use via `-fa`. Attention memory becomes O(C*head_dim), independent of L.
- **P1's remaining role:** bounding linear activations (FFN, per-layer inputs) per chunk.
- **Generalization target:** one fused attention primitive parameterized by mask shape
  (causal, window) and KV layout (GQA, shared-KV, partial rotary, and MLA latent-KV for
  DeepSeek-V3/Kimi-K2-class models). Not a gemma4 special case.
- **Status:** P2 design dispatched to proxima-architect.

gemma4 TOKENIZER BUG, proven 2026-09-29 by a debugger.
- **Mechanism:** gemma4 is routed through GPT-2 byte-level BPE (`gguf.rs:98`,
  `vocab.rs:258-264`, `bpe.rs:22-25`, `pipe.rs:73-75`). The GGUF's merges are UTF-8-char
  keyed with `▁`, so every non-ASCII character and every newline mis-tokenizes.
- **Examples:** ` “` becomes 3 ids instead of `999`; `\n\n` becomes
  `[247723, 247723]` instead of `[108]`.
- **Effect on the harness:** 11-15% more niah prompt tokens than Ollama (32,417 vs
  28,168; 7,898 vs 7,070). Every NIAH and TTFT comparison so far is not token-matched.
- **Control:** ASCII single-space text is identical, 6,539 ids.
- **Fix:** being written (no cargo during slot-0-ea's GPU window) in long_ctx_main2
  proxima-tokenizer. Validation waits for the window.
- **Notified:** slot-0-3f.

ORACLE RULE CHANGE, owner, 2026-09-29, relayed by slot-0-3f and verified in memory
`feedback_no_llama_cpp_anywhere.md`: "cpu is not oracle btw. llama.cpp is."
- llama.cpp is the correctness oracle, via vendored fixtures from the external build
  (commit and command recorded).
- It is still never a runtime, crate, vendored code or dependency.
- proxima-internal comparisons are consistency checks only: CPU path, f64 reference,
  fused-vs-unfused, HEAD tokens.
- Affected: the tokenizer fix now gates on llama-tokenize id equality; the P2 v2 oracle
  gets the same treatment.
- **AC12/AC13 as written** (HEAD-token parity) are consistency checks, NOT oracle gates.
  An amendment is due: a llama.cpp oracle for ring-window correctness.
  WRITTEN: SPEC "amendment 2026-09-29: llama.cpp oracle for ring-window correctness"
  (R12L, AC12L-a..d). Slice 6L runs AC12L-a..d. It needs the tokenizer fix landed and the
  GPU window closed, because 12L-b runs llama.cpp on Metal. Owner: this session, next
  after tokenizer validation.
  Tools in the external build (commit f1ea20621): `llama-perplexity`
  (`--kl-divergence-base` saves all logits for a teacher-forced pass), `llama-eval-callback`
  and `llama-tokenize`. AC12L-a/b use llama-perplexity's saved logits over the fixed
  2,304-id sequence, with `-ngl 0` and `-ngl 99`.

Tokenizer fix WRITTEN (2026-09-29) in $HOME/repos/slot-0/.long_ctx_backups/tree_main2/proxima-tokenizer.
Uncompiled, because of the GPU window.
- **Oracle:** 15 llama.cpp fixtures (f1ea20621) in
  `tests/fixtures/llama-gemma4-tokenize/`. The prebuilt `llama-tokenize` predates gemma4,
  so a gemma4-capable build was compiled from tools/tokenize, vocab-only on CPU. It is
  preserved at `<session>/long_ctx_backups/tools/llama-tokenize-f1ea20621`.
- **Mechanism:** the char-level BPE is detected by the vocab's own shape probe (`▁` and
  `<0x0A>` present). The pre-split is newline-runs only. BPE seeds UTF-8 characters
  (heap-based), with byte fallback for unmatched symbols.
- **Second bug fixed:** decode of a literal non-ASCII token went through the GPT-2 inverse
  remap, so `é` became 0xE9.
- **Validation:** after the window, run `cargo nextest run -p proxima-tokenizer --features
  gguf gemma4_tokenizer`. Expected 33 tests, and the 20 real-blob tests must report ok
  with no SKIP. Also run the full proxima-tokenizer suite, a no_std check, and clippy.
- **Obsolete diagnostics:** the three examples `gemma4_tokenizer_{wholestring_bpe,
  spm_vs_bpe,spm_no_prefix}` examine the byte-seeded path this replaces. Remove them in
  the validation slice.

P2 v2 NOT ADMITTED (critique 2026-09-29; findings in the v2 design header).
P2 decision: two owner rules pull in opposite directions for fused attention:
- 2026-09-21: "byte equivalence stays binding; fusion NOT promoted until it matches
  unfused Metal bytes" (memory `project_gemma4_attn_fusion_recognizer_gap`).
- 2026-09-29: "cpu is not oracle btw. llama.cpp is."
Online softmax cannot be bit-equal to the unfused tree reduce by construction.

ANSWERED 2026-09-29 by the owner: **the llama.cpp oracle governs fused attention.**
Bit-equality to unfused Metal is dropped. Speculative verify-vs-decode byte identity (R1)
still binds. Memory updated. v3 design dispatched.

slot-0-ea GPU window DONE (2026-09-29). Its finding (n=2, plausible): a single ~2 s GPU
dispatch in any process triggers GPU recovery that kills every process's in-flight
buffers (`CommandBufferFailed 'Internal Error (0e)'`).
- **Constraint for this spec:** no single dispatch may run for seconds. P2 v3 must bound
  the per-dispatch duration.
- **Do not set `PROXIMA_METAL_ENCODER_ERROR_STATUS=1`** until slot-0-ea's fix lands; it
  panics on main.
- **Resumed:** tokenizer validation (dispatched) and the attention dispatch census.

Tokenizer fix VALIDATED on CPU (2026-09-29, long_ctx_main2).
- **Oracle gate:** 15/15 `gemma4_tokenizer_matches_llama_*` pass, so ids are identical to
  llama.cpp f1ea20621 on all fixtures. The filter matched 28 tests, 28 passed, 0 SKIP.
- **Full suites:** proxima-tokenizer `178 passed, 12 skipped`; proxima-model-interop
  (std) `287 passed, 34 skipped`.
- **Tiers and lint:** the no_std check and clippy `-D warnings` both exit 0.
- **Cleanup:** the three obsolete gemma4_tokenizer_* examples are deleted.
- **Backup:** `long_ctx_on_73fd1cbd_tok.patch`, 4,139,391 B.
- **Pending:** the Ollama prompt_eval_count parity measurement, after slot-0-3f's GPU
  window.

Attention census MEASURED (2026-09-29; `gemma4_attention_chain_census`, CPU bind, in long_ctx_main2):
- **Totals:** 1661 BoundOps per decode step.
- **Attention chain ops:** 609 total (36.7%), by the test's own direct-operand rule,
  which it labels HEURISTIC.
- **Per layer:** layers 0-14 (own KV) have 19-21 chain ops plus 8 other; layers 15-34
  (shared KV) have 16 chain ops plus 3 other.
- **Anomaly:** the fused_feature_on arm fused 0 ops (1661 = 1661, 0 absorbed). The
  2026-09-21 base recorded 1173 fused. Being checked on pristine main 73fd1cbd to see
  whether this spec's ring work (the `cached_len_swa` mask input) made the recognizer
  decline.
  RESOLVED: the anomaly was a degenerate run, because `--features metal` does not compile
  `metal-fuse-attn-decode`. With `--features metal,metal-fuse-attn-decode`, pristine main
  73fd1cbd and long_ctx_main2 print IDENTICAL results: unfused 1661, fused 1416,
  absorbed 245 (7 per layer), `sum_chain_ops_fused_feature_on=0`. The ring work does not
  change fusion.
  Note for P2: today's Candidate B decode fusion absorbs 245 of the 609 heuristic
  chain ops (40%).

P2 v3 NOT ADMITTED (critique 2026-09-29: 3 blockers, 12 major, 9 minor).
- **B3, decisive:** the premise that dispatch COUNT is the tok/s lever is unproven and
  contradicted by our own record. llama.cpp issues about 2k dispatches per token and is
  still faster (memory project_gemma4_decode_dispatch_type).
  - Measured slopes: about 1.8-4.9 us per removed dispatch (Candidate B: 245 removed for
    -1.0 to -1.2 ms; plan_time_constants: 262 removed for -0.47 ms).
  - Extrapolating linearly, even v3's T2 (1154 to 447) buys about 1.3-3.5 ms of
    21.2 ms/token. That leaves about 17.7 ms against Ollama's 10.0.
- **B1:** several D-slice eliminations are contradicted by the code:
  - The RoPE plane reads the norm node twice, so the epilogue is declined.
  - The D2 tail is the reverted `epilogue_sources` patch.
  - D3 counts Identity copies that belong to `identity-copy-alias`.
  - D5 has no primitive.
  - Flash is 2 dispatches per layer at P > 1, not 1.
  - T1 is at least 702, not 602.
- **B2:** v3 narrowed to gemma4 only, but the SPEC covers qwen3.6 and dense at 131K/262K.
- **Major:** `AttentionFold` duplicates `NumericRewrite` admission (`numeric.rs:137-139`).
  The oracle threshold is chosen, not derived. The 2 s trigger scope is unmeasured, and
  one prefill command buffer is about 6.4 s at 131K. The decode gate cannot resolve a
  1% effect with n=3.

DESIGN ITERATION PAUSED. The next step is measurement, not v4.
- **Why:** no design can be judged until per-op GPU time exists.
- **Slice M0:** per-kernel GPU time for one gemma4 decode step. Replay each distinct
  emitted kernel in isolation on captured buffers, N iterations each, timed with GPU
  timestamps. That avoids the 50-100x contaminated per-dispatch profiler.
  - An existing precedent is `omega/examples/attn_fused_replay.rs`.
  - Reconcile the sum of per-kernel times against the measured whole-step gpu_exec.
  - Output: ms per op class (matvec by codec, norm, rope, softmax, attention dot/AV,
    elementwise, head).
- **GPU:** M0 needs the GPU and waits for slot-0-3f's "done".
- **Owner-only alternative:** an Xcode Metal GPU capture of one decode step gives the
  same breakdown directly. It is a GUI tool.

Open question, if any: none

## struck

-

REBOOT LOSS, 2026-09-30.
- **What happened:** /private/tmp was wiped by a reboot. It took the port trees
  (long_ctx_main, long_ctx_main2), every backup, P1 chunked prefill, the tokenizer fix and
  its llama fixtures, the draft-slack pricing, the census tests and the M0 harness.
- **Rule (owner):** never put a work tree, export, patch, backup, fixture or tool in /tmp.
  Durable path: `~/repos/slot-0/.long_ctx_backups/`. Only cargo target dirs may stay in
  /tmp. All former `/private/tmp/long_ctx_*` paths in this spec now point there.
- **Survivors:** the root checkout (df3766dd-based slices 1-5, ring, harness, parity
  example) and this spec dir. They are backed up in `.long_ctx_backups/`.
- **LANDED 2026-09-30** on local main (unpushed): 9 commits, 73fd1cbd..ad787c34.
  Each was verified green from its own `git archive` tree (check std,metal all-targets;
  interop nextest 247 at base, growing to 282; tensor 690 at base, growing to 693; 0 failed).
- **Redo next, in root, as new commits:** P1 chunked prefill, then the gemma4 tokenizer
  fix with its llama.cpp fixtures.
- **Earlier plan (done):** land the root onto main in coherent commits (owner-authorized
  2026-09-30), then redo P1, the tokenizer fix and draft-slack in the root.

## resume (2026-09-30)

- **Landed on local main (unpushed):** 9 commits, 73fd1cbd..ad787c34.
- **On a detached HEAD above ad787c34** (main is held for slot-0-84, which lands its
  ~50 commits next):
  - 2ed803b1 chunked prefill (ubatch)
  - 2bbdc657 gemma4 tokenizer with llama.cpp fixtures
  - a0e4a5f1 obsolete tokenizer diagnostics removed
  - Each was verified green from its own `git archive` tree.
- **Durable tools:** `~/repos/slot-0/.long_ctx_backups/tok/llama-tokenize`, a
  gemma4-capable build of f1ea20621.
- **Next:**
  1. When slot-0-84 sends its sha, rebase the detached commits onto it.
  2. After its GPU window closes:
     - the AC23 Metal regression;
     - Ollama token-count parity;
     - niah at 8K/16K/32K with chunked prefill;
     - M0, the per-kernel GPU-time census that must precede any P2 v4.
- **P2 (fused attention):** v1-v3 are not admitted. Design is paused until M0 data exists.
- Tokenizer cross-check on slot-0-84's gemma4 26B MoE prompts (blob ea549b76…),
  HEAD 95519693 against llama.cpp's ids:
  - p0/p1/p2 are exactly equal (198/1000/2426 tokens). Old proxima gave 204/1013/2470
    and diverged at index 3.
  - The `\n` + 4 spaces + `--` case gives [140,726], as llama.cpp does.
  - The gemma4-capable llama-tokenize equals HEAD on 15 probe files.
  - Evidence: `.long_ctx_backups/tok/peer_check/`.
- **Ring-parity oracle available (2026-09-30, from slot-0-84):**
  `../../../../proxima-speculative-decode-evidence/preland/ring_prompt_llama_ids.json`
  (llama-server f1ea20621, Metal, greedy: 51 ids ending at `<turn|>` 106) and
  `ring_prompt_ids.json` (the 2048 prompt ids).
  - slot-0-84's branch, which carries the truncation fix, matches all 51 on the full-KV arm.
  - Main diverges at index 0.
  - proxima keeps generating past 106. Known: gemma4's 106 is text-suppressed but not a
    stop (the stop-set is downstream chat policy). The re-record is compared up to and including the
    first 106.
