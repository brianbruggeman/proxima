# long-context

status: audited (spec-auditor ADMIT on pass 8, 2026-09-29; R 28, AC 33, 0 orphaned, 0 dangling)
owner: brian
created: 2026-09-29

## problem

On the 64 GB M1 Max, proxima serving gemma4-E2B, qwen3.6-35b-a3b or qwen3-8b reaches
each model's target context (131072, 262144, and 131072 via YaRN) with needle recall no
lower than Ollama at the same `num_ctx`, measured by the `long_context_niah` harness's
found/N count and its KV byte count.

## refutation condition

Any one of these proves this was the wrong thing to build:

- at native length, proxima finds fewer needles than Ollama on the same prompts with the
  same `num_ctx`, and the gap persists with f32 KV. That means the long-context path is
  wrong, not the storage.
- on qwen3-8b at 131072, the YaRN arm finds no more needles than the unscaled arm, run
  with `ContextLength::Extrapolate`. That means YaRN bought nothing (see "out of scope": DCA).

## measured inputs (read 2026-09-29 from `ollama /api/show`, not assumed)

| model | arch | GGUF context_length | layers | KV-bearing layers | kv heads x head dim | sliding window |
|---|---|---|---|---|---|---|
| gemma4:e2b-it-qat | gemma4 | 131072 | 35 | 15 own (12 sliding, 3 full); 20 share | 1 x 256 (swa), 1 x 512 (full) | 512, pattern 4:1 |
| qwen3.6:35b-a3b | qwen35moe | 262144 | 40 | 10 (`head_count_kv` nonzero every 4th) | 2 x 256 | none (GDN in the other 30) |
| qwen3:8b | qwen3 (dense family, `architecture.rs:262`) | 40960 | 36 | 36 | 8 x 128 | none |

Qwen3-8B's YaRN setting uses `original_max_position_embeddings = 32768`, not the GGUF's
40960 (https://huggingface.co/Qwen/Qwen3-8B). So the scaled limit comes from the scaling
config's own `original_context`, not from the GGUF value.

The window is 512 on the real gemma4 E2B checkpoint. `GEMMA4_SLIDING_WINDOW = 1024` at
`proxima-tensor/src/spec/descriptor.rs:143` is a descriptor fixture, not this value.

## today's blockers (read, with file:line)

- **Trained context is never read.** It is not parsed from the GGUF, and the default is
  a hard-coded 131072 (`serving.rs:566`).
- **KV is over-charged.** `MemoryBudget::derive` charges KV to every layer
  (`memory_fit.rs:124-125`).
- **Sliding layers are mask-only.** They store and scan the full sequence
  (`gemma4/bind.rs:770`; no window-bounded allocation in `decode.rs:5470-5539`).
- **KV is f32 only.** `serving.rs:689-707`.
- **The default config is rejected by its own admission check** (`serving.rs:568,570`
  vs `:689-718`).
- **RoPE has no scaling.** `residency_caches.rs:1126-1130`.

## requirements

The limit is `L = original_context x factor` under scaling, and `L = trained` (the GGUF
`{arch}.context_length`) without scaling.

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `Architecture::trained_context_length` returns `{arch}.context_length` from the GGUF | yes |
| R2 | With `context_length = ContextLength::Native` (the default), the requested context is L | yes |
| R3 | The requested context is then clamped by `fit_context_length` to what memory admits | yes |
| R4 | `ContextLength::Within(n)` with n above L is rejected with `ContextExceedsTrained { requested, limit, scaling }`. `ContextLength::Extrapolate(n)` admits any n | yes |

Amended 2026-09-29: ~~`context_length: Option<u32>` plus `allow_extrapolation: bool`~~ became
one `ContextLength` enum. The pair could represent a meaningless state
(`allow_extrapolation = true` with `None`), and the bool was `ServingConfig`'s 13th, which
tripped `clippy::struct_excessive_bools`. The enum makes the invalid state
unrepresentable (`feedback_capability_is_a_composable_type_not_a_bool`). AC2 and AC4 keep
their case counts; `--allow-extrapolation` in AC21 maps to `Extrapolate`.
| R5a | `MemoryBudget` charges gemma4 KV only to its 15 own-KV layers, not the 20 shared ones | yes |
| R5b | `MemoryBudget` caps each gemma4 sliding layer's KV at `sliding_window` rows | yes |
| R6 | `MemoryBudget` charges qwen35moe KV only to layers with `head_count_kv[i] != 0` | yes |
| R7 | `ServingConfig::default()` passes `apply_serving_config` | yes |
| R8 | `RopeScaling` (`none`, `linear { factor }`, `yarn { factor, original_context, extrapolation_factor, attention_factor, beta_fast, beta_slow }`) is parsed from GGUF `{arch}.rope.scaling.*` | yes |
| R9 | A per-call `RopeScaling` override replaces the GGUF value | yes |
| R10a | YaRN inverse frequencies match `transformers` `_compute_yarn_parameters` on `yarn-worked-example.md` | yes |
| R10b | The YaRN attention factor matches the same reference (0.1 x ln(factor) + 1 when no mscale keys) | yes |
| R10c | `build_position_inputs` emits cos/sin equal to `cos/sin(pos x inv_freq) x attention_factor`, within the derived f32 tolerance | yes |
| R11 | gemma4 sliding layers allocate `min(positions_needed, sliding_window)` rows | yes |
| R12 | With the ring, gemma4 still answers the 4 correctness-gate facts, and finds needles past the window at least as often as Ollama | yes |
| R13 | The placed-KV Metal path stores and reads K/V as F16 | yes |
| R14 | `ServingConfig::default()` KV type is F16, matching Ollama's `OLLAMA_KV_CACHE_TYPE` default | yes |
| R15 | The placed-KV Metal path stores K/V as Q8_0 (quantize on write, dequantize in the cached-attention kernel) | yes |
| R16a1 | The niah haystack source is a checked-in Project Gutenberg text whose sha256 matches a pinned constant | yes |
| R16a2 | The haystack is trimmed to within 1% of the requested token count | yes |
| R16a3 | N needles land at token depths `(i + 0.5) / N`, within one needle length | yes |
| R16b1 | The niah harness prints `kv_bytes=` for every proxima arm | yes |
| R16b2 | The niah harness prints `peak_metal_bytes=` for every proxima arm | yes |
| R16c1 | The Ollama request's prompt bytes equal the proxima arm's prompt bytes | yes |
| R16c2 | The Ollama request sets `num_ctx` to `--ctx` and `temperature` to 0 | yes |
| R16c3 | Both arms are scored by one exact-match function | yes |
| R17 | proxima's decode-time ratio of Q8_0 to F16 KV at 65536 is no worse than Ollama's same ratio on the same model | yes |
| R18 | Every test that passed before slice 1 still passes after slice 13 | yes |

## architecture

Everything here is format parsing, a generic op, a kernel, the existing runtime, or a bug
fix. All of it lives in proxima (`feedback_model_work_lives_in_ragd_not_proxima`).

- **R1-R4.**
  - `Architecture::trained_context_length(&self) -> Option<u32>`.
  - `ServingConfig.context_length` becomes `ContextLength` (`Native` default, `Within(n)`,
    `Extrapolate(n)`); `Extrapolate` is the llama.cpp behaviour of running past the trained
    length, made explicit instead of silent.
  - `fit_context_length` (`memory_fit.rs:196-245`) runs after L is resolved.
  - The rejection is a new `InteropError` variant beside `SequenceExceedsContextLength`
    (`error.rs:276`).
- **R5/R6.** `MemoryBudget::derive` takes `&[u64]` per-layer KV row bytes in place of
  `block_count x kv_row_bytes`. Each architecture supplies the slice from its existing
  layout:
  - gemma4: `KvCacheShape::Custom` (`gemma4/bind.rs:1069`);
  - qwen35moe: per-layer `head_count_kv`;
  - dense: `block_count` copies.
- **R8/R9.** `RopeScaling` is an enum field in `ServingConfig`, serde plus conflaguration
  (principle 4). GGUF keys use llama.cpp names: `rope.scaling.type`, `.factor`,
  `.original_context_length`, `.attn_factor`, plus `yarn_*` when present.
- **R10.** YaRN goes into `build_position_inputs` (`residency_caches.rs:1103-1142`) as an
  inv_freq substitution plus cos/sin multiplied by attention_factor, which scales logits
  by af^2. No kernel changes. The gemma4 sliding table (`gemma4/program.rs:27-44`) is
  untouched.
- **R11/R12.** gemma4 never reaches `run_decode_loop_placed_kv`: `pregather.rs:2607` gives
  the single-range program only to `KvCacheShape::Uniform` architectures, and gemma4's is
  `Custom` (`gemma4/bind.rs:1068`). Its KV is the host `LayerCache` of the two-range loop
  (`run_decode_loop_observed_seeded`), copied into a scratch and re-bound as named blocks every
  step. The ring lives there. Sliding layers hold `min(positions_needed, window)` rows
  (`LayerCache::ring`), position p is written to row `p % capacity` (`append_at`), and a read
  unrolls the newest `min(cached_len, window)` rows oldest-first into the scratch
  (`unroll_live_rows`). The program takes those rows under a second extent slot
  (`SLIDING_KV_SYMBOL`) and their count as `cached_len_swa`; its mask is the existing
  `causal_mask_cached_windowed`, whose query-to-key distance does not change when the evicted
  prefix is dropped. No kernel change: the fused cached-attention op already reads its live row
  count from a rank-0 input, and `cached_attention_candidates` now accepts either
  `cached_len` or `cached_len_swa` as that input.
- **R13/R15.** The KV element type becomes a parameter of the placed-KV buffers and of
  `render_cached_attention` (`omega/src/msl/cached_attention_render.rs`).
  - F16 is a half load.
  - Q8_0 uses 32-element blocks with an f16 scale, dequantized inside the key/value load.
  - Keys stay split into `k_even`/`k_odd`, and each half is block-quantized on its own.
  - The threadgroup budget (`omega/omega-runtime.toml:283`, 32768 B) is unchanged,
    because dequant happens in registers.

### KV bytes the harness must report (formula)

| model @ length | f32 | f16 | q8_0 |
|---|---|---|---|
| gemma4 E2B @ 131072, ring on | 12x2048x512 + 3x4096x131072 = 1,623,195,648 | 811,597,824 | 431,161,344 |
| qwen3.6 @ 262144 | 10x4096x262144 = 10,737,418,240 | 5,368,709,120 | 2,852,126,720 |
| qwen3-8b @ 131072 | 36x8192x131072 = 38,654,705,664 | 19,327,352,832 | 10,267,656,192 |

q8_0 is 34 B per 32 elements. The k_even/k_odd halves are 128/256 (gemma4), 128
(qwen3.6) and 64 (qwen3-8b) elements, all multiples of 32.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| default KV type | F16 | Q8_0 default diverges from the incumbent's default (principle 14). Q8_0 is opt-in, as in Ollama |
| limit under scaling | `original_context x factor` | `trained x factor` would allow qwen3 163840 (40960 x 4), past the 131072 Qwen validated |
| running past L | reject unless `ContextLength::Extrapolate` | silent extrapolation is today's defect. An explicit opt-in keeps the unscaled arm of the refutation runnable |
| where YaRN lives | host angle table plus af on cos/sin | a kernel-side rope duplicates `fused_rope_pair` for one scaling law |
| sliding ring vs mask-only | ring | mask-only stores and scans 24,576 B/token of sliding KV at every length |
| R12 oracle | external: the 4 world-fact answers of `gemma4_correctness_gate.rs`, plus Ollama's needle count at 8192 (AC12x). HEAD token parity (AC12) and its offset control (AC13) stay as a determinism check, labelled as one | audit 6 was right. Principle 14 names external incumbents, and `feedback_incumbent_is_llama_not_us` records that a self-referential gate cost 2 weeks. HEAD parity alone could pass a ring that is broken in the same way HEAD is |
| R7 oracle | the admission function, with a paired rejection control re-run by slices 2, 8b, 9 and 14 (TASKS.md) | R7 is not behaviour-preserving; it repairs a self-consistency defect (the default violates the repo's own admission contract). Slices 8 and 9 widen admission to F16/Q8_0, so the control pins the configs that must still be refused (Q4_0 KV, `parallel_sequences > 1`) and fails loudly if admission is loosened to pass the default |
| theta precision | f32, as today and as the reference | f64 angles would diverge from the torch/ggml incumbents at long positions |
| int4 KV | not in this spec | arXiv 2604.16957 reports 25-100x more directional error on Gemma 4 than on Llama under angular quantization. It needs `/discovery-loop` with a kill criterion |

## acceptance criteria

Commands run from `proxima/` after `source proxima-tensor/specs/long-context/env.sh`. That
file sets `CARGO_TARGET_DIR`, the absolute blob paths `$GEMMA4`, `$QWEN36`, `$QWEN3_8B`
and `$LONG_CTX_SPEC`, and defines the shell function `niah`. That function runs
`cargo run -p proxima-model-interop --release --example long_context_niah --features std,metal -- "$@"`.

Recall gates are about needles found. `X` is proxima's count, `Y` is Ollama's, and
`A`/`B`/`C` are proxima's f32/f16/q8_0 arms.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-model-interop --features std trained_context_read` | 3 passed: gemma4 131072, qwen35moe 262144, qwen3 40960, each from real GGUF metadata bytes |
| AC2 | R2 | `cargo nextest run -p proxima-model-interop --features std context_default_resolves` | 2 passed: qwen3 unscaled -> 40960; qwen3 yarn(4, 32768) -> 131072 |
| AC3 | R3 | `cargo nextest run -p proxima-model-interop --features std context_fit_clamps` | 1 passed: qwen3.6 f32 KV (40,960 B/token over 10 layers), weights 0, arena 0, `limit_bytes = 4,096,000,000`, headroom 0, requested 262144 -> `ReducedContext { from: 262144, to: 100000 }` (4,096,000,000 / 40,960, by `memory_fit.rs:236`) |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --features std context_over_limit` | 3 passed: 40961 unscaled rejected; 131073 with yarn(4) rejected; 131072 unscaled with `Extrapolate` admitted |
| AC5a | R5a | `cargo nextest run -p proxima-model-interop --features std memory_budget_gemma4_own_layers` | 1 passed: @ 131072 f32 with no window cap = 15 own layers = (12x2048 + 3x4096) x 131072 = 4,831,838,208 B (today's formula over 35 layers is larger) |
| AC5b | R5b | `cargo nextest run -p proxima-model-interop --features std memory_budget_gemma4_window_cap` | 1 passed: @ 131072 f32 with cap 512 = 1,623,195,648 B |
| AC6 | R6 | `cargo nextest run -p proxima-model-interop --features std memory_budget_qwen35moe` | 1 passed: @ 262144 f32 = 10,737,418,240 B (today's formula gives 42,949,672,960) |
| AC7 | R7 | `cargo nextest run -p proxima-model-interop --features std serving_default_admission` | 2 passed: `apply_serving_config(&ServingConfig::default(), 1)` is `Ok`; and the control, `ServingConfig { parallel_sequences: 2, ..default }` and `ServingConfig { kv_cache_key_quant: Q4_0, ..default }`, are each `Err`, so the admission function still rejects what it must at every slice |
| AC8 | R8 | `cargo nextest run -p proxima-model-interop --features std rope_scaling_from_gguf` | 4 passed: none, linear, yarn, unknown type rejected |
| AC9 | R9 | `cargo nextest run -p proxima-model-interop --features std rope_scaling_override` | 1 passed |
| AC10a | R10a | `cargo nextest run -p proxima-model-interop --features std yarn_inv_freq` | 2 passed: pairs 0/20/30/40/63 to 1e-6 relative, and the `low == high` guard |
| AC10b | R10b | `cargo nextest run -p proxima-model-interop --features std yarn_attention_factor` | 1 passed: factor 4 -> 1.138629436111 to 1e-6 |
| AC10c | R10c | `cargo nextest run -p proxima-model-interop --features std yarn_scaled_rope_table` | 1 passed: the 11 position/pair cells of the worked example within `\|theta\| x 2^-22 x af + 1e-6` (two f32 roundings; see `yarn-worked-example.md`) |
| AC11 | R11 | `cargo nextest run -p proxima-model-interop --features std gemma4_ring_rows` | 1 passed: at positions_needed 2048, 12 sliding layers allocate 512 rows and 3 full layers 2048 |
| AC12 | R12 (determinism check, not the oracle) | `cargo run -p proxima-model-interop --example gemma4_ring_parity --features std,metal -- $GEMMA4 --expect $LONG_CTX_SPEC/gemma4_head_tokens.txt` (2,048-token prompt, 256 greedy tokens) | `full vs head: K/K ring vs head: K/K` (K = recorded length, EOS or `--max-tokens`). Amended 2026-09-30: the 256-token HEAD/base recordings were computed through the >2^32-thread truncation and are deleted as corrupt; the head file is re-recorded on main after the truncation fix lands, and is accepted only if it matches llama.cpp on the same ids (R12L) |
| AC12x | R12 (external oracle) | with the ring enabled (default after slice 6b), three commands. (a) incumbent control: `cargo run -p proxima-model-interop --example gemma4_ring_parity --features std,metal -- "$GEMMA4" --ollama-facts \| grep -c '^ollama answered_correctly=true'`. (b) `cargo nextest run -p proxima-model-interop --features std --features metal -E 'test(gemma4_e2b_answers)' --no-capture 2>&1 \| grep -c 'answered_correctly=true'`. (c) `niah --model "$GEMMA4" --ctx 8192 --needles 10` (8192 is 16 windows of 512, so every needle but the last sits outside the sliding window) | (a) `4`: Ollama passes the same four substring checks (Paris, soliloquy, briefcase, hippopotamus from `tests/gemma4_correctness_gate.rs:1-108`, world facts) on the same prompts. (b) `4`, which also rules out the skip path at `gemma4_correctness_gate.rs:112-117`, where 0 lines print. (c) `proxima found=X/10 ollama found=Y/10` with Y >= 9 and X >= Y |
| AC13 | R12 control | the same with `--ring-offset 1` | `full vs head: K/K` and `ring vs head:` below K |
| AC13b | R16a1 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_haystack_sha` | 1 passed |
| AC13c | R16a2 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_haystack_trim` | 1 passed: 8192 requested -> 8110..8274 tokens |
| AC13d | R16a3 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_needle_depths` | 1 passed: 10 needles at `(i + 0.5) x 819.2` +/- one needle length |
| AC13e | R16c1 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_ollama_request_prompt` | 1 passed |
| AC13f | R16c2 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_ollama_request_options` | 1 passed: `num_ctx == 131072`, `temperature == 0` in the serialized body for `--ctx 131072` |
| AC13g | R16c3 | `cargo nextest run -p proxima-model-interop --features std,metal --example long_context_niah niah_scoring_shared` | 3 passed: exact needle, needle with surrounding prose, wrong digits |
| AC14 | R16c3 negative control | `niah --model "$GEMMA4" --ctx 8192 --needles 10 --control` | `proxima found=0/10 ollama found=0/10` |
| AC15 | R16c1-R16c3 positive control | `niah --model "$GEMMA4" --ctx 8192 --needles 10` | `ollama found=Y/10` with Y >= 9, and proxima X >= Y. At 8K the incumbent must find the needles, or the harness is broken |
| AC16 | R13 | `niah --model "$GEMMA4" --ctx 32768 --needles 10 --kv f32,f16` | `f32 found=A/10 f16 found=B/10 ollama found=Y/10`; gate B >= Y, B == A, and Y > 0 |
| AC17 | R14 | `cargo nextest run -p proxima-model-interop --features std default_kv_is_f16` | 1 passed |
| AC18 | R15 | `niah --model "$GEMMA4" --ctx 32768 --needles 10 --kv f16,q8_0` | `f16 found=B/10 q8_0 found=C/10 ollama found=Y/10`; gate C >= Y and Y > 0 |
| AC19 | R16b1, R16b2, R2, R11, R13 | `niah --model "$GEMMA4" --ctx 131072 --needles 10 --kv f16` | `kv_bytes=811597824` and `peak_metal_bytes=P` with P > 811597824; gate X >= Y and X > 0 |
| AC20 | R16b1, R6, R13 | `niah --model "$QWEN36" --ctx 262144 --needles 10 --kv f16` | `kv_bytes=5368709120`; gate X >= Y and X > 0 |
| AC21 | R8-R10c, R15 | `niah --model "$QWEN3_8B" --ctx 131072 --needles 10 --kv q8_0 --rope-scaling none,yarn:4:32768 --allow-extrapolation` | `kv_bytes=10267656192`, plus `ollama found=Y/10` (Ollama runs the GGUF unscaled, because it carries no `rope.scaling.*` keys); gate yarn found > none found and yarn found >= Y and yarn found > 0 |
| AC22 | R17 | `cargo run -p proxima-model-interop --release --example kv_dtype_decode_bench --features std,metal -- --model $GEMMA4 --ctx 65536 --iterations 5` (proxima f16/q8_0 and Ollama f16/q8_0, interleaved per iteration) | 4 arms with ms/token and CoV; gate proxima q8/f16 <= ollama q8/f16 |
| AC23 | R18 | parser proof: `$LONG_CTX_SPEC/passed_tests.sh --self-test \| wc -l`. Baseline (slice 0a, against HEAD): `(cd $HOME/repos/slot-0/.long_ctx_backups/tree_head && CARGO_TARGET_DIR=/private/tmp/cargo_target_long_ctx_head /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/long-context/passed_tests.sh) > $LONG_CTX_SPEC/before.txt && wc -l < $LONG_CTX_SPEC/before.txt`. Final (slice 14, working tree): `$LONG_CTX_SPEC/passed_tests.sh > $HOME/repos/slot-0/.long_ctx_backups/tree_after.txt && comm -23 $LONG_CTX_SPEC/before.txt $HOME/repos/slot-0/.long_ctx_backups/tree_after.txt \| wc -l` | parser proof `2`; baseline a count > 0; final `0` |

### amendment 2026-09-29: llama.cpp oracle for ring-window correctness (R12L / AC12L)

The owner rule is "llama.cpp is the oracle; proxima's CPU/HEAD output is not". AC12 and
AC13 are therefore consistency checks only. R12L supplies the oracle.

| id | discharges | command | expected |
|---|---|---|---|
| AC12L-a | R12L fixture | the external build `others/llama.cpp/build/bin`, CPU backend (`-ngl 0`), temperature 0, teacher-forced over the fixed 2,048-token gemma4 prompt plus 256 continuation tokens: dump the per-position argmax. Vendor it in `proxima-model-interop/tests/fixtures/llama-gemma4-ring/` with the llama.cpp commit, command and prompt ids. The prompt is tokenized by the FIXED tokenizer, so its ids equal llama-tokenize's | the fixture has 2,304 argmax ids |
| AC12L-b | R12L bar | the same, with the llama.cpp Metal backend (`-ngl 99`) against its own CPU backend | the agreement count A_ll/2304 is recorded. This is the incumbent's own cross-backend agreement, and it is the bar |
| AC12L-c | R12L | proxima with the ring on (default), teacher-forced over the same ids, per-position argmax vs AC12L-a | agreement >= A_ll, and the first disagreement index is printed |
| AC12L-d | R12L control | the same with `--ring-offset 1` | agreement < A_ll. If it is not lower, the check does not see the ring |

- R12L: with the ring on, gemma4's teacher-forced top-1 agreement with llama.cpp's CPU
  backend is no lower than llama.cpp's own Metal-vs-CPU agreement on the same prompt.
- Prerequisites: the tokenizer fix (so the prompt ids match llama.cpp), and slot-0-3f's
  grid-overflow fix (so a 2,048-row evaluation is not truncated). Chunked prefill avoids
  the truncation for chunks under 1,365 rows.

## out of scope

- **Downstream wiring** of the new `ServingConfig` fields. Separate repo; its own spec after
  this lands.
- **int4 KV** (see decisions).
- **Dual chunk attention, Jet-Long, Self-Extend.** Triggered only if AC21 refutes YaRN.
- **KV eviction** (SnapKV, DuoAttention, H2O). These lose recall by design, and nothing
  in this spec's targets needs it at 64 GB.
- **Sparse prefill** (MInference).
- **`parallel_sequences > 1`**, and paged or multi-slot KV.
- **Trained-sparse attention** (NSA, MoBA, DSA). Not applicable to existing checkpoints.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| qwen3-8b is covered by the dense family (`architecture.rs:262`) but never loaded from its real checkpoint | medium | AC21 has no model | slice 0b loads it with `bench_local`. A load failure is fixed inside this spec |
| Ollama cannot serve qwen3.6 @ 262144 on 64 GB | medium | AC20 loses Y | AC20 is recorded as failed with Ollama's error verbatim, and an amendment adds an AC at the largest `num_ctx` Ollama does serve, with the same X >= Y gate. The oracle stays external |
| prefill at 262144 overruns the transient workspace (mlx-lm #1480 hit this at 176K) | medium | AC20 OOM | the harness sets `prefill_chunk_positions` (`serving.rs:452`; 0 = off today) and records peak bytes per chunk size |
| Q8_0 dequant slows decode at depth (a CUDA blog shows 55% of f16 at 64K) | high | decode regression | AC22 gates it against Ollama's own ratio, measured interleaved |
| another session holds the GPU (slot-0-3f timing) | certain today | cargo and Metal slices wait | slices start after slot-0-3f sends "done", because it asked for cargo builds to wait too |
| gemma4 greedy disagrees with Ollama early for reasons unrelated to the ring (`project_gemma4_correctness_gate`) | medium | the niah gates (AC15-AC19) fail for a non-context reason | AC15's 8K positive control separates the two: if proxima misses needles at 8K, the fault is not context length |

## context

- `proxima-model-interop/src/serving.rs:212,452,566-570,657-718`: context field, prefill chunking, defaults, admission
- `proxima-model-interop/src/memory_fit.rs:50,124-125,196-245`: KV pricing and fit
- `proxima-model-interop/src/generate/residency_caches.rs:1103-1142`: RoPE angle table
- `proxima-model-interop/src/generate/decode.rs:5470-5539`: placed-KV allocation
- `proxima-model-interop/src/gemma4/bind.rs:296,689,770,1069`: gemma4 KV layout and window mask
- `proxima-model-interop/src/architecture.rs:262`: dense family covers qwen3
- `omega/src/msl/cached_attention_render.rs`: online-softmax cached-attention kernel
- `omega/omega-runtime.toml:188-283`: chunk, split and threadgroup limits
- `yarn-worked-example.md` (this directory): R10 walk and tolerance
- survey 2026-09-29: Qwen3 YaRN (https://huggingface.co/Qwen/Qwen3-8B); DCA
  (https://arxiv.org/abs/2402.17463); Gemma 4 KV quant risk
  (https://arxiv.org/abs/2604.16957); fused int4 Metal KV (https://arxiv.org/abs/2605.05699)
