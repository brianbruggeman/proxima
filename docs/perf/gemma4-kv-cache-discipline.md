# gemma4 kv-cache discipline log

Correctness is the first gate for this initiative; performance is second.
The active component engages gemma4 decode with the existing single-range
cached-attention family instead of the cacheless full-reprefill path.

**Deviation from the skill default:** this log lives in-repo at
`docs/perf/gemma4-kv-cache-discipline.md` rather than the slot-0 Obsidian
vault, matching the existing convention of `docs/model-interop/discipline.md`
-- kept with the code on this branch by explicit instruction for this slice.

13-point gate cells (see `/disciplined-component`): Build / Tests / Clippy /
Micro-bench / Compare-bench / E2E / Opt / SIMD-SM-no-Box / O(1) / Cfg-API /
Home-turf / Δ / Notes.

## C1 -- gemma4 single-range cached decode (`gemma4-kv-cache` feature)

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cargo check -p proxima-model-interop --features std,metal,gemma4-kv-cache` = 0 (also `--all-targets` = 0); flag-off `--features std,metal` = 0 (also `--all-targets` = 0); `cargo check -p proxima-tensor --features std,config` (+ `--all-targets`) = 0 | `cargo nextest run -p proxima-model-interop --features std,metal`: **261 passed, 0 failed, 59 skipped** (this slice's required gate); `cargo nextest run -p proxima-tensor --features std,config,test-support,cached-attention-streaming,kv-capacity-bucket`: 669/670 passed, 1 PRE-EXISTING unrelated failure (confirmed identical on an unmodified `9868d3bb` worktree, not caused by this change) | Not run this slice | PENDING -- no isolated micro-bench of the new cached attention mixer exists yet | Ollama/llama.cpp and MLX decode tok/s captured (Baseline step) but **GATED**: correctness failed before a compare verdict could be drawn (see Δ) | `bench_local`, real gemma4 26B-A4B blob, metal backend, flag-off (cacheless, oracle) and flag-on (cached) binaries both run end to end; flag-on output is WRONG (see Δ) | Reuse-first: no new type/trait minted; extended the existing `lfm2_forward_program_with_experts` machinery (shared `build_attention_layer_resources`, shared `append_lfm2_layer_ffn`) rather than forking | Not evaluated this slice -- correctness gate stops before an opt pass is meaningful | Decode step is DESIGNED O(1) in prior sequence length (merged kv-cache leaves grown once per step, no re-prefill) but this is UNVERIFIED as correct -- see Δ | `gemma4-kv-cache` cargo feature on `proxima-model-interop`, default-off; no matching feature added to `proxima-tensor` (new spec module compiles unconditionally there, matching every sibling `*_cached` module's own convention) | Ollama `batiai/gemma4-26b:latest`: 56.62 tok/s decode. MLX `mlx-community/gemma-4-26b-a4b-it-4bit`: 67.667 tok/s decode. Both measured (Baseline step) but **not usable as a verdict** because the cached arm they'd be compared against is incorrect | **FAILED -- token_identical=false.** Cacheless oracle (flag off) reproduces its own prior baseline byte-for-byte: ids `[818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106, 106, 45518, 107, 45518, 107, 101, 818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]`, text `"The capital of France is **Paris**...."`. Cached (flag on) diverges at generated-token index 0 (818 vs 236772) and collapses into a 4-token repeating cycle `[236772, 236771, 236771, 236770]`, decoding to `"-001-001-001..."`. Raw cached ttnt (93.538 ms/token, ~44x faster than cacheless 4152.904 ms/token) is **not reported as a performance result** -- it describes how fast the wrong tokens were produced | Component spans two commits on this branch: `9db00f837` (this slice's implementation, compiles clean, passes the full CPU spec/bind suite) and a not-yet-landed root-cause fix for the correctness failure this log records. See Investigate/Blockers/Fix-plan below for the seven architectural gaps that had to be closed to get this far, and the open item for what's still wrong. |

**Incumbent design point(s):**
- **llama.cpp / Ollama** (`batiai/gemma4-26b:latest`) -- decode one token at a
  time against a resident KV cache; their headline metric is per-token
  `eval_duration`/`eval_count` throughput. Measured 56.62 tok/s.
- **MLX** (`mlx-community/gemma-4-26b-a4b-it-4bit`) -- Apple Silicon unified
  memory, KV-cached autoregressive decode; `mlx_lm.generate`'s own reported
  metric is `tokens-per-sec`. Measured 67.667 tok/s. Caveat: different 4-bit
  quantization than the local GGUF blob and a different sampling/chat-template
  path (`--ignore-chat-template`), so this is an honest home-turf throughput
  number, not a token-identity comparison.

**Tier evidence:** `cargo check -p proxima-model-interop --features std,metal,gemma4-kv-cache` and the same command with `--all-targets` both exit 0 (compiles the new cached gemma4 bind branch, its tests, and its examples). Flag-off equivalents also exit 0 (cacheless branch untouched). `cargo check -p proxima-tensor --features std,config` (+`--all-targets`) exits 0 -- the new `lfm2_single_range_cached.rs` module compiles unconditionally in that crate, matching the existing `single_range_moe_cached.rs`/`mistral_forward_cached.rs` convention (no crate-level feature gate needed there).

**Test N:** `proxima-model-interop` gate for this slice: **261 passed, 0 failed** (`cargo nextest run -p proxima-model-interop --features std,metal`, log at `/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/9049d06b-8620-4a97-8fec-5659655eee9d/scratchpad/nextest_gate.log`). `proxima-tensor` CPU spec/bind suite: 669/670 (1 pre-existing unrelated failure, `bind::tests::single_range_cached_attention_fuses_one_step_per_layer_on_the_real_openchat_shape`, reproduced on the unmodified base commit).

**Opt-sweep findings:** Not applicable yet -- the correctness gate failed before any tuning pass would be meaningful. The one structural choice made (extracting `build_attention_layer_resources` and `append_lfm2_layer_ffn` as shared helpers between the prefill and cached builders, rather than duplicating them) was verified behavior-preserving on the prefill side by the full `proxima-tensor` suite staying at 669/670 before and after.

**SIMD/SM/no-dyn pass:** Not evaluated -- this is graph-spec composition (`Vec<Op>` construction), not a hand-rolled hot loop; no dynamic dispatch was introduced (no new trait objects, no `Box<dyn ..>`).

**O(1):** Designed as O(1) per decode step in prior sequence length (`kv_cache.{layer}.k_even/k_odd/v` merged-cache leaves grown once per step via the existing `cached_len` contract, same shape as qwen35moe/mistral). **This is unverified as correct** -- the correctness failure below means the actual per-step read is producing wrong values, so the O(1) claim describes the intended data-flow shape, not a validated property of a working cache.

**Internal-primitive audit:** No new type or trait was minted. Every knob threaded (`LayerAttentionConfig`, `LayerFfnConfig`, `ValueSource`, `AttentionScoreScale`, `RopePairing`, `FfnCombination`) already existed on the prefill (`lfm2_forward_program_with_experts`) side; the new file composes them into a cached-attention builder the same shape as the existing `single_range_moe_cached.rs`/`mistral_forward_cached.rs` siblings. `find_or_insert`/`AttentionLayerResources` were promoted `private -> pub(crate)` (visibility only) so the extracted resource pre-pass could be shared, not to host a new abstraction.

**Tunable axes:** None new -- no magic numbers were introduced; the change composes existing per-layer config types, all of which already resolve through the existing gguf-metadata-driven bind path, not hardcoded constants.

**Re-prove command:**
```text
cd /Users/brianbruggeman/repos/slot-0/proxima-kv-cache && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-kv-cache/target-kv cargo nextest run -p proxima-model-interop --features std,metal
```
(gate above; the GPU correctness comparison additionally requires the model
gate lock and both `bench_local` binaries described in the Verify+Bench
evidence below -- not re-run as part of this log's own gate command since it
needs the gemma4 blob and exclusive GPU access.)

### Investigate -- why gemma4 is cacheless today

`proxima-model-interop/src/gemma4/bind.rs:607-622` calls
`lfm2_forward_program_with_experts` (the prefill-only engine,
`proxima-tensor/src/spec/attention_forward.rs:1016`) and leaves
`layer_roots: Vec::new()` on the returned `BoundProgram`
(`bind.rs:649`). With no cache leaves for the generic decode loop
(`proxima-model-interop/src/generate/decode.rs`) to grow,
`cached_len` never advances past 0 and the full growing token sequence is
re-bound every step -- an O(n^2) re-prefill, not an explicit
`if architecture == "gemma4"` branch.

### Blockers closed by this slice (all seven, at the spec/graph level)

1. **Sliding-window mask over a merged cache** -- `causal_mask_merged` took no
   `window` parameter; closed by a new `causal_mask_merged_windowed`
   composing `causal_mask_merged`'s future-check with
   `causal_mask_windowed`'s too-old check (delegates to
   `causal_mask_merged` when unwindowed).
2. **Dual RoPE tables** (freq_base 1e6 full / 1e4 SWA) -- the cached engines
   declared exactly one `rope_cos`/`rope_sin` pair per program; closed by
   extracting the per-attention-config resource pre-pass
   (`build_attention_layer_resources`) so `RopeTableSel` (already used by the
   prefill engine) is available to the cached builder for free.
3. **Shared-KV value source** (gemma4 has no `attn_v.weight`,
   `ValueSourceKind::SharedWithKey`) -- neither cached engine took a
   `wv: NodeId`; closed by reusing the existing `ValueSource` type in the new
   attention mixer.
4. **`AttentionScoreScale::Unscaled`** (gemma4 hard-codes `scaling=1.0`) --
   both cached engines unconditionally computed `1/sqrt(head_dim)`; closed by
   threading the existing `AttentionScoreScale` config through the shared
   resource pre-pass.
5. **`value_norm`** (per-kv-head RMSNorm on V post-projection) -- absent from
   both cached engines; closed by reusing the existing
   `rmsnorm_per_head_no_scale` primitive in the new mixer.
6. **`FfnCombination::ParallelDenseMoe`** (gemma4's dense-SwiGLU + routed-MoE
   parallel-sum FFN) -- the cached engines only supported the prefill
   engine's exclusive-OR dense-XOR-MoE switch; closed by extracting the FFN
   match into a shared `append_lfm2_layer_ffn`, called identically by both
   engines, verified behavior-preserving on the prefill side by the
   `proxima-tensor` suite staying at 669/670.
7. **`final_logit_softcapping=30`** -- neither cached forward-program took a
   `logit_softcap` parameter; closed by copying the existing tanh-softcap
   composition from the prefill tail into the new cached program.

Not a blocker: 128-expert/8-used MoE routing itself -- `expert_count`/
`expert_used_count` and `append_mistral_cached_moe_layer` already exist and
are exercised by qwen35moe's own cached decode path.

### Correctness verification -- FAILED

**Repro (all under `cd /Users/brianbruggeman/repos/slot-0/proxima-kv-cache &&`, branch `perf/gemma4-kv-cache` at `9db00f837ae5c96b22b1229638b01dda7c78743b`, `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-kv-cache/target-kv` on every cargo call):**

Two distinctly-fingerprinted release binaries were built and run back to
back under one hold of the model gate lock
(`/Users/brianbruggeman/repos/slot-0/.model.lock`):

- flag-off (`--features std,metal`, cacheless, the oracle):
  `ids=[818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106, 106, 45518, 107, 45518, 107, 101, 818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]`,
  text `"The capital of France is **Paris**.thought\nthought\n<channel|>The capital of France is **Paris**."` --
  byte-for-byte identical to the prior Baseline step's run, confirming the
  oracle is deterministic across sessions.
- flag-on (`--features std,metal,gemma4-kv-cache`, cached):
  `ids=[236772, 236771, 236771, 236770, 236772, 236771, 236771, 236770, ...]` --
  a 4-token repeating cycle from the very first generated token, decoding to
  `"-001-001-001..."`.

**Divergence point: generated-token index 0.** This is not late-decode drift
-- the cached path is wrong from the first step, and the fixed low-entropy
4-cycle (not a near-miss logit) is consistent with the cache buffer being
read as constant/zero/misaligned rather than holding real per-layer K/V
history -- e.g. a merged-cache read at the wrong offset, a RoPE-table
mismatch between the prefill-time and cached-decode mixers, or the cache
tensor never being populated from the prefill step.

Per the task's own binding rule ("if they differ, the cache is wrong ...
STOP, do not report a perf win on incorrect output"), no speedup or
scorecard is computed. The raw cached-path ttnt (93.538 ms/token vs the
oracle's 4152.904 ms/token, ~44x) is recorded above **only as evidence of
what was measured**, explicitly disqualified as a performance result.

## Bench table (informational only -- gated, no verdict)

| Arm | design-favors | Decode tok/s | Status |
|---|---|---|---|
| proxima cacheless (ours, oracle) | neutral | 0.2408-0.2415 | Correct, reproduced twice; ~234-280x slower than the incumbents |
| proxima cached (`gemma4-kv-cache`, ours) | ours | 10.69 (raw, invalid) | **INCORRECT OUTPUT -- disqualified, not a result** |
| Ollama / llama.cpp (`batiai/gemma4-26b:latest`) | incumbent | 56.62 | Measured, correct (their own decode) |
| MLX (`mlx-community/gemma-4-26b-a4b-it-4bit`) | incumbent | 67.667 | Measured, correct (their own decode, different quant/sampling) |

### Changelog

| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-09-18 | Baseline: measured cacheless gemma4 decode (oracle), Ollama, and MLX home-turf decode throughput; no code changed | cacheless 0.2415 tok/s; Ollama 56.62 tok/s; MLX 67.667 tok/s | single run each, no repeats yet | local Metal host, model gate held for cacheless+Ollama runs |
| 2026-09-18 | Implement (`9db00f837`): landed `gemma4-kv-cache` feature -- new `causal_mask_merged_windowed` primitive, new `lfm2_single_range_cached.rs` cached forward-program builder, `Gemma4Arch::bind()` gains a flag-gated branch populating real `CachedLayerRoots`; all seven architectural blockers closed at the spec level | compiles clean with and without the flag; CPU spec/bind suite 669/670 (1 pre-existing unrelated failure); no GPU/model-load run this step (explicitly out of scope) | deterministic build+test counts | local build host, no GPU run |
| 2026-09-18 | Verify+Bench: ran both flag-off and flag-on `bench_local` binaries against the real gemma4 blob under the model gate | **token_identical=false** -- cached path diverges at generated-token index 0, collapses to a 4-token repeating cycle; cacheless oracle reproduced its own baseline byte-for-byte | one comparison run each; cacheless reproduced across two independent sessions | local Metal host, model gate held for both runs (0s wait, held ~5.6s total for the cached run) |
| 2026-09-18 | This log: ran the required crate test gate and recorded the discipline log; no code changed | `cargo nextest run -p proxima-model-interop --features std,metal`: 261 passed, 0 failed, 59 skipped | deterministic | local build host, no GPU run |

**Honest read:** the `gemma4-kv-cache` feature compiles clean on both sides
of its flag and closes all seven architectural gaps needed to *express*
gemma4's dual-RoPE, sliding-window, shared-KV-value, unscaled-score,
value-norm, parallel-dense-MoE-FFN, softcapped-logit shape inside the
existing cached-attention spec family, without minting any new type -- but
the cached decode does **not** meet the correctness bar: it is not
token-identical to the proxima cacheless oracle, diverging at the first
generated token and collapsing into a degenerate 4-token cycle. Per the
project's own correctness-first rule, this DISQUALIFIES any performance
claim: cached gemma4 decode does **not** meet or beat llama.cpp/Ollama
(56.62 tok/s) or MLX (67.667 tok/s) -- it produces wrong output, so no
speed comparison against either incumbent stands. The 44x raw speed
difference over the cacheless oracle is real as a number but worthless as a
result until the cache reads correct K/V history.

**Implication:** the next slice is a root-cause dig into the cached
attention mixer's numerical output, starting from
`proxima-tensor/src/spec/lfm2_single_range_cached.rs` and the
`CachedLayerRoots` population/consumption path in
`proxima-model-interop/src/gemma4/bind.rs:604-687`, specifically whether
gemma4's dual-RoPE per-layer leaf naming (`rope_cos`/`rope_sin` vs a
SWA-specific pair) is actually bound into the cache-consuming graph the way
the prefill-time mixer binds it. The natural first artifact is a
CPU-synthetic differential test extending the existing
`gemma4_synthetic_parity` harness (`proxima-tensor/src/spec/tests.rs`) to
compare `append_lfm2_single_range_cached_attention`'s output against the
prefill mixer's own reference at `cached_len=0` (degenerate case) and
`cached_len>0` -- this is the gap that let the spec-level test suite pass
(669/670) while the real-model decode was wrong, and closing it is required
before any further GPU-level correctness or performance claim.
