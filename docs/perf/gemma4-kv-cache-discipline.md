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

## C3 -- 24-token VerifyBench (this slice): token-identical confirmed, still short of the bar

**Correctness (binding oracle check), 24 tokens greedy, real gemma4 26B-A4B
blob, two distinctly-fingerprinted release `bench_local` binaries
(`md5 76ca0c62...` flag-off, `md5 2b3c8968...` flag-on, matching the fix
commit's own reported fingerprints), templated France prompt:**
**token-identical at all 24 positions**, ids
`[818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106, 106, 45518, 107, 45518,
107, 101, 818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]`, text
`"The capital of France is **Paris**.thought\nthought\n<channel|>The capital
of France is **Paris**."` on both arms, matching the C1 Baseline step's
oracle byte-for-byte. This is a stronger correctness result than C2's
8-token confirm: the flag-on path now holds token identity over 3x the
sequence length, across the SWA-history boundary.

**Performance (only a valid claim because correctness above holds):**
cached decode measured **9.368 tok/s** (`ttnt_ms=106.749`, `ttft_ms=3408.376`,
`total_ms=5881.558` for 20 prompt tokens + 24 generated). Cacheless
(this same clean run) measured **0.2166 tok/s** (`ttnt_ms=4617.054`),
consistent with the task's stated 0.24 tok/s baseline -- cached is
**43.25x faster** than this run's own cacheless arm and **39.03x faster**
than the stated baseline.

**Meets-or-beats verdict: NO.** 9.368 tok/s vs Ollama/llama.cpp's 56.62
tok/s is **0.1654x** -- Ollama is 6.05x faster. 9.368 tok/s vs MLX's
67.667 tok/s is **0.1384x** -- MLX is 7.22x faster. Correct output, real
~39-43x win over the cacheless path, but the decode hot path remains
roughly one order of magnitude below both incumbents.

**Environment finding (not a code finding):** across 10 total attempts in
this verification, 8 produced degenerate all-zero-id output on BOTH arms
symmetrically (including a full 5/5-failed retry loop per binary), coincident
with two concurrently-resident Ollama `llama-server` processes consuming
~30GB RSS and driving load average above 12 -- the known
`project_ollama_loader_is_the_judge_hook.md` contention pattern (its
judge hook curls `/api/generate` on every Stop). One partially-contended
attempt showed a correct prefix collapsing mid-decode to zero, evidence the
corruption is a mid-run Metal-residency failure under memory pressure (the
13GB mmap'd weights losing residency and reading back as zero), not a logic
bug in either arm -- symmetric across flag-off and flag-on, and the fix
diff (`d9f20f107`) touches only `#[cfg(feature = "gemma4-kv-cache")]` code,
so it structurally cannot be the cause of a flag-off failure. The
token-identical and 9.368 tok/s numbers above come from one clean
back-to-back pair (cacheless immediately followed by cached, no intervening
tool call) captured once host load dropped to ~30% CPU and both
`llama-server` RSS footprints stopped growing.

**Gate:** `cargo nextest run -p proxima-model-interop --features std,metal`
re-run this slice: **261 tests run: 261 passed, 0 failed, 59 skipped**, exit
0 (log at
`/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/9049d06b-8620-4a97-8fec-5659655eee9d/scratchpad/nextest_gate2.log`).
No code changed this slice -- doc-only update.

**Updated gate cells (supersede the C1 row's Compare-bench/Home-turf/
Correctness cells, current truth as of this slice):**

| Cell | C1 (superseded) | Current (C3) |
|---|---|---|
| Correctness | FAILED -- token_identical=false, diverged at index 0 | **PASSED -- token_identical=true across 24/24 positions**, two independent confirms (8-token in C2, 24-token here) |
| Compare-bench | Gated -- no verdict possible, output was wrong | **Computed and NEGATIVE**: cached 9.368 tok/s = 0.1654x Ollama (56.62), 0.1384x MLX (67.667) -- correct but 6.05-7.22x short of the bar |
| Home-turf | Ollama/MLX numbers measured but unusable (cached arm disqualified) | Same Ollama (56.62) / MLX (67.667) numbers now usable as the denominator of an actual (failing) compare-bench verdict |

## Bench table (current)

| Arm | design-favors | Decode tok/s | Status |
|---|---|---|---|
| proxima cacheless (ours, oracle) | neutral | 0.2166-0.2415 | Correct, reproduced across 3 sessions; ~234-280x slower than the incumbents |
| proxima cached (`gemma4-kv-cache`, ours) | ours | **9.368** | **Correct (token-identical, 24/24), still 6.05x below Ollama and 7.22x below MLX -- does not meet or beat** |
| Ollama / llama.cpp (`batiai/gemma4-26b:latest`) | incumbent | 56.62 | Measured, correct (their own decode) |
| MLX (`mlx-community/gemma-4-26b-a4b-it-4bit`) | incumbent | 67.667 | Measured, correct (their own decode, different quant/sampling) |

### Changelog

| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-09-18 | Baseline: measured cacheless gemma4 decode (oracle), Ollama, and MLX home-turf decode throughput; no code changed | cacheless 0.2415 tok/s; Ollama 56.62 tok/s; MLX 67.667 tok/s | single run each, no repeats yet | local Metal host, model gate held for cacheless+Ollama runs |
| 2026-09-18 | Implement (`9db00f837`): landed `gemma4-kv-cache` feature -- new `causal_mask_merged_windowed` primitive, new `lfm2_single_range_cached.rs` cached forward-program builder, `Gemma4Arch::bind()` gains a flag-gated branch populating real `CachedLayerRoots`; all seven architectural blockers closed at the spec level | compiles clean with and without the flag; CPU spec/bind suite 669/670 (1 pre-existing unrelated failure); no GPU/model-load run this step (explicitly out of scope) | deterministic build+test counts | local build host, no GPU run |
| 2026-09-18 | Verify+Bench: ran both flag-off and flag-on `bench_local` binaries against the real gemma4 blob under the model gate | **token_identical=false** -- cached path diverges at generated-token index 0, collapses to a 4-token repeating cycle; cacheless oracle reproduced its own baseline byte-for-byte | one comparison run each; cacheless reproduced across two independent sessions | local Metal host, model gate held for both runs (0s wait, held ~5.6s total for the cached run) |
| 2026-09-18 | This log: ran the required crate test gate and recorded the discipline log; no code changed | `cargo nextest run -p proxima-model-interop --features std,metal`: 261 passed, 0 failed, 59 skipped | deterministic | local build host, no GPU run |
| 2026-09-18 | Fix (`d9f20f107`) + docs (`d23af282b`): root-caused the zero-cache read (single-range engine never folds its own call's new K/V before scoring, generic decode loop never pre-folds), landed `append_lfm2_two_range_cached_attention` generalizing the proven qwen35moe two-block online-softmax combine; 8-token real-checkpoint confirm **token_identical=true** | cacheless 3160.764 ms/token vs cached 107.383 ms/token (~29x, now valid since output is correct); superseded the prior 44x-but-wrong number | `proxima-tensor` 672/673 (1 pre-existing unrelated failure); `proxima-model-interop` 261/261 | local Metal host, model gate held |
| 2026-09-18 | C3 VerifyBench: 24-token real-checkpoint run, two fresh-fingerprint binaries, clean back-to-back pair after host contention (2 concurrent Ollama `llama-server` processes) cleared | **token_identical=true (24/24)**; cached 9.368 tok/s vs cacheless 0.2166 tok/s (43.25x); vs Ollama 56.62 tok/s = 0.1654x (6.05x short); vs MLX 67.667 tok/s = 0.1384x (7.22x short) -- **meets-or-beats: NO** | 1 clean comparison pair; 8/10 total attempts degenerate under host contention (documented, ruled out as a logic bug, symmetric across both arms) | local Metal host, model gate held; two Ollama judge-hook `llama-server` processes contending for ~30GB RSS during 8 of 10 attempts |
| 2026-09-18 | Gate re-run for this doc-only slice, no code changed | `cargo nextest run -p proxima-model-interop --features std,metal`: 261 passed, 0 failed, 59 skipped, exit 0 | deterministic | local build host, no GPU run |

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

## C2 -- root cause and fix

**Root cause (proven, not inferred):** `lfm2_single_range_cached.rs`'s own
module doc names the shape correctly -- `append_lfm2_single_range_cached_attention`
scores ONLY against the merged `kv_cache.{layer}.*` leaves, and explicitly
never reads its own freshly-rotated `rotated_k_new`/`v_new` for its own
call's score ("a query never attends a key that does not exist yet"). That
contract is correct ONLY when a caller pre-folds this call's OWN new
positions into the merged cache leaves before evaluating (proven by the
existing `single_range_vs_two_range_decode` test's own "folded in by hand
the way write-placement would fold them in at runtime" pattern,
`proxima-tensor/src/spec/tests.rs`). `proxima-model-interop`'s GENERIC
(non-placed) decode loop (`push_kv_named_blocks`/`KvPadScratch::fill`,
`proxima-model-interop/src/generate/residency_caches.rs:149-170`) never does
that pre-fold: `LayerCache` starts empty and only grows AFTER a step
evaluates. Only the Metal `metal-output-placement` `run_decode_loop_placed_kv`
path (dense-only, MoE excluded, `load_model.rs:1260`) achieves the
write-then-read ordering, via a fused kernel writing new K/V into the same
device buffer a plan reads back. `Gemma4Arch::bind` wired the single-range
engine into the GENERIC loop -- so at `cached_len=0`, `new_count=` the whole
prompt (`single_position_step: false`), literally every attention layer's
merged-cache leaf was entirely zero on the very first (and every
subsequent) step, at EVERY layer, collapsing the whole network to residual
+ FFN only.

**Execution evidence** (`proxima-tensor/src/spec/tests.rs`,
`gemma4_synthetic_parity` module, synthetic 2-layer/SWA+full gemma4-shaped
fixture): feeding the single-range engine the SAME all-zero merged cache
the real decode loop provides at `cached_len=0` diverges from the prefill
oracle by `max_abs_diff=0.394` (`single_range_cached_gemma4_diverges_on_zero_cache_matches_when_self_range_is_folded`);
hand-folding this call's own new K/V into the SAME leaves before evaluating
matches the oracle to float noise (`2.98e-8`) -- proving the mechanism
directly, node values in hand.

**Fix** (`proxima-tensor/src/spec/lfm2_single_range_cached.rs`,
`proxima-model-interop/src/gemma4/bind.rs`): a new
`append_lfm2_two_range_cached_attention` +
`lfm2_two_range_cached_forward_program_with_experts`, generalizing
`single_range_moe_cached::append_mistral_cached_moe_layer`'s own
already-proven two-block online-softmax combine (reuse-first, no new Op
variant, no new type) with gemma4's existing knobs (`ValueSource`,
`RopePairing`, `value_norm`, dual RoPE tables). The cache block scores ONLY
genuine history -- a new `causal_mask_cached_windowed` excludes every row at
or past `cached_len` unconditionally (`is_padding`), composed with the same
too-old windowed check `causal_mask_merged_windowed` already established for
SWA layers. The local block scores this call's own new positions against
its own in-graph `rotated_k_new_even`/`rotated_k_new_odd`/`v_new`, never
round-tripped through a cache leaf -- so `kv_cache.{layer}.*` may be fed
EXACTLY what the existing growing-cache decode loop already provides (real
`[0, cached_len)`, zero padding past it), with **no decode-loop change**.
`Gemma4Arch::bind` now calls the two-range builder in place of the
single-range one.

**Fix verification** (same synthetic fixture, `proxima-tensor` CPU suite):
`two_range_cached_gemma4_matches_prefill_oracle_with_decode_loop_realistic_zero_padding`
(cached_len=0, the exact zero-padded cache the real loop feeds, no pre-fold)
matches the oracle to `3.9e-8`;
`two_range_cached_gemma4_two_step_decode_matches_one_shot_prefill_oracle` (a
genuine two-step decode, `cached_len=0` then `cached_len=2` with real folded
SWA-windowed history) matches to `4.9e-8`. `cargo check -p proxima-model-interop
--features std,metal,gemma4-kv-cache` (+ `--all-targets`) and the flag-off
equivalent both exit 0. `cargo nextest run -p proxima-tensor --features
std,config,test-support,cached-attention-streaming,kv-capacity-bucket`:
672/673 (the same PRE-EXISTING `single_range_cached_attention_fuses_...`
875-vs-939 failure, confirmed unrelated). `cargo nextest run -p
proxima-model-interop --features std,metal`: 261/261, 0 failed, 59 skipped.

**Real-checkpoint confirm** (model gate held, two distinctly-fingerprinted
release `bench_local` binaries, `--features std,metal` vs `--features
std,metal,gemma4-kv-cache`, templated France prompt, 8 tokens, greedy):
**token-identical.** Both arms produce `ids=[818, 5279, 529, 7001, 563,
5213, 50429, 84750]`, text `"The capital of France is **Paris**."`.
`ttnt_ms`: cacheless 3160.764 ms/token, cached 107.383 ms/token (~29x) --
now a VALID performance observation, since the output is correct (the
44x-faster-but-wrong number from the prior slice is superseded, not
reused). A full 24-token VerifyBench run and a meets-or-beats scoreboard
against Ollama (56.62 tok/s) / MLX (67.667 tok/s) is the next slice's own
gate, not claimed here.

**Update (this slice, C3): the 24-token VerifyBench ran** -- see the "C3 --
24-token VerifyBench" section above for the full evidence. Result:
token-identical confirmed at 24/24 positions; cached decode measures
9.368 tok/s, which does **not** meet or beat either incumbent (0.1654x
Ollama, 0.1384x MLX). This closes the item this note deferred; it is not
still open.

**Current honest read (supersedes the C1 honest read above): the fix is
real and verified twice at two different sequence lengths (8-token and
24-token real-checkpoint runs, both token-identical to the cacheless
oracle) -- gemma4 cached decode is now CORRECT, not still-incorrect. It is
also NOT yet a meets-or-beats result: 9.368 tok/s is 6.05x below Ollama's
56.62 tok/s and 7.22x below MLX's 67.667 tok/s. Status: correct-but-slower.
The ~39-43x win over the cacheless path is real progress and the right
foundation (O(1)-per-step decode, no re-prefill), but the initiative's
stated bar -- meet or beat both incumbents -- is not met, and the component
stays behind its default-off `gemma4-kv-cache` flag. The next divergence to
chase for further speedup, not correctness, is decode-step dispatch count
and kernel occupancy on the two-range cached mixer (unmeasured this slice --
no micro-bench or profile was run to say where the remaining ~6-7x lives).**
