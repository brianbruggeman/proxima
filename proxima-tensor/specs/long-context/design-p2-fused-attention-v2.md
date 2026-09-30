# P2 v2: L-independent fused attention (Q-tiled, one canonical fold)

status: DESIGN, not audited. Supersedes `design-p2-fused-attention.md` (v1, rejected 2026-09-29).

## v2 critique outcome (proxima-critic, 2026-09-29): NOT ADMITTED

Blockers:
1. **Irreversible steps run before evidence.** Candidate B (a measured -1.0 to -1.2
   ms/tok, byte-identical over 105/105 cells) is deleted in P2-14 BEFORE the decode
   measurement (P2-16), and no decode outcome can fail. The online kernel family has a
   recorded gemma4 decode REGRESSION: wall 23.57 to 33.17 ms/tok, because parallelism
   drops from the unfused AV's 2048 threadgroups to one per layer. v2 omits it. gemma4
   has Hkv=1, so C=1 decode is a single threadgroup unless M2 split-K fires, and M2's
   fill threshold is undefined.
2. **G1/G2 do not make llama.cpp the oracle for what this change can break.**
   - G1 feeds proxima's own q/k/v and mask.
   - G2 at layer 4 has an unmeasured shared-error floor.
   - 5e-4 is llama's f16-KV flash-attn bar, about 9 orders looser than f32 reorder
     spread.
   - The llama gate cannot fail the NPV=1 regression v2 itself found.
   - G2 (a gemma4 full-model llama run) is unrun; the prebuilt llama tools predate gemma4.
3. **The R1 evidence is tautological.** `fold_row` has no row-count input.
   - The pipeline cache key lives in `omega/src/identity.rs:524-595` and bakes
     `new_upper`/`context_chunks`; v2 never names that file.
   - T2 passes with zero accepted drafts.
   - C=32 (the default ubatch) is not in T1.

Major:
- **B1 cap wrong.** B1's "28/49 rows differ" uses chunk cap 4, but real full-layer cap
  is 1 (`signature_tokens_prelude.rs:1378-1382`).
- **Owner bar retired unannounced.** v2 retires the OWNER DECISION of 2026-09-21
  ("byte equivalence stays binding; fusion NOT promoted until it matches unfused Metal
  bytes") without surfacing it.
- **Other models ungated under default policy.** qwen3.6 and dense are ungated, including
  the dense single-range path of the 131072 YaRN target. The parity tests P2-14 would
  delete are not ported, and neither are the WGSL/CUDA matches.
- **grid2d unguarded.** The grid2d dispatch branch returns before any width check, so the
  typed width error misses the flash path.
- **Ollama may not be llama.cpp.** Ollama gates remain even though Ollama's gemma4 engine
  identity is unestablished.
- **SPEC drift.** SPEC R12L/AC12L is not reconciled, and AC23 contradicts P2-14's ABSENT
  list.
- **NPV/Ls cost unchecked.** NPV/Ls register feasibility and merge cost are unverified,
  and Safe math costs 179 vs 241 GB/s on the matvec.
- **Slices not minimal.**

Open decision for the owner: see TASKS.md "P2 decision". v3 must not start until it is
answered.
author: proxima-architect pass, 2026-09-29. Read-only design; the only writes are this file and scratch experiments under `E/` (C and awk; two of the C programs link the external llama.cpp build's ggml on its CPU backend for seconds on tiny synthetic inputs; no GPU, no cargo).

## 0. Citation and provenance conventions

- `T:path:line` is the code tree `/private/tmp/long_ctx_main2` (non-git export: main 73fd1cbd plus this spec's work).
  Every code cite below is `T:` unless prefixed otherwise. Cites were opened this session, not inferred.
- `S:path:line` is the root checkout `/Users/brianbruggeman/repos/slot-0/proxima` (spec files only). `S:.../SPEC.md` is
  byte-identical to the `T:` copy (checked with `diff`). The speculative spec exists only in `T:`.
- `E/` is `<session>/long_ctx_backups/p2_experiments/v2/`. Every `[M]` number here has its source file and output file there
  (`fold_contract.c/.out`, `err_stat.c/.out`, `ls_sweep.out`, `npv_sweep.out`, `sub_sweep.out`, `support_exact.c/.out`,
  `exact.awk/.out`, `llama_oracle_demo.c/.out`, `llama_npv.out`, `llama_worked_example.c/.out`). The `fold_contract`, `err_stat` and `support_exact`
  programs are scalar-f32 C simulations of the SPECIFIED arithmetic (`-ffp-contract=off`, explicit `fmaf`); the two `llama_*` programs run the external
  llama.cpp build (CPU backend, commit f1ea20621) on synthetic bytes. None is a measurement of the Metal kernel or of the real payload.
- Tags: [R] read from source, [M] measured (C simulation, stated), [D] derived from other numbers (never a mechanism basis),
  [A] assumed (must be measured before it is acted on), [P] a number recorded by a prior session and read from a memory note, not
  re-measured here (pointer to the record, not an artifact I opened).
- Oracle rule (owner, 2026-09-29, binding): llama.cpp is the correctness oracle, via an EXTERNAL build (never a runtime, crate, vendored code or
  dependency of proxima); proxima-internal references (CPU path, f64 definition, fused-vs-unfused, resumed-vs-fresh) are internal consistency checks and
  labelled so; Ollama is the performance incumbent only. Section 4 applies it. The kernel is derived from the FlashAttention papers (Section 3.1); llama.cpp
  source was read only for facts about the oracle (its tolerance and tensor names), nothing is ported. The existing kernel's doc comments cite llama.cpp
  (21 mentions in `T:omega/src/msl/cached_attention_render.rs`); the arms this design deletes take those comments with them, and the arm it keeps is
  re-documented (slice P2-14).

## 1. Decision, and the one contested call

Shape: ONE op (`BoundOpKind::CachedAttention`), lowered by the `NumericPolicy` that already selects lowerings. Under a policy that
withholds `reassociation` the existing strictly-sequential per-key body stays (the bit-exact lowering). Under a policy that grants it
(the shipped default is `llama_relaxed`, `T:proxima-model-interop/src/serving.rs:992`) ONE new tiled lowering replaces the three
reassociating arms that exist today (chunked, block-staged, split-plus-merge). That lowering is FlashAttention-2's online-softmax tile
update, Q-tiled through threadgroup memory, with a canonical fold whose association order is a function of the absolute key index
only. Decode, speculative verify and prefill chunks run that one pipeline. Candidate B (`CachedSoftmaxWeights`) is retired.
Recognizer outcomes are values (`Fused | Kept | Declined(reason)`), and the existing `ServingConfig.flash_attention` bool becomes the
"demand" bit (no new config enum). Full-attention anchors are routed by policy now; windowed anchors are `Kept` until a measured
decode-throughput rule flips them, and the flip path (ring phase alignment, gates) is designed here, not deferred (Section 5.1).

Why decode throughput is in scope. The owner's summary is "the memory is excellent, tok/s is not". The same unfused attention chain
that scales memory with L is also the largest identified block of decode dispatches: 15 BoundOps per attention layer, 525 of 1661
per decode token [P, from a census recorded on this program by a prior session]. Section 3.8 counts that population before and after
from the real program, measures decode ms/token against Ollama's `eval_duration`, and names the follow-on chains.

The contested decision: one lowering versus a second tiled lowering bolted onto the gemma4 path only. The gemma4-only option has the
smaller blast radius and the worse design. It leaves two reassociating implementations of one op selected by which recognizer arm
matched (the "two half-primitives" debt, principle 1), and it leaves dense speculative verify (spec R2, not yet built:
`T:proxima-model-interop/src/architecture.rs:318-332`) with decode and verify kernels of different arithmetic. Decision: one
lowering. If a decode non-regression gate fails for a model family the follow-up is to tune the decode-shaped mode, never to keep the
old arm for that family (P2-16e).

## 2. Finding to answer map

| v1 finding | answer | where |
|---|---|---|
| B1 speculative byte-identity | Candidate B retired; decode, verify and prefill share one pipeline whose fold is a function of key index only. Simulated: 0 of 49 rows differ; the legacy extent-keyed partition differs on 28 of 49 at short context. Kernel gate and model gate are nextest tests | 3.2, 3.3, AC25, AC26 |
| B2 oracle: wrong baseline, underived 2x/4x, no real payload, "Not met" bar | oracle is llama.cpp (external build, vendored fixtures), on the real payload dumped from proxima's unfused Metal path and on a real gemma4 model run; statistic (NMSE, llama's own metric) and threshold derived from K=64 legitimate reorderings before the kernel exists; principle-14 position rests on the incumbent's own criterion | 4.1 to 4.4 |
| B3 routing misses the real gate | the gate at `:1241` and every branch named; the Candidate B arm is deleted, not bypassed | 5.2 |
| M1 C-invariance vs chunked partition | partition is by absolute key index (`Ls`), never by dispatch extent | 3.2 |
| M2 no admission site, declines discarded | outcomes are values; `flash_attention` is the demand bit with a typed refusal | 5.3 |
| M3 blast radius, PASS-only baseline | per-model table; baseline records PASS/FAIL/SKIP | 6.1, 6.2 |
| M4 no performance model | FA-2 Q-tiling; tile sizes derived from the 32768 B budget; roofline; home-turf arms (prefill and decode) | 3.4 to 3.8 |
| M5 silent 768-wide clamp | typed error at dispatch; limit read from the compiled pipeline | 5.5 |
| M6 window gate too weak | real-model gate, prompt over 3W, in-window fact vs out-of-window distractor, llama.cpp answers as oracle, with controls | 4.7 |
| M7 poison overflows f16/Q8_0, one-sided | no magnitude poison: exact-count signature (both directions) plus NaN in masked rows | 4.5 |
| M8 S5 ignores FFN | program-wide max at C=512 is the FFN gate/up reduce, 1,610,612,736 [D] | 6.3 |
| M9 P2 not in SPEC | R19 to R28, AC24 to AC37, SPEC.md:131 amended | 7.1 |
| m1 mixed trees | every cite prefixed `T:`/`S:` | 0 |
| m2 2.2e-5 noise figure | not reproduced; retracted | 4.4 |
| m3 scope covers 28 sliding layers | sliding layers `Kept` by default, three reasons, measured flip with the path designed | 5.1 |
| m4 generalization unmeasured | dropped; only what is designed and gated is claimed | 8 |
| owner: tok/s | dispatch census, decode pairs vs Ollama, follow-on chains | 3.8 |

## 3. Shape

### 3.1 Sources

- FlashAttention-2: Dao, arXiv:2307.08691, Algorithm 1 (forward): per query block, iterate key/value blocks, keep the running row max
  `m`, row sum `l` and unnormalised output `O`; rescale by `exp(m_old - m_new)`; divide by `l` once at the end. The
  parallelisation-over-query-blocks and work-partitioning discussion is the basis for row tiles and for the key partition.
  [A] the algorithm line numbers are quoted from memory; slice P2-6 opens the paper and records the exact lines before code.
- FlashAttention: Dao et al., arXiv:2205.14135 (I/O complexity in terms of on-chip memory M).
- Online normaliser: Milakov and Gimelshein, arXiv:1805.02867. Key-split merge (m, l, O form a monoid under merge): the FlashDecoding
  formulation (Dao et al., 2023, an engineering write-up, not a paper) [A].
- No code is ported from llama.cpp. Its source was read for two facts about the oracle only: its own tolerance for this op (`tests/test-backend-ops.cpp:7915-7917`) and the tensor names its graph exposes (`src/llama-graph.cpp:2809-2811`).

### 3.2 The arithmetic contract (the definition of the op under reassociation-admitting policies)

Notation: query vector `q` (head dim `hd`, scale `s`), absolute key index `j` (cached live rows first, then new rows; for a merged
single-range buffer the buffer row), `visible(row, j)` from the bands, tile width `Bk = 32 = SIMD_WIDTH`, partition `Ls` keys (a
multiple of 32), `NPV` partial accumulators. Every multiply-add is an explicit `fma`; masked means SELECTED away, never multiplied
by zero.

1. Score. Lane `l` accumulates `a_l = fma(q[d], k_j[d], a_l)` over `d = l, l+32, ...` ascending (the K row is the concatenation
   `[even plane | odd plane | pass plane]`, width `hd`). The 32 lane partials are reduced by the xor butterfly `16, 8, 4, 2, 1`
   (`simd_shuffle_xor`, one add per stage). `score_j = visible ? butterfly * s : -inf` (a select).
2. Tile `t = floor(j / 32)`. `m_t = max` over the tile. `m' = max(m, m_t)`. `alpha = (m == -inf) ? 0 : exp(m - m')`.
   `p_j = (score_j == -inf) ? 0 : exp(score_j - m')`. `l_t` = butterfly sum of the 32 `p_j`. `l = fma(l, alpha, l_t)`.
3. Value fold, per output dim `d` (owned by lane `d mod 32`): `NPV` interleaved chains
   `part[j mod NPV] = fma(p_j, v_j[d], part[j mod NPV])` over the tile's visible keys in ascending `j`, then combined pairwise
   `part[a] = part[a] + part[a + w]` for `w = NPV/2 .. 1`; `o[d] = fma(o[d], alpha, part[0])`.
4. Block `b = floor(j / Ls)`. A block folds its tiles from the identity state `(m=-inf, l=0, o=0)`.
5. Partition merge, strictly left to right over blocks starting from the identity: `m = max(mA, mB)`,
   `wA = (mA == -inf) ? 0 : exp(mA - m)`, `wB` likewise, `l = fma(lA, wA, lB * wB)`, `o[d] = fma(oA[d], wA, oB[d] * wB)`. The identity is
   an exact two-sided neutral (`wA = 0, wB = 1` gives `fma(0, 0, x*1) = x`), so blocks wholly beyond a row's visible range change nothing.
6. Output `o[d] / l` (`0` when `l == 0`).

What is and is not arithmetic. Arithmetic (changing it changes bits, so it is pinned in the pipeline key and the fixtures): `Bk`, `Ls`,
`NPV`, the lane-to-dim map, the butterfly order, the merge expression. Not arithmetic (free to tune, bit-identical output by
construction): the row tile `Br`, the staging width `Sk`, the number of simdgroups, and the execution mode. The score of a key is
independent of `Sk` because the transposed reduce network performs the same add tree per key (16, 8, 4, 2, 1); addition is commutative
bitwise, so operand order inside a stage cannot matter.

The constants live in `omega/omega-runtime.toml [flash_attention]` with `OMEGA_FLASH_ATTENTION_*` build-time overrides and
`rerun-if-env-changed` (principle 12); they replace `[attention_context_chunks]`, `[attention_block]` and `[attention_splits]`
(`T:omega/omega-runtime.toml:188-269`).

Two execution modes, one arithmetic. Resident (M1): one threadgroup folds all blocks of its (row tile, kv head) in kernel, block-fresh
state then merge (two accumulator sets). Split (M2): threadgroups compute block partials into scratch, a merge kernel folds them left to
right. The planner picks the mode from shape only (a row-tile count below a fill threshold and a scratch size under a budget select M2;
verify at 49 rows stays M1 because its M2 scratch would be 825 MB [D]). [M] `E/fold_contract.out`: resident versus split-merge differ on
0 of 49 rows at base 20 and at base 611, for `Ls` in {128, 256, 512}.

### 3.3 Speculative R1: the argument and the gates

Speculative R1 (`T:proxima-tensor/specs/speculative-decode-llama-parity/SPEC.md:30`; R20 at `:53`; default-on at `:98`) requires verify
rows and 1-row decode to be byte-identical. The code shows how that breaks today. The two-range fused kernel's entry name carries
`q{query_rows}_c{cached}_n{new}` (`T:omega/src/msl/signature_tokens_prelude.rs:340-343`), so decode and verify are different pipelines,
and the chunk count is `context_chunks_for(cached + new)` (`T:cached_attention_render.rs:258-263`, `signature_tokens_prelude.rs:1348-1360`),
a function of the dispatch extent. [M] `E/fold_contract.out` (legacy chunk model, `chunks = clamp(ceil(extent/16), 1, 4)`): at 20 cached
rows, 28 of 49 rows differ between the 1-row dispatch and the 49-row dispatch; at 611 cached rows the chunk count saturates and 0 differ.
The hazard is real but bounded to extents under `4 * keys_per_chunk = 64` keys, plus whatever a differently specialised compile does
(unmeasured). The v1 blocker is sharper: with fusion on, decode would take Candidate B (`T:dead_code_cached_attention.rs:1366`) while
verify would take the online kernel: two arithmetics.

The contract removes the hazard by construction and by test:

- One pipeline. The flash entry name has no `q/c/n` token. Row count, live cached rows and `cached_len` are runtime uniforms. Its cache
  key is identical for the decode BoundOp and the verify BoundOp of one layer; a test asserts key equality.
- The fold is defined over absolute key index (Section 3.2). Identical key set gives identical bits, whatever the dispatch's row count,
  tile row count, mode or partition count. [M] `E/fold_contract.out`: decode versus verify differ on 0 of 49 rows at both bases; changing
  `Ls` changes bits (`ls-changed differing=49` at 611 cached rows), which is why `Ls` is arithmetic and pinned.
- Math mode pinned. The flash pipelines compile with `MathMode::Safe` whatever `Plan::set_math_mode` says. The default `MathMode` is
  `Relaxed` (`T:omega/src/metal/pipeline_buffers_upload.rs:44-45`), which permits reassociation (`:19-27`); the contract's adds must not
  be re-treed by the compiler, and M1 and M2 are different kernels. The cache-key token is fixed (`S`). The Safe-versus-Relaxed cost is a
  measurement (P2-16a); adopting Relaxed would need a proof that M1 and M2 stay bit-equal, which is not derivable, so Safe stays.
- Under a policy that withholds reassociation the bit-exact sequential arm serves every row count as a per-key left fold; R1 there does
  not depend on this design.
- Masked and padded rows are never used: a masked key is a select on the score and a skipped fma on the value; loads beyond the live
  range are predicated off. NaN in masked or padded K/V rows must therefore never reach the output (AC29).

Gates (nextest tests, not examples; both are internal invariants and cannot show correctness, per the oracle rule in Section 4, where correctness is 4.4 and 4.7):

- T1 `flash_rows_independent_of_row_count` (omega, metal): for hd/G/Hkv in {512/8/1, 256/8/1, 128/4/8} and C in {1, 2, 3, 8, 17, 49, 64,
  512}, every row's output bits equal the 1-row dispatch of that row. `flash_resident_equals_split_merge` compares M1 and M2.
- T2 `speculative_matches_plain_on_synthetic_gemma4` (interop): a synthetic gemma4-shaped GGUF written through
  `proxima_gguf::write_complete` like `T:proxima-model-interop/tests/support/mod.rs`. Shape: 10 layers, period-5 pattern (full at 4 and
  9; `T:gemma4/bind.rs:690-708`), `shared_kv_layers = 4` (sources layer 5 and layer 4), `head_count_kv = 1`, `Hq = 4`, `key_length_swa =
  32`, `key_length = 64`, `sliding_window = 16`, F32 weights, vocab 257, no PLE (`ple_dim` absent is legal, `gemma4/hparams.rs`). Prompt of
  100 tokens (a repeated 10-token pattern so `ngram-simple` drafts; over 6 W, and past the first `Ls = 128` boundary once 28 or more tokens
  are generated), 64 generated tokens, greedy and sampled configs, speculative on versus off, on CPU and on Metal. Assert token ids equal,
  `tokens == 64`, `verify_steps >= 1`. The fixture asserts at build time that the greedy run reaches 64 tokens without EOS (else the
  fixture seed is invalid: an N==0 guard). Control `flash_vs_unfused_bits_differ`: fused and unfused logits differ in bits on the same
  context, which proves the equality assertions are not vacuous. The accepted-draft count is reported, not gated (a random model may
  accept none; verify row 0 is still computed by the verify program and written to KV, so R1 is exercised). No such gemma4-shaped GGUF
  writer exists in the tree today (`tests/support` holds only the dense writer); writing it is part of slice P2-11.

### 3.4 Tile derivation from the 32768 B budget

Budget `T = 32768` (`T:omega/omega-runtime.toml:283`, `sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES`). Storage is `e` bytes per
element (f32 = 4 today; f16 = 2 and Q8_0 = 1.0625 arrive with spec R13/R15). K rows and V rows are both `hd` wide, where `hd` is
`rotary_dim + pass_dim` (`T:cached_attention_render.rs:83-87`); qwen3.6's K row is 64 rotary + 192 pass = 256.

Each simdgroup instruction reads one contiguous K or V row (lane to dim mapping), so no bank padding is needed. One staging buffer holds
a K stage, then a V stage (never both at once). Rule: `Sk = min(32, pow2_floor(T / (s * hd * e)))`, stage bytes `Sk * hd * e`, where `s`
is the number of threadgroups targeted resident per core.

| model / layer | hd | G | Hkv | Sk (s=1 / s=2, f32) | stage bytes (s=2) | Br | simdgroups | threads/TG |
|---|---|---|---|---|---|---|---|---|
| gemma4 full | 512 | 8 | 1 | 16 / 8 | 16,384 | 2 | 16 | 512 |
| gemma4 sliding (Kept unless flipped) | 256 | 8 | 1 | 32 / 16 | 16,384 | 2 | 16 | 512 |
| qwen3-8b | 128 | 4 | 8 | 32 / 32 | 16,384 | 4 | 16 | 512 |
| qwen3.6 attention | 256 (64 rot + 192 pass) | from the op | 2 | 32 / 16 | 16,384 | 16/G | 16 | 512 |

`Br = simdgroups / G` with `simdgroups = 16` (`G` divides 16; otherwise `Br = 1` and each simdgroup owns `ceil(G/16)` query vectors).
Each simdgroup owns one (row, group) query vector, so `Rq = Br * G = 16` vectors share every staged K/V tile [D]. `s` and `simdgroups`
are non-arithmetic constants and are swept (P2-16a). The `s = 2` column assumes the per-core pool is at least `2T`, which is [A]. At
`s = 1` and hd 512, `Sk = 16` uses all 32768 B and leaves nothing for the compiler, so `Sk = 8` is used there too. With f16 the stage
width doubles (cap 32). The kernel carries `[[max_total_threads_per_threadgroup(512)]]`.

Registers (not derived, [A]): per lane `Rv * (hd/32)` each for `q`, `o_run`, `o_blk`, plus `NPV * hd/32` value partials if the loop nest is
keys-outer. At hd 512 and `NPV = 4` that is 112, too many. The output dims are independent, so the value fold is emitted as a dim-slab loop
nest (arithmetic-neutral). Whether 512 threads fit is decided by the compiler and read from the pipeline (Section 5.5), never assumed.

### 3.5 Geometry, memory, threads

Grid: `threadgroups_x = ceil(rows / Br)` (fastest, so co-resident threadgroups stream the same K/V), `y = kv_head`, `z = partition` (M2 only).
This uses the existing `GridSpec::grid2d`/`depth` path (`T:omega/src/metal/resident_nocopy_cache.rs:1318-1329`), which avoids the 1-D `uint
gid` linearisation that truncates above 2^32 threads. Uniforms: `query_rows`, live cached rows, partition count.

| dispatch (gemma4 E2B, full layer, hd 512, G 8) | threads | derivation |
|---|---|---|
| prefill chunk C=512, M1 | 131,072 | 256 row tiles * 512 |
| verify C=49, M1 | 12,800 | 25 * 512 |
| decode C=1, M2 at L=131072, Ls=128 | 524,288 (+256 merge) | 1024 partitions * 512 |
| unfused control C=512, L keys | 262,144 * L | `C*L*Hq*64`; reaches 2^32 at L=16384 exactly [D] |

Transient bytes at C=512: output 512*8*512*4 = 8 MiB (full), 4 MiB (sliding). The unfused score buffer at L=131072 is 512*131072*8*4 =
2,147,483,648 B [D]. M2 scratch for decode at L=131072: 8 heads * 1024 partitions * 514 floats * 4 B = 16,842,752 B [D]. v1's "4 MiB decode
scratch at 131072" belonged to Candidate B and no longer applies. `transient_cap` is 172,812,125 (`T:omega/omega-runtime.toml:176`); v1's qwen3
single-range split scratch of 272,629,760 B at C=512 disappears because prefill chunks use M1 (no scratch).

### 3.6 Performance model and the prefill home-turf arm

Per (query vector, key): 4*hd FLOPs and 8*hd bytes of K+V at f32. A tile shared by `Rq` vectors has intensity `Rq/2` FLOP per device byte
(f32) [D]: 8 at `Rq = 16`. Attention FLOPs per full layer over a whole prompt are `2*hd*Hq*L^2`; gemma4 E2B has 7 full layers: 3.85e12 at
L=8192, 6.16e13 at 32768, 9.85e14 at 131072 [D]. Sliding layers add `28 * 4*hd*Hq*W*L` (0.96e12 at 8192).

Traffic. Tile loads per full-layer chunk are `ceil(C/Br) * L * kv_row_bytes` = 256 * 131072 * 4096 B = 128 GiB (algorithmic, hd 512, f32),
against 2 TiB for today's one-simdgroup-per-(row, group) walk (4096 vectors * 131072 * 4096 B) [D]. This is an `Rq = 16` reduction, not
"O(L*kv_width) independent of C": per chunk it is `C/Br` passes over K/V, as in FA-2's own accounting; over a whole prompt it is about
`L^2 * kv_row_bytes / (2 Br)` independent of `C` [D]. What P2 removes is the L-scaling of memory (no `[C, L]` tensor) and an `Rq`-fold cut
of traffic. Whether the remaining traffic hits DRAM or the system-level cache is [A]: the row-tile-fastest launch order gives co-resident
threadgroups a shared stream, and the effect is measured through time, since per-dispatch DRAM counters are unavailable.

Instruction-mix bound [D]: per (vector, key) per lane about `hd/16` FMA issues, `hd/16` LDS reads at one vector per simdgroup, and about 3.5
shuffle and exp issues. That caps the FMA-issue fraction near 45 to 70 percent unless LDS co-issues [A]. No claim is made that the kernel
reaches any fraction of peak; the gate below is a measurement.

Prefill home-turf arm (measurement slice P2-16b), pre-stated rule:

- Arms: proxima (the `niah` harness, new `--prefill-pairs N`) and Ollama `/api/generate` on the same prompt bytes (`R16c1`; the pinned haystack reduced to ASCII with single spaces, whitespace runs collapsed and non-ASCII dropped, sha recorded, until the gemma4 tokenizer fix lands), `raw`,
  `num_predict = 1`, `num_ctx` = ctx, `keep_alive = 0` between pairs. Lengths 8192 and 32768. Pairs interleaved, order alternating, 3 pairs
  minimum, on a quiet box per the speculative SPEC protocol (`ollama ps` before and after, GPU idle window under 5 percent).
- Recorded per pair: `proxima_prefill_ms`, Ollama `prompt_eval_duration_ms`, `prompt_eval_count` (the harness already parses it,
  `T:proxima-model-interop/examples/long_context_niah/ollama.rs:47-58`) and proxima's prompt token count.
- Validity gate (the pass condition of the slice): 3 pairs printed per length, `prompt_eval_count == proxima tokens` (else Ollama served a
  cached prefix and the pair is void), pair-ratio CoV at or under 5 percent, `contaminated = 0`. Zero valid pairs is RED. There is no win
  threshold: no N can be derived (the achievable fraction of peak is [A]).
- Follow-up rule, derived logically. Let `gap = proxima_ms - ollama_ms` where positive and `a` the attention kernels' measured ms in the
  proxima prefill. If `a <= gap`, attention alone cannot close the gap even at zero cost: record it, name the non-attention share, hand it to
  the GEMM owner. If `a > gap`, open the tile sweep and the MMA arm (Section 3.7). Both outcomes are recorded either way. Per-dispatch
  timing is 50 to 100 times contaminated on this box [P], so `a` comes from the fused-versus-unfused build difference (same interleaving),
  not from per-op timers.

### 3.7 Abandoned inside the kernel: `simdgroup_matrix` for QK and PV

The tiled GEMM already uses `simdgroup_multiply_accumulate` (`T:omega/src/msl/tiled_gemm_cooperative_scan.rs:505,1360`). It is not used in
the first arm: the summation order inside an 8x8 multiply-accumulate is hardware-defined, and independence of a row's result from the other
rows of the fragment is unproven; that independence is exactly what R1 needs. It is a follow-up arm gated on a hardware probe (the same
row at different fragment positions and with different neighbours must be bit-equal) plus the accuracy gate, taken only if the follow-up
rule above says attention can close the gap.

### 3.8 Decode throughput: dispatch census, decode pairs vs Ollama, follow-on chains

Recorded facts [P] (memory notes `project-gemma4-decode-dispatch-type` and `project-gemma4-attn-fusion-recognizer-gap`, measured 2026-09-21
to 2026-09-26 at base 39d8709e1, not re-measured here): gemma4-E2B decode is about 21.4 to 22.4 ms per token wall against Ollama's effective 158 GB/s versus proxima's 66 GB/s (2.4 times; the brief's Ollama figure of about 6.4 ms per token does not reconcile with 158 GB/s over the recorded 1.42 GB per token, which is 9.0 ms, so no Ollama millisecond figure is asserted here); Q4_0/Q8_0 matvecs are about 39.6 percent of GPU exec at about 150 GB/s in situ; the remainder
is about 1386 unfused non-matvec dispatches; host encode is about 25 percent of wall (1.75 ms for 1661 dispatches, about 1.05 us each [D]).
The census recorded 1663 BoundOps unfused (1661 after `prune_dead`) and 1173 with the online kernel, i.e. exactly 14 nodes absorbed per layer,
490 per token [P]. So an attention layer's unfused chain is 15 BoundOps (14 absorbed plus the anchor), not the 25 of the earlier hypothesis;
the census in P2-12 prints both.

(a) Dispatch census (slice P2-12, requirement R28), from the real program through `bind_with_fusion` and `omega::emit`, at `new_count` in
{1, 49, 512}. Expected from the recorded numbers [D from P]: before, 35 layers * 15 = 525 attention-span BoundOps per decode token (31.6
percent of 1661). After routing full layers only: 7 * 1 + 28 * 15 = 427 (98 fewer, 5.9 percent of the token's dispatches). After routing all
layers: 35 (490 fewer, 29.5 percent). Physical dispatches: 1 per fused layer under M1, 2 under M2 (split plus merge; `ENCODE_DISPATCH_CALLS`
counts `encode_op` calls, split plus merge counts once, and the physical dispatch happens inside `dispatch` (`resident_nocopy_cache.rs:1310-1352`, three call sites), so the
census also prints `PHYSICAL_DISPATCH_CALLS` under `instrument`). Prefill chunks have the same BoundOp counts (same program) with different
grids. This is why routing only 7 of 35 layers moves decode dispatches by 6 percent: the tok/s case rests on the sliding layers, and that is
what Section 5.1's measured flip decides.

(b) Decode pairs against Ollama (slice P2-16c). Harness: `niah --decode-pairs N --decode-tokens 64` (same prompt-bytes guarantee as R16c1, on the same ASCII single-space haystack as 3.6; the
proxima arm reports ms per token over the decode window only, as `(elapsed_last - elapsed_first) / 63` from `TokenEvent.elapsed_ms` (`T:proxima-model-interop/src/generate/residency_caches.rs:2975-2999`, a `u64` millisecond clock, so a window statistic, not a per-token one; `speculative_bench.rs:1290-1308` already computes the same), with `SpeculativeConfig::none()` pinned so speculation does not enter the number; the Ollama arm reports `eval_duration /
eval_count`). Contexts 2048 and 32768. The repo's interleaving rule applies: one iteration runs both arms back to back, order alternating
each pair, warm-up pairs discarded, per-pair ratio (a back-to-back arm block measured a fake 2.7x that was 1.1x [P]); box load, GPU utilisation
and `ollama ps` are checked before each arm; the Ollama model is unloaded (`keep_alive = 0`) and the GPU allowed to idle under 5 percent before
the proxima arm, and no compile or agent runs during a timing phase. Three builds per context: unfused baseline, full layers fused, all layers
fused (the last only once P2-17 exists). Validity gate (the slice's pass condition): 3 pairs printed per context, Ollama `eval_count == 64` and
proxima decode tokens `== 64`, pair-ratio CoV at or under 5 percent, `contaminated = 0`; zero valid pairs is RED. No win threshold is stated.
Follow-up rule (derived): fit `ms_per_token = a + c * N_dispatch` over the three builds at each context (dispatch counts from (a), three points
and two parameters, so one residual degree of freedom, printed). Let `G = proxima - ollama` for the best build. Predicted saving of fusing the
remaining chains in (c) is `c * N_remaining_chain_dispatches`. If it is at least `G`, dispatch fusion alone can close the gap and (c) is the
next slice; if it is below `G`, the bandwidth term is binding and the next slice is the matvec, not fusion. The fit's residual and `c` are
recorded even when the rule is moot.

(c) Follow-on chains (short; attention is the scope). No new primitive is needed. What remains after attention, per the recorded population
[P]: per-layer RMSNorm chains, GeGLU and residual/scale elementwise chains, the per-layer-embedding injection, and the Q6_K head. The primitives
that fuse them already exist and are bit-exact by construction, so they need byte-equality gates, not an accuracy oracle:
`ReduceEpilogueFusion` (a `Reduce` with `epilogue_body` and broadcast epilogue operands; RMSNorm is a sum-of-squares reduce plus a broadcast
epilogue) and `ChainFusion` (elementwise chains), both admitted with no permission (`T:proxima-tensor/src/numeric.rs:161-166,191-193`). The
natural follow-on is extending those two recognizers to the gemma4 shapes the P2-12 census still shows unfused, ordered by dispatch count per
token, and it is chosen by that census, not by this document. The Q6_K head is a matvec-tier item, outside both.

## 4. Oracle

Rule (owner, 2026-09-29, binding): llama.cpp is the correctness oracle. It is never a runtime, crate, vendored code or dependency of proxima. Reference outputs are
generated by external tools compiled against the external build (`/Users/brianbruggeman/repos/others/llama.cpp`, commit `f1ea20621`, `build/bin` and `ggml/include`), CPU
backend only, and vendored as fixtures with the commit and command recorded, following the precedent `T:proxima-tokenizer/tests/fixtures/llama-ngram/{generator,fixtures}`.
proxima-internal references are internal consistency checks and are labelled as such wherever they appear: the f64 definition, the CPU path, fused-versus-unfused
comparisons, resumed-versus-fresh, T1 and T2. They cannot establish correctness: two proxima paths can agree while both are wrong (resumed and fresh decode "matched"
for 1,365 to 1,872-token gemma4 prompts while both went through the same truncated dispatch [P: memory `feedback_no_llama_cpp_anywhere`]). Ollama remains the runtime
incumbent for performance arms only (the repo cites Ollama's own Go implementation of gemma4 as `gemma4.go`, `T:gemma4/bind.rs:690-694`, so an Ollama run is not a llama.cpp run and is not the correctness oracle either [A: read from those citations, not run]).

### 4.1 Principle 14: the incumbent's own criterion

The v1 bar "byte-identical to feature-off Metal" (`T:proxima-tensor/Cargo.toml:40`, `T:proxima-model-interop/Cargo.toml:177-181`, "not yet met") cannot stand, and
the proof is the incumbent's own behaviour, not an assertion:

1. llama.cpp compares two implementations of flash attention by normalised mean squared error, never by bits. NMSE is `mse(a, b) / mse(a, 0)` with `a` the reference
   (`/Users/brianbruggeman/repos/others/llama.cpp/tests/test-backend-ops.cpp:301-313`), and the acceptance bar for `FLASH_ATTN_EXT` against its CPU reference is
   `max_nmse_err() = 5e-4` (`:7915-7917`; the default for other ops is 1e-7, `:1214-1216`). [R]
2. [M] Measured with llama's own kernels on identical bytes (external CPU-backend tool `E/llama_oracle_demo.c`, synthetic Q/K/V, hd 512, 64 rows, 1500 keys,
   `E/llama_oracle_demo.out`): llama's unfused f32 chain versus its `GGML_OP_FLASH_ATTN_EXT` with f16 K/V gives NMSE 9.8e-6 (peaked scores) and 3.2e-5 (flat). Two
   implementations of one op inside the incumbent are not bit-equal. Its CPU flash op also rejects f32 K (`ggml-cpu/ops.cpp:7101`, "fattn: unsupported K-type", hit
   while building the tool), so the kernel-level reference for proxima's f32 KV is llama's unfused f32 chain (the `-fa off` structure), and the f16-K/V flash op becomes the
   reference for the f16 arms of spec R13.
3. The repo's own algebra also says a reassociating lowering needs an explicit permission: `NumericRewrite::TreeReduce`, `ContextChunkMerge`, `ContextSplitMerge` require
   `reassociation` (`T:proxima-tensor/src/numeric.rs:169-178,206-212`) and the shipped default grants it (`serving.rs:992`); under a policy that withholds it the bit-exact
   sequential arm stays.

So the bar adopted is the incumbent's kind of criterion (NMSE to the oracle's output), applied on the real payload with thresholds derived before the kernel exists (4.4),
plus llama's own absolute bar. Bit identity is kept only where the arithmetic is unchanged: the bit-exact arm, and decode versus verify under R1 (an internal invariant).

### 4.2 Reference generation (external, vendored)

Two generators, C++ sources checked in beside their fixtures (`tests/fixtures/llama-attention/{generator,fixtures,README.md}`; the README records the llama.cpp commit and each
command line). They run once, on a quiet box, on the CPU backend; the tests read the fixtures and need no llama.cpp at run time.

- G1 `llama-attention-kernel`: reads proxima's `payload.manifest` and dumped tensors (4.3), builds the ggml graph on those exact bytes (unfused f32 chain `mul_mat`,
  `soft_max_ext(mask, scale, 0)`, `mul_mat`; and, for f16 arms, `GGML_OP_FLASH_ATTN_EXT` with an F16 mask padded to `GGML_KQ_MASK_PAD`) and writes `y_llama_*.bin`.
  Feasibility is demonstrated: `E/llama_oracle_demo.c` builds against the external `libggml`/`libggml-cpu` and runs.
- G2 `llama-attention-model`: the real gemma4 GGUF, token ids from `llama-tokenize`, `llama_decode` in 512-token chunks on the CPU backend, with a `cb_eval` copying
  `kqv_out-0` and `kqv_out-4` of the last chunk. `kqv_out` is `build_attn_mha`'s output before the `wo` projection (`src/llama-graph.cpp:2809-2811`), the same tensor as
  proxima's anchor output. The arch exists in this build (`src/llama-arch.cpp:59`, `src/models/gemma4.cpp`). [R]
- Tokenizer. proxima's gemma4 tokenizer mis-tokenises newlines and non-ASCII (being fixed separately). Every real-text fixture is ASCII with single spaces (or is generated
  after the fix), token ids come from `llama-tokenize`, and each gate first asserts proxima's ids equal them (`token_ids_equal=true n=2048`; otherwise RED).

### 4.3 proxima payload capture (before any kernel)

Existing hook, no new instrument in omega: `PROXIMA_CAPTURE_NODES`, `PROXIMA_CAPTURE_STEPS`, `PROXIMA_CAPTURE_DUMP_DIR` (`T:omega/src/metal/arena_encode_dispatch_finish.rs:626-705,820-954`,
omega feature `instrument`, reachable as `proxima-model-interop/instrument`, `T:proxima-model-interop/Cargo.toml:65`). It dumps every input binding, the output binding and the
uniforms of each captured dispatch to `node{N}_step{S}_buf{i}_off{o}_len{L}.bin` plus `node{N}.meta`; `S` mirrors the decode loop's step counter
(`T:omega/src/metal/device_buffers_arena_plan.rs:611-619`). Each buffer is capped at 64 MiB.

1. Layers. gemma4 E2B layer 4 (first full layer, hd 512) and layer 0 (first sliding layer, hd 256). Both own their KV (own-KV layers are 0..14, `T:gemma4/bind.rs:305-310`).
2. Shape. The 2048-token ASCII prompt at `ubatch_size = 512` (P1 chunked prefill), capturing the last chunk (cached 1536, new 512; layer 0's ring holds 512 cached rows), the
   same chunking as G2. NOT one-evaluation prefill: `rows * 8960 * 256` exceeds 2^32 at 1873 rows (`S:TASKS.md:322-323`), so a one-evaluation 2K payload is silently corrupt
   from layer 1 on.
3. Node ids. `attention_outcomes` at the chunk shape (5.3) returns an `AttentionAnchor` for every one of the 35 layers on the unflipped tree, because the decline there
   (`softmax_weights_decode_only`, `:1221`) happens after source resolution (`:1066`). The anchor's 8 source nodes and output node name the dispatches that consume or produce them;
   `T:proxima-model-interop/tests/gemma4_attention_chain_census.rs:52-53` enumerates the chain for layers 0 and 4 as a cross-check.
4. Harness `examples/attention_payload_oracle.rs` (a consumer of the hook, not a new dump mechanism). RED unless it finds the 9 tensors (q_even, q_odd, k_cached even/odd, k_new
   even/odd, v_cached, v_new, unfused output) for both layers (18) and writes `payload.manifest` (layer, kind, hd, G, Hkv, cached_len, new_rows, window, scale, step, sha256s).
   The mask is not a buffer: it is reconstructed from the manifest.
5. The hook is an env-gated file dump, the pattern the repo's rules forbid for new work. It exists and is used as is; replacing it with a structured event and a file-sink
   `Exporter` is a separate change, not part of P2.

### 4.4 Statistic, derivation, pre-registration

Definitions. Kernel level: `y_llama` = G1 output on proxima's bytes. Model level: `y_llama` = G2's `kqv_out-il`. `x_inc` is proxima's unfused Metal output, `x_fus` the flash kernel's
(kernel level: run on the payload bytes; model level: the layer's attention output from a proxima run with the flash arm). Per tensor (full, sliding):

- `nmse_x = NMSE(y_llama, x)` with llama's own definition.
- `rho_max = max |x_fus - y_llama| / max |x_inc - y_llama|`.

Threshold derivation. Build `K = 64` equally legitimate reorderings of the incumbent (its width-64 strided folds with a random element-to-lane assignment, f32, evaluated on the same
payload). Their `nmse_k` and `rho_max_k` against `y_llama` measure the variability between correct implementations of the same op. Acceptance: `nmse_fus <= max_k nmse_k`,
`rho_max <= max_k rho_max_k`, and `nmse_fus <= 5e-4` (llama's absolute bar, 4.1). For exchangeable legitimate orders a new one exceeds the maximum of `K` with probability `1/(K+1)` =
1.5 percent [D]. Emulation check (independent of the oracle): the real Metal `x_inc` must lie within the reorder class's radius of the CPU emulation, `NMSE(x_emul, x_inc) <= max_k NMSE(x_emul, x_k)`;
else the emulated class does not represent the incumbent and the slice is RED. At model level, layer 4's inputs are identical in the fused and unfused proxima runs before P2-17 (layers 0 to 3 are sliding
and `Kept`), so the common upstream difference to llama is a floor that cancels between the two arms; the same acceptance rule applies with `y_llama = kqv_out-4`.

Ordering. `payload_thresholds.toml` is committed by P2-2, from the vendored llama fixtures and the unfused proxima payload, before any kernel. AC28 checks
`git rev-list --count <thresholds commit>..<kernel commit>` is at least 1.

What the oracle cannot see, and the tightening gate [M]. Errors add: `NMSE(llama, x) ~ NMSE(llama, truth) + NMSE(x, truth)`, so the distance to the oracle is dominated by the oracle's own rounding error.
Measured with llama's kernels (`E/llama_oracle_demo.out`, `E/llama_npv.out`; synthetic, hd 512, 1500 keys): `NMSE(llama unfused, f64)` = 5.56e-13 (peaked) and 1.23e-13 (flat); the contract
fold's distance to llama is 7.00e-13 and 1.38e-13 at `NPV = 4`, versus its own distance to f64 of 1.15e-13 and 1.6e-14. Across `NPV` = 1, 2, 4, 8 in the flat regime the distance to llama moves
1.438e-13, 1.407e-13, 1.383e-13, 1.366e-13 (5 percent, inside the reorder spread), while the distance to f64 moves 2.15e-14 to 1.51e-14 (43 percent). So an oracle-defined bar is
necessary and cannot alone select the arithmetic constants. Selection uses the internal f64 definition as a tightening gate (labelled internal): among candidates that satisfy the llama-defined region,
take the smallest `NPV` whose f64 `rho_rms` (below) is inside the K-order control region against f64. Acceptance stays with the oracle.

Internal demonstration of the tightening procedure on synthetic data (f64 reference; incumbent modelled as two width-64 folds, two-pass softmax with width-64 strided sums; hd 512, 128 rows, 1500
visible keys; `E/err_stat.out`, `E/npv_sweep.out`, `E/ls_sweep.out`; internal consistency only):

| regime (score sd) | incumbent rms err vs f64 | control rho_rms (24 orders) min / p50 / p95 / max | candidate rho_rms by `NPV` = 1 / 2 / 4 / 8 |
|---|---|---|---|
| 3.0 (peaked) | 1.23e-7 | 0.904 / 0.993 / 1.060 / 1.106 | 1.034 / 1.018 / 1.007 / 1.002 |
| 0.057 (flat) | 3.24e-9 | 0.985 / 1.007 / 1.022 / 1.024 | 1.164 / 1.042 / 0.990 / 0.971 |

With `NPV = 1` the value fold is one sequential fma chain per tile, and in the flat regime the contract fold is 16 percent worse in rms than the incumbent model, outside every control order (max
1.024). `NPV = 2` still fails the flat regime (1.042); `NPV = 4` passes both. A sub-tile fold that adds into one accumulator is worse than `NPV = 4` (1.076 to 1.078; `E/sub_sweep.out`). Smaller
`Ls` raises error (1.59 at `Ls = 32`, flat), larger `Ls` saturates (1.15 to 1.17 at 128 to 512, `NPV = 1`). P2-8 sweeps `NPV` in {2, 4, 8} and `Ls` in {128, 256, 512} on the real payload;
`NPV = 4`, `Ls = 128` are starting values only. The synthetic incumbent is a single-range model, so only the procedure is demonstrated.

Retraction (v1 minor 2). v1 claimed a leaked masked key at long L is invisible under an accumulation noise of about 2.2e-5. Not reproduced: against f64 (internal) the incumbent-model rms error is
1.2e-7 (peaked) and 3.2e-9 (flat), one leaked masked key inflates `rho_rms` by 2,941 and 203,156 at 1500 keys, and a dropped last key by 13,217 and 211,315 (`E/err_stat.out`). Against the oracle a
leaked key of weight `w` contributes about `(w * |v|)^2 / mean(y^2)` to the NMSE, `w` near 1e-3 to 1e-4 at these lengths, so 1e-7 to 1e-8 against a floor of 1e-13 [D]. O3 stays because it is exact,
regime-independent and dtype-safe.

### 4.5 O3: exact support (internal, exact)

An arithmetic identity, so no oracle is needed. Scores are forced equal (all keys share one K row), so every visible weight is `exp(0) = 1` and the fold is integer arithmetic in f32 (exact below 2^24).
Each V row carries a `{0,1}` signature: three coprime moduli whose sum is at most `hd` (hd 512: 163, 167, 173, product 4.7M, unique past 262144; hd 256: 79, 83, 89 [583K]; hd 128: four moduli 23,
29, 31, 37 [765K]). Output dim `d` must equal `count_d / n_visible`, where `count_d` counts visible keys whose residue code hits `d`. Masked and padded rows carry NaN in K and V. Nothing large is stored,
so f16 and Q8_0 hold every value (a Q8_0 block dequantises `1.0` to 0.99994; the test dequantises through the same codec). A leaked key adds `1/n` to at least one bin; a dropped key removes it; a
compensating (leak, drop) pair is caught because the three residue codes cannot all collide below the CRT bound. [M] `E/support_exact.out` (contract fold, C simulation):

| case | n visible | rounding error vs exact count | over-include | over-mask | swap pair |
|---|---|---|---|---|---|
| window 5000 of 65536 | 5,000 | 3.6e-8 | 3.6e-2 | 3.4e-2 | 3.6e-2 |
| window 5000 of 131072 | 5,000 | 3.6e-8 | 3.6e-2 | 3.4e-2 | 3.6e-2 |
| full, n=131072 | 131,071 | 3.8e-8 | 1.3e-3 | 1.3e-3 | 1.3e-3 |
| full, n=8192 | 8,191 | 1.5e-8 | 2.1e-2 | 2.1e-2 | 2.1e-2 |

`nan_in_output = 0` in all four: NaN in masked rows never leaked. Defect deviations exceed the rounding error by 3.4e4 or more. A first version with two moduli (residues mod 256) missed the (leak, drop)
pair at n = 131072 because keys 65535 and 131071 share both codes: the CRT bound is required. Test cases (AC29): full support; window edge; a cached/new boundary inside a tile; padding NaN; a partition
boundary at `Ls`; n = 131072. Controls (AC29): over-include, over-mask and swap pair each deviate by at least 1e-4.

### 4.6 Worked example (kept from v1: all 20 cells reproduced, plus 20 more; oracle-checked)

Setup as v1 (Hkv=1, G=2, hd=2, W=3, scale = ln 2; two cached keys, five new keys at positions 2 to 6; k_even = 1 0 1 0 1 0 1, k_odd = 0 1 2 0 1 2 0; V(k) = (k, 6-k); h0 query (1,1), h1 query (0,-1) on
even rows and (1,-1) on odd). Re-derived with exact integer arithmetic (`E/exact.awk`; weights are powers of two): the 20 windowed cells match v1's table cell for cell
(`S:design-p2-fused-attention.md`, "Worked example"). The same keys with bands `cached{MIN,MAX} new{MIN,0}` (a full layer):

| row | h0 | h1 |
|---|---|---|
| 0 | (3/2, 9/2) | (4/7, 38/7) |
| 1 | (21/13, 57/13) | (9/8, 39/8) |
| 2 | (37/17, 65/17) | (24/13, 54/13) |
| 3 | (19/7, 23/7) | (13/7, 29/7) |
| 4 | (3, 3) | (53/18, 55/18) |

Oracle check [M]: llama.cpp's unfused f32 chain on its CPU backend (`E/llama_worked_example.c`, commit f1ea20621) reproduces all 40 cells with maximum absolute error 7.5e-7
(`E/llama_worked_example.out`). Tolerance 1e-5 is therefore 13 times the oracle's own deviation, which is the derivation of that number. The check is vendored as a fixture from G1 (AC30).

### 4.7 Sliding-window real-model gate

The prompt exceeds W = 512 by construction (3W or more, so P1 chunking crosses the ring three times), ASCII with single spaces (tokenizer note, 4.2), token ids from `llama-tokenize`. Structure
per case: a distractor fact early (token about 200, more than 1,300 tokens from the question, outside every sliding window), filler generated from a fixed pool of ASCII sentences, a correction fact
inside the window near the end, then a question whose answer is the in-window value ("The access code is 4417." ... "Correction: the access code is 9082." ... "What is the access code?"). Five cases with
different values. Full layers can still see the distractor, so the case is a discriminating end-to-end guard, not a mask proof (the mask proof is AC29 plus AC30).

- Oracle: llama.cpp (`llama-server` or `llama-cli` from the external build, CPU backend, temperature 0, speculation off) on the same token ids, answers vendored as a fixture with the commit and command.
  Positive control: llama must answer the in-window value in 4 or more of 5, else the prompts are invalid.
- Negative control: the same prompts with the correction removed; llama and proxima must answer the distractor value (shows the prompt discriminates).
- Degenerate control: the existing ring-offset hook (`LoadedModel::with_ring_write_offset_for_parity_control`, `S:TASKS.md:206-207`) set to 1 must lower proxima's found count; if it does not, the gate
  measures something else.
- Gate: `proxima found = X/5`, `llama found = Y/5`, `Y >= 4`, `X >= Y`. Runs before the flip (fusion off, baseline `X0` recorded) and after.

## 5. Scope, routing, values

### 5.1 Per-layer policy (precise), and the designed flip

Per recognized anchor, decided after the structural match:

| anchor | outcome | why |
|---|---|---|
| no window (`cached_lower == MIN`, `local_row_bound == MAX`) | `Fused` (flash lowering) | its unfused score memory is `C*L*Hq*4`; L-scaling |
| window `W` (sliding) | `Kept { window: W }` (unfused, shipped ring path) until P2-16d flips it | see below |
| any structural or eligibility failure | `Declined { why }` | a value; loud under demand |

Why `Kept` by default: (a) Memory. With the ring a sliding layer's scores are `C*(W+C)*Hq*4` = 16.8 MB at C=512 regardless of L [D]. (b) R1 under the ring.
The unrolled scratch's row 0 is position `cached_len - live_rows` (`S:SPEC.md:126-129`, `unroll_live_rows`), which moves as the window slides, so a tile
grid anchored at scratch row 0 would put a key in different tiles for a decode step and for a verify row at the same step. Full layers have row 0 =
position 0 always. [D] (c) Nothing yet measured says routing sliding pays, and the recorded decode data says it may (Section 3.8(a): 28 of the 35
attention chains, 420 of 525 attention BoundOps per token, are sliding).

The flip path, designed now. Routing windowed anchors needs the tile grid anchored to absolute position. Decision: phase-align the ring scratch on the
host: `unroll_live_rows` writes the oldest live row at scratch row `pad = base mod Ls`, where `base = cached_len - live_rows` (positions evicted), so scratch
row index is congruent to absolute position modulo `Ls`; the program's `cached_len_swa` input becomes `pad + live_rows`; the first `pad` rows are never read
(below the lower band) and each windowed layer's scratch grows by at most `Ls - 1` rows (12 own-KV layers * 127 rows * 256 * 4 B * 2 = 3.1 MB [D]). The
kernel is unchanged (absolute index = scratch index) and query-to-key distance is unchanged by a uniform shift, so the existing `causal_mask_cached_windowed`
(the program's own mask) stays correct. This needs no new operand and no new kernel parameter. Gates for the flip (P2-17): T2 runs with sliding routed and
kept (4 passed), the ring parity example (AC12/AC13 numbers) is re-run, and `flash_rows_independent_of_row_count` gains ring-offset cases (same key set at
scratch offsets differing by non-multiples of 32, phase-aligned by the same helper) with `mismatching_rows=0`.

Measured reason to flip (P2-16d, decode ms/token first, then prefill). Two interleaved builds (sliding routed via the one-line predicate, sliding kept),
5 or more runs each, on a quiet box, at 2048 and 32768 context. Flip iff either the decode ms/token or the prefill chunk time improves by more than
`2 * sigma_total` (the std of that arm's per-token or per-chunk time across runs) and neither regresses by more than `2 * sigma_total`. An improvement
under that noise floor is not measurable, so it is not a reason. If it flips, P2-17 lands the phase alignment and the routing predicate in one slice.

### 5.2 The routing edit

Gate location: `T:proxima-tensor/src/bind/dead_code_cached_attention.rs`, function `cached_attention_candidates` (`:645`), loop over `output_position`
(`:669`). For a gemma4 full-layer prefill anchor (C=512 new rows, bucket-padded cached rows, `via_gemma_template = true`, `cached_lower = MIN`,
`local_row_bound = u64::MAX`):

| branch | line | result today |
|---|---|---|
| anchor detection (`not_online_softmax_add` ... `score_denominator_mismatch`) | 674 to 773 (11 `continue;`) | pass |
| `mask_select_shape`, `mask_select_arity`, `mask_form` | 775, 784, 802 | pass |
| cached padding walk `cached_padding_mask_lower_bound` | 828 to 836 | pass (via_gemma_template) |
| scale / score-source / query / pass / value / product checks | 851 to 1057 | pass |
| `source_not_found`, `source_indirect_or_negative_stride`, `scale_not_constant` | 1081, 1096, 1113 | pass |
| `local_window_not_vacuous` | 1145 | pass (`local_row_bound == MAX`) |
| `cached_len_precision` (2^24) | 1156 | pass |
| `cached_len_operand_required` | 1175 | pass |
| `window_mismatch` | 1184 | pass (both `None`) |
| `softmax_weights_decode_only` (`via_gemma_template && new_key_rows != 1`) | 1221 | DECLINE: where prefill stops today |
| `softmax_weights_scale_not_unity` | 1231 | pass at gemma4's scale 1.0 (`Unscaled`, `T:gemma4/bind.rs:799`) |
| `softmax_weights_eligible = via_gemma_template` | 1241 | the real gate: routes to Candidate B |
| Candidate B arm | 1366 to 1580 (ends `continue` at 1579) | never reaches the online kernel |
| pass strides, dependencies, requested-output, removable, resolved, ninth operand | 1581 to 1702 | unreachable for gemma |
| `CachedAttention` build | 1712 to 1742 | unreachable for gemma |

Why v1's edit fails: deleting `:1221-1240` alone lets the prefill candidate reach `:1241` with `eligible = true`, so it enters the Candidate B arm at
`:1366`, which builds a `CachedSoftmaxWeights` pair assuming one new key per row (`run_cached_softmax_weights` reads `new_scores[row, 0]`). It would
produce a structurally wrong op for 512 new rows and never the online kernel.

The v2 edit. (1) Replace `:1201-1242` with the route decision: window present gives `Kept`; else continue to the fused build. (2) Delete
`softmax_weights_decode_only` and `softmax_weights_scale_not_unity` (their only purpose was Candidate B's missing proof; the flash kernel takes any scale,
and the worked example, scale ln 2, covers a non-unity scale in AC30). (3) Delete `softmax_weights_eligible`, the Candidate B arm `:1365-1580`, the third element of
`cached_padding_mask_lower_bound`'s return, and the pins at `T:cached_attention_epilogue_liveness.rs:74-109`. (4) Fold the gemma-shaped matchers
(`cached_padding_mask_lower_bound`, `local_causal_mask_new_row_bound`, `cached_len_padding_template`) into `cached-attention-streaming` and delete the
feature `metal-fuse-attn-decode` (`grep -rl` finds it in 19 files; 16 references in `dead_code_cached_attention.rs`, 14 in `generate/decode.rs`, 6 in
`tests/attn_launch_shape_census.rs`, 6 in `proxima-model-interop/Cargo.toml`). (5) `BoundOpKind::CachedSoftmaxWeights` (96 references in 34 files) becomes
unreachable and is deleted (P2-14; gate `git grep -c CachedSoftmaxWeights` = 0). Candidate B is retired, not "kept for decode", because decode on Candidate
B with verify on flash is the R1 break.

### 5.3 Outcomes as values; the demand bit

Information destroyed today: the two-range recognizer has 64 `continue;` sites (`:645-1745`): 11 are anchor non-matches (`:674-773`), 53 are declines of a
recognized anchor, of which 36 carry a `stage = "..."` name only inside an `instrument`-gated `debug!` and 17 carry nothing; the single-range recognizer has 28
with no name (`:1767-2032`). The reason is computed and then dropped at `continue`.

```rust
pub struct AttentionAnchor {
    pub output: NodeId,
    pub sources: SmallVec<[NodeId; 11]>,
    pub scale: f32,
    pub window: Option<u64>,
    pub cached_len: Option<NodeId>,
}

#[non_exhaustive]
pub enum AttentionOutcome {
    Fused { anchor: AttentionAnchor, op: BoundOp, absorbed: BTreeSet<NodeId> },
    Kept { anchor: AttentionAnchor, window: u64 },
    Declined { output: NodeId, why: DeclineReason, anchor: Option<AttentionAnchor> },
}

#[non_exhaustive]
pub enum DeclineReason { MaskForm, CachedPaddingForm, ScaleShape, ScaleMismatch, PolicyTooStrict { rewrite: NumericRewrite }, WindowMismatch, CachedLenPrecision /* one variant per stage name */ }

pub fn attention_outcomes(program: &[Op], shapes: &Shapes, outputs: &[NodeId], numeric_policy: NumericPolicy)
    -> Result<Vec<AttentionOutcome>, TensorError>;
```

`DeclineReason` promotes each existing `stage` string (36) and names the 17 unnamed sites. The first-pass discovery outcome is reported, except for anchors
pass 1 fused, which report their second-pass outcome (`output_not_resolved` and the like; `cached_attention_epilogue_liveness.rs:33-40,113-117`).
`Declined.anchor` is `Some` when the failure is after source resolution (`:1066`). The split is three-way, not a bool or a `Policy` variant with
`is_policy()`, because "unfused by policy" must not be a failure under demand. `bind_with_fusion` keeps its signature and its `fuse_cached_attention: bool`
(`T:proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:62-68`) and consumes `Fused` outcomes as it consumes candidates today. The recognizer's matching phase
is split into `match_anchor` (structural, `Result<AttentionAnchor, DeclineReason>`), `route`, and `build_fused`. Under a policy without `reassociation`
the recognizer still emits `Fused` and the sequential arm lowers it, exactly as today (no policy gating at bind: `NumericPolicy::default()` is used at 180
sites in `omega/src/msl/tests.rs` alone, 164 `default()` plus 16 `bit_exact()`).

Demand. `ServingConfig.flash_attention: bool` (`T:serving.rs:588`, `-fa`) is rejected unconditionally today (`serving.rs:1113-1121`, "requires a new fused
Op variant"), which is stale. New meaning: `true` requires every full-attention anchor to be `Fused`. `BackendRuntime` copies the bit at construction (as it
does `fuse_cached_attention`, `T:generate/residency_caches.rs:1575`) and, when set, runs `attention_outcomes` once per new plan key (cold path). Any
`Declined` returns `InteropError::FlashAttentionDeclined { declined: Vec<(u32, DeclineReason)> }`, a typed error carrying the values rather than a formatted
`String`. `flash_attention = true` with `cached_attention_fusion = false` is rejected in `apply_serving_config` (one typed check).

No new config enum. Call sites both ways:

```rust
ServingConfig { cached_attention_fusion: true, flash_attention: true, ..ServingConfig::default() }
ServingConfig { attention_fusion: AttentionFusion::Require, ..ServingConfig::default() }
```

The three states (off, auto, require) are all expressible with the two bools that already exist (47 references to `flash_attention` in 25 files, measured). The
enum would remove exactly one representable state, `(false, true)`, which one admission check already covers. A caller can do nothing with the enum that they
cannot do with the bools, so it is a relocation and is not added. This departs from the `ContextLength` precedent (`S:SPEC.md:64-69`) on purpose: that pair was
`Option` plus bool with a genuinely meaningless state and a bool-count lint; here both bits are shipped and meaningful, and the invalid combination is refused
at admission. Principle 4 parity: a test `flash_attention_agrees_across_literal_and_default_override` following `serving.rs:1659-1735`.

### 5.4 `bands` (kept from v1)

`BoundOpKind::CachedAttention { cached_lower_inclusive, new_upper_inclusive }` (`T:proxima-tensor/src/bind/types_layout_boundop.rs:210-211`) becomes
`bands: [CausalBand; 2]` (cached range, new range). `CausalBand` exists and the CPU reference already takes the pair (`T:proxima-tensor/src/physical.rs:308-311,538-541`),
yet the CPU executor rebuilds it from hard-coded `MAX`/`MIN` (`T:proxima-tensor/src/cpu/run_node.rs:795-807`) and the bind step drops the new-range lower bound: a
rich to poor boundary, an op that cannot express a sliding window on its new range. 72 references in 12 files (`grep -rn`, measured). The kernel predicate is one band
test, `relative < lower || relative > upper`, plus loop bounds. What a caller can do that they could not: a windowed layer with a chunk wider than its window
(`ubatch_size > W`, a config value) is expressible; today it hits `local_window_not_vacuous` (`:1145`), a value-dependent cliff. The runtime ninth operand (two
scalars discriminated by `cached_key_rows == 0 && operands.len() == 9`, computed differently at `signature_tokens_prelude.rs:292-293` and
`cached_attention_render.rs:103-106`) is read through one helper in the new lowering. Windowed layers stay `Kept` until P2-16d flips them; `bands` is what makes
the flip arm and the worked example expressible.

### 5.5 No silent threadgroup clamp

Today `dispatch` clamps a REQUIRED width: `Some(width) => width.min(pipeline.maxTotalThreadsPerThreadgroup())` (`T:omega/src/metal/resident_nocopy_cache.rs:1334-1338`,
copied at `arena_encode_dispatch_finish.rs:751-755`), while its own doc says a required width is honoured exactly (`:1305-1309`). A cooperative kernel that assumes
`simdgroups * 32` then indexes uninitialised threadgroup memory. The real limit is `pipeline.maxTotalThreadsPerThreadgroup()`, read from the compiled pipeline (it
depends on register use, so it is unknowable at emit time; `pipeline_for` already reads it at `pipeline_buffers_upload.rs:452-454`).

Fix: `dispatch(..) -> Result<(), MetalError>`; `Some(width) > limit` returns `MetalError::ThreadgroupWidthExceedsPipelineLimit { entry, required, limit }`; `None`
(grid-derived width, an occupancy hint) keeps its `min`. `pipeline_for` performs the same check right after `compile_pipeline`, so the error names the entry once,
at creation. The flash kernel additionally carries `[[max_total_threads_per_threadgroup(512)]]`, so the compiler constrains registers to the declared width or
compilation fails loudly. Three call sites change (`device_buffers_arena_plan.rs:1667`, `arena_encode_dispatch_finish.rs:1566,1622`). This also repairs every
existing cooperative kernel with a required width.

## 6. Blast radius

### 6.1 Per model (what the crate serves)

Registered architectures: `QWEN35`, `QWEN35MOE`, `GEMMA4`, `DENSE` (`T:proxima-model-interop/src/architecture.rs:617-619`, `dense.rs:59`). LFM2 is not an
`Architecture` impl. "Fusion goes default-on" means: the gemma4-shaped matchers join `cached-attention-streaming` (already in `metal`,
`T:proxima-model-interop/Cargo.toml:128-147`), and the reassociating lowering is the new tiled one.

| model | attention program and recognizer arm | fused today (`metal` default) | after |
|---|---|---|---|
| gemma4 E2B | two-range (`lfm2_two_range_cached_forward_program_with_experts`), 7 full and 28 windowed layers, scale 1.0 | 0 of 35: the recognizer declines 28 at `mask_form` and 7 at `cached_scale_shape` [P: memory `project-gemma4-attn-fusion-recognizer-gap`; re-measured by P2-1] | 7 full layers `Fused`, 28 windowed `Kept` (until P2-16d); decode, verify and prefill on one pipeline; decode attention-span BoundOps 525 to 427 [D] (Section 3.8); only gemma4 has a verify program today (`architecture.rs:318-332`, `gemma4/bind.rs:1146-1152`) |
| qwen3.6 (`QWEN35MOE`, `QWEN35`) | two-range with pass plane; `single_position_step: true` (`qwen35moe/bind.rs:289`, `qwen35.rs:708`), so prefill is one position per evaluation (C = 1) | fused (bare padding select), legacy arms | same op, new lowering: arithmetic changes (reassociated, about 1e-6 [A]); no Q-tiling benefit at C = 1, decode-shaped M2 only; decode non-regression measured (P2-16e) |
| qwen3-8b, mistral, qwen2 (`DENSE`) | prefill: two-range program; decode: single-range placed KV with a runtime `new_upper` (`load_model.rs:1295`) | fused, legacy arms | same; both forms lowered by the new pipeline; the single-range form maps the merged buffer as the "new" range with a runtime upper band |
| LFM2 (`lfm2.rs`) | `run_lfm2_prefill`: cacheless whole-sequence program per generated token (`lfm2.rs:1-12`) | no cached anchor exists | unchanged; `attention_outcomes` returns an empty list (asserted, count 0). Its attention does scale with L; routing it needs a cached LFM2 program, which is an LFM2 port, named here and not designed |

Speculative decoding on non-gemma4 architectures (spec R2) is not built; when a dense verify program lands it inherits R1 from this lowering instead of needing
its own proof, which the gemma4-only alternative would not give. Under a policy without `reassociation` nothing changes for any model except that gemma4 full
layers become fused (sequential arm) instead of unfused.

### 6.2 Baseline and `passed_tests.sh`

`S:passed_tests.sh` keeps only PASS lines (`parse_passed` prints `$(NF-1), $NF` for `PASS`) and runs nextest at `--status-level pass`, so FAIL and SKIP never enter
`before.txt`, and a test compiled out under a feature does not appear at all (the R18 baseline never exercised `metal-fuse-attn-decode`, which `metal` does not
include). Changes:

1. `--status-level skip`; the parser emits `STATUS crate test` for PASS, FAIL and SKIP, sorted, and prints `# totals pass=P fail=F skip=S`, asserting
   `P + F + S == lines` and `P > 0` (N == 0 is RED).
2. Self-test: the 5-line sample prints 4 lines (2 PASS, 1 FAIL, 1 SKIP), not the 2 it prints today.
3. Two baselines at the current tree: `before_status.txt` (as shipped) and `before_status_fuse.txt` (the same run with `proxima-model-interop/metal-fuse-attn-decode`
   added). Tests behind the feature appear only in the second; its FAIL and SKIP lines are the concrete blast-radius evidence P2-10 must turn green or delete by name.
4. The regression compare is a join on `(crate, test)` and prints transitions: `PASS -> FAIL`, `PASS -> SKIP`, `PASS -> ABSENT` are regressions; `SKIP -> PASS` and
   `ABSENT -> PASS` are gains. Deleted tests (Candidate B and the legacy arms) are listed by name in the commit that deletes them, and the count of `PASS -> ABSENT`
   must equal that list's length.

### 6.3 Thread counts and the 64-bit grid fix

Program-wide max per dispatch at C=512, gemma4 E2B, after routing [D from the coefficients recorded in `S:TASKS.md:322-323,350-356`; the census test recomputes
it in P2-12]:

| dispatch | threads |
|---|---|
| FFN gate/up reduce, 12288-wide layers (`rows * 3,145,728`, 12288 * 256 lanes) | 1,610,612,736 (37.5 percent of 2^32), the program-wide max |
| per-layer-embedding projection, node 17 (`rows * 8960 * 256`) | 1,174,405,120 |
| sliding attention, unfused, ring (L = 1024 keys: `512*1024*8*64`) | 268,435,456 |
| flash attention, full layers (M1) | 131,072 |

None scales with L after routing. The row limit from the FFN is `floor((2^32 - 1) / 3,145,728) = 1365` rows, matching the recorded "about 1,365 rows" [D]. Any chunk
of at most 1365 rows is safe; C=512 is.

Dependency on slot-0-3f's 64-bit grid fix (in flight): no P2 slice requires it to land first, by design. (a) Payload capture runs at `ubatch = 512`, not
one-evaluation. (b) The unfused control at L >= 16,384 runs at C <= 128 (`128 * 32768 * 8 * 64 = 2.1e9 < 2^32`), not C = 512. (c) The flash grid is a 3-D
`dispatchThreadgroups`, so it never linearises to a `uint gid`. (d) Each gate asserts `max_threads < 2^32` as a precondition from the census (a count, so it fails
RED if the census emits nothing). Gates that WOULD need the fix and are therefore not defined: any arm at `ubatch > 1365` or one-evaluation prefill above 1365 rows;
re-recording `gemma4_base_tokens.txt` (AC12/AC13), which belongs to that fix (`S:TASKS.md:316-330`). After the fix lands, control arms may move to C = 512 and the
precondition becomes a tripwire.

## 7. Spec integration and slices

### 7.1 SPEC.md additions

Amend `S:SPEC.md:131` (the R11/R12 paragraph). Replace "No kernel change: the fused cached-attention op already reads its live row count from a rank-0 input, and
`cached_attention_candidates` now accepts either `cached_len` or `cached_len_swa` as that input." with: "The ring itself changes no kernel: the fused cached-attention
op already reads its live row count from a rank-0 input, and `cached_attention_candidates` accepts either `cached_len` or `cached_len_swa` as that input. The prefill and
verify attention kernel changes (Q-tiled lowering, canonical fold, routing) are R19 to R28, designed in `design-p2-fused-attention-v2.md`." Line `:119` ("No kernel
changes") is about YaRN and stays true. Add `omega/omega-runtime.toml [flash_attention]` and this file to the context list. SPEC status returns to `draft` until
re-audited; the re-audit is a gate before P2-6.

| id | requirement | testable in isolation |
|---|---|---|
| R19 | Attention memory of every full-attention layer of the served models is independent of context length at fixed chunk C: no dispatch scales with L | yes |
| R20 | One pipeline and one association order serve decode, verify and prefill: a row's output bits are a function of its key set only (speculative R1 under fusion) | yes |
| R21 | Full-attention anchors route to the flash lowering, windowed anchors are `Kept`, every other outcome is a typed `DeclineReason`; `flash_attention = true` refuses on any `Declined` | yes |
| R22 | The flash kernel's output on the real gemma4 payload is no further from llama.cpp's output than the unfused Metal incumbent's (kernel level on the same bytes, model level on a real run), by NMSE thresholds fixed before the kernel exists, and inside llama.cpp's own 5e-4 bar | yes |
| R23 | The fold uses exactly the visible keys: exact-count support test, both directions, dtype-safe; the worked example holds | yes |
| R24 | The windowed real-model gate: an in-window fact beats an out-of-window distractor, at least as often as llama.cpp (fixture answers) | yes |
| R25 | A required threadgroup width above the compiled pipeline's limit is a typed error, never a clamp | yes |
| R26 | The regression baseline records PASS, FAIL and SKIP with a printed total | yes |
| R27 | Prefill and decode against Ollama (2K, 8K, 32K), sliding routing, and per-family decode non-regression are recorded as measurements with pre-stated validity rules | yes |
| R28 | The attention dispatch population per token is counted before and after from the real program | yes |

### 7.2 Acceptance criteria (count-bearing)

Run from the working tree root after `source proxima-tensor/specs/long-context/env.sh`. `$F` = `--features std,metal`.

| id | discharges | command | expected |
|---|---|---|---|
| AC24 | R19 | `cargo nextest run -p proxima-model-interop $F flash_attention_census` | 3 passed (L = 8192, 32768, 131072), each printing `anchors=35 fused=7 kept=28 declined=0 max_threads=1610612736 over_u32=0 emitted=N` with N > 0 |
| AC25 | R20 (internal invariant) | `cargo nextest run -p omega flash_rows_independent_of_row_count flash_resident_equals_split_merge flash_pipeline_key_shared_by_decode_and_verify` | 7 passed (3 shapes each for the first two, 1 for the key test), printing `dispatch_shapes=8 mismatching_rows=0` and `mismatching_elements=0` |
| AC26 | R20 (internal invariant) | `cargo nextest run -p proxima-model-interop $F speculative_matches_plain_on_synthetic_gemma4 flash_vs_unfused_bits_differ` | 3 passed (cpu, metal, control); the first two print `tokens=64 verify_steps>=1 identical=true`, the control prints `differing_elements>0` |
| AC27 | R21 | `cargo nextest run -p proxima-tensor --features std,cached-attention-streaming attention_outcomes`; `cargo nextest run -p proxima-model-interop $F flash_attention_demand flash_attention_agrees` | 7 passed; 3 passed. The synthetic gemma4 prints `anchors=10 fused=2 kept=8 declined=0`; an unmatchable-mask fixture prints `declined=1` and the demand returns `FlashAttentionDeclined` with `declined.len() == 1`; a table-driven test prints `variants_covered == variants_total` |
| AC28 | R22 | `cargo nextest run -p omega flash_error_no_worse_than_incumbent` (kernel level, G1 fixtures) and `cargo nextest run -p proxima-model-interop $F flash_model_error_no_worse_than_incumbent` (model level, G2 fixtures); `git rev-list --count <thresholds commit>..<kernel commit>` | 2 passed (full, sliding) and 2 passed (layer 4, layer 0), each printing `nmse_fus<=T_nmse rho_max<=T_max nmse_fus<=5e-4 controls=64 emulation_check=ok fixture_commit=f1ea20621` and, labelled internal, `rho_rms_f64<=T_f64`; count >= 1 |
| AC29 | R23 | `cargo nextest run -p omega flash_support_exact_count flash_support_controls` | 6 passed and 3 passed |
| AC30 | R23 | `cargo nextest run -p proxima-tensor -p omega flash_worked_example worked_example_matches_llama_fixture` | 5 passed (windowed and full tables on CPU and Metal, plus the llama.cpp fixture check), 40 cells within 1e-5 |
| AC31 | R24 | `niah --model "$GEMMA4" --window-facts --cases 5`; then `--drop-correction`; then `--ring-offset 1` | `proxima found=X/5 llama found=Y/5` (Y from the vendored llama.cpp fixture) with Y >= 4 and X >= Y; control: both answer the distractor in 4 or more of 5; ring-offset X strictly below the unmodified X |
| AC32 | R25 | `cargo nextest run -p omega dispatch_rejects_width_over_pipeline_limit flash_pipeline_supports_declared_width` | 2 passed; the second prints `required=512 limit>=512` |
| AC33 | R26 | `$LONG_CTX_SPEC/passed_tests.sh --self-test` (count its lines), then the baseline totals line | 4 lines; `# totals pass=P fail=F skip=S` with P > 0 and P+F+S = lines |
| AC34 | R27 | `niah --model "$GEMMA4" --ctx 8192 --prefill-pairs 3`; then `--ctx 32768 --prefill-pairs 3` | 3 pairs each, `prompt_eval_count == proxima tokens`, `pair_cov <= 0.05`, `contaminated=0`; then the recorded ratio and the applied follow-up rule (3.6) |
| AC35 | R27 | `niah --model "$GEMMA4" --ctx 2048 --decode-pairs 3 --decode-tokens 64`; then `--ctx 32768` | 3 pairs each, `ollama eval_count=64 proxima decode_tokens=64`, `pair_cov <= 0.05`, `contaminated=0`, per-build ms/token, the `a + c*N` fit with its residual, and the applied rule (3.8b) |
| AC36 | R27 | the P2-16d and P2-16e tables | 5 or more runs per arm, `sigma_total` printed, flip decision by the 2-sigma rule, decode arms for gemma4, qwen3.6 and qwen3-8b |
| AC37 | R28 | `cargo nextest run -p proxima-model-interop $F attention_dispatch_census` | 3 passed (`new_count` 1, 49, 512), each printing `chain_ops_per_layer=K before=35*K after_full_only=7+28*K after_all=35`; asserts the identities `after_full_only == before - 7*(K-1)` and `after_all == 35`; expected `K=15`, `before=525` [P]; a different K is itself the finding. Under `instrument` also prints `physical_dispatch_calls` |

### 7.3 Slices

Each slice is one commit and one behaviour change. Cargo and Metal slices wait for slot-0-3f's timing window, as in `S:TASKS.md`. Writing code is sonnet, running and
measuring is haiku, judgment stays here.

| # | slice | validation | expected |
|---|---|---|---|
| P2-0 | `passed_tests.sh` records PASS/FAIL/SKIP and totals; two baselines (`before_status.txt`, `before_status_fuse.txt`) | AC33 | 4 lines; P+F+S = lines, P > 0 (F for the fuse baseline recorded as data) |
| P2-1 | `AttentionAnchor`, `AttentionOutcome`, `DeclineReason`; recognizer split into `match_anchor`/`route`/`build_fused`; no behaviour change; `attention_outcomes` public | `attention_outcomes` tests; `awk` count of `continue;` in `cached_attention_candidates` | 7 passed; 11 (only the anchor non-matches remain). Real gemma4 prints the current decline histogram as data |
| P2-2 | payload capture example; llama.cpp generators G1 and G2 run once on the CPU backend and their outputs vendored (commit and command in the README); incumbent-vs-llama NMSE and (internal) incumbent-vs-f64 error; 64 control reorderings; `payload_thresholds.toml` committed | the example run, the generator runs, the manifest check | 18 tensors found (9 per layer); G1 files 4 (full and sliding, f32 unfused and f16 flash); G2 files 2 (`kqv_out-0`, `kqv_out-4`); `token_ids_equal=true n=2048`; `controls=64`; `emulation_check=ok`; thresholds file written |
| P2-3 | window real-model harness (`--window-facts`, `--drop-correction`) with llama.cpp answers vendored, baseline at the unflipped tree | AC31 on the unflipped tree | 5 cases and 5 controls in the fixture; `Y >= 4` from llama; `X0` recorded |
| P2-4 | `bands` replaces the two scalars (72 refs, 12 files); the CPU executor reads the pair; worked example both tables on CPU | `git grep -c` for the two old field names; `flash_worked_example` (cpu) | 0; 2 passed; baseline compare 0 regressions |
| P2-5 | `dispatch` returns `Result`; typed width error; `pipeline_for` check | AC32, first test | 1 passed; baseline compare 0 regressions |
| P2-6 | audit gate first (re-audit the SPEC, open FA-2 Algorithm 1 and record the lines); then the flash lowering M1: `render_flash_attention`, entry name without q/c/n, Safe pin, 3-D grid, sized constants, policy-selected next to the sequential arm | AC30 (metal), AC29, AC32 second test | 2 + 6 + 3 + 1 passed |
| P2-7 | M2: partition partials to scratch, merge kernel, planner mode rule | `flash_resident_equals_split_merge` | 3 passed, `mismatching_elements=0` |
| P2-8 | accuracy gate: sweep `NPV`, `Ls` on the real payload; fix the constants in `omega-runtime.toml` | AC28 | 2 passed; chosen `NPV`/`Ls` recorded with both `rho` values |
| P2-9 | T1 row-count independence and pipeline-key equality | AC25 | 7 passed |
| P2-10 | routing edit (Section 5.2), Candidate B unreachable, feature folded, real gemma4 outcomes | AC27 first command; real gemma4 outcomes | 7 passed; `fused=7 kept=28 declined=0`; every FAIL in `before_status_fuse.txt` resolved or named |
| P2-11 | synthetic gemma4 GGUF writer; T2 speculative test and control | AC26 | 3 passed |
| P2-12 | census: threads and memory (real gemma4, L in {8192, 32768, 131072}, C=512) and attention dispatch count (`new_count` in {1, 49, 512}) | AC24, AC37 | 3 passed and 3 passed; `max_threads=1610612736`; `K` and `before` printed |
| P2-13 | demand bit: `BackendRuntime` check, `InteropError::FlashAttentionDeclined`, admission check, parity test | AC27 second command | 3 passed |
| P2-14 | delete `CachedSoftmaxWeights` (96 refs), the legacy chunk/block/split arms, the `context_chunks_for` family, `[attention_*]` config, the feature `metal-fuse-attn-decode`; re-document the sequential arm citing no llama.cpp | `git grep -c` over those five names; full compare with statuses | 0; `PASS -> ABSENT` equals the named deleted list, every other transition 0 |
| P2-15 | real-model gates: four facts (each answer also checked against llama.cpp's fixture answer), window gate, needles at 8192 (ASCII haystack until the tokenizer fix), AC19 | `gemma4_e2b_answers`; AC31; `niah --ctx 8192 --needles 10`; AC19 | 4; X >= Y >= 4; X >= Y with Y >= 9; `kv_bytes=811597824` and X >= Y |
| P2-16 | measurements: (a) tile sweep over `s`, simdgroups, `Sk`, Safe vs Relaxed compile; (b) prefill pairs 8K and 32K; (c) decode pairs 2K and 32K; (d) sliding routing 2-sigma rule; (e) decode non-regression per family (gemma4, qwen3.6, qwen3-8b) | AC34, AC35, AC36 | recorded with CoV; no verdict; follow-up rules applied; (c) re-run with the third build after P2-17 |
| P2-17 | conditional on the P2-16d rule: route windowed anchors, phase-align the ring scratch (Section 5.1), ring-offset T1 cases, T2 in both routings | AC25 (ring cases), AC26 (4 passed), ring parity numbers of AC12/AC13 | ring cases `mismatching_rows=0`; 4 passed; `full vs head` and `ring vs head` at their recorded values |

## 8. Structural checks

Central claim as a lint. Not pipes, and why each is justified:

- `BoundOpKind::CachedAttention`, its recognizer and its MSL lowering are tensor-graph algebra (`Op`/`BoundOp`), not `Pipe`s. proxima-tensor and omega are the
  tensor compiler; a `Pipe` is a runtime request/response surface. The consumer-facing edge (`LoadedModel`, `ServingConfig`) is unchanged and stays the existing
  surface.
- `AttentionAnchor`, `AttentionOutcome`, `DeclineReason` are data, not behaviour. Second question, call sites both ways: before, `bind_with_fusion(..)` returns ops
  and the reason for each non-fusion is gone; after, `attention_outcomes(..)` returns it. What a caller can do that they could not: refuse loudly under demand, list
  anchors for payload capture and census, count outcomes in a gate. `AttentionAnchor` exists because the recognizer already holds those 20 locals and loses them;
  splitting the function without it would be the compensator.
- `MetalError::ThreadgroupWidthExceedsPipelineLimit` and `InteropError::FlashAttentionDeclined` are error variants.
- No new config type, no new `Pipe`, no blanket impl, no wrapper type. `KeepReason` was considered and folded into `Kept { window }`. `FlashAttention::{Auto, On, Off}`
  was rejected by the call-site test (Section 5.3).

Information destroyed, found and closed: the decline reasons at `continue`; the new-range lower bound at `types_layout_boundop.rs:210-211`; the ninth operand's
meaning inferred from `cached_key_rows == 0`; the required threadgroup width lost to a `min`; the baseline's FAIL and SKIP lines.

Abandoned designs (constraints applied before the design, not after):

1. v1: route prefill to the existing online kernel and keep Candidate B for decode. Abandoned: two arithmetics for one row (R1), and the kernel re-reads K/V per row
   (2 TiB per full layer-chunk at L=131072 [D]).
2. Keep Candidate B and extend it to multi-row for verify. Abandoned: it materialises exp per key, so verify at 49 rows and 131072 keys is 49 times a per-row weight tensor
   per layer, and it still differs from prefill.
3. Make decode agree with verify by pinning the legacy chunk count. Abandoned: the count is a function of the dispatch extent (`context_chunks_for`), measured to
   differ on 28 of 49 rows at short context; the partition must move off the extent onto the key index.
4. One sequential fold over all keys, no `Ls`. Abandoned: decode at 131072 keys would run one simdgroup per (row, group) with no key parallelism and cannot fill the machine.
5. `simdgroup_matrix` for QK and PV in the R1-bearing kernel: abandoned for the first arm (Section 3.7).
6. `FlashAttention::{Auto, On, Off}` (v1): abandoned (Section 5.3).
7. Routing all 35 layers unconditionally (v1): replaced by `Kept` plus a measured flip with the ring phase alignment designed (Section 5.1); a tile grid anchored to the ring's
   moving scratch row 0 was abandoned because it breaks R1.
8. Gating recognizer admission through a new `NumericRewrite` so `bit_exact` declines fusion: abandoned. The bit-exact lowering already exists at emit, and gating
   admission would flip fused-op expectations across the 180 default-policy sites in `omega/src/msl/tests.rs` alone for no gain.
9. A two-modulus O3 signature: abandoned after the measured (leak, drop) alias at n = 131072.
10. The f64 definition or proxima's own CPU path as the accuracy oracle (v1's oracle stack): abandoned under the owner's 2026-09-29 rule. The f64 reference stays as a labelled internal
    tightening check, because the llama-defined distance is dominated by llama's own rounding and cannot alone select `NPV` (Section 4.4, measured).
11. Ollama as the correctness oracle for the facts and window gates: abandoned. Ollama's gemma4 appears to be its own Go engine (the `gemma4.go` citations in `gemma4/bind.rs`) rather than llama.cpp [A]; it stays as the performance incumbent only.
12. Live llama.cpp at test time (a runtime or dev-dependency link): abandoned; the generators run once, externally, and only their outputs are vendored.

Observations not acted on, one line each: the capture hook is an env-gated file dump (Section 4.3); `NumericPolicy::llama_relaxed` names an Apple math mode with a llama
label (`numeric.rs:64`), a rename outside this design; `serving.rs:1113-1121` documents `flash_attention` as unimplementable, which this design makes stale and slice P2-13
corrects.

Unmeasured or assumed, that could flip a decision: the real-payload thresholds (only the procedure is demonstrated); whether 512 threads fit the compiled pipeline; the
per-core threadgroup memory pool (the `s = 2` column); system-level cache reuse of the shared K/V stream; the FA-2 algorithm line numbers quoted from memory; the `Safe`
compile cost; the FFN coefficient and the 15-op chain (recorded by prior sessions, re-run by P2-12 and not here); decode non-regression on qwen3.6 and qwen3-8b; the head
grouping `G` of qwen3.6 (read from the op); the real G1 and G2 fixtures (only the procedure and a synthetic run against the external build are shown; whether llama.cpp's gemma4 `kqv_out-4` is comparable tensor-for-tensor with proxima's anchor output is read from source, not run); the gemma4 tokenizer fix, on which every real-text fixture depends; the `a + c*N` linearity of decode time in dispatch count (three points fit two parameters). Nothing about MLA or architectures
beyond those served is claimed.
