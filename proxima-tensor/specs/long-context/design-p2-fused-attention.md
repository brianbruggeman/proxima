# P2: L-independent fused attention (v1 REJECTED 2026-09-29; v2 in progress)

## Why v1 is rejected (proxima-critic, same day)

The core premise survives. The existing online-softmax `CachedAttention` plus mask bands
is the right substrate, and the thread and memory figures reproduce. The worked example
reproduces in all 20 cells.

Blockers:
1. **Speculative byte-identity.** Routing multi-row attention to the online kernel moves
   speculative verify rows (k+1) onto different arithmetic than 1-row decode (Candidate B).
   That breaks main's speculative SPEC R1, which is default-on, and no gate sees it
   (`dead_code_cached_attention.rs:1201-1230`).
2. **Oracle evidence.**
   - Wrong baseline: the CPU harness modeled a sequential f32 "two-pass", but the incumbent
     unfused path is a width-64 cooperative tree reduce.
   - The 2x/4x thresholds have no derivation. Per-row ratios reach 10.2.
   - No real gemma4 payload was used, and Metal relaxed math and chunk merge are not
     modeled.
   - "Not met" (`Cargo.toml:37-38`) was turned into "wrong bar" without principle-14 proof.
3. **The routing edit misses the real gate.** `softmax_weights_eligible = via_gemma_template`
   (`:1241`) sends everything down the Candidate B branch. Deleting the decode-only decline
   alone never reaches the online kernel.

Major:
- The C-invariance and old-band bit-identity claims contradict the chunked key partition.
- `FlashAttention::On` has no admission site, and the decline reasons are discarded
  (about 49 `continue`s).
- The flip's blast radius over lfm2/qwen35 is unenumerated, and the baseline records only
  PASS lines.
- **No performance model.** The two-range kernel re-reads all K/V per row
  (4096x re-read, about 2 TiB per full layer-chunk at 131072). Frontier prefill Q-tiles.
- A 768-wide threadgroup is silently clamped, and the merge then reads uninitialized
  memory (`resident_nocopy_cache.rs:1334-1338`).
- O4 is too weak for the sliding window: the fact prompts are shorter than W.
- O3 poison 1e30 overflows f16/Q8_0 storage and detects over-inclusion only.
- S5's "max threads 393,216" ignores the FFN (1.61e9 at C=512).
- P2 is not in SPEC (no R/AC). SPEC.md:131 says "No kernel change".

Minor:
- Mixed-tree line numbers.
- The 2.2e-5 noise figure is not reproduced.
- Scope covers the 28 sliding layers, which gain nothing.
- The generalization claims are unmeasured.

v2 must answer every item. Oracles stay llama-free (`feedback_no_llama_cpp_anywhere`):
an f64 definition reference, proxima's own unfused Metal path on real payloads, and the
Ollama runtime gates.

---

(v1 text follows)

Source: a proxima-architect pass on 2026-09-29, read at main 73fd1cbd plus this spec's
work in /private/tmp/long_ctx_main2. Tags: [R] read, [M] measured, [D] derived, [A] assumed.
The experiments are preserved in `<session>/long_ctx_backups/p2_experiments/`
(`softmax_orders.c`, `isolate.c`, `ratio.c`, `bound.c`, `walk.awk`).

## Thesis

No new primitive. The existing `CachedAttention` op and its Metal kernel are already the
flash shape: online softmax, multi-row, per-row register state
(`omega/src/msl/cached_attention_render.rs`). P2 makes three changes:
1. `BoundOpKind::CachedAttention` carries `bands: [CausalBand; 2]` (cached range, new
   range) in place of two scalars. The CPU reference already uses `[CausalBand; 2]`
   (`physical.rs:308-311`), and the bind step currently destroys that information
   (`types_layout_boundop.rs:210-211`).
2. The gemma recognizer routes prefill to the online kernel.
   - Delete `local_window_not_vacuous` (`dead_code_cached_attention.rs:1145-1155`). It is
     a sound vacuity guard, redundant once the new-side lower band exists.
   - Turn `softmax_weights_decode_only` (`:1221-1240`), a missing proof, into a
     fall-through.
3. The raw-bit fused-vs-unfused bar is replaced by a four-part oracle stack (O1-O4 below).

## Premise corrections [R]

- The window guard declines only when `new_key_rows > W`. With the ring, sliding-layer
  scores are already L-independent (about 16.8 MB at C=512). Only the 7 full layers scale.
- The window admits W keys including self: `q-k > W-1` is masked (`primitives.rs:1164-1170`).
- Unfused thread count is `C*L*Hq*64` for full layers (wide cooperative reduce, width 64).
  It hits 2^32 at C=512 and L=16384. At 131072 the largest safe power-of-two C is 32.
- The documented review behind the default-off feature (`attn_decode_fusion_review.md`)
  does not exist in the tree. The "bar" is only harness code (`first_diff_f32`,
  `decode.rs:123-131`).
- The R18 baseline never exercised `metal-fuse-attn-decode`-on tests, because `metal`
  does not include that feature.

## Why bit identity is the wrong oracle [M]

The measurement is a C harness, `-ffp-contract=off`, libm `expf`, D=8. It compares online
softmax (the MSL order) and two-pass f32 against an f64 reference.

| n | sigma | bit-equal rows | max err, two-pass | max err, online |
|---|---|---|---|---|
| 16 | 1 | 94/2000 | 3.4e-7 | 3.7e-7 |
| 512 | 1 | 12/2000 | 3.4e-7 | 3.3e-7 |
| 8192 | 1 | 6/2000 | 2.8e-7 | 4.2e-7 |
| 131072 | 1 | 0/200 | 2.6e-7 | 2.3e-7 |
| 131072 | 4 | 1/200 | 3.44e-4 | 3.39e-4 |

- The ratio of mean errors is 0.95-1.02 and the ratio of max errors is 0.77-1.47.
- The dominant error is the sequential f32 fold, which both orders share.
- A tolerance cannot detect a leaked masked key at long L: one key's weight is about 7.6e-6
  against an accumulation noise of about 2.2e-5 [D].

## Oracle stack

- **O1:** an f64 definition reference, plus an exact-rational worked example.
- **O2:** per (n, sigma) cell, >= 1000 rows. Mean err <= 2x the two-pass mean, and max err
  <= 4x the two-pass max. These are pre-registered from the CPU experiment and must be
  re-checked on the Metal grid.
- **O3:** a support test. Masked keys carry a 1e30 poison V, so any leak is visible.
- **O4:** external. The four world facts (AC12x) and NIAH X >= Y (AC15).
- Bit identity is kept only where the arithmetic is unchanged: Candidate B decode, the new
  kernel vs the old kernel on the old bands, and C-invariance.

## Bounds after P2 [D]

- Fused threads are `C*Hq*chunks*32`, with chunks capped independently of L.
- For E2B at C=512: sliding layers 393,216 threads (768-wide threadgroup), full layers
  131,072. That is about 10,900x under 2^32 at any L.
- Transient output is 4 MiB (sliding) and 8 MiB (full).
- For comparison, the unfused score buffer at C=512, L=131072 is 2 GiB.
- Decode stays on Candidate B, with O(L*Hq) scratch of 4 MiB at 131072.
- Unverified: whether a 768-wide threadgroup fits the pipeline's max threads.
- Adjacent: qwen3 single-range split scratch at C=512 is 272,629,760 B. It is
  L-independent but exceeds `transient_cap`; S8 measures it.

## Signatures

```rust
pub struct CausalBand { pub lower_inclusive: i64, pub upper_inclusive: i64 } // alloc tier POD
BoundOpKind::CachedAttention { /* existing fields */ bands: [CausalBand; 2] }
pub enum FlashAttention { #[default] Auto, On, Off } // replaces flash_attention: bool (-fa auto|on|off)
```

- MSL: a single band predicate, `relative < lower || relative > upper => continue`, plus
  first/last-key loop bounds.
- `FlashAttention::On` returns `UnsupportedServingConfig` when any layer declines fusion,
  because a silent decline is a silent return to O(C*L).

## Slices

| # | slice | validation | expected |
|---|---|---|---|
| S0 | feature-on regression baseline `before_fuse.txt` | passed_tests.sh with `+metal-fuse-attn-decode` | N_f > 0 |
| S1 | `bands` replaces the two scalars, with no behaviour change | grep of the old field names; `comm` against both baselines | 0; 0; 0 |
| S2 | kernel band predicate plus the worked example, on CPU and Metal; bit identity to S1 | the three tests | 2; 2; 3 passed |
| S3 | recognizer routing; window-guard deletion; shared-KV prefill fixture | counts test | prefill(600,0) attention = 2 (was 0); shared fixture = 3 |
| S4 | oracle harness O2/O3 plus the Metal grid | three tests | 9; 12; 12 passed |
| S5 | no-GPU census of the real gemma4 at C=512, L=8192 and 131072 | `attn_prefill_l_independence` | `cached_attention_ops=35`; max threads 393,216 at both L; all < 2^32 |
| S6 | real-model gates | the four facts; niah 8192; AC19 at 131072 | 4; Y >= 9, X >= Y; kv_bytes and X >= Y, X > 0 |
| S7 | flip `metal-fuse-attn-decode` into `metal`; `FlashAttention` enum | passed_tests.sh diff; flag tests | 0 except the counts test; 3 passed |
| S8 | measurement only: ms per chunk and the transient at L 8K/32K/131K; qwen3 scratch | table | no verdict |

## Worked example (S2 test)

- **Setup:** Hkv=1, G=2, hd=2, W=3, scale=ln2. Two cached keys, five new keys at
  positions 2-6.
- **Keys:** k_even = 1 0 1 0 1 0 1; k_odd = 0 1 2 0 1 2 0. V(k) = (k, 6-k).
- **Queries:** h0 query (1,1). h1 query (0,-1) on even rows, (1,-1) on odd rows.
- **Bands:** cached {-2, MAX}, new {-2, 0}.

Expected outputs, all exact rationals:

| row | h0 | h1 |
|---|---|---|
| 0 | (3/2, 9/2) | (4/7, 38/7) |
| 1 | (21/11, 45/11) | (9/4, 15/4) |
| 2 | (35/13, 43/13) | (22/7, 20/7) |
| 3 | (13/3, 5/3) | (11/3, 7/3) |
| 4 | (24/5, 6/5) | (37/7, 5/7) |

- An independent awk evaluator matches all 20 cells.
- **Control:** the pre-P2 bands differ on rows 3-4, with a smallest gap of 0.30.
- **Tolerance:** 1e-5 absolute.

## Generalization

One primitive with four parameters:
- **Mask bands:** causal, sliding, bidirectional. Chunk-local attention (llama4) needs a
  `#[non_exhaustive]` variant later.
- **Key layout:** rotary/pass planes and GQA groups.
- **value_dim:** needed for MLA.
- **KV layout:** two-range or single-range. Paged KV would come in through the existing
  Lookup slot.

MLA in absorbed form (DeepSeek-V3/Kimi-K2) is MQA over a latent: `value_dim`, V aliasing
the pass plane, strides from Layout. It must use the registers-only two-range arm, because
the single-range `shared_o` exceeds the threadgroup budget at G=128.

Teaching hook: the (m, l, o) kernel state is a monoid under merge. That is why chunk merge
and KV split are legal, and why the mask only sets the domain of the fold.

## Abandoned

- **Extend Candidate B to prefill:** it materializes exp per key, so O(C*L).
- **A two-pass recompute bit-identical kernel:** 2-3x the QK cost, and coupled to unfused
  topologies.
- **A new FlashAttention BoundOpKind:** the existing kind already has the flash shape.
- **P1 with C(L) shrinking:** C=32 means 4096 chunks at 131K, each re-streaming all weights.
