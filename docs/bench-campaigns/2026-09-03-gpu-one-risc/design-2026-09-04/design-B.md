# design-B: the Metal decode path as bytes, pipes, sans-IO FSMs, and RISC algebra

Every `file:line` below was opened in this session. Numbers carry provenance tags:
MEASURED (a counter/timer someone ran), DERIVED (computed from other numbers),
ASSUMED (spec sheet or literature). No verdicts.

## The shape chosen, and the one contested decision

**Shape.** One new bound kind — `BoundOpKind::Loop`, an ordered list of accumulation
*stages* over one shared iteration space — replaces `Elementwise`, `Reduce`, and
`CachedAttention` (net −2 kinds). Attention is not an op; it is a five-stage `Loop`,
and the flash kernel is a *lowering theorem* over `(Maximum-fold, exp-of-difference,
Add-fold)` that any program with that shape gets and any emitter may decline without
changing results. The four census fusion levers and attention are then the same
mechanism. Planning becomes pure (`proxima-tensor`, no device), the schedule (arena,
barriers, uniforms, commands) becomes plan-time data, and `omega` shrinks to one
driver with one edge: `submit(&[Command])`.

**Contested.** Whether attention needs a *tuple-valued reduce monoid* in the algebra
(a `(m, l, acc)` accumulator as a first-class `ScalarOp`-level thing). It does not,
and admitting one would be the fourth macro-op in a row. The accumulator tuple is a
*register allocation inside one kernel*, exactly as a matmul's accumulator is today —
invisible to `Op`, invisible to `BoundOp`. What the IR does need is the ability to say
"these five folds share one traversal", which is `Loop { stages }`, and that same
sentence is what rmsnorm, SwiGLU-epilogue and residual-add need. One extension, five
consumers. The rejected design is written out in §E.

**The linkage that decides the target.** 5x llama = 3.5 ms/token. Weight streaming at
the measured achieved bandwidth is 24.991 ms of the 33.0 (dispatch census, ROW 281/282);
dispatch overhead is ~6.3 ms of it. Sections A/B buy the 6.3 ms (616 → 264 dispatches).
Section D buys the 25 ms (4.17 GB → ≤1.4 GB). **Neither alone reaches 3.5 ms**; the
arithmetic is in §D.7 and it names the residual.
Two preconditions gate §D and are named there rather than assumed: the packed-row kernel
has no s-axis fold (`msl.rs:3171-3200`), so a k-token pass re-streams weights k times
until it does; and `classify_packed_row_block` requires `gather_count == 0`
(`msl.rs:1039-1052, 1389`), so a gathered weight falls to a serial kernel — which is the
arm that already measured dispatch-bound (ROW 180/181, `discipline.md:16430, 16471`).

---

# D. Bytes — the primary section

Every "exists today" claim here is anchored to a `file:line` I opened or that the
byte-levers probe verified; where the probe and my own reading differ, the probe's
line wins and I say so.

## D.0 The floor, and where 4.169 GB comes from

MEASURED (given, task line 91): 4.169 GB moved per generated token; M1 Max spec
400 GB/s ⇒ 10.4 ms floor if all of it moves. Target 3.5 ms ⇒ ≤ 1.4 GB/token at spec,
≤ 1.0 GB at a 300 GB/s realistic ceiling.

DERIVED decomposition (openchat-3.5 / Mistral-7B geometry read from
`proxima-tensor/specs/mistral_layer.toml:7-11`: `block_count=32, embedding=4096,
feed_forward=14336, head_count=32, head_count_kv=8, head_dim=128, vocab=32002`):

| group | params | bpw | bytes/token |
|---|---|---|---|
| FFN gate+up+down (3 x 4096x14336) x32 | 5.637 G | 4.5 (Q4_K) | **3.171 GB** |
| attn wq+wk+wv+wo (16.78M+4.19M+4.19M+16.78M) x32 | 1.342 G | 4.5 | **0.755 GB** |
| output.weight 32002x4096, Q6_K today (probe line 32) | 0.131 G | 6.5625 | **0.1075 GB** |
| norms (f32) | 0.00027 G | 32 | 0.001 GB |
| token_embd | 0.131 G | 4.5 | **not streamed** — one gathered row, 16 KB |
| KV read, t=512 ctx: 32 x 2 x t x 8 x 128 x 4 B | | f32 | **0.134 GB** |
| **total** | | | **4.168 GB** |

4.168 against the given 4.169 reconciles to 0.03%, which is the check that this is the
right decomposition to reason about. It says: **FFN is 76% of the per-token stream**, and
**KV grows 0.262 MB per token of context** (t=8192 ⇒ 2.147 GB, larger than every
attention weight combined).

Achieved bandwidth, MEASURED: packed-row matvecs move ~4.03 GB in 24.991 ms
(dispatch-census.md:27) = **161 GB/s against 400 GB/s spec**. That ratio is a lever in
its own right (D.6) and it is the one term I have no mechanism for.

## D.1 Multi-token per pass — lossless, and the one that needs a kernel

**Bytes.** Weight and KV bytes are paid once per *pass*. With mean acceptance `a`,
bytes/token = pass_bytes / a. At a = 2.5: 4.168 → **1.667 GB/token**. DERIVED.

**What exists (probe lines 4-12, verified):** the program and the plan cache are already
`new_count`-generic — `symbols = [new_count, kv_bound_extent]` (`generate.rs:2394`,
opened) and the cache key is `(new_count, bucket)` (`generate.rs:1248, 1323`). The `s`
axis is symbolic through every layer builder (`mistral_layer.toml:12-14`).

**What does not exist, and it is not the algebra:**

1. *The steady-state loop hardwires one token.* `next_ids = alloc::vec![token_id]`
   (`generate.rs:2503`, opened; also `2069`) and `sample_next_token` consumes a single
   logits row (`proxima-tokenizer/src/sample.rs:277`). §B.3's `Cursor { position,
   accepted }` is exactly this field, so acceptance is a value in the FSM, not a branch.
2. **The packed-row kernel has no s-axis fold.** It batches 4 *output* rows per simdgroup
   over ONE activation vector (ggml `nr0`; `msl.rs:3171-3200`), so a pass with `s = k`
   re-streams every weight row `k` times and the amortization does not happen. This is
   the load-bearing correction to any "multi-token is free" arithmetic: **D.1's byte win
   is conditional on an `nr1` s-fold in that kernel.** A weight-once multi-token kernel
   does exist — `metal-tiled-gemm` (`simdgroup_matrix`, Q4_K only, min-token gate;
   `omega/Cargo.toml:129`, `msl.rs:1575-1608, 3961-4030`) — and is unreachable from
   decode.
3. *No draft/verify machinery anywhere* (probe line 12).

**Algebra needed: none.** The matvec's iteration space already carries `s`
(`spec.rs:2404+` builds `"si->shdi"` / `"shdi->shd"` maps whose outer axis is `s`), so
the s-fold is an *emitter* capability, and under §A's `Loop` it is simply "this Loop has
one more free axis in `output_axes` that the kernel may tile". Tree-shaped proposals need
an arbitrary `[s,s]` acceptance mask, which is already an `Op::Input` on the two-range
path (`spec.rs:2360-2364` names `is_future` as an input built by `causal_mask`), so a
tree mask is **data**; §A.4's R4 simply finds no derivable band for it and the traversal
stays full over `s` (s = 8, negligible). Verification is host-side O(k) integer work in
`advance`, and placed-KV rollback is "do not advance `cached_len` past the accepted
count" (probe line 38) — one `Cursor` field, because the KV rows past the accepted prefix
are never read again (the band's upper bound is `cached_len + q`).

**Two binary questions:** nothing is added. The drafter is a second `Plan` in the
fixed-capacity plan table (§B.7); the acceptance test is
`Pipe<In = SampledStep, Out = Accepted>` composed with `.and_then`; an n-gram
prompt-lookup drafter and a small draft model are the same pipe shape (probe line 39).

**Quality gate: byte-identical.** Exact verification reproduces the target model's
sequence. Gate: 100 tokens byte-identical to `k=1` greedy at three seeds. Kill: any
divergence — it means verification is wrong, not that the lever is lossy.

## D.2 Lower-bit codecs — FFN is 76% of the stream

**Bytes.** DERIVED, FFN only (attention weights stay Q4_K; they are 18%):

| FFN codec | bpw | FFN bytes | total weights | probe factor (line 42) |
|---|---|---|---|---|
| Q4_K (today) | 4.5 | 3.171 GB | 4.034 GB | — |
| Q3_K (110 B/256) | 3.4375 | 2.422 GB | 3.285 GB | x0.75 |
| Q2_K (82 B/256) | 2.5625 | 1.806 GB | 2.669 GB | x0.6 |

**What exists (probe lines 28-32, verified):** `QuantizedBlock` (`cpu.rs:3091`) and
`PackedCodec` (`msl.rs:788-805`) carry F32/Q4_K/Q5_K/Q6_K/Q8_0 plus F16/BF16, and
`metal.rs:445-459` (opened) is the single place that maps block to codec.
**Q2_K/Q3_K/Q4_0/Q5_0 already parse** in `proxima-gguf/src/types.rs:109-133, 245-281`
and are rejected at bind with `UnrepresentableGgmlType`
(`proxima-model-interop/src/bind.rs:70-73`, `capability.rs:141-156`). So the work is
narrower than "add a codec": lift one rejection, add one dequant body per backend
(`cpu.rs`, `msl.rs`, `wgsl.rs`, `cuda.rs`), and — because the checkpoint on disk is
Q4_K_S — a requantizer in `proxima_gguf::quant` (probe line 42).

Pipe question: a codec is a *tag* on an operand read consumed inside a kernel body, not
behaviour composed at a boundary; it extends a closed set that already exists. Second
question: a caller can bind a Q3_K checkpoint that `bind.rs:70-73` rejects today. Both
answered.

**Quality gate (lossy — "text identical" is unavailable):**
- Metric 1: exact-match rate of greedy continuations vs the Q4_K_S model, 200 held-out
  prompts x 64 tokens, reported as a match-at-token-n curve, never one number.
- Metric 2: mean KL(full || quantized) on the last-position logits over the same set —
  the mechanism number that explains metric 1.
- Kill: exact-match-at-64 < 0.90, or mean KL > 0.05 nats.

## D.3 Dynamic row elision — the largest lever, and the one with a measured warning

**Bytes.** Eliding fraction `e` of the 14336 FFN rows reads `(1-e)` of gate/up rows and
`(1-e)` of down's columns. DERIVED:

| elision | FFN bytes | + predictor | total weights |
|---|---|---|---|
| 0% | 3.171 GB | — | 4.034 GB |
| 80% | 0.634 GB | 0.043 GB | 1.541 GB |
| 90% | 0.317 GB | 0.043 GB | 1.224 GB |

Predictor DERIVED: rank-128 scorer per layer = 4096x128 + 128x14336 = 2.36 M params, at
Q4_K = 1.33 MB/layer x 32 = 42.5 MB/token.

**The measured warning, first.** This was probed and it lost: the landed elision bench
(`proxima-tensor/benches/bench_dynamic_elision.rs`, ROW 180/181,
`discipline.md:16430, 16471`) built the skip set on the HOST, ran CPU-only, and measured
**dispatch-bound: 0.2-0.29 ns/element against a 0.057 ns/element DRAM baseline** (probe
lines 23-25). Reading the mechanism rather than the number: per-element gather addressing
cost more than the bytes it saved. So D.3's gate is **ns/element against the dense arm**,
not bytes saved — a bytes-only gate would have passed the arm that already lost.

**Algebra: the gather form exists and is in production.** `IndexMap::Computed { indices,
index_map, base, gathered_dim }` (`map.rs:134-151`, opened) is live for MoE expert
routing on the CPU (`run_reduce_quantized`'s gather arm, `cpu.rs:7091`, sizing
`7205-7261`) and expresses exactly `y[j] = sum_k W[idx[j], k] * x[k]` — today only for
whole-expert slabs (probe lines 15-18). `bind::build_operand` (`bind.rs:1030-1057`,
opened) already turns it into `Lookup { indices, index_layout, element_stride, extent }`
with `element_stride = row_major_strides(shape)[gathered_dim]` (`bind.rs:1046`), and
`shape.rs:297-310` bounds the gathered extent at `GATHER_EXTENT_EXACT_FLOAT_LIMIT`;
14336 and 32002 both pass.

**The selector is a program in the existing algebra — no `ScalarOp` added.** The probe is
right that there is no top-k, threshold-count or argsort, and that `Keep` is only
`Reduce | Scan` (`op.rs:60-78, 142-147`, opened). The construction below needs none of
them, and it is the non-obvious part of this design, so it is written out:

```
scores[r]     = Reduce(Add) over the rank-128 sketch          -- an ordinary matvec
selected[r]   = Greater(scores[r], threshold)                 -- ScalarOp::Greater
prefix[r]     = Reduce { body: Add, keep: Keep::Scan } over selected
                                                              -- exclusive prefix sum
slot[r]       = Select(selected[r], prefix[r], DUMP)           -- ScalarOp::Select
rows[c]       = Reduce { body: Maximum, init: Zero,
                         operand: Iota(14336),
                         out_map: IndexMap::Computed { indices: slot, gathered_dim: 0 },
                         keep: Keep::Reduce }
```

Four existing node forms. Three details make it correct rather than nearly correct:
- **A scatter writes on every iteration step**, so unselected rows must be sent
  somewhere harmless: `DUMP` is one extra destination slot, and the destination extent
  travels in `IndexMap::scatter_extent` (`map.rs:167-198, 209-222`, opened, including the
  `base`-at-`gathered_dim` offset convention). Colliding writes fold with the reduce's
  own body (`map.rs:109-115`), so every unselected row folds into `DUMP` and is dropped.
- **Indices must be an integer dtype** — `shape.rs:283-291` (opened) enforces it. This is
  the same constraint that forces §C.3's `cached_len: Int32`, one mechanism, not two.
- **Fixed budget, dynamic membership.** The gathered output axis extent must be static at
  bind time, so the budget `c` is fixed (e.g. `c = 2867` = 20% of 14336) and the
  threshold is tuned per layer so roughly `c` rows pass. Overflow rows land in `DUMP`;
  underflow leaves slots pointing at row 0 with a 0/1 mask operand zeroing their
  contribution. The consequence matters beyond correctness: **the elided program has a
  fixed shape, so it is one plan with no recompile**, which is what keeps §B.7's
  plan-hit invariant (and the zero-allocation budget) intact under elision.

**What must change is an emitter, and the probe names it exactly.** Metal's
`classify_packed_row_block` requires `gather_count == 0` (`msl.rs:1039-1052, 1389`), so
any gathered weight falls to the serial one-thread-per-output kernel with element-granular
fetches (`push_gather_fetch`, `msl.rs:2286-2318`) and a per-operand fault slot
(`metal.rs:3221-3233`). That fallback is the same shape as the CPU arm that measured
dispatch-bound. So the card is not "turn on the gather"; it is **"the packed-row-blocked
kernel accepts a row-axis `Lookup`"**: fetch `row = rows[j]` once per output row, then
read that row's quant blocks contiguously exactly as today. Bytes per row unchanged, row
count falls, addressing cost amortized over 4096 elements instead of paid per element —
which is the specific difference from the arm that lost.

**Quality gate (discovery-loop: the correct output is unknown).**
- Pre-registered hypothesis: at `e = 0.80` with a rank-128 scorer, greedy
  exact-match-at-64 vs dense ≥ 0.95 on 200 held-out prompts.
- Mechanism metric: **mass recall** — the fraction of dense `|gate_i * up_i|` L1 mass
  captured by the selected set, per layer per token, reported as a distribution.
- Performance metric: **ns/element vs the dense packed-row arm** at the same `e`, which
  is what ROW 180/181 says the bytes alone will not tell you.
- Ablation: `e ∈ {0.5, 0.7, 0.8, 0.9}`; rank ∈ {64, 128, 256}. Degenerate control: a
  **random** row set at the same `e`, which must fail. If random passes, the metric is
  measuring something other than sparsity and the result is void.
- Kill: exact-match-at-64 < 0.90 at e=0.80, or mass-recall p10 < 0.80, or ns/element
  above the dense arm.

## D.4 output.weight — 107 MB/token for one argmax

Binds today at Q6_K (probe line 32).

**(a) Q4_K instead of Q6_K:** 107.5 → 73.7 MB, saves 34 MB, existing codec path. Lossy;
metric family as D.2, kill at exact-match-at-64 < 0.98 (this tensor sits directly on the
sampled token, so it gets the tighter bar).

**(b) Candidate-set logits:** read only `c` rows of `output.weight` chosen by a cheap
sketch. At c=512, Q6_K: 512 x 4096 x 6.5625/8 = **1.72 MB**, saving ~105 MB. DERIVED.
Algebra: the *same* row gather as D.3 over the vocab axis — `check_gather_extent` passes
at 32002 (`shape.rs:297-310`) — and the same fixed-budget/DUMP construction, so it rides
D.3's emitter card rather than needing its own.
Quality: metric = top-1 recall of the candidate set vs the full argmax, per token, 200
prompts x 64 tokens. Kill: recall < 0.999. A miss is not a small error; it is a different
token and the sequence diverges.

## D.5 KV bytes — the term that grows with context

**Bytes.** 0.262144 MB per context-token per generated token at f32. DERIVED from
32 layers x 2 x kv_heads 8 x head_dim 128 x 4 B.

| KV dtype | bytes/ctx-token | t=512 | t=2048 | t=8192 |
|---|---|---|---|---|
| f32 (today) | 4096 B | 134 MB | 537 MB | 2.147 GB |
| f16 | 2048 B | 67 MB | 268 MB | 1.074 GB |
| Q8_0 | ~1088 B | 36 MB | 143 MB | 570 MB |

**Algebra: existing.** `PackedCodec::Float16` is already a codec slot with a stated
reason (`metal.rs:441-444`, opened: it needs a non-`float` binding type even with no
unpack function). The cache write is an ordinary op carrying its own `BoundOp::dtype`
(`bind.rs:206-212`, opened, which documents dtype-per-node precisely so a narrower node
emits a narrower kernel). Q8_0 KV additionally needs the write side to quantize — one
epilogue stage under §A's `Loop`, not a new path.

**Second KV lever, free from §A.1.** The cache is two buffers per layer today
(`kv_cache.{layer}.k_even` / `.k_odd`, `generate.rs:677-680`, opened) because RoPE's
`2i`/`2i+1` split could not survive a fusion (`is_identity_projection`,
`bind.rs:1156-1164`). With `IndexPattern::compose`, K is one interleaved buffer read with
stride-2 maps: one buffer per layer instead of two, half the placements
(`generate.rs:2378-2391` pushes 6 per layer today), and the `k_even`/`k_odd` vocabulary
leaves 15+ sites (`spec.rs:2333`). Bytes unchanged; bindings, placements and dispatches
down.

Quality: f16 KV, kill at exact-match-at-64 < 0.99. Q8_0 KV, kill at < 0.98.

## D.6 The residual: achieved bandwidth 161 vs 400 GB/s

MEASURED: 4.03 GB in 24.991 ms = 161 GB/s (dispatch-census.md:27, packed-row arm). I have
no mechanism for the 2.5x gap and will not supply one — a causal claim needs an artifact
in the same breath. The instrumentation that would produce one: a pure-read kernel arm
with the *identical* access pattern (same quant block stride, same threadgroup width, no
arithmetic), three problem sizes, against a `memcpy`-shaped control that should saturate.
Until that runs, every projection below is given at both 161 (measured) and 300 GB/s
(ASSUMED).

## D.7 The arithmetic — does the product reach ≤1.4 GB/token?

Two routes, both at t=512. Bytes DERIVED from the tables above; times DERIVED by division.
Both assume D.1's `nr1` s-fold has landed, without which the `÷ a` row does not apply.

**Route 1 — no discovery-loop lever (D.1 + D.2-Q3_K + D.4a + D.5-f16):**

| step | weights | KV | pass bytes |
|---|---|---|---|
| baseline | 4.034 | 0.134 | 4.168 GB |
| + Q3_K FFN | 3.285 | 0.134 | 3.419 |
| + output.weight Q4_K | 3.251 | 0.134 | 3.385 |
| + f16 KV | 3.251 | 0.067 | 3.318 |
| ÷ acceptance a = 2.5 | | | **1.327 GB/token** |

**Route 2 — with elision (D.1 + D.3@80% + D.4a + D.5-f16):**

| step | weights | KV | pass bytes |
|---|---|---|---|
| baseline | 4.034 | 0.134 | 4.168 GB |
| + FFN elision 80% + rank-128 predictor | 1.541 | 0.134 | 1.675 |
| + output.weight Q4_K | 1.507 | 0.134 | 1.641 |
| + f16 KV | 1.507 | 0.067 | 1.574 |
| ÷ a = 2.5 | | | **0.630 GB/token** |

**With the dispatch term.** Non-matvec dispatch cost MEASURED 6.344 ms (2.978 elementwise
+ 0.851 cooperative + 3.515 attention, dispatch-census.md:26-27, biased high by
serialized command buffers). §A/§B take 616 → 264 (census line 49); DERIVED residual for
the non-matvec ops ≈ 0.65 ms.

| route | bytes/token | @161 GB/s | @300 GB/s | + ~0.7 ms dispatch |
|---|---|---|---|---|
| today | 4.168 | 25.9 | 13.9 | **33.0 MEASURED** |
| Route 1 | 1.327 | 8.24 | 4.42 | 8.9 / 5.1 |
| Route 2 | 0.630 | 3.91 | 2.10 | 4.6 / 2.8 |

**Reading, stated plainly.** ≤1.4 GB/token is reached by Route 1, which contains no
discovery-loop lever. 3.5 ms/token is reached only by Route 2 *and* only if achieved
bandwidth moves off 161 GB/s. D.6 is therefore on the critical path to the owner's
number, and it is the one term I cannot currently explain. Two preconditions gate the
whole table and are not assumptions I get to make quietly: the `nr1` s-fold
(`msl.rs:3171-3200`) and the gather-capable packed-row kernel
(`msl.rs:1039-1052, 1389`). Without the first, D.1 contributes nothing; without the
second, D.3 contributes negative time at positive byte savings, which is exactly what ROW
180/181 measured.

**Order (risk-ascending):**
1. D.5 f16 KV — 67 MB at t=512, 1.07 GB at t=8192; near-lossless; one dtype change.
2. D.4a output.weight Q4_K — 34 MB; existing codec.
3. D.1a the `nr1` s-fold in the packed-row kernel — 0 bytes by itself, and the
   precondition for everything multiplicative. Gate: `s=4` pass streams weights once
   (weight bytes MEASURED flat vs `s`).
4. D.1b n-gram drafter + verify in `Cursor` — `÷ a`, byte-identical gate, zero added
   weight bytes.
5. D.6 bandwidth instrumentation — 0 bytes, potentially 1.9x on the largest term.
6. D.2 Q3_K FFN — 749 MB; one rejection lifted, one dequant body per backend, plus a
   requantizer.
7. D.3 elision — 2.5 GB; needs the gather-capable packed-row kernel first, then runs as a
   discovery-loop with a kill criterion and a degenerate control.

**What is *not* a byte lever, said so it is not oversold:** all of §A/§B. Epilogue fusion
removes `residual1`/`x_next` materialization = 2 x 4096 x 4 B written and reread per
layer = 2.1 MB/token DERIVED, which is 0.05% of the stream. §A/§B buy the 6.3 ms of
dispatch time; §D buys the 25 ms of streaming. Neither substitutes for the other, and the
33.0 ms → 3.5 ms target needs both.

# A. Attention in the RISC algebra

## A.0 What attention IS, in Op/ScalarOp/IndexMap

For output `o[q,h,g,d]` over query rows `q`, kv-heads `h`, query groups `g`, head_dim
`d`, keys `t`, and contraction axis `p`:

```
s[q,h,g,t] = scale * SUM_p  Q[q,h,g,p] * K[t,h,p]        Reduce(Add) over p
b[q,h,g,t] = Select(band(q,t), -inf, s)                  Elementwise(Select)
m[q,h,g]   = MAX_t b                                     Reduce(Maximum, NegativeInfinity) over t
w[q,h,g,t] = exp(b - m)                                  Elementwise(Subtract, Exponential)
l[q,h,g]   = SUM_t w                                     Reduce(Add, Zero) over t
a[q,h,g,d] = SUM_t w * V[t,h,d]                          Reduce(Add, Zero) over t
o[q,h,g,d] = a / l                                       Elementwise(Divide)
```

Every line is an existing `Op` with existing `ScalarOp`s (`op.rs:60-78`) and existing
index patterns; `band` is `Greater` over two `Iota`s, which `Op::Iota`'s own doc
(`op.rs:209-231`) states is exactly what that variant was added for, and
`specs/causal_attention.toml` already evaluates it. **Attention needs no new algebra to
be *expressed*.** What it needs is a way to say those seven lines share one traversal.

## A.1 Extension 1 (required): `IndexPattern::compose`

```rust
// proxima-tensor/src/map.rs — pure, no_std, no alloc beyond the axes Vec it returns.
/// Substitutes `outer` into `inner`: given `inner` addressing an operand from an
/// iteration space I2, and `outer` addressing I2 from I1, returns the pattern
/// addressing that operand directly from I1. Affine composition — coefficients
/// multiply, offsets accumulate. `None` when `inner` names an axis `outer` does not
/// project (a rank mismatch the caller must materialize through instead).
#[must_use]
pub fn compose(outer: &IndexPattern, inner: &IndexPattern) -> Option<IndexPattern>;
```

Semantics: for each `inner` axis `sum_j c_j * i_j + k`, replace each `i_j` by
`outer.axes[j]` = `sum_m e_m * u_m + f`, yielding `sum_{j,m} (c_j * e_m) * u_m +
(k + sum_j c_j * f_j)`. Terms with equal `axis` merge.

**Why required, and why nothing smaller works.** Fusion today is gated on
`is_identity_projection` (`bind.rs:1156-1164`: every axis `offset == 0` and a single
`coeff == 1` term), read at the two fusion sites (`bind.rs:699-701` elementwise,
`bind.rs:776-778` reduce). That predicate is literally "the map is the identity", and it
is the *whole* reason RoPE's `2i`/`2i+1` and GQA's `h = group*u + g` decline to fuse
(dispatch-census.md:11 names both). Anything that admits a non-identity map must know
how to push it through the held node's own map — that operation is substitution, and
there is no weaker operation than substitution that produces a correct address. The
`eliminate_masked_window_reduce` special case (`bind.rs:734-773`), which exists solely
to force materialization before a two-term window map, is subsumed.

Two binary questions. (1) Can an existing primitive express it? No — `IndexPattern` has
no composition operator; `compose_operand` (`bind.rs:1177+`) recurses only through maps
it has already proven to be the identity, i.e. it *avoids* composition rather than
performing it. (2) What can a caller do that they could not? Fuse a strided or offset
operand read. The RoPE ops (4/layer, 128/token) and the GQA group map exist as separate
dispatches only because of this. New capability, measurable in dispatch count.

## A.2 Extension 2 (required, consequence of A.1): strided extent resolution

`unify_iteration_space` (`shape.rs:224-244`) binds an iteration axis's extent *only*
from an operand axis that is a single `coeff == 1`, `offset == 0` term. After A.1 an axis
can appear only under a strided map, and inference then reports `UnconstrainedDim`
(`shape.rs:250-253`). This is audit item 13's first blocker, verified at those lines.

```rust
// shape.rs — inside unify_iteration_space, a second pass over axes no unit
// projection pinned, run only for still-unresolved slots:
//   single term `c*i + k` against operand extent E, c > 0  =>  N = (E - 1 - k)/c + 1
// Precedence: a unit projection always wins. Existing programs bind byte-identically.
```

The precedence rule is the non-regression proof obligation, and it is mechanical: run
`bind` over `specs/mistral_layer.toml` at real dims before and after and assert the full
`Vec<BoundOp>` compares equal (the existing fixture at `spec.rs:10313-10327` already
builds that program).

## A.3 Extension 3 (required): `BoundOpKind::Loop` and `StepArg::Stage`

```rust
// proxima-tensor/src/bind.rs
pub use crate::sized::MAX_INLINE_STAGES;   // build.rs const, see §C.4

/// One accumulation stage of a `Loop`. A stage with `fold: None` is a pure map over
/// the axes still live at that point; a stage with `fold: Some(..)` folds
/// `reduced_axes` away and its result is addressable by later stages.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// Iteration axes this stage folds. Empty for a map stage. Order is the loop
    /// nesting order an executor walks, outermost first.
    pub reduced_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
    /// The per-step scalar body. `StepArg::Operand` reads this Loop's operands;
    /// `StepArg::Step` reads an earlier step of THIS body; `StepArg::Stage` reads an
    /// earlier stage's accumulator (backwards-only, same rule as `Op`'s NodeIds).
    pub body: ComposedBody,
    pub fold: Option<Fold>,
    /// Restriction of the traversal, when an analysis proved one. Purely a
    /// profitability hint: the body still contains the masking `Select`, so an
    /// executor that ignores this produces identical results, slower.
    pub band: Option<Band>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold { pub op: ScalarOp, pub init: ReduceInit, pub keep: Keep }

/// A half-open restriction on ONE reduced axis, in that axis's own coordinates:
/// `lower(coord) <= axis < upper(coord)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Band { pub axis: u16, pub lower: BandBound, pub upper: BandBound }

/// Affine in the iteration coordinate, plus optionally a runtime scalar read from one
/// of the Loop's own operands — which is what `cached_len` is, and why no sentinel and
/// no operand-count discriminator is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandBound {
    /// `sum(terms) + offset` over iteration axes.
    Affine(AxisIndex),
    /// `sum(terms) + offset + operands[slot][0]`; `slot`'s dtype must satisfy
    /// `DType::is_integer` (`dtype.rs:57`), checked when the band is built.
    Dynamic { slot: u16, plus: AxisIndex },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepArg { Operand(u16), Step(u16), Stage(u16) }   // Stage is the added variant

pub enum BoundOpKind {
    Loop {
        stages: SmallVec<[Stage; MAX_INLINE_STAGES]>,
        operands: BoundOperands,
        /// Iteration axes surviving to the output, in `out_map` operand-axis order —
        /// unchanged semantics from today's `Reduce::output_axes`.
        output_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        out_layout: Layout,
        out_scatter: Option<Lookup>,
    },
    Iota,
    Constant { value: f32 },
}
```

`Elementwise` = one stage, `fold: None`, `reduced_axes: []`. `Reduce` = one stage with a
`Fold`. `CachedAttention` = **deleted**. Net: 5 kinds → 3.

**Attention as a `Loop`,** iteration space `(q,h,g,t,p,d)`, operands `[Q, K, V,
cached_len]`:

| stage | reduced_axes | body | fold |
|---|---|---|---|
| 0 | `[p]` | `Multiply(Operand(0), Operand(1))` then `Multiply(Step(0), scale)` | `Add`/`Zero` |
| 1 | `[t]` | `Select(band_expr, -inf, Stage(0))` | `Maximum`/`NegativeInfinity` |
| 2 | `[t]` | `Exponential(Subtract(Stage(0'), Stage(1)))` | `Add`/`Zero` |
| 3 | `[t]` | `Multiply(Step(exp), Operand(2))` | `Add`/`Zero` |
| 4 | `[]` | `Divide(Stage(3), Stage(2))` | `None` |

`Band { axis: t, lower: Affine(0), upper: Dynamic { slot: 3, plus: q } }` on stages 1-3.
Strides for Q/K/V come from `build_operand` (`bind.rs:1030-1057`), i.e. from `Layout`,
which is exactly what `render_cached_attention` ignores today (`msl.rs:2578` hardcodes
`qbase`/`kbase` arithmetic; the matcher compensates with eight literal stride tuples at
`bind.rs:2451-2472`). Those tuples and the rank gates (`bind.rs:2419-2433`, including
`head_dim = query_shape[3] * 2`) delete.

**Two binary questions for `Loop`.** (1) Pipe? No — it is a value in a program, not
behaviour; data flowing through pipes is not itself a pipe. (2) What can a caller do?
Express five folds sharing one traversal, which is what removes 96 + 64 + 128 + 64
dispatches per token (§A.5) and what lets attention stop being a macro-op. Before: the
caller either accepts 7 dispatches or asks for a hand-written kind. After: one kind, one
rule set. That difference is written in dispatch counts, not in prose.

## A.4 The fusion RULES (not matchers), and the lowering theorem

All rules operate on the *bound* stream inside `BoundOpBuilder` (`bind.rs:992-1003`,
already a `Pipe<In = (Op, Shapes), Out = ReadyBatch>` with a `held` map). **No new pipe
stage** — that was an abandoned design (§E.2).

```rust
/// Which structural rewrites this consumer's backend can render. Replaces
/// `bind_with_fusion(.., fuse_cached_attention: bool)` (`bind.rs:2635-2640`), which
/// collapsed a capability SET to one bit and made wgpu/cuda decline prologue fusion
/// they can render, merely because they cannot render attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_FUSION")]
pub struct FusionRules {
    /// Absorb a producer elementwise into a consumer's per-step body.
    #[setting(default = true)]  pub prologue: bool,
    /// Absorb a consumer elementwise into a producing fold as a later stage.
    #[setting(default = true)]  pub epilogue: bool,
    /// Allow an epilogue whose iteration space is the fold's INPUT space (the fold
    /// result broadcasts back) — rmsnorm's shape.
    #[setting(default = true)]  pub broadcast_epilogue: bool,
    /// Derive a `Band` from a masking `Select` and restrict the traversal.
    #[setting(default = true)]  pub band: bool,
    /// Execute a `Maximum` fold and the `Add` folds depending on it in ONE traversal.
    #[setting(default = true)]  pub online_softmax: bool,
}
```

Defaults are seeded from `sized::` consts so the no_std/no_alloc tier has the same knobs
(conflaguration bridge, §C.4). Every field is a *rendering capability*, so a backend
publishes `const RULES: FusionRules` and gets each independently — wgpu sets
`online_softmax: false` and still receives prologue/epilogue/band.

**R1 PROLOGUE** (exists; generalized by A.1). Legality: the producer is elementwise, its
map composes (A.1), it is not data-dependent. Profitability: §A.6 cost model, replacing
`quarantine_broadcast_operands` (`bind.rs:935-973`) and the `StillLive` veto
(`bind.rs:698-701, 775-778`).

**R2 EPILOGUE** (new). If a `Loop` `L` whose last stage folds is consumed by exactly one
elementwise `E`, and `E`'s iteration space equals `L`'s `output_axes` space, then `E`
becomes a `fold: None` stage appended to `L`, reading `StepArg::Stage(last)`, and `L`'s
`out_layout` becomes `E`'s. Removes `residual1`, `x_next`, `ffn_hidden`
(dispatch-census.md:33-37) = 3/layer = 96/token.

**R3 BROADCAST-EPILOGUE** (new). Same, except `E`'s iteration space is `L`'s *full*
space: the fold result is broadcast back over the reduced axes. Appended as a
`fold: None` stage with `reduced_axes: []` reading `Stage(k)` — an executor runs the
fold to completion, then a second traversal of the same axes with the accumulator in a
register/threadgroup. rmsnorm (`sum_squares` then `normed`) = 2/layer = 64/token
(dispatch-census.md:46-48).

**R4 BAND** (new, analysis only). Scan a stage's body for
`Select(Greater(A, B), NEG_INF, x)` where `A`, `B` are affine in iteration axes plus at
most one rank-0 integer operand. Emit `Band`. **The body keeps the `Select`.** That
single decision is what makes this rule unable to change a result: an emitter that
ignores `band` is correct-and-slower, and a band that is derived too wide is
correct-and-slower. This replaces `exact_merged_causal_mask_cached_len`
(`bind.rs:2353`), the `i64::MIN`/`i64::MAX` sentinels (`bind.rs:2510-2511`,
`msl.rs:2530-2534`), and the `operands.len() == 8 | 9` discriminator read in four files
(`bind.rs:237-239`, `cpu.rs:4847`, `msl.rs:2030`, `msl.rs:2542`).

**R5 ONLINE-SOFTMAX — a lowering theorem, per emitter, not an IR fact.**

> Let stage `i` fold `Maximum` with `init = NegativeInfinity` over axis set `T`, and let
> stages `j > i` fold `Add` over the same `T` with bodies of the form
> `g(...) * exp(x - Stage(i))` where `x` does not depend on `Stage(i)`. Then stages
> `i..=j` may be executed in ONE traversal of `T`, maintaining `m` and each `Add`
> accumulator, rescaling every accumulator by `exp(m_old - m_new)` whenever `m` rises.

Any program with that shape gets it. No model name, no operand count, no stride literal
appears in the rule. An emitter that declines runs two traversals of `T` inside the same
kernel — **same dispatch count, same buffers, different arithmetic order**. That is what
makes the theorem a schedule rather than semantics, and it gives a two-sided parity gate
that today does not exist:
- fused-CPU vs unfused-CPU isolates *rule* error,
- Metal vs fused-CPU isolates *emitter* error.

Today a single Metal-vs-CPU comparison mixes both.

The loop-count defect disappears by construction: today's single-range fusion sets
`cached_key_rows = new_key_rows = key_shape[0]` and `cached_lower = i64::MAX`
(`bind.rs:2504-2506, 2510`) so the kernel at `msl.rs:2586` iterates
`cached_key_rows + new_key_rows` = 2t and `continue`s past the first t. Under `Loop`,
the traversal bound is `extents[t]` — t — and the band restricts it further.

## A.5 The census, rule by rule

| census lever | expressed today? | rule | needs | dispatches removed |
|---|---|---|---|---|
| epilogue (residual1, x_next, ffn_hidden) | no | R2 | `Loop` + `StepArg::Stage` | 96/token |
| prologue-with-broadcast (`normed`) | yes, vetoed by heuristic | R1 | cost model (§A.6) | 64/token |
| RoPE (rotated q/k even/odd) | no | R1 + R2 | A.1 + A.2 | 128/token |
| rmsnorm reduce+broadcast | no | R3 | `Loop` + `Stage` | 64/token |
| attention | as 7 ops | R1+R2+R4+R5 | all of the above | 32 macro-ops become 32 Loops |

19 → 8 bound ops/layer, 616 → 264/token (census line 49). Each card's gate is the
measured count, not the projection.

## A.6 Profitability: a cost model, not a heuristic

`quarantine_broadcast_operands` declines when `child_extent < reduce_extent`
(`bind.rs:954-955`) — a proxy that also declines profitable fusions, and the
`StillLive` veto (`bind.rs:698`) declines every multi-consumer node outright, which is
exactly why `normed` (5 consumers) materializes.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_FUSION_COST")]
pub struct FusionCost {
    /// Bytes/second the target moves, for weighing a materialization against recompute.
    #[setting(default = 161_000_000_000)] pub bytes_per_second: u64,
    /// Scalar ops/second one lane sustains.
    #[setting(default = 400_000_000_000)] pub flops_per_second: u64,
    /// Fixed cost of one dispatch, nanoseconds. MEASURED default from the census.
    #[setting(default = 10_300)] pub dispatch_ns: u64,
}
```

Decision, per candidate: absorb iff
`recompute_flops * consumers / flops_per_second <= (2 * bytes(node) / bytes_per_second) + dispatch_ns`.
Every input is already in hand at bind time (extents, dtype, body step count, consumer
count from `live::annotate`). The defaults are the measured constants and they are
build-time config, not literals in source (§12).

## A.7 Migrating off `BoundOpKind::CachedAttention`

The variant has ten companions, which is the finding: `cached_attention_candidates` +
`cached_attention_single_range_candidates` (two matchers, `bind.rs:2653-2655`),
`attention_dependencies` (`bind.rs:2520`), `attention_consumers` (`bind.rs:2559`),
`has_external_attention_consumer` (`bind.rs:2545`), `removable_attention_dependencies`
(`bind.rs:2585`), `exact_merged_causal_mask_cached_len` (`bind.rs:2353`), eight literal
stride tuples (`bind.rs:2451-2472`), `pack_cached_attention_uniforms`
(`metal.rs:2197`), `render_cached_attention` (`msl.rs:2513`), plus two total-accessor
special cases (`BoundOp::element_body` returning `EMPTY_BODY`, `bind.rs:326`;
`split_axis` returning `None`, `bind.rs:409`). Follow the compensators inward: the defect
is the one, not the eleven.

Migration, in the card order of §E:
1. `Loop` lands; `Elementwise`/`Reduce` become one-stage `Loop`s; `CachedAttention` still
   exists and still matches. Gate: 100x byte-identical logits, dispatch count still 616.
2. R2/R3 land. Gate: 616 → 520, parity ≤1e-4 vs unfused CPU.
3. R4/R5 land; the two matchers, both `attention_*` helpers, the stride tuples, the
   uniform packer, the renderer, and the variant delete in one commit. Gate: attention
   op count still 32; `gpu_exec_ms` for the attention arm falls (2t → t traversal);
   text identical over 100 tokens.
4. `attention_dependencies`/`removable_attention_dependencies` are *replaced* by the
   generic single-consumer absorption the builder already performs for elementwise
   chains — not ported. If any behaviour of theirs cannot be expressed generically, that
   is a finding about the generic rule and it gets fixed there.

---

# B. The decode step as FSM x orchestration over pipes

## B.1 The planning chain — already pipes, and no new stage

`ShapeTable: Pipe<In = Op, Out = (Op, Shapes)>` (`shape.rs:323-335`) and
`BoundOpBuilder: Pipe<In = (Op, Shapes), Out = ReadyBatch>` (`bind.rs:992-1003`) compose
with `AndThen` (`primitives.rs:203-221`). Fusion rules R1-R5 are *methods on the existing
builder*, because R2/R3 need to see a consumer that arrives later — which is precisely
the `held` mechanism the builder already runs for producers (`bind.rs:724-731`). One
`held` map, symmetric use. Adding a `Fuse` stage was rejected: §E.2.

`bind_with_fusion`'s double pass (`bind.rs:2641` then `bind.rs:2678` re-runs `bind_plain`
over an extended output set) disappears — there is one pass.

## B.2 `plan()` becomes pure, and `Schedule` is plan-time data

`omega::metal::plan` calls `device_and_queue()` at `metal.rs:490` to build the arena and
uniform buffers (`metal.rs:488-499`). Split:

```rust
// proxima-tensor — no_std + alloc, no device, no `omega` dependency.
#[must_use]
pub fn plan(
    program: &[Op], symbols: &[u64], codecs: &[Option<PackedCodec>],
    outputs: &[NodeId], config: &PlanConfig,
) -> Result<Plan, TensorError>;

pub struct Plan {
    program: Vec<Op>, shapes: Shapes, resolved: Vec<BoundOp>,
    retires: Vec<Vec<NodeId>>, effective_outputs: Vec<NodeId>,
    block_nodes: Vec<NodeId>, block_dtypes: Vec<DType>,
    schedule: Schedule,
}

/// Everything today's driver recomputes per token or builds with device IO, computed
/// once as pure data. Zero `Retained<_>`, zero `MTLBuffer`, compiles under
/// `--no-default-features --features alloc`.
pub struct Schedule {
    /// Arena slot per resolved position; `slots.len() < position_slot.len()` records
    /// reuse. Pure sizing — `build_buffer_arena` (`metal.rs:3555-3612`) minus the
    /// `allocate_buffer` call.
    pub position_slot: Vec<SlotId>,
    pub slot_bytes: Vec<usize>,
    pub peak_bytes: usize,
    /// Barrier-before-position, and WHICH slots it must cover. Static: hazards are a
    /// function of (position -> slot) reads/writes, which the plan fixes.
    pub barriers: Vec<BarrierSet>,
    /// Per-position uniform bytes, with the fields that vary per token named rather
    /// than rebuilt: see `UniformPatch`.
    pub uniforms: Vec<UniformBlob>,
    pub uniform_patches: Vec<UniformPatch>,
    /// Emitter routing decided HERE, not recovered by substring-grepping MSL
    /// (`metal.rs:1631-1692`).
    pub kernels: Vec<KernelSpec>,
}

#[derive(Clone, Copy, PartialEq, Eq)] pub struct SlotId(pub u32);
#[derive(Clone, Copy, PartialEq, Eq)] pub struct KernelId(pub u32);
pub struct BarrierSet { pub slots: SmallVec<[SlotId; MAX_INLINE_BARRIER_SLOTS]> }
```

**HazardTracker becomes a pure function.** Today it walks pointer identity per op per
token (`metal.rs:1094-1125`), and `reset` clears *both* sets on any barrier
(`metal.rs:874-877`) so one hazard forces a full drain. At plan time the same three
hazards (RAW/WAW/WAR, `metal.rs:861-867`) are computed over `SlotId`, and the barrier
carries the *specific* slots — `memoryBarrierWithResources` instead of
`MTLBarrierScope::Buffers` (audit 11 notes the API exists). Information restored: which
resources, and which hazard.

```rust
/// Pure, no_std, testable with `SlotId = u32` and no device — which is the capability
/// today's `HazardTracker<Id>` gestures at (it is generic "so this logic is testable
/// without a real Metal device", `metal.rs:824-827`) but cannot deliver, because the
/// walk that uses it lives inside `execute_plan_with_placements`.
#[must_use]
pub fn barrier_schedule(resolved: &[BoundOp], position_slot: &[SlotId]) -> Vec<BarrierSet>;
```

**KernelSpec, not a string.** `classify_kind` re-renders MSL and greps it for
`simdgroup_multiply_accumulate` / `q4k_pair_dot(blk` / `acc1_0` to recover a decision the
emitter itself made (`metal.rs:1631-1692`); the comment there records that this label was
wrong for 216 of 225 ops until an arm was added. The emitter returns
`KernelSpec { id: KernelId, family: KernelFamily, entry: ArrayString<..> }` where
`KernelFamily` is an enum, and the profiler groups by that.

## B.3 The step FSM — typestate for the path, one enum for suspension

Exactly one path exists (bind → encode → submit → complete → sample), so the transitions
are typestate (P11). One enum exists, and only because a step must be *stored* across a
`Poll::Pending` while the GPU runs:

```rust
/// Owned, `!Send`, borrows the plan for `'p`. Each transition consumes the old value.
pub struct ReadyStep<'p>   { plan: &'p Plan, cursor: Cursor, scratch: &'p mut StepScratch }
pub struct BoundStep<'p>   { plan: &'p Plan, cursor: Cursor, bindings: Bindings<'p> }
pub struct EncodedStep<'p> { plan: &'p Plan, cursor: Cursor, commands: Commands<'p> }
pub struct InFlightStep<'p>{ plan: &'p Plan, cursor: Cursor, ticket: SubmitTicket }
pub struct SampledStep     { cursor: Cursor, logits: LogitsView, counters: StepCounters }

/// The suspension form: the ONLY runtime state discriminator in this design, and it
/// exists because `poll_advance` must resume a step it did not start.
pub enum Step<'p> {
    Ready(ReadyStep<'p>), Bound(BoundStep<'p>), Encoded(EncodedStep<'p>),
    InFlight(InFlightStep<'p>), Sampled(SampledStep),
}

impl<'p> Step<'p> {
    /// Advances as far as it can without blocking; `Pending` only inside `InFlight`.
    /// Exhaustive match, transitions consume via `core::mem::replace`.
    pub fn poll_advance(&mut self, cx: &mut Context<'_>) -> Poll<Result<TokenId, StepError>>;
}
```

`Cursor { position: u32, accepted: u16 }` is the newtype carrying "how many tokens are
committed" — the value D.1's acceptance test writes, so multi-token verification is a
field, not a code path.

**Counters by ownership, not by protocol.** `metal_stage_totals()` is
snapshot-and-reset and its doc states it "must happen exactly once per step"
(`metal.rs:2719`, and `generate.rs:2468-2482` is the call site honouring it in a
comment). `SampledStep` **owns** `StepCounters` by value, produced by the transition.
Reading a step's counters twice is not a discipline; it does not compile.

## B.4 The pipe composition of the driver, by In/Out

Every stage is `Pipe`; the chain is `AndThen`, which is itself a `Pipe`
(`primitives.rs:203-221`), and `.and_then` comes from `PipeExt` (`ext.rs:45-54`).

| stage | In | Out | form | pure? |
|---|---|---|---|---|
| `BindInputs` | `ReadyStep<'p>` | `BoundStep<'p>` | transform | yes, 0 alloc |
| `Encode` | `BoundStep<'p>` | `EncodedStep<'p>` | transform | yes, 0 alloc |
| `Submit` | `EncodedStep<'p>` | `InFlightStep<'p>` | transform | **the Metal edge** |
| `Complete` | `InFlightStep<'p>` | `SampledStep` | transform | **the Metal edge** |
| `Sample` | `SampledStep` | `TokenId` | transform | yes, 0 alloc |
| `StopPolicy` | `TokenId` | `TokenId` | observe | yes — `Err(StepEnd::Eos)` at stop |
| `Advance` | `TokenId` | `ReadyStep<'p>` | transform | yes, 0 alloc |

```rust
let decode_step = BindInputs
    .and_then(Encode)
    .and_then(driver.submit())     // Pipe<In = EncodedStep, Out = InFlightStep>
    .and_then(driver.complete())   // Pipe<In = InFlightStep, Out = SampledStep>
    .and_then(Sample::from(sample_config))
    .and_then(StopPolicy::new(&vocab, stop));
```

`StopPolicy` is an observe pipe (`Out = In`) that returns `Err(StepEnd::Eos)`. That is
the `Shed` lesson applied, not re-litigated: `and_then` + render the `Err` at the edge,
no `Decide` type.

**The Metal edge is two methods on one type.** Nine entry points today
(`metal.rs:519` `execute`, `535` `execute_plan`, `978` `execute_plan_with_placements`,
`1177` `plan_named`, `1192` `execute_plan_named`, `1210`
`execute_plan_named_with_placements`, `1409` `execute_plan_op_timed`, `1484`
`execute_plan_named_op_timed`, `1509` `execute_plan_with_placements_op_timed`, `1613`
`execute_plan_named_with_placements_op_timed`) are one driver x {named, placed, timed}.
Each axis becomes data, not an entry point:

- **named** — `resolve_named_blocks` already exists and is already shared with the CPU
  evaluator (`metal.rs:1183`). It runs once at bind time into the owned `BindingTable`.
- **placed** — a binding is `Slot::Arena(SlotId)` or `Slot::External(BufferId)`. The
  three-hazard argument in `execute_plan_with_placements`'s doc (`metal.rs:921-960`)
  becomes two `barrier_schedule` inputs (external slots are never retired), not prose.
- **timed** — `SubmitConfig { per_op_command_buffers: bool }`. One command buffer per
  dispatch is a config field, which also deletes the
  `PROXIMA_METAL_OP_PROFILE_STEP` env branch (`generate.rs:2428-2455`) — a structured
  config field, not a hand-rolled env-gated dump.

```rust
pub struct MetalDriver {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue:  Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: KernelTable,     // was PIPELINE_CACHE      (metal.rs:266)
    residency: ResidencyTable,  // was NOCOPY/RESIDENT/CHECKPOINT (metal.rs:3007, 3150, 2933)
    slots:     SlotTable,       // was OUTPUT_BUFFER_POOL + BufferArena (metal.rs:2477, 3509)
    uniforms:  UniformTable,    // was UNIFORM_BUFFERS + CLOCK (metal.rs:3246, 3266)
    config: SubmitConfig,
}

impl MetalDriver {
    /// Binds a pure `Plan` to device resources ONCE: allocates arena slots from
    /// `schedule.slot_bytes`, uniform buffers from `schedule.uniforms`, compiles
    /// `schedule.kernels`. Replaces the device IO inside `plan()` (metal.rs:488-499)
    /// AND `mark_resident`'s two-phase init (metal.rs:373).
    pub fn residency(&self, plan: &Plan, resident: &ResidentSet) -> Result<Residency, MetalError>;

    pub fn submit(&self, step: EncodedStep<'_>) -> impl Future<Output = Result<InFlightStep<'_>, MetalError>>;
    /// Waker-driven: `addCompletedHandler` wakes the task. Replaces
    /// `command_buffer.waitUntilCompleted()` (metal.rs:1146), which parks the thread.
    pub fn poll_complete(&self, step: &mut InFlightStep<'_>, cx: &mut Context<'_>)
        -> Poll<Result<SampledStep, MetalError>>;
}
```

Both are `Pipe` impls on small handle values borrowed from the driver; `poll_complete` is
the P20 reactor-driven surface, and it is what lets the D.1 drafter run on the CPU while
the GPU pass is in flight.

**The eight thread-locals** (`metal.rs:252-274` PIPELINE_CACHE + DEVICE_AND_QUEUE,
`2477` OUTPUT_BUFFER_POOL, `2933` checkpoint mapping, `3007` nocopy, `3150` resident
copies, `3246` UNIFORM_BUFFERS, `3266` UNIFORM_CACHE_CLOCK) become the fields above.
`register_checkpoint_mapping` (`metal.rs:2957`), a side channel that mutates a global,
becomes `ResidentSet` passed to `residency()`. Lock discipline (§21): the driver is
`!Send`, per-core, shared-nothing — no mutex is introduced, which is the point of not
replacing thread-locals with `Arc<Mutex<..>>` (abandoned, §E.4).

## B.5 The encode loop as a pure function over a caller-owned buffer

```rust
/// Backend-neutral command stream. `&mut [Command]` is the encode target: zero
/// allocation, and the buffer is sized at plan time from `schedule.position_slot.len()
/// + schedule.barriers count`.
pub enum Command<'p> {
    Barrier { slots: &'p [SlotId] },
    Dispatch { position: u32, kernel: KernelId, bindings: &'p [Binding],
               grid: Grid, uniforms: UniformRef },
}

/// Pure. no_std. This is `execute_plan_with_placements`'s loop (`metal.rs:1074-1140`)
/// with every device call removed and every per-token `Vec` removed.
#[must_use]
pub fn encode<'p>(plan: &'p Plan, bindings: &'p Bindings<'p>, out: &'p mut [Command<'p>])
    -> Result<&'p [Command<'p>], TensorError>;
```

The hazard walk, the `hazard_inputs: Vec<_>` built per op (`metal.rs:1095-1100`), and the
`placement_dump` env branch (`metal.rs:1065-1089`) all leave this loop: the first two are
plan-time, the third is a telemetry event with a file-sink exporter, never a hand-rolled
`eprintln!`.

## B.6 Allocation budget, and the counter test

Target: **zero heap allocations per token on a plan hit.** Named sources removed:

| site | today | after |
|---|---|---|
| `generate.rs:2301-2307` | `named_blocks: Vec` rebuilt every step | owned `BindingTable`, built once |
| `generate.rs:2323-2324` | `cached_len_scalar` f32 array pushed | Int32 leaf in the owned table |
| `generate.rs:2347-2366` | KV scratch `resize` + 3 pushes/layer | placements are bindings, no scratch |
| `generate.rs:2370-2373` | two placement `Vec`s per step | owned, offsets patched in place |
| `generate.rs:2406-2413` | `roots: Vec` per step | `plan.effective_outputs` |
| `generate.rs:2503` | `next_ids = vec![token_id]` | `Cursor` + owned ids buffer |
| `metal.rs:541 / 996` | `device_buffers: BTreeMap` per call | `SlotTable`, indexed |
| `metal.rs:1073` | `pending_faults: Vec` | `ArrayVec` sized by plan gather count |
| `metal.rs:1095-1100` | `hazard_inputs: Vec` per op (616/token) | plan-time `BarrierSet` |
| `metal.rs:2183` | `pack_uniforms -> Vec<u8>` per op | `write_uniforms(&mut [u8])` |
| `metal.rs:3326` | `Vec<u8>` uniform cache KEY per op | index-keyed `UniformTable` |

The last two are 616 allocations/token each, DERIVED from the dispatch count.

```rust
/// Reads as documentation of the contract (P17): a decode step that hits its plan
/// performs no heap allocation, and the test asserts the STEP COUNT so that a loop
/// which ran zero steps cannot pass. Zero work and successful work do not share a
/// signal here.
#[proxima::test]
async fn a_plan_hit_decode_step_allocates_nothing_and_runs_every_step() {
    let counting = CountingAllocator::install();
    let session = fixture_session();                  // real openchat geometry, real gguf bytes
    session.decode_steps(WARMUP).await.expect("warmup");   // first step builds the plan
    let before = counting.total();
    let produced = session.decode_steps(STEPS).await.expect("steady state");
    assert_eq!(produced.len(), STEPS, "N==0 is RED: the loop must have run every step");
    assert_eq!(session.plan_misses_since(before_mark), 0, "steady state is a plan hit");
    assert_eq!(counting.total() - before, 0, "steady-state decode allocates nothing");
}
```

## B.7 The plan cache, and why it stops missing

Today the cache is one entry cleared on miss at three sites (`generate.rs:1325, 1373,
1417`), keyed partly on `kv_bound_extent` (`generate.rs:2344, 2394`), which changes every
`KV_BUCKET_TOKENS` = 32 tokens (`sized.rs:326, 390`). Each miss rebuilds the plan, the
arena and the uniform buffers, and recompiles kernels.

`BandBound::Dynamic` removes the reason: bind K/V leaves at the session's **capacity**,
fixed for the whole generation, and let the band supply the live length at run time from
the `cached_len` operand. The traversal reads `[0, cached_len + q]` regardless of the
declared extent, so no extra bytes move, and the masking `Select` remains as the
correctness backstop (R4). Then `symbols = [new_count, capacity]` is constant in steady
state: **one plan, one arena, one uniform set, zero rebuilds.**

`PlanCache<const N>` is fixed-capacity (`sized::PLAN_CACHE_ENTRIES`), holding the k=1
plan and the k>1 speculative plans of D.1 side by side — multi-token is two table
entries, not a branch. Gate: assert `plan_misses == 1` over N steps and `N != 0`.

---

# C. Generic model programs

## C.1 Production decode from spec data

`ProgramSpec` already carries the conflaguration house pattern:
`#[derive(Debug, Clone, Default, PartialEq, Builder, Deserialize, Serialize, Settings)]`
with `#[settings(prefix = "TENSOR")]` (`spec.rs:69-72`), `NodeSpec` is the tagged node
enum (`spec.rs:123-173`), and `Vec<Op>: TryFrom<&ProgramSpec>` is exercised at
`spec.rs:10313-10327` on `specs/mistral_layer.toml` at real dimensions. The data path
exists; production does not use it.

The one missing piece is stacking: the TOML is one layer, production needs 32 with
per-layer weight names. Add composition, not a template language:

```rust
impl ProgramSpec {
    /// Concatenates `other` after `self`, prefixing every id in `other` with `prefix`
    /// and rewiring `other`'s named inputs from `self`'s ids per `wiring`. Pure;
    /// `TryFrom` still does all lowering, so a stacked spec is an ordinary spec.
    pub fn extend(&self, prefix: &str, other: &ProgramSpec, wiring: &Wiring)
        -> Result<ProgramSpec, TensorError>;

    /// 32 layers = `fold` over 32 `extend`s with `{i}` substituted in weight names.
    pub fn stack(&self, count: u32, wiring: &StackWiring) -> Result<ProgramSpec, TensorError>;
}

#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_STACK")]
pub struct StackWiring {
    /// Id of the layer's carried-in activation (`"x"`).
    pub carry_in: String,
    /// Id of the layer's carried-out activation (`"x_next"`).
    pub carry_out: String,
    /// Weight-name template containing `{layer}` (`"blk.{layer}.ffn_down.weight"`).
    pub weight_names: Vec<String>,
    /// Cache-root ids the driver places per layer.
    pub cache_roots: Vec<String>,
}
```

**What a new architecture needs.** Data only, if it is expressible with the existing
`ScalarOp` set (`op.rs:60-78`, deliberately closed) and the existing map grammar
(`spec.rs:221-260` already parses `"s,2*i->si"` — coefficients and offsets). That covers
RoPE variants, GQA/MQA, RMSNorm/LayerNorm, SwiGLU/GeGLU, MoE routing
(`specs/moe_block.toml`), sliding-window and tree masks. **Rust is needed only for:** a
new `ScalarOp` (a scalar machine primitive that is genuinely not composable — the enum's
own doc says this set stays closed), a new `PackedCodec` (D.2), or a new emitter kernel
family. An SSM/Mamba layer is data: its scan is `Keep::Scan` (`op.rs:139-147`), its
conv1d is a two-term window axis (`map.rs:328-338` proves the form). That is the test of
"generic": `qwen35_forward_program`'s hand-written Rust (`generate.rs:932`,
`spec.rs:898..7184`) must be replaceable by a TOML, and the card gate is node-for-node
equality against the Rust builder before the Rust is deleted.

## C.2 Typed roots and typed layer inputs

```rust
/// Replaces `pub type CachedLayerRoots = (NodeId, NodeId, NodeId)` (`spec.rs:2333`,
/// consumed at 15+ sites) and `Qwen35DenseAttentionRoots` (`spec.rs:2344`, a 4-tuple
/// for the partial-rotary remainder). ONE key node, not two: with
/// `IndexPattern::compose` (§A.1) the interleaved cache is read with stride-2 maps, so
/// `k_even`/`k_odd` (`generate.rs:677-680`) stop existing as separate tensors, and a
/// partial-rotary remainder is an affine slice of the same buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheRoots { pub key: NodeId, pub value: NodeId }

/// Replaces `append_mistral_cached_layer`'s 23 positional parameters and its
/// `#[allow(clippy::too_many_arguments)]` (`spec.rs:2379-2404`). Named fields make
/// swapping two `NodeId`s a compile error (P11 newtype ids); `Option` is not used as a
/// flag — `qk_norm` carries the three nodes it needs or the variant that has none.
#[derive(Debug, Clone, Builder)]
pub struct LayerInputs {
    pub carry: NodeId, pub inv_dim: NodeId, pub eps: NodeId,
    pub inv_sqrt_head_dim: NodeId, pub rope_cos: NodeId, pub rope_sin: NodeId,
    pub attn_norm: NodeId, pub ffn_norm: NodeId,
    pub wq: NodeId, pub wk: NodeId, pub wv: NodeId, pub wo: NodeId,
    pub w_gate: NodeId, pub w_up: NodeId, pub w_down: NodeId,
    pub cache: CacheRoots, pub mask: MaskSource, pub head_norm: HeadNorm,
}

pub enum HeadNorm { None, PerHead { q: NodeId, k: NodeId, inv_head_dim: NodeId } }
/// A mask is either derived (Iota-built, band-analysable) or supplied (a tree mask for
/// D.1). Rich, not a `bool`.
pub enum MaskSource { Causal, Supplied(NodeId) }
```

## C.3 `cached_len` as an integer leaf

`generate.rs:2323-2324` binds `cached_len` as `QuantizedBlock::Float32(&[cached_len as
f32])`. It becomes `DType::Int32` (`dtype.rs:22`). Forced by two things, not taste:
`BandBound::Dynamic` requires `DType::is_integer` (`dtype.rs:57`), and `IndexMap::
Computed`'s indices already require it (`shape.rs:283-291`) — D.3's selector output is
the same shape. The float-only executor gate that today exempts index nodes
(`bind::index_node_ids`, `bind.rs:2746`) extends to band-bound operands by the same
mechanism, one function, not a second exemption list.

## C.4 Sizing and capability config

`KV_BUCKET_TOKENS` lives in the IR crate (`proxima-tensor/src/sized.rs:326`, asserted
= 32 at `sized.rs:390`) though it is a Metal plan-cache-key policy. It moves to
`omega/src/sized.rs` as `KV_EXTENT_BUCKET_TOKENS`, and §B.7 makes its default 1: with a
dynamic band the plan key no longer varies with KV length, so bucketing has nothing to
buy. Better framing in one sentence: the right fix is deletion, and the move is what the
card lands so that the deletion is a measured default change rather than an assumption.

New build-time consts, all through the existing `build.rs` + `sized` pattern (§12), never
literals in source: `MAX_INLINE_STAGES` (proxima-tensor, const-generic-shaped, sits
beside `MAX_INLINE_RANK = 4` at `sized.rs:72` and `READY_BATCH_CAPACITY = 3` at
`sized.rs:78`), `MAX_INLINE_BARRIER_SLOTS`, `PLAN_CACHE_ENTRIES`,
`COMMAND_BUFFER_CAPACITY` (omega). Runtime `FusionRules`/`FusionCost`/`SubmitConfig`
defaults are seeded FROM those consts via `fn default_x() -> T { sized::X }` so the
no_std+no_alloc tier and the std tier cannot disagree, and a parity test pins it
(`defaults_track_the_sized_floor`).

---

# E. Ordering, gates, and what the constraints changed

## E.1 Cards, each with its gate

Standing gates on every card: parity ≤1e-4 vs the CPU evaluator on the real program;
100 tokens byte-identical where the card is lossless; `ms/token` and `gpu_exec_ms` not
worse beyond CoV over 5 runs; allocation counter unchanged or lower; every asserted
count non-zero.

| # | card | gate specific to it |
|---|---|---|
| 0 | `IndexPattern::compose` + strided extent resolution | `bind` output over `mistral_layer.toml` at real dims compares **equal** to pre-card, node for node; a stride-2 fixture that previously returned `UnconstrainedDim` now binds |
| 1 | `BoundOpKind::Loop` + `StepArg::Stage`; Elementwise/Reduce become 1-stage Loops; 5 emitters + CPU ported | dispatch count still **616**; 100x byte-identical logits; ms/token within CoV |
| 2 | R2 epilogue + R3 broadcast-epilogue | 616 → **520** MEASURED; parity ≤1e-4 vs *unfused* CPU |
| 3 | R4 band + R5 online-softmax; delete `CachedAttention` and its 11 companions | attention op count still 32; attention arm `gpu_exec_ms` falls (2t→t); text identical over 100 tokens; `grep -c CachedAttention` = 0 |
| 4 | K/V bound at capacity; band supplies live length; `KV_EXTENT_BUCKET_TOKENS` moves to omega, default 1 | `plan_misses == 1` over 128 steps, and `steps == 128` |
| 5 | pure `plan()` + `Schedule` + `Command`; one driver; thread-locals → fields | `--no-default-features --features alloc` builds `proxima-tensor` **and names the modules it built**; entry-point count 10 → 1; allocation counter **0** over 128 steps with the step count asserted |
| 6 | `poll_complete` replaces `waitUntilCompleted` | CPU time during a pass is available to the drafter: measured overlap > 0 |
| 7 | `FusionCost` replaces `quarantine_broadcast_operands` + `StillLive` veto; R1 over RoPE/GQA | 520 → **264** target, MEASURED per card, not projected |
| 8 | `CacheRoots`/`LayerInputs`/`MaskSource`; `cached_len` Int32; `ProgramSpec::stack`; production from TOML | TOML-built program equals Rust-built program node-for-node, THEN the Rust builders delete |
| D5 | f16 KV (`PackedCodec::Float16`, `metal.rs:441-444`) | exact-match-at-64 ≥ 0.99 over 200 prompts; KV bytes/token halved MEASURED |
| D4a | output.weight Q6_K → Q4_K (probe line 32) | exact-match-at-64 ≥ 0.98 |
| D1a | `nr1` s-axis fold in the packed-row kernel (`msl.rs:3171-3200`) | weight bytes/pass MEASURED **flat** as `s` goes 1→4; precondition for every `÷ a` row in §D.7 |
| D1b | n-gram drafter + verify + acceptance in `Cursor`; `next_ids` stops being `vec![token_id]` (`generate.rs:2503`) | 100 tokens **byte-identical** to k=1 greedy at 3 seeds; acceptance reported as a distribution, not a mean |
| D6 | bandwidth instrumentation (pure-read arm + memcpy control) | a mechanism for 161 vs 400 GB/s, or the written statement that none was found |
| D2 | lift `UnrepresentableGgmlType` for Q3_K (`proxima-model-interop/src/bind.rs:70-73`) + dequant body per backend + requantizer | exact-match-at-64 ≥ 0.90 AND mean KL ≤ 0.05 nats |
| D3a | packed-row-blocked kernel accepts a row-axis `Lookup`; `classify_packed_row_block`'s `gather_count == 0` gate (`msl.rs:1039-1052, 1389`) lifts | **ns/element ≤ the dense arm** at e=0.80 — the gate ROW 180/181 (`discipline.md:16430, 16471`) failed at 0.2-0.29 vs 0.057 ns/element |
| D3b | selector program (Greater → `Keep::Scan` → DUMP-slot scatter) + fixed budget `c` | pre-registered: exact-match-at-64 ≥ 0.95 at e=0.80; mass-recall p10 ≥ 0.80; **random-row control must fail**; plan_misses still 1 (fixed shape) |

## E.2 What each constraint changed, and what it killed

**no_std + alloc tier.** Forced `Schedule`, `Command`, `Band` and `KernelSpec` to be pure
data with caller-owned buffers, which is what forced `plan()` out of the device.
*Abandoned:* keeping today's `Plan`, which owns `BufferArena` and `PlanUniforms` of
`Retained<MTLBuffer>` (`metal.rs:341-347`). It cannot compile below std, cannot be
constructed without a GPU, and is why `plan()` does device IO at all.

**The pipe question.** *Abandoned:* a `Fuse: Pipe<In = ReadyBatch, Out = ReadyBatch>`
stage between `BoundOpBuilder` and the executor. It answered the first question honestly
(it is algebra, and it is a pipe) but failed the second: `BoundOpBuilder` already holds
producers in `held` (`bind.rs:724-731`) and already implements `Pipe`
(`bind.rs:992-1003`), so the call site
`shapes.and_then(builder)` is the identical line before and after. The stage moved code;
it gave no caller anything. *Also abandoned:* a `Decide`-shaped stop type — stop is
`Err(StepEnd::Eos)` on an observe pipe.

**Reuse-first.** *Abandoned:* `BoundOpKind::FlashAttention` — a cleaner macro-op that
takes its strides from `Layout` and drops the operand-count discriminator. It fixes audit
items 2 and 3 and leaves items 1 and 9 exactly where they are: still a kind named after a
model concept, still needing a matcher, still invisible to any program that lays the same
attention out differently. Writing the paragraph that defended it was the finding.

**A measured loss beats a clean argument.** *Abandoned twice, by the byte probe:* (i) that
a k-token pass amortizes weights for free — the packed-row kernel batches 4 *output* rows
over one activation (`msl.rs:3171-3200`), so `s = k` re-streams every weight row k times,
and D.1's entire byte win is conditional on an `nr1` fold that does not exist; (ii) that
row elision is a bytes win judged by bytes — the landed CPU probe measured 0.2-0.29
ns/element against a 0.057 ns/element dense baseline (ROW 180/181,
`discipline.md:16430, 16471`), so D.3's gate is ns/element, and a bytes-only gate would
have passed the arm that already lost.

**Lock-free / lock discipline.** *Abandoned:* replacing the eight thread-locals with
process-wide `Arc<Mutex<..>>` caches so the driver could be `Send`. The driver is
per-core and shared-nothing (prime), so ownership removes the question instead of
answering it — a lock is a missing owner, and the owner is `MetalDriver`.

**"Less work + same output = keep."** Forced R4's shape: the band restricts the traversal
but the body **keeps the masking `Select`**, so the rule cannot change output, only work.
*Abandoned:* deleting the `Select` when a band is derived, which is what today's matcher
effectively does by absorbing the whole mask chain
(`removable_attention_dependencies`, `bind.rs:2585-2618`) — and which is why a band that
is wrong is silently wrong instead of merely slow.

## E.3 The central claim as a lint: what is NOT a pipe here

1. `Op`, `BoundOp`, `Loop`, `Stage`, `Band`, `Plan`, `Schedule`, `Command`, `CacheRoots`,
   `LayerInputs` — **values**. Data that flows through pipes is not itself a pipe.
   Structural, not habit.
2. `FusionRules`, `FusionCost`, `PlanConfig`, `SubmitConfig`, `StackWiring` — **config**,
   which is the other first-class surface (P4), and each is `Settings + Builder` with
   defaults seeded from `sized::`.
3. `MetalDriver` — a **resource owner** for `!Send` device handles. Its behaviour
   (`submit`, `poll_complete`) is a pipe; the ownership is not.
4. `enum Step` — the **suspension form**, and the one runtime state discriminator in the
   design. Justified by cancellation: `poll_advance` must resume a step it did not start,
   so something must store a step of statically-unknown state. The transitions themselves
   are typestate, so no `match` decides *what may happen next* — only *where to resume*.
5. `ByteStreamParser` is not used and not extended: nothing here is fed bytes and polled
   an unbounded number of times per feed (`sans_io.rs:41-52` states that boundary).

**No behaviour in this design is a non-pipe.** Every stage of the driver, every rule
application boundary, and the single Metal edge are `Pipe` impls composing with
`AndThen`.

## E.4 Where information is destroyed today, and what restores it

| destroyed | site | restored by |
|---|---|---|
| backend capability SET → `bool` | `bind.rs:2639` | `FusionRules` |
| band static-vs-dynamic → `operands.len() == 8 \| 9` | `bind.rs:237-239`, `cpu.rs:4847`, `msl.rs:2030`, `msl.rs:2542` | `BandBound::{Affine,Dynamic}` |
| band bounds → `i64::MIN`/`i64::MAX` sentinels | `bind.rs:2510-2511`, `msl.rs:2530` | `Band` |
| emitter routing → substring grep of MSL | `metal.rs:1631-1692` | `KernelSpec { family: KernelFamily }` |
| hazard identity/scope → `reset()` clears both sets; `MTLBarrierScope::Buffers` | `metal.rs:874-877`, `metal.rs:1105` | `BarrierSet { slots }` + `memoryBarrierWithResources` |
| which root is which → `(NodeId, NodeId, NodeId)` | `spec.rs:2333` | `CacheRoots` |
| an integer → `f32` | `generate.rs:2323` | `DType::Int32` |
| two distinct facts → one `None` (feature-off vs out-of-range) | `metal.rs:421-432` | the field exists unconditionally in `Schedule` |
| step counters → a "read exactly once" protocol in a comment | `metal.rs:2719`, `generate.rs:2468` | `SampledStep` owns `StepCounters` |

## E.5 Worked example, which is the test (P17)

Tiny attention at `q=2, h=1, g=1, t=4, p=2, d=2`, real interleaved RoPE layout, written
as `ProgramSpec` TOML so it exercises the same path production will.

```rust
/// Reads as documentation: the seven-op attention chain and its fused one-`Loop` form
/// must agree, and the fused form must be ONE bound op with FIVE stages — not a
/// macro-op, and not seven dispatches.
#[proxima::test]
#[case::unfused(FusionRules::none(),  7, 0)]
#[case::prologue_only(FusionRules::builder().prologue(true).build(), 5, 0)]
#[case::full(FusionRules::all(), 1, 5)]
async fn attention_fuses_by_structure_not_by_name(
    #[case] rules: FusionRules, #[case] bound_ops: usize, #[case] stages: usize,
) {
    let spec: ProgramSpec = toml::from_str(include_str!("../specs/causal_attention.toml"))
        .expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers");
    let shapes = shape::infer(&program, &[SEQ]).expect("infers");

    let resolved = bind::with_rules(&program, &shapes, &[root], rules).expect("binds");

    assert_eq!(resolved.len(), bound_ops, "fusion is counted, not asserted in prose");
    if stages > 0 {
        let BoundOpKind::Loop { stages: got, operands, .. } = &resolved[0].kind
            else { panic!("attention must be a Loop, not a named kind") };
        assert_eq!(got.len(), stages);
        assert_eq!(operands.len(), 4, "Q, K, V, cached_len — no duplicated operands");
        assert!(matches!(got[1].band, Some(Band { upper: BandBound::Dynamic { .. }, .. })),
                "the live length is an operand read, not a compiled-in sentinel");
        // strides come from Layout, so a transposed-K program fuses identically
        assert_eq!(operands[1].1.strides.len(), resolved[0].extents.len());
    }

    // the values agree at every rule setting — a rule may cost time, never accuracy
    let reference = cpu::evaluate_with_rules(&program, &shapes, &blocks, FusionRules::none());
    let fused     = cpu::evaluate_with_rules(&program, &shapes, &blocks, rules);
    assert_close(&fused, &reference, 1e-6);
}

/// The transposed-K twin: the SAME assertions against a program whose K operand is
/// laid out `[h, t, p]` instead of `[t, h, p]`. Today this program falls back to the
/// 7-op chain because `bind.rs:2465-2472` compares eight literal stride tuples.
#[proxima::test]
async fn the_same_attention_laid_out_differently_fuses_the_same_way() { /* .. */ }
```

The second test is the one that says the design is a rule and not a matcher, and it is
the test that cannot pass on main.

## E.6 Prerequisites this work absorbs rather than defers

Three surfaced while reading and are inside the cards above, not filed:
`eliminate_masked_window_reduce` (`bind.rs:734-773`) is subsumed by A.1 and deletes with
card 0; the `element_body`/`split_axis` total-accessor special cases
(`bind.rs:326, 409`) delete with card 3; and `#[ignore]` on
`a_mistral_layer_written_as_toml_evaluates_at_its_real_dimensions`
(`spec.rs:10339-10341`) is removed by card 7 — its stated reason is that unfused FFN
recompute makes it too slow, which is exactly what `FusionCost` fixes, so the ignore
becomes a gate rather than staying a comment.
