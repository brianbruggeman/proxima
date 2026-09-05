# design-B2 — the Metal decode path as pipes, sans-IO FSMs, RISC algebra, generic

Grounding: every file:line below was opened in this session. `Pipe` is
`proxima-primitives/src/pipe/primitives.rs:88-99` — no lifetime parameters on the
trait, `fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out,
Self::Err>>`, **no `Send` bound** (primitives.rs:19-25: local is root, `SendPipe`
is the additive form). Every signature here type-checks against that shape:
implementing *types* may carry lifetimes; the trait does not.

**The shape chosen, in one paragraph.** The decode path becomes one driver built
from eight pipes, threaded by one `Step` enum whose transitions consume the old
variant. `BoundOpKind::CachedAttention` (bind.rs:225-251) is deleted and replaced
by `BoundOpKind::ReduceChain` — a *shared-axis multi-reduce* with a structurally
derived `Band` — so the fused kernel becomes a rewrite RULE over the existing
algebra rather than a model matcher. Zero new `Op`/`ScalarOp`/`IndexMap` variants
are required; one existing function (`unify_iteration_space`, shape.rs:208-229)
must be extended, and the proof that nothing smaller works is in §A.3.

**The contested decision** (stated first because it is the one a reader will
attack): I put the online-softmax tuple accumulator `(m, l, acc)` in the
**emitter**, not in the IR. The alternative — a tuple-valued reduce monoid, a new
`ScalarOp`/`ReduceInit` pair — was the obvious way to make the fused kernel
"legal". I rejected it: op.rs:53-57 states `ScalarOp` is "the one closed set in
this crate that stays closed... not an extension point", and op.rs's `Keep` doc
already sets the precedent that "log-depth prefix sum is a *scheduling* decision
about a `Keep::Scan` reduce, not a different operation". Online softmax is the
same kind of decision: a legal reassociation of an associative reduce chain
(`ScalarOp::is_associative`, op.rs:112-118 already exists to authorise exactly
this). Putting it in the IR would have bought one kernel and cost the closed
vocabulary.

---

# D. BYTES — PRIMARY

## D.0 The arithmetic that must reconcile

All bytes are 10^6 = 1 MB. Model: openchat-3.5 Q4_K_S, 32 layers, d_model 4096,
d_ffn 14336, 32 query heads, 8 KV heads, head_dim 128, vocab 32000.

Codec byte-per-element ratios (SOURCED, `omega/src/msl.rs:439-444`: Q4_K =
144 bytes / 256 elements; the K-quant super-block family is 256 elements wide):

| codec | bytes/256 | bytes/elt | ratio vs Q4_K |
|---|---|---|---|
| Q6_K | 210 (`msl.rs:447-455`) | 0.8203 | 1.458 |
| Q4_K | 144 (`msl.rs:439`) | 0.5625 | 1.000 |
| Q3_K | 110 (ggml, not yet in `PackedCodec`) | 0.4297 | 0.764 |
| Q2_K | 84 (ggml, not yet in `PackedCodec`) | 0.3281 | 0.583 |
| f32 (KV today) | 1024 | 4.0 | — |
| f16 KV | 512 | 2.0 | 0.500 of f32 |
| Q8_0 KV | 34 B / 32 elt | 1.0625 | 0.266 of f32 |

Baseline stream per generated token — this table is the reconciliation the task
requires, and it closes to 4,167.6 MB exactly:

| block | elements | codec | MB | SOURCED / DERIVED |
|---|---|---|---|---|
| FFN gate (32 × 4096×14336) | 1,879,048,192 | Q4_K | 1,056.96 | DERIVED from dims × msl.rs:439 |
| FFN up | 1,879,048,192 | Q4_K | 1,056.96 | DERIVED |
| FFN down | 1,879,048,192 | Q4_K | 1,056.96 | DERIVED |
| **FFN subtotal** | 5,637,144,576 | | **3,170.9** | matches the task's 3,170.9 |
| attn q (32 × 4096×4096) | 536,870,912 | Q4_K | 302.0 | DERIVED |
| attn o | 536,870,912 | Q4_K | 302.0 | DERIVED |
| attn k (32 × 4096×1024) | 134,217,728 | Q4_K | 75.5 | DERIVED |
| attn v | 134,217,728 | Q4_K | 75.5 | DERIVED |
| **attn subtotal** | 1,342,177,280 | | **755.0** | matches 755.0 |
| output.weight (32000×4096) | 131,072,000 | Q6_K | 107.5 | matches 107.5 |
| **matvec subtotal** | | | **4,033.4** | |
| KV read (32 lyr × 2 × 8 heads × 128 × 512 pos × 4 B) | | f32 | 134.2 | matches 134.2; **implies cached_len = 512** |
| **TOTAL** | | | **4,167.6** | |

The KV figure pins the operating point: 512 cached positions × 262,144 B/position
= 134,217,728 B. Every ms/token number below is *at cached_len = 512*; KV bytes
grow linearly with context and every other term does not, so the byte budget is
context-dependent and the cards state the context they measured at.

Cross-check against measurement (SOURCED, `dispatch-census.md:32-35`): MATVEC 225
ops = 23.24 ms (CoV 0.7%). 4,033.4 MB / 23.24 ms = **173.6 GB/s** achieved on the
matvec stream. ATTN 32 ops = 3.0-3.6 ms for the 134.2 MB KV stream = 37-45 GB/s
achieved. Steady-state wall 28.82 ms = gpu_exec 27.52 + host 1.30
(`dispatch-census.md:25-30`). The gap to llama.cpp 17.45 is 11.4 ms and it is
GPU-side.

**Two independent gaps, and the design must attack both.** M1 Max spec is 400
GB/s. The matvec stream achieves 173.6 GB/s = **43.4% of spec** (MEASURED /
DERIVED from the census). So there is a *bytes* gap and an *efficiency* gap, and
a bytes-only plan divides an already-inefficient number.

## D.1 The budget, stated three ways

Let `H` = host ms/token, `B_eff` = achieved GB/s, `M` = MB streamed per generated
token. Then `ms/token = M / B_eff + H`.

Target 3.5 ms/token. Host today is 1.30 ms (census:27), and under a k-token pass
host amortizes across accepted tokens (one command buffer, one readback, one
plan lookup per pass — census:27 itemises emit 0.418 + op_setup 0.158 +
encode_dispatch 0.209 + pipeline_lookup 0.013 + readback 0.005 + greedy 0.044 +
build_position_inputs 0.002 + unexplained 0.453, all per *pass* not per token).

| condition | host/token | GPU budget | MB allowed |
|---|---|---|---|
| B_eff = 173.6 (MEASURED today), A = 1 | 1.30 | 2.20 ms | **382 MB** |
| B_eff = 173.6, A = 3.0 accepted/pass | 0.43 | 3.07 ms | **533 MB** |
| B_eff = 300 (ASSUMED, card D-1 measures) , A = 3.0 | 0.43 | 3.07 ms | 921 MB |
| B_eff = 400 (spec ceiling), A = 3.0 | 0.43 | 3.07 ms | 1,228 MB |
| B_eff = 400, A = 1, H = 0 | 0 | 3.50 ms | 1,400 MB (the task's figure) |

The task's "≤ 1.4 GB/token" is the **B_eff = 400, H = 0** cell. It is the loosest
of the five. The binding cell is the first: **382 MB at today's measured
efficiency**, which is a 10.9× byte reduction from 4,167.6.

## D.2 Lever pricing — each lever, bytes before → after, algebra, gate

Formula, one line, every symbol measured by a named card:

```
MB/token = [ (gate+up) · c_ffn · d_u
           + down      · c_ffn · g_u
           + attn      · c_attn · h_u
           + out       · c_out  · s
           + KV        · c_kv ] / A
```

- `A` = accepted tokens per weight pass (lever 1)
- `d_u` = **union** row density over the k tokens of a pass, gate/up (lever 2)
- `g_u` = union *block-group* density for ffn_down (lever 2b — the contraction-axis case)
- `h_u` = head-group density for the attention projections (lever 2c)
- `c_*` = codec ratios from D.0 (lever 3)
- `s` = output.weight shortlist fraction (lever 4)
- `c_kv` = KV codec ratio (lever 5)

### Lever 1 — multi-token pass (draft + verify). A: 1 → 2.0-3.0. LOSSLESS.

Bytes before: 4,167.6 / token. After: 4,167.6 / A. This is the only lever that is
**exactly lossless** under greedy decoding: verification accepts a drafted token
only where the verifier's argmax equals the draft, so the emitted text is
byte-identical to A = 1. Gate: **text identical, 100× byte-identical** — the
strongest gate available, and it is available only here.

Algebra required: **none new.** The `s` axis exists; the program and plan cache
are already `new_count`-generic (byte-levers-probe.md:4-5, `generate.rs:2394`,
cache key `(new_count, bucket)` at `generate.rs:1248,1323`).

What blocks it is a **kernel**, not the algebra: the packed-row kernel batches 4
*output* rows per simdgroup over ONE activation and has no s-axis fold
(byte-levers-probe.md:8-11, `msl.rs:3171-3200` — `push_packed_row_blocked_body`,
opened, `#[allow(clippy::too_many_arguments)]` at msl.rs:3170). With s = k and no
fold, weights re-stream k times and `A` cancels: **factor 1.0, not k.** The nr1
fold is therefore card D-3 and is a prerequisite of every other lever's
denominator.

Draft source: `Pipe<In = DraftRequest, Out = DraftSpan>` — a transform pipe. An
n-gram prompt-lookup draft needs no second model and no second weight stream (0
added bytes); a small draft model is the *same pipe shape* with its own bytes
added to the numerator. Start with n-gram: it keeps the numerator untouched.

Quality gate: text identical. Kill criterion: if measured `A < 1.6` at k = 4 on
the held-out prompt set, the lever does not pay for the verify dispatch and the
card is killed (a k = 4 pass costs ~1.05× the single-token GPU work for the
folded matvec; below A = 1.6 the pass is a loss).

`A` is **ASSUMED 2.0 at k = 4 / 3.0 at k = 8** pending card D-2.

### Lever 2 — dynamic row elision, FFN. This is where the union density bites.

The FFN is 3,170.9 MB = **76.1%** of the stream. Contextual sparsity elides
hidden neurons whose SwiGLU gate is below threshold.

**2a — gate/up (2,113.9 MB).** Row-granular: hidden neuron j selects row j of
gate and row j of up, each a contiguous 4096-element row = 2,304 B of Q4_K. Row
granularity is coarser than the 256-element block, so byte savings are linear in
density. Algebra: `IndexMap::Computed { indices, index_map, base, gathered_dim }`
(map.rs:134-151) — **exists, and is live in production** for MoE expert routing
on CPU (byte-levers-probe.md:16-18, `cpu.rs:7091`). Nothing new.

**2b — ffn_down (1,056.96 MB): the contraction-axis case, and it is decisive.**
`down` is [4096, 14336] contracting over 14336. Eliding hidden neuron j removes
*column* j of down — i.e. one element out of every 14336-long contracted row. The
Q4_K super-block is 256 elements **along that same contraction axis**
(msl.rs:439-444). For an unstructured selection of density d, the probability a
given 256-block is entirely unselected is `(1-d)^256`:

| d | (1-d)^256 | fraction of down's bytes still read |
|---|---|---|
| 0.25 | 1.0e-32 | 1.000 |
| 0.10 | 2.0e-12 | 1.000 |
| 0.02 | 0.0055 | 0.994 |

**Unstructured elision saves zero bytes on ffn_down.** The only way down
participates is *block-granular* selection: the selector picks groups of 256
consecutive hidden neurons, so `g_u` is a group density and blocks are dropped
whole. Group density is strictly worse than element density (a group survives if
any of its 256 members does), so `g_u ≥ d_u` always, and the card that measures
one must measure both. This is the single most likely place the plan fails, and
it is why D-5 (below) is gated on a *measured* `g_u`, not a hoped one.

**2c — multi-token × elision UNION density.** Under a k-token pass the row set is
the **union** over k tokens. Independent per-token selection gives an upper bound
`d_u = 1 - (1-d)^k`:

| d | k=2 | k=4 | k=8 |
|---|---|---|---|
| 0.25 | 0.438 | 0.684 | 0.900 |
| 0.15 | 0.278 | 0.478 | 0.728 |

At d = 0.25, k = 4 the union costs 0.684 where the per-token density was 0.25:
the union eats 2.7× of the elision gain. Adjacent-token neuron sets are
correlated (consecutive tokens share most of the residual stream), so the true
`d_u` is below the independent bound — but *how far below is the load-bearing
unmeasured number of this entire plan*. Card D-2 measures `d_u(k)` and `g_u(k)`
directly by instrumenting the gate activations on the held-out prompt set; it
runs **before** any kernel work, because if `d_u(4) > 0.75` the elision lever is
dead and the ordering changes.

Values used below are **ASSUMED**: `d_u(4) = 0.55`, `d_u(8) = 0.45` (i.e.
substantially correlated), `g_u(8) = 0.45` for 256-groups.

Metal blocker: `classify_packed_row_block` requires `gather_count == 0`
(byte-levers-probe.md:19-21, and the same gate is visible on the cooperative
route at `msl.rs:1039-1052` which I opened — `gather_count(resolved) == 0`), so
*any* gathered weight falls to the element-granular serial kernel
(`push_gather_fetch`, msl.rs:2286-2318). Card D-4 teaches the packed-row kernel a
**row-axis** gather: one index read per 4096-element row, amortized 1/4096, which
is a different cost structure from the element-granular gather the landed probe
measured at 0.2-0.29 ns/element against DRAM 0.057 (byte-levers-probe.md:24-25,
`benches/bench_dynamic_elision.rs`, "generic DISPATCH-BOUND"). That amortization
is a **mechanism claim and it is ASSUMED**: D-4's gate is the measured GB/s of
the gathered packed-row kernel, and if it does not clear 0.85× the ungathered
kernel's GB/s the lever is killed regardless of the byte saving.

Is the selector a pipe? **Yes, and it is also expressible in-graph — I wrote
both.** In-graph, no host round trip:

```rust
// selector, entirely in the existing algebra: threshold the gate activation,
// scan it to a compacted index list, gather rows with it.
//  %g   = Reduce(Add) over Multiply(normed, W_gate)      // the existing matvec
//  %k   = Elementwise(Greater, [(%g, affine), (%tau, broadcast)])   // op.rs ScalarOp::Greater
//  %pos = Reduce { body: Add, keep: Keep::Scan, operand: %k, .. }   // op.rs Keep::Scan
//  %idx = Reduce { body: Add, keep: Keep::Reduce,
//                  out_map: IndexMap::scatter(%pos, .., destination_extent) }  // map.rs:160-190
//  %up  = Reduce { operand: W_up,
//                  in_map: IndexMap::Computed { indices: %idx, gathered_dim: 0, .. } }
```

Every node above is an existing variant: `ScalarOp::Greater` (op.rs:78),
`Keep::Scan` (op.rs:142-147), `IndexMap::scatter` (map.rs:160-190),
`IndexMap::Computed` (map.rs:134-151). **No new type.** The index tensor is
logically `DType::Int32` and is carried as f32 by every backend — map.rs:99-107
documents exactly this, with the `2^24` exactness ceiling enforced by
`shape::infer` (`TensorError::GatherExtentExceedsExactFloat`). 14,336 ≪ 2^24, so
**the "no backend plumbs integer buffers" constraint does not bind this lever**;
it binds only a hypothetical > 16.7M-row gather.

Quality gate (text identical is NOT available): **exact-match rate vs the full
model**, defined as the fraction of generated tokens whose greedy argmax equals
the full model's greedy argmax, on a fixed 200-prompt held-out set at 128 tokens
each (25,600 comparisons). Kill criterion: **exact-match < 0.98**.

### Lever 3 — lower-bit codecs. c_ffn 1.0 → 0.764 (Q3_K) or 0.583 (Q2_K).

Bytes: FFN 3,170.9 → 2,423.0 (Q3_K) or 1,848.7 (Q2_K). Algebra: **none new** —
the program already abstracts the codec. `PackedCodec` (msl.rs:788-805) and CPU
`QuantizedBlock` (cpu.rs:3091) carry F32/Q4_K/Q5_K/Q6_K/Q8_0; Q2_K/Q3_K *parse*
in `proxima-gguf/src/types.rs:109-133,245-281` and are rejected at bind with
`UnrepresentableGgmlType` (`proxima-model-interop/src/bind.rs:70-73`;
`capability.rs:141-156`) (byte-levers-probe.md:29-33). The work is a decoder body
in `proxima_gguf::quant` plus an MSL unpack, mirroring the Q6_K body already at
msl.rs:447-470. This is the **lowest-risk large lever**: no new algebra, no new
kernel structure, no selector.

Gate: exact-match ≥ 0.98 as above, plus perplexity delta on the held-out set
reported (not gated — reported, so the owner sees the cost). Kill: exact-match <
0.98 at Q3_K on FFN. Q2_K applied to `down` only is a separate card with its own
gate, because down's contraction-axis role makes it the most error-tolerant of
the three (its errors average over 14336 terms).

### Lever 4 — output.weight (107.5 MB, Q6_K). c_out 1.0 → 0.681; s 1.0 → 0.1.

Requantize Q6_K → Q4_K: 107.5 → 73.7 MB (ratio 0.6857). Shortlist: rank the vocab
with a low-rank probe and evaluate only the top-N rows — but under lever 1 the
verifier needs an argmax it can trust, so a shortlist must carry a *guarantee*.
The honest form: shortlist to N = 3200 (s = 0.1) and gate on exact-match; a
shortlist miss shows up directly as an argmax disagreement, so the metric already
measures the failure. 73.7 × 0.1 = 7.4 MB. Algebra: a shortlist is the same
`IndexMap::Computed` gather as lever 2, over the vocab axis. **No new type.**

### Lever 5 — KV. c_kv 1.0 → 0.5 (f16) → 0.266 (Q8_0).

134.2 → 67.1 → 35.6 MB. At cached_len = 512 this is 3.2% of the stream, but it is
the only term that grows with context: at 4096 positions KV is 1,073.7 MB and
becomes the *second* largest block. The KV path today carries f32 (the 134.2 MB
reconciliation proves it), and the ATTN kernel achieves only 37-45 GB/s
(census:33) — so KV is *also* the efficiency outlier, and halving its bytes may
matter less than fixing its access pattern. Card D-7 measures both arms.

## D.3 The ladder — the product must reach the budget. Here is the arithmetic.

Baseline **4,167.6 MB, 28.82 ms/token** (census:26-27).

| # | config | A | c_ffn | d_u | g_u | c_attn | c_out·s | c_kv | MB/token | ms @173.6 | ms @300 | ms @400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | today | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4,167.6 | 25.3+1.3 = 26.6 | — | — |
| L1 | + nr1 k=4 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2,083.8 | 12.0+0.65 = **12.65** | 7.6 | 5.9 |
| L2 | + Q3_K FFN | 2.0 | 0.764 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1,709.6 | 9.85+0.65 = **10.50** | 6.35 | 4.92 |
| L3 | + KV f16 | 2.0 | 0.764 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1,676.0 | 9.66+0.65 = **10.31** | 6.24 | 4.84 |
| L4 | + gate/up elision | 2.0 | 0.764 | 0.55 | 1.0 | 1.0 | 1.0 | 0.5 | **1,312.8** | 7.56+0.65 = **8.21** | 5.03 | 3.93 |
| L5 | + out Q4_K | 2.0 | 0.764 | 0.55 | 1.0 | 1.0 | 0.681 | 0.5 | **1,295.9** | 7.47+0.65 = **8.12** | 4.97 | 3.89 |
| L6 | + k=8, block-gran down | 3.0 | 0.764 | 0.45 | 0.45 | 1.0 | 0.681 | 0.266 | **703.5** | 4.05+0.43 = **4.48** | 2.78 | 2.19 |
| L7 | + Q3_K attn, shortlist | 3.0 | 0.764 | 0.45 | 0.45 | 0.764 | 0.0681 | 0.266 | **644.0** | 3.71+0.43 = **4.14** | 2.58 | 2.04 |
| L8 | + Q2_K down | 3.0 | see note | 0.45 | 0.45 | 0.764 | 0.0681 | 0.266 | **541.8** | 3.12+0.43 = **3.55** | 2.24 | 1.79 |

Worked arithmetic for **L5** (the row I would defend):
```
gate+up : 2113.9 · 0.764 · 0.55 =  888.2
down    : 1056.96 · 0.764 · 1.0 =  807.5
attn    :  755.0 · 1.0 · 1.0    =  755.0
out     :  107.5 · 0.681 · 1.0  =   73.2
KV      :  134.2 · 0.5          =   67.1
sum                              = 2591.0 ;  / A=2.0  = 1295.5 MB
1295.5 MB / 173.6 GB/s = 7.46 ms ; + host 1.30/2.0 = 0.65 → 8.11 ms/token
```
Worked arithmetic for **L8** (`c_ffn` = 0.764 on gate/up, 0.583 on down):
```
gate+up : 2113.9 · 0.764 · 0.45 =  726.7
down    : 1056.96 · 0.583 · 0.45=  277.3
attn    :  755.0 · 0.764        =  576.8
out     :  107.5 · 0.681 · 0.1  =    7.3
KV      :  134.2 · 0.266        =   35.7
sum                              = 1623.8 ;  / A=3.0 = 541.3 MB
541.3 MB / 173.6 GB/s = 3.12 ms ; + host 1.30/3.0 = 0.43 → 3.55 ms/token
```

## D.4 Is 3.5 ms/token reachable? Stated plainly.

**Not on any number measured today.** The measured state is 28.82 ms/token at
4,167.6 MB and 173.6 GB/s. Reaching 3.5 requires **L8**: eight levers stacked, of
which five are lossy, and whose two load-bearing factors — the union density
`d_u`/`g_u` and the acceptance rate `A` — are **ASSUMED, not measured**.

Under which measured conditions it becomes reachable, precisely:

1. `A ≥ 3.0` at k = 8 on the held-out set (card D-2). If `A = 2.0`, L8 becomes
   812 MB → 4.68 + 0.65 = **5.33 ms**, and 3.5 is out of reach at 173.6 GB/s.
2. `d_u(8) ≤ 0.45` and `g_u(8) ≤ 0.45` (card D-2). At the *independent* bound
   `d_u(8) = 0.90` and no block-granular structure, L8 becomes 1,146 MB → 6.6 +
   0.43 = **7.03 ms**. This factor alone spans 3.55 → 7.03.
3. The gathered packed-row kernel clears 0.85× the ungathered kernel's GB/s
   (card D-4). Below that, the byte saving is spent on dispatch and the elision
   levers are net-negative — which is what the landed CPU probe already measured
   in its element-granular form (byte-levers-probe.md:24-25).
4. Q3_K on FFN + attn and Q2_K on down together hold exact-match ≥ 0.98
   (cards D-6, D-8). Five lossy levers compound; the gate is on the **stack**,
   not on each lever alone, and the stack is re-measured after each addition.

The result that does **not** depend on the union-density or acceptance
assumptions: L1 alone (nr1 fold + draft/verify) is worth 28.82 -> 12.65 ms at
measured bandwidth (2.28x) and is gated on *text identical*, the strongest gate
available. Adding L2 (Q3_K FFN) and L3 (KV f16) reaches 10.31 ms but those two
are Q-gated, not lossless. L1 is the part of this plan I would land first and
would expect to survive.

Efficiency, separately: 173.6 GB/s is 43.4% of the 400 GB/s spec. The nr1 fold
(card D-3) raises arithmetic intensity per streamed weight byte by `k`, which is
the standard mechanism for moving a matvec off the latency-bound regime toward
the bandwidth-bound one — **ASSUMED**, measured by card D-1 (a device-ceiling
probe: a pure streaming kernel over 4 GB, no compute) and card D-3's own GB/s
gate. If `B_eff` reaches 300, L5 already gives **4.97 ms** with only two lossy
levers. That path — fold + codec + fix efficiency — is materially safer than the
elision path and I would run D-1/D-3 before D-4/D-5.

## D.5 Every lever is a program, not a special case

Each lever above is a *program-level* change, which is the property the owner
directive demands. The union of them adds **zero** `Op`, `ScalarOp`, or `IndexMap`
variants:

| lever | algebra used | new variant |
|---|---|---|
| multi-token | the existing `s` axis; `new_count` symbol (generate.rs:2394) | none |
| elision selector | `Greater` + `Keep::Scan` + `IndexMap::scatter` (map.rs:160-190) | none |
| elided matvec | `IndexMap::Computed` (map.rs:134-151) | none |
| codecs | `PackedCodec`/`QuantizedBlock` table extension | none |
| shortlist | `IndexMap::Computed` over the vocab axis | none |
| KV codec | same table | none |

---

# A. Attention in the RISC algebra

## A.1 What attention *is*, structurally

A softmax-weighted banded reduction over a key axis is three reduces over **one
shared axis** `j` (the key axis), plus one elementwise divide:

```
m[q]      = Reduce(Maximum, init=NegativeInfinity) over j of  s[q,j]
l[q]      = Reduce(Add,     init=Zero)             over j of  exp(s[q,j] - m[q])
acc[q,d]  = Reduce(Add,     init=Zero)             over j of  exp(s[q,j] - m[q]) · v[j,d]
out[q,d]  = acc[q,d] / l[q]
   where   s[q,j] = Reduce(Add) over c of q[q,c]·k[j,c]        (the score contraction)
   and     j ranges over Band(q)                               (the causal / cached band)
```

Every line is `Op::Reduce` + `Op::Elementwise` with the existing `ScalarOp`s
(`Maximum`, `Add`, `Exponential`, `Multiply`, `Subtract`, `Divide` — op.rs:60-78)
and the existing `ReduceInit` (`NegativeInfinity`, `Zero` — op.rs:122-131). The
band is `Select(Greater(Iota_key, Iota_query + offset), -inf, s)` — `Op::Iota`
(op.rs:207-230) exists precisely for this and its own doc names the causal mask as
its reason to exist. **The algebra already says attention.** What it cannot say
today is "these three reduces share one loop over `j` and the intermediates are
never materialized", and that is a *binding*-level fact, not an IR fact.

## A.2 The bound form — `ReduceChain` replaces `CachedAttention`

```rust
/// One reduce stage inside a shared-axis chain. Data legal for exactly this
/// stage: no attention words, no model names, no sentinels.
#[derive(Debug, Clone, PartialEq)]
pub struct ReduceStage {
    /// The per-step combine of this stage's operands before accumulating —
    /// the same `ComposedBody` `BoundOpKind::Reduce` already carries.
    pub body: ComposedBody,
    pub reduce_op: ScalarOp,
    pub init: ReduceInit,
    pub keep: Keep,
    /// Operand slice of the chain's `BoundOperands` this stage reads, as a
    /// half-open range. A stage that reads a previous stage's accumulator
    /// names it by `StageId`, never by operand position.
    pub operands: core::ops::Range<u16>,
    pub carries: SmallVec<[StageId; 2]>,
}

/// Position of a stage within one `ReduceChain`. Newtype so a stage index and
/// an operand index cannot be swapped at a call site (P11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StageId(pub u16);

/// A band on one reduced axis, as affine bounds over the iteration space.
/// Replaces `cached_lower_inclusive: i64` / `new_upper_inclusive: i64` and
/// their `i64::MIN`/`i64::MAX` sentinels (bind.rs:2510, bind.rs:243-248).
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    pub axis: u16,
    pub lower: BandBound,
    pub upper: BandBound,
}

/// A bound is either static (an affine function of the iteration space, reusing
/// `map::AxisIndex` — no new grammar) or a runtime scalar node. The runtime case
/// is what `cached_len` is; making it a VARIANT is what deletes the
/// `operands.len() == 8 | 9` discriminator read in four files
/// (bind.rs:237, cpu.rs:4847, msl.rs:2030, msl.rs:2543).
#[derive(Debug, Clone, PartialEq)]
pub enum BandBound {
    Static(crate::map::AxisIndex),
    Runtime { node: NodeId, plus: i32 },
}

pub enum BoundOpKind {
    Elementwise { body: ComposedBody, operands: BoundOperands },
    Reduce { /* unchanged */ .. },
    /// N reduces sharing ONE loop over `band.axis`, with the intermediates
    /// never materialized. Softmax attention is the 3-stage instance; a
    /// mean/variance layer-norm is the 2-stage instance; a plain reduce is the
    /// 1-stage instance and is exactly `Reduce` above, which is why `Reduce`
    /// stays (it is the hot common case and pays no SmallVec).
    ReduceChain {
        stages: SmallVec<[ReduceStage; 3]>,
        operands: BoundOperands,
        band: Band,
        /// Elementwise consumer folded onto the chain's output (`acc / l`).
        epilogue: Option<ComposedBody>,
    },
}
```

**Where the tuple accumulator went.** It is not here. `ReduceChain` says only
"these stages share an axis"; an emitter that keeps one register per stage and
walks `j` once *is* online softmax. `ScalarOp::is_associative` (op.rs:112-118)
authorises the rescale-on-new-max reassociation. The CPU evaluator implements the
identical rule by walking the same loop with the same per-stage accumulators, so
CPU and every emitter share one rule and one parity test.

**Strides come from `Layout`.** `render_cached_attention` (msl.rs:2578) ignores
operand `Layout.strides` and the matcher compensates with eight literal stride
tuples (bind.rs:2452-2474) plus rank gates (bind.rs:2417-2427). `ReduceChain`
carries no shape fields at all: `query_rows`, `cached_key_rows`, `new_key_rows`,
`kv_heads`, `query_groups`, `head_dim` are all recoverable from
`BoundOp::extents` and the operands' `Layout`. Deleting them is what makes "the
same attention laid out differently" fuse instead of falling back.

**The 2t-for-t defect dies structurally.** bind.rs:2504-2510 sets
`cached_key_rows = new_key_rows = key_shape[0]` with `cached_lower_inclusive =
i64::MAX`, and the kernel loop (msl.rs:2586) skips the first half — 2t iterations
for t of work. With one `Band` whose bounds are `BandBound`, a single-range
program produces one range and the loop runs t iterations. There is no second
range to skip because there is no fixed 8/9-operand signature to pad.

## A.3 The one REQUIRED algebra extension, and the proof nothing smaller works

`unify_iteration_space` (shape.rs:208-229, opened) resolves an iteration axis
extent **only** from an operand axis that is a single term with `coeff == 1` and
`offset == 0`:

```rust
if let [term] = axis.terms.as_slice()
    && term.coeff == 1
    && axis.offset == 0
{ /* only here does an extent get resolved */ }
```

Two structures the fused rule needs are excluded by exactly this gate:

- RoPE's even/odd split addresses `2i` and `2i+1` — `coeff == 2`, `offset ∈
  {0,1}`. The census records the consequence: `is_identity_projection` fails on
  the 2i/2i+1 stride (dispatch-census.md:11, bind.rs:1153-1164), so rotated_q/k
  even/odd stay four separate dispatches per layer = 128/token.
- The cached/new key split addresses `j` and `cached_len + j` — `offset ≠ 0`.

**Proof that nothing smaller works:** the extent of the iteration axis is
information that exists *only* in the strided operand's shape. Any design that
does not read it there must be handed it out-of-band — and the 10-field
`CachedAttention` macro-op is precisely that out-of-band channel
(`head_dim = pairs*2` hardcoding the even/odd split, bind.rs:2417-2427). So the
choice is: extend the resolver, or keep a macro-op forever. There is no third
option, because there is no other holder of the extent.

The extension, minimal and local to that function:

```rust
/// Resolve `iter_extent` from an operand axis that is one term with a positive
/// coefficient and a constant offset: the axis covers
/// `[offset, offset + coeff*(iter_extent-1)]` and must fit the operand extent
/// exactly, so `iter_extent = (operand_extent - offset + coeff - 1) / coeff`
/// and `offset + coeff*(iter_extent-1) < operand_extent`. Exactness is required:
/// a partial cover would silently shrink an iteration space.
fn resolve_strided_axis(operand_extent: u64, term: AxisTerm, offset: i32)
    -> Option<u64>;
```

Everything else in §A is a **rule**, not a variant. That is the whole point.

## A.4 The fusion rule *is* a pipe

`fuse_cached_attention: bool` (bind.rs:2635-2640, opened — and `bind_plain` plus
both matchers run twice per plan, bind.rs:2641/2678) collapses "which fused kinds
can this backend render" into one bit. The replacement is not a `FusionKinds`
set type. **A fusion rule is a transform pipe, and "which rules" is which pipes
you compose:**

```rust
/// Every fusion rule has this shape. `BoundProgram` in, `BoundProgram` out —
/// a transform (In -> Out), the load-bearing one of the four forms.
pub struct ShareAxisChain;      // the softmax/layer-norm rule (A.2)
pub struct FuseEpilogue;        // elementwise consumer of a Reduce (census:46-50)
pub struct FusePrologue;        // elementwise producer into reduce operands (today's only rule)

impl Pipe for ShareAxisChain {
    type In = BoundProgram;
    type Out = BoundProgram;
    type Err = TensorError;
    fn call(&self, program: BoundProgram)
        -> impl Future<Output = Result<BoundProgram, TensorError>>
    { async move { rewrite_shared_axis_chains(program) } }
}
```

A backend states its capability by *composing the rules it can render*:

```rust
// metal
let rules = FusePrologue.and_then(FuseEpilogue).and_then(ShareAxisChain);
// wgpu / cuda today: no ReduceChain renderer, so they simply do not compose it
let rules = FusePrologue.and_then(FuseEpilogue);
```

Two binary questions for a hypothetical `FusionKinds` bitflag set: (1) can an
existing primitive express it? Yes — the composition above. (2) What can a caller
do that they could not? Nothing. **So there is no set type.** The bool is deleted
and nothing replaces it. `bind_plain` runs once; the rules run once each, in a
fixed order, over its output.

`ReduceChain` itself must answer the same two questions:
1. *Expressible with an existing primitive?* Attempt written: a `Vec<BoundOp>` of
   three `BoundOpKind::Reduce`s — which is exactly what `bind_plain` produces
   today, and it materializes the `[1, 32, 512]` score tensor. The fact "these
   three share a loop" is the payload; a flat `Vec` has no place to put it. Not
   expressible.
2. *What can a caller do that they could not?* Fuse an attention whose Q/K/V
   layout is not one of the eight stride tuples at bind.rs:2452-2474; fuse a
   layer-norm's sum/sum-of-squares pair (2 dispatches/layer, census:59-61) with
   the *same* rule; emit the chain from wgsl/cuda, which cannot lower
   `CachedAttention` at all (lowering-audit.md:26). Different call sites.
   Passes.

Net type count: `CachedAttention` (10 fields, model semantics) → `ReduceChain` (3
fields, structural). **One type out, one in, ten fields to three.**

## A.5 Migration from `BoundOpKind::CachedAttention`

Ordered so every step has a parity gate against the step before it.

1. Land `Band`/`BandBound`/`ReduceStage`/`ReduceChain` alongside `CachedAttention`;
   nothing emits `ReduceChain` yet. Gate: workspace builds, tier builds unchanged
   (tiers-census.md rows).
2. Add `ShareAxisChain` and make it produce `ReduceChain` from the *same* program
   the matcher matches. Gate: for the production program, the `ReduceChain` and
   the `CachedAttention` bound ops agree on operands and extents (a structural
   equality test, not a numeric one).
3. Implement `ReduceChain` in the CPU evaluator. Gate: **≤ 1e-4 vs the existing
   `CachedAttention` CPU arm on the real program**.
4. Implement `render_reduce_chain` in msl.rs reading strides from `Layout`. Gate:
   ≤ 1e-4 vs CPU; ms/token and gpu_exec_ms not worse beyond CoV (census CoV 0.7%
   for matvec, 9.9% for attn — the attn CoV is the binding one and the gate must
   use it).
5. Flip the rule composition to `ShareAxisChain`; delete `CachedAttention`, the
   two matchers, the eight stride tuples (bind.rs:2452-2474), the rank gates
   (bind.rs:2417-2427), the `operands.len() == 8 | 9` reads in four files, and
   the `i64::MIN`/`MAX` sentinels. Gate: **100× byte-identical, text identical**.
6. Land `resolve_strided_axis` (A.3) and let `ShareAxisChain` fold RoPE even/odd.
   Gate: ≤ 1e-4, dispatch count drops by 128/token, text identical.

---

# B. The decode step as FSM × orchestration over pipes

## B.1 The sans-IO step FSM

Transitions consume the old variant; each variant owns only its legal data; no
runtime "am I in the right state" boolean.

```rust
/// One decode pass. `'p` is the plan borrow; the trait carries no lifetimes, the
/// implementing pipes do (primitives.rs:88-99).
#[derive(Debug)]
#[must_use]
pub enum Pass<'p> {
    /// Nothing bound yet. Carries what a plan key needs and nothing else.
    Requested { context: ContextCursor, draft: DraftSpan },
    /// A plan is in hand; inputs are laid out. No device call has happened.
    Planned { plan: &'p Plan, inputs: PassInputs },
    /// A command list exists as DESCRIPTORS. Still no device call: this is the
    /// sans-IO boundary, and this variant is what a DPDK/SPDK/fuzzer consumer
    /// would drive instead of Metal.
    Encoded { plan: &'p Plan, commands: CommandList },
    /// Handed to the device. The single Metal edge produced this.
    Submitted { plan: &'p Plan, fence: FenceId },
    /// Device signalled; outputs are readable in place.
    Completed { plan: &'p Plan, logits: LogitsView<'p> },
    /// The draft has been checked against the logits.
    Verified { accepted: AcceptedRun },
}

/// Newtype ids so two u64s cannot be swapped at a call site (P11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct FenceId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct PlanKey(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct BufferId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct StepIndex(pub u64);

/// How many drafted tokens the verifier accepted, and where the KV cursor lands.
/// NOT a bool and NOT a count alone: the rejected suffix is what a rollback
/// needs, and destroying it here is what would force reconstruction downstream.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedRun {
    pub tokens: SmallVec<[TokenId; 8]>,
    pub rejected: SmallVec<[TokenId; 8]>,
    pub cursor: ContextCursor,
}
```

## B.2 The driver: one pipeline, four forms, ONE Metal edge

Nine executor entry points (metal.rs:519, 535, 978, 1177, 1192, 1210, 1409, 1484,
1509, 1613 — one driver × {named, placed, timed}) collapse to one composition.
`{named, placed, timed}` become **fields on the request**, which is P4
config-as-composition: a variant is config, not a recompile, and not a fourth
entry point.

| stage | type | In | Out | form | touches device |
|---|---|---|---|---|---|
| plan lookup | `ResolvePlan` | `Pass::Requested` | `Pass::Planned` | transform | **no** |
| encode | `EncodePass` | `Pass::Planned` | `Pass::Encoded` | transform | **no** |
| submit | `SubmitPass` | `Pass::Encoded` | `Pass::Submitted` | transform | **YES — the only one** |
| await | `AwaitFence` | `Pass::Submitted` | `Pass::Completed` | transform (poll) | yes (fence poll) |
| verify | `VerifyDraft` | `Pass::Completed` | `Pass::Verified` | transform | no |
| advance | `AdvanceCursor` | `Pass::Verified` | `Pass::Requested` | transform | no |
| counters | `RecordCounters` | `Pass<'p>` | `Pass<'p>` | **observe** | no |
| draft | `NgramDraft` | `()` | `DraftSpan` | **source** | no |

```rust
pub struct ResolvePlan { cache: PlanCache }          // pure; sans-IO
impl<'p> Pipe for &'p ResolvePlan {
    type In = Pass<'p>;
    type Out = Pass<'p>;
    type Err = DecodeError;
    fn call(&self, pass: Pass<'p>) -> impl Future<Output = Result<Pass<'p>, DecodeError>> {
        async move {
            match pass {
                Pass::Requested { context, draft } => {
                    let plan = self.cache.get(PlanKey::of(&context, &draft))
                        .ok_or(DecodeError::PlanAbsent)?;
                    Ok(Pass::Planned { plan, inputs: PassInputs::of(&context, &draft) })
                }
                other => Err(DecodeError::WrongState { got: other.name() }),
            }
        }
    }
}
```

`Pass` is `#[non_exhaustive]`-free on purpose (it is a closed protocol), and the
`WrongState` arm is the *only* runtime state check in the design — it exists
because `Pipe::call` takes `Self::In` and a pipeline is composed by type, so a
mis-composition is caught at the first call. A typestate encoding (one pipe per
concrete state type) removes even that; I chose the enum because the loop must
carry an *unknown* next state at the `AcceptedRun`/`Requested` join and typestate
there costs a second enum anyway.

**The single Metal edge:**

```rust
/// The ONLY type in the decode path that calls Metal. Everything upstream is
/// sans-IO and drivable from any loop — that is the P11 clause, and this type is
/// the facade being the codec's first consumer.
pub struct SubmitPass { session: core::cell::RefCell<MetalSession> }

/// `RefCell`, not a mutex: `Pipe` is `!Send` by root (primitives.rs:19-25), the
/// session is per-core shared-nothing, and `call(&self, ..)` forces the mutation
/// inward. A `Mutex` here would be a lock where the real answer is an owner (§21).
pub struct MetalSession {
    arena: BufferArena,
    uniforms: UniformRing,
    outputs: OutputPool,
    pipelines: PipelineCache,
    queue: CommandQueue,
}
```

That struct is where **all eight `thread_local! RefCell` globals go**
(metal.rs:266, 273, 2551, 2939, 3038, 3162, 3260, 3266 — six confirmed by
tiers-census.md:14-15 at metal.rs:252, 2477, 2933, 3007, 3150, 3246), plus
`register_checkpoint_mapping`'s side channel. A `Plan` becomes a self-contained
value because nothing it needs lives in a thread-local any more.

## B.3 `plan()` becomes pure

`plan()` today performs device IO — `device_and_queue` + arena + uniform buffers
at metal.rs:490-500 — and the arena is built for all four executors and used by
two (audit-2026-09-04.md:19). Planning splits in two:

```rust
/// Sans-IO. No device, no allocation of device memory, no queue. Deterministic:
/// same program + same shapes + same rules -> same Plan, byte for byte.
pub fn plan(program: &[Op], shapes: &Shapes, rules: &FusionPipeline)
    -> Result<Plan, TensorError>;

/// The device half, owned by SubmitPass and run once per plan, not per token.
impl MetalSession {
    pub fn admit(&mut self, plan: &Plan) -> Result<Residency, DriverError>;
}
```

`Plan` gains the two things that were being recomputed per token:

```rust
pub struct Plan {
    pub ops: Vec<BoundOp>,
    /// Barrier schedule as a STATIC property (audit finding 11/29). Today
    /// `HazardTracker::reset` clears both sets on any barrier and the scope is
    /// `MTLBarrierScope::Buffers` = all buffers (metal.rs:874-877, 1105), and
    /// the schedule is recomputed every token even though it depends only on
    /// the program.
    pub barriers: SmallVec<[BarrierPoint; 32]>,
    /// Which buffer each op reads/writes, resolved at plan time — this is what
    /// makes the barrier a fixpoint rather than a loop-mutated set.
    pub residency: ResidencyPlan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BarrierPoint {
    /// Index into `ops` this barrier precedes.
    pub before: u32,
    /// The resources that actually conflict — `memoryBarrierWithResources`
    /// exists and the all-buffers scope is throwing this information away.
    pub resources: SmallVec<[BufferId; 8]>,
    /// Why, kept because a barrier with no reason is unauditable.
    pub cause: Hazard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hazard { ReadAfterWrite, WriteAfterWrite, WriteAfterRead }
```

`HazardTracker` (metal.rs:839-887) — a struct with methods and loop-mutated sets
— becomes `fn schedule_barriers(&[BoundOp], &ResidencyPlan) -> SmallVec<[BarrierPoint; 32]>`,
a pure function run once per plan. That kills, structurally, the identity defect
the audit records at metal.rs:1103 vs 1121 (checks output identity from
`placement`, records from `device_buffers`, so WAW/WAR are skipped when placement
is `None`) and the ABA stale-pointer case at metal.rs:1131: there are no live
pointers at plan time, only `BufferId`s.

## B.4 `classify_kind` stops grepping MSL

`classify_kind` (metal.rs:1631-1692) substring-greps generated MSL to recover the
emitter's own routing decision, and the profiler groups by `&str`
(lowering-audit.md: "4 substrings of generated source"). The emitter already knows
what it chose; the fix is to **return it** rather than reconstruct it:

```rust
/// The emitter's routing decision, returned WITH the source instead of being
/// re-derived from it. This is the "find where information is destroyed" case:
/// rich (an enum the emitter computed) -> poor (a String) -> reconstructed
/// (a substring grep). Deleting the grep deletes the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelRoute {
    Elementwise, CooperativeReduce, PackedRowBlocked, ReduceChain,
    SerialReduce, Iota, Constant, TiledGemm,
}
pub struct Emitted { pub source: String, pub route: KernelRoute, pub key: KernelKey }
```

`KernelKey` is a struct, not a `String`: `kernel_cache_key` (msl.rs:930-981)
allocates Strings per op per step (lowering-audit.md:31-33, metal.rs:3772-3773)
**and** omits the reduce extents while `cooperative_reduce_width` bakes the
extent-derived width into the source (msl.rs:4390-4406, 4540) — a latent
wrong-kernel-served path that `pipeline_for` (metal.rs:2374-2400) serves from
cache without re-emitting. A `#[derive(Hash, PartialEq, Eq)]` struct containing
the reduce extents fixes the correctness hole and the per-op allocation in one
change.

## B.5 Allocation budget and the test that proves it

Per token, plan hit: **zero allocations.** Per plan miss: bounded, recorded.

Where the storm is today (audit finding 10): `generate.rs:2301-2306` rebuilds
`named_blocks` every step; `2323, 2361-2364, 2400-2406, 2504`; `metal.rs:1032-1040,
1095-1100, 1073`; plus `kernel_cache_key` Strings per op per step
(metal.rs:3772-3773). Each has the same shape and the same fix: a `Vec` rebuilt
per step becomes a slot in `MetalSession` cleared and refilled, and a `String` key
becomes a POD struct key.

```rust
#[cfg(test)]
mod tests {
    /// Reads as English: after the plan is warm, generating 128 tokens performs
    /// zero heap allocations. 128, not 1: a per-token leak of one allocation is
    /// invisible at n=1 and obvious at n=128.
    #[test]
    fn warm_decode_allocates_zero_bytes_per_token() {
        let session = fixture_session();          // real openchat-3.5 program
        let warm = session.generate(prompt, 8);   // plan miss + warm-up, unmeasured
        let counter = AllocCounter::install();
        let _ = session.generate_more(&warm, 128);
        assert_eq!(counter.allocations(), 0, "warm decode allocated");
        assert_eq!(counter.bytes(), 0);
    }
}
```

`AllocCounter` is a `#[global_allocator]` shim in the test harness (std-only, test
cfg). The gate asserts `N == 128` tokens generated as well as `allocations == 0`,
because zero work and successful work emit the same zero.

---

# C. Generic model programs

## C.1 Production decode builds from spec data

`proxima-tensor/specs/mistral_layer.toml` plus the test at spec.rs:10317 prove
the data path works and production does not use it (audit finding 23; the
model-named builders live at spec.rs:898..7184, `generate.rs:624
SingleRangeProgram`, `731 Qwen35SsmShape`).

**The tier constraint decides the shape here.** `config` is std-only
(`proxima-tensor/Cargo.toml:37`: `config = ["std", "dep:bon", "dep:conflaguration",
"dep:serde", "smallvec/serde"]`; `lib.rs:103-104`: "the TOML/serde face, std-only
... The `alloc` tier never sees it"). So **nothing on the alloc-tier bind path may
derive `Deserialize`/`Settings`.** The split:

```rust
/// std-only. A transform pipe: spec source in, program out.
#[cfg(feature = "config")]
pub struct LoadSpec;
#[cfg(feature = "config")]
impl Pipe for LoadSpec {
    type In = SpecSource;                 // Toml(&str) | Baked(&'static [Op])
    type Out = Vec<Op>;
    type Err = SpecError;
    fn call(&self, source: SpecSource) -> impl Future<Output = Result<Vec<Op>, SpecError>>
    { async move { lower_spec(source) } }
}

/// alloc tier. Takes `&[Op]`. Knows nothing about TOML, serde, or bon.
pub fn plan(program: &[Op], shapes: &Shapes, rules: &FusionPipeline)
    -> Result<Plan, TensorError>;
```

Below std, the same TOML is resolved at **build time** into a baked `&'static
[Op]` — the conflaguration tier-2 pattern (`build.rs` IS the config surface), the
same mechanism `proxima-telemetry`'s `sized` module uses. One source of truth, two
tiers, and the alloc-tier bind path never sees `serde`.

## C.2 The positional types

```rust
/// `CachedLayerRoots = (NodeId, NodeId, NodeId)` (spec.rs:2333) consumed at 15+
/// sites: three same-typed positions, so any two can be swapped silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedLayerRoots { pub hidden: NodeId, pub key: NodeId, pub value: NodeId }

/// The 23-NodeId layer builder (spec.rs:3517, `#[allow(too_many_arguments)]`)
/// becomes one struct with a `bon` builder — P4, both surfaces first-class, and
/// the `allow` disappears rather than being justified.
#[derive(Debug, Clone, bon::Builder)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerBindings {
    pub attn_norm: NodeId, pub q: NodeId, pub k: NodeId, pub v: NodeId,
    pub o: NodeId, pub ffn_norm: NodeId, pub gate: NodeId,
    pub up: NodeId, pub down: NodeId, /* .. */
}
```

Note the `cfg_attr`: `LayerBindings` is a *spec-side* type (std/config tier), so
the serde derive is legal there and never reaches the alloc-tier bind path. If
this struct were needed at the alloc tier the derive would have to go — that is
the constraint doing its job.

`cached_len` moves from `DType::Float32` to `DType::Int32`. **This costs no
backend work**: map.rs:99-107 states every backend carries every buffer as f32
including `indices`, with exactness guaranteed below 2^24; `cached_len` is a
sequence position, far below. The DType change is what lets the band read it as
`BandBound::Runtime` without a float comparison.

`KV_BUCKET_TOKENS` moves out of `proxima-tensor/src/sized.rs` into `omega`'s
`sized` (audit finding 20): it is a Metal *cache-key policy*, and an IR crate that
holds a driver's policy cannot be reused by a driver with a different one.

## C.3 What a new architecture needs to add

With A.2 + A.4 + C.1 landed:

| new architecture needs | today | after |
|---|---|---|
| a new layer shape (different norm, different FFN) | Rust: a `append_*` fn in spec.rs | **data only** — a TOML layer |
| GQA with a different group map | Rust: a ninth stride tuple at bind.rs:2452 | **data only** — `Layout` carries it |
| a different attention band (sliding window) | Rust: new fields on `CachedAttention` | **data only** — `Iota`/`Greater`/`Select` |
| a genuinely new scalar primitive | Rust: `ScalarOp` variant | Rust — and op.rs:53-57 says this set stays closed, so the answer is "desugar it" |
| a new weight codec (Q3_K) | Rust: `PackedCodec` + unpack body | **Rust** — one table row + one MSL body |
| a new backend | Rust: a driver | **Rust** — but it composes the fusion rules it can render (A.4) instead of passing a bool |

Two Rust-required rows remain, and both are honest: a codec is machine-level, and
a backend is a driver.

---

# E. Ordering, cards, tripwires

Every card is a **30-minute slice with ONE gated deliverable**. Gate abbreviations:
**P** = parity ≤ 1e-4 vs CPU on the real program; **B** = 100× byte-identical;
**T** = text identical; **S** = ms/token and gpu_exec_ms not worse beyond CoV
(matvec CoV 0.7%, attn CoV 9.9%, census:32-34 — the *attn* CoV binds any card
touching attention); **A0** = allocation counter == 0 with N asserted;
**Q** = exact-match ≥ 0.98 on the 200-prompt / 128-token held-out set;
**X** = the measurement itself is the deliverable.

## Phase 0 — measure before designing further (these gate everything in D)

| card | deliverable | gate |
|---|---|---|
| D-1 | device ceiling probe: a pure streaming kernel over 4 GB, 3 rounds, quiet box | **X**: GB/s with CoV; the number that converts every MB row in D.3 to ms |
| D-2 | union-density + acceptance harness: instrument gate activations and n-gram draft acceptance on the 200-prompt set | **X**: `d_u(k)`, `g_u(k)` for k ∈ {2,4,8}, `A(k)`; **kill D-4/D-5 if d_u(4) > 0.75** |
| E-0 | exact-match harness itself (full-model reference logits cached for the 200-prompt set) | **X**: harness reproduces full-model text at exact-match 1.000 (the degenerate control — if this does not read 1.000, the metric measures something else) |

D-1 and D-2 run first because they are the two ASSUMED factors that span 3.55 →
7.03 ms in §D.4. Designing kernels before they land is spending on an unmeasured
premise.

## Phase 1 — structure (no byte change; every card is `S`-gated)

| card | deliverable | gate |
|---|---|---|
| B-1 | `Pass` enum + `ResolvePlan`/`EncodePass` pipes, old path still live | builds; **P** unchanged |
| B-2 | `SubmitPass` + `MetalSession`; 3 of 8 thread-locals moved in | **P**, **S** |
| B-3 | remaining 5 thread-locals + `register_checkpoint_mapping` moved in | **P**, **S**, `Plan` is `Send`-free-standing |
| B-4 | nine executors → one composition; {named, placed, timed} as request config | **B**, **S** |
| B-5 | `schedule_barriers` pure fn; `Plan.barriers` static | **B**, **S**, barrier count logged |
| B-6 | `memoryBarrierWithResources` with per-point `resources` | **P**, **S** (expect gain; CoV-gated) |
| B-7 | `KernelKey` struct replaces the String key; reduce extents included | **B**, and the wrong-kernel-served hole at msl.rs:930-981 closes |
| B-8 | `Emitted { route }` replaces `classify_kind`'s grep | **B**, profiler groups by enum |
| B-9 | per-step `Vec`s → `MetalSession` slots | **A0** with N = 128 asserted |
| A-1..A-6 | the six migration steps of §A.5 | as listed there |

## Phase 2 — bytes, in the order the arithmetic demands

| card | deliverable | gate | MB after |
|---|---|---|---|
| D-3 | nr1 s-axis fold in `push_packed_row_blocked_body` (msl.rs:3171) | **P**, **S**, GB/s reported | — |
| D-4 | verify FSM: `VerifyDraft` + KV rollback (do not advance past accepted) | **T** (lossless), **B** | — |
| D-5 | n-gram `NgramDraft` source pipe, k = 4 | **T**, A measured | 2,083.8 |
| D-6 | Q3_K decoder in `proxima_gguf::quant` + MSL unpack | **Q**, **P** | 1,709.6 |
| D-7 | KV f16 (arm A) vs KV access-pattern fix (arm B), interleaved | **P**, **Q**, **S** | 1,676.0 |
| D-8 | row-axis gather in the packed-row kernel (relax `gather_count == 0`) | **X**: GB/s ≥ 0.85× ungathered, else **kill** | — |
| D-9 | in-graph selector program (the `Greater`/`Scan`/`scatter` chain of §D.2) | **P**, **Q** | 1,312.8 |
| D-10 | output.weight Q6_K → Q4_K | **Q** | 1,295.9 |
| D-11 | k = 8 + 256-block-granular selection for `down` | **Q**, A and g_u measured | 703.5 |
| D-12 | Q3_K attn + vocab shortlist | **Q** | 644.0 |
| D-13 | Q2_K on `down` only | **Q** — expect this to be where the stack breaks | 541.8 |

Tripwire on the lossy stack: **Q is re-measured on the whole stack after every
card from D-6 onward, never on the card alone.** Five lossy levers compound and a
per-card gate would pass all five while the stack fails.

## E.1 What each constraint CHANGED, and what it killed

**The pipe question (P1).** Changed: `fuse_cached_attention: bool` was going to
become a `FusionKinds` bitflag set. I wrote the composition
(`FusePrologue.and_then(FuseEpilogue).and_then(ShareAxisChain)`), it expresses the
capability exactly, so **there is no set type** — the bool is deleted and nothing
replaces it. *Abandoned:* a `Selector` type for row elision. A selector is
`Pipe<In = Activation, Out = RowSet>` — and better still, the in-graph
`Greater`/`Keep::Scan`/`IndexMap::scatter` chain of §D.2, which I wrote out. Both
expressions exist; the type does not.

**The second binary question (what can a caller do?).** *Abandoned:* an
`Executor` trait with `run_named` / `run_placed` / `run_timed`, the obvious
collapse of the nine entry points. I wrote both call sites:
`executor.run_placed(plan, inputs)` vs `driver.call(Pass::Requested { .. })` with
placement as a request field. The trait line adds nothing, and a trait over
backends would need `Box<dyn>` (P20 forbids it in this crate). Deleted before it
was written.

**P11 sans-IO.** Changed the FSM boundary: my first cut had `Encoded` already
holding `MTLCommandBuffer`. That makes the encode stage untestable off-device and
forfeits the kernel-bypass floor. `CommandList` is now descriptors, and
`Pass::Encoded` is exactly the variant a DPDK/SPDK/fuzzer consumer drives.
*Abandoned:* putting the online-softmax tuple monoid in `ScalarOp` (op.rs:53-57
says the set stays closed; the `Keep::Scan` precedent puts scheduling in the
emitter).

**P3 tiers.** Changed C.1 decisively. My first cut had `LayerBindings` and the
spec types on the bind path with `#[derive(Deserialize)]`. `Cargo.toml:37` +
`lib.rs:103-104` make `config` std-only, so that would have dragged std onto the
alloc tier. The split — `LoadSpec` as a std-gated pipe producing `Vec<Op>`, the
alloc-tier `plan()` taking `&[Op]` — is what the constraint produced, and it is
better: the bind path is now provably free of serde. It also handed me the no_std
answer for free (build.rs bakes the same TOML to a `&'static [Op]`).

**Lock discipline (§21).** *Abandoned:* `Arc<Mutex<MetalSession>>`, which was the
reflex once `call(&self, ..)` forced the mutation inward. `Pipe` is `!Send` by
root (primitives.rs:19-25) and the session is per-core shared-nothing, so the
answer was an **owner**, not a lock: `RefCell` inside `SubmitPass`. The
mutex-shaped question ("which lock guards the arena?") was the wrong question.

**Reuse-first (P1).** Changed `Band`: my first cut had `lower: i64, upper: i64`,
which is what `CachedAttention` has today with `i64::MIN`/`MAX` sentinels
(bind.rs:2510). `map::AxisIndex` already expresses "an affine function of the
iteration space", so `BandBound::Static(AxisIndex)` reuses it and the sentinels
have nowhere to live.

## E.2 What in this design is NOT a pipe — the central-claim lint

1. **`Op`, `BoundOp`, `Plan`, `Pass`, `AcceptedRun`, `Band`, `ReduceStage`.**
   Data. These are the values pipes carry. A pipe that is also its own payload is
   not a form the algebra has. **Structurally justified.**
2. **`MetalSession` (arena, uniforms, output pool, pipeline cache, queue).** A
   *resource*, owned by the one pipe that touches the device. It is the I/O edge
   the sans-IO discipline requires to exist somewhere. **Structurally justified** —
   and note it is the only place in the path where I/O lives, which is the
   property that was missing.
3. **`plan()` and `schedule_barriers()`, plain functions.** They *could* be
   `Pipe<In = &[Op], Out = Plan>`. They are not, and this is the one I will name
   as a **judgement call rather than a structural necessity**: they run once per
   plan, not per token, they are synchronous and infallible-modulo-`Result`, and
   wrapping them in a future buys nothing at their call site (I wrote it both
   ways: `plan(program, shapes, &rules)?` vs `Planner.call(program).await?`). By
   the second binary question that identical line means the pipe wrapper is a
   relocation. If a caller ever needs to compose planning with a remote or
   deferred source, they become pipes and nothing else changes.
4. **`resolve_strided_axis`, `KernelKey`, `KernelRoute`.** A function and two POD
   values inside the emitter. Not algebra.

No defects in that list. Item 3 is the one a reviewer should push on.

## E.3 Where information is currently destroyed (and what this design does)

| rich → poor | today | after |
|---|---|---|
| emitter's routing decision → `String` → substring grep | metal.rs:1631-1692 | `KernelRoute` returned with the source (B.4) |
| fused-kind capability → `bool` | bind.rs:2635-2640 | rule composition (A.4) |
| band bounds → `i64::MIN`/`MAX` sentinels | bind.rs:2510 | `BandBound` enum (A.2) |
| single/two-range state → `operands.len() == 8 \| 9` | bind.rs:237, cpu.rs:4847, msl.rs:2030, 2543 | one `Band`; the discriminator has no referent (A.2) |
| operand layout → 8 literal stride tuples | bind.rs:2452-2474 | `Layout.strides` read by the renderer (A.2) |
| conflicting resources → `MTLBarrierScope::Buffers` (all) | metal.rs:874-877, 1105 | `BarrierPoint.resources` (B.3) |
| verifier outcome → an accepted count | (not yet built) | `AcceptedRun { tokens, rejected, cursor }` (B.1) |

## E.4 The number that hurts, stated first

At today's measured 173.6 GB/s and 4,167.6 MB/token, the design's first three
rungs (L1 nr1 fold + draft/verify, L2 Q3_K on FFN, L3 KV f16) reach **10.31
ms/token**: 2.79x today's 28.82, 1.69x llama.cpp's 17.45, and **2.95x short of
the 3.5 ms target**. Only L1 of those three is lossless (gate: text identical);
L2 and L3 are Q-gated. Reaching 3.5 needs L8 — eight levers, five of them lossy —
whose two dominant factors (`A` and `d_u`/`g_u`) are ASSUMED, not measured, and
span 3.55 to 7.03 ms between them. Cards D-1 and D-2 convert those two
assumptions into measurements before any kernel is written, and D-2 carries an
explicit kill that changes the whole ordering.
