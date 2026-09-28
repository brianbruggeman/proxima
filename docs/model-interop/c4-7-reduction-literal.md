# C4.7 — packed-row reduction-length literal ("nb specialization")

Status: spec + worked example, written before implementation. Owner gates:
no commit, no default change.

## 1. What failed before, measured

The killed experiment (`metal-q4_0-execfix`, patch saved at session scratchpad
`nb/original_execfix.patch`, base `39d8709e1`) baked
`int nb = extents[reduce_dim] / 32;` into `push_q4_0_native_body`. Reproduced
on the current tree (session scratchpad `nb/repro/REPORT.md`), gemma4-E2B
four-prompt gate, one test run per arm:

| arm | features | gate | cache-reuse mismatches | wrong-value mismatches |
|---|---|---|---|---|
| A | `metal` | 4/4 | n/a | n/a |
| B | `metal,metal-q4_0-native` | 4/4, token-identical to A | n/a | n/a |
| C | `metal,metal-q4_0-execfix` | fails 3/4, garbage | 682 / 3025 op resolutions | 385 / 3025 |
| D | C + `_nb{n}` appended to the cache key | fails 3/4, garbage | 0 | 385 / 3025 |

First differing output: generated token index 1 on all four prompts (token 0
comes from prefill, which renders `push_packed_row_multi_row_body`, untouched
by the patch). Example, paris: B `[9079, 236761, ...]`, C `[9079, 87832, ...]`.

Two independent defects:

1. **Key omits the baked value.** `kernel_identity` (`omega/src/identity.rs`)
   carries rank, axes, codecs and Metal extras but no extent. The first op to
   miss a key compiles its own `nb`; every later op with the same key and a
   different K runs that `nb`. Observed collisions: K 6144 / 12288 on one key
   (`..._c0fB1_da2_w6403R`), K 1536 / 256 on another, K 256 / 512 on a third.
2. **The baked value is the wrong number.** `PackedRowBlock::reduce_dim` is
   the innermost reduce axis only (`classify_packed_row_block`,
   `emit_and_classify.rs:2203`). The runtime `u.reduction_total` is the
   product over every folded reduce axis (`pack_reduce_uniforms`,
   `prepare_uniforms_pack.rs:1594`). For 35 ops per forward (rank-5,
   `ax0_4`), extents[reduce_dim] = 256 or 512 while the uniform is 2048 or
   4096. Arm D proves this defect alone breaks the gate.

Mechanism common to both: the value in the source and the value in the key
were derived by two different pieces of code, and the value in the source
was derived by a third expression that did not match the runtime uniform.

## 2. Audit of the specialization / identity machinery

- `kernel_cache_key` (`emit_and_classify.rs:1201`) builds a
  `MetalOnlyExtras` from 17 predicate calls, then `kernel_identity` renders it.
  It never renders source (ROW 92/93: a hit must not pay `emit`).
- `emit_inner` never sees that record. Each render site re-derives its own
  decision: `packed_row_block_shape_token`, `packed_row_block_direct_axis`,
  `packed_row_block_grouped_axes`, `packed_row_block_stride_is_one`,
  `elementwise_addressing_cache_token`, `coord_index32_active`,
  `tiled_gemm_q4_0_active` have no caller inside the renderer; key and source
  agree only because two code paths happen to reach the same answer.
- Launch configuration: `GridSpec::threadgroup_width` is computed by
  `tiled_gemm_threadgroup_width` separately in `emit_inner`
  (`emit_and_classify.rs:108`), `kernel_dispatch_shape` (`:1298`) and the key
  (`:1215`).
- Numerical options: `numeric_policy` enters the key through
  `numeric_policy_token`; `MathMode` is appended by hand at each
  `PIPELINE_CACHE` call site (`resolve_steps`, `encode_op`,
  `Plan::kernel_keys`).
- Expert-source mode: the key is always computed with
  `expert_source_mode = false`; `emit_with_expert_sources_mode` renders with
  `true` and a modified `packed_operands`.

## 3. Design

One record, three consumers.

- `MetalOnlyExtras` gains `reduction_literal: Option<u64>`.
- `metal_specialization(resolved, packed_operands, numeric_policy)` is the one
  function that builds the record (today's body of `kernel_cache_key`).
  `kernel_cache_key` = `kernel_identity(metal_specialization(..))`.
- `emit_inner` calls the same function once and hands `&MetalOnlyExtras` down
  to the packed-row renderers. A renderer may bake a reduction length only by
  reading `reduction_literal`; every packed-row site that today reads
  `u.reduction_total` reads it through one helper that returns either
  `u.reduction_total` or `{n}L`.
- `GridSpec::threadgroup_width` in `emit_inner` and `kernel_dispatch_shape`
  reads `cooperative_width` from the same record.
- `kernel_identity` appends `_rl{n}` when `reduction_literal` is `Some`.
- Value: `reduction_len(resolved, output_axes)` — the same product over
  `reduction_dims` that `pack_reduce_uniforms` writes into `u.reduction_total`.
  Never `extents[reduce_dim]`.
- Admission: compile-time feature `metal-reduction-literal` (default off),
  then runtime A/B `PROXIMA_REDUCTION_LITERAL=1` inside it, then
  `packed_row_block(resolved, quantized)` admitted. Independent of
  `expert_source_mode`, so the key-time and render-time records agree.
  Feature off: the field is always `None`, every emitted byte and every key is
  unchanged.

Why this is not a parallel key: the renderer cannot bake a value the key does
not contain, because the only source of the baked value is the field the key
serializes. The remaining 16 fields are still derived in parallel; section 5's
key-completeness audit is the mechanical check that covers them.

Why the arithmetic order is preserved: the loop bound is the only token that
changes, and it changes from a uniform load to a literal of the same value for
the same op (both are `reduction_len`). Iteration set and order per lane are
unchanged. What source-level reasoning cannot rule out: under `Relaxed`/`Fast`
math the compiler may schedule a known-trip-count loop differently. That is
measured, not assumed (section 5, every output word, every policy).

## 4. Worked example (real gemma4-E2B shapes from `nb/repro/ops_C.tsv`)

Model-fixed vs live values:

- model-fixed: a packed weight's reduction length (1536, 6144, 12288, 2048,
  4096, 256, 512 in this checkpoint), codec, operand layout strides. These are
  constant for the life of the loaded model, so baking them compiles at most
  one pipeline per (existing key, distinct K).
- live runtime: operand bases (which layer's weight), token position, KV
  length / bucket (attention reduces), output base. These stay in the
  `Uniforms` buffer; nothing here bakes them. Different layers with the same
  shape keep sharing one pipeline.

Shape 1 — FFN down projection, narrow vs wide layers. Today both resolve to
`omega_reduce_r3_ax0_2_n2_multiply_add_zero_wide_c0fB1_da2_w6403R`.

| | narrow layer | wide layer |
|---|---|---|
| reduction_len (uniform) | 6144 | 12288 |
| default arm trip bound today | `(int)u.reduction_total / 256` = 24 | = 48 |
| default arm, literal | `(int)6144L / 256` | `(int)12288L / 256` |
| native arm, literal | `(int)6144L / 32` = 192 | `(int)12288L / 32` = 384 |
| key, literal | `..._w6403_rl6144R` | `..._w6403_rl12288R` |
| output | every word equal to the runtime-bound kernel | same |

Insert narrow then wide, or wide then narrow: two distinct keys, two
pipelines, neither order can serve the other's bound.

Shape 2 — multi-axis fold (rank 5, `ax0_4`, node 168 in the repro).

| | value |
|---|---|
| `extents[reduce_dim]` (what the killed patch baked) | 256 |
| `reduction_len` = product over folded reduce axes | 2048 |
| `u.reduction_total` at runtime | 2048 |
| literal | `2048L`, key `..._da4_w6403_rl2048R` |

Boundary: K = 256 is the minimum admitted length (one super-block; only lane
group `ix = 0` iterates). A reduction length that is not a multiple of 256 is
refused by `classify_packed_row_block` (`ExtentNotBlockMultiple`), so the
field is `None` and nothing changes for that op.

## 5. Acceptance criteria (each names a count)

- AC1 feature-off emit: the full existing `omega` msl test suite passes with
  the same executed test count as before the change (count recorded).
- AC2 insertion order: for each of the three real collision pairs, both
  orders through one cache produce 2 distinct keys, 2 pipeline misses, and
  every output word equal to the runtime-bound kernel (0 differing words),
  repeated 3 hits each.
- AC3 policies: AC2 repeated under `bit_exact`, `llama_relaxed`, `fast`;
  differing-word count reported per policy (expected 0; a nonzero count is a
  finding, not a pass).
- AC4 independent oracle: every AC2 output also compared against proxima's
  CPU evaluator on real blk weights; max_abs reported.
- AC5 boundary shapes: K = 256, 2048 as 256x8 multi-axis, 12288; a K = 288
  op yields `reduction_literal = None` and an unchanged key.
- AC6 key completeness on the real graph: with the feature on and off, every
  pipeline-cache hit during the gemma4 gate re-renders its source and compares
  it to the source that created the pipeline; audited-hit count > 0,
  mismatches = 0.
- AC7 gemma4 gate: 4/4 with the feature compiled in, A/B off and on;
  generated token ids identical between A/B arms on all four prompts.

## 5b. Results (session scratchpad `nb/`, raw logs kept)

Correctness (isolated copy `trees/impl`, real gemma4-E2B weights):

- AC2-AC5: 3 real collision pairs x both insertion orders: 2 distinct keys
  each, 6 audited hits each, 0 mismatches; 0 differing words in 18 decode
  cells (6 shapes x 3 NumericPolicy presets) and 9 prefill-shaped cells
  (token_total 8, generic / Q4_0-hoist / unroll arms); CPU-oracle max_abs
  1.4e-6 .. 2.0e-5. K = 288 yields `None` and an unchanged key.
- AC6 key audit, real 16-token decode: audited 3261 / 3255 / 3258 (arms
  unset / `=1` / `=decode`), mismatched 0 in each.
- AC7 gate: 4/4 in every arm; generated token-id arrays identical across
  plain `metal`, feature-compiled unset, `=1`.
- omega nextest `--no-fail-fast`: 371/371 (`metal`), 373/373
  (`metal,metal-reduction-literal`).

What the specialization removes (`nb/fix/AIR.md`, `nb/prefill/RESULTS.md`
section 2; `xcrun metal -O2 -S -emit-llvm -fmetal-math-mode=relaxed`):

- decode single-token body, K 1536 / 6144 / 12288: one uniform load, one
  `sdiv`, one branch (the `ix < super_blocks` guard, provably true once the
  bound is constant). Loop count unchanged (11 `llvm.loop` both), no unroll.
  The three literal variants differ by one immediate.
- prefill multi-row body: one uniform load; the per-lane `k` loop's entry
  guard becomes unconditional and its exit test is rewritten from
  signed-post-increment-vs-uniform to unsigned-pre-increment-vs-immediate
  (`icmp ult i64 %88, 1504`). Block and branch counts unchanged. What this
  does in the GPU ISA is not observable from the LLVM stage — residual.

Timing (quiet box each run, Ollama stopped, arms interleaved, same binary):

- isolated decode probe, 6 shapes, >= 2 GB per sample, 12 samples, 2
  process runs: paired literal - runtime medians between -175 and +80 ns per
  dispatch, sign flips between runs on 4 of 6 shapes, every range spans 0.
  No separable effect.
- isolated prefill probe (`reduction_literal_prefill_speed_probe`): literal
  slower on 48/48 pairs: attn_k K 1536 +1.02% (t 26), +6.76% (t 510);
  ffn_down K 12288 +2.24% (t 26), +4.91% (t 510). One pipeline per arm in
  the timed loop, so pipeline switching is not the cause.
- whole generation, prompt4 (26 tokens, 105 generated), 8 A/B/C triples,
  warm: prefill A 3203.0 / B 3292.5 (+85..+100 ms, 0/8 faster) / C 3203.0
  (-3..+4 ms); wall A 5321.4 / B 5418.1 / C 5329.9; ordinary decode
  18.819 / 18.802 / 18.832 ms per token (B 4/8, C 2/8 faster — no signal);
  first decode step 51-52 vs 52-54 ms (A vs B, first run set).
- whole generation, prompt5 (510 tokens, 3 generated), 3 triples: B +2.8
  .. +3.4 s on a 60.8 s prefill (0/6); C equal to A in triple 2 (+4, -1 ms),
  outliers in triples 1 and 3 (+4175 cold, +810 / +1271 warm) that the same
  binary did not reproduce in triple 2 — not established.
- text_hash identical in every run of every arm.

Measured summary: `=1` slows prefill in every pair on both prompts;
`=decode` matches the runtime arm on prompt4 and on one of three prompt5
triples; no arm shows a whole-generation or decode reduction outside noise.
The parts with no timing cost are the key/source structure (one record;
the renderer bakes only what the key serializes) and the key audit.

## 6. Operation elimination in one real specialized decode graph

Record: session scratchpad `nb/graph/` — `decode_ops.tsv` (1661 resolved
BoundOps of the decode plan's first call, i.e. generated token 2,
`--features metal,instrument`), `known_scalars.tsv` (702 rows),
`census.md` (every count with its command), `graph_dump.diff` (throwaway
instrumentation). Kinds: reduce 903, elementwise 488, constant 262, iota 8.

### Known scalars and their consumers

| scalar | value (bits) | count | consumer |
|---|---|---|---|
| GELU-tanh `0.5`, `sqrt(2/pi)`, `0.044715`, `1` | `0x3f000000`, `0x3f4c422a`, `0x3d372713`, `0x3f800000` | 70 each (+3 of `1`) | epilogue operands of the FFN gate/up reduce (e.g. position 828 steps 1-9) and the per-layer input-gate reduce (879) |
| `1/sqrt(2)` | `0x3f3504f3` | 35 | per-layer GELU epilogue |
| RMSNorm `1/N` | `1/1536` `0x3a2aaaab`, `1/256` `0x3b800000`, `1/512` `0x3b000000` | 105 / 52 / 13 element-body uses | step 0 `Multiply(sum_of_squares, 1/N)` of every norm elementwise |
| embedding scale, softcap, bucket bound | `39.191837`, `30`, `16`, `511` | 1-2 each | embedding scale, logit softcap, mask compare |
| reduction length K (this component) | 256 ... 12288 | 275 packed-row matvecs | trip bound of every packed-row body |
| layout scalars: extents, strides, base of `Identity` operands | see below | 106 | `Identity` copies |

### Candidate rewrites, with what each removes

1. Constant leaves hoisted out of the per-step stream: **already
   implemented** — `Plan::mark_plan_time_constants_resident`
   (`omega/src/metal/device_buffers_arena_plan.rs:1563`) is called for decode
   (`proxima-model-interop/src/generate/decode.rs:304,722,860`) and
   `resident_skip` (`placements_execute_named.rs:1126`) skips the 262
   dispatches on warm calls. Removes nothing further.
2. Inline scalar constants as literals in consuming bodies: removes 262
   four-byte buffers and ~386 operand bindings, no dispatch (the leaves are
   already resident). Changes consumer source text under Relaxed/Fast math
   (contraction with a literal may differ from a buffer read). Not selected:
   no dispatch or bandwidth removed.
3. **Contiguous `Identity` elision (selected).** All 106 elementwise ops
   with body `0:Identity(operand0)` read their operand in its own
   row-major element order (strides equal the contiguous strides of the
   extents, ignoring unit axes): 72 at base 0 (rope'd K/V reshaped
   `[1,8,D] -> [1,1,8,D]`, e.g. positions 805/806), 34 contiguous slices of
   node 17 at base `256*L` (per-layer embedding input, e.g. 839). Every one
   has 1-3 in-graph consumers, all reduce folds. Rewrite: when the body is
   exactly `Identity(operand0)`, dtypes match, there is no gather, and the
   operand layout is contiguous in output element order, replace each
   consumer's reference to the copy with the source node composed with the
   copy's layout (base offset + strides), extend the source's arena lifetime
   to the copy's last consumer, and drop the copy. If the copy is also a
   plan output or caller-placed (KV cache write-back), the placement moves to
   the producer or the copy stays — not verified yet from the dump.
   Removes up to 106 dispatches and 106 intermediate buffers per decode step
   (1-4 KB each).
   Byte-preservation argument: an `Identity` of equal dtype moves each 32-bit
   word without arithmetic, so the consumer reading the source through the
   composed layout reads the same words at the same logical coordinates, in
   the same order. The consumer's arithmetic and fold order are unchanged;
   only layout-derived specializations (base uniform, stride-is-one tokens)
   can change its source, and those select addressing, not arithmetic. This is
   proven only by the same every-word comparison as AC2, not by this
   argument.
   What it is not: a time claim. The 106 copies are small; their cost in the
   chunked serial stream has not been measured and is not inferred from the
   count.
