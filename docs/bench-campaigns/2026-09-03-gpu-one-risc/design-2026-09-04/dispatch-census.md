# Dispatch census, default Metal decode, main af918bb (read-only reconciliation, exact to 616)

Per layer (19 bound ops; x32 = 608; + tail 8 = 616): source
`proxima-tensor/src/spec.rs:3517-3921` (68 raw RISC ops/layer) folded by `BoundOpBuilder`.

| # | node | kind | fuses | out shape | why separate |
|---|---|---|---|---|---|
| 1 | sum_squares (attn norm) | reduce-cooperative | Reduce(Add) + x*x | 1x1 | reduce boundary (`spec.rs:12700` reduces never fuse) |
| 2 | normed | elementwise | 6-op chain (mean, +eps, sqrt, recip, *, *gamma) | 1x4096 | consumes a materialized reduce; broadcast s->sd |
| 3-5 | q, k_new, v_new | reduce-packed-row-blocked | Reduce(Add)+Multiply | 1x32x128 / 1x8x128 x2 | matvec |
| 6-9 | rotated_{q,k}_{even,odd} | elementwise | 3-op RoPE each | 1x32x64 / 1x8x64 | `is_identity_projection` fails on the 2i/2i+1 stride and the GQA group map `h=group*u+g` (`bind.rs:1153-1164`) |
| 10 | attended | cached-attention | 10 bound / 17 raw ops | 1x8x4x128 | matcher `bind.rs:2295-2517` |
| 11 | attn_out | reduce-packed-row-blocked | Reduce+Multiply | 1x4096 | matvec |
| 12 | residual1 | elementwise | Add(attn_out, x) | 1x4096 | `StillLive`: x read twice (`bind.rs:701,778`) |
| 13 | sum_squares (ffn norm) | reduce-cooperative | | 1x1 | as 1 |
| 14 | normed2 | elementwise | 6-op | 1x4096 | as 2 |
| 15-16 | gate, up | reduce-packed-row-blocked | | 1x14336 x2 | matvec |
| 17 | ffn_hidden (SwiGLU) | elementwise | 6-op (neg, exp, +1, recip, *, *up) | 1x14336 | `quarantine_broadcast_operands`: child extent 14336 < reduce extent 14336x4096 (`bind.rs:935-973`) |
| 18 | ffn_out | reduce-packed-row-blocked | | 1x4096 | matvec |
| 19 | x_next | elementwise | Add(ffn_out, residual1) | 1x4096 | StillLive |

Tail: 2 iota, 1 mask elementwise, final norm (1 coop + 1 elementwise), lm_head matvec, 2 constants.
Totals: reduce-cooperative 65, packed-row 225, cached-attention 32, elementwise 290, constant 2, iota 2.

STEADY-STATE WALL (ROW 286 run, score-logs/default-r3.log lines 15-24, plan-HIT steps 3..7 —
THE token-time decomposition of record; the 33.0 ms "steps 1..7" mean includes two plan-miss
steps that pay bind+compile and must not be used): wall 28.82 ms = gpu_exec 27.52 + host 1.30
(emit 0.418 + op_setup 0.158 + encode_dispatch 0.209 + pipeline_lookup 0.013 + readback 0.005 +
greedy 0.044 + build_position_inputs 0.002 + unexplained 0.453). There is NO 5.96 ms host
residual. Steady-state ratio to llama.cpp 17.45 = 1.65x. Gap 11.4 ms is GPU-side.

IN-BUFFER (ROW 287, the production single command buffer, quiet box, 3 rounds): ALL 616 ops 27.04 ms (CoV 1.0%); MATVEC 225 ops 23.24 ms (0.7%) =
179 GB/s over 4.169 GB; NOT-MATVEC 391 ops 4.05 ms (1.5%); ATTN 32 ops 3.0-3.6 (CoV 9.9%);
COOP 65 ops 1.3-1.8 (18%); ELEM 290 ops 1.76 ms (1.1%, ~6 µs each). Residual ALL − (MATVEC +
NOT-MATVEC) = −0.25 ms (concurrent overlap). Whole token wall 33.0 ms (ROW 286).

Serialized per-op profile (ROW 281/282, one command buffer per op, biased high for small ops —
SUPERSEDED for time decomposition, kept for op counts only):
cached-attention 32 ops 3.515 ms; elementwise 290 ops 2.978 ms (~10 µs each);
reduce-cooperative 65 ops 0.851 ms; packed-row 225 ops 24.991 ms.

llama.cpp b25346221 (`llm_build_llama`, flash-attn off, ggml-metal has ZERO fusion at this
checkout): 23 ggml ops/layer = 740 dispatches/token. We already dispatch fewer (616).

## Fusion levers that are GENERIC RULES over the algebra (not model matchers)
- EPILOGUE fusion: an elementwise consumer of a Reduce whose iteration space equals the
  reduce's output space, with the reduce output having no other consumer, becomes the reduce's
  epilogue. Removes per layer: residual1 (#12), x_next (#19), ffn_hidden (#17 as the epilogue
  of `up` reading materialized `gate`) = 3/layer = 96/token. `BoundOpBuilder` today only fuses
  elementwise INTO reduce operands (prologue), never out of them.
- PROLOGUE-with-broadcast: `normed` (#2/#14) into the q/k/v (and gate/up) matvec prologue —
  blocked by `quarantine_broadcast_operands`; relaxing it needs a cost bound (recompute is a
  per-row 3-op chain on an already-loaded activation vs. a 16 KB materialization + dispatch).
  Would remove 2/layer = 64/token if all consumers absorb it.
- RoPE (#6-9): blocked by the stride/group-map identity rule and by `CachedLayerRoots` needing
  even/odd as separate outputs (spec.rs:2333 consumed at 15+ sites). Fusing even+odd into one
  dispatch = 2/layer; into the q/k matvec epilogue = 4/layer (needs a non-identity epilogue map,
  i.e. an IndexMap-aware epilogue — the same extension as above generalized).
- rmsnorm sumsq (#1/#13): a reduce whose consumer needs a broadcast back — a two-phase
  threadgroup op (reduce then normalize in one dispatch) = the "reduce with broadcast epilogue"
  form; 2/layer = 64/token.
Ceiling if all four land: 19 -> 19 - 3 - 2 - 4 - 2 = 8/layer (7 matvec + 1 attention) = 264/token.
