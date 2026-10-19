# Ouro dense projection math: Omega and llama.cpp

This note records the mathematical comparison between the shared Ouro
projection and its Metal lowerings. Source is checked in the current Proxima
worktree and against the pinned llama.cpp source revision named below. Saved
serving timings are kept separate from source and lowering evidence.

## Concrete Ouro projection

The checkpoint config gives hidden width `K = 2048`, feed-forward width
`N = 5632`, 24 physical full-attention layers, and four recurrent visits.
For a 256-token prefill chunk, the captured shared-program projection has
activation node 111 in F32, weight node 112 in BF16, and bound extents
`[256, 2048, 5632]` (recorded in `CALLER-C7-AUDIT.md`, lines 103–109).
Its output-axis interpretation is:

```text
X ∈ F32^(B×K)       B = 256, K = 2048
W ∈ BF16^(N×K)      N = 5632
Y ∈ F32^(B×N)

Y[b,n] = Σ(k=0..K-1) X[b,k] · decode_bf16(W[n,k])
```

The result has `256 × 5632 = 1,441,792` elements. The reduction performs
`1,441,792 × 2,048 = 2,952,790,016` multiply-accumulates, or
`5,905,580,032` scalar FLOPs when one multiply and one add count separately.
These are shape-derived operation counts, not measured device throughput.

For a complete 1,018-token prompt, the same single projection represents
`1,018 × 5,632 × 2,048 = 11,741,954,048` multiply-accumulates. With 256-token
microbatches, this is four executions of the projection (three full chunks
and one 250-token tail), subject to the runtime's actual chunk formation.
The checkpoint has 24 physical layers and four recurrent visits, so a single
projection site executed once per logical layer visit accounts for
`1,127,227,588,608` multiply-accumulates over the full prompt. Other projection
sites and attention work are additional; this count is not the whole-model
FLOP total.

## Omega lowering: original behavior and configured BF16 path

The original classifier rejected F16/BF16 dense operands when
`omega_dense_float_storage` was absent. That gate has been removed from dense
float operand admission. `RuntimeConfig::dense_mma_input_layout` now has
`auto`, `f32`, and `bf16` choices. `auto` selects BF16 fragments per bound
operation when it reads a BF16 operand, otherwise F32. A BF16 projection does
not narrow unrelated F32 dense operations in the same plan. Explicit `f32`
with a BF16 block is refused during plan construction so a BF16 weight cannot
silently widen into F32 MMA. The selected layout flows through `resolve_step`,
MSL emission, and pipeline identity.

Rounding an F32 activation to BF16 is a separate numerical rewrite from FMA
contraction and reduction reassociation. `NumericRewrite::Bf16InputRounding`
requires its own `NumericPolicy::bf16_input_rounding` permission. The configured
planner derives that permission from `auto`/`bf16` fragment selection and
passes it through the common `admit` check; direct emitters without that
permission decline the BF16 dense tile. The permission is part of numeric
kernel identity.

For BF16 selection, the dense tiled body stages `bfloat` values,
`simdgroup_bfloat8x8` fragments, and an F32 accumulator. BF16 bytes are
reassembled from the `uchar*` buffer ABI directly into a BF16 value. F32
activations are rounded to BF16 at the staging boundary. This implements the
following operation:

```text
A_bf16[b,k] = round_to_bf16(X_f32[b,k])
W_bf16[n,k] = decode_bf16(stored_weight_bytes[n,k])
Y_f32[b,n]  = f32_accumulate Σ(k=0..K-1) A_bf16[b,k] · W_bf16[n,k]
```

Worked boundary example: `X=1.00390625` is exactly halfway between BF16 `1.0`
and the next BF16 value `1.0078125`; round-to-nearest-even selects `1.0`.
With stored BF16 `W=2.0`, the device multiplies `1.0 × 2.0` and the F32
accumulator receives `2.0`. The old F32-fragment path instead multiplies
`1.00390625 × 2.0 = 2.0078125`. This difference is expected and must be
covered by an output bound that includes activation rounding.

The fragment format is included in dense kernel identity, so a pipeline built
for F32 fragments cannot satisfy a BF16 plan. The old F32-fragment path remains
available for plans whose input layout resolves to F32 and whose operands do
not violate the BF16 no-widen contract.

The BF16 device primitive passed a standalone MSL compile and M1 Max dispatch
before this source change. That established native `bfloat` fragment support,
not correctness or performance of Omega's generated kernel. The configured
Omega device integration fixture is the first check of this lowering.

With the configured tile sizes `M=64`, `N=32`, `K=32`, a 256 by 5632 output
requires `ceil(5632/64) × ceil(256/32) = 704` output threadgroups per K
iteration. Each group advances through `2048/32 = 64` K tiles. This describes
the configured tiling geometry; it does not establish that the tiled kernel is
faster on the measured device.

## llama.cpp lowering and numerical difference

At pinned revision `1550360e089185632c2036c908411b05fafa1119`,
`ggml/src/ggml-metal/kernels/mul_mm.metal` defines
`kernel_mul_mm_bf16_f32` with BF16 matrix fragments and an F32 destination
(lines 66–78 and 851–857). The BF16 specialization stages activations through
the `S1=bfloat` fragment type (lines 255–280). Its corresponding equation is

```text
Y[b,n] = F32-accumulate Σ(k=0..K-1)
         round_bf16(X[b,k]) · W[n,k]
```

That differs numerically from Omega's existing dense tile, which preserves
F32 activation values in its F32 fragment, and from the fallback, whose
accessor widens the stored BF16 weight value to F32 before multiplication.
Llama's kernel source is a candidate lowering, not proof that the saved Llama
TTFT run selected it. The pinned source sets `GGML_METAL_HAS_BF16` from the
runtime device's `has_bfloat` capability when it creates the Metal library
(`ggml-metal-device.m:446-460`). `has_bfloat` depends on reported Metal GPU
families and can be disabled by `GGML_METAL_BF16_DISABLE`; when tensor API and
BF16 are both enabled, Llama compiles a BF16 tensor-matmul probe and disables
BF16 if that probe fails (`ggml-metal-device.m:1133-1147,1214-1260`). The
saved TTFT context has no device capability fields, disable-environment values,
binary hash, or selected pipeline, so neither the fragment specialization nor
the tensor path is established as the kernel used for that timing.

## What the math settles, and what remains empirical

The symbolic `BoundOp` expresses the same contraction as the Llama matrix
operation: one feature row of `W` dotted with each token row of `X`, producing
`Y[b,n]`. The concrete gap in the earlier Omega lowering was fragment/storage
treatment: a default-off admission gate, followed when enabled by F32 fragments
rather than the BF16 fragment specialization present in llama.cpp source. The
gate has now been removed and the generic tiled lowering emits native BF16
fragments for matching bound operands. The algebra itself does not require a
new Ouro-specific operation.

This derivation does not assign the end-to-end TTFT gap to the BF16 lowering.
The Metal integration test
`programmable_tiled_bf16_weight_matches_f32_within_derived_bound` compiled and
dispatched Omega's emitted BF16 kernel and passed its derived numerical bound.
That establishes this fixture's numerical condition, not an E2E output or
performance result. Paired TTFT runs should compare the saved baseline with
the generic BF16-fragment path while retaining a generic F32-fragment control.
The existing saved Llama result is
not re-established here, and its selected pipeline is still unknown; no
end-to-end TTFT gain is claimed from the math or kernel-source change.

## October 18 configured-placement fast fail

The first release probe exposed a runtime integration refusal before request
timing: configured Omega plans rejected external input placements. The
configured planner now shares its input resolution, numeric policy and layout
selection with the placed-input path. Plans that request resource ceilings
still refuse caller-owned placements because those bytes are not included in
the current physical census. Each placed input now checks its planned shape,
dtype width, offset and buffer capacity before encoding.

The release fast probe used `/private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf`
with observed GGUF SHA-256
`d3cdf276005d768c5def9cfb6679eb5600a424c5cd23b45157e65204708ae478`,
configured `relaxed-metal.toml`, Metal, `gpu-profile`, one generated token and
the frozen `ball24` case. It completed one warmup and five measured requests.
The five callback TTFT observations were 2326.682, 2325.099, 2324.346,
2331.868 and 2323.741 ms; mean 2326.347 ms, sample CoV 0.141%. The saved
per-request records are `/tmp/ouro-bf16-probe-after/cached.jsonl`. This prompt
has 24 tokens, below `DENSE_TILED_MIN_TOKENS=160`, so it is a placement-path
smoke observation and does not exercise the long-prompt tiled path.

The frozen `repeated-ball256` release attempt used the same model/config and
one-token budget. Its warmup callback recorded TTFT 15593.180 ms and wall time
15698.681 ms. The five-sample run was interrupted before a measured sample to
keep the cycle below one minute; `/tmp/ouro-bf16-probe-256/cached.jsonl`
contains setup, diagnostic and warmup records only. This value is not a
measured-cohort statistic. Captured MSL includes both BF16 and F32 matrix
fragment kernels, including `omega_reduce_r4_o3_n2_multiply_add_zero` and
`omega_reduce_r5_o4_n2_multiply_add_zero`; the source files are under
`/tmp/ouro-bf16-probe-sources/`. A per-request dispatch-to-source manifest is
still needed to attribute the TTFT to each emitted specialization.

The new `--cached-probe --samples 1` mode skips the separate logits diagnostic
and warmup, records one production callback observation, and avoids turning a
fast-fail check into a five-request cohort. Its single sample has no variance
estimate; use the existing five-sample mode for cohort summaries. Neither run
provides an equivalent pre-change TTFT: the previous configured placement path
refused before execution. These observations therefore do not establish a
before/after speedup or explain the remaining long-prompt latency.
