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

## Omega lowering recorded in source and capture

The current classifier rejects F16/BF16 dense operands when
`omega_dense_float_storage` is absent (`omega/src/msl/emit_and_classify.rs`,
lines 4407–4412). The checked-in runtime sizing config sets that build option
to zero (`omega/omega-runtime.toml`, lines 680–684). The rejected operation
continues through the cooperative-reduction renderer.

The recorded gate-off dispatch for this shape used the generic reduce entry
`omega_reduce_r3_o2_n2_multiply_add_zero`. The gate-on dispatch rendered a
simdgroup matrix body, but still used `float` staging, 4-byte tile elements,
and `simdgroup_float8x8` (`omega/src/msl/tiled_gemm_cooperative_scan.rs`,
lines 1458–1467). Its BF16 accessor calls `bf16_element`, which produces
`float` (`omega/src/msl/signature_tokens_prelude.rs`, lines 1111–1116).
Therefore the existing gate has two distinct effects to measure:

1. Off: BF16 dense GEMM is refused and falls through to cooperative reduction.
2. On: dense GEMM is tiled, but stored BF16 weights are decoded and staged as
   F32; F32 activations remain F32.

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
TTFT run selected it. The saved run context identifies model, prompt, revision,
and ubatch, but has no binary hash or selected pipeline. The available local
llama build cache records `GGML_METAL_USE_BF16=OFF` at
`/Users/brianbruggeman/repos/others/llama.cpp/CMakeCache.txt:501-502`, but is
not tied to that run record. Neither that flag nor the source specialization
can identify the pipeline used by the saved timing.

## What the math settles, and what remains empirical

The symbolic `BoundOp` expresses the same contraction as the Llama matrix
operation: one feature row of `W` dotted with each token row of `X`, producing
`Y[b,n]`. The concrete gap visible in current Omega source is lowering policy
and fragment/storage treatment: a default-off admission gate, followed when
enabled by F32 fragments rather than the BF16 fragment specialization present
in llama.cpp source. The algebra itself does not require a new Ouro-specific
operation.

This derivation does not assign the end-to-end TTFT gap to either difference.
The smallest causal comparison is three cells over one current-main captured
BoundOp and identical stored operands:

1. Current cooperative-reduction fallback.
2. Existing dense tile with F32 fragments.
3. Dense tile with BF16 weight and activation fragments, F32 accumulation.

Compare literal output coordinates against an F64 reference computed from the
actual stored BF16 weight bytes and the exact activation bytes. Cell 1 versus
2 isolates tiled lowering. Cell 2 versus 3 measures the combined BF16-fragment
change, including F32 activation rounding. Keep TTFT measurements separate
from the replay used for output comparison, and capture the actual selected
Llama pipeline before attributing its timing to the source specialization.
