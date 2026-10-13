# Work packet: float-storage codecs in dense tiled GEMM

Status: executable implementation brief; source edits and validation have not started.

## Owned slice

Own `omega/src/msl/emit_and_classify.rs`,
`omega/src/msl/tiled_gemm_cooperative_scan.rs`,
`omega/src/msl/elementwise_reduce_core.rs`,
`omega/src/msl/signature_tokens_prelude.rs`, `omega/src/identity.rs`, and
existing MSL unit tests in `omega/src/msl/tests.rs`. Changes outside the
classifier/codec/policy call chain in these files are not authorized by this
packet. The root integration supplies the caller's
`NumericPolicy`, runtime bindings, and config selection. Do not add a second
kernel model or infer permissions from codec choice.

## Observed source contracts

- `proxima-tensor/src/numeric.rs` defines
  `bit_exact`, `llama_relaxed`, the `FmaContraction` and `TreeReduce` rewrites,
  and `admit`. `llama_relaxed` grants those two rewrites; bit-exact grants
  neither. Admission must call `admit`, not duplicate its permission logic.
- `omega/src/msl/emit_and_classify.rs:1654-1695` shows structural specialized
  reductions (including dense GEMM and packed-row paths) bypassing the generic
  TreeReduce policy check. This packet changes dense tiled GEMM admission only;
  do not broaden the edit to unrelated packed Q4 lowering. Since the dense MMA
  path contracts and changes the accumulation tree for every input storage
  type, require both permissions for dense tiled GEMM, including its existing
  F32/F32 case. A denied policy must decline this specialization and use the
  existing policy-compatible fallback.
- `omega/src/msl/emit_and_classify.rs:2741-2772` classifies packed Q4 GEMM and
  must remain separate. `:3472-3628` defines the dense rejection/block types
  and currently rejects all `Some(codec)` operands. Extend that classifier to
  allow `None`, `Some(Float16)`, and `Some(BFloat16)` independently for each
  operand, including mixed pairs; reject all other codecs. Preserve its shape,
  layout, epilogue, and axis checks.
- `proxima-primitives/src/codec.rs:15-43` has Float16 and BFloat16 but
  no FP8 codec. Do not add speculative FP8 behavior.
- `omega/src/msl/signature_tokens_prelude.rs:605-622,908-956` binds Float16 as
  `half` and reads BFloat16 through `bf16_element`. In
  `omega/src/msl/tiled_gemm_cooperative_scan.rs:1073-1317`, dense GEMM already
  uses float scratch tiles and float fragments; only its two `operand_read`
  calls currently pass `None`. Pass the codecs selected into the classified
  block at those calls. Keep existing zero-fill K-tail masks.
- `omega/src/msl/emit_and_classify.rs:1558-1565` documents the shared grid
  contract: one grid result feeds rendering, launch dispatch shape, and cache
  identity. The dense block is consulted from classification, policy
  dispatch, grid/depth/width helpers, and rendering (`:1404-1418,1686-1695,
  1826+,2979-3041,3092,3141,3267,3766-3825`). Thread the caller's
  `NumericPolicy` through these call paths. Each must use the same policy-aware
  dense classifier; do not recompute codec or admission decisions in a renderer
  or grid helper.
- `omega/src/identity.rs:237-248,790-796,821-824` includes dense axes,
  operand codec tokens, and the numeric-policy token in identity. Preserve
  these paths: include dense axes only when the same classifier admits the
  specialization; existing per-operand codec tokens distinguish F16/BF16/mixed
  inputs; the policy token keeps different permission sets distinct. A cache
  token is identity, not permission to lower.
- Existing MSL fixtures include `real_shaped_tiled_gemm_op` at
  `omega/src/msl/tests.rs:2565-2615`, dense admission tests around `:1720-1768`,
  and the dense batched shape helper at `:1621-1678`. Reuse the existing
  fixtures and minimum-token constants rather than inventing threshold values.
- The GPU fixture comparator is `omega/tests/metal_parity.rs:32-61`: it rejects
  empty or length-mismatched vectors and compares maximum absolute error with a
  tolerance. Keep this slice's numerical fixture CPU-only; mirror that
  comparator's checks for the positive oracle and corrupt-cell control. This is
  not evidence of GPU parity. llama.cpp is a performance oracle only, never an
  accuracy oracle.

## Required implementation

1. Use the existing build-time configuration mechanism for a default-off
   compile gate: add `[experimental] dense_float_storage = 0` to
   `omega/omega-runtime.toml`. In `omega/build.rs`, resolve it through
   `SizingSource` (including its existing environment override and rerun
   mechanism), require 0 or 1, declare `rustc-check-cfg` for
   `omega_dense_float_storage`, and emit that cfg only for 1. The override is
   `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE`. Root's config worker owns those
   two sizing files. The tiled worker uses `cfg!(omega_dense_float_storage)`
   and tests use `#[cfg(omega_dense_float_storage)]`. No Cargo manifest edit
   or exception is needed. Uncompressed dense GEMM retains its existing gate.
2. Extend `DenseBatchedGemmBlock` to retain the exact two operand codecs chosen
   by classification. Extend the classifier to accept only the supported
   float-storage set above when the new feature is enabled. With the feature
   disabled, any `Some(codec)` remains rejected. Q4K/Q4_0 and every other
   unsupported packed codec remain rejected by dense GEMM and continue through
   their existing routes.
3. Make the dense classifier take the caller-provided `NumericPolicy`. Require
   `admit(policy, NumericRewrite::FmaContraction)` and
   `admit(policy, NumericRewrite::TreeReduce)` before returning an admitted
   block. Represent a denied rewrite in the dense rejection result, so tests
   can distinguish it from unsupported codec/shape rejection. Never widen or
   replace the policy. A policy granting exactly one rewrite still declines
   dense tiled GEMM.
4. Pass the classified operand codecs to the existing dense renderer's
   `operand_read` calls. Keep decode-to-float in the existing float tiles;
   introduce no low-precision scratch tile, bespoke codec decoder, or new
   kernel model.
5. Thread the same `NumericPolicy` through every dense classification use,
   including `reduce_is_cooperative_dispatch`, `metal_specialization`, dense
   grid depth/width/threadgroup helpers, direct-store helpers, and the renderer.
   The policy-aware classifier must be the sole source for emit choice, grid,
   and dense cache specialization. A bit-exact request must not get a tiled
   dense grid while rendering the serial/policy-compatible fallback.

## Hand-worked CPU oracle

Use the small uncompressed matmul fixture independently of Metal code. The
weight is `[feature,k] = [[1,2],[3,4]]`; activation is `[token,k] =
[[5,6],[7,8]]`; output order is `[token,feature]`. Scalar products give
`out[0,0]=1*5+2*6=17`, `out[0,1]=3*5+4*6=39`,
`out[1,0]=1*7+2*8=23`, and `out[1,1]=3*7+4*8=53`; flattened output is
`[17,39,23,53]`. Compare the CPU result with that literal oracle using the
same nonempty/length/max-absolute-error comparator shape as the GPU fixtures.
Then corrupt one cell (for example, change 23 to 24) and assert that the same
comparator rejects it. Do not consume emitted MSL or `operand_read` to produce
the CPU expected values.

## New test inventory and acceptance commands

Put explicit cases in `omega/src/msl/tests.rs` under exactly these names. Eight
run with the new feature on, and one default-off control runs without it.

1. `programmable_tiled_bf16_classifier_accepts_supported_operand_positions`:
   BF16 weight, BF16 activation, and mixed BF16/F16 codec selections produce
   classified blocks with the same codecs retained in the block.
2. `programmable_tiled_f16_renderer_uses_existing_float_tiles`:
   admitted F16 operands render reads through `operand_read`; assert no new
   half scratch tile is emitted.
3. `programmable_tiled_unsupported_codec_is_rejected_by_dense_classifier`:
   Q4K and at least one other unsupported packed codec decline dense
   classification; the packed Q4 classifier does not admit BF16/F16.
4. `programmable_tiled_noncontiguous_reduction_is_rejected`:
   use a two-axis reduction fixture and a deliberately invalid stride relation;
   assert the existing noncontiguous-reduction rejection.
5. `programmable_tiled_ragged_k_and_feature_tail_keep_bounds_masks`:
   use admitted token count, feature extent 5, and K extent
   `TILED_GEMM_BLOCK_K + 3`; assert classification and the rendered K and
   feature bounds masks.
6. `programmable_tiled_numeric_policy_requires_both_rewrites`:
   assert bit-exact, FMA-only, and TreeReduce-only policies decline dense
   classification; a policy granting both admits it. Assert the denied cases
   do not emit dense MMA.
7. `programmable_tiled_grid_emit_and_cache_follow_one_admission`:
   for a policy granting both, assert dense classification, emitted dense
   source, dense grid shape, dense axes identity, and codec identity agree; for
   bit-exact, assert fallback source/grid and no dense axes specialization.
8. `programmable_tiled_cpu_oracle_and_corrupt_cell_control`:
   execute the hand-worked CPU fixture, compare against `[17,39,23,53]`, then
   assert the same comparator rejects the one-cell corruption.
9. `programmable_tiled_float_storage_feature_off_declines` (compiled only
   without `omega_dense_float_storage`): assert BF16/F16 decline while
   uncompressed dense classification remains available under a policy granting
   both rewrites.

The enabled build must print exactly `8 passed`; the disabled build exactly
`1 passed`:

```text
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo nextest run -p proxima-omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 cargo nextest run -p proxima-omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture
```

These are count-bearing acceptance commands, not claims already run. Do not add
tests elsewhere or rename them without updating the exact expected counts.

## Root integration dependency and limits

The root integration must pass a runtime-selected `NumericPolicy` into this
slice. To request the dense MMA route it must explicitly provide both
`FmaContraction` and `TreeReduce` permissions; this packet adds no default,
automatic relaxation, or policy override. The root owns config bindings and
backend-wide resource/grid selection. This slice only guarantees that Metal
dense classification, source selection, grid derivation, and dense identity
agree for the policy and codecs they receive.

No GPU accuracy claim follows from the CPU fixture. If a later integration
adds an actual Metal parity case, use the existing GPU comparator and trace the
payloads. llama.cpp may compare execution performance only. No build, test, or
benchmark has been run for this packet.
