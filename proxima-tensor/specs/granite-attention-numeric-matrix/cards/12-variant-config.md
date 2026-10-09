# Card 12: variant-config

**Owner:** GPT-6 Luna
**Dependency:** none
**Commit:** `feat(omega): name attention variant axes`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Add seven independent named variant fields and the default-off feature.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:1630-1674,1792-1831`
- `omega/Cargo.toml:18-39`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/Cargo.toml`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Add metal-attn-variants feature and `AttentionVariant` with `kv_storage`, `mma_precision`, `kv_reuse`, `tile_height`, `query_parallelism`, `simd_topology`, `prefetch`. Defaults preserve the current F32 cache storage, sized-rule MMA precision, legacy K/V reuse, legacy tile height, legacy query schedule, legacy SIMD topology, and prefetch off. Declare explicit values from SPEC.md. Keep validation at the typed form/grid boundary; this card establishes the value types and default-off feature, while each dependent card implements only its own admission rule. Test default values and one value per field; one-row admission behavior belongs to card 16. No inner-loop environment reads.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC12 | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_12_variant_config)'` | filter selects 2 tests; 2 passed; all seven defaults and explicit values |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
