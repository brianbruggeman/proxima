# Card 17: query-parallelism

**Owner:** GPT-6 Luna
**Dependency:** 12,14,15,16
**Commit:** `feat(omega): parallelize attention query rows`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Select concurrent simdgroup ownership of query rows independently of tile height and lane topology.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `omega/src/msl/signature_tokens_prelude.rs:47-75,1957-2005`
- `omega/src/msl/cached_attention_decode_split.rs:130-151`
- `proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out:12` for captured extents `[1000,8,2,64]`

## Edit

- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/cached_attention_row_tiled.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Map `query_parallelism=simdgroup_rows` so simdgroups work on distinct query rows inside a threadgroup while the selected tile height remains unchanged. It must retain shared K/V threadgroup staging when `kv_reuse` is enabled. `tile_height` controls the number of rows owned by the threadgroup; `simd_topology` controls lane-to-head mapping inside each simdgroup. Assert exact once-only coverage of a Granite 8-KV-head, 2-query-group fixture and unchanged staged K/V identity. Explicit `simdgroup_rows` on `query_rows=1` returns a typed decline; the separate `query_parallelism=legacy` control still selects the legacy decode-split form.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC17 | `cargo nextest run -p omega --features metal-attn-split-decode,metal-attn-split-rows,metal-attn-variants --lib -E 'test(~card_17_query_parallelism)'` | filter selects 2 tests; 2 passed; row coverage plus explicit one-row decline and legacy decode-split control |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
