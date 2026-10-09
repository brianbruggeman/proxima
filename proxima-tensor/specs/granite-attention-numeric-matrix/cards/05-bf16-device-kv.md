# Card 05: bf16-device-kv

**Owner:** GPT-6 Luna
**Dependency:** 04
**Commit:** `feat(interop): retain bf16 device kv`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Select BF16 DeviceKv storage using the new placed helpers.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `proxima-model-interop/src/generate/device_kv.rs:1-27,49-127,232-279,365-369`

## Edit

- `proxima-model-interop/src/generate/device_kv.rs`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Extend element_bytes, store/load, staged commit, and cache_codec. Preserve prefix rewind and ring offsets. Test adopt/append/flush of BF16 values and unsupported codec decline.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC05 | `cargo nextest run -p proxima-model-interop --features std,metal,metal-attn-split-rows --lib -E 'test(~card_05_bf16_device)'` | filter selects 2 tests; 2 passed |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
