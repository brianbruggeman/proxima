# Card 00: bf8-contract

**Owner:** GPT-6 Luna
**Dependency:** none
**Commit:** `docs(tensor): bind scalar bf8 contract`
**Budget:** at most 30 minutes front-to-back, from first source read through AC, review, and TASKS.md update. If the work cannot fit, split this card before implementation; do not ship a partial commit.

## Purpose

Commit exact scalar E5M2 BF8 byte contract and a 16-vector CSV.

## Scope

In scope: one behavior, the two acceptance tests (or two checker gates for card 00), and the TASKS.md row/resume update in the same coherent commit. Out of scope: CUDA, benchmarks, default activation, unrelated source and later cards. Keep the all-legacy baseline selectable.

## Read

- `proxima-onnx/src/types.rs:33-40`
- `proxima-tensor/src/convert.rs:22-31`

## Edit

- `proxima-tensor/specs/granite-attention-numeric-matrix/BF8_CONTRACT.md`
- `proxima-tensor/specs/granite-attention-numeric-matrix/bf8_vectors.csv`
- `proxima-tensor/specs/granite-attention-numeric-matrix/check_bf8_contract.py`
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

Write bit field equations, bias, all specials and RNE ties. Copy the 16 exact vectors from SPEC.md. Checker parses each row, requires distinct cases, and recomputes expected bits from its own integer reference function; negative control mutates a vector and must be caught.

Use the existing Proxima numeric and Metal abstractions. Do not introduce num-traits, model-name special cases, or a silent fallback. Storage width, MMA operand width and f32 accumulator are different contracts. The two AC checks must cover a normal case and a refusing or degenerate case. Inspect assertions and emitted payloads, not merely the runner summary.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC00 | `python3 proxima-tensor/specs/granite-attention-numeric-matrix/check_bf8_contract.py` | checks=2 vectors=16 |

A zero-test match or a checker that does not open its fixture is failure. Run this local command after the edit. The expected count is part of the criterion.

## Residual

The card establishes behavior and structural or numeric payload evidence only; timing, throughput and model-quality claims require separate measured cells.
