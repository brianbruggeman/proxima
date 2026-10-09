# Card 02: add a second Granite prefill length

**Owner:** GPT-6 Luna
**Dependency:** 00,01
**Commit:** `perf(interop): compare two granite prefill lengths`
**Budget:** at most 30 minutes front-to-back, including AC2a/AC2b, review, and TASKS.md update.

## Purpose

Extend the Card 00 measurement to a second, shorter prefix of the same Sherlock passage. Keep each length's real legacy/SharedK output gate and raw replay cells separate in one checked report.

## Read

- `../SPEC.md` for the two-shape report contract and exact acceptance counts.
- Card 00's reusable shape measurement function and Card 01's `--expected-shapes` checker.
- `proxima-model-interop/tests/granite_attention_variant_prefill.rs:113-135` for the tokenizer-driven Sherlock prompt builder.

## Edit

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: parameterize the prompt builder by target token count without changing existing callers; add one test that calls the existing shape measurement function for nominal 256 and 971 token targets and writes both to one report.
- `proxima-tensor/specs/granite-attention-variant-performance/TASKS.md`: tick after the measurement and checker counts are inspected.

## Steps

1. Build both prompts from the same repeated Sherlock passage and checkpoint tokenizer. Require actual counts at least 256 and 971, short count less than long count, and short prompt text and encoded IDs a prefix of the long. Record actual counts and prompt-ID digests, not only nominal targets.
2. On one loaded model, run the Card 00 capture, exact output/ID gate, 20 alternating replay samples per arm, and `Cell` resource observation for the short shape, then the long shape. Keep each shape's owning legacy/selected `CapturedDispatch` handles alive until its 20 rounds and resource cells finish. Drain captures between arms/shapes. Both shapes use Card 00's exact `round=0..19`, `arm_order`, and per-arm `position` schema. Preserve both shape records in the same `version=1` JSON report; fail on any nonreplayable, fault-bound, or mismatched dispatch.
3. Run Card 01's checker with `--expected-shapes 2`. Read every raw arm row and round difference; report gains and losses as observations without a speed verdict.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC2a | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_02_granite_two_prefill_shapes)' -j 1 --success-output immediate` | filter selects exactly 1 test and reports 1 passed; output prints 4 `granite ab arm` lines; report has 2 distinct verified actual prompt counts, 4 captured arms, 20 raw samples and 1 resource line per arm, 2 equal output/ID pairs, 0 replay errors |
| AC2b | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 2 /private/tmp/granite-attention-ab.json` | prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0` |

## Residual

Two prefill lengths from one passage are the first shape contrast. Isolated dispatch replay does not measure whole-layer or public-request latency, and the measured signs alone do not establish the causal mechanism of a performance gap.
