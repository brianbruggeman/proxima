# granite-attention-variant-performance -- slices

Each card is one coherent commit, at most 30 minutes front-to-back. Update its checkbox, observation, and resume lines in the same commit. Numeric-matrix Card 26 must land before Card 00.

| # | card | dependency | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|---|
| 00 | [card](cards/00-one-shape-replay-cell.md) | numeric-matrix Card 26 | R1,R2,R3 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_00_granite_attention_replay_cell)' -j 1 --success-output immediate` | selects 1 test, 1 passed; 1 real prompt shape, 2 captured arms, 20 raw samples and 1 resource observation per arm, 1 exact output/ID pair, 0 errors | [ ] | pending Card 26 output/fault gate |
| 01 | [card](cards/01-check-report.md) | 00 | R4 | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 /private/tmp/granite-attention-ab.json` and `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --negative-controls /private/tmp/granite-attention-ab.json` | positive prints `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0`; control prints `negative_controls=6 rejected=6` | [ ] | pending Card 00 report artifact |
| 02 | [card](cards/02-second-prefill-shape.md) | 00,01 | R5 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_02_granite_two_prefill_shapes)' -j 1 --success-output immediate`; then `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 2 /private/tmp/granite-attention-ab.json` | test selects 1, 1 passed; checker prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0` | [ ] | pending Card 01 checker |

## resume

Last landed slice: none in this measurement spec
Next action: after numeric-matrix Card 26 lands, implement Card 00's one-shape raw capture/replay report and run AC0
Open question, if any: whether the real captured pair remains replayable with equal complete output bytes; numeric-matrix Card 26 gates measurement on that observation

## struck

- Timing Cards 27–28 were removed from `granite-attention-numeric-matrix` before implementation because the numeric-matrix spec forbids timing in implementation cards; this spec owns those measurements.
