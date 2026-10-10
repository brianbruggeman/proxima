# granite-attention-variant-performance -- slices

Each card is one coherent commit, at most 30 minutes front-to-back. Update its checkbox, observation, and resume lines in the same commit. Numeric-matrix Card 26 must land before Card 00.

| # | card | dependency | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|---|
| 00 | [card](cards/00-one-shape-replay-cell.md) | numeric-matrix Card 26 | R1,R2,R3 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_00_granite_attention_replay_cell)' -j 1 --success-output immediate` | selects 1 test, 1 passed; 1 real prompt shape, 2 captured arms, 20 raw samples and 1 resource observation per arm, 1 exact output/ID pair, 0 errors | [x] | `AC0: 1 passed, 11 skipped; report=/private/tmp/granite-attention-ab.json; checkpoint metadata and SHA identify Granite 3.1 1B A400M Instruct Q8_0; prompt=972 actual tokens; node=89 extents=[972,8,2,64]; equal full-output hashes and IDs [322]; 20 samples/arm, round signs=20 positive/0 zero/0 negative. F32 KV, isolated Metal replay p50 observations: legacy=1,068,250 ns, shared_k=2,339,500 ns; per-arm max=1,110,333 ns and 2,548,250 ns. These are one-shape observations. Run log=/private/tmp/granite-attention-ab-card00-final2.nextest.log. Output-path control rejected an attempt to use the checkpoint itself as the report; checkpoint SHA remained cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80.` |
| 01 | [card](cards/01-check-report.md) | 00 | R4 | `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 /private/tmp/granite-attention-ab.json` and `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 1 --negative-controls /private/tmp/granite-attention-ab.json` | positive prints `prompt_shapes=1 arms=2 samples_per_arm=20 resource_cells=2 matching_pairs=1 errors=0`; control prints `negative_controls=12 rejected=12` | [x] | AC1a printed the required counts and `round_differences_positive=20 negative=0 zero=0`; AC1b rejected all 12 in-memory mutations. Report SHA256 was `1bb5d3dd3fbcc19a9f27272ee73e48a87d897be03f86d1909079a3e62a82d788` before and after AC1b. |
| 02 | [card](cards/02-second-prefill-shape.md) | 00,01 | R5 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT=/private/tmp/granite-attention-ab.json cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_card_02_granite_two_prefill_shapes)' -j 1 --success-output immediate`; then `python3 proxima-tensor/specs/granite-attention-variant-performance/check_report.py --expected-shapes 2 /private/tmp/granite-attention-ab.json` | test selects 1, 1 passed; checker prints `prompt_shapes=2 arms=4 samples_per_arm=20 resource_cells=4 matching_pairs=2 errors=0` | [ ] | pending Card 01 checker |

## resume

Last landed slice: Card 00 one-shape capture and timing report, AC0 selected 1 test and passed 1
Current uncommitted slice: Card 01 report checker, AC1a accepted 1 shape and AC1b rejected 12/12 malformed copies
Next action: implement Card 02's second tokenizer-verified prompt shape and check the resulting two-shape report
Open question, if any: the actual tokenizer count and captured dispatch identity for Card 02's shorter prompt

## struck

- Timing Cards 27–28 were removed from `granite-attention-numeric-matrix` before implementation because the numeric-matrix spec forbids timing in implementation cards; this spec owns those measurements.
