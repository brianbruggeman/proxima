# Card 25: measure the Groups4 tail with 1,000 paired rounds

## Question

At fixed F16-MMA/F32-KV and otherwise Legacy schedule, does configured Groups4 retain a lower nearest-rank p99 cached-attention timestamp than sized Legacy across 1,000 alternating pairs at actual 256 and 972 prompt tokens, in each of three serial processes?

## Pre-registered hypothesis and kill criterion

Hypothesis: Groups4 p99 is lower than Legacy p99 in all six process/shape cells. Kill this tail direction if any selected p99 is greater than or equal to Legacy p99, if output/ID equality fails, or if any report is missing one of its 1,000 paired samples. Also report p50, p90, CoV, and every signed paired difference; the p99 test does not erase those records.

Card24's 20-round p99 is the maximum sample under nearest-rank calculation. That card's fresh p99 values were lower in all six cells, while earlier Card17 run1 had one long-prompt crossing. This card raises the measured-round count to 1,000 so the p99 rank is based on the 990th sample rather than the maximum of 20.

## One-variable contract

Both arms use F16 MMA, F32 K/V, Legacy K/V reuse, tile height, query parallelism, topology, and prefetch off. Selected changes only the simdgroup count to Groups4 through `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4`. Each prompt shape gets two warmup pairs and 1,000 alternating measured pairs per arm. Resource context remains five additional replays per arm.

## Commands

Build one focused target using one Cargo job, copy it, then run three Metal processes serially. Reports contain the complete sample arrays and paired deltas.

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill --no-run
cp /tmp/cargo_target/debug/deps/granite_attention_variant_prefill-4c29247f309c5b09 /tmp/card25-granite-attention-tests
for run in 1 2 3; do
  PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT="/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card25-groups4-run${run}-report.json" /tmp/card25-granite-attention-tests perf_granite_simdgroups4_tail_revalidation_1000_pairs_two_shapes --nocapture
done
```

For each report, run `check_report.py --expected-shapes 2 --expected-samples 1000 --selected-arm simdgroups4`, then repeat with `--negative-controls`.

## Acceptance

- Three serial processes each select one test and pass one; each report contains two actual prompt sizes, two warmup pairs, and 1,000 measured pairs per arm/shape.
- Each report passes its positive checker and rejects all 13 negative controls without changing the raw report.
- Both shape pairs preserve complete output hashes and generated IDs; retain the exact IDs, hashes, source identity, grid, pipeline resources, and process context.
- Evaluate the p99 kill criterion and preserve all paired signs and raw per-round samples.

## Result

The focused target compiled in 6.11 seconds using one Cargo job. Three serial Metal processes each selected one test, passed one, and filtered 35. Each actual shape has two warmup pairs and exactly 1,000 alternating measured pairs per arm. All three reports passed the positive checker with `samples_per_arm=1000` and rejected 14/14 negative controls, including the new `measured_rounds` metadata mutation.

Both outputs and generated IDs `[322]` matched within all six pairs. At actual 256 tokens the output SHA is `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6`; at 972 tokens it is `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae`. Each report preserves every arm timestamp, paired delta, output identity, dispatch entry/source, grid, pipeline resources, and process context.

Cells list Legacy/Groups4 nearest-rank p50/p90/p99 GPU nanoseconds, CoV percent, and Groups4-minus-Legacy paired signs (+/−/0):

| Run/report SHA256 | Actual tokens | p50 ns | p90 ns | p99 ns | CoV % | signs +/−/0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 `ed2bce4ff5d8…b7cfc11f` | 256 | 184,500.0 / 145,958.4 | 185,666.7 / 147,375.0 | 187,875.0 / 149,083.3 | 2.31 / 0.80 | 0/1000/0 |
| 1 | 972 | 1,091,583.3 / 1,057,583.4 | 1,124,875.0 / 1,104,624.9 | 1,189,750.0 / 1,143,333.4 | 3.33 / 3.24 | 253/747/0 |
| 2 `fd9585749a81…4052b77d` | 256 | 184,708.3 / 145,750.0 | 186,291.6 / 147,333.3 | 187,999.9 / 149,625.0 | 1.95 / 5.61 | 3/997/0 |
| 2 | 972 | 1,100,875.0 / 1,049,125.0 | 1,125,625.0 / 1,096,625.0 | 1,152,000.0 / 1,129,875.1 | 2.50 / 2.51 | 132/868/0 |
| 3 `1d47df1cf790…b3c8410e0` | 256 | 184,875.1 / 146,125.0 | 186,166.6 / 147,416.7 | 188,499.9 / 149,000.0 | 0.68 / 0.70 | 0/1000/0 |
| 3 | 972 | 1,098,916.6 / 1,049,208.3 | 1,123,083.3 / 1,095,250.0 | 1,142,041.6 / 1,133,625.0 | 2.27 / 2.57 | 142/858/0 |

The pre-registered p99 condition holds in these six cells: Groups4 p99 is lower in each. At 256 tokens every one of 3,000 paired deltas is negative. At 972 tokens the per-process positive (selected-slower) deltas are 253/1,000, 132/1,000, and 142/1,000. The selected p50, p90, and p99 are lower in all six cells. This records a repeated timestamp distribution for this setup; it does not establish a cause or a whole-request speed result.

I split the 972-token paired deltas by alternating arm order directly from each report's 1,000 raw `rounds` records. The `[legacy, simdgroups4]` and `[simdgroups4, legacy]` halves had positive counts 116/500 vs 137/500 in run1, 67/500 vs 65/500 in run2, and 66/500 vs 76/500 in run3. Their respective median deltas were −30,583/−26,917 ns, −46,667/−47,625 ns, and −45,000/−42,375 ns. This split does not expose a large order-linked difference in these summaries; it does not identify why the 972-token cells still have positive paired rounds. Complete order and sample arrays remain in the reports.

### Additional raw-round inspection: chronological windows

To check whether selected-slower rounds cluster over a run, I grouped each report's 972-token `rounds` by consecutive 100-round windows and computed, from the paired `selected_minus_legacy_ns` values, the positive-delta count and the median delta. The command was:

```sh
for run in 1 2 3; do
  jq -r 'def median: sort as $values | if length % 2 == 1 then $values[(length/2|floor)] else (($values[(length/2|floor)-1] + $values[(length/2|floor)])/2) end; .shapes[1].rounds | group_by(.round / 100 | floor)[] | [((.[0].round / 100 | floor) * 100), (length), ([.[] | select(.selected_minus_legacy_ns > 0)] | length), ([.[] | .selected_minus_legacy_ns] | median), ([.[] | select(.arm_order[0] == "legacy" and .selected_minus_legacy_ns > 0)] | length), ([.[] | select(.arm_order[0] == "simdgroups4" and .selected_minus_legacy_ns > 0)] | length)] | @tsv' "evidence/card25-groups4-run${run}-report.json"
done
```

The following values are for actual 972-token Granite prompts, Groups4-minus-Legacy GPU timestamp deltas, 1,000 paired rounds per process. Each cell lists `selected-slower count / median delta ns / positive count with Legacy first / positive count with Groups4 first`:

| Rounds | Run 1 | Run 2 | Run 3 |
|---:|---:|---:|---:|
| 0–99 | 30 / −30,667 / 14 / 16 | 13 / −48,125 / 7 / 6 | 15 / −47,771 / 7 / 8 |
| 100–199 | 28 / −25,375 / 12 / 16 | 10 / −50,313 / 6 / 4 | 20 / −39,167 / 11 / 9 |
| 200–299 | 19 / −29,583 / 13 / 6 | 17 / −40,480 / 10 / 7 | 13 / −50,104 / 5 / 8 |
| 300–399 | 22 / −27,396 / 8 / 14 | 12 / −49,250 / 6 / 6 | 14 / −41,167 / 4 / 10 |
| 400–499 | 22 / −28,687 / 13 / 9 | 15 / −47,354 / 9 / 6 | 13 / −49,354 / 5 / 8 |
| 500–599 | 28 / −26,146 / 9 / 19 | 9 / −51,583 / 4 / 5 | 16 / −36,021 / 8 / 8 |
| 600–699 | 28 / −32,750 / 13 / 15 | 14 / −40,292 / 4 / 10 | 13 / −41,437 / 8 / 5 |
| 700–799 | 23 / −28,375 / 9 / 14 | 13 / −46,396 / 6 / 7 | 17 / −44,000 / 7 / 10 |
| 800–899 | 30 / −29,604 / 15 / 15 | 14 / −50,229 / 10 / 4 | 14 / −38,188 / 6 / 8 |
| 900–999 | 23 / −25,083 / 10 / 13 | 15 / −46,146 / 5 / 10 | 7 / −47,979 / 5 / 2 |

In this 100-round grouping, all thirty window medians are negative, while every window contains some positive paired deltas (7–30 per window). Positive counts are distributed across both arm orders. This rules out a large, sustained 100-round interval where the selected arm's median reverses, for these three processes; it does not explain individual positive rounds or establish which GPU mechanism produced the distributions. The underlying individual observations remain in `evidence/card25-groups4-run{1,2,3}-report.json`.

Selected source SHA is `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`; Legacy source SHA is `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`. Groups4's captured launch uses threadgroup width 128, 32,768 total grid threads at 256 tokens and 124,928 at 972 tokens; pipeline `(max_threads, tg_static_bytes, exec_width)=(576,9600,32)`. Legacy uses width 64, 16,384/62,464 total grid threads, and `(448,9600,32)`. The MSL changes `simdgroups` from 2 to 4 and derives two depth/key fragments per group rather than four. These are dispatch/source differences; the Mac exposes timestamp counters only, so the timing mechanism remains unexplained.

Report SHA256: run1 `ed2bce4ff5d8bc9ceea8fe5baf57069a2cc3faafba69970e9d82d4dfb7cfc11f`, run2 `fd9585749a812492e398831b35cda07a7c3ae06313e894801b7d715d4052b77d`, run3 `1d47df1cf790ef40462bbc81611369a08821653205b444c73b389b3ebc8410e0`. Logs, reports, build output, and checker output are retained in `evidence/card25-*`.

## Residual

Even 1,000 rounds produce only ten observations at or above nearest-rank p99. The Mac exposes no shader counters or labels, so any timestamp distribution remains without a device-level cause. Cross-process and longer-context behavior beyond these three processes and two prompt sizes is unmeasured.
