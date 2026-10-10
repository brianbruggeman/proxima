# Card 24: revalidate Groups4 p50 and tail records

## Question

On three new serial Granite replay processes, does the configured Groups4 count retain lower paired-cell p50 than sized Legacy at actual 256 and 972 prompt tokens, while preserving complete output and generated IDs? What do p90, p99, CoV, and the signed pairs show in those same cells?

## Pre-registered hypothesis and kill criterion

Hypothesis: at fixed F16 MMA/F32 K/V and otherwise Legacy schedule, Groups4 has lower p50 than Legacy at both prompt sizes in each of three new processes. Kill this direction if any one of the six cells has selected p50 greater than or equal to its Legacy p50, or output/ID equality fails. p90/p99 and paired signs are required counter-evidence fields; they are not omitted when they disagree with p50.

This question follows Cards 16–18 and 21: those records show lower Groups4 p50 observations, with occasional p99 crossings and mixed signs at the longer prompt. It does not authorize changing the serving default.

## One-variable contract and source mapping

Baseline is F16 MMA, F32 K/V, sized Legacy count, Legacy K/V reuse, tile height, query parallelism, and SIMD topology, with prefetch off. Selected changes only `simdgroup_count` through `PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4`. The test asserts the configured count reaches its selected `AttentionVariant`.

For the captured head dimension 64 and block 64, `omega/src/msl/cached_attention_row_tiled.rs:447-449` derives four simdgroups, 128 threads, two depth fragments per group, and two key fragments per group. Card18's saved Groups4 MSL diff against Legacy shows the entry name, simdgroup constant, and generated fragment formulas as the dispatch-source changes. These formulas describe work assignment; they do not explain device timing.

## Commands

The focused copied binary built for Card23 includes the Groups4 configured replay. Run the three Metal processes serially, then check each report with `--expected-shapes 2 --selected-arm simdgroups4` and the same arguments plus `--negative-controls`.

```sh
for run in 1 2 3; do
  PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4 PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_GRANITE_AB_REPORT="/private/tmp/proxima-granite-main/proxima-tensor/specs/granite-attention-variant-performance/evidence/card24-groups4-run${run}-report.json" /tmp/card23-granite-attention-tests perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy --nocapture
done
```

## Acceptance

- Each process selects one test, passes one, and records two warmup pairs plus 20 alternating measured pairs per arm and shape.
- Each report passes the positive checker and rejects all 13 negative controls.
- Each of six cells retains paired raw samples, output/ID identity, source/entry, grid, pipeline resources, and process resource context.
- Evaluate the pre-registered p50 kill criterion and report p90/p99/sign counter-evidence without assigning a GPU mechanism from timing alone.

## Result

The copied focused binary ran three serial processes; each selected one test, passed one, and filtered 34. Each report captured actual counts 256/972 with two warmup pairs and 20 paired samples per arm. The positive checker printed the expected two-shape, 20-sample counts; all three negative-control invocations rejected 13/13 mutations. Complete output SHA and generated IDs `[322]` matched within each pair in every cell. Output SHAs are `454ca3f4483db67f72f42baa6f7625c46853a9be768e4bbcdd35116789f72ba6` at 256 tokens and `970fa9d575423441d23b639a5831938d9e9c46aa8c3c6e84fb2d4f7fe0dfbcae` at 972 tokens.

Cells list Legacy/Groups4 p50, p90, p99 GPU nanoseconds; CoV percent; and Groups4-minus-Legacy paired signs (+/−/0):

| Run/report SHA256 | Actual tokens | p50 ns | p90 ns | p99 ns | CoV % | signs +/−/0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 `88cd9a01ad96…c8696c81` | 256 | 181,541.7 / 144,625.0 | 185,500.0 / 145,500.1 | 226,875.0 / 146,125.0 | 5.38 / 0.52 | 0/20/0 |
| 1 | 972 | 1,082,625.0 / 1,063,291.7 | 1,113,625.0 / 1,081,500.0 | 1,122,583.3 / 1,099,375.1 | 1.84 / 1.62 | 4/16/0 |
| 2 `f955d4594af0…4460a412` | 256 | 182,000.0 / 144,750.0 | 183,041.7 / 146,208.3 | 183,875.0 / 147,375.1 | 0.34 / 0.79 | 0/20/0 |
| 2 | 972 | 1,092,291.7 / 1,050,208.4 | 1,124,125.0 / 1,080,666.7 | 1,148,208.4 / 1,105,791.7 | 2.65 / 2.16 | 3/17/0 |
| 3 `31bd5e3cb5a9…a221706a` | 256 | 181,916.6 / 144,750.0 | 182,500.0 / 145,208.3 | 182,875.0 / 146,000.0 | 0.36 / 0.48 | 0/20/0 |
| 3 | 972 | 1,096,833.3 / 1,072,250.0 | 1,152,000.0 / 1,137,750.0 | 1,494,000.0 / 1,280,500.0 | 9.79 / 5.30 | 5/15/0 |

The pre-registered p50 condition holds for these six new process/shape cells. In the 972-token cells, paired signs remain mixed. For the same six cells, selected p99 is below Legacy p99; the long run 3 p99 values and CoV retain a high-tail record on both arms. Older Card17 runs include a selected p99 crossing in long run 1, while Card21's fresh report has lower selected p99 at both lengths. These records do not establish a stable tail bound or explain the time distribution.

Captured selected source SHA is `637d62efe5ffb3c07f0df1b406cd5a4171c337a73177b80e3737b4eecbcd956f`, Legacy is `b85db41a81b3503641fbdb453cff0b7dd32e442085614eef5ab20fe1b1e0ba5d`. At actual 256 tokens, total grid threads are 32,768 vs Legacy 16,384; at 972, 124,928 vs 62,464. Selected threadgroup width is 128 vs Legacy 64. Selected pipeline resources are `(max_threads, tg_static_bytes, exec_width)=(576,9600,32)` versus Legacy `(448,9600,32)`. Source derives two depth and two key fragments per simdgroup for Groups4, versus four and four for sized Legacy. This is the source change associated with the selected dispatch; its causal contribution to elapsed time is not measured.

Report SHA256: run 1 `88cd9a01ad96f9d7ac1d2a41c3a0338e11955dd5e5689bf8068e05cac8696c81`; run 2 `f955d4594af0ecfe1ff29afe7603d4249da2fe4112db5cc8b952f2c44460a412`; run 3 `31bd5e3cb5a94d56be22b8af92b781a8c07567619deeb5cb8fcea8b9a221706a`. Raw per-round timing, warmup, output, source, grid, resource, and process records remain in the matching `evidence/card24-groups4-run{1,2,3}-report.json` files.

## Residual

The M1 Max probes recorded timestamp counters but no shader counters or labeled intervals. Repeated direction can describe the measured cells; it cannot establish why the kernel takes that time or support a default change on its own.
