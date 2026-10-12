# Omega programmable execution evidence

## Experiment conditions

The two projection arms run the existing configured Omega Metal plan on an
Apple M1 Max. The checkpoint is Ouro-1.4B revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
The matrix is `model.layers.0.self_attn.q_proj.weight`, BF16 storage,
2,048 input and output channels, file bytes `[488687986,497076594)`.
Activations are Float32 checkpoint embeddings from the frozen prefix IDs.
These facts are recorded in each arm's `projection_setup` rows 2, 11 and 20.

Both builds enable `metal-tiled-gemm`, explicitly set minimum tokens to 1,
and use [relaxed-metal.toml](relaxed-metal.toml): contraction and reassociation
permitted, other numeric rewrites denied. Only
`OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0/1` differs. The default production
minimum remains 160 and the float-storage extension is off by default.

## Initial projection cohort, including losses

Each cell contains five measured executions after one separately recorded
warmup. The timer ends after synchronous output readback. Means and CoV below
are **derived** from the raw durations in the cited rows; they are not GPU
occupancy measurements. No outliers were removed. These are fixed-prefix
projection fixtures, not a held-out language-quality evaluation.

| Activation rows | Scalar mean ms | Scalar CoV% | Tiled mean ms | Tiled CoV% | Source rows |
|---|---:|---:|---:|---:|---|
| 1 | 0.3656584 | 11.87944 | 2.4979416 | 1.80241 | [scalar](results/micro-disabled-v1/microbench.jsonl):9; [tiled](results/micro-enabled-v1/microbench.jsonl):9 |
| 5 | 0.7357250 | 1.08799 | 1.2775248 | 10.26723 | same artifacts, row 18 |
| 24 | 2.7412670 | 0.42010 | 2.8032000 | 0.94361 | same artifacts, row 27 |

This initial cohort precedes the operand-role cache-key repair. It is retained
separately from the rerun against the integrated implementation. The recorded
means are higher in the tiled arm for all three shapes. This
observation does not establish the cause of the latency difference. Raw CPU
time, RSS, host load, Metal counters, output vectors, per-coordinate signed
errors, and arithmetic error envelopes accompany every execution record.
Counter snapshots reflect the executor's per-call reset; they must not be
treated as monotonic process counters and blindly subtracted across calls.

## Executed lowering and literal payloads

[The captured device run](results/validation/omega-root-ac6-captured.txt)
contains eight test executions, including four real Metal cases. Lines 32,
42, 52 and 62 record node, source hash, extents, grid and operand codecs.
The adjacent `literal_matrix` rows contain all encoded inputs and output
coordinates. The simple oracle repeats `[17,39]` / `[23,53]`; the ragged
oracle repeats `[35,70,105,140,175]` / `[70,140,210,280,350]`.
Positive and one-coordinate-corruption controls use the same comparator.

The BF16 source [pipeline_74eada85477b2aa7.metal](results/literal-sources/pipeline_74eada85477b2aa7.metal)
reads BF16 into float tile storage at line 896 and calls
`simdgroup_multiply_accumulate` at line 930. The actual captured launch has
640 threads, width 128, depth 1 for extents `[160,2,2]`. An auxiliary
[checkpoint projection run](results/micro-enabled-source-capture/microbench.jsonl)
captured the same source hash. Its timings remain a separate cohort.
The disabled checkpoint arm captured
[scalar source](results/micro-disabled-v1/pipeline_a07d5d715b8c0579.metal).

## Paired Ouro execution after the cache repair

The shorter prefix has a higher recorded mean with float-storage tiling enabled.
The longer prefix has a lower recorded mean. Both remain above the pinned
llama performance oracle. These are observations for frozen prefixes, not a
language-quality score or a general scheduling decision.

Each arm uses the same contraction/reassociation permissions, minimum-token
override, requested roots, checkpoint, and four recurrent loops. Each cell
contains five measured executions after its own warmup. The two source
manifests were compared before their builds. The primary data lives in
[enabled samples](results/ouro-configured/samples.jsonl) and
[disabled samples](results/ouro-configured-control/samples.jsonl); each row
contains complete output arrays, CPU/RSS observations, driver counters, and
configuration. The following values are derived from those durations, also
serialized under `case_id`/`method` in each arm's `summary.json`.

| Prefix / method | Disabled mean ms | Disabled CoV% | Enabled mean ms | Enabled CoV% |
|---|---:|---:|---:|---:|
| france / baseline4 | 479.8250834 | 0.31401 | 568.3160332 | 3.44911 |
| france / fixed-logits-0.5 | 482.1731414 | 0.59749 | 567.7937916 | 0.20498 |
| france / adaptive-logits-cap-1 | 481.9720918 | 0.32959 | 573.4946500 | 2.72554 |
| france / fixed-hidden-0.5 | 482.2238418 | 0.60372 | 562.5381420 | 0.69260 |
| ball / baseline4 | 2063.8102916 | 0.34079 | 615.0300334 | 0.39112 |
| ball / fixed-logits-0.5 | 2067.8812916 | 0.32748 | 622.5347914 | 0.33696 |
| ball / adaptive-logits-cap-1 | 2071.7889332 | 0.32607 | 622.2731418 | 0.42551 |
| ball / fixed-hidden-0.5 | 2069.6805834 | 0.42356 | 616.1276750 | 0.45402 |

The source condition is Ouro-1.4B BF16 weights with Float32 activation arithmetic
on the M1 Max. The prefixes contain respectively 5 and 24 frozen IDs in
[frozen-prefixes.json](frozen-prefixes.json). Status: the recorded durations
and derived statistics are observations; a general performance claim remains
unmeasured. The experimental gate remains off by default.

The independent native [timing recount](results/recount/timing-recount.jsonl)
reconciles the reported means, sample SD, CoV and percentiles against every raw
duration. It contains 25 records. Its pinned llama cells (lines 23 and 24) use
the same checkpoint, prefix IDs and recurrent depth, with the arithmetic and
final-layer work differences recorded explicitly:

| Pinned llama baseline4 case | Mean ms | CoV% | Samples |
|---|---:|---:|---:|
| france | 43.3585084 | 1.66875 | 5 |
| ball | 103.0217582 | 3.33784 | 5 |

The [coordinate reconciliation](results/recount/coordinate-comparisons.jsonl)
contains 18 paired projection vectors, one comparator-control record and
168 model vectors across all 48 paired warmup/timed invocations. It retains
each changed coordinate, both values and its signed difference. Argmax did
not change between the two build-gate arms in these records, but the vectors
are not equal. For example, `ball:baseline4:sample:0`, field `logits`,
coordinate 0 changes from `5.995109558105469` to `5.995102405548096`; coordinate
1 changes from `-15.224569320678713` to `-15.224539756774902`. The two signs
are both retained. These differences are compatible with the granted
reduction/contraction rewrites, but no model-quality tolerance is inferred
from matching selected tokens.

For a concrete execution trace, `ball:baseline4:sample:0` records 96 layer
executions, 3,181 physical dispatches, and 681 readbacks in both arms.
Disabled wall time is 2064.508458 ms, of which 2039.320208 ms is host
commit-through-wait time; enabled wall time is 619.036083 ms with
602.727750 ms in that bracket. `france:baseline4:sample:0` records the same
dispatch/readback counts, with disabled 477.898709 ms and enabled 556.562000 ms.
The graph work and readback boundaries were retained. Kernel source selection
changes through the shared dense classifier. These host brackets do not
distinguish GPU arithmetic, scheduling, or command latency and are not device
busy-time measurements.

Both arms' `records.jsonl` begin with CPU and Metal model/paper controls.
The hidden readout now records
`[-0.2324952781200409, 2.789943218231201, 2.557447910308838]` against the
independent expected
`[-0.2324952781200409, 2.789943218231201, 2.557448148727417]`, within the
unchanged recorded tolerance. Wrong-sign and wrong-normalization controls
retain their mismatching payloads.

## Projection repetitions after the cache repair

All three cohorts retain five timed samples and one warmup per shape and arm.
No cohort replaces another. These ranges span cohort means, not individual
latencies; the raw vectors and per-sample CPU/RSS/load records remain in each
JSONL. The comparison uses the same checkpoint projection and conditions
described above. Source records are `projection_summary` at lines 9, 18 and 27
in each of `micro-enabled`, `micro-enabled-repeat2`, `micro-enabled-repeat3`,
and their `micro-disabled` counterparts.

| Rows | Disabled cohort means ms | Disabled CoV% by cohort | Enabled cohort means ms | Enabled CoV% by cohort |
|---|---|---|---|---|
| 1 | 0.4420666, 0.3723666, 0.4078418 | 11.48442, 10.47442, 13.38393 | 1.5656834, 2.4462834, 1.3215338 | 13.98773, 0.99287, 7.84690 |
| 5 | 1.0155666, 0.7733832, 0.7357418 | 10.98024, 3.32362, 1.93040 | 2.9362502, 2.5071332, 1.4698000 | 3.62771, 0.57055, 7.24155 |
| 24 | 2.8551414, 2.6700166, 2.7121080 | 1.84508, 0.63083, 0.80212 | 2.4889164, 2.1199918, 2.8804334 | 14.96851, 14.32074, 1.56317 |

The recorded small-row tiled projection means exceed the disabled means in
every cohort. The 24-row comparison changes direction across cohorts. Device
occupancy was unavailable; neither the variation nor its cause is hidden by
pooling these samples into one headline.

## Failures retained

- The [first admission audit](results/validation/omega-programmable-spec-audit.txt)
  refused incomplete specification details. The revised admission artifacts
  are retained beside it.
- The [first device test run](results/validation/omega-root-ac6.txt) rejected
  the expected tiled identity. It used a 160-row Metal fixture. The classifier
  assigned weight role to operand zero, inspecting the two-column axis as
  tokens for activation-first projections. The classifier now uses exclusive
  output-axis ownership; both operand orders are asserted.
- The [initial microbench launch log](results/validation/omega-micro-enabled-console.txt)
  records an output-directory collision before inference. That failure occurs
  before the runner's `failure.json` writer, so no such JSON file is claimed.
- The first completed enabled microbench used an incorrect source-capture
  directory. Its timing/payload cohort is retained; source capture was rerun
  separately rather than fabricating missing files.
- The [configured paper-control failure](results/ouro-configured-failed-paper-control/failure.json)
  records hidden logits with the first value repeated in every channel. The
  dense renderer selects operand roles, but its cache key initially omitted
  those roles. The existing identity tuple now records weight and activation
  indices, with reversed-order source/key assertions in the tiled fixture.
- The first native recount command used the wrong oracle directory and failed
  after writing partial output. Its console log and `recount-failed-path/`
  artifacts are retained. The repeated command uses the pinned
  `ouro-llama-oracle/results/llama-v1/llama.jsonl` and emits the complete
  25-record timing artifact.
- A separate main-workspace recount encountered a stale copy containing only
  47 of the completed run's 48 records. Its count guard rejected that copy.
  The complete frozen source artifact was restored; the failed log is retained
  as `omega-final-recount-failed-missing-row.txt`. No inference sample was
  fabricated or omitted from the canonical recount.

## Main-workspace validation records

These are the counts printed by commands run from
`/Users/brianbruggeman/repos/slot-0/proxima`, with dedicated target directories.
The fixture assertions and literal vectors are in the work packets and source;
the records below identify what executed rather than claiming complete program
correctness.

| Surface | Observed execution count | Raw artifact under `results/validation/` |
|---|---:|---|
| Config, common planner, CPU budgets and selector | 34 passed | `omega-main-config-cpu-selection.txt` |
| Binder fixtures, existing binder cases and packed regressions | 45 passed | `omega-main-binding-legacy-packed.txt` |
| Configured Metal and wgpu device fixtures | 8 passed | `omega-main-metal-wgpu.txt` |
| Enabled tiled admission/source/grid/cache fixtures | 8 passed | `omega-final-ac4.txt` |
| Compile-off storage refusal | 1 passed | `omega-final-ac5.txt` |
| Literal CPU/Metal vectors and comparator controls | 8 passed, 0 skipped | `omega-final-ac6.txt` |
| Configured API rustdoc | 1 passed | `omega-main-configured-doctest.txt` |
| CUDA availability probe | 1 executed, missing driver returned | `omega-main-cuda-availability-probe.txt` |
| CUDA hardware fixtures | 4 compiled/listed; 0 device executions | `omega-main-cuda-four-list.txt` |
| Alloc-only library build | 1 Omega artifact, 0 compiler errors | `omega-main-ac7-json-only.txt` |
| Std/CPU library build | 1 Omega artifact, 0 compiler errors | `omega-main-ac8-json-only.txt` |

The final all-driver Clippy command emitted no diagnostics
(`omega-main-final-clippy-r4.txt`). Its earlier combined-feature compilation
found missing CUDA arms in Metal-specific diagnostic APIs; those APIs now
return an explicit unsupported-backend error for CUDA. The failed compiler
output remains in `omega-main-final-clippy.txt`. Source review boundaries and
the exact inspected assertions are retained in
`omega-programmable-final-review.txt`.

The device fixture log retains the actual buffer rows. Metal's affine fixture
has six physical buffers, with capacities `16,16,16,48,16,32`. Wgpu has twelve
rows, including explicit uploads and readback staging, totalling 208 bytes.
The wgpu one-byte-below controls record `(allocation,upload,dispatch,submit)`
as `(0,0,0,0)`; the admitted call records `(12,4,2,1)` and its repeat records
`(8,4,2,1)`. Thus the refusal observation uses counters also exercised by an
admitted call. Metal likewise records zero upload/allocation/dispatch counters
on its refusal path.

The same log's `heterogeneous_selection_raw` contains actual CPU, Metal and
wgpu candidate durations for the literal add/negate fixture. The selected
CPU record has the minimum retained sum; its output is checked against
`[11,18,33,36]` and `[-11,-18,-33,-36]`. This is a fixture observation, not a
recommendation to select CPU for model inference.

CUDA's observed error is `Driver("cuda driver library unavailable")` on this
M1 Max. CUDA device execution and NPU/ANE execution are unmeasured. The latter
has no registered driver and its config request is rejected explicitly.

## Interpretation limits

llama.cpp remains the performance oracle. Its pinned native run uses different
matrix arithmetic and avoids some final-layer work performed by the Proxima
runner. A wall-time comparison is not numerical equivalence. Projection
channel indices are not vocabulary token IDs. Resource ceilings concern
owned payloads and physical buffers, not total process RSS or all driver
memory. No measurement here establishes full device utilization or a globally
optimal schedule.
