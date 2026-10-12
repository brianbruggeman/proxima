# Ouro through the shared pipeline

## Implementation boundary

The lab `Model` now calls the existing `LoadedModel` loader and diagnostic API.
The real generation benchmark calls `LoadedModel::generate_from_ids`. The
Ouro registry adapter expresses physical layers, logical recurrent visits,
normalization and readout policy as shared descriptor data.

```text
GGUF / safetensors + recurrent settings
                  |
        existing architecture / weight loader
                  |
        shared ModelDescriptor + graph builder
        (physical weights, distinct logical KV owners)
                  |
        existing LoadedModel serving lifecycle
        prefill -> cached decode -> sampler -> tokens
                  |
        existing BackendRuntime plan cache
                  |
        Omega configuration -> backend lowering -> kernels
```

The lab's independent tiny controls and the native numerical oracle remain
reference consumers. They do not execute the production transformer. Existing
graph/attention/binding/cache/planning owners contain the reusable changes.

## First end-to-end checkpoint: default planner

The larger prefix is a synthetic length control: the frozen ball prompt IDs
are repeated and truncated. Neither case is a held-out language-quality corpus.
Both runtimes use the pinned Ouro GGUF, four recurrent passes, F32 KV, greedy
selection, empty KV at the start of each request, a 64-token output budget,
one separate warmup and five measured requests. Runs are serialized on the
same M1 Max. Proxima has the experimental dense-float tile extension off and
the default tile threshold of 160; this is not an enabled-tile result.

| Case | Runtime | Actual generated | Mean wall ms | Wall CoV% | Mean first-token ms | Source |
|---|---|---:|---:|---:|---:|---|
| repeated-ball256 | Proxima default | 64, budget exhausted | 37033.9562000 | 0.0966251 | 26586.6373754 | `results/proxima-cached-r2/cached.jsonl:17`; requests 12–16 |
| repeated-ball256 | pinned llama | 64, budget exhausted | 2908.8302502 | 2.5782207 | 449.4634502 | `results/llama-cached-r2/llama.jsonl:18`; requests 13–17 |
| ball24 | Proxima default | 24, EOS | 5684.1928336 | 0.1485965 | 2403.8031086 | `results/proxima-cached-r2/cached.jsonl:9`; requests 4–8 |
| ball24 | pinned llama | 24, EOS | 1042.4503832 | 6.5557086 | 114.4813250 | `results/llama-cached-r2/llama.jsonl:10`; requests 5–9 |

Status of these cells: observed native request durations and summary-derived
statistics. CoV uses sample standard deviation with the N-1 denominator. The
records retain every token ID, per-token latency, CPU/RSS/load observation and
error field. The table is not an attribution of the gap to a particular kernel.
For example, the first short Proxima sample has token IDs
`[441,4415,281,260,1977,30,378,2988,314,389,30,378,4405,314,2382,30,1116,198,504,2988,314,365,50,25]`;
the first llama sample carries the same array. Signed logit comparison is
separate from that trajectory observation.

Exact commands, failed attempts and source hashes are in
`results/cached-checkpoint/COMMANDS.md`. The default production run has 12
completed request records and the native oracle has 12 request records plus
two diagnostic requests. The output budget and the actual output counts are
recorded separately.

## Literal graph and loader evidence

`results/shared-graph/validation/ouro-shared-tests-r3.txt` contains the
literal vectors and frozen incumbent graph comparisons. The later enum-tier
rerun is recorded beside it. The frozen dense, routed and Gemma graphs compare
all Ops and roots, not just operation counts. Separate logical histories
produce `[3,0]` and `[7,0]`; deliberately using the wrong history produces
`[5,0]`. The full epsilon-one recurrent fixture produces
`[0.58834845,0.7844646]` against the hand-derived
`[0.5883484,0.78446454]`.

Interop CPU and Metal fixtures compare full cache rows after full prefill,
chunked continuation and reset. The stored HF split-half and GGUF interleaved
RoPE layouts have a width-four asymmetric control. The ordinary HF loader
has its own positive control. Runtime/serving identity mismatch is refused
before executing a differently configured graph.

## Repairs exposed by validation

The live multi-mode HF run reached a concrete ownership failure after its
numeric settings were aligned: a second variant offered a different embedding
allocation under `token_embd.weight` (`402653184` bytes on both sides). The
failure is retained in `results/cached-checkpoint/ouro-mainline-methods-live-r2.txt`.
Forward variants now share an immutable `ModelWeights` owner and retain separate
graphs. Only recurrence/readout changes are admitted by the reuse method.

`results/weight-owner/ouro-owner-tests-r5.txt:59` records the literal CPU
baseline `[0.58834845,0.7844646]`, fixed-logit output
`[0.47427437,0.6323658]` and hidden-guided output
`[-0.26182377,-0.34909838]`. Lines 82–84 retain every resident name/byte size,
the actual Metal survivor output `[-0.26182398,-0.34909844]`, the separate-owner
embedding identity error, and the empty final residency census. The assertions
check shared host payload identity, rejected physical descriptor changes,
survival after one graph is dropped and cleanup after final owner release.
The run executes the selected owner/recurrent cases plus the repaired task
output case: 15 executed, 15 passed, 553 excluded by the filter. This is fixture
coverage, not a multiple-conversation serving measurement.

The strict C7 recount is recorded in `results/cached-matched-batch-recount-r5`.
It preserves 36 request records, six independently recomputed timing cells,
four complete diagnostic vector comparisons, 24 trajectory pairs and one
comparator-control record. The earlier parser lost low bits from two recorded
floating-point durations; exact JSON float round-tripping and first-sample
centering of the independent Welford calculation repair that reproduction
without changing the comparison bound. The two literal regression cases and
their recovered values are in
`results/cached-checkpoint/ouro-mainline-recount-tests-r5.txt`.
The older mismatched-batch cohort is rejected with
`matched batch or instrumentation condition differs` in
`results/cached-checkpoint/ouro-mainline-cac6-wrong-batch-negative-r5.txt`.

- The caller's direct native safetensors dependency was absent; added with
  `cargo add` and rebuilt.
- The migration left unused declarations from the lab executor. Removed the
  redundant declarations; the independent oracle reads the retained typed HF
  metadata fields.
- The recurrent mode enum was hidden behind the tensor graph's std/config
  gate. Moved the plain enum to the tensor root and re-exported the old path.
- The cached runner duplicated the entry point's fresh-directory check and
  rejected every fresh run. The entry point now owns directory creation.
- Prefill and Token callbacks both carry the first ID. Token accounting now
  filters `Phase::Token`, while preserving all phase events in the trace.
- The native cached branch was placed before log initialization. Source review
  moved it after setup logging and into the selected execution branch before
  the end-to-end oracle run.

## Shared-owner live method panel

`results/mainline-methods-live-r3/samples.jsonl` contains eight warmups and forty
measured full-prefix forwards, retaining the method graphs together in one
process. This is the repaired ownership reproduction: the preceding r2 process
failed on its second method. The r3 condition is Metal, F32 activation policy
with bit-exact permissions, `MathMode::Safe`, no configured Omega policy, and
the dense-float tile extension off. It is not cached conversation latency or a
comparison with the configured C7 condition. France is a five-token numerical
fixture; ball uses the frozen 24-token prefix. Each cell has five samples and
sample CoV with the N-1 denominator; raw duration arrays remain in the summary.

| Prefix | Method | Mean forward ms | CoV% | Summary source |
|---|---|---:|---:|---|
| France5 | baseline | 2167.967267 | 0.639207 | `results/mainline-methods-live-r3/summary.json:47` |
| France5 | fixed logits, strength .5 | 2168.041859 | 0.240965 | `summary.json:158` |
| France5 | adaptive logits, cap 1 | 2219.730366 | 0.256463 | `summary.json:269` |
| France5 | fixed hidden, strength .5 | 2161.495634 | 0.100152 | `summary.json:380` |
| ball24 | baseline | 7121.910242 | 0.064311 | `summary.json:510` |
| ball24 | fixed logits, strength .5 | 7125.777142 | 0.023930 | `summary.json:640` |
| ball24 | adaptive logits, cap 1 | 7176.487042 | 0.058684 | `summary.json:770` |
| ball24 | fixed hidden, strength .5 | 7124.359909 | 0.068680 | `summary.json:900` |

The abbreviated summary paths in this table all refer to
`results/mainline-methods-live-r3/summary.json`. Its records label head/layer
execution counts as derived graph metadata rather than observed dispatches.
The adaptive cap in this timing panel is one; the independent paper comparison
uses the separately declared cap of .5.

## Shared-owner cached generation checkpoint

`results/owner-checkpoint-cached-on-r1/cached.jsonl` contains the new source
checkpoint after shared ownership: load/setup, two complete diagnostic vectors,
two warmups, ten measured generation requests and the completion footer. It
uses configured Metal, the enabled dense-float gate, tile floor 160,
batch/microbatch 256, context 320, F32 KV, four recurrent visits and greedy
selection. Profile capture is all nodes at step zero, as in C7. The five
measured requests in each row start with empty KV and retain every emitted ID,
token latency, phase event and resource observation.

| Prefix / actual output | Mean wall ms | Wall CoV% | Mean first-token ms | First-token CoV% | Source |
|---|---:|---:|---:|---:|---|
| ball24 / 24 tokens, EOS | 5879.053850 | 0.098613 | 2461.055383 | 0.123410 | `results/owner-checkpoint-cached-on-r1/cached.jsonl:9` |
| repeated-ball256 / 64 tokens, budget | 16287.704633 | 0.067111 | 5573.721983 | 0.149588 | `results/owner-checkpoint-cached-on-r1/cached.jsonl:17` |

These are observed durations with sample N-1 CoV, not an attributed performance
change relative to earlier cohorts. Every request records zero dropped profile
events and drains outside its timer; the footer records twelve requests and
zero errors. The complete emitted arrays retain the short response ending
`[1116,198,504,2988,314,365,50,25]` and the long synthetic repetition; all
positions remain in the request records. The original C7 gate pair is retained
unchanged for its controlled kernel comparison.

### Final lab helper checkpoint, r2

After the lab's compiler findings were repaired, the same configured cached
command ran again into `results/owner-checkpoint-cached-on-r2`. This is a new
source condition: named tap alias, helper argument grouping and iterator
rewrites are captured in `results/weight-owner/lab-*-clippy.diff`. The strict
lab clippy log has no diagnostics; the four caller and two recount cases execute
in `ouro-owner-lab-caller4-r3.txt` and `ouro-owner-lab-recount2.txt`.

The cached footer at line 18 records twelve requests and zero errors. All six
short-case trajectories end at the same EOS after 24 IDs; all six long-case
trajectories emit the same 64 IDs and stop at budget. Every request has zero
profile drops and records marker draining outside its timer. Setup remains
Metal, four loops, greedy, F32 KV, configured relaxed numeric policy, gate one,
tile floor 160, batch/ubatch 256, context 320, step-zero capture. Five measured
requests follow one excluded warmup per cell.

| Prefix / actual output | Mean wall ms | Wall CoV% | Mean first-token ms | First-token CoV% | Source |
|---|---:|---:|---:|---:|---|
| ball24 / 24 tokens, EOS | 5901.535808 | 0.150746 | 2461.350175 | 0.154676 | `results/owner-checkpoint-cached-on-r2/cached.jsonl:9` |
| repeated-ball256 / 64 tokens, budget | 16351.733817 | 0.164925 | 5587.418708 | 0.037361 | `results/owner-checkpoint-cached-on-r2/cached.jsonl:17` |

These durations and N-1 CoVs describe this repeated source checkpoint; the
earlier r1 and frozen C7 samples are preserved separately. No performance change
is attributed to the helper rewrites, and none of these requests is a
multi-stream serving measurement.

## Configured checkpoint C6: recorded microbatch limit

Both configured arms completed the same requests. The setup records show
`batch_size=32` and `ubatch_size=32`; llama used 256. The production chunk loop
in `proxima-model-interop/src/generate/decode.rs` splits on microbatch width.
These timed prefill chunks are below the 160-token dense tile threshold.
The separate full-prefix diagnostic does not show which kernels timed requests
executed. There is no dispatch capture in this cohort.

| Case | Dense-float gate | Mean wall ms | Wall CoV% | Mean first-token ms | Source |
|---|---|---:|---:|---:|---|
| repeated-ball256 | on | 37517.337525 | 0.3753333 | 26883.9315336 | `results/proxima-configured-on/cached.jsonl:17` |
| repeated-ball256 | off | 37345.7189168 | 0.1526966 | 26834.2269668 | `results/proxima-configured-off/cached.jsonl:17` |
| ball24 | on | 5824.31445 | 0.2458060 | 2442.2950916 | `results/proxima-configured-on/cached.jsonl:9` |
| ball24 | off | 5813.0320082 | 0.2842910 | 2442.3646 | `results/proxima-configured-off/cached.jsonl:9` |

Each cell contains five measured requests and one excluded warmup, with the
same pinned GGUF, configured Metal policy, four passes, F32 KV, greedy decode
and 64-token budget as above. Actual outputs are 24 tokens/EOS for ball24 and
64 tokens/budget for repeated-ball256. Statistics were independently recomputed
from request records using Welford sample variance; raw values and source lines
are in `results/cached-recount/timing-recount.jsonl`. These are observations of
this condition, not a tile-speed claim.

`results/cached-recount/coordinate-comparisons.jsonl` retains all signed
differences and paired trajectories. All twelve gate pairs have equal emitted
ID arrays; the twelve off/llama pairs also have equal arrays. The short initial
diagnostic has zero changed coordinates across gate arms, while the long
diagnostic has 48,757 changed coordinates of 49,152. At coordinate zero, the
long diagnostic is 12.74315071105957 off and 12.74316692352295 on. All 49,152
initial coordinates differ from llama for both cases; short coordinate zero is
5.995109558105469 off and 5.971109867095947 in llama. These comparisons do not
establish universal numerical equivalence from matching generated tokens.

The same comparator records an empty identical-vector difference and rejects
the deliberately changed coordinate `[1,2]` versus `[1,3]`. A duplicated off
arm is also rejected by the configuration check. C7 is a separately registered
condition with 256-token microbatches, an explicit context limit of 320 and
dispatch capture around actual generation requests.

## Independent C2 vectors after shared weight ownership

The Metal candidate and independent native F32 reference have 890,648
non-exact coordinates among 901,120 compared coordinates. The complete signed
records contain 485,283 positive and 405,365 negative differences. No coordinate
exceeds the admitted `0.003 + 0.003 * max(abs(actual), abs(reference))`
bound. These are bounded comparisons, not bit equality or reproduction of the
author's BF16 activation arithmetic. Sources: `results/weight-owner/c2-counts.json:1`,
the 26 rows in `results/weight-owner/c2-vector-ledger.jsonl`, and the complete
`results/c2-compare-owner-*-r1/differences.jsonl` payloads.

The eight cells cover France's five-token and ball's 24-token frozen prefixes,
four iterations, and baseline/fixed-logit/adaptive-logit/fixed-hidden methods;
all contrast strengths are .5. Their scalar files retain twelve comparisons
and their selection files retain eight ID/byte comparisons. France selects
7042 (` Paris`) in each method. Ball baseline selects 441 (` not`); each
contrast method selects 2382 (` red`). Each matches its independent reference.
The wrong-prefix control rejects with `matching paper candidate record absent`
in `results/weight-owner/ouro-c2-wrong-prefix-control-r1.txt:1`.

The largest signed witnesses are retained in both directions for every cell in
`results/weight-owner/final-artifact-audit.md`. For France adaptive guided logits,
coordinate 5 is actual -24.80272102355957 versus reference -24.801918029785156;
coordinate 7042 is actual 17.106033325195312 versus reference 17.105283737182617.
The adaptive scalar output and host-recomputed probability metadata are separate
observations; the scalar files retain their differences rather than treating
the metadata as device intermediates. No claim about general answer quality
follows from these two synthetic prefixes.

### Adaptive replay exposed numeric-policy bypass

The untimed `results/weight-owner/adaptive-graph-replay-r1.json:1` replays the
exact saved terminal logits through an ancestor-closed partition of the shared
production graph. It retains both prefixes, all nineteen operations, complete
CPU/Metal payloads and signed differences, strength-only/all-output controls,
a changed-input control, and a zero-cap graph rebuilt by the shared builder.

For France, strength-only Metal reproduces the saved C2 scalar
0.13502368330955505 exactly; requesting all graph outputs gives
0.1349496841430664. The all-output denominator is 1.3364638090133667 on Metal
and 1.336696982383728 on CPU. The host serial diagnostic gives strength
0.1349497139453888. For ball, strength-only Metal likewise reproduces the saved
0.4852037727832794, while all-output Metal gives 0.4852014183998108.
These are different requested-output conditions, not repeated timing samples.

The actual captured denominator-plus-division kernel is
`results/weight-owner/adaptive-graph-msl-r1/pipeline_09423c18f970ab52.metal:895`:
each lane folds every 256th element, then `simd_sum` and a partial-group merge
combine the results. The materialized denominator kernel at
`pipeline_847fe6d7d88a7778.metal:878` visits inputs serially. This difference in
fold order is reachable under the all-false numeric permission set because
`omega/src/msl/emit_and_classify.rs:1687` admits the broadcast-epilogue structure
without the ordinary `TreeReduce` check. `proxima-tensor/src/numeric.rs:31`
explicitly requires reassociation permission for such a fold.

This is a shared lowering policy bypass revealed by the paper check. Passing
the registered C2 numerical bound did not establish compliance with that
policy. The original replay, C2 comparisons and timings remain frozen evidence
of their source condition. The bounded repair has its own spec and checkpoint.

### Broadcast-fold repair: literal device checks

The shared Metal emitter now extracts the existing increasing-index fold into
`push_ordered_reduce_fold`. The ordinary serial renderer and the policy-denied
generic broadcast branch call that same helper. The latter runs the fold on
lane zero, publishes one threadgroup scalar, synchronizes all valid lanes and
uses the existing parallel broadcast write tail. Tiled, dense and packed
specialized returns precede this branch; expert-source mode is excluded.

The new emitter run selects three cases (three passed, 298 excluded), the
specialized control run selects three (three passed, 298 excluded), and the
device target selects five (five passed, none excluded). Full source/payload
logs are `results/omega-broadcast-{emitter,specialized,device}-r1.txt`.
These are selected-target counts, not a whole-workspace result.

The device log's cancellation fixture at line 9 retains both full 512-wide
rows. Strict fused and materialized outputs equal their registered literal
vectors. The first row's leading strict value is 16777216; the permitted tree
control produces 8388608 because the two small contributions survive a
different reduction order. The second row's leading value is 8388608 in both
paths, exercising distinct per-row scalars. The positive fixture at line 18
retains both 256-wide rows: each strict leading value is 1, while each permitted
tree leading value is printed as 0.9999925. These are exact Float32 vector
equality assertions against the literals, not a claim about every Float32
bit pattern or cross-backend bit equality.

Root reread the actual extraction/branch diff against the saved source and the
test assertions, then parsed every printed fixture coordinate. The generated
strict branch adds no SIMD fold after publication; its output-group guard is
uniform and before the barrier.

### Broadcast-fold repair: saved-input replay

`results/broadcast-fold-replay-r3.json:1` retains the actual frozen terminal
logits, every intermediate CPU/Metal vector and both requested-output sets for
France5 and ball24, depth4, cap0.5, Float32, all numeric permissions false.
This replay is untimed. Extracted scalar records live in
`results/broadcast-fold-replay-ledger-r3.jsonl:1` (France) and `:2` (ball).

| Frozen input | Metal strength-only | Metal all intermediates | CPU both output sets |
|---|---:|---:|---:|
| France5 | 0.1349496841430664 | 0.1349496841430664 | 0.13501334190368652 |
| ball24 | 0.4852014183998108 | 0.4852014183998108 | 0.48520341515541077 |

Each case retains nineteen node payloads, a maximum-logit mutation that changes
CPU strength, and a zero-cap graph returning zero. The CPU/Metal differences
remain; output-set equality is not cross-backend equality. Earlier replay r2
produced the same scalar observations but no kernel files: the capture writer
requires an existing directory. R3 creates that directory before the run and
retains the explicit instrument feature. R2 is preserved as numerical evidence
without source-capture attribution.

The emitted denominator-plus-divide kernel is
`results/broadcast-fold-msl-r3/pipeline_b1e949158ae47530.metal:882`, SHA256
`b1e949158ae475304e01da843c09f88e322484166ec9d20644e4ff8724f1e643`.
Root read its complete body: lane zero visits `r=0..reduction_total` in order,
publishes `ordered_fold`, and every valid lane synchronizes before the existing
stride-256 divide write. This is the changed arithmetic mechanism relative to
the frozen SIMD partial-merge source, while retaining the broadcast write
geometry.

### Broadcast-fold repair: strict full-prefix timing

Every recorded mean in this strict cohort is above its frozen live-r3 mean.
The condition is Metal, Ouro-1.4B, depth4, Float32 activation arithmetic, all
numeric permissions false, dense-float gate off, no Omega runtime config and
no KV reuse. These are full-prefix forward/readout durations, not cached
conversation latency. Each cell contains five measured requests after one
excluded warmup. CoV uses sample standard deviation.

| Prefix | Method | Frozen live-r3 mean ms | Frozen CoV% | Repaired live-r2 mean ms | Repaired CoV% |
|---|---|---:|---:|---:|---:|
| France5 | baseline4 | 2167.9672666 | 0.639207 | 2246.9800248 | 0.204034 |
| France5 | fixed logits 0.5 | 2168.0418586 | 0.240965 | 2252.0201170 | 0.207001 |
| France5 | adaptive logits cap1 | 2219.7303662 | 0.256463 | 2315.9123502 | 0.180243 |
| France5 | fixed hidden 0.5 | 2161.4956336 | 0.100152 | 2245.9181416 | 0.103488 |
| ball24 | baseline4 | 7121.9102418 | 0.064311 | 7203.4536332 | 0.055176 |
| ball24 | fixed logits 0.5 | 7125.7771416 | 0.023930 | 7209.5682836 | 0.034355 |
| ball24 | adaptive logits cap1 | 7176.4870418 | 0.058684 | 7276.5641000 | 0.083966 |
| ball24 | fixed hidden 0.5 | 7124.3599086 | 0.068680 | 7208.4046998 | 0.044116 |

Sources are `results/mainline-methods-live-r3/summary.json` and
`results/broadcast-fold-live-summary-ledger-r2.jsonl:1` through `:8`, which
preserve conditions, raw durations and original sample IDs. The arithmetic
mechanism is the ordered fold in the shared emitter; this also reaches generic
broadcast normalization within the forward graph. The timing comparison is
between retained source checkpoints, not an alternating kernel-only toggle,
so these observations do not isolate the entire duration difference to that
fold. The strict policy remains binding despite the higher observed means.

### Broadcast-fold repair: configured cached generation

The fresh cached checkpoint is `results/broadcast-fold-cached-r2/cached.jsonl`.
Its setup at line1 records Metal, GGUF BF16 storage, Float32 KV, depth4, greedy
sampling, batch/ubatch256, context320, dense-float gate on and the configured
`llama_relaxed` numeric permissions. Prompt reuse and speculative decoding are
disabled. Request profile drains occur outside the response timer; production
instrumentation remains enabled. This is a separate condition from strict live
forward evaluation.

| Prefix / emitted tokens | Mean response ms | Response CoV% | Mean first-token ms | First-token CoV% | Source |
|---|---:|---:|---:|---:|---|
| ball24 / 24, EOS | 5875.7635916 | 0.140898 | 2458.9336916 | 0.136402 | `cached.jsonl:9` |
| repeated-ball256 / 64, budget | 16205.2848418 | 0.138653 | 5577.7028416 | 0.166113 | `cached.jsonl:17` |

Each cell has five measured requests and one excluded warmup. The footer at
line18 records twelve completed requests and zero errors. Root read all twelve
generated-ID vectors, EOS flags, raw wall/first-token durations and profile-drop
fields. Every short request has the same complete 24-ID vector and every long
request the same complete 64-ID vector; all report zero dropped profile events.
The longer synthetic output repeats the input construction; this is retained
payload evidence, not an answer-quality claim. Pooled decode samples cover
different token positions and are not repeated-request CoV.

These source-checkpoint timings are not a causal speedup claim for the strict
arithmetic repair. The relaxed path permits the incumbent tree reduction.
The earlier owner-r2 cached measurements remain separately labeled above, and
the pinned llama comparison remains the separate matched C7 cohort below.

### Broadcast-fold repair: fresh independent C2 comparison

The fresh candidate remains numerically non-exact against the independent F32
reference: 890894 of 901120 coordinates differ, comprising 438241 positive and
452653 negative differences. The other 10226 coordinates are exact. These are
the new `c2-compare-broadcast-*-r2` records; the earlier owner cohort's 890648
non-exact coordinates remain a distinct source condition.

Root read all twenty-six vector comparison records, twelve scalar comparison
records and eight selection records from the eight fresh case/method cells.
Each vector record reports zero nonfinite coordinates and zero exceedances of
the unchanged `0.003 + 0.003 * max(abs(actual), abs(reference))` bound. All
scalar records are within that same rule and all eight selected IDs/byte
sequences match their reference. These counts do not establish bit parity or
answer quality beyond the frozen synthetic prefixes.

The actual France adaptive scalar is 0.13495200872421265, versus the independent
reference's 0.1349518597126007 (`c2-compare-broadcast-france-adaptive-logits-r2/`
`scalar-comparisons.jsonl:1`). Ball is 0.4852010905742645 versus
0.48520249128341675 in the corresponding ball file. These are fresh full-forward
outputs, not the frozen terminal-logit inputs used by replay r3. The shared
normalization folds also changed, so the two experiments must not be treated
as having identical terminal inputs.

France selects ID7042 / bytes ` Paris` in all four methods. Ball selects
ID441 / bytes ` not` for baseline, and ID2382 / bytes ` red` for each contrast
method. Every comparison retains the candidate line271 and exact independent
reference location, with full signed coordinate rows in `differences.jsonl`.
The paper candidate binary was frozen for these runs and compared to the same
independent oracle/reference artifacts used before the repair.

The final independent artifact audit is
`results/broadcast-fold-final-audit.md`. Its native extractions retain one
positive and one negative actual/reference coordinate witness for each of the
26 fresh C2 vectors (`broadcast-fold-c2-witnesses-r2.json`). The separate strict
live join retains 5996544 coordinates: 2420196 positive, 3511926 negative and
64422 exact, with bidirectional witnesses grouped by case/method/output field
in `broadcast-fold-live-witnesses-bidirectional-r2.json`. These are separate
comparisons: fresh C2 uses the frozen independent arithmetic reference; strict
live uses the earlier shared-owner source checkpoint. The current cached
profile ledger is `broadcast-fold-cached-pairs-r2.json`, whose twelve records
retain equal begin/end request fields and ordered timestamps.

## C7: matched batch/context, captured production dispatch

The native oracle remains below Proxima's recorded response durations. This
cohort uses the same checkpoint, frozen prefixes, greedy requests and output
counts as C6, with batch/ubatch 256 and context 320 in both runtimes. Both Proxima
arms enable identical step-zero dispatch capture; llama does not have that
instrumentation. These are recorded measurements, not a general speed verdict
or an uninstrumented cross-runtime ratio. Five measured requests follow one
excluded warmup in each cell; sample CoV uses N-1 variance.

| Case | Runtime/condition | Mean response wall ms | Wall CoV% | Mean first-token ms | Source |
|---|---|---:|---:|---:|---|
| repeated-ball256 | Proxima gate on, instrumented | 16401.66715 | 0.2448765 | 5574.7170166 | `results/proxima-matched-batch-on-r3/cached.jsonl:17` |
| repeated-ball256 | pinned llama, uninstrumented | 2838.8440332 | 1.4032889 | 412.8548666 | `results/llama-matched-batch-r3/llama.jsonl:18` |
| repeated-ball256 | Proxima gate off, instrumented | 37987.5544664 | 0.0697712 | 27340.3244664 | `results/proxima-matched-batch-off-r3/cached.jsonl:17` |
| ball24 | Proxima gate on, instrumented | 5888.4711668 | 0.2554215 | 2461.4119334 | `results/proxima-matched-batch-on-r3/cached.jsonl:9` |
| ball24 | pinned llama, uninstrumented | 1001.1678834 | 0.4717571 | 104.8375668 | `results/llama-matched-batch-r3/llama.jsonl:10` |
| ball24 | Proxima gate off, instrumented | 5889.6048334 | 0.4388880 | 2460.1295832 | `results/proxima-matched-batch-off-r3/cached.jsonl:9` |

The response improvement occurs before the first token. Pooled subsequent-token
latencies for repeated-ball256 are 168.0912737 ms off and 170.9063356 ms on;
llama records 38.5077540 ms. These averages combine different token positions
within the five requests and are not repeated-sample uncertainty estimates.
The short-case first-token average is slightly higher with the gate on, as shown
above; neither that cell nor decode exhibits the long-prefill reduction.

### Exact dispatch mechanism

The repeated-ball256 warmup's BF16 feed-forward projection has node 116,
extents `[256,2048,5632]`, activation operand 111 and BF16 weight operand 112
in each arm. The actual records are at off `gpu-events.jsonl:90054` and on
`gpu-events.jsonl:90066`; their following dispatch rows carry step zero and
the complete source digest.

- Off: grid 369098752, width 256, SHA
  `a07d5d715b8c0579f2bf1ce6ce631acfb76769bdcd29eec383c14256ff55b22d`.
  The captured source performs BF16 element loads, scalar accumulation and
  cooperative SIMD reduction.
- On: grid 90112, width 128, SHA
  `74eada85477b2aa77cd1e2bac1562a15b9fe6d72bd6c94b829ce1f9b3513ade8`.
  The captured source's lines 884–930 stage weight/activation tiles in
  threadgroup memory and use `simdgroup_float8x8` fragments with
  `simdgroup_multiply_accumulate`.

The files are under `results/cac7-pipeline-{off,on}-r3`, named with their
digest prefixes. Other operations use matrix instructions even in the off arm;
their presence somewhere in generated source would not establish this change.
The comparison above identifies the same operation's actual request dispatch.

Initial capture attempts remain separate: `packed-multi-token` excludes BF16
through its packed-codec classifier. The missing-feature explanation was
retracted after inspecting the dependency feature graph. The completed pair
uses `PROXIMA_CAPTURE_NODES=all`, `PROXIMA_CAPTURE_STEPS=0`; no buffer dump is
required. Exact commands and source manifests are in
`results/cached-checkpoint/COMMANDS.md`.
