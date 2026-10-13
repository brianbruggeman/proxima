# policy-preserving broadcast fold

status: admitted; registered checks executed
owner: execution worker
created: 2026-10-03

## problem

The existing generic Metal broadcast-reduce emitter changes an Add fold's order
under a policy that withholds reassociation; requesting intermediate roots must
not grant that missing permission.

## refutation condition

Refute this repair if the literal ordered-fold outputs differ, the cooperative
write geometry changes, the permitted tree path disappears, a specialized
packed/tiled/expert arm changes, either saved-input strict Metal strength changes
with requested output set, or the real-model checkpoint fails its registered
payload/count checks. A latency increase is retained as evidence rather than
used to relax the arithmetic requirement.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | A generic broadcast fold denied `NumericRewrite::TreeReduce` uses exactly the ordinary serial renderer's ordered reduction loop, seed, operand reads and body rounding. | yes |
| R2 | Preserve existing cooperative output coordinate/width selection and parallel broadcast epilogue writes. Publish one completed ordered scalar per output threadgroup with a barrier before its lanes read it. | yes |
| R3 | A policy granting TreeReduce retains the current lane-strided tree. Specialized packed/tiled/expert paths retain their existing emission. | yes |
| R4 | Both frozen adaptive inputs yield exactly equal strength-only/all-taps Metal scalar roots under the strict policy. Preserve full intermediates and CPU/host differences without widening any bound. | yes |
| R5 | Each product checkpoint runs fresh strict four-arm live48, relaxed configured cached12 and all eight native C2 cells against the frozen references, retaining failures, signed payload differences, raw durations and CoV. | yes |
| R6 | Preserve alloc-only emitter compilation and strict production lint boundaries; add no API, config field, runtime allocation, driver buffer or lock. | yes |

## architecture

Owned product files are `omega/src/msl/elementwise_reduce_core.rs` and
`omega/src/msl/tiled_gemm_cooperative_scan.rs`; the dispatch classifier gets a
comment correction if needed, with its routing unchanged. Owned assertions are
`omega/src/msl/tests.rs` and `omega/tests/reduce_epilogue_fusion_parity.rs`.

Extract the existing seed/ordered loop from `push_serial_reduce_body` into a
private emission helper `push_ordered_reduce_fold(source, resolved, quantized,
indent) -> Result<(), EmitError>`. The helper derives axes, body, init and dtype
from the existing BoundOp, and reuses `fold_init_tokens`, operand/gather readers,
coordinate decomposition, `push_body_steps`, and `scalar_op_expr`. The serial
renderer composes that helper with its current preamble and output tail.

In `push_cooperative_reduce_body`, after existing specialized early returns and
the existing cooperative preamble, generic broadcast folds with TreeReduce
denied emit:

1. one `threadgroup` scalar;
2. lane zero's original ordered fold, using the extracted helper;
3. lane zero publishes the completed accumulator;
4. all lanes reach `threadgroup_barrier(mem_flags::mem_threadgroup)`;
5. all lanes load that scalar and reuse `push_broadcast_epilogue_write`.

No lane exits between publication and barrier. The existing pinned width equals
the output's cooperative width (`tiled_gemm_threadgroup_width`), so each scalar
belongs to one output row; a two-row device fixture checks this ownership.
Do not run a SIMD sum over zeros plus the completed scalar: that would introduce
an additional arithmetic operation, including signed-zero handling.

The generic branch applies only to broadcast epilogues without expert-source
substitution; specialized early returns and the existing expert path stay in
place. Other generic folds continue their current classifier route. No caller
or tensor graph changes are required. The new branch repairs an existing
permission contract; it is not a new experimental optimization.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| fold order | existing ordered loop reused by both renderers | a second copied arithmetic loop would drift |
| write ownership | existing parallel broadcast tail, one shared scalar per output group | globally disabling fusion changes unrelated lowering and materialization |
| policy | existing `admit(policy, TreeReduce)` | no Ouro-specific rule or new policy/config type |
| strict structure | lane-zero fold inside cooperative geometry | ordinary serial renderer has no broadcast output tail |
| allocation | one per-group scalar in generated MSL, no device allocation | no new scratch arena or driver resource consumer |

Guiding principles rejected global fusion disablement, a second softmax formula,
and a new reduction IR/type. The alloc emitter remains the source owner; native
GPU dispatch remains its existing consumer. Cold String emission may allocate,
as it already does; no zero-allocation claim is made for compilation.

### worked literals before source edits

Cancellation row has width 512. All entries are zero except indices
`0=16777216`, `1=1`, `256=-16777216`, `257=1`.
Float32 ordered sum: `0 + 16777216 = 16777216`; adding one rounds back to
16777216; zeros preserve it; adding -16777216 yields zero; the last one yields
one. Divide the row by its sum: the expected 512 outputs equal the original
row exactly. A second row has `0=16777216`, `1=-16777216`, `2=1`, `3=1`;
its ordered sum is two and expected outputs are the row divided by two.
The second row prevents a shared-scalar alias from satisfying the assertion.

For the first row, the current width-32 or width-256 lane walk cancels the large
terms on lane zero and collects both ones on lane one: its tree sum is two.
That is the negative control; strict output at index zero would be 8388608
instead of the required 16777216.

Positive row has width 256, `0=1` and indices 1 through 255 equal `2^-25`.
Each small addition is below half an ULP at one, so the ordered sum stays one.
Its expected normalized row is unchanged. A second row doubles every entry;
ordered sum is two and expected normalized outputs equal the first row.
The permitted tree retains small terms, so its first normalized value must
differ from one; preserve that actual relaxed result without using it as the
strict expected vector.

### pseudocode and walk

For strict generic broadcast: compute output coordinates; lane zero seeds the
existing accumulator; visit flattened reduction positions from zero to N-1;
read operands and evaluate the existing composed body once per position; combine
in that order; publish accumulator; synchronize; each lane walks its existing
output positions and evaluates the epilogue with the published scalar.
For the cancellation rows this publishes one and two respectively. Their
parallel divide writes then produce the two hand-derived rows above.
For the positive rows it publishes one and two, producing identical normalized
rows. Tree admission preserves the incumbent loop and combine.

## acceptance criteria

Commands run natively from the main repository, target
`/private/tmp/proxima-ouro-mainline-target`; GPU runs serialize with the other
worker. `NATIVE_CONFIG` below denotes the literal existing path
`proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml`.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1,R2,R3 | `cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-omega --features 'reduce-epilogue-fusion metal-wide-cooperative-reduce metal-tiled-gemm' --lib -E 'test(broadcast_fold_policy_)' --no-capture --target-dir /private/tmp/proxima-ouro-mainline-target`, then specialized command below | 3 new cases plus3 existing specialized controls; report exclusions separately |
| AC2 | R1,R2,R3 | `cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-omega --features 'reduce-epilogue-fusion metal-wide-cooperative-reduce metal-tiled-gemm' --test reduce_epilogue_fusion_parity --no-capture --target-dir /private/tmp/proxima-ouro-mainline-target` | 5 selected cases: existing3 plus `broadcast_fold_policy_cancellation_rows` and `broadcast_fold_policy_positive_rows`; full two-row literal assertions, fused/all-roots strict equality and relaxed negative controls |
| AC3 | R4 | native diagnostic command below, followed by native jq checks | 2 cases, 19 nodes per case, 2 exact strict Metal output-set strength equalities, 4 mutation/cap controls; full vectors retained |
| AC4 | R5 | cached command below | 12 requests (10 measured,2 warmup),2 diagnostics,2 summaries,1 completion with0 errors; all12 full IDs/EOS and durations/drop fields retained; 12 ordered profile pairs and25 source hashes unchanged across run |
| AC5 | R5 | eight candidate and eight comparison invocations registered in `BROADCAST-FOLD-C2-COMMANDS.md` | 8 selected records +8 comparisons,26 vectors,901120 coordinates,12 scalars,8 selections; same abs/rel .003, no new tolerance; signed differences and failures retained |
| AC6 | R6 | `cargo check -p proxima-omega --no-default-features --features alloc --lib --message-format json --target-dir /private/tmp/proxima-ouro-mainline-target` | 1 omega library artifact,0 error diagnostics, build-finished true |
| AC7 | R6 | `cargo clippy -p proxima-omega --features 'reduce-epilogue-fusion metal-wide-cooperative-reduce metal-tiled-gemm' --lib --target-dir /private/tmp/proxima-ouro-mainline-target -- -D warnings` | 0 production diagnostics; all-target compile/lint runs retain command-line test expectation exceptions and0 other diagnostics |
| AC8 | R5 | strict live command below | 48 request records (40 measured,8 warmups),8 timing cells; all4 retained graph arms, raw vectors/counts/durations/CoV and signed changes against immutable live-r3, no CPU fallback |

AC1 names: `broadcast_fold_policy_ordered_structure`,
`broadcast_fold_policy_permitted_tree`, `broadcast_fold_policy_serial_helper`.
Source assertions cover no simd_sum in the strict generic branch, seed/order,
shared publish/barrier, unchanged width/grid, broadcast write loop, and existing
serial-source loop. The permitted path retains simd_sum.

R3 additionally runs these three existing assertions: Q8 packed decoder,
Q4K tiled multiply/staging/bounds, and mixed expert descriptor codec/hoisting.
Their source assertions remain unchanged; retain full emitted failure payloads.
```text
cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-omega --features 'reduce-epilogue-fusion metal-wide-cooperative-reduce metal-tiled-gemm' --lib -E 'test(push_packed_row_blocked_body_emits_a_q8_0_row_blocked_kernel) | test(many_token_matmul_takes_the_tiled_gemm_path) | test(expert_source_uses_descriptor_codec_instead_of_checkpoint_codec)' --no-capture --target-dir /private/tmp/proxima-ouro-mainline-target
cargo clippy -p proxima-omega --features 'reduce-epilogue-fusion metal-wide-cooperative-reduce metal-tiled-gemm' --all-targets --target-dir /private/tmp/proxima-ouro-mainline-target -- -D warnings -A clippy::expect_used -A clippy::unwrap_used
```
Expected specialized count3; all-target diagnostics0 with only the user's
test expectation/unwrap command-line exceptions. Strict lib command stays intact.

AC4/AC8 before/after25-row source manifests, native cmp commands, keyed48-request
baseline comparison,168 complete vector differences/5996544 coordinates, and
8 independently recounted raw timing/CoV cells are executable in
`BROADCAST-FOLD-CHECKPOINT-COMMANDS.md`. Run those bracketing commands around the
two timed invocations below. The immutable baseline census was read before edits.

AC3:
```text
mkdir -p proxima-tensor/specs/ouro-mainline/results/broadcast-fold-msl-r3
env PROXIMA_PIPELINE_CAPTURE=proxima-tensor/specs/ouro-mainline/results/broadcast-fold-msl-r3 cargo run --release --example ouro_adaptive_graph_diagnostic --features 'proxima-tensor omega/instrument' --target-dir /private/tmp/proxima-ouro-mainline-target -- /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/models/ouro-1.4b proxima-tensor/specs/ouro-mainline/results/c2-owner-france-adaptive-logits-r1/records.jsonl proxima-tensor/specs/ouro-mainline/results/c2-owner-ball-adaptive-logits-r1/records.jsonl proxima-tensor/specs/ouro-mainline/results/broadcast-fold-replay-r3.json
jq -e '{cases:(.records|length), nodes:[.records[].operations|length], equal_strength_cases:([.records[]|select(.metal_strength_only==.metal_strength_all_nodes_requested)]|length), controls:([.records[]|select(.negative_control.cpu_strength_delta!=0 and .negative_control.zero_cap.cpu_strength==0)]|length)} | .cases==2 and .nodes==[19,19] and .equal_strength_cases==2 and .controls==2' proxima-tensor/specs/ouro-mainline/results/broadcast-fold-replay-r3.json
```

AC4 (the runner owns the fresh output leaf):
```text
mkdir -p proxima-tensor/specs/ouro-mainline/results/broadcast-fold-cached-pipeline-r2
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_PIPELINE_CAPTURE=proxima-tensor/specs/ouro-mainline/results/broadcast-fold-cached-pipeline-r2 cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm omega/instrument' --target-dir /private/tmp/proxima-ouro-mainline-target -- --cached-performance --gpu-profile --backend metal --samples 5 --max-new-tokens 64 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/ouro-mainline/results/broadcast-fold-cached-r2
```

AC8 is a separate strict/gate-zero source checkpoint, compared only with
`results/mainline-methods-live-r3` under that same strict uncached condition:
```text
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 cargo run --release --example ouro_loopcd_lab --features proxima-tensor --target-dir /private/tmp/proxima-ouro-mainline-target -- --performance --backend metal --samples 5 --output proxima-tensor/specs/ouro-mainline/results/broadcast-fold-live-r2
```

The source manifest expands the previous sixteen files by all three lowering
owners, two repair test owners, canonical numeric policy, diagnostic source and two
additional test-only compile repairs (qk_norm initializer and config initializer).
Exactly25 rows must compare unchanged before/between/after timed runs:
```text
shasum -a 256 proxima-model-interop/src/generate/model_weights.rs proxima-model-interop/src/generate/load_model.rs proxima-model-interop/src/generate/pregather.rs proxima-model-interop/src/generate/decode.rs proxima-model-interop/src/generate/residency_caches.rs proxima-model-interop/src/generate/mod.rs proxima-model-interop/src/ouro.rs proxima-model-interop/src/hf_bind.rs examples/ouro_loopcd_lab.rs examples/ouro_loopcd_lab/model.rs examples/ouro_loopcd_lab/engine.rs examples/ouro_loopcd_lab/panel.rs examples/ouro_loopcd_lab/performance.rs proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/lfm2_single_range_cached.rs omega/src/backend.rs omega/src/msl/elementwise_reduce_core.rs omega/src/msl/tiled_gemm_cooperative_scan.rs omega/src/msl/emit_and_classify.rs omega/src/msl/tests.rs omega/tests/reduce_epilogue_fusion_parity.rs proxima-tensor/src/numeric.rs examples/ouro_adaptive_graph_diagnostic.rs omega/tests/gemma4_rows_support/mod.rs omega/tests/programmable_execution_vectors.rs
```

## out of scope

New softmax formulas, Ouro graph changes, new runtime/backend/config/IR, changed
independent oracle formulas, relaxed C2 thresholds, and scheduler changes.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| ordered norm folds increase strict latency | measurable | changed real-model wall times | retain strict live48 raw records against strict live-r3; keep relaxed cached12 separate; no latency verdict |
| a barrier sees only part of an output group | source-sensitive | wrong scalar or deadlock | unchanged pinned group width, uniform group guard, two-row real device assertions |
| fold helper changes ordinary scalar rounding | source-sensitive | generic reductions change | derive expected literals first, compare serial emitted source structure and real controls |
| instrumentation materializes and changes arithmetic | observed | false explanation | strict strength-only/all-node equality assertion plus full MSL sources |

## context

Read guiding-principles, algorithm-development, spec-first; local bootstrap,
index/routes/invariants and `ai_docs/{omega-programmable,ouro-mainline}.jsonl`.
The policy and permission gate are `proxima-tensor/src/numeric.rs` and
`omega/src/msl/emit_and_classify.rs:1640`. The broadcast bypass is at1670.
`push_serial_reduce_body` is at `elementwise_reduce_core.rs:518`;
cooperative body/tail/write helpers are in `tiled_gemm_cooperative_scan.rs`.
Original replay/source artifacts remain under `results/weight-owner/`.

## execution evidence

`results/BROADCAST-FOLD-COMMANDS.md` binds the actual logs and retained failures.
Selected cases:3 new emitter,3 specialized controls and5 device cases. The
alloc-only check emitted1 library artifact/0 errors; strict library and
all-target lint reruns emitted0 diagnostics (test-only command-line exceptions).
Replay-r3 retained2 cases/38 operations/4 controls and2 exact strict Metal
output-set scalar equalities, with actual captured denominator/write source.
Strict live-r2 retained48 requests/168 numeric vectors/5996544 coordinate joins
and8 raw timing cells against immutable live-r3; all8 observed means increased.
Relaxed cached-r2 retained12 requests/12 matched profile pairs/0 recorded errors.
All25 source rows were identical before/after each timed run and between runs.
Fresh C2-r2 retained8 cells/26 vectors/901120 coordinates/12 scalars/8 selections;
890894 coordinates are non-exact,0 exceed the unchanged absolute/relative .003
bounds. Wrong-prefix control refused with one missing-match error.
These observations do not assert cross-backend byte equality or attribute the
whole strict wall-time change to one mechanism.
