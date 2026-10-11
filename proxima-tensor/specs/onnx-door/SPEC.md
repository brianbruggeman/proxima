# onnx-door

status: audited
owner: Brian Bruggeman
created: 2026-10-11

Owner direction, 2026-10-11, verbatim: "we really fucking need omega to include onnx or something so it's present and viable." Frame: one generic IR with many doors; every door is a proxima pipe with an FSM over its stages; generic (no per-model code); meet or beat the incumbent per door and cell.

## problem

omega cannot run the BAAI/bge-small-en-v1.5 ONNX model (`onnx/model.onnx` at Hugging Face revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 133093490 bytes, sha256 `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`) through one ONNX door pipe on `Engine::Cpu` and on `Engine::Gpu` (Metal), at batch 1 and batch 32 over a 64-sentence corpus, with every L2-normalized CLS embedding within max abs 1e-5 (CPU) and 1e-3 (Metal) of the output tract-onnx 0.22.4 recorded on the same input tensors, and with the median time per sentence at or below tract's recorded median plus max(MAD, 2%) in each of the four engine-by-batch cells, measured by the `onnx_door_bench` example under the parity rounds' bench discipline (interleaved arms, box-load refusal, CoV, resource columns, bound lines).

## refutation condition

Any one of these refutes the door as present and viable:

1. a sentence of the 64 whose embedding exceeds the stated tolerance against tract's record, at batch 1 or batch 32, on CPU or Metal;
2. a `bound metric=ms_per_sentence` line with `within=false` in any of the four cells, or a `bound metric=rss_peak_mb` line against the 500 MB cap with `within=false`;
3. mnist.onnx, a convolutional model of another family, needing any model-specific branch in the door (AC13 counts model names in the door sources and the classification accuracy);
4. tract's recorded cosines on the three bge_eval sentences disagreeing with the onnxruntime cosines already recorded in `proxima-tensor/docs/discipline.md:17482` by more than 1e-4: the oracle would then be wrong and every parity row void;
5. the census (R1) reporting an unlowerable op in either model, which would put the lowering (R2) back on the critical path.

Oracles, none sharing code with the path under test: tract-onnx 0.22.4 as an incumbent BINARY in the indexer repo (its output recorded once as files, never re-run inside a proxima gate, like the llama numbers); onnxruntime's three recorded cosines (`proxima-tensor/docs/discipline.md:17482`); the real MNIST `t10k` labels; the ONNX operator definitions in `docs/Operators.md` of github.com/onnx/onnx (hand-derived worked examples); the hand-computed (`bc -l`) two-layer-MLP softmax values already used at `omega/tests/onnx_metal_parity.rs`. Python is never run; tract runs only as the indexer repo's Rust binary.

Controls, by kind of check. A comparison against a recorded or vendored external file carries an identity control (the record compared to itself through the same comparison function passes) and a degenerate control (a copy altered in one place fails). A comparison against a hand-derived value carries a degenerate control (the expected value altered fails). A structural assertion against a constant stated in its own AC (a count, a shape, a file name) carries none. Each control is a counted test case inside its AC.

## requirements

Each R is one consumer-visible behaviour; its single AC enumerates the named test cases that exercise it. The Rs fail independently of one another: each has a different producer module, a different recorded file, or a different measured quantity.

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Op coverage census as a measurement: `proxima_onnx::census` counts the nodes and distinct op types of a graph and names every op type `lower_node` cannot lower; `SUPPORTED_OPS` equals the arms of `lower_node`; the `onnx_census` example prints `census nodes=<n> distinct=<d> unlowerable=<u>` for a model, and prints the unlowerable names | yes |
| R2 | The four standard-domain ops BERT-family exporters emit that `lower_node` lacks (`LayerNormalization`, `Gelu`, `Squeeze`, `Expand`) lower as compositions of the existing op algebra, with no new `Op` form or `ScalarOp`, and `docs/op_coverage.md` is regenerated | yes |
| R3 | The ONNX door is a `Pipe` (`In = DoorBatch`, `Out = DoorOutput`, `Err = DoorError`) over an enum FSM `DoorStage` of parse, validate, lower, plan, execute; a failure names the stage that raised it and carries the stage's own error; the door composes behind another pipe with `AndThen` | yes |
| R4 | `onnx_door_walkthrough` drives the FSM through every legal transition and every stage's failure exit and prints one line per transition (principle 11) | yes |
| R5 | The door keys plans by the pin set: a repeated pin set performs zero lowerings and zero plan builds, a new pin set exactly one of each, initializer bytes are decoded once per door regardless of pin sets, and the number of retained pin sets is capped by a sizing-TOML constant with evictions counted | yes |
| R6 | The real-world corpus is on disk and pinned: the model file at the recorded revision has the recorded length and sha256, a 4096-byte header and the hash are vendored, and `corpus.jsonl` holds the 64 sentences with their WordPiece ids | yes |
| R7 | tract's output is recorded once: an indexer-repo harness writes 64 batch-1 and 2 batch-32 raw CLS embeddings with their input tensors and a per-pass timing file, and the proxima side validates the files | yes |
| R8 | CPU parity: the door on `Engine::Cpu` reproduces tract's recorded embeddings within 1e-5 max abs per sentence at batch 1 and batch 32, with the identity, degenerate and mask-leak controls | yes |
| R9 | Metal parity: the door on `Engine::Gpu` with `GpuDriver::Metal` reproduces tract's recorded embeddings within 1e-3 max abs per sentence at batch 1 and batch 32, with the same controls and a proof the GPU ran | yes |
| R10 | The bench cell prints, for tract (recorded), omega CPU and omega Metal at batch 1 and batch 32, wall, ms per sentence, CPU %, peak and steady RSS, GPU bytes and CoV in one table | yes |
| R11 | Time bound: in each of the four omega-versus-tract cells the median ms per sentence is within the decode_arms bound rule of tract's recorded median | yes |
| R12 | Memory bound: peak RSS of each of the four omega cells is within the 500 MB cap of `AGENTS.md` | yes |
| R13 | Generic-ness: mnist.onnx (a convolutional model of another family) runs through the same door constructor on CPU and Metal with top-1 accuracy at or above 0.98 over the first 1000 `t10k` images, and the door sources name no model | yes |
| R14 | No tract and no onnxruntime in proxima: no crate named `tract*`, `ort`, `ort-sys` or `onnxruntime*` appears in proxima's `Cargo.lock` or `fuzz/Cargo.lock`, while the same check finds them in the indexer repo's lock | yes |
| R15 | The `fuzz_onnx` target still reaches no panic after the lowering changes, and gains a sweep that chains `parse_complete` into the census | yes |

## architecture

```
proxima-onnx/src/census.rs                                    // new: Census { nodes, op_counts, unlowerable }, census(&GraphProto), render_census
proxima-onnx/src/lower.rs                                     // SUPPORTED_OPS const beside lower_node (lower.rs:465); four new arms; lower_layer_normalization, lower_gelu, lower_squeeze, lower_expand
proxima-onnx/examples/onnx_census.rs                          // new: prints the census line of a model file
proxima-onnx/docs/op_coverage.md                              // regenerated by examples/generate_op_coverage_doc.rs
omega/Cargo.toml                                              // feature onnx-door = ["std", "cpu", "dep:proxima-onnx", "dep:bon"]; proxima-onnx becomes an optional regular dependency by cargo add (it is a dev-dependency at Cargo.toml:673)
omega/src/onnx_door/mod.rs                                    // new: OnnxDoor, the Pipe impl
omega/src/onnx_door/stage.rs                                  // new: DoorStage enum and advance
omega/src/onnx_door/error.rs                                  // new: DoorError { stage, source }
omega/src/onnx_door/config.rs                                 // new: DoorConfig (bon builder + conflaguration Settings)
omega/src/onnx_door/cache.rs                                  // new: pin-set keyed plan cache, counters, eviction
omega/omega-runtime.toml                                      // new section [onnx_door] max_pin_sets
omega/examples/onnx_door_walkthrough.rs                       // new: R4
omega/examples/onnx_door_bench.rs                             // new: R10 to R12, shares cell.rs by #[path]
proxima-model-interop/examples/cell_resources/cell.rs         // gains rss_steady_mb (resident size after the last pass)
omega/tests/onnx_door.rs                                      // new: R3, R5 on synthetic graphs
omega/tests/onnx_door_bge.rs                                  // new: R6 to R9 on the real model
omega/tests/onnx_door_generic.rs                              // new: R13
omega/tests/support/bge.rs                                    // new: CLS pooling, L2 normalization, tolerance compare, record readers
omega/tests/fixtures/bge-small/                               // new: README.md, model.sha256, model_header.bin, corpus.jsonl, tract_b1.jsonl, tract_b32.jsonl, tract_timing.jsonl
fuzz/src/bin/fuzz_onnx.rs                                     // third sweep: parse_complete into census
<indexer crate>/examples/corpus_build.rs, tract_record.rs     // new, in the indexer repo only: tokenizes the corpus with tokenizer.json, runs tract
```

### the door

`OnnxDoor::from_bytes(bytes, DoorConfig)` holds the model bytes and a config; `Pipe::call(&self, DoorBatch)` returns `DoorOutput { evaluated, outputs }`. `DoorBatch` is `{ pins: BTreeMap<String, u64>, inputs: Vec<(String, Vec<f32>)> }`: the pins bind the symbolic dimensions of the graph inputs (`lower_graph_pinned`, `lower.rs:243`), the inputs are the per-call tensors by graph input name. Integer graph inputs (bge declares int64 ids) are fed as `f32` carrying the exact integer, as `bge_eval.rs:124` does. `DoorOutput.outputs` is `Lowered::graph_outputs` (`lower.rs:168`). `DoorConfig` carries `engine: Engine`, `gpu_driver: Option<GpuDriver>`, `numeric_policy`, `max_pin_sets` and is a `bon` builder that also loads through conflaguration (principle 4).

The FSM is one enum, each variant carrying only the data legal in its state, transitions consuming the old value:

| state | holds | entered by | leaves by |
|---|---|---|---|
| `Parsed` | `ModelProto` borrowed from the door's bytes (`pipe.rs:23`) | `parse_complete` | validate, or `DoorError { stage: Parse }` |
| `Validated` | the graph and its `Census` | census has no unlowerable op, every symbolic input dimension is in the pins | lower, or `DoorError { stage: Validate }` naming every unlowerable op and every unpinned dimension |
| `Lowered` | `Lowered` for this pin set | `lower_graph_pinned` | plan, or `DoorError { stage: Lower }` |
| `Planned` | `Lowered` and the omega `Plan` | `plan_named` (`backend.rs:405`) | execute, or `DoorError { stage: Plan }` |
| `Executed` | the plan and the `Evaluated` | `execute_plan_named` (`backend.rs:480`) | back to `Planned` for the next batch with the same pins, or `DoorError { stage: Execute }` |

A pin set already seen starts at `Planned`; a new one starts at `Parsed`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| a door type at all | `OnnxDoor` as a `Pipe` | answered by writing the call site both ways. Before: `parse_complete`, `lower_graph_pinned`, `plan_named`, `execute_plan_named` called by hand, four error types, the caller owning pins and a cache (`examples/support/wire_to_weights_pipeline.rs` and `bge_eval.rs:238` each hand-wrap them). After: `door.call(batch)`. The lines differ, and a caller can now do three things it could not: learn which stage failed from one error type (rich-to-poor information loss today), reuse plans by pin set, and compose the door behind a tokenizer pipe with `AndThen` (AC3). `backend.rs:56-66` adjudicated that wrapping `plan_named`/`execute_plan_named` alone is a relocation; this door adds the stages around them, which that adjudication did not cover |
| `Pipe`, not `SendPipe` | `Pipe` | `Plan` is `!Sync` (`backend.rs:56-66`); a `SendPipe` bound cannot be met without a lock |
| FSM as one enum | `DoorStage` with consuming `advance` | principle 11: typestate parameters cannot express the pin-set loop `Executed -> Planned`; exhaustive `match` can |
| census inside the validate stage | yes | the census is the same information the validate stage needs to name every unlowerable op; a second walk would be a second source of the op list |
| `SUPPORTED_OPS` as a const | a const checked equal to `parse_supported_ops(lower.rs source)` by AC1 | `lower_node` is a private match (`op_coverage.rs` module doc); a table-driven dispatch would thread a runtime table through every `lower_*` helper. The equality test is the same cannot-drift guarantee the doc generator uses |
| four extra ops although the census finds none missing for bge-small | included | measured: the 1244 nodes of this model use 19 op types, all among the 48 in `op_coverage.md`. BERT-family exports from other exporters (opset 17 `LayerNormalization`, opset 20 `Gelu`, `Squeeze`, `Expand`) do need them; R2 is the generic claim, and the owner may strike it |
| pooling and normalization outside the door | in `tests/support/bge.rs` | CLS pooling and L2 are the bge recipe; the door returns the graph outputs. Generic-ness is counted by AC13 |
| one lowering and one plan per pin set | cache keyed by the pin map | the lowering binds concrete shapes (`lower.rs:243`); a symbolic plan is not what the lowering produces. Bucketing is the caller's: the bench pads each batch to the next multiple of 16 tokens, so the corpus needs at most 4 pin sets per cell |
| initializers decoded once per door | shared across pin sets | `lower_graph_pinned_cached` doc: ~26-30 ms per decode of the real weights, and a second f32 copy of 133 MB per pin set breaks the 500 MB cap (R12) |
| model delivery | file at the Hugging Face cache layout, hash and 4096-byte header vendored | the file is 133 MB; the repo vendors nothing larger than its header and a hash |
| tolerance | 1e-5 max abs (CPU), 1e-3 (Metal), on L2-normalized 384-vectors, pre-registered | CPU: the same graph's f32 embeddings differed from onnxruntime by at most 6.5e-8 (`discipline.md`, first 8 dims, three sentences); Metal: `omega/tests/onnx_metal_parity.rs:265` uses 1e-3 for an f32 chain. A failing measurement is reported with its number; the tolerance is not widened by the implementer |
| time bound | the rule of `decode_arms.rs:508-527`: arm median at or below reference median plus max(MAD, 2% of reference median) | the same rule the parity rounds' bound lines use |
| padding | each batch padded to the next multiple of 16 tokens (minimum 16, truncation at 64), mask zero on pad; tract is fed the identical tensors | tract timed on shorter unpadded rows would do less work than omega; identical tensors make the cells comparable, and padding invariance is checked separately (AC7) |
| bench discipline | arms interleaved per iteration, refuses to start above `--max-load`, no other process during the timed run, ollama off | recorded rules: interleaved GPU arms, load refusal, bench alone |

## acceptance criteria

Paths and variables: `BGE_MODEL_PATH` is `$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx`; `MNIST_MODEL` is `/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx`; the MNIST idx files are under `$HOME/.cache/burn-dataset/mnist/test/`; `FIX` is `omega/tests/fixtures/bge-small`; `OUT` is a scratch directory the runner names; `INDEXER_REPO` is the indexer repo checkout root; `INDEXER_CRATE` is the manifest directory of the indexer library crate holding `src/bin/cpu.rs`. Tests that need a model or a recorded file open it and fail, never skip, so each N counts runs that read the real file. Every AC is one command and one count; the parenthetical names every test case that count contains, and a slice that adds a different number updates its AC in the same commit.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-onnx --features std -E 'test(/onnx_census_/)'` | `8 passed` (counts_ops_of_a_hand_encoded_graph: nodes, distinct, per-op counts; names_every_unlowerable_op_once: a graph with two unsupported op types, three nodes each, yields both names once; supported_ops_equals_lower_node_arms: `SUPPORTED_OPS` equals `parse_supported_ops` of the `lower.rs` source; degenerate control: that source with the `"Erf"` arm deleted parses to one op fewer and the equality fails; empty_graph_reports_zero; unlowerable_agrees_with_lower_graph: the names equal the `op_type` of `LowerError::UnsupportedOp`; bge_small_reports_1244_nodes_19_distinct_0_unlowerable: reads `BGE_MODEL_PATH`, asserts the rendered line `census nodes=1244 distinct=19 unlowerable=0`; mnist_reports_13_nodes_6_distinct_0_unlowerable: reads `MNIST_MODEL`, asserts `census nodes=13 distinct=6 unlowerable=0`; the 1244 and 13 and the op lists were counted from the files' bytes while writing this spec) |
| AC2 | R2 | `cargo nextest run -p proxima-onnx --features std -E 'test(/onnx_lower_extra_/) + binary(op_coverage_doc_drift)'` | `13 passed` (layer_normalization_axis_last_with_scale_and_bias; layer_normalization_without_bias; gelu_erf_default; gelu_tanh_approximation; squeeze_axes_input; squeeze_without_axes_drops_every_unit_dimension; expand_broadcasts_to_a_larger_shape; expand_rejects_a_shape_that_cannot_broadcast with a typed `LowerError`; each of the four numeric cases (the two layer normalizations and the two gelus) compared with a value hand-derived from the onnx `Operators.md` formula, and one degenerate control per numeric case with the expected value altered, 4 tests; the existing drift test `op_coverage_doc_drift`, which fails unless `docs/op_coverage.md` lists the 52 ops `lower_node` now dispatches) |
| AC3 | R3 | `cargo nextest run -p omega --features onnx-door,cpu -E 'test(/onnx_door_contract_/)'` | `12 passed` (mlp_matches_hand_computed_softmax_values: the two-layer MLP of `onnx_metal_parity.rs` through `door.call` equals 0.5998884, 0.4001116, 0.43474925, 0.56525074 within 1e-3; mlp_control_altered_expected_value_fails; parse_failure_reports_stage_parse; validate_failure_lists_every_unlowerable_op; unpinned_symbolic_dimension_reports_stage_validate_with_the_dimension_name; lower_failure_reports_stage_lower; plan_failure_reports_stage_plan: `Engine::Gpu` with a driver not compiled in; execute_failure_reports_stage_execute: a graph input not supplied; stages_are_ordered_parse_validate_lower_plan_execute; door_is_a_pipe_with_in_door_batch_out_door_output_err_door_error; door_composes_after_a_pipe_with_and_then; config_builder_equals_conflaguration_loader) |
| AC4 | R4 | `cargo run -p omega --features onnx-door,cpu --example onnx_door_walkthrough \| grep -c '^transition '` | `10` (5 legal: parsed to validated, validated to lowered, lowered to planned, planned to executed, executed to planned; 5 failure exits, one per stage: parse, validate, lower, plan, execute, each line naming the stage) |
| AC5 | R5 | `cargo nextest run -p omega --features onnx-door,cpu -E 'test(/onnx_door_plan_cache_/)'` | `4 passed` (second_call_same_pins_runs_zero_lowerings_and_zero_plans; new_pins_run_exactly_one_lowering_and_one_plan; initializers_decode_once_across_three_pin_sets; exceeding_max_pin_sets_evicts_the_oldest_and_counts_one_eviction) |
| AC6 | R6 | `cargo nextest run -p omega --features onnx-door -E 'test(/bge_fixture_/)'` | `7 passed` (model_file_length_is_133093490; model_file_sha256_equals_the_recorded_hash; altered_byte_changes_the_sha256 as the degenerate control; vendored_header_equals_the_first_4096_bytes_of_the_model; corpus_has_64_sentences; corpus_ids_start_101_end_102_and_fit_64_tokens; first_three_corpus_rows_equal_the_hard_coded_id_arrays_of_bge_eval_rs) |
| AC7 | R7 | `cargo nextest run -p omega --features onnx-door -E 'test(/tract_record_/)'` | `8 passed` (b1_record_has_64_rows_of_384_finite_floats; b32_record_has_2_batches_of_32_rows_of_384; header_names_tract_onnx_0_22_4_and_the_model_sha256; timing_has_b1_and_b32_cells_with_3_warmup_and_5_timed_passes_each; padding_invariance_each_b32_row_equals_its_b1_row_within_1e-5_after_normalization; three_sentence_cosines_equal_the_recorded_onnxruntime_values_0_936311_0_378777_0_334176_within_1e-4; identity control: the record's cosines compared with themselves pass; degenerate control: cosine(A,B) altered by 1e-3 fails) |
| AC8 | R8 | `cargo nextest run --release -p omega --features onnx-door,cpu -E 'test(/bge_cpu_parity_/)'` | `6 passed` (b1_all_64_sentences_within_1e-5_of_tract; b32_all_64_sentences_within_1e-5_of_tract; identity control: tract's record compared with itself passes; degenerate control: one element altered by 1e-3 fails; mask_leak_control: a b32 run with the attention mask set to ones on pad positions exceeds the tolerance on at least one sentence; outputs_are_finite_and_unit_norm) |
| AC9 | R9 | `cargo nextest run --release -p omega --features onnx-door,cpu,metal -E 'test(/bge_metal_parity_/)'` | `7 passed` (b1_all_64_sentences_within_1e-3_of_tract; b32_all_64_sentences_within_1e-3_of_tract; identity control; degenerate control with one element altered by 1e-2; mask_leak_control; outputs_are_finite_and_unit_norm; gpu_ran_control: `omega::metal::current_allocated_size()` is greater than 0 after the run) |
| AC10 | R10 | `cargo run --release -p omega --features onnx-door,cpu,metal --example onnx_door_bench -- --runs 5 --warmups 3 --max-load 3.0 \| tee "$OUT/bench.log" \| grep -c -E '^table arm=(tract\|omega_cpu\|omega_metal) cell=b(1\|32) .*ms_per_sentence=[0-9.]+ .*wall_ms_median=[0-9.]+ .*cpu_pct=[0-9.]+ .*rss_peak_mb=[0-9.]+ .*rss_steady_mb=[0-9.]+ .*gpu_alloc_mb=[0-9.]+ .*cov_pct=[0-9.]+'` | `6` (3 arms by 2 batch sizes, each row carrying all seven columns; the tract rows are read from `tract_timing.jsonl`, never re-run) |
| AC11 | R11 | `grep -c -E '^bound metric=ms_per_sentence arm=omega_(cpu\|metal) vs=tract cell=b(1\|32) .* within=true$' "$OUT/bench.log"` | `4` (the log is the one the AC10 command wrote at the same commit) |
| AC12 | R12 | `grep -c -E '^bound metric=rss_peak_mb arm=omega_(cpu\|metal) vs=cap500 cell=b(1\|32) .* within=true$' "$OUT/bench.log"` | `4` (reference median is the constant 500) |
| AC13 | R13 | `cargo nextest run --release -p omega --features onnx-door,cpu,metal -E 'test(/onnx_door_generic_/)'` | `6 passed` (mnist_cpu_top1_at_least_0_98_over_first_1000_t10k_images; mnist_metal_top1_at_least_0_98; labels_compared_with_themselves_score_1_0 as the identity control; labels_shuffled_by_one_position_score_below_0_98 as the degenerate control; door_sources_name_no_model: the count of `bge`, `bert` and `mnist` (case-insensitive) in `omega/src/onnx_door/*.rs` is 0; model_name_counter_control: the same counter over a string containing `Mnist` returns 1) |
| AC14 | R14 | `for lock in Cargo.lock "$INDEXER_REPO/Cargo.lock" fuzz/Cargo.lock; do grep -c -E '^name = "(tract[a-z0-9_-]*\|ort\|ort-sys\|onnxruntime[a-z0-9_-]*)"$' "$lock"; done \| paste -sd' ' -` | `0 7 0` (proxima's lock, the indexer repo's lock as the degenerate control that must match, the fuzz lock) |
| AC15 | R15 | `( cd fuzz && cargo run --release --bin fuzz_onnx ) \| grep -c 'panics=0$'` | `3` (onnx::parse_complete::random, onnx::parse_complete::then_lower_graph, onnx::parse_complete::then_census; each line is `SweepReport::print_line`, `fuzz/src/lib.rs:67`) |

GPU cells: AC9, AC10 (and so AC11, AC12) and AC13's Metal tests run behind `gpu.lock`; AC1 to AC8, AC14 and AC15 are CPU-only.

## out of scope

- training graphs, `Loop`-driven training, an ONNX writer (`proxima-onnx/src/writer.rs` is untouched)
- quantized ONNX (QDQ, `QLinear*`): the census finds no such op in bge-small (the 19 op types are listed in AC1)
- ONNX Runtime contrib-domain ops (`SkipLayerNormalization`, `Attention`, `FastGelu`)
- tokenization inside proxima: `proxima-tokenizer` has no WordPiece (`bge_eval.rs` module doc); the corpus carries ids made by the indexer repo's tokenizer
- any proxima dependency on tract or onnxruntime (R14)
- a symbolic (non-pinned) plan, dynamic output shapes, data-dependent control flow (`LowerError::DataDependentControlFlow`)
- sentence-quality or retrieval metrics (hit rates): parity is by embedding values
- changing tolerances or bounds after a measurement; a failing number is reported, not widened
- other doors (GGUF, safetensors, MLX): the door frame is the generic IR with many doors, and this spec is the ONNX door only

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| omega has never run bge-small on `Engine::Cpu` through `plan_named` or on Metal (the repo's only bge runs use `proxima_tensor::cpu` directly, `bge_eval.rs:238`; the only onnx-on-Metal test is a 4-node MLP) | high | AC8 to AC9 fail on a plan or an op the Metal emitter rejects | the failing `DoorError` names stage and node; the fix is a rule over the algebra in the emitter or lowering, with a synthetic single-op test, never a model branch |
| the Metal result misses 1e-3, or f16 accumulation moves it | medium | AC9 fails | report the per-sentence max abs; the numeric policy is a `DoorConfig` field, so an f32 policy is tried before any tolerance discussion |
| tract's single-thread executor is slower than omega CPU anyway, or faster than Metal at batch 1 (launch overhead on a 12-layer graph) | medium | AC11 has `within=false` cells | the bench trace (per-stage and per-dispatch) decides the slice; the bound is not relaxed |
| peak RSS exceeds 500 MB: model bytes 133 MB, decoded f32 initializers 133 MB, the plan arena, plus Metal buffers | high on CPU | AC12 fails | R5 decodes initializers once; slices measure the footprint columns first and drop the model bytes after lowering if the trace shows they are held |
| the pin-set cache holds one arena per pin set and each arena binds all weights | medium | RSS grows by 133 MB per pin set | `max_pin_sets` default comes from the sizing TOML; the AC5 eviction case counts evictions; S4 measures bytes per pin set |
| `ModelProto` borrows the bytes, so a new pin set re-parses (zero-copy by the `lib.rs` doc, cost unmeasured) | low | extra milliseconds on a cache miss | S4 prints the parse time per pin set; unmeasured until then |
| `Expand` may not be expressible as a pure op without an operand of the target shape | medium | AC2 expand cases fail | land it as a broadcast view like `lower_unsqueeze` (`Value.view`), or a typed `LowerError::UnsupportedShape` recorded in the notes; no new `Op` form (`lib.rs` doc) |
| integer inputs carried as f32 lose ids above 2^24 | low for a 30522-entry vocabulary | wrong embedding for a larger vocabulary | the validate stage rejects an integer input whose declared range exceeds 2^24 |
| the corpus builder finds fewer than 61 usable sentences in `SHAPE.md`, `FEATURES.md`, `SECURITY.md` and `book/src/algebra/*.md` | low | AC6 corpus count is not 64 | the builder reads further `book/src` pages in name order; the source list is recorded in `FIX/README.md` |
| the Hugging Face `onnx/model.onnx` is replaced at the revision | low | hash mismatch | AC6 fails on the hash first; the revision is pinned in the URL |
| tract timings recorded on a loaded box | medium | the incumbent looks slower than it is | the harness prints the one-minute load before and after each pass and refuses above its own limit; the passes are recorded raw |
| the sentences contain text the repository content gate rejects | low | the commit is blocked | the builder skips such sentences and takes the next |

## context

- `proxima-onnx/src/lower.rs:64` (`LowerError`), `:168` (`Lowered`), `:227` (`lower_graph`), `:243` (`lower_graph_pinned`), `:450` (`lower_graph_pinned_cached`), `:465` (`lower_node`, 48 arms)
- `proxima-onnx/src/pipe.rs:23` (`parse_complete`); `proxima-onnx/src/op_coverage.rs` (`parse_supported_ops`); `proxima-onnx/docs/op_coverage.md:5` ("48 ops lower today"); `proxima-onnx/tests/op_coverage_doc_drift.rs`
- `proxima-onnx/examples/bge_eval.rs:90` (`MODEL_PATH_ENV`), `:96` (`sentences`), `:124` (`dynamic_inputs`), `:196` (`cls_normalize`), `:238` (`run_pass_arena`)
- `omega/src/backend.rs:99` (`Engine`), `:173` (`GpuDriver`), `:347` (`Plan`), `:405` (`plan_named`), `:480` (`execute_plan_named`); `omega/tests/onnx_metal_parity.rs:265` (the existing onnx-on-Metal test); `omega/Cargo.toml:673` (`proxima-onnx` dev-dependency), `:525` (`conflaguration`); `omega/omega-runtime.toml`
- `proxima-primitives/src/pipe/primitives.rs:91` (`Pipe`)
- `examples/wire_to_weights.rs` and `examples/support/wire_to_weights_pipeline.rs` (the hand-wrapped onnx forward the door replaces)
- `fuzz/src/bin/fuzz_onnx.rs:24` (`run`), `:36` (`parse_then_lower_sweep`); `fuzz/src/lib.rs:67` (`print_line`)
- `proxima-model-interop/examples/cell_resources/cell.rs:1` (the resource columns), `proxima-model-interop/examples/decode_arms.rs:484,508` (the bound lines)
- `proxima-tensor/docs/discipline.md:17482` (onnxruntime cosines on the three bge_eval sentences); `scripts/onnx_reference/README.md` (the onnxruntime and torch arms)
- `proxima-tensor/specs/mlx-cold-export/SPEC.md` and `proxima-tensor/specs/fp6-e3m2-codec/SPEC.md` (shape of this spec)
- outside this repository, not resolvable with `git show` and not a path in it: the incumbent, tract-onnx 0.22.4 (locked 0.22.4) as the indexer repo's cpu binary calls it (`model_for_read`, `into_optimized`, `into_runnable` at line 36 of that binary), indexer repo commit `359f96c`; the model's tokenizer is `tokenizer.json` of the same Hugging Face revision
