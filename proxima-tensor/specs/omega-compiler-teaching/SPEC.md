# omega-compiler-teaching

status: draft
owner: Brian Bruggeman
created: 2026-10-11

## problem

A reader with a GGUF file and this repository alone cannot trace how the file becomes a dispatched kernel, because the runtime side has a curriculum (`docs/tutorials/00-foundations.md` through `11-any-transport-agnostic.md`, `book/src/SUMMARY.md`, `examples/<topic>/`) and the compile side, the 10 stages of omega's pipeline (read, lower, ir, infer, bind, rewrite, render, emit, plan, dispatch; censused from source below), has none; this spec adds a 10-document, 10-example, 1-end-to-end curriculum under `docs/omega-compiler/`, measured by: the end-to-end example taking the granite 3.1 1b Ollama blob through all 10 stages on the CPU engine and printing exactly 10 stage records, ~~its generated ids equal to the first 4 of the vendored llama.cpp ids~~ (struck: byte-exact model responses are the wrong correctness bar (owner 2026-10-11: "we cannot really judge responses even from the same model on the same stack"); exactness applies only where the arithmetic is exact) replaced by a top-1 agreement rate of at least 0.95 between its greedy generated ids and the 224 recorded llama.cpp ids of the granite parity fixture (status plausible), and 56 cited `file:line` symbols (44 stage rows and 12 answer-key rows) all found on the cited lines by a test.

## refutation condition

The claim "the compile side is teachable from the repo alone, in 10 stages" is refuted by any one of these observations:

- the end-to-end example needs a stage that is not one of the 10 (its output carries an eleventh record kind, or a stage record names a function that no census row cites);
- the answer key's 12-step chain from a dispatched kernel to the layout struct and the pipe skips a hop that exists in source (a reader following the chain with `grep` finds a function between two cited rows that the key does not name);
- ~~AC4's generated ids differ from the vendored llama.cpp ids at any of the first 4 positions~~ (struck: byte-exact model responses are the wrong correctness bar (owner 2026-10-11: "we cannot really judge responses even from the same model on the same stack"); exactness applies only where the arithmetic is exact). AC4's top-1 agreement rate over the 224 recorded positions falls below 0.95 on the CPU engine while the stage records say every stage ran, which means the records describe a pipeline that is not the one producing the tokens;
- the citation test's wrong-symbol control (AC7) passes a row whose symbol is not on its line, which means the test cannot fail and the docs can drift unseen.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | one teaching document per pipeline stage (10), each naming the primitive the stage composes and linking the next stage | yes |
| R2 | one runnable example per stage (10), each printing exactly one structured stage record | yes |
| R3 | one end-to-end example takes the real granite 3.1 1b GGUF through every stage on the CPU engine and prints exactly 10 stage records | yes |
| R4 | ~~the end-to-end example's generated token ids equal the vendored llama.cpp ids for the same prompt~~ (struck: byte-exact model responses are the wrong correctness bar (owner 2026-10-11: "we cannot really judge responses even from the same model on the same stack"); exactness applies only where the arithmetic is exact); the top-1 agreement rate between the end-to-end example's greedy generated ids and the vendored llama.cpp ids over all 224 recorded positions of the granite parity fixture (`granite_moe/llama_ids.json` 3 cases x 32 = 96, `granite_moe/long_prompt_llama_ids.json` 1 case x 128 = 128) is at least 0.95 | yes |
| R5 | the public compile-side API of `omega` carries doctests that compile and pass | yes |
| R6 | the public compile-side API of `proxima-tensor` carries doctests that compile and pass | yes |
| R7 | every `file:line` citation in the stage documents is found on its line by a test, and the test can fail | yes |
| R8 | a "trace a kernel back" exercise carries an answer key of 12 `file:line` hops from a dispatched kernel to the layout struct and the pipe, checked by the same kind of test | yes |
| R9 | the curriculum teaches the one-struct / one-pipe shape the admitted `float-layout-dtype` spec lands: the ir and render documents name `FloatLayout`, the plan and dispatch documents name the one evaluate pipe | yes |
| R10 | every curriculum document links into the runtime-side curriculum (`docs/tutorials/00-foundations.md`) | yes |
| R11 | the book index links the curriculum | yes |

## architecture

Layout, mirroring the runtime curriculum (`docs/tutorials/NN-<topic>.md` with prerequisites / you will learn / new concepts / contents, every block cited by `file:line`; `book/src/SUMMARY.md` as the index; `examples/<topic>/` as the runnable half):

```
docs/omega-compiler/index.md                 overview, the 10-stage map, the runtime-side links
docs/omega-compiler/NN-<stage>.md            NN = 01..10, one per stage
docs/omega-compiler/trace-a-kernel.md        exercise, then "## answer key" with 12 numbered hops
proxima-model-interop/examples/compile_stage_NN_<stage>.rs     10 stage examples
proxima-model-interop/examples/compile_walkthrough.rs           the end-to-end example
omega/tests/compile_citations.rs             the citation test
omega/tests/fixtures/compile_citations.toml  rows: group, file, line, symbol (44 stage rows, 12 key rows)
```

Every stage document has two machine-checked lines, `Composes: <the primitives, each as file:line>` and `Next: [<next stage>](NN-<stage>.md)` (stage 10's `Next` links `trace-a-kernel.md`). The examples live in `proxima-model-interop` because the only crate that can bind a GGUF's tensors by name (`LoadedModel::load_with_descriptor`, `proxima-model-interop/src/generate/pregather.rs:2572`) is downstream of `omega`; they drive `omega::backend` with `Engine::Cpu`, so no Metal device is touched.

A stage record is one JSON line, `{"stage":"<name>","index":<1..10>,"engine":"cpu","in":<count>,"out":<count>,"detail":{...}}`; `in` and `out` are the counted objects the stage consumed and produced (ops, bound ops, kernels, plan nodes), and `detail` carries the stage's own facts (stage 7: the codec tag and block bytes of the first packed weight node; stage 8: the first kernel's `entry` and `bindings.len()`). The end-to-end example prints the 10 records, then one `{"check":"llama_ids",...}` line that is not a stage record.

### the stage census (read from source, this session)

| # | stage | primitive composed | citations (44) |
|---|---|---|---|
| 1 | read | `parse_complete` over the mmap'd file into `ParsedGguf`; `DoorStage::Parsed` is the same step for ONNX | `proxima-gguf/src/pipe.rs:106` parse_complete; `proxima-gguf/src/pipe.rs:25` ParsedGguf; `omega/src/door/reader.rs:30` Reader; `omega/src/door/stage.rs:18` DoorStage |
| 2 | lower | `build_forward` from a `ModelDescriptor` into `ForwardProgram`; `Lowered` is the ONNX door's equivalent | `proxima-tensor/src/spec/descriptor.rs:679` build_forward; `proxima-tensor/src/spec/descriptor.rs:572` ForwardProgram; `proxima-tensor/src/lowered.rs:12` Lowered; `omega/src/onnx_door/mod.rs:45` OnnxDoor |
| 3 | ir | `Op` program in an append-only `Vec`, `IndexMap` indexing maps, `Extent` symbols, `DType` | `proxima-tensor/src/op.rs:191` Op; `proxima-tensor/src/op.rs:45` Extent; `proxima-tensor/src/map.rs:218` IndexMap; `proxima-tensor/src/dtype.rs:17` DType; `proxima-tensor/src/op.rs:335` append |
| 4 | infer | one `ShapeTable::push` per op, driven by `infer` | `proxima-tensor/src/shape.rs:707` infer; `proxima-tensor/src/shape.rs:99` ShapeTable; `proxima-tensor/src/shape.rs:46` Shapes |
| 5 | bind | `bind_with_fusion` resolves each `Op` to a `BoundOp` whose operands are `Layout` strides | `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:62` bind_with_fusion; `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:5` fn bind; `proxima-tensor/src/bind/types_layout_boundop.rs:126` BoundOp; `proxima-tensor/src/bind/types_layout_boundop.rs:14` Layout; `proxima-tensor/src/bind/types_layout_boundop.rs:146` BoundOpKind |
| 6 | rewrite | the worklist engine that admits epilogue-fuse and layer-norm-cluster rewrites over the bound ops | `proxima-tensor/src/cpu/epilogue.rs:820` run_rewrite_worklist; `proxima-tensor/src/cpu/epilogue.rs:945` rewrite_engine_depth_fires |
| 7 | render | classify a `BoundOp` against its operand codecs, then render the codec's decode into the kernel body | `omega/src/msl/emit_and_classify.rs:2877` classify_packed_row_block; `omega/src/msl/packed_row_blocked_ggml.rs:4` push_packed_row_blocked_body; `omega/src/msl/scaled_float_render.rs:17` push_scaled_float_decoders; `omega/src/msl/kernel_types_identity.rs:1282` Q4_0_UNPACK_MSL; `proxima-primitives/src/float_layout.rs:20` FloatLayout; `proxima-primitives/src/codec.rs:17` Codec |
| 8 | emit | the three text emitters over one `BoundOp`: MSL, CUDA, WGSL | `omega/src/msl/emit_and_classify.rs:43` fn emit; `omega/src/msl/kernel_types_identity.rs:184` Kernel; `omega/src/msl/kernel_types_identity.rs:239` GridSpec; `omega/src/cuda.rs:183` emit_cuda; `omega/src/wgsl.rs:182` emit_wgsl |
| 9 | plan | `plan_named` resolves names to blocks and builds the reusable `Plan`; the CPU arm checks a cached arena out and in, the Metal arm lays out its buffer arena | `omega/src/backend.rs:413` plan_named; `omega/src/backend.rs:355` Plan; `proxima-tensor/src/cpu/arena.rs:1358` checkout_arena; `proxima-tensor/src/cpu/arena.rs:1410` checkin_arena; `omega/src/metal/device_buffers_arena_plan.rs:773` Plan; `omega/src/metal/arena_layout.rs:50` lay_out_whole_slots |
| 10 | dispatch | `execute_plan_named` runs the plan: CPU `run_node` per node, Metal `execute_plan` per kernel; `Door` is the `Pipe` that wraps plan and execute | `omega/src/backend.rs:488` execute_plan_named; `omega/src/metal/execute_and_hazards.rs:114` execute_plan; `proxima-tensor/src/cpu/run_node.rs:4` run_node; `omega/src/door/mod.rs:210` Pipe for Door |

Citation count per stage: 4, 4, 5, 3, 5, 2, 6, 5, 6, 4 = 44. Each row above was checked this session with `sed -n "<line>p" <file>` against its symbol: 45 of 45 matched (the 45th is the answer-key row `omega/src/msl/emit_and_classify.rs:1791` kernel_cache_key).

### the answer key chain (12 hops, reading from the kernel back to the pipe)

1 `omega/src/metal/execute_and_hazards.rs:114` execute_plan, 2 `omega/src/msl/kernel_types_identity.rs:184` Kernel, 3 `omega/src/msl/emit_and_classify.rs:43` emit, 4 `omega/src/msl/emit_and_classify.rs:1791` kernel_cache_key, 5 `omega/src/msl/emit_and_classify.rs:2877` classify_packed_row_block, 6 `omega/src/msl/packed_row_blocked_ggml.rs:4` push_packed_row_blocked_body, 7 `omega/src/msl/kernel_types_identity.rs:1282` Q4_0_UNPACK_MSL (the authoring slice substitutes the unpack constant of the codec that dominates the granite blob's tensor directory, read from the stage 1 record), 8 `proxima-primitives/src/codec.rs:17` Codec, 9 `proxima-primitives/src/float_layout.rs:20` FloatLayout, 10 `proxima-tensor/src/bind/types_layout_boundop.rs:126` BoundOp, 11 `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:62` bind_with_fusion, 12 `omega/src/door/mod.rs:210` Pipe for Door.

### the dependency on `float-layout-dtype`

The walkthrough teaches the one-struct shape that `proxima-tensor/specs/float-layout-dtype/SPEC.md` lands, and stage content that would be wrong before it lands waits for it (TASKS.md orders those slices last):

- stage 3 (ir) and its example: after float-layout-dtype slice 2 (`DType` float variant carrying a `FloatLayout`, `proxima-tensor/specs/float-layout-dtype/TASKS.md:10`);
- stage 7 (render) and its example: after float-layout-dtype slice 4 (the one MSL emitter over a layout, `proxima-tensor/specs/float-layout-dtype/TASKS.md:12`);
- stages 9 (plan) and 10 (dispatch) and their examples: after float-layout-dtype slice 18 (one evaluate pipe per back end, `proxima-tensor/specs/float-layout-dtype/TASKS.md:26`).

The other six stages do not touch what those slices change and land first.

A finding recorded while censusing: the doc comment and doctest of `msl::emit` sit at `omega/src/msl/kernel_types_identity.rs:324-373` but are attached to the constant `Q3K_UNPACK_MSL` at `omega/src/msl/kernel_types_identity.rs:398`, while `emit` at `omega/src/msl/emit_and_classify.rs:43` carries no doc. Slice 14 of TASKS.md moves them onto `emit`.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| where the examples live | `proxima-model-interop/examples/`, driving `omega::backend` with `Engine::Cpu` | `omega/examples/` cannot depend on `LoadedModel`; `omega` is upstream of `proxima-model-interop`, so a GGUF with real weights cannot be bound there |
| what a stage record carries | counted `in`/`out` plus a `detail` object of facts read from the stage's own output | wall-clock fields: timing is not what a stage teaches, and it would make the record non-deterministic |
| how drift is caught | a fixture table of `(group, file, line, symbol)` rows checked by a test, plus a test that every `file:line` in the documents is a table row | prose review: the citation drift this curriculum exists to avoid is exactly what review misses |
| stages as a count of 10 | the census above, with `Door` (`omega/src/door/mod.rs:210`) inside stages 1, 2, 9 and 10 rather than a stage of its own | an eleventh "door" stage: the door is the sans-IO wrapper over four of the ten, not a transformation of its own, so the end-to-end record count would double-count them |
| Metal | cited and explained in stages 9 and 10 by `file:line`, never run in an AC | the ACs must run on any box, and the curriculum teaches the engine-neutral `Plan` first (`omega/src/backend.rs:355`) |

## acceptance criteria

All commands run from the repository root.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `grep -l '^Composes: ' docs/omega-compiler/[0-9][0-9]-*.md \| xargs grep -l '^Next: \[' \| wc -l` | `10` |
| AC2 | R2 | `for example in $(ls proxima-model-interop/examples/compile_stage_[0-9][0-9]_*.rs \| xargs -n1 basename \| sed 's/\.rs$//'); do cargo run -q -p proxima-model-interop --features std,metal --example $example; done \| grep -c '^{"stage":'` | `10` |
| AC3 | R3 | `cargo run -q -p proxima-model-interop --features std,metal --example compile_walkthrough \| grep -c '^{"stage":'` | `10` |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/compile_walkthrough_ids_/)'` | `3 passed, 0 skipped`: `compile_walkthrough_ids_agreement_rate_at_least_bound` (free-running greedy CPU-engine generation for the 4 recorded prompts, position `i` agrees when the generated id equals the recorded `generated_ids[i]`, no teacher forcing; asserts the agreement count over 224 positions divided by 224 is at least 0.95, prints `agree=<n> of 224 rate=<r>`), `compile_walkthrough_ids_control_corrupted_run_falls_below_bound` (control that must fail the bound: every recorded id incremented by 1 through the same comparator gives rate 0.0, below 0.95), `compile_walkthrough_ids_counter_control_identity_run_is_1_0` (the vendored ids themselves through the comparator give exactly 224 of 224, rate 1.0). The 0.95 bound is plausible: no agreement rate between llama.cpp and proxima on this fixture is recorded in the repository (grep of `proxima-tensor/specs` and the fixture directory for a recorded `of 224` count returned 0 hits), so the bound is unmeasured until the first run; the first run's printed rate is recorded in the slice result, and any change to the bound is a strike-and-amend with that rate as its reason |
| AC5 | R5 | `cargo test --doc -p proxima-omega \| grep '^test result'` | `test result: ok. 7 passed; 0 failed; 0 ignored` (1 existing at `omega/src/msl/kernel_types_identity.rs:331`, 6 new: `Kernel`, `GridSpec`, `Binding`, `plan_named`, `execute_plan_named`, `context_chunks_for`) |
| AC6 | R6 | `cargo test --doc -p proxima-tensor --features std \| grep '^test result'` | `test result: ok. 8 passed; 0 failed; 0 ignored` (1 existing at `proxima-tensor/src/lib.rs:124`, 7 new: `infer`, `bind`, `bind_with_fusion`, `BoundOp`, `Layout::offset_of`, `Lowered`, `cpu::evaluate_named`) |
| AC7 | R7 | `cargo nextest run -p proxima-omega --no-default-features --features std -E 'test(/compile_citations_(stage_rows_44\|docs_cite_only_table_rows\|refuses_wrong_symbol)/)'` | `3 passed, 0 skipped` (the first asserts 44 stage rows each found on its line; the second that every `file:line` in `docs/omega-compiler/*.md` is a table row; the control feeds a row with the right line and a wrong symbol and expects a refusal) |
| AC8 | R8 | `cargo nextest run -p proxima-omega --no-default-features --features std -E 'test(/compile_citations_key_rows_12/)' && awk '/^## answer key/{f=1;next} f&&/^[0-9]+\. /{n++} END{print n}' docs/omega-compiler/trace-a-kernel.md` | `1 passed, 0 skipped` then `12` |
| AC9 | R9 | `echo "$(grep -l 'FloatLayout' docs/omega-compiler/[0-9][0-9]-*.md \| wc -l) $(grep -l 'one evaluate pipe' docs/omega-compiler/[0-9][0-9]-*.md \| wc -l)"` | `2 2` (the ir and render documents; the plan and dispatch documents) |
| AC10 | R10 | `grep -l 'tutorials/00-foundations.md' docs/omega-compiler/*.md \| wc -l` | `12` (index, 10 stage documents, the exercise) |
| AC11 | R11 | `grep -c 'omega-compiler/' book/src/SUMMARY.md` | `12` |

AC4 is the correctness AC. Bit-exactness is kept where the arithmetic is exact (the stage records' counts and the AC7 and AC8 citation rows); a model response is judged by agreement rate, not equality. Its oracle, `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`, is llama.cpp's output for the prompt "The capital of France is" and shares no code with the path under test; the counter-control runs that oracle's own ids through the same comparator and expects exactly 1.0, and the control corrupts every id and expects a rate below the bound.

## out of scope

- any new compiler behaviour: no change to `infer`, `bind_with_fusion`, the worklist, an emitter, the plan or the dispatch loop
- running Metal, CUDA or WGSL devices in an AC; stages 8 to 10 cite them, the records come from the CPU engine
- the door specs' own work (`onnx-door`, `litert-door`, `mlir-interop`, `coreml-ane-lane` under `proxima-tensor/specs/`); stages 1, 2, 9 and 10 cite the door as it exists
- prose about intent, history, or comparisons with other compilers
- bench numbers; the curriculum states no performance figure

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| a cited line drifts when source above it changes | high | a document teaches a wrong location | AC7 fails on the first drifted row and names it; rows are updated in the commit that moves the line |
| the granite blob is absent on a box | medium | AC3 and AC4 cannot run | both read `PROXIMA_ARCH_GRANITE_MOE_GGUF` and fall back to the staged Ollama blob path used at `proxima-model-interop/examples/model_config_load.rs:26`; a missing file is a named error, not a skip |
| the CPU engine's agreement rate with llama.cpp is below 0.95 | medium | AC4 fails | that is a finding recorded as the slice's result with the printed rate and the first diverging position per prompt; the bound changes only by strike-and-amend citing the measured rate, never by loosening the comparator |
| float-layout-dtype slices 2, 4 and 18 land later than expected | medium | stages 3, 7, 9, 10 stay unwritten | their slices are ordered last in TASKS.md; the other slices land independently |
| the 6 new omega doctests need a macOS-only path | low | AC5 differs across hosts | the doctests use `omega::backend` with `Engine::Cpu` and `omega::msl::emit`, both compiled by `alloc`/`std`/`cpu` without a device |

## context

- `docs/tutorials/00-foundations.md`, `docs/tutorials/03-native-runtime.md`: the runtime-side tutorial shape (prerequisites, you will learn, new concepts, contents, every block cited)
- `scripts/tutorials-gate.sh`: how the runtime side compile-checks its tutorial blocks
- `book/src/SUMMARY.md`: the runtime-side book index this curriculum joins
- `proxima-tensor/docs/a-model-is-a-configuration.md`, `proxima-tensor/docs/rewrite-algebra.md`: existing compile-side prose, treated as suspect and re-derived from source
- `omega/examples/door_walkthrough.rs`, `omega/tests/door_support/walkthrough.rs`: the one existing state-machine walkthrough (principle 11 shape) the stage examples mirror
- `proxima-model-interop/examples/model_config_load.rs`: loads the granite blob and lowers it; the end-to-end example extends it
- `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`: the AC4 oracle
- `proxima-tensor/specs/float-layout-dtype/SPEC.md`, `proxima-tensor/specs/float-layout-dtype/TASKS.md`: the dependency
- `proxima-tensor/specs/onnx-door/SPEC.md`, `proxima-tensor/specs/litert-door/SPEC.md`, `proxima-tensor/specs/mlir-interop/SPEC.md`, `proxima-tensor/specs/coreml-ane-lane/SPEC.md`: the admitted door specs
