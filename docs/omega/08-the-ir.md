# 08: the IR, from the lowest algebra up

Omega has no IR of its own. The IR is `proxima-tensor`'s program (`Vec<Op>`, `proxima-tensor/src/op.rs:208`) and its resolved form (`Vec<BoundOp>`, `proxima-tensor/src/bind/types_layout_boundop.rs:199`). A backend reads a `BoundOp` and writes its machine's text. This page builds both from the bottom, so you can say what any op is made of and what a lowering must do with it.

You need Rust and nothing else. Every claim has a `file:line`; paths are from the repo root. Every name is public API except those marked as a private test fixture.

## 1. The lowest algebra

A program is a flat `Vec<Op>`. A reference is a `NodeId`, a position in that vector, and may only point backwards (`op.rs:3-17`). `Op` has five variants (`op.rs:208-299`):

| variant | what it is |
|---|---|
| `Input` | a leaf: data enters, by name (`op.rs:218`) |
| `Elementwise` | a map: one `ScalarOp` over an index space (`op.rs:227`) |
| `Reduce` | a fold of one `ScalarOp` over axes (`op.rs:237`, struct at `op.rs:184`) |
| `Iota` | a leaf: `output[i] = i` (`op.rs:264`) |
| `Constant` | a leaf: one literal in every element (`op.rs:294`) |

`ScalarOp` (`op.rs:77`) is the machine's scalar vocabulary: 17 variants, `Identity Add Subtract Multiply Divide Maximum Minimum Negate Reciprocal Exponential Logarithm SquareRoot Tanh Erf Greater Equal Select`. It is closed on purpose: `gelu` is several elementwise ops, not a variant (`op.rs:67-73`). `ScalarOp::arity` gives the operand count (`op.rs:112`); `is_associative` says whether a fold may be reordered (`op.rs:139`).

Every op carries a `dtype: DType` (`Op::dtype`, `op.rs:331`). `DType` is `Bool` or `Layout(FloatLayout)` (`proxima-numeric/src/dtype.rs:12-15`). A `FloatLayout` (`proxima-numeric/src/float_layout.rs:93`) is the structure of one number: `exponent_bits`, `mantissa_bits`, `specials`, `bias_override`, `sign`, `nan`. An integer is the layout with zero exponent bits (`float_layout.rs:88-90`). So a new element type is a layout constant, not a new variant (`dtype.rs:3-7`). Note the spelling: `DType::FLOAT32` is `FloatLayout::BF32` (`dtype.rs:72`, `float_layout.rs:164`), binary32 with exponent 8 and mantissa 23.

`DType` names only one-number formats. The wider `NumericFormat` also has `Block` (a run of codes under a shared scale, `BlockLayout`) and `Levels` (`proxima-numeric/src/format.rs:251-253`). A quantized weight such as Q4_K is a `NumericFormat`, and no `DType` can say it. Remember this for section 5.

## 2. Composition and addressing

Two things turn single scalar ops into tensor ops: how an operand is addressed, and how bodies chain.

**Addressing.** In a program an operand carries an `IndexMap` (`proxima-tensor/src/map.rs:218`): `Affine(IndexPattern)`, or `Computed` for a data-dependent index, which is how gather needs no op of its own (`map.rs:164-167`). An `IndexPattern` says which iteration axis feeds each operand axis. `map::projection(3, &[0, 2])` reads axes 0 and 2 of a three-axis space (`map.rs:351`). A pattern that reorders is a transpose; one that skips an axis is a broadcast (`map.rs:348-349`, `op.rs:224-226`).

After binding, the pattern becomes a `Layout`: a `base` offset and one `stride` per iteration axis, 0 for an axis the operand does not vary along (`types_layout_boundop.rs:5-20`). The address is `base + sum(index * stride)` (`Layout::offset_of`, `types_layout_boundop.rs:25`). Same idea as `torch.Tensor.stride()`.

**Bodies.** `ComposedBody` is an ordered list of `BodyStep { op: ScalarOp, args }`; the last step is the result (`types_layout_boundop.rs:97-110`). An arg is `StepArg::Operand(slot)` or `StepArg::Step(earlier)` (`types_layout_boundop.rs:90-93`), backwards only, the same rule as `NodeId`. An unfused op is the one-step case, `ComposedBody::leaf(op)` (`types_layout_boundop.rs:116`). So the whole fusion machinery is one shape: a short list of scalar ops.

## 3. The two general ops

**`Elementwise`**: for every point of an index space, read each operand at its address, apply the body, write one value (`op.rs:227-232`).

**`Reduce`**: for every output point, fold the operand over the axes the output drops, starting from `init` (`ReduceInit`, `op.rs:151`), combining with `body`. `Keep::Reduce` keeps the last accumulator; `Keep::Scan` keeps every prefix, so a scan is a reduce and not a separate op (`op.rs:165-178`). A data-dependent `out_map` makes it a scatter (`op.rs:234-236`). The bound form records the accumulator's own format in `accumulator: DType` (`types_layout_boundop.rs:552`): a norm sums in f64 while its elements are f32.

**A matmul is a `Reduce` over a product body.** The program is two ops, quoted from the test fixture `matmul_program` (private, `proxima-tensor/src/bind/tests.rs:1877-1921`):

```rust
// iteration axes: 0 = row, 1 = column, 2 = contraction
let product = append(&mut program, Op::Elementwise {
    dtype: DType::FLOAT32, body: ScalarOp::Multiply,
    operands: vec![(lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                   (rhs, IndexMap::Affine(map::projection(3, &[2, 1])))],
    name: None });
let sum = append(&mut program, Op::Reduce(Reduce {
    dtype: DType::FLOAT32, body: ScalarOp::Add, init: ReduceInit::Zero,
    operand: product,
    in_map:  IndexMap::Affine(map::projection(3, &[0, 1, 2])),
    out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
    keep: Keep::Reduce, name: Some("matmul".into()) }));
```

Binding fuses the two into one op. The test `matmul_resolves_to_one_fused_op_not_two` asserts one `BoundOp`, kind `Reduce`, whose `element_body` is the multiply (`proxima-tensor/src/bind/tests.rs:1958-1985`). The crate says the same in one line: "`matmul` is a `Reduce(+)` over an `Elementwise(*)`" (`proxima-tensor/src/lib.rs:44`). There is no matmul op.

## 4. Fused forms

`BoundOpKind` (`types_layout_boundop.rs:221`) has 12 variants: the two general ops, two leaves, and eight fused forms. A fused form is a pattern of layers 1-3 collapsed so a backend can write one kernel. Each is still defined by the ops it replaces; that unfused chain is its reference.

| `BoundOpKind` | in terms of sections 1-3 |
|---|---|
| `Elementwise` (`:434`), `Reduce` (`:456`) | the two general ops, with fused bodies. `Reduce` also fuses a prologue (`element_body`) and an epilogue (`epilogue_body`) |
| `Iota` (`:667`), `Constant` (`:671`) | the two leaves, resolved |
| `ElementwiseTwin` (`:447`) | two `Elementwise` over one operand list, two outputs, one dispatch |
| `RoundBatchedReduce` (`:568`) | `round_count` sibling `Reduce`s differing only in the gather index, plus a round axis |
| `CachedAttention` (`:293`) | score fold, softmax, value fold over a KV cache (softmax is two reduces and three elementwise, `lib.rs:44-45`) |
| `CachedSoftmaxWeights` (`:637`) | the softmax-normalize stage of that chain alone |
| `GatedDeltaNet` (`:346`) | one recurrence step, a chain of about twelve ops (`:311-313`) |
| `MoeTopK` (`:405`) | unrolled rounds of argmax-with-exclusion: `Equal`, `Multiply`, `Reduce(Maximum)` (`:372-377`) |
| `TopFractionSelect` (`:427`) | `spec::top_fraction_mask`: a 0/1 mask on the `keep_count` highest scores (`:423-426`) |
| `BlockQuantize` (`:674`) | dense f32 to blocks of a `BlockLayout`; no program op yields it, a backend derives it (`:672-673`) |

Each fused form's parameters sit on the variant, so a backend admits a form by data and never by a branch; `CachedAttention`'s `planes` and `arithmetic` are the worked case (`types_layout_boundop.rs:124-126`, `:140-142`).

## 5. Binding

`BoundOp` (`types_layout_boundop.rs:199-215`) adds what a program leaves open:

| field | meaning |
|---|---|
| `node: NodeId` | the program op it answers for |
| `dtype: DType` | its output element type, copied from `Op::dtype` (`:203-208`) |
| `extents: Vec<u64>` | the resolved iteration space; symbolic extents are now numbers (`:209-213`) |
| `kind: BoundOpKind` | section 4. Operands are `(NodeId, Layout, Option<Lookup>)` (`:58`): an address, plus gather addressing when the index is computed (`Lookup`, `:51`) |

`bind` builds them from a program and its inferred shapes (`proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:20`; `bind_unfused` at `:7` skips fusion).

**Where each operand's format comes from today.** The output's format is `BoundOp.dtype`. An operand's format is not on the bound op: the operand tuple holds a node, a layout and a lookup, and no format. It is found somewhere else:

- A float operand: the `dtype` of the `BoundOp` or `Op::Input` that produced it.
- A quantized weight: bytes that arrive beside the program. The CPU takes `QuantizedBlock::Packed { codec, bytes }` (`proxima-tensor/src/cpu/epilogue.rs:1460-1466`). Omega takes `PackedOperands = BTreeMap<NodeId, Codec>` (`omega/src/lowering.rs:77`). `Codec` is a re-export of `NumericFormat` (`proxima-numeric/src/lib.rs:46-48`).

That is the gap behind the "no fixed weights" rule. The format exists, keyed by `NodeId` outside the op, so a lowering that needs it gets a side table or picks one in code. Until each bound operand carries its `NumericFormat`, a format cannot reach a kernel as a parameter of the op. `BlockQuantize` is the one kind that carries a layout (`proxima-tensor/src/bind/types_layout_boundop.rs:674`), and its own docs say it has no `NumericFormat` (`:793-794`).

## 6. Lowering is a transform into an instruction set

Lowering reads a `BoundOp` and writes text in a target language. Two shared pieces already do this for numeric formats.

**`Dialect`** (`omega/src/render/dialect.rs:28`) spells tokens only: scalar and integer type names, literals, casts, bit reinterpretation, five intrinsics (`Intrinsic`, `omega/src/render/dialect.rs:15`), a function head, a declaration, a select, a branch. Three impls:

| target | type | impl |
|---|---|---|
| Metal | `Msl` | `omega/src/msl/dialect.rs:19` |
| CUDA and HIP | `Cuda` | `omega/src/backend/cuda/dialect.rs:61` (one dialect; only the bfloat name differs, `:1-7`) |
| WGSL | `Wgsl` | `omega/src/backend/wgpu/dialect.rs:30` |

**`render::load` and `render::store`** (`omega/src/render/layout.rs:354`, `:551`) take a `FloatLayout` and render the function that decodes or encodes its code in integer steps, generic over `D: Dialect`. They are the CPU algorithm in `proxima_numeric::float_layout_codec::{decode, encode}` rendered step for step (`omega/src/render/mod.rs:1-8`). Metal calls them at `omega/src/msl/layout_render.rs:145,155`; CUDA at `omega/src/cuda_layout.rs:28,44`. The only `Wgsl` callers are tests (`omega/src/layout_text_tests.rs:83,87`). `render::Refusal` says why a layout cannot be spelled (`omega/src/render/refusal.rs:5`).

That is the target shape: a backend is a `Dialect` plus shared steps. Machines differ in facts, not code. Matrix-unit shapes and the formats they accept, wave width, shared-memory size and async copy are device data that the shared lowering reads. The instruction words are the dialect's spelling. If a backend lags its incumbent, the fix is a missing fact or schedule transform in the shared lowering, never a per-backend kernel.

Not everything has reached that shape yet. The map of debt:

| not yet a transform | where |
|---|---|
| `ScalarOp` to text is written three times; `Dialect` has no method for it | `omega/src/wgsl.rs:753`, `omega/src/cuda.rs:2380`, `omega/src/msl/signature_tokens_prelude.rs:632` |
| each backend's own elementwise body | `omega/src/msl/elementwise_reduce_core.rs:3`, `omega/src/wgsl.rs:1334`, `omega/src/cuda.rs:3114` |
| each backend's own reduce body | `omega/src/msl/elementwise_reduce_core.rs:325`, `omega/src/wgsl.rs:1425`, `omega/src/cuda.rs:4424` |
| fused forms rendered per backend; WGSL declines them (`UnsupportedOpKind`) | Metal `omega/src/msl/cached_attention_render.rs`, CUDA `omega/src/cuda_cached_attention.rs`, WGSL `omega/src/wgsl.rs:237` |
| `Plan::Cpu` wraps the proxima-tensor evaluator: `CpuPlan` keeps the program and calls it | `omega/src/backend.rs:581`, `:1324`, `:2188`; evaluator `proxima-tensor/src/cpu/quantized_eval.rs:1883` |
| per-format kernels inside that evaluator | `proxima-tensor/src/cpu/gemm_dot_quant.rs:1308` (`matmul_q4k_f32`), `proxima-tensor/src/cpu/gemm_q8_0_act.rs:813` |

Each row is a place where a format or an op is spelled in backend text instead of rendered from the shared pieces. New work does not add a row (`AGENTS.md`, "No bespoke paths").

## 7. Before you change a lowering

1. Which layer expresses it: a `ScalarOp` body, a `Layout`, an `Elementwise` or `Reduce`, or one of the eight fused forms?
2. Which shared piece renders it: `Dialect`, `render::load` or `render::store`? If none exists, build that piece; a per-backend kernel is not an interim.
3. Where does each format come from? A bound tensor or codec, passed as a parameter. Never a name, a constant, or a search for the widest or narrowest.
4. Which incumbent number is the acceptance (llama.cpp, whisper.cpp, sd.cpp)? See [07](./07-comparisons.md).
5. What bespoke code does the change delete? Name it, with a row from the debt map above.
