# proxima TODO

## Payload layer — no-cell design (2026-07-07)

Status: **design settled, not code.** This note is a public design backlog,
not part of the shipped API contract.

### The decision
The pipe algebra moves **`P` and nothing else** — a pipe is `P -> Q`.
No `Envelope` / `Payload` container cell. No `Body` type.

- **Metadata is a capability, not a slot.** Concerns are composable wrapper types
  `W<P>` — `Header<P>`, `Trailer<P>`, `Framed<P>`, `Sealed<P>` — and each is
  *itself a `P`*, so it pushes into the next layer through the same `P -> Q`
  pipes with zero special-casing.
- **Presence** = `P` (empty is `P = ()`, absence, not a variant).
- **Delivery** = the *type* of `P` (streamed = `P` is a stream type), never an enum arm.
- **Nesting = protocol layering.** HTTP's control-headers vs `Content-*`
  representation-headers split = outer-wrap vs inner-wrap. gzip = `map_inner`
  on the inner wrapper — structurally cannot touch control headers.

### Load-bearing constraint (the thing that can break it)
Wrapper **forwarding**: a pipe that doesn't care about the header must see
through `Header<Bytes>` to `Bytes`, or every wrapper becomes unwrap-boilerplate.
- functor `map_inner` (transform inner, wrapper preserved), and/or
- `Deref<Target = P>` transparency.

### Naming (owner-settled)
- reject `Envelope` (wrapper/contents metaphor breaks under self-similar nesting;
  fights already-landed `body -> payload` vocabulary)
- reject `Metadata` (weasel-word)
- content is `P`; wrapper noun TBD (`Header` favored)
- names (`Body`, `Request`, `Response`) = type aliases; newtype ONLY to enforce an invariant

### Open sub-questions
- [ ] `context` — ambient (context var) vs field
- [ ] dynamic-depth boundary — where the type bottoms out to opaque `Bytes` + `Box<dyn>` at the edge

### Next action — prove it before any code
- [ ] Model **multipart**, a **CONNECT tunnel**, and **h2 frames** as nested wrapper
      `P`s against how proxima does them today. Confirm: no loss, no per-chunk
      metadata, and no pipe hand-unwraps (forwarding holds).
- [ ] Only then scope the migration (blast radius: replaces main's fat
      `Request<P> { method, path, metadata, payload, stream, context }` — every
      listener/upstream/middleware).

---

## Doc consolidation — comb + collapse every .md

**297 tracked `.md` files** (147 under `docs/`). Comb through all of them and
systematically collapse them **one by one** into their precise needed form —
merge overlapping discipline logs, drop superseded/landed records, keep only
what's load-bearing. Do it per-file, not in a bulk sweep.

- [ ] `docs/` (126) — the bulk; per-subsystem discipline logs, likely heavy overlap
- [ ] top-level (`SHAPE.md`, `FEATURES.md`, `parking-lot.md`, `README.md`)
- [ ] per-crate `*.md` (telemetry 5, intercept 5, benches 8, examples 5, ai_docs 3)
- [ ] `spec/`, `scenarios/`

---

## RFC 9112 — authoring form, and the half-done citation migration (2026-07-16)

Two items that look like one. Neither is built.

### 1. The message syntax exists on the wire, not as an authoring surface

proxima already encodes and parses HTTP/1.1 message syntax —
`proxima-protocols/src/http1_codec/h1_client.rs` (`encode_request_head`,
`parse_response_head`) and `h1_body.rs` (chunked framing). What does not exist
is that same syntax as a thing a human *writes* to drive the client:

```
GET /path HTTP/1.1
Host: example.com
Accept: application/json

<body>
```

The codec that reads this shape is already here, so the seam is wiring rather
than new machinery. `.http` / `.rest` files (JetBrains HTTP Client, VS Code
REST Client) are the established convention for the same text, so the form has
prior art and tooling.

Unresolved: where it attaches. An `H1ClientConfig` variant? A `Pipe` whose `In`
is the raw text? A `Spec` form? Config and code are isomorphic here, so the
choice decides whether a request-as-text is config, input, or spec — pick
deliberately.

### 2. The RFC citations mostly point at superseded documents

RFC 9110 (Semantics) and RFC 9112 (HTTP/1.1) replaced RFC 7230/7231 in June
2022. Counts in the tree today:

| cited | count | status |
| --- | --- | --- |
| RFC 7230 | 24 | superseded by 9112 (syntax) + 9110 (semantics) |
| RFC 7231 | 6 | superseded by 9110 |
| RFC 9110 | 6 | current |
| RFC 9112 | 1 | current |

The implementation follows the current syntax; the citations largely do not.
Someone began the migration and stopped. Section numbers were reorganised
between the old and new documents, so this is a read-and-remap job per
citation — a find/replace of the RFC number alone would produce confidently
wrong section references, which is worse than the stale ones.

---

## DynPipe / SendDynPipe — the last two blanket impls (2026-07-17)

Status: **open design question, deliberately unresolved.** `algebra-lint`
flags both and should keep flagging them until this is answered.

`proxima-primitives/src/pipe/alloc_tier.rs` erases a pipe for `dyn` dispatch
through two traits — `DynPipe`, `SendDynPipe` — each a restatement of `Pipe`
with `In`/`Out` demoted to generics, `Err` pinned to `ProximaError`, and the
future boxed. They are implemented as blanket impls over an open set (`impl<P:
Pipe> DynPipe for P`), which the no-blanket rule forbids.

Both remedies are refused, and that is the finding:
- deleting the blanket impl needs a type to host the impl instead — that
  newtype IS the blanket impl renamed (`Erased<P>`, tried and reverted in
  `39e7c035`/`bd69cb5f`: `into_handle(pipe)` was the identical call site
  before and after, so nothing was gained).
- so the defect is upstream: the two traits themselves. The question is
  whether erasure can key off `Pipe` directly (or whether `Pipe` can be made
  object-safe enough) so the restatement disappears, rather than being
  relocated.

Note `alloc_tier.rs` already asserts the equivalence it is compensating for,
ten lines below the impls: `impl<In, Out> SendPipe for dyn SendDynPipe<In,
Out>` — the erased handle IS a pipe of the same form. That round-trip is the
evidence the two traits are redundant, not the justification for them.

Do not "fix" this by exempting it in the lint or by adding an adapter. It is a
real design decision about the erasure boundary.

## Also pending
- Review pre-public cleanup branches and cherry-pick only still-relevant changes.
- Keep scratch worktrees and private assistant state out of public commits.


## NPU lanes for omega (2026-10-08)

Status: **owner's ask, researched, not spec'd.** omega lowers to Metal today;
NPUs are the next placement lanes. Placement is data (the WHERE-pipe), never a
model branch. Sources for every number: `docs/research/npu-lanes-census-2026-10-08.md`.

### Lane census (sourced; "unsourced" where no primary source was found)
| lane | entry | kind | quant | measured floors |
| --- | --- | --- | --- | --- |
| ANE via CoreML | `MLComputeUnits.cpuAndNeuralEngine`, coremltools | compiled model, placement fixed at compile | fp16 native; palettization 1-8 bit; W8A8 on A17 Pro / M4; GGUF blocks direct: no source | dispatch floor: no CoreML number found; IOSurface `MLMultiArray` zero-copy: unsourced |
| ANE via private API | `_ANEClient` / MIL to E5 microcode (private, not shippable) | compiled kernel | fp16; int8 contested between sources | 0.095 ms dispatch, 2.3 ms IOSurface round trip per dispatch, 5.76 ms/tok decode (Orion); fixed `[1,C,1,S]`, ~119 compiles per process then silent failure |
| Hexagon, raw FastRPC | IDL + QAIC stub/skel, `rpcmem` (ION/dma-heap), Hexagon SDK | raw kernel (ggml-hexagon works this way) | Q4_0/Q8_0/MXFP4 repacked into DSP buffers; HMX detail unsourced | ~100 us per call, 82-92 us tuned, 0.2 ms per forward in ggml-hexagon (community numbers); ~3.5-4 GiB address space per session |
| Hexagon, QNN / AI Engine Direct | QNN C API, context binary, ONNX Runtime QNN EP | compiled graph, HTP-specific | u8/u16 matmul, fp16; 16a4w block in ExecuTorch; GGUF direct: unsourced | INT8 graph execute 3.4-3.5 ms (V79, community); GPU-NPU sync ~400 us (HeteroLLM) |
| Hexagon, TVM | `tvm.target.hexagon`, launcher over FastRPC | compiled kernel | HVX via LLVM intrinsics; HMX/block-quant unsourced | launcher runs one layer at a time, no number |
| Hexagon, `qualcomm/hexagon-mlir` ("hexagonmlir") | Triton and torch-mlir front ends; v73-v81 | compiled kernel, TCM mega-kernels with DMA | fp16/fp32 in the paper; no int4/int8, no end-to-end LLM result | none reported |
| Intel NPU | OpenVINO NPU plugin on Level Zero | compiled model | INT4-FP16 group-wise; INT8 weight-only unsupported | static shapes only; prefill chunk 1024 |
| AMD XDNA | MLIR-AIE / IRON, `xclbin` + insts | raw kernel | int8/int16/bf16; bfp16 on XDNA2 | 54.8 ms/tok Qwen3-0.6B after cutting configures 366 to 170 (community) |

"exagonrpc": no project of that name; nearest is Hexagon FastRPC. "raw-fastrpc":
FastRPC with a hand-written IDL and skel, no QNN, which is how llama.cpp's
ggml-hexagon backend runs (one RPC per forward pass).

### The first use: shunt an expert
CoreML (and Hexagon) as a lane an MoE expert, or a set of experts, is shunted
to, so the GPU and the NPU run experts concurrently. Routing is on the GPU, so
the shunt is per layer or static per expert, not per token. The floors above
say the granule must be whole-expert or whole-layer: every lane's dispatch is
0.1-3.5 ms, against a 10-40 us Metal dispatch.

### Shape 1: split by stage (front on the NPU, expert FFNs on the GPU)
No paper found that puts attention and router on an NPU and expert FFNs on a GPU
with a per-token handoff number. Nearest measured:
- NPUMoE (Apple silicon, CoreML, arXiv 2604.18788): attention and hot experts
  on the ANE, routing/top-k/norms/cold experts on the CPU, GPU unused; latency
  1.32-5.55x lower than the baselines; CPU-NPU sync is over 60% of runtime in
  the worst case; naive CoreML spends 4.4-5.7x more energy on data movement.
- HeteroLLM (Snapdragon 8 Gen 3, arXiv 2501.14794): GPU-NPU sync ~400 us fixed;
  concurrent bandwidth 43.3 to 59.5 GB/s; W4A16 unsupported on the NPU at decode.
- llm.npu (ASPLOS'25): QNN lacks KV cache, SiLU, RMSNorm, RoPE; prefill 22x.
First measurement on this box: the handoff cost per token between a Metal
buffer and a CoreML input, both directions, before any graph is split.

### Shape 2: the NPU predicts the route (pre-gating / expert prefetch)
None of these run the predictor on an NPU; the signal and the hit rate are the
transferable parts.
- Pre-gated MoE (ISCA'24): a learned pre-gate in block N picks block N+1's
  experts from block N's activations; 1.7x latency vs on-demand, 42x vs prefetch;
  hit rate not reported, accuracy "comparable".
- SiDA-MoE (MLSys'24): offline LSTM over token embeddings predicts all active
  experts; top-3 hit 91.7-99.0%; up to 3.93x throughput, 80% GPU memory saved.
- EdgeMoE: offline table keyed on the two previous layers' activations; one
  example at 87.1% hit; top 20% of activation paths cover 99% of activations;
  1.19-2.77x.
- MoE-Infinity: request-level expert activation traces matched by cosine;
  2.7-16.7x per-token latency vs offloading baselines; no hit rate.
First measurement on this box: hit rate of a SiDA-style or EdgeMoE-style
predictor against the real top-k on the granite route captures already in
`proxima-tensor/specs/decode-prefill-parity/evidence/attr3/`, before any NPU
work; the prefetch pays only if hit rate times expert time exceeds the lane's
dispatch floor.

### Measure before any slice
- [ ] CoreML prediction dispatch floor on this box (no published number exists)
- [ ] Metal buffer to CoreML input and back, per token, zero-copy or not
- [ ] what a Q4_0 expert becomes on each lane (palettized / W8A8 / repacked blocks)
- [ ] route-prediction hit rate on the captured granite routes

### Tiered MoE, what exists (sourced, `docs/research/tiered-moe-2026-10-08.md`)
- Gating with 3+ levels in an LLM: none found. DeepSeek-V3 is a 2-stage select
  (node set, then top-8 of 256) plus one shared expert; PEER is 2 sub-key sets.
- Expert classes, 3+: MoE++ has 4 (FFN, zero, copy, constant); NPUMoE has 3
  static capacity tiers from offline popularity.
- Placement tiers, 3: HOBBIT (GPU hi/lo precision, DRAM, SSD; next-layer
  predictor top-1 ~96%), PowerInfer-2 (NPU hot, CPU cold, flash; 95% cache hit),
  NPUMoE (ANE hot experts + attention, CPU router/top-k/cold experts, GPU idle).
- Router on a different device than the experts: NPUMoE (CPU router, ANE
  experts) and ProMoE (CPU predictor ~200 us, GPU experts). An NPU router with
  GPU experts: none found. That is the open cell the ANE cards measure.
- Cards: `docs/bench-campaigns/2026-10-08-ane-lane/plan.md`.

### LiteRT (owner, 2026-10-08; from memory, unverified)
Google's on-device runtime (the TensorFlow Lite rebrand; LiteRT Next API;
LiteRT-LM for LLMs). A compiled-model lane like CoreML and QNN: flatbuffer
models, acceleration by delegate (GPU, CoreML, vendor NPU paths compiled
ahead of time per SoC). Not a kernel lane, so not something omega runs on.
Three possible relations, none started: (1) a backend lane on Android, the
public route to Qualcomm and MediaTek NPUs without the vendor SDKs; (2) a
`.tflite` codec in the any-to-any format goal, after GGUF and safetensors;
(3) an incumbent (LiteRT-LM phone decode numbers) for an Android build. On a
Mac it adds nothing CoreML does not. Verify the current API surface and the
NPU delegate list before any card.

## MLIR and omega (owner question, 2026-10-11)

Recorded so the position is in the tree, not only in session memory. Status: no slice, no card before this one.

- Positioning (owner, 2026-09-09): proxima-tensor's algebra sits in the MLIR Linalg / Halide / TVM tradition: a generic expression (indexing maps + scalar body, parallel and reduction dims), with the schedule separate from the algorithm. A specialized executor must account for the whole composed expression and its addressing, never recognize a familiar fragment. Omega is that lowering for Metal today; CPU, wgpu and cuda back ends exist with narrower codec coverage.
- Where MLIR already appears: as other compilers in the NPU lane census above (`qualcomm/hexagon-mlir`, AMD MLIR-AIE / IRON). Those lanes consume MLIR; omega does not emit it.
- Two directions worth a card each, neither started:
  1. Export: omega IR -> MLIR text (Linalg generic + tensor/memref, or StableHLO) so MLIR-consuming back ends (hexagon-mlir, MLIR-AIE, IREE) become lanes without a bespoke driver each. First measurement: take one captured decode step (granite or gemma4 E2B, the same captures the parity rounds use), emit it as Linalg text, and count the ops that have no faithful Linalg form. Expected gaps to count, not assume: the block-quantized codecs (Q4_0, Q8_0, bf8, bf4, fp6 e3m2 are byte layouts Linalg has no type for), gather-based MoE routing and the expert slab, KV residency and the hazard tracker's barrier policy, the 32-bit index class. The count decides whether export is a lane or a paper.
  2. Import: MLIR (torch-mlir / StableHLO) -> omega IR, so a PyTorch graph lowers through omega's existing codec and residency machinery. First measurement: how many ops of a torch-mlir dump of the same two models map onto omega's op set with no new op.
- Non-goals until a measurement says otherwise: no MLIR (LLVM) dependency in the core crates; textual emission and parsing only; no dialect registration.
- Unsourced: whether hexagon-mlir or MLIR-AIE accept Linalg text at the level omega would emit (their user guides were read for the lane census, not for this question).

## Interop doors: any format in, any hardware out (owner frame, 2026-10-11)

Owner: "from a gguf or from a onnx or any other format - the point here is interop and also GENERIC and proxima pipes with fsm." One generic IR (proxima-tensor's algebra, lowered by omega) with many doors; every door is a proxima pipe over that IR with an FSM for its stages, measured by the same oracles (recorded llama ids for tokens, vendored reference bytes for formats); no format- or vendor-specific code outside its pipe. State at `f82e9ab2`:

| door | direction | state | note |
|---|---|---|---|
| GGUF | in | landed | the primary reader; e3m2 blocks ride under a proxima-own type id (65568) that ggml consumers reject |
| HF safetensors | in | landed, interim | dense leaves; granite MoE stacks bound as views (down) and one bf16 copy per expert (gate/up); peak RSS 3.24 GB vs 26 MB for the lazily mapped GGUF, zero-copy fused binding in the expert slab still owed; dense HF alias set and rank-3 weights owed for the round trip |
| MLX plain (bf16/f16/f32) | in | via the HF path | same gaps as HF |
| MLX affine quantized | in | refused by design | `MlxQuantizedNotLoadable`; native affine-with-bias blocks are a kernel codec, a separate spec |
| ONNX | in | none | carries a computation graph, not named tensors (`proxima-model-interop/src/lib.rs:14`); same problem class as the MLIR import direction above: graph -> omega IR, measured by ops mapped with no new op |
| GGUF | out | landed | writer used by the e3m2 round trip |
| HF safetensors | out | landed | shapes HF row-major since `0fb46a31` |
| MLX plain and affine 4-bit directories | out | landed, bytes only | shards + index + `config.json` (every GGUF-derivable key) + tokenizer files; never yet loaded by mlx-lm (installed in a venv on this box; the integration check for MLX slice 13) |
| Metal | hardware out | landed | every codec including the e3m2 render (row-blocked, tiled, grouped) |
| CPU | hardware out | landed | Rust arms per codec |
| cuda, wgpu | hardware out | builds; narrower | CUDA C / WGSL text emission compiled by the vendor toolchain; bf8/bf4/fp6 typed-rejected; no parity run on this box |
| CoreML / ANE | hardware out | none; census + cards | floor 0.27 ms per predict; CPU predictor 81 us beats it; top-k/gather/scatter unsupported on the M1 ANE; cards 0.1, 1.1, 5.1 filled (`docs/bench-campaigns/2026-10-08-ane-lane/plan.md`) |
| Hexagon (fastrpc, hexagon-mlir, QNN), AMD XDNA, LiteRT | hardware out | none; census only | `docs/research/npu-lanes-census-2026-10-08.md` |
| LLVM IR | hardware out | none | omega has no llvm/inkwell/cranelift dependency; an LLVM lane would be the same shape as the MSL/WGSL/CUDA emitters (text out, vendor toolchain compiles) and would reach CPU via llc and Hexagon via the lowering end hexagon-mlir already uses ("LLVM-IR with runtime calls") |
| MLIR (Linalg / StableHLO) | both | none; card above | export counts unrepresentable ops first; import counts unmapped ops first |

Pipe/FSM shape today: the transforms are sans-IO functions (`In = ParsedGguf -> Out = MlxExport`, and the inverse), composed by callers; they are not yet expressed as proxima pipes with an FSM over the stages (parse, bind, transform, pack, assemble, write). That expression is the next structural slice for the formats layer and is what makes a new door a config composition rather than a new crate.
