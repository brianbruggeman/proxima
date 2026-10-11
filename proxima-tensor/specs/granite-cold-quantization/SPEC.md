# granite-cold-quantization

status: complete
owner: brian
created: 2026-10-10

## problem

Proxima cannot reproducibly convert a pinned Granite 4 BF16 checkpoint into E5M2, E3M2, or E2M2 tensor artifacts, exercise scalar and scaled representations, and compare reconstructed tensor values against the pinned source with retained evidence. Conversion, scaling, comparison, and evidence capture must run without Python.

## refutation condition

The work is unnecessary if Proxima already converts the pinned checkpoint into all requested formats, records per-tensor scale metadata, reconstructs every artifact against the source payload, and retains the raw comparison evidence without Python.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Pin Granite 4 Micro BF16 weights, config, tokenizer, source revision, and hashes as the immutable source checkpoint. | yes |
| R2 | Provide a reproducible native Proxima command path for all supported cold-conversion and source-comparison runs; Python is not a runtime or build dependency. | yes |
| R3 | Encode/decode E5M2, E3M2, and E2M2 values with explicit rounding, overflow, NaN, subnormal, and packed-bit-order contracts. | yes |
| R4 | Convert F32 and BF16 tensor payloads to each selected scalar format with shape preservation, per-tensor hashes, and a report that distinguishes bytes read/written from derived sizes. | yes |
| R5 | Proxima compares every converted tensor with its F32 or BF16 source payload, retaining per-tensor changed/zeroed/saturated counts, max-error source/reconstruction pairs, and first-eight raw value pairs. | yes |
| R6 | Support an explicit per-tensor maximum scale mode and persist the scale beside each tensor record; scalar mode remains separately selectable. | yes |
| R7 | Verify every output file hash and byte length against the manifest, and verify all pinned source-file hashes against the checkpoint directory. | yes |

## architecture

The source is `ibm-granite/granite-4.0-micro-base` at revision `98e1d641451818f6c8e6b17c1c359b5b4ac495d0`, sourced as BF16 safetensors. Proxima owns conversion, artifact IO, tensor reconstruction comparison, and evidence capture in this card. A converter consumes a named tensor's source dtype, shape, and bytes and writes caller-owned packed bytes plus a report record; E3M2 and E2M2 bit order and numerical behavior are fixed by golden vectors. The source checkpoint remains immutable. Converted artifacts live under a distinct experiment directory and include the source revision and SHA256 in their manifest. Model loading, inference, MLX comparison, and prefill timing are separate work and are not acceptance criteria here.

The initial comparison boundary is reconstructed tensor payloads against the exact BF16/F32 source values. This isolates scalar-format and scaling behavior before model-runtime support is considered. Model-logit and prefill comparisons are outside this cold-conversion card. Python MLX-LM is excluded.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| first Granite 4 target | Micro base, all-attention topology | isolates attention and weight-format behavior before the hybrid Mamba/MoE state path |
| source checkpoint | pinned BF16 safetensors revision | do not requantize a prequantized GGUF and attribute its prior error to the new format |
| runtime | Proxima performs conversion and tensor comparison without Python | a reference runtime cannot stand in for Proxima's own artifact and codec behavior |
| precision names | `e5m2`, `e3m2`, `e2m2` in machine records; human aliases BF8/BF6/BF5 are metadata only | prevents a short label from silently selecting a different bit contract |
| MSL dispatch | preserve the existing configured attention selector; quantization and dispatch are separate experiment axes | changing storage format and threadgroup schedule together would make either result unattributable |

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1,R2 | `RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo run --release -p proxima-model-interop --features std --bin granite-cold-quantize -- /private/tmp/proxima-granite4-micro-bf16-hf /private/tmp/granite4-cold-e5m2-release e5m2 32768` | native Rust command hashes pinned config/tokenizer/shards, converts every indexed source tensor, and writes a manifest with input/output hashes and byte counters |
| AC2 | R3,R4 | `RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo test -p proxima-gguf --lib quant::fp -- --nocapture` and `RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo test -p proxima-model-interop --features std --lib cold_artifact::tests -- --nocapture` | codec tests report nonzero counts; the four focused artifact tests include BF16 and F32 source tensors and exact reconstruction comparisons |
| AC3 | R4 | Run AC1 serially with `e3m2` and `e2m2` and distinct new output directories | all indexed source tensors are emitted; each output manifest records source shape, input/output hashes, and bytes processed/written |
| AC4 | R5,R6 | Run the native `granite-cold-quantize compare` command serially for E5M2/E3M2/E2M2 in scalar and `tensor_maximum` modes; run `RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo test -p proxima-model-interop --features std --lib cold_artifact::tests -- --nocapture` | each JSON report has 322 per-tensor rows, raw first-eight value pairs, maximum-error source/reconstruction values, and aggregate payload counters; the focused command reports 4 passing tests |
| AC5 | R7 | `bash proxima-tensor/specs/granite-cold-quantization/evidence/verify-converted-output-hashes.sh` | all six artifacts' output hashes/lengths match their manifests, and all 10 source artifact hashes match the pinned checkpoint |

## out of scope

- CUDA, generic dtype registration unrelated to these three scalar formats, or model families other than Granite 4 Micro.
- Python-based conversion or inference; treating MLX affine quantization, MXFP8, or mixed-bit recipes as equivalent to scalar E5M2/E3M2/E2M2.
- Changing attention tile height, simdgroup count, K/V reuse, prefetch, or MMA precision in the same experiment as weight quantization.
- Reporting a model quality or performance win from tensor reconstruction aggregates; those require a separate native model execution experiment with output payload evidence.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| BF6/BF5 aliases hide distinct rounding or exponent rules | high | incompatible weight bytes | name the binary formats explicitly and retain golden bytes before whole-model conversion |
| Tensor reconstruction differences are mistaken for model behavior | high | unsupported quality or runtime inference | keep this card's claims at tensor-payload level and require a separate native Proxima invocation for model-output claims |
| Granite 4 hybrid-family tags are mistaken for model support | medium | a model parses but cannot execute | keep model loading and inference outside this conversion card |

## context

- `proxima-tensor/specs/granite-attention-numeric-matrix/SPEC.md` defines the existing Proxima E5M2 scalar contract and explicitly places model-weight quantization outside that spec.
- `proxima-gguf/src/quant/bf8_e5m2.rs` and `proxima-gguf/src/quant/bf4_e2m1.rs` contain the current scalar format helpers; `proxima-model-interop/src/bind.rs::quantize_to_kind` currently returns `UnsupportedCodec` for these custom formats.
- `proxima-model-interop/src/hf_bind.rs` binds fixed-width safetensors through a dense model path; this card's scope is native tensor conversion and fidelity evidence, not model execution.
- The required workflow and evidence use native Proxima commands without Python. No Granite 4 inference baseline is included in this tensor-conversion card.
- The official Granite 4 Micro GGUF file is available at the pinned inventory revision, but the cold-conversion source of truth for this work is the BF16 safetensors checkpoint, not that GGUF.
