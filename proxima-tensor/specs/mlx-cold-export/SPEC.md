# mlx-cold-export

status: draft
owner: Brian Bruggeman
created: 2026-10-10

## problem

proxima-model-interop cannot write an MLX model directory for granite 3.1 1b moe or gemma4 E2B (plain MLX safetensors, or mlx-lm's affine-quantized layout), measured by byte agreement of tensor names, shapes, dtypes, packing, config keys and tokenizer files with mlx-community's conversions of the same models and by a greedy round trip back through proxima that reproduces the recorded llama ids.

## refutation condition

mlx-community's conversion of the same source model (a reference directory read as bytes,
never run) has tensor names, shapes, dtypes, packing, config keys or tokenizer arrays that this
export cannot reproduce for granite 3.1 1b moe or gemma4 E2B, or a plain export read back by
proxima does not reproduce the recorded llama greedy ids. Either refutes the export as
MLX-compatible; it would then be a proxima-only format and must say so.

Oracles: the vendored mlx-community files (bytes, never executed; two conversions per model, a
plain bf16 one and a 4-bit affine one, both recorded in the fixture README) and the recorded
llama ids in `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`
(fields `prompt_ids`, `generated_ids`), `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json`
(`prompt_ids`) and `proxima-model-interop/tests/fixtures/gemma4_e2b_llama_greedy_ids.json`
(`prompt_ids`, `oracle_ids`). proxima's own tokenizer output is never an oracle. The workspace
runs no Python; MLX, Ollama and llama-server are never executed.

Controls, by kind of check. A comparison against a vendored or recorded external file carries an
identity control (the reference compared to itself through the same comparison function passes)
and a degenerate control (a copy altered in one place fails). A comparison against a hand-derived
or transcribed value carries a degenerate control (the expected value altered in one byte fails).
A structural assertion against a constant stated in its own AC (a shape, a dtype, a file name, a
default, a grid) carries none. Each control is a counted test case inside its AC.

## requirements

Each R is one consumer-visible behaviour; its single AC enumerates the named test cases that
exercise it. The Rs fail independently of one another: each has a different producer module or a
different reference file.

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Shard assembly: the export cuts tensors into safetensors shards under mlx-lm's cap, names them, tags them `__metadata__ {"format": "mlx"}` and emits the index file when there is more than one shard (`transform/mlx.rs`, new) | yes |
| R2 | Dense name mapping: every dense GGUF leaf of both models maps through the binding profile in reverse (`BindingProfile::aliases_for`, `profiles/binding.rs:90`) to the tensor name the vendored plain header uses; an unknown leaf is reported, never invented | yes |
| R3 | MoE binding: a `granitemoe.toml` binding profile under `proxima-model-interop/src/profiles/binding/` and an MoE arm in `hf_bind` bind routed expert stacks and routers to literal HF names, while a dense profile still refuses MoE leaves with `HfMoeWeightsUnsupported` (`hf_bind.rs:921`, `error.rs:486`) | yes |
| R4 | MoE name mapping: every granite MoE leaf maps to the tensor name the vendored plain header uses | yes |
| R5 | Plain export: float tensors are emitted as bf16 (default), f16 or f32, and Q8_0 and Q4_0 tensors are dequantized to f32 through `proxima_gguf::quant::dispatch::dequantize` before the float cast | yes |
| R6 | Affine packing (mlx-lm affine mode): `pack_affine` produces `weight` uint32 words, `scales` and `biases` with the layout, shapes, parameter grid, default and float fallback mlx-lm uses, and the edge-anchored arithmetic of mlx `quantize` (`mlx/backend/cpu/quantized.cpp:1165-1233` at 06eb7483): the group edge of larger magnitude anchors the bias, the scale is negative when that edge is the maximum, dequant is `w = scale * q + bias` | yes |
| R7 | Config writing: `transform/mlx_config.rs` (new) writes the HF `config.json` from GGUF metadata, with `quantization` keys on quantized exports only, and copies `config.json` for HF-directory sources | yes |
| R8 | Tokenizer writing: `proxima-tokenizer` (new `hf_write.rs`) serializes a `Vocab` to `tokenizer.json` and `tokenizer_config.json` and agrees field by field with the vendored mlx-community files | yes |
| R9 | Tokenizer export: the interop writes the tokenizer files for a GGUF source and copies them for an HF-directory source; the written files agree with the vendored reference and encode the fixture prompt to the recorded llama ids | yes |
| R10 | Round trip: a plain export read back by proxima's safetensors loader generates greedily the recorded llama ids | yes |
| R11 | Reference header agreement: the plain and (4, 64) affine exports' tensor names, shapes, dtypes and config keys agree with the vendored plain and affine4 references | yes |
| R12 | Packing agreement: the `weight`, `scales`, `biases` the export packs at (4, 64) from the vendored plain bf16 tensor are bit-equal to the vendored affine4 reference's | yes |
| R13 | CLI export: a CLI subcommand writes the directory for a (4, 64) affine export of each model, and the exported tensor count agrees with the source GGUF header and the affine4 reference | yes |
| R14 | Loading an mlx-lm quantized directory back into proxima is refused with `InteropError::MlxQuantizedNotLoadable` (new) naming the tensor (native affine import is a separate spec) | yes |

## architecture

```
proxima-model-interop/src/transform.rs                      // add `mod mlx; mod mlx_names; mod mlx_config;` beside `mod tests;` (transform.rs:153; src/transform/ holds only tests.rs today)
proxima-model-interop/src/transform/mlx.rs                  // new: MlxExport, export_plain, export_affine, pack_affine, shard assembly
~~proxima-model-interop/src/transform/mlx_affine.rs      // affine packing (pack_affine, export_affine, AffineParams) lives here, not in mlx.rs~~
proxima-model-interop/src/transform/mlx_names.rs            // new: reverse of hf_bind through BindingProfile::aliases_for, dense and MoE
proxima-model-interop/src/transform/mlx_config.rs           // new: config.json writer sharing a field table with the hf_config reader
proxima-model-interop/src/profiles/binding/granitemoe.toml  // new binding profile (the family profile src/profiles/granitemoe.toml, registered at profiles/mod.rs:30, is a different file and is untouched)
proxima-model-interop/src/hf_bind.rs                        // MoE arm
proxima-tokenizer/src/hf_write.rs                           // new: tokenizer.json and tokenizer_config.json writer from Vocab (feature hf)
tools/proxima-cli/src/main.rs                               // Command gains a Model variant: ModelCommand::ExportMlx with model, out, quantize, dtype
proxima-model-interop/tests/mlx_cli_export.rs               // new: the integration test behind AC13
proxima-model-interop/tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/plain/    // new: config.json, tokenizer.json, tokenizer_config.json, safetensors header JSON, one bf16 tensor
proxima-model-interop/tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/affine4/   // new: config.json, safetensors header JSON, the same tensor's packed weight, scales, biases
proxima-model-interop/tests/fixtures/mlx-reference/gemma4-e2b/plain/                   // new: as above
proxima-model-interop/tests/fixtures/mlx-reference/gemma4-e2b/affine4/                  // new: as above
```

`plain/` is mlx-community's bf16 conversion of the model and `affine4/` is its 4-bit conversion
(group size 64, mlx-lm's affine default); the README in each model directory names both
upstream repos and their revision shas (the upstream repo names are recorded there when the files
are fetched; the directory names above are proxima-side names). The float source for R12 is the
bf16 tensor in `plain/`, so the packing input is an external file, not a dequantized Ollama GGUF.

Signatures, in words: `export_plain` takes a `ParsedGguf` and the file bytes it was parsed from
(the pair `gguf_to_safetensors` takes, `transform.rs:53`), a `BindingProfile` and a
`proxima_tensor::DType` (`proxima-tensor/src/dtype.rs:17`, restricted to BF16, F16 or F32 and
refused with a typed error otherwise) and returns an `MlxExport` or an `InteropError`;
`export_affine` takes the same with `AffineParams` in place of the dtype; `pack_affine` takes an
f32 slice, `out`, `inp`, `AffineParams` and a caller buffer. `MlxExport` is new and holds the
shards (`proxima_safetensors::SafetensorsModel` values, `proxima-safetensors/src/writer.rs:33`,
borrowing tensor payloads), the optional shard index, the config and the optional tokenizer files.

Dequantization is one entry point: `proxima_gguf::quant::dispatch::dequantize(kind, data,
output)` (`proxima-gguf/src/quant/dispatch.rs:16`, `pub`, with Q4_0 at `:24` routing to
`q4_0::dequantize` at `proxima-gguf/src/quant/q4_0.rs:111` and Q8_0 at `:31`), a dependency
`proxima-model-interop` already has (`proxima-model-interop/Cargo.toml:339`). The proxima-tensor
CPU paths are not used: `codec_dequantize_fn` (`proxima-tensor/src/cpu/epilogue.rs:1452`) is
`pub(super)` and returns `None` for Q4_0, Q5_0 and Q4_1, and `dequantize_row`
(`proxima-tensor/src/cpu/run_node.rs:2083`) is `pub(super)` and returns an error for Q4_0 and
Q5_0. No proxima-tensor surface is added.

No new trait. `AffineParams { bits: u8, group_size: u32 }` is the one new value type, a pair
of integers the config and CLI both carry; the transform is a pipe (`In` the `ParsedGguf` plus its bytes,
`Out = MlxExport`) expressed as plain functions. Packing is bytes-first into caller buffers;
shards borrow tensor payloads and copy only when a dtype changes.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| affine mode only | yes | mxfp4/mxfp8/nvfp4 need e8m0 or e4m3 scales, a codec each; affine is mlx-lm's default and what mlx-community ships |
| bits grid {2,3,4,5,6,8} | yes | the mlx docs list these; the kernel error string also admits 1, a disagreement recorded in facts.md; the docs are the contract |
| bf16 default float dtype | yes | mlx-lm's default; f16 and f32 by option |
| names through the binding profile, extended to MoE | yes | the profile is the loader's single source of HF names; a hand table would be a second source; the oracle for the result is the vendored plain header (R2, R4) |
| dequantization through the gguf dispatch seam | yes | the tensor CPU tables are `pub(super)` and skip Q4_0/Q5_0; the gguf seam is public and covers both |
| a config writer sharing a field table with `hf_config` | yes | `hf_config` only reads; a shared table keeps the two from drifting; the oracle is not the table but the GGUF-side reader and the vendored configs (R7, R11) |
| a tokenizer writer in `proxima-tokenizer` | yes | an MLX directory without tokenizer files is unusable by mlx-lm; the crate owns the `Vocab` and the reader |
| tokenizer oracle is the vendored mlx-community file, ids oracle is the recorded llama ids | yes | proxima's own tokenizer output compared to itself is circular |
| packing float source is the vendored bf16 tensor | yes | the packing input must be shared with the reference; an Ollama Q4_0 or Q8_0 dequantization is not what mlx-community packed |
| quantized form not loaded back here | yes | native affine-with-bias blocks are a kernel codec; the fp6 spec is this round's codec work |
| reference compared as bytes with identity and degenerate controls | yes | the workspace runs no Python; a comparison with no control cannot tell truth from self-agreement |
| the CLI check is an integration test, not a shell chain | yes | a test asserts the counts and files itself and prints one N |
| no shared-expert leaf test | yes | granite's GGUF metadata carries `granitemoe.expert_count = 32` and `expert_used_count = 8` and no shared-expert key (`proxima-model-interop/tests/fixtures/llama-parity/granite_moe/gguf_kv.txt:29-30`), so no vendored model exercises one |

## acceptance criteria

Paths: `GRANITE` is `~/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`,
`E2B` is `~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`.
Tests that need a GGUF open these two paths and fail, never skip, when a blob is absent, so
each N counts runs that read a model. Every AC is one command and one count; the parenthetical
names every test case that count contains, and a slice that adds a different number updates
its AC in the same commit.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_shard_/)'` | `12 passed` (metadata on a single shard; metadata on each of two shards; the shard cap constant equals 5368709120 bytes, which is mlx-lm's `MAX_FILE_SIZE_GB = 5` shifted left by 30 bits (`mlx_lm/utils.py:67,824` on main, read this session); degenerate control: 5000000000 does not equal the constant; a model under an injected cap yields 1 shard; a model over it yields 2; single-shard name `model.safetensors`; multi-shard names `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`; each tensor in exactly one shard; index `weight_map` names the right shard for every tensor; index `metadata.total_size` equals the summed payload bytes; no index file for a single shard) |
| AC2 | R2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_names_dense/)'` | `5 passed` (E2B dense leaves map to the vendored plain header's names with 0 unmapped; granite dense leaves the same; identity control: the reference names compared to themselves pass; degenerate control: one altered name fails; a leaf with no profile entry lands in the unmapped list and no name is produced for it) |
| AC3 | R3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/granitemoe_binding_/)'` | `5 passed` (`binding_profile("granitemoe")` returns the new profile, not the default; the routed gate and up leaves bind to parts 0 and 1 of 2 of `model.layers.N.block_sparse_moe.input_linear.weight`; the routed down leaf binds to `model.layers.N.block_sparse_moe.output_linear.weight`; the router leaf binds to `model.layers.N.block_sparse_moe.router.layer.weight`; a dense profile given MoE leaves and the vendored header's manifest returns `HfMoeWeightsUnsupported` because none of its alias sources is stored; the expected names are HF's granitemoe tensor names and the slice confirms each against the vendored plain header before the test is fixed) |
| AC4 | R4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_names_moe/)'` | `4 passed` (granite MoE leaves map to the vendored plain header's names; the unmapped list is empty; identity control passes; degenerate control with one altered name fails) |
| AC5 | R5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_plain_/)'` | `10 passed` (bf16 default bytes equal hand-derived bytes; f16 option; f32 option; a Q8_0 block equals hand-derived f32 values from the ggml Q8_0 layout; a Q4_0 block equals hand-derived f32 values from the ggml Q4_0 layout; and one degenerate control per case, an expected value altered in one byte, which fails) |
| AC6 | R6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_affine_/)'` | `21 passed` (bits 2, 4, 8 and bits 3, 5, 6 each unpack through the transcribed mlx `quantized.cpp` expressions to the source q values, 6 tests; degenerate control: element order reversed does not unpack; `weight` shape `(out, in x bits / 32)` for bits 2,3,4,5,6,8; `scales` shape `(out, in / group_size)`; `biases` shape `(out, in / group_size)`; `scales` and `biases` dtype equals the export float dtype; a hand-derived group walked through `quantized.cpp:1165-1233` gives the expected scale, bias and levels (scale -0.25, bias 2.75, levels 15 - (i % 16)), plus a degenerate control; reconstruction within |scale|/2 for unclamped elements and within |scale| for the edge-clamped ones, plus a degenerate control with a q off by one; all 18 pairs of bits {2,3,4,5,6,8} and group_size {32,64,128} accepted; bits 7 refused; group_size 48 refused; `AffineParams::default()` is bits 4, group_size 64; last dim 100 at group 64 stays float with no `scales` or `biases`; control: last dim 128 is quantized) |
| AC7 | R7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_config_/)'` | `13 passed` (granite config and E2B config each carry `architectures`, `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `rms_norm_eps`, `rope_theta` and the MoE fields where routed, with the GGUF metadata's values, 2 tests; identity control; degenerate control with one altered value; the written config read by `parse_hf_config` and `architecture_from_hf_config` (`hf_config.rs:136,178`) gives a `ModelHparams` equal to the one `architecture_from_metadata` (`bind.rs:598`) gives from the GGUF, for granite and E2B, 2 tests; degenerate control: written `hidden_size` altered gives a different `ModelHparams`; an unknown family yields `UnknownFamilyConfig` (new); a quantized export carries `quantization {group_size, bits, mode: "affine"}` and the `quantization_config` mirror; a plain export carries neither; an HF-directory source's `config.json` is copied byte-equal; copy identity control; copy degenerate control with a one-byte-different source) |
| AC8 | R8 | `cargo nextest run -p proxima-tokenizer --features hf,gguf -E 'test(/hf_write/)'` | `14 passed` (writer output parsed by `vocab_from_tokenizer_json` (`proxima-tokenizer/src/hf.rs:53`) gives an equal `Vocab`; a vocab with added tokens round-trips; for granite and E2B the writer's `model.vocab`, `model.merges` (order preserved), `added_tokens`, bos/eos/unk strings and `add_bos_token` each equal the vendored plain `tokenizer.json` or `tokenizer_config.json`, 10 tests; identity control; degenerate control with one altered vocab entry) |
| AC9 | R9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_tokenizer_/)'` | `14 passed` (tokenizer.json written from GGUF metadata equals the vendored plain reference byte-structurally for granite and E2B, 2 tests; identity control; degenerate control; the written tokenizer encodes `The capital of France is` to `prompt_ids` of `granite_moe/llama_ids.json` and of `llama-parity/gemma4_e2b/llama_ids.json`, 2 tests; identity control: recorded ids compared to themselves pass; a one-token-altered copy fails; `tokenizer.json` of an HF-directory source copied byte-equal, plus identity and degenerate controls; `tokenizer_config.json` copied byte-equal, plus identity and degenerate controls) |
| AC10 | R10 | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/mlx_round_trip/)'` | `4 passed` (granite plain export read back generates `generated_ids` of `granite_moe/llama_ids.json`; E2B generates `oracle_ids` of `gemma4_e2b_llama_greedy_ids.json`; identity control: recorded ids compared to themselves pass; a one-token-altered copy fails) |
| AC11 | R11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_reference_header/)'` | `16 passed` (for granite and E2B: plain export shapes, plain export dtypes, affine export names, affine export shapes and affine export dtypes against the vendored headers, 10 tests, each asserting the measured relation exactly: no exported name is absent from the header; the E2B header holds 1411 (plain) and 1415 (affine) vision and audio tower names a text GGUF cannot export, counted by prefix; a (4, 64) export differs from the E2B affine4 header exactly in the 105 mlp weights its config sets to 8 bits, and matches it when exported at the config bits; the granite affine4 reference is synthetic; plain `config.json` and affine `config.json` key sets for each model equal the vendored keys minus the recorded unwritten keys plus the recorded top-level text keys, 4 tests; identity control; degenerate control with one renamed tensor) |
| AC12 | R12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_reference_pack/)'` | `8 passed` (for granite and E2B, packed `weight`, `scales` and `biases` bit-equal to the affine4 reference's, 6 tests; identity control: the reference compared to itself passes; degenerate control: a copy with one flipped bit fails) |
| AC13 | R13 | `cargo nextest run -p proxima-model-interop --features std,metal --test mlx_cli_export` | `10 passed` (granite and E2B: the CLI with `--quantize 4,64` writes exactly `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, 2 tests; float tensors plus quantized leaves equal N, the `ParsedGguf.tensor_count` (u64) from `proxima_gguf::parse_complete(&[u8])` returning a `ParsedGguf` or a `GgufError` (`proxima-gguf/src/pipe.rs:106`, re-exported at `proxima-gguf/src/lib.rs:67`; `tensor_count` at `pipe.rs:27`, `tensors` as a Vec of `TensorInfo` at `pipe.rs:30`), 2 tests; identity control: the header's count compared to itself passes; degenerate control: a count off by one fails; the exported tensor count equals the affine4 header's tensor count, 2 tests; reference identity control; reference degenerate control with one tensor removed) |
| AC14 | R14 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_quantized_load_refused/)'` | `1 passed` (`MlxQuantizedNotLoadable` naming the tensor) |

## out of scope

- loading mlx-lm quantized weights natively (R14 reserves it)
- mxfp4, mxfp8, nvfp4 modes; the 1-bit affine kernel path
- running mlx or mlx-lm (no Python); agreement is by bytes against vendored reference files with controls
- per-layer quantization predicates (mlx-lm's per-module quantization keys inside `quantization`)
- shared-expert leaves (no vendored model carries one)
- LoRA adapters, vision towers, the gemma4 26B model, the 16 large-checkpoint parity tests

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the granitemoe binding profile needs leaf shapes the loader does not expose for stacked experts | medium | AC3 and AC4 cannot pass | slice 3 reads the stacked-expert leaves as `expert_slab.rs` and `expert_sidecar.rs` bind them and names each as a profile entry; a leaf with no HF counterpart is reported, not invented |
| the HF granitemoe tensor names in AC3 are from the HF modeling code, not yet read from the vendored header | medium | AC3 expected names wrong | slice 2 vendors the plain header first; slice 3 fixes the literal names from it |
| mlx-community has no bf16 and 4-bit conversions at the exact source revision | medium | AC2, AC4, AC8, AC9, AC11 to AC13 have no reference | vendor the closest revisions' files, record repos and shas in the fixture README; the failing comparison then reports the revision gap |
| the affine4 conversion was quantized from a source other than the vendored bf16 tensor | medium | AC12 fails without a code defect | the README records both shas; a mismatch refutes R12 for that model and is reported, not hidden |
| `proxima_safetensors::write_complete` always writes its own format-version entry into `__metadata__` and rejects a caller that sets that key (`proxima-safetensors/src/writer.rs`, doc on `write_complete`) | medium | the shard metadata holds `format: mlx` plus an extra key mlx may not expect | slice 1 reads the key name from `sized` and AC1's metadata cases assert `format` equals `mlx`; if mlx-lm's loader rejects the extra key the writer gains an option, decided from the reference header's `__metadata__` |
| GGUF tokenizer metadata lacks fields tokenizer.json needs (pre-tokenizer, added tokens, normalizer) | high for gemma4 | AC9 fails | the failing comparison names the missing field in the test message; the field is added to the writer, not skipped |
| the mlx 3, 5, 6-bit kernel forms were read from an excerpt without line numbers (facts.md) | medium | AC6 oracle is a transcription | slice 6 re-fetches `quantized.cpp`, records the commit and line range in the test, and AC12 cross-checks one real packed tensor at the reference's bits |
| a vendored `tokenizer.json` is large | medium | repo size | slice 2 checks the repo's fixture storage policy before adding the file and follows it |
| bf16 export of a Q8_0 source loses the quantization's information | certain, by design | the plain export is a dequantized model | the CLI prints source codecs and target dtype; the quantized form is the lossless-in-bits case |
| ~~`gguf_to_safetensors` wrote GGUF `dims` unreversed, so shapes disagreed with the HF header (granite `k_proj` `[1024, 512]` vs `[512, 1024]`)~~ | fixed | every shape the transform wrote was ggml order | the transform writes reversed dims and `safetensors_to_gguf` reverses them back; tests `transform_shapes_follow_hf_row_major_*` |

## context

- `proxima-model-interop/src/transform.rs:1-12,153` (sans-IO GGUF to safetensors and back; block-quantized types fail with `UnrepresentableGgmlType`, `error.rs:21`; the only `mod` line is `mod tests;` at `:153`); `transform` is a private module (`lib.rs:90`)
- `proxima-model-interop/src/profiles/binding.rs:22-27` (embedded binding table), `:37` (`TensorAlias`), `:67` (`BindingProfile`), `:90` (`aliases_for`), `:132` (`binding_profile`); `hf_bind.rs:49` imports them and `:280` calls `aliases_for`
- `proxima-model-interop/src/hf_bind.rs:771-819` (the dense name builders), `:921` (`HfMoeWeightsUnsupported`)
- `proxima-model-interop/src/profiles/binding/` (default.toml, gemma4.toml, lfm2.toml, qwen35.toml, qwen35moe.toml; no granitemoe)
- `proxima-model-interop/src/hf_config.rs:136` (`parse_hf_config`), `:178` (`architecture_from_hf_config`); `proxima-model-interop/src/bind.rs:338` (`ModelHparams`, derives `PartialEq` at `:337`), `:598` (`architecture_from_metadata`)
- `proxima-model-interop/Cargo.toml:339,378` (`proxima-gguf` dependency; `proxima-tokenizer` dev-dependency with feature `hf`)
- `proxima-tokenizer/src/hf.rs:53` (`vocab_from_tokenizer_json`, reader only), `:158` (`bos_eos_policy_from_tokenizer_config`), `proxima-tokenizer/src/vocab.rs:101` (`Vocab`)
- `proxima-gguf/src/quant/dispatch.rs:16,24,31` (dequantize seam), `proxima-gguf/src/pipe.rs:25-30,106` (`ParsedGguf`, `parse_complete`)
- `proxima-tensor/src/cpu/epilogue.rs:1452` and `proxima-tensor/src/cpu/run_node.rs:2083` (the `pub(super)` tensor-side paths this spec does not use)
- `tools/proxima-cli/src/main.rs:42` (`Command` enum; no `Model` yet)
- `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`, `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json`, `proxima-model-interop/tests/fixtures/gemma4_e2b_llama_greedy_ids.json` (the recorded ids)
- mlx `python/mlx/nn/layers/quantized.py`, `mlx-lm/mlx_lm/utils.py`, `mlx/backend/cpu/quantized.cpp` (layout, packing order, modes, shard cap; URLs in `.long_ctx_backups/mlx_fp6/facts.md`)
