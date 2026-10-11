# mlx-cold-export -- slices

Each slice: one commit, one requirement, one validation command, the AC it repeats. Update the
checkbox and the note IN THE SAME COMMIT as the slice. Slice N discharges ACN and requirement RN.
Slices 10 and 13 load models on Metal and run behind gpu.lock; slices 1 to 9, 11, 12 and 14 are
CPU-only and may be written in parallel exports once their dependencies have landed. Each slice
lands its tests in the order its AC lists them, so it stays inside ~30 minutes by adding the
cases in separate steps of the same commit.

Dependencies: 1 none; 2 none (creates `mlx_names.rs`, vendors `plain/`); 3 on 2; 4 on 2 and 3;
5 on 2; 6 none; 7 on 2; 8 on 2; 9 on 7 and 8; 10 on 1, 2, 4, 5, 7, 9; 11 on 2, 4, 5, 6, 7
(vendors `affine4/`); 12 on 6 and 11; 13 on 10 and 11; 14 none.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | `transform/mlx.rs` (new): `MlxExport`, shard cut, names, `__metadata__ format mlx`, index file; `mod` lines beside `mod tests;` in `transform.rs` | AC1 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_shard_/)'` | `12 passed` | [x] | sans-IO assembly; `format` and `proxima_format_version` are distinct keys so `write_complete` accepts `format: mlx`; cap cut mirrors make_shards (new shard when running total + next tensor exceeds cap) |
| 2 | `transform/mlx_names.rs` (new): dense leaf to HF name through `BindingProfile::aliases_for` and the unmapped list; vendors the `plain/` reference files into `tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/` and `tests/fixtures/mlx-reference/gemma4-e2b/` with a README naming upstream repos and shas | AC2 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_names_dense/)'` | `5 passed` | [x] | dense names via aliases_for in reverse; layouts are data (STANDARD_LAYOUT, SANDWICH_NORM_LAYOUT), HF prefix a parameter; gemma4 reference is mlx-community/gemma-4-E2B-it-qat-bf16, granite is ibm-granite (no mlx-community conversion exists); gemma tokenizer.json is 32 MB, the repo has no LFS policy |
| 3 | `src/profiles/binding/granitemoe.toml` registered in the `binding.rs` table and the MoE arm of `hf_bind` | AC3 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/granitemoe_binding_/)'` | `5 passed` | [x] | names read from the vendored header (input_linear.weight [32,1024,1024] fused gate+up, output_linear.weight [32,1024,512], router.layer.weight [32,1024]), all three confirmed; 24 literal per-layer aliases x4 as lfm2.toml does; MoE arm = routed_expert_count, moe_stack_source (alias whose source the manifest stores, else HfMoeWeightsUnsupported), bind_safetensors_expert_stack (decode, cut part, transpose to [experts,in,out]); every stacked leaf has one HF counterpart (gate and up share one) |
| 4 | MoE leaf names in `mlx_names.rs` vs the vendored plain header | AC4 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_names_moe/)'` | `4 passed` | [ ] |  |
| 5 | `export_plain`: bf16/f16/f32 casts and Q8_0/Q4_0 dequantization through `proxima_gguf::quant::dispatch::dequantize` | AC5 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_plain_/)'` | `10 passed` | [ ] | hand-derived block oracles |
| 6 | `pack_affine`, `export_affine`, `AffineParams` (bits, group size, default, float fallback) | AC6 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_affine_/)'` | `21 passed` | [x] | affine packing lives in `transform/mlx_affine.rs` (not `mlx.rs`); oracle transcribed from mlx 06eb7483 `quantized.cpp` 65-93, 99-160, 1165-1233; `tests/support/trial.rs` moved to a `#[path]` mod in its one user so the `capability_matrix` target builds under `-D warnings` |
| 7 | `transform/mlx_config.rs` (new): HF config writer, `quantization` keys, `UnknownFamilyConfig`, config copy | AC7 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_config_/)'` | `13 passed` | [x] | e2b input is `sliding_pattern_hparams` (architecture_from_metadata fails on the gemma4 `feed_forward_length` array); keys drawn from `HF_CONFIG_KEYS` in hf_config.rs; gemma4 also nests `text_config`; quantization is `Option<(bits, group_size)>` until slice 6 lands `AffineParams` |
| 8 | `proxima-tokenizer/src/hf_write.rs` (new): `tokenizer.json` and `tokenizer_config.json` writer from `Vocab`, feature `hf`; agreement with the vendored plain files | AC8 | `cargo nextest run -p proxima-tokenizer --features hf,gguf -E 'test(/hf_write/)'` | `14 passed` | [x] | Vocab::merge_pairs added; reader now accepts merges as ["l","r"] pairs (vendored files use them; string form failed); writer emits null normalizer/pre_tokenizer/post_processor/decoder and model = type+vocab+merges, config = bos/eos/unk strings + add_bos/eos_token only; granite added_tokens ids 49152-49154 lie beyond model.vocab so a Vocab cannot hold them; reader ignores added_tokens so tests restore token types; R9 encode-equality is where the dropped pipeline blocks show |
| 9 | interop tokenizer export from GGUF metadata and copy for HF sources; encode of the fixture prompt; enable feature `hf` on the interop's tokenizer dependency with `cargo add` | AC9 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_tokenizer_/)'` | `14 passed` | [ ] | external ids: `granite_moe/llama_ids.json` and `gemma4_e2b/llama_ids.json` |
| 10 | round trip on both models | AC10 | `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/mlx_round_trip/)'` | `4 passed` | [ ] | GPU; behind gpu.lock |
| 11 | vendor the `affine4/` reference files; header, name and config-key agreement for the plain and affine exports | AC11 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_reference_header/)'` | `16 passed` | [ ] | README records the 4-bit repo and sha |
| 12 | packing agreement against the affine4 reference | AC12 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_reference_pack/)'` | `8 passed` | [ ] | input is the vendored plain bf16 tensor read as exact f32 |
| 13 | CLI `model export-mlx` and `tests/mlx_cli_export.rs` | AC13 | `cargo nextest run -p proxima-model-interop --features std,metal --test mlx_cli_export` | `10 passed` | [ ] | GPU; behind gpu.lock |
| 14 | `InteropError::MlxQuantizedNotLoadable` refusal on load | AC14 | `cargo nextest run -p proxima-model-interop --features std -E 'test(/mlx_quantized_load_refused/)'` | `1 passed` | [x] | refused in hf_bind::mlx_refusal, called first by both safetensors binders; tests/support/trial.rs moved to a per-target mod (dead-code deny) |

## resume

Last landed slice: 1, 2, 3, 6, 7, 8 and 14
Next action: slice 4, 5, then 9 to 13
Open question, if any: none for slice 3 (each stacked-expert leaf has one HF counterpart; gate and up are parts 0 and 1 of 2 of one fused tensor)

## struck

- R6 and AC6 as first written: min-anchored arithmetic (`s = (max - min) / (2^bits - 1)`, `q = round((w - min) / s)`, `w = s q + min`). That was the spec's error, found by reading the incumbent: mlx 06eb7483 `quantized.cpp:1165-1233` anchors the bias on the edge of larger magnitude (scale negative when that edge is the maximum). R6, AC6 and `pack_affine` now follow the incumbent.
