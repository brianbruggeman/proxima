# mlx-reference

External reference files for the MLX export. Fetched with `curl` from huggingface.co and never run.
`plain/` is the bf16 conversion of each model. Safetensors headers are the first `8 + N` bytes of a
shard (`N` is the little-endian u64 in bytes 0..8), saved as the JSON object only; no shard is vendored.
`data_offsets` in a header are relative to the first byte after the header.

## gemma4-e2b/plain

- upstream: `mlx-community/gemma-4-E2B-it-qat-bf16`
- revision: `7794c5ea0f9cf1b92f074f314ac4e461f2fa87b2` (last modified 2026-06-05)
- the local GGUF is Ollama `library/gemma4:e2b-it-qat`; this is the bf16 conversion of the same QAT release
- files: `config.json`, `tokenizer.json` (32169626 bytes), `tokenizer_config.json`,
  `header-0000N-of-00003.json` (the three shards' headers, header lengths 70222, 130214 and 45516 bytes)
- tensor: `language_model.model.layers.0.self_attn.k_proj.weight`, BF16, shape `[256, 1536]`, 786432 bytes,
  file `language_model.model.layers.0.self_attn.k_proj.weight.bf16.bin`, taken from
  `model-00001-of-00003.safetensors` at absolute bytes 4202494104..=4203280535
  (`data_offsets` [4202423874, 4203210306] plus 8 plus the 70222-byte header)

## granite-3.1-1b-a400m-moe/plain

- upstream: `ibm-granite/granite-3.1-1b-a400m-instruct`
- revision: `0da7a48b0276d500ce5922fd2b33944091fc6c09`
- gap: mlx-community publishes no conversion of this model (searches `granite-3.1`, `a400m`, `granite-moe`,
  `granitemoe`, `granite-3.0` return none for 1b-a400m). These are the original IBM bf16 safetensors, one shard,
  header `{"__metadata__":{"format":"pt"}}` instead of `mlx`. mlx-lm's bf16 conversion keeps the dense names
  and renames only the routed-expert stacks, so slice 4 (`mlx_names_moe`) compares expert names against
  a header with a revision gap that has to be recorded when it lands.
- files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `header.json` (header length 26384 bytes)
- tensor: `model.layers.0.self_attn.k_proj.weight`, BF16, shape `[512, 1024]`, 1048576 bytes,
  file `model.layers.0.self_attn.k_proj.weight.bf16.bin`, taken from `model.safetensors` at absolute bytes
  201428760..=202477335 (`data_offsets` [201402368, 202450944] plus 8 plus the 26384-byte header)

The gemma4-e2b `plain/tokenizer.json` is stored minified (`jq -c`, same JSON value) to keep it under the repository's largest tracked file; the upstream file is pretty-printed at 32 MB.

## gemma4-e2b/affine4

- upstream: `mlx-community/gemma-4-E2B-it-qat-4bit`
- revision: `42f62737af7a9fd8c1d55d79666c1a217be4e2e2` (last modified 2026-06-05); one shard, `model.safetensors`
- files: `config.json` (31198 bytes; carries `quantization` and `quantization_config`, `mode` affine, `group_size` 64, `bits` 4, plus 105 per-module overrides at `bits` 8 for every layer's `mlp.gate_proj`, `mlp.up_proj` and `mlp.down_proj`), `header.json` (header length 320311 bytes, 2511 tensors, `__metadata__` `{"format":"mlx"}`)
- tensor: `language_model.model.layers.0.self_attn.k_proj`, the same module `plain/` vendors, taken from `model.safetensors` at absolute bytes (`data_offsets` plus 8 plus the 320311-byte header):
  - `weight`, U32, shape `[256, 192]`, 196608 bytes, file `...k_proj.weight.u32.bin`, bytes 2600571573..=2600768180
  - `scales`, BF16, shape `[256, 24]`, 12288 bytes, file `...k_proj.scales.bf16.bin`, bytes 2553717787..=2553730074
  - `biases`, BF16, shape `[256, 24]`, 12288 bytes, file `...k_proj.biases.bf16.bin`, bytes 2605032131..=2605044418
- the QAT release is mixed precision: this conversion is not uniform (4, 64). The 105 mlp weights are `[rows, in * 8 / 32]` words, and the header tests derive that set from the vendored config instead of assuming it

## granite-3.1-1b-a400m-moe/affine4

mlx-community has no 4-bit conversion of this model, and none for the nearest architecture: the granite 4-bit repos
found by search (`granite-3.2-8b-instruct-preview-4bit`, `granite-3.3-*-4bit`, `granite-4.0-*-4bit`,
`granite-4.0-tiny-preview-4bit`, `granite-4.0-h-*-4bit`) are dense, hybrid-mamba or `granitemoehybrid` models, not `granitemoe`.
Every granite `affine4` row is therefore self-referential and says nothing about mlx-lm's output. What the files are:

- `header.json` is written by `cargo run -p proxima-model-interop --features std --example mlx_granite_affine_header -- <this directory>` from the vendored
  `plain/header.json`: `sanitize` of the installed mlx-lm `granitemoe.py` splits `input_linear` into `switch_mlp.gate_proj` and `switch_mlp.up_proj` along
  the row axis and renames `output_linear` to `switch_mlp.down_proj`; then each `.weight` with at least two dimensions and a last dimension divisible by 64
  becomes `weight` (U32, last dim `x bits / 32`), `scales` and `biases` (BF16, last dim `/ 64`), other tensors are copied; names sorted; offsets assigned
  contiguously; `__metadata__` `{"format":"mlx"}`. The rule is transcribed from mlx-lm and mlx `quantize`, not from this crate's output
- `config.json` is `plain/config.json` plus `quantization` and `quantization_config` `{group_size 64, bits 4, mode affine}` and, in both blocks, one `{group_size 64, bits 8}` entry per layer for `model.layers.N.block_sparse_moe.router.layer`
- `model.layers.0.self_attn.k_proj.{weight.u32,scales.bf16,biases.bf16}.bin` are the output of this crate's `export_affine` at (4, 64) over
  `plain/model.layers.0.self_attn.k_proj.weight.bf16.bin` read as exact f32; shapes `[512, 128]`, `[512, 16]`, `[512, 16]`, 262144, 16384 and 16384 bytes.
  They are the crate agreeing with itself, so a slice 12 comparison on these rows is an identity check and only the gemma rows are external
- mlx-lm's `granitemoe.py` quantizes `router.layer` at 8 bits (`quant_predicate`) and the rest at the export's width; the installed copy under
  `~/.venvs/mlx` was read 2026-10-12 and carries no `o_proj` or `down_proj` rule. The export and this header now share that layout, and `config.json` lists the
  routers per module as the loader reads them
