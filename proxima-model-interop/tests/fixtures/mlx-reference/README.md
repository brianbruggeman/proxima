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
