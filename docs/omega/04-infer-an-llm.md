# Omega 04: inference with an LLM

**Prerequisites:** [page 02](./02-programs-as-data.md) (binds, outputs, `plan_parts`) and [page 03](./03-train.md) (a model is a program over named buffers; `State` is named buffers).

**You will:** load a real GGUF checkpoint, see the three facts the checkpoint does not carry and where they live as a TOML file, generate tokens through the serving path, read the `ServingSettings` surface that mirrors llama.cpp's flags, and measure milliseconds per token on Llama-3.2-1B-Instruct.

**New concepts (in order):** a checkpoint is bytes plus a binding · `CheckpointMapping` · the architecture file: `src/arch/llama.toml` · `ServingSettings`, the flag surface as config · `ServingConfig`, the resolved per-call shape · the four generation entry points · `generate_with_serving_config` and what it returns · decode timing · what omega does not do yet.

## 1. A checkpoint, an architecture file, and a serving config

Omega does not know what a transformer is. Serving does: `proxima-model-interop` reads a checkpoint, binds each stored tensor to a named input of the graph the architecture implies, and hands omega a program. Three artifacts make that work.

**The checkpoint.** A GGUF file is parsed once (`proxima_gguf::parse_complete`) and stays mmap'd, so loading a 2.3 GB model does not copy 2.3 GB. `CheckpointMapping` (`proxima-model-interop/src/source.rs:29`) owns the file and the mapping, and `load_serving` (`:91`) turns it into a `LoadedModel`.

**The architecture.** A checkpoint does not say everything: whether positions rotate pairwise or interleaved, which activation the FFN uses, whether values are normalised. Those live in a small TOML per family, keyed on `general.architecture`. For llama it is `proxima-model-interop/src/arch/llama.toml:5-11`, and it is three lines:

```toml
rope_layout = "adjacent"
ffn_activation = "Silu"
value_norm = false
```

Each line carries the llama.cpp source it came from in a comment (`llama.toml:4-5`, `:7-8`, `:10-11`), so if the two disagree you can diff against the line rather than argue from memory. `ArchConfig::load` (`proxima-model-interop/src/arch_config.rs:278`) and `load_from` (`:294`) read the file; a family with no file is refused rather than guessed.

**The serving config.** `ServingSettings` (`proxima-model-interop/src/serving_settings.rs:157`) is llama.cpp's flags as a config, in four groups: how much runs and where (`-c`, `-np`, `-ngl`, `--fit`), how the KV cache is stored (`--cache-type-k`/`-v`), how execution is shaped (`-fa`, `-b`, `-ub`), and how a token is sampled. It derives `Builder`, `Settings` and `Validate`, its env prefix is `PROXIMA_SERVING` (`:154`), and `as_serving_config` lowers it to the `Copy` shape the decode loop reads.

The GPU hook is in that lowering: `ServingConfig::omega_runtime` (`serving.rs:804`) is an `Option<&omega::config::RuntimeConfig>`. The engine and driver a model serves on are the same `RuntimeConfig` from page 00.

## 2. Load, then generate

The four generation entry points are all in `proxima-model-interop/src/generate/decode.rs`:

| entry point | line | shape |
|---|---|---|
| `generate_with_serving_config` | `:2677` | prompt text in, `(ids, text, stopped_by_eos)` out |
| `generate_from_prefix` | `:2926` | continue from a token prefix |
| `generate_from_ids` | `:2982` | exact ids in, no tokenizer round trip |
| `generate_streaming` | `:3561` | yields tokens as they are produced |

`generate_from_ids` is the one a parity harness uses: it removes the tokenizer from the comparison and leaves only the graph.

`proxima-model-interop/examples/decode_loop_wall_clock.rs` is the smallest end-to-end runnable program: one `load_with_serving`, one generation, and a printed timing line. Its inputs are the checkpoint path (`PROXIMA_OPENCHAT_GGUF`), the prompt (`PROXIMA_PROMPT`) and the token count (`PROXIMA_MAX_TOKENS`); nothing else is compiled in.

```sh
PROXIMA_OPENCHAT_GGUF=$HOME/models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf \
PROXIMA_MAX_TOKENS=24 \
PROXIMA_PROMPT="The capital of France is" \
cargo run --release -p proxima-model-interop --features std,metal --example decode_loop_wall_clock
```

```text
prompt="The capital of France is" max_tokens=24
decode_summary tokens_generated=24 stopped_by_eos=false total_wall_clock_ms=501.920 mean_ms_per_token=20.913 generated_text=" Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in Paris. The famous" ids=[12366, 13, 578, 469, 3168, 301, 22703, 374, 7559, 304, 12366, 13, 578, 9928, 49606, 16730, 374, 1101, 7559, 304, 12366, 13, 578, 11495]
```

That is a warm run. The first request after a rebuild printed `mean_ms_per_token=116.152` on the same command and the same checkpoint, because Metal compiles its pipelines the first time a shape is planned and caches them from then on (`pipeline_cache`, page 01 section 1). Three warm runs of the same command, in one quiet window, printed 20.913, 20.920 and 21.416 ms per token: a mean of 21.08 and a sample coefficient of variation of 1.4%. A loaded machine changes those numbers, and page 07 records by how much. Page 07 also puts this beside llama.cpp on the same weights.

The generated text is the checkpoint answering: prompt in, ids out, decoded by the same tokenizer. The `ids` are the contract: page 07's comparison and the parity harnesses in `proxima-model-interop/tests` compare ids.

## 3. What the config decides

The settings that change a decode, and the field each one is:

- How much context. `context_length` (`serving_settings.rs:167`) with `context_extrapolate` (`:171`); the memory fit then clamps whichever length results, which is why a 131072-token request either loads or fails at validation, with the field named, instead of dying halfway through a decode.
- How the KV cache is stored. `kv_cache_key_quant` and `kv_cache_value_quant` (`:183`, `:188`), defaulting to `f16` when the `gpu` feature is on and `f32` without it. The pair is admitted by `kv_cache_format::admit_pair`, so a mismatched pair is a config error.
- Whether attention is fused. `flash_attention` (`:192`), default `false`: the naive multiply-then-reduce graph, which is the shape every parity test can read.
- How much is offloaded. `gpu_layers` (`:205`), `gpu_memory_fit` (`:209`) and `gpu_memory_limit_bytes` (`:211`).
- How a prompt is chunked. `batch_size` (`:196`) and `ubatch_size` (`:200`); an architecture that is not single-position evaluates a long prompt in `ceil(prompt_tokens / ubatch_size)` evaluations.
- How a token is picked. `temperature` (`:227`), `top_k` (`:231`), `top_p` (`:235`), `min_p` (`:239`), `repeat_last_n` (`:243`), `repeat_penalty` (`:247`), the two OpenAI-style penalties (`:251`, `:255`) and `seed` (`:259`). These are read into `SampleConfig` when the loop is built (`proxima-model-interop/src/generate/decode.rs:3981-3985`), so greedy is `temperature = 0.0`.

## 4. What omega does not do yet

- No server. `proximad` mounts `/pipelines`, `/events` and the tensor-plan route (`tools/proxima-cli/src/tensor_run.rs:94`); there is no HTTP chat or completion route in this workspace. Serving over HTTP is llama.cpp's job today.
- KV cache formats apply to the whole model. `kv_cache_key_quant`/`kv_cache_value_quant` (`serving_settings.rs:183-188`) are set per model, and there is no per-layer override.
- No conversion command in the CLI. The library can write GGUF (`proxima_gguf::write_complete`, `proxima-gguf/src/writer.rs:65`) and convert safetensors into it (`safetensors_to_gguf`, `proxima-model-interop/src/transform.rs:109`). The CLI exposes `model export-mlx` only (`tools/proxima-cli/src/main.rs:157`), so producing a GGUF from an HF checkpoint means writing Rust against the library.

Next: [05: speech, whisper](./05-speech-whisper.md) teaches the pieces that exist for speech and names the ones that do not.
