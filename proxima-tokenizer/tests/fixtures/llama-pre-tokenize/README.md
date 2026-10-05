# llama.cpp pre-tokenizer token-id fixtures

Oracle: llama.cpp commit f1ea206218210afb913ae2f5d2c51faed35915da (`llama-tokenize`, built out of tree,
CPU-only, Release: `cmake -S . -B <build> -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release`,
then `cmake --build <build> --target llama-tokenize`). `tools/tokenize/tokenize.cpp:146` loads with
`vocab_only = true`, so no weights are read.

Command per fixture, no BOS (the Rust tests call `encode`, which adds none):

    llama-tokenize -m <gguf> --ids --log-disable --no-escape --no-bos -f texts/<name>.txt > <family>/<name>.ids

| family dir | gguf | `tokenizer.ggml.pre` | llama.cpp pre type |
|---|---|---|---|
| llama3 | `llama.cpp/models/ggml-vocab-llama-bpe.gguf` (vocab only) | `llama-bpe` | LLAMA3 |
| qwen2 | Ollama blob sha256 c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515 (qwen2.5:0.5b) | `qwen2` | QWEN2 |
| qwen3 | Ollama blob sha256 a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f (qwen3:8b) | `qwen2` | QWEN2 |
| qwen35 | Ollama blob sha256 afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5 | `qwen35` | QWEN35 |
| qwen35moe | Ollama blob sha256 f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d | `qwen35` | QWEN35 |
| granite_moe | Ollama blob sha256 cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 (granite3.1-moe:1b) | `refact` | REFACT |
| deepseek_coder_33b_no_pre | `~/.lmstudio/models/TheBloke/deepseek-coder-33B-instruct-GGUF/deepseek-coder-33b-instruct.Q4_K_S.gguf` (vocab only; the file carries no `tokenizer.ggml.pre`, llama.cpp logs "missing pre-tokenizer type, using: 'default'") | none | DEFAULT (four-pass regex) |

`texts/*.txt` are the exact input bytes, shared by every family: prices, dates and times, phone-like numbers,
version strings, 1..8 digit runs (2 vs 3 digit boundary), large and decimal numbers, code and CSV with numbers,
contractions beside digits, non-ASCII digits, combining marks beside digits, Devanagari/Thai/Arabic with marks
beside digits, and roman numerals, circled letters and circled digits. Used only to generate these vendored
files; llama.cpp is not a runtime dependency.

`src/unicode_tables.rs` is not hand-written: `cargo run -p proxima-tokenizer --example gen_unicode_tables -- <llama.cpp>/src/unicode-data.cpp f1ea206218210afb913ae2f5d2c51faed35915da proxima-tokenizer/src/unicode_tables.rs`
regenerates it from llama.cpp's `unicode-data.cpp` at the same commit.

`starcoder/` and `refact/` use the vocab-only GGUFs from llama.cpp's `models/` (read via `LLAMA_CPP_MODELS_DIR`, not vendored: `ggml-vocab-starcoder.gguf`, `ggml-vocab-refact.gguf`,
`tokenizer.ggml.pre` = `starcoder` / `refact`, llama.cpp pre type STARCODER / REFACT), same command and the same 13 texts.

`texts_non_ascii_digits/` (Arabic-Indic and Persian digits, Devanagari digits, fullwidth digits, superscripts, fractions, circled digits, ASCII beside non-ASCII digits)
are run the same way against the starcoder and refact vocabs only: those vocabs have no multi-digit-only and no space+digit tokens, so ASCII digit grouping
is invisible in their ids and only non-ASCII digit runs distinguish `\p{N}` isolation from the other gpt2-family splits.
