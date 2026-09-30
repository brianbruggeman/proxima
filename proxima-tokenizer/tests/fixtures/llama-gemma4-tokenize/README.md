# llama.cpp gemma4 token-id fixtures

Oracle: llama.cpp commit f1ea20621 (`llama-tokenize`, built CPU-only, Release,
`-DGGML_METAL=OFF -DLLAMA_CURL=OFF`).

Model: gemma4 GGUF blob `sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`
(Ollama blob store).

Command per fixture, default BOS prepended:

    llama-tokenize -m <blob> --ids --log-disable --no-escape -f <name>.txt > <name>.ids

Each `<name>.txt` is the exact input bytes (no trailing newline added unless the
input itself has one); `<name>.ids` is the oracle output. The war_and_peace inputs
are the first 2000 / 30000 bytes of `proxima-model-interop/examples/data/war_and_peace.txt`.
Used only to generate these vendored files; llama.cpp is not a runtime dependency.
