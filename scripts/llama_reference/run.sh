#!/usr/bin/env bash
# Committed incumbent-arm invocation for the GPU-decode lane (ROW 116/193's
# own cited numbers, previously unreproducible -- see README.md in this
# directory). Runs llama.cpp's OWN llama-bench binary against the SAME
# openchat-3.5-1210.Q4_K_S.gguf checkpoint the proxima decode loop loads,
# never our FFI (our FFI links a CPU-only ggml sibling tree -- see README).
set -euo pipefail

LLAMA_CPP_CHECKOUT="${LLAMA_CPP_CHECKOUT:-$HOME/repos/others/llama.cpp}"
LLAMA_BENCH="${LLAMA_BENCH:-$LLAMA_CPP_CHECKOUT/build/bin/llama-bench}"
MODEL_PATH="${MODEL_PATH:-$HOME/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf}"
N_GEN="${LLAMA_REF_N_GEN:-32}"
REPETITIONS="${LLAMA_REF_REPETITIONS:-5}"
THREADS="${LLAMA_REF_THREADS:-8}"

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "llama-bench not found or not executable at $LLAMA_BENCH" >&2
    echo "build it first: cd $LLAMA_CPP_CHECKOUT && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "gguf checkpoint not found at $MODEL_PATH" >&2
    exit 1
fi

checkout_sha=$(git -C "$LLAMA_CPP_CHECKOUT" rev-parse HEAD 2>/dev/null || echo "unknown")
model_bytes=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH")

echo "llama.cpp checkout: $LLAMA_CPP_CHECKOUT @ $checkout_sha" >&2
echo "llama-bench binary: $LLAMA_BENCH" >&2
echo "model: $MODEL_PATH ($model_bytes bytes)" >&2
echo "invocation: llama-bench -m <model> -n $N_GEN -r $REPETITIONS -t $THREADS (default -ngl 99 -b 2048 -ub 512)" >&2

"$LLAMA_BENCH" -m "$MODEL_PATH" -n "$N_GEN" -r "$REPETITIONS" -t "$THREADS"
