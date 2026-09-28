#!/usr/bin/env bash
# builds the llama-ngram fixture generator against llama.cpp's own
# common/llama static libraries (CPU only). usage:
#
#   LLAMA_CPP_SRC=/path/to/llama.cpp \
#   LLAMA_CPP_BUILD=/path/to/out-of-tree/build \
#   ./build.sh
#
# LLAMA_CPP_SRC must be checked out at commit f1ea20621 (recorded in every
# fixture header). LLAMA_CPP_BUILD is an out-of-tree cmake build directory;
# it is configured and built here if libllama-common.a is not already there.

set -euo pipefail

SRC="${LLAMA_CPP_SRC:?set LLAMA_CPP_SRC to a llama.cpp checkout}"
BUILD="${LLAMA_CPP_BUILD:?set LLAMA_CPP_BUILD to an out-of-tree cmake build dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$BUILD/common/libllama-common.a" ]; then
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
        -DGGML_METAL=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF
    cmake --build "$BUILD" --target llama-common -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
fi

clang++ -std=c++17 -O2 \
    -I"$SRC/common" -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/vendor" \
    "$HERE/main.cpp" -o "$HERE/fixturegen" \
    "$BUILD/common/libllama-common.a" \
    "$BUILD/common/libllama-common-base.a" \
    "$BUILD/src/libllama.a" \
    "$BUILD/ggml/src/libggml.a" \
    "$BUILD/ggml/src/libggml-cpu.a" \
    "$BUILD/ggml/src/libggml-base.a" \
    -lpthread -lcurl -framework Accelerate -framework Foundation

echo "built $HERE/fixturegen"
