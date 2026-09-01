#!/usr/bin/env bash
# BGE-small incumbent-arm harness runner -- see README.md.
#
# Creates (or reuses) a pinned venv, installs pinned deps, runs bench.py
# against BGE_MODEL_PATH, prints the final arm | ms/sentence | CoV% |
# n_runs | threads table.
#
# Usage:
#   BGE_MODEL_PATH=/path/to/model.onnx ./run.sh
#
# Env vars:
#   BGE_MODEL_PATH       required -- path to a BGE-small-en-v1.5 model.onnx
#                         (never written into a tracked file; if you don't
#                         have one, `python export_model.py` produces one
#                         from the cached HF safetensors checkout)
#   ONNX_REF_VENV_DIR     venv location (default: $HERE/.venv)
#   ONNX_REF_PYTHON       interpreter to build the venv with (default:
#                         python3.12 -- torch/onnxruntime wheel availability
#                         lags newer CPython releases; 3.14 has none as of
#                         this writing)
#   ONNX_REF_RUNS          runs per arm (default: 5, matches bge_eval.rs's
#                         own BGE_EVAL_RUNS default)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ONNX_REF_VENV_DIR:-$HERE/.venv}"
PYTHON_BIN="${ONNX_REF_PYTHON:-python3.12}"

if [[ -z "${BGE_MODEL_PATH:-}" ]]; then
    echo "skipping: set BGE_MODEL_PATH to a local BGE-small-en-v1.5 model.onnx checkout"
    echo "  (no model.onnx? run: $PYTHON_BIN $HERE/export_model.py -- it exports one"
    echo "   from the cached HF safetensors checkout into ./model/model.onnx)"
    exit 0
fi
if [[ ! -f "$BGE_MODEL_PATH" ]]; then
    echo "skipping: BGE_MODEL_PATH=$BGE_MODEL_PATH does not exist"
    exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "creating venv at $VENV_DIR ($PYTHON_BIN)…"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi

PIP_INSTALL=(torch==2.5.1 transformers==4.46.3 "onnxruntime==1.20.1" "numpy<2" onnx==1.17.0)
echo "installing pinned deps: ${PIP_INSTALL[*]}"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_DIR/bin/python" "${PIP_INSTALL[@]}"
else
    "$VENV_DIR/bin/pip" install --quiet "${PIP_INSTALL[@]}"
fi

echo
echo "running bench.py (BGE_MODEL_PATH=$BGE_MODEL_PATH, runs=${ONNX_REF_RUNS:-5})…"
echo
BGE_MODEL_PATH="$BGE_MODEL_PATH" ONNX_REF_RUNS="${ONNX_REF_RUNS:-5}" "$VENV_DIR/bin/python" "$HERE/bench.py"
