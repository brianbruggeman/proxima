#!/usr/bin/env bash
# bench-vs-ggml: proxima-tensor's packed-int8 kernels vs ggml (Accelerate-
# backed) on ggml's own home turf — batch-1 decode GEMM at real quantized
# weight shapes, plus bare f32 GEMM and a gather/scale/reduce workload.
#
# Registered under `ggml-bench` since 2026-08 and never invoked anywhere in
# this repo or its CI: no gate script, no workflow job ever ran it. When it
# was finally run by hand this session it produced the session's most
# important measurement (our int8 kernel 4.2x faster than ggml on real decode
# shapes) — evidence that existed the entire time and never ran.
#
# This script is a LOCAL-ONLY runner, not a CI job, because the six benches
# below have two requirements CI cannot satisfy without adding brand-new,
# out-of-scope infrastructure:
#
#   1. `proxima-tensor/build.rs` links a STATICALLY-built ggml checkout
#      (`GGML_BUILD_DIR`) via `-lstatic=ggml-cpu/-base/-ggml` plus the macOS
#      `Accelerate`/`Foundation` frameworks. No ggml source is vendored in
#      this repo; building one means cloning a third-party C++ project,
#      running cmake, and pinning its commit — a new CI capability, not a
#      bench-wiring change.
#   2. Five of the six benches (every one except `bench_vs_ggml` itself) read
#      REAL packed tensor bytes out of a live 4.1 GiB GGUF checkpoint
#      (`openchat-3.5-1210.Q4_K_S.gguf`) rather than synthetic data — no
#      checkpoint is vendored or fetchable by a CI job either.
#
# Both requirements are operator-machine assets. The path they resolve
# through here (an explicit precondition check, loud failure, one
# documented command) is the correct fix per guiding-principle 15: a bench
# that silently no-ops when its precondition is missing is the same defect
# as one nobody ever runs. See CI: five of these six benches print
# "nothing to bench" and exit 0 when the GGUF file is absent — this script
# treats that as a hard failure, not a pass, because a zero-arm run reports
# the same signal as a passing one.
#
# usage:
#   GGML_BUILD_DIR=/path/to/built/ggml \
#   PROXIMA_BENCH_GGUF_PATH=/path/to/checkpoint.gguf \
#     scripts/bench-vs-ggml.sh
#
# building a static ggml checkout (the exact command this repo's own
# discipline log used, proxima-tensor/docs/discipline.md:4114):
#   cmake -S ggml -B ggml/build -DBUILD_SHARED_LIBS=OFF \
#     -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
#   cmake --build ggml/build
#   export GGML_BUILD_DIR=/path/to/that/ggml
#
# PROXIMA_BENCH_GGUF_PATH defaults to the hardcoded path each bench file
# used to carry verbatim — overriding it is what makes this runnable on any
# machine that has its own checkpoint, not only the one that hardcoded it.
#
# outputs: benches/RESULTS_bench-vs-ggml_<platform>.md
# env vars:
#   GGML_BUILD_DIR         — required, no default (see build command above)
#   PROXIMA_BENCH_GGUF_PATH — optional, defaults to the in-source path

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

if [[ -z "${GGML_BUILD_DIR:-}" ]]; then
  printf 'RED: GGML_BUILD_DIR is unset.\n' >&2
  printf 'this bench links a statically-built ggml checkout; see this script'"'"'s\n' >&2
  printf 'header for the exact cmake command and re-run with GGML_BUILD_DIR set.\n' >&2
  exit 1
fi

if [[ ! -f "${GGML_BUILD_DIR}/build/src/libggml.a" ]]; then
  printf 'RED: %s/build/src/libggml.a not found.\n' "$GGML_BUILD_DIR" >&2
  printf 'GGML_BUILD_DIR must point at a ggml checkout whose build/ directory\n' >&2
  printf 'already holds the static libs (BUILD_SHARED_LIBS=OFF cmake build).\n' >&2
  exit 1
fi

GGUF_PATH="${PROXIMA_BENCH_GGUF_PATH:-/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf}"
export PROXIMA_BENCH_GGUF_PATH="$GGUF_PATH"

PLATFORM="$(detect_platform)"
RESULTS="benches/RESULTS_bench-vs-ggml_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-vs-ggml-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-vs-ggml: platform=%s\n' "$PLATFORM"
printf 'GGML_BUILD_DIR=%s\n' "$GGML_BUILD_DIR"
printf 'PROXIMA_BENCH_GGUF_PATH=%s\n' "$GGUF_PATH"
printf 'output: %s\n' "$RESULTS"

# run_bench <bench_name> <features> <needs_gguf>
# asserts a nonzero criterion arm count — a bench that runs zero arms exits 0
# and reads exactly like one that ran and passed; that silent-green is the
# defect this whole gate exists to close.
run_bench() {
  local bench_name="$1"
  local features="$2"
  local needs_gguf="$3"

  if [[ "$needs_gguf" == "yes" && ! -f "$GGUF_PATH" ]]; then
    printf 'RED: %s needs a real GGUF checkpoint at %s (not found).\n' "$bench_name" "$GGUF_PATH" >&2
    printf 'set PROXIMA_BENCH_GGUF_PATH to a checkpoint on this machine.\n' >&2
    exit 1
  fi

  printf -- '--- %s (features=%s) ---\n' "$bench_name" "$features"
  rm -rf "${CRITERION_DIR:?}/${bench_name}"
  cargo bench \
    -p proxima-tensor \
    --features "$features" \
    --bench "$bench_name" \
    2>&1 | tee "${LOGS_DIR}/${bench_name}.log"

  # criterion's group names come from the bench source, not the binary name,
  # so scope the arm count to estimates.json files written by THIS run.
  local found
  found="$(find "$CRITERION_DIR" -name estimates.json -newer "${LOGS_DIR}/${bench_name}.log.start" 2>/dev/null | wc -l | tr -d ' ')"

  if [[ "${found:-0}" -lt 1 ]]; then
    printf 'RED: %s produced zero criterion arms. exit 0 with N=0 is not a pass.\n' "$bench_name" >&2
    exit 1
  fi
  printf '%s: %s arms recorded.\n' "$bench_name" "$found"
}

# stamp a start marker per bench so `find -newer` scopes the arm count to
# this run and not to a stale criterion tree from a previous invocation.
touch_start_marker() {
  local bench_name="$1"
  touch "${LOGS_DIR}/${bench_name}.log.start"
}

cat > "$RESULTS" << HEADER
# bench-vs-ggml results — ${PLATFORM}

proxima-tensor's packed-int8 kernels vs ggml (Accelerate-backed) on real
decode shapes and real quantized weight bytes.

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
GGML_BUILD_DIR: \`${GGML_BUILD_DIR}\`
PROXIMA_BENCH_GGUF_PATH: \`${GGUF_PATH}\`

---

HEADER

for entry in \
  "bench_vs_ggml|ggml-bench|no" \
  "bench_q4k_matmul|ggml-bench|yes" \
  "bench_q5k_matmul|ggml-bench,q5k-int8-dot|yes" \
  "bench_q6k_matmul|ggml-bench,q6k-int8-dot|yes" \
  "bench_q4k_cold_cache|ggml-bench,q4k-int8-dot|yes" \
  "bench_q4k_superblock_phases|ggml-bench,q4k-int8-dot|yes"
do
  IFS='|' read -r bench_name features needs_gguf <<< "$entry"
  touch_start_marker "$bench_name"
  run_bench "$bench_name" "$features" "$needs_gguf"
  printf '## %s\n\nsee %s/%s.log\n\n' "$bench_name" "$LOGS_DIR" "$bench_name" >> "$RESULTS"
done

printf '\n---\nRun completed: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RESULTS"
printf 'bench-vs-ggml complete. results: %s\n' "$RESULTS"
