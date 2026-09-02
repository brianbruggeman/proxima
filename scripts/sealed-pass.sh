#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/brianbruggeman/repos/slot-0/proxima-wt-seal"
SLOT_ZERO_ROOT="/Users/brianbruggeman/repos/slot-0"
PROXIMA_MAIN_ROOT="/Users/brianbruggeman/repos/slot-0/proxima"

LOAD_ONE_MINUTE_THRESHOLD="${SEALED_PASS_LOAD_THRESHOLD:-5.0}"
QUIET_BOX_WAIT_TIMEOUT_SECONDS="${SEALED_PASS_WAIT_TIMEOUT_SECONDS:-600}"
QUIET_BOX_POLL_INTERVAL_SECONDS="${SEALED_PASS_POLL_INTERVAL_SECONDS:-15}"

LLAMA_BENCH_BINARY="/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench"
OPENCHAT_GGUF_MODEL="/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
TORCH_REFERENCE_ROOT="$PROXIMA_MAIN_ROOT/proxima-onnx/scripts/torch_reference"
TORCH_VENV_PYTHON="$TORCH_REFERENCE_ROOT/venv/bin/python"
TORCH_TRAIN_BENCH_SCRIPT="$TORCH_REFERENCE_ROOT/train_bench.py"

ORCH2_WORKTREE="$SLOT_ZERO_ROOT/proxima-wt-orch2"
NANOFIX_WORKTREE="$SLOT_ZERO_ROOT/proxima-wt-nanofix"
TRANSA_WORKTREE="$SLOT_ZERO_ROOT/proxima-wt-transa"
TRAIN_WORKTREE="$SLOT_ZERO_ROOT/proxima-wt-train"

# same constants the discipline log (ROWs 193/221) uses; nothing under
# proxima-model-interop or omega computes GB/s or GMAC/s, so a cell using
# different numbers here is not comparable to the incumbent rows.
MACS_PER_TOKEN=7110402048
WEIGHT_BYTES_PER_TOKEN_GB=3.9996

RUN_TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="$REPO_ROOT/target/sealed-pass/$RUN_TIMESTAMP"
CARGO_TARGET_DIR="$OUTPUT_ROOT/cargo-target"

mkdir -p "$OUTPUT_ROOT" "$CARGO_TARGET_DIR"
export CARGO_TARGET_DIR

fail_with_missing_artifact() {
  local description="$1"
  local path="$2"
  echo "FATAL: missing artifact for $description: $path" >&2
  exit 1
}

require_file() {
  local description="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    fail_with_missing_artifact "$description" "$path"
  fi
}

require_executable() {
  local description="$1"
  local path="$2"
  require_file "$description" "$path"
  if [ ! -x "$path" ]; then
    echo "FATAL: artifact for $description is not executable: $path" >&2
    exit 1
  fi
}

preflight_required_artifacts() {
  require_executable "incumbent llama-bench binary" "$LLAMA_BENCH_BINARY"
  require_file "openchat Q4_K_S gguf model" "$OPENCHAT_GGUF_MODEL"
  require_executable "torch reference venv interpreter" "$TORCH_VENV_PYTHON"
  require_file "torch train_bench.py" "$TORCH_TRAIN_BENCH_SCRIPT"
  require_file "torch diagnostics.py" "$TORCH_REFERENCE_ROOT/diagnostics.py"
}

current_load_one_minute() {
  sysctl -n vm.loadavg | awk '{print $2}'
}

concurrent_build_process_line() {
  pgrep -fl 'cargo|rustc' || true
}

box_is_quiet() {
  local load_one_minute="$1"
  local build_process_line="$2"
  awk -v load="$load_one_minute" -v threshold="$LOAD_ONE_MINUTE_THRESHOLD" \
    'BEGIN { exit !(load < threshold) }' \
    && [ -z "$build_process_line" ]
}

# the script refuses to measure on a box it has not proven quiet: bounded
# poll on 1-min load AND absence of a concurrent cargo/rustc, loud exit
# non-zero on timeout, never a silent proceed.
wait_for_quiet_box() {
  local deadline_epoch_seconds
  deadline_epoch_seconds=$(( $(date +%s) + QUIET_BOX_WAIT_TIMEOUT_SECONDS ))

  while true; do
    local load_one_minute build_process_line
    load_one_minute="$(current_load_one_minute)"
    build_process_line="$(concurrent_build_process_line)"

    if box_is_quiet "$load_one_minute" "$build_process_line"; then
      echo "$load_one_minute"
      return 0
    fi

    if [ "$(date +%s)" -ge "$deadline_epoch_seconds" ]; then
      echo "FATAL: box never went quiet within ${QUIET_BOX_WAIT_TIMEOUT_SECONDS}s" >&2
      echo "FATAL: last 1-min load $load_one_minute, threshold $LOAD_ONE_MINUTE_THRESHOLD" >&2
      if [ -n "$build_process_line" ]; then
        echo "FATAL: concurrent build process(es):" >&2
        echo "$build_process_line" >&2
      fi
      exit 1
    fi

    sleep "$QUIET_BOX_POLL_INTERVAL_SECONDS"
  done
}

# every arm of a pair runs inside this one script invocation so both cells
# of a pair share box conditions; a paired arm measured under a different
# run is not a pair.
run_cell() {
  local cell_name="$1"
  local working_dir="$2"
  shift 2
  local log_file="$OUTPUT_ROOT/$cell_name.log"

  {
    echo "cell: $cell_name"
    echo "working_dir: $working_dir"
    echo "command: env -C $working_dir $*"
    echo "started_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$log_file"

  local load_before
  load_before="$(wait_for_quiet_box)"
  echo "load_one_minute_before: $load_before" >> "$log_file"

  local command_exit_code=0
  env -C "$working_dir" "$@" >> "$log_file" 2>&1 || command_exit_code=$?

  local load_after
  load_after="$(current_load_one_minute)"
  {
    echo "load_one_minute_after: $load_after"
    echo "finished_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "exit_code: $command_exit_code"
  } >> "$log_file"

  if [ "$command_exit_code" -ne 0 ]; then
    echo "FATAL: cell '$cell_name' exited $command_exit_code, see $log_file" >&2
    exit "$command_exit_code"
  fi

  echo "sealed: $cell_name (load before $load_before, after $load_after) -> $log_file"
}

skip_cell() {
  local cell_name="$1"
  local reason="$2"
  local log_file="$OUTPUT_ROOT/$cell_name.log"
  {
    echo "cell: $cell_name"
    echo "status: SKIPPED"
    echo "reason: $reason"
    echo "skipped_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$log_file"
  echo "skipped: $cell_name ($reason) -> $log_file"
}

require_optional_worktree_file() {
  local worktree_dir="$1"
  local relative_path="$2"
  if [ ! -d "$worktree_dir" ]; then
    echo "MISSING_WORKTREE:$worktree_dir"
    return 1
  fi
  if [ ! -e "$worktree_dir/$relative_path" ]; then
    echo "MISSING_FILE:$worktree_dir/$relative_path"
    return 1
  fi
  return 0
}

echo "sealed-pass: output directory $OUTPUT_ROOT"
echo "sealed-pass: MACS_PER_TOKEN=$MACS_PER_TOKEN WEIGHT_BYTES_PER_TOKEN_GB=$WEIGHT_BYTES_PER_TOKEN_GB"

preflight_required_artifacts

run_cell "01-incumbent-gpu-ngl99-t1" "$REPO_ROOT" \
  "$LLAMA_BENCH_BINARY" -m "$OPENCHAT_GGUF_MODEL" -ngl 99 -t 1 -n 32 -r 5

run_cell "02-incumbent-gpu-ngl99-t8" "$REPO_ROOT" \
  "$LLAMA_BENCH_BINARY" -m "$OPENCHAT_GGUF_MODEL" -ngl 99 -t 8 -n 32 -r 5

# live-flag control: at ngl 99 the thread flag changes nothing (measured
# 57.02 vs 57.24 tok/s), so without an arm that DOES scale with -t (measured
# 5.62 -> 25.48 tok/s, 4.53x at ngl 0) there is no way to tell "threads do
# not matter on this path" apart from "the -t flag never reached the
# runtime".
run_cell "03-control-cpu-ngl0-t1" "$REPO_ROOT" \
  "$LLAMA_BENCH_BINARY" -m "$OPENCHAT_GGUF_MODEL" -ngl 0 -t 1 -n 32 -r 5

run_cell "04-control-cpu-ngl0-t8" "$REPO_ROOT" \
  "$LLAMA_BENCH_BINARY" -m "$OPENCHAT_GGUF_MODEL" -ngl 0 -t 8 -n 32 -r 5

if orch2_missing_reason="$(require_optional_worktree_file "$ORCH2_WORKTREE" \
  "proxima-model-interop/src/bind.rs")"; then
  run_cell "05-proxima-gpu-decode-orch-threads-1" "$ORCH2_WORKTREE" \
    env PROXIMA_PREFAULT=1 PROXIMA_MAX_TOKENS=8 PROXIMA_ORCH_THREADS=1 \
    cargo test -p proxima-model-interop --release --lib --features metal,instrument \
    -- --ignored --exact --nocapture \
    bind::real_openchat_file::runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache

  run_cell "06-proxima-gpu-decode-orch-threads-8" "$ORCH2_WORKTREE" \
    env PROXIMA_PREFAULT=1 PROXIMA_MAX_TOKENS=8 PROXIMA_ORCH_THREADS=8 \
    cargo test -p proxima-model-interop --release --lib --features metal,instrument \
    -- --ignored --exact --nocapture \
    bind::real_openchat_file::runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache
else
  skip_cell "05-proxima-gpu-decode-orch-threads-1" "$orch2_missing_reason"
  skip_cell "06-proxima-gpu-decode-orch-threads-8" "$orch2_missing_reason"
fi

# this cell's fixed-cost-cancelled marginal arm settles the GPU lane's
# direction: it reads either ~70.76 GB/s (the kernel IS the decode gap) or
# ~230 (the kernel already matches llama.cpp's 228.9 parity and the 3.24x
# gap lives elsewhere) -- two contended runs gave each answer, so this
# cell only means anything measured on a quiet box.
if nanofix_missing_reason="$(require_optional_worktree_file "$NANOFIX_WORKTREE" \
  "omega/examples/q4k_matvec_nsg_probe.rs")"; then
  run_cell "07-q4k-nano-probe-baseline" "$NANOFIX_WORKTREE" \
    cargo run -p omega --release --features metal,cpu,instrument \
    --example q4k_matvec_nsg_probe

  run_cell "08-q4k-nano-probe-packed-row-nsg2" "$NANOFIX_WORKTREE" \
    cargo run -p omega --release --features metal,cpu,instrument,metal-packed-row-nsg2 \
    --example q4k_matvec_nsg_probe
else
  skip_cell "07-q4k-nano-probe-baseline" "$nanofix_missing_reason"
  skip_cell "08-q4k-nano-probe-packed-row-nsg2" "$nanofix_missing_reason"
fi

TRAIN_STEP_WORKTREE=""
if [ -d "$TRANSA_WORKTREE" ]; then
  TRAIN_STEP_WORKTREE="$TRANSA_WORKTREE"
elif [ -d "$TRAIN_WORKTREE" ]; then
  TRAIN_STEP_WORKTREE="$TRAIN_WORKTREE"
fi

if [ -n "$TRAIN_STEP_WORKTREE" ] && \
  train_step_missing_reason="$(require_optional_worktree_file "$TRAIN_STEP_WORKTREE" \
    "proxima-autograd/benches/train_step_lane.rs")"; then
  run_cell "09-proxima-train-step-bench" "$TRAIN_STEP_WORKTREE" \
    cargo bench -p proxima-autograd --bench train_step_lane --features train-step-bench
else
  if [ -z "$TRAIN_STEP_WORKTREE" ]; then
    train_step_missing_reason="MISSING_WORKTREE:$TRANSA_WORKTREE and $TRAIN_WORKTREE"
  fi
  skip_cell "09-proxima-train-step-bench" "$train_step_missing_reason"
fi

# accelerate spawns its own threads regardless of torch.set_num_threads, so
# an unpinned "t=1" arm is not actually single-threaded and the pair would
# be a lie; diagnostics.py:report_and_verify_threads prints
# VECLIB_MAXIMUM_THREADS as <unset> unless this pins it alongside
# OMP_NUM_THREADS.
run_cell "10-torch-train-step-threads-1" "$TORCH_REFERENCE_ROOT" \
  env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$TORCH_VENV_PYTHON" "$TORCH_TRAIN_BENCH_SCRIPT" --threads 1 --warmup 20 --steps 200

run_cell "11-torch-train-step-threads-8" "$TORCH_REFERENCE_ROOT" \
  env VECLIB_MAXIMUM_THREADS=8 OMP_NUM_THREADS=8 \
  "$TORCH_VENV_PYTHON" "$TORCH_TRAIN_BENCH_SCRIPT" --threads 8 --warmup 20 --steps 200

echo "sealed-pass: complete, logs under $OUTPUT_ROOT"
