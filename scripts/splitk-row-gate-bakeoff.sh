#!/usr/bin/env bash
# splitk-row-gate-bakeoff.sh
# One-shot driver for the split-K row-gate bake-off: llama-bench (home-turf
# incumbent) plus our four arms (default / splitk-4096 / splitk-1024 /
# splitk-0), 3 interleaved rounds, plus one per-op profile run per arm.
# Not part of any CI gate -- a measurement harness, run manually.
set -uo pipefail

GGUF="/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
LLAMA_BENCH="/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench"
LOG_DIR="/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6e203711-bd50-48cc-9ade-409668bdafdd/scratchpad/splitk-logs"
mkdir -p "${LOG_DIR}"

declare -A ARM_BIN=(
  [default]="/Users/brianbruggeman/repos/slot-0/proxima-wt-splitkrows/target-arms/default/release/deps/proxima_model_interop-cde21533ced4e3ef"
  [splitk-4096]="/Users/brianbruggeman/repos/slot-0/proxima-wt-splitkrows/target-arms/splitk-4096/release/deps/proxima_model_interop-4a2c155b41d76c50"
  [splitk-1024]="/Users/brianbruggeman/repos/slot-0/proxima-wt-splitkrows/target-arms/splitk-1024/release/deps/proxima_model_interop-4a2c155b41d76c50"
  [splitk-0]="/Users/brianbruggeman/repos/slot-0/proxima-wt-splitkrows/target-arms/splitk-0/release/deps/proxima_model_interop-4a2c155b41d76c50"
)
ARM_ORDER=(default splitk-4096 splitk-1024 splitk-0)

# cdb-daemon and sccache live under ~/.cargo/bin and match a naive
# `cargo|rustc|nextest` grep on their own PATH, even though neither is a
# measurer -- exclude both by name so the quiet check reflects actual
# build/test/bench activity, not permanently-resident tooling.
loadout() {
  printf 'loadout at %s:\n' "$(date)"
  pgrep -fl 'cargo|rustc|nextest' 2>&1 | grep -v grep | grep -v 'cdb-daemon\|sccache' \
    || printf '(quiet -- no cargo/rustc/nextest)\n'
}

wait_for_quiet() {
  local waited=0
  while true; do
    local others
    others=$(pgrep -fl 'cargo|rustc|nextest' 2>/dev/null | grep -v grep | grep -v 'cdb-daemon\|sccache' | grep -v "$$" || true)
    if [[ -z "${others}" ]]; then
      return 0
    fi
    if [[ "${waited}" -ge 1500 ]]; then
      printf 'proceeding after %ss wait, loadout still non-empty:\n%s\n' "${waited}" "${others}"
      return 0
    fi
    sleep 120
    waited=$((waited + 120))
  done
}

for round in 1 2 3; do
  printf '\n=== ROUND %s ===\n' "${round}" | tee -a "${LOG_DIR}/bakeoff-summary.log"
  wait_for_quiet
  loadout | tee -a "${LOG_DIR}/bakeoff-summary.log"

  printf '-- llama-bench round %s --\n' "${round}" | tee -a "${LOG_DIR}/bakeoff-summary.log"
  "${LLAMA_BENCH}" -m "${GGUF}" -n 32 -p 0 -r 5 -t 8 -ngl 99 \
    > "${LOG_DIR}/llama-bench-round${round}.log" 2>&1
  echo "EXIT=$?" >> "${LOG_DIR}/llama-bench-round${round}.log"
  tail -6 "${LOG_DIR}/llama-bench-round${round}.log" | tee -a "${LOG_DIR}/bakeoff-summary.log"

  for arm in "${ARM_ORDER[@]}"; do
    printf -- '-- arm=%s round=%s decode loop --\n' "${arm}" "${round}" | tee -a "${LOG_DIR}/bakeoff-summary.log"
    PROXIMA_MAX_TOKENS=8 "${ARM_BIN[${arm}]}" --exact --nocapture --ignored \
      'bind::real_openchat_file::runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache' \
      > "${LOG_DIR}/decode-${arm}-round${round}.log" 2>&1
    echo "EXIT=$?" >> "${LOG_DIR}/decode-${arm}-round${round}.log"
    grep -E 'metal_decode_summary|token_breakdown_metal|test result' "${LOG_DIR}/decode-${arm}-round${round}.log" | tail -3 \
      | tee -a "${LOG_DIR}/bakeoff-summary.log"
  done
done

printf '\n=== PER-OP PROFILE (one run per arm) ===\n' | tee -a "${LOG_DIR}/bakeoff-summary.log"
for arm in "${ARM_ORDER[@]}"; do
  printf -- '-- arm=%s profile --\n' "${arm}" | tee -a "${LOG_DIR}/bakeoff-summary.log"
  PROXIMA_METAL_OP_PROFILE_STEP=3 "${ARM_BIN[${arm}]}" --exact --nocapture --ignored \
    'bind::real_openchat_file::profiles_one_real_decode_step_by_per_op_gpu_time' \
    > "${LOG_DIR}/profile-${arm}.log" 2>&1
  echo "EXIT=$?" >> "${LOG_DIR}/profile-${arm}.log"
  grep -E 'op_profile_family |test result' "${LOG_DIR}/profile-${arm}.log" \
    | tee -a "${LOG_DIR}/bakeoff-summary.log"
done

printf '\n=== DONE ===\n' | tee -a "${LOG_DIR}/bakeoff-summary.log"
