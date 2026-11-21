#!/usr/bin/env bash
# omega-train-gate.sh — the training surface's feature/target gate.
#
# `proxima-autograd` is a real dependency of omega only behind `train`, so the
# tier build without it must still compile. The tests and the benchmark are the
# two consumers of the config surface: the tests pin `benches/llama-finetune.toml`
# to llama.cpp's finetune defaults, and the benchmark runs one real step under
# the same file. `cargo nextest` exits 0 on an empty selection, so every step
# asserts its own count instead of trusting the exit code alone.
#
# Usage: bash scripts/omega-train-gate.sh

set -euo pipefail

export CARGO_TERM_COLOR=never

crate="proxima-omega"
crate_dir="omega"
features="train,cpu"

if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${repo_root}"

# `git wt cargo` pins CARGO_TARGET_DIR to the worktree; bare cargo would build
# into whatever the shell exports, which is the shared target the .wt rule bans.
case "${repo_root}" in
    */.wt/*) cargo_cmd=(git wt cargo) ;;
    *) cargo_cmd=(cargo) ;;
esac

passed_count() {
    local output="$1"
    printf '%s\n' "${output}" | grep -oE '[0-9]+ tests run: [0-9]+ passed' | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | tail -1
}

run_and_print() {
    local label="$1"
    shift
    local output status=0
    output="$("$@" 2>&1)" || status=$?
    printf '%s\n' "${output}"
    if [ "${status}" -ne 0 ]; then
        printf 'ERROR: %s exited %s\n' "${label}" "${status}" >&2
        exit "${status}"
    fi
}

printf '\n== %s training gate ==\n' "${crate}"

printf '\n[1/6] std tier still builds without the train feature\n'
run_and_print "std tier" "${cargo_cmd[@]}" check -p "${crate}" --no-default-features --features std

printf '\n[2/6] train graph builder unit tests\n'
unit_output="$("${cargo_cmd[@]}" nextest run -p "${crate}" --features "${features}" --lib -E 'test(train::tests)' 2>&1)" || {
    printf '%s\n' "${unit_output}"
    printf 'ERROR: the train unit tests failed\n' >&2
    exit 1
}
printf '%s\n' "${unit_output}"
unit_passed="$(passed_count "${unit_output}")"
if [ "${unit_passed:-0}" -ne 9 ]; then
    printf 'ERROR: expected 9 train unit tests, got %s\n' "${unit_passed:-0}" >&2
    exit 1
fi

printf '\n[3/6] training API and llama config tests\n'
api_output="$("${cargo_cmd[@]}" nextest run -p "${crate}" --features "${features}" --test training_api 2>&1)" || {
    printf '%s\n' "${api_output}"
    printf 'ERROR: the training API tests failed\n' >&2
    exit 1
}
printf '%s\n' "${api_output}"
api_passed="$(passed_count "${api_output}")"
# the Metal parity test compiles only on macOS with the default `metal` feature
expected_api=15
if [ "$(uname -s)" = "Darwin" ]; then
    expected_api=16
fi
if [ "${api_passed:-0}" -ne "${expected_api}" ]; then
    printf 'ERROR: expected %s training_api tests, got %s\n' "${expected_api}" "${api_passed:-0}" >&2
    exit 1
fi

printf '\n[4/6] numerics stay proxima-numeric: no format-name token in the surface\n'
if [ ! -f "${crate_dir}/src/train.rs" ]; then
    printf 'ERROR: %s/src/train.rs is missing, so the format-token check proves nothing\n' "${crate_dir}" >&2
    exit 1
fi
format_tokens="$(grep -Eio '\b(f16|bf16|q2_k|q3_k|q4_0|q4_1|q4_k|q5_0|q5_1|q5_k|q6_k|q8_0|bf32)\b' "${crate_dir}/src/train.rs" | sort -u || true)"
if [ -n "${format_tokens}" ]; then
    printf 'ERROR: train.rs names a format instead of a proxima-numeric type: %s\n' "${format_tokens}" >&2
    exit 1
fi

printf '\n[5/6] the module doc example compiles and runs\n'
doc_output="$("${cargo_cmd[@]}" test --doc -p "${crate}" --features "${features}" -- src/train.rs 2>&1)" || {
    printf '%s\n' "${doc_output}"
    printf 'ERROR: the train doc example failed\n' >&2
    exit 1
}
printf '%s\n' "${doc_output}"
doc_passed="$(printf '%s\n' "${doc_output}" | grep -E 'test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ "${doc_passed:-0}" -ne 1 ]; then
    printf 'ERROR: expected 1 train doc example, got %s\n' "${doc_passed:-0}" >&2
    exit 1
fi

printf '\n[6/6] llama-config benchmark runs a real step\n'
bench_output="$("${cargo_cmd[@]}" bench -p "${crate}" --features "${features}" --bench train_step_llama -- --test 2>&1)" || {
    printf '%s\n' "${bench_output}"
    printf 'ERROR: the training benchmark failed to run\n' >&2
    exit 1
}
printf '%s\n' "${bench_output}"
config_rows="$(printf '%s\n' "${bench_output}" | grep -c '^train_step_llama: optimizer=adamw ' || true)"
if [ "${config_rows:-0}" -ne 1 ]; then
    printf 'ERROR: expected 1 bench config row, got %s\n' "${config_rows:-0}" >&2
    exit 1
fi
if ! printf '%s\n' "${bench_output}" | grep -q 'train_step_llama: optimizer=adamw .*batch=2048 ubatch=512'; then
    printf 'ERROR: the bench did not print llama.cpp b11671 finetune defaults (adamw, batch 2048, ubatch 512)\n' >&2
    exit 1
fi

printf '\n== omega-train-gate: all green ==\n'
