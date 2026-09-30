#!/usr/bin/env bash
set -euo pipefail

parse_passed() {
    awk '$1 == "PASS" { print $(NF-1), $NF }' | LC_ALL=C sort
}

sample_output() {
    cat <<'EOF'
        PASS [   0.004s] proxima-model-interop memory_fit::tests::a
 PASS [   0.011s] proxima-tensor shape::tests::rank_of_scalar_is_zero
        FAIL [   0.020s] omega graph::tests::broken
        SKIP [         ] proxima-tensor slow::tests::ignored
     Summary [   0.500s] 4 tests run: 2 passed, 1 failed, 1 skipped
EOF
}

interop_features() {
    local features="proxima-model-interop/std"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        features+=",proxima-model-interop/metal"
    fi
    printf '%s' "$features"
}

run_nextest() {
    cargo nextest run -p proxima-model-interop -p proxima-tensor -p omega \
        --features "$(interop_features)" \
        --no-fail-fast --status-level pass --final-status-level none 2>&1 || {
        local status=$?
        [[ $status -eq 100 ]] || return "$status"
    }
}

main() {
    if [[ "${1:-}" == "--self-test" ]]; then
        sample_output | parse_passed
        return
    fi
    local passed
    passed="$(run_nextest | parse_passed)"
    if [[ -z "$passed" ]]; then
        echo "passed_tests.sh: nextest reported zero passing tests" >&2
        return 1
    fi
    printf '%s\n' "$passed"
}

main "$@"
