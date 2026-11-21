#!/usr/bin/env bash
# PROXIMAOMEGA-71 acceptance: a decodable codec binds through its decoder,
# an undecodable one still refuses, and the bind stays green.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

cargo_test() {
    git wt cargo test -p proxima-model-interop --features std --lib "$1"
}

decoded="$(cargo_test codec_without_a_packed_kernel 2>&1)"
grep -q 'test result: ok\. 2 passed' <<<"$decoded"

bench="$(git wt cargo bench -p proxima-model-interop --features std --bench codec_decode_fallback -- --test 2>&1)"
grep -q 'mismatch=0' <<<"$bench"
grep -q 'batch=2048 ubatch=512' <<<"$bench"

for filter in 'bind::' 'bind_leaves::'; do
    out="$(cargo_test "$filter" 2>&1)"
    grep -q 'test result: ok\.' <<<"$out"
    if grep -Eq 'test result: FAILED|[1-9][0-9]* failed' <<<"$out"; then
        printf '%s\n' "$out" >&2
        exit 1
    fi
done

echo "proximaomega71-gate: all green"
