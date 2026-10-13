#!/usr/bin/env bash
# wire-to-weights-gate.sh
# the smoke test compiles to zero tests without the wire-to-weights-demo feature, so no default-feature run can fail it; this gate runs it with the feature and asserts its count.
#
# usage: bash scripts/wire-to-weights-gate.sh
#
# fetches the real mnist.onnx and the MNIST t10k idx files into a cache dir,
# each pinned by sha256, then runs `--test wire_to_weights_smoke` and fails
# unless exactly 1 test ran and passed.

set -euo pipefail

export CARGO_TERM_COLOR=never

fixtures="${WIRE_TO_WEIGHTS_FIXTURES:-${HOME}/.cache/proxima-wire-to-weights}"
burn_commit="993e212d8d706268750e3760413c1d9ff589c6ca"
model_url="https://raw.githubusercontent.com/tracel-ai/burn/${burn_commit}/examples/onnx-inference/src/model/mnist.onnx"
model_sha="5de2936be8f335e42049fc72a268d951481c782b113846e8c0021d8af7b4b519"
mnist_url="https://ossci-datasets.s3.amazonaws.com/mnist"
images_sha="0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"
labels_sha="ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

verify() {
    local path="$1" expected="$2"
    local actual
    actual="$(sha256_of "${path}")"
    if [ "${actual}" != "${expected}" ]; then
        printf 'ERROR: %s sha256 %s != pinned %s\n' "${path}" "${actual}" "${expected}" >&2
        exit 1
    fi
}

fetch_pinned() {
    local url="$1" dest="$2" expected="$3" gunzip_first="$4"
    if [ -f "${dest}" ] && [ "$(sha256_of "${dest}")" = "${expected}" ]; then
        printf 'cached: %s\n' "${dest}"
        return
    fi
    mkdir -p "$(dirname "${dest}")"
    if [ "${gunzip_first}" = "gz" ]; then
        curl -fsSL "${url}" | gunzip > "${dest}.part"
    else
        curl -fsSL "${url}" -o "${dest}.part"
    fi
    mv "${dest}.part" "${dest}"
    verify "${dest}" "${expected}"
    printf 'fetched: %s sha256=%s\n' "${dest}" "${expected}"
}

printf '\n== wire-to-weights gate ==\n'

fetch_pinned "${model_url}" "${fixtures}/mnist.onnx" "${model_sha}" raw
fetch_pinned "${mnist_url}/t10k-images-idx3-ubyte.gz" "${fixtures}/mnist/test/t10k-images-idx3-ubyte" "${images_sha}" gz
fetch_pinned "${mnist_url}/t10k-labels-idx1-ubyte.gz" "${fixtures}/mnist/test/t10k-labels-idx1-ubyte" "${labels_sha}" gz

export PROXIMA_MNIST_ONNX="${fixtures}/mnist.onnx"
export PROXIMA_MNIST_DIR="${fixtures}/mnist"

printf '\n-- wire_to_weights_smoke --\n'
nextest_output="$(cargo nextest run --color never -p proxima-lib --test wire_to_weights_smoke --features wire-to-weights-demo 2>&1)"
printf '%s\n' "${nextest_output}"
passed_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests? run: [0-9]+ passed' | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ "${passed_count:-}" != "1" ]; then
    printf 'ERROR: wire_to_weights_smoke expected exactly 1 passed, got "%s"\n' "${passed_count:-}" >&2
    exit 1
fi
printf 'wire_to_weights_smoke passed: %s\n' "${passed_count}"

printf '\n== wire-to-weights gate: PASS ==\n'
