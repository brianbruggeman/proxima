#!/usr/bin/env bash
# proxima-tensor-gate.sh — feature/target matrix gate for proxima-tensor.
#
# Why it exists: measured 2026-08-19, `proxima-tensor` appeared in ZERO
# workflows and ZERO gate scripts, while carrying seven features and more
# `target_arch` branching than any other crate in the tree. Two holes that
# cost real breakage:
#
#   1. `tensor-bgpool` is referenced exactly once in the entire repository —
#      its own definition in proxima-tensor/Cargo.toml. Nothing turns it on.
#      It swaps `cpu::run_chunks_threaded` from `std::thread::scope` onto
#      prime's crossbeam-deque `ProximaBackgroundPool`, so it is a whole
#      alternate execution backend that no command has ever compiled, never
#      mind run.
#   2. Nothing ever cross-compiled. `cpu.rs` gates the NEON width tile, the
#      dot tile and their plans on `target_arch = "aarch64"`, and the
#      non-aarch64 arm was dead-code-broken on an ordinary
#      `cargo check --target x86_64-*`. Every developer box here is arm64, so
#      the host build stayed green while the portable arm did not compile at
#      all. `no-std.yml`'s matrix cross-compiles `prime` only.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with its
# siblings is a feature nobody has proven.
#
# The doctest cell asserts a NONZERO count. `cargo test --doc` exits 0 when it
# matches nothing, so a bare cell would be the zero-passed-reads-as-all-passed
# trap AGENTS.md warns about — the same defect that let this workspace ship a
# doctest gate which never ran a doctest.
#
# No `ggml-bench` cell: that feature's build.rs wants a statically linked ggml
# checkout on disk (plus, for five of its six benches, a real multi-GiB GGUF
# checkpoint), so it belongs with the bench harnesses, not a correctness
# gate. Named here so its absence is a decision on the record, not an oversight.
# The local-only execution path (this is a CI-unsatisfiable dependency, not a
# missing wire) is `scripts/bench-vs-ggml.sh` -- read its header for the exact
# GGML_BUILD_DIR/PROXIMA_BENCH_GGUF_PATH re-prove command.
#
# Usage:  bash scripts/proxima-tensor-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# checked, never linked, so this runs from an arm64 host as well as from
# ubuntu CI where it is the native target.
PORTABLE_TARGET="x86_64-unknown-linux-gnu"

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-tensor --all-targets"
    "default tests|cargo nextest run -p proxima-tensor --no-fail-fast"
    "default clippy|cargo clippy -p proxima-tensor --all-targets -- -D warnings"
    "default rustdoc|cargo doc -p proxima-tensor --no-deps"

    # the portable arm. every NEON path is cfg'd out here, so this is the only
    # cell that compiles the fallback code an aarch64 dev box never builds.
    #
    # lib only, deliberately: `--all-targets` pulls dev-dependencies, and
    # `alloca`'s build script does not cross-compile from arm64-darwin (it
    # exits 1 before rustc is reached), which would report a toolchain limit as
    # a source defect. On CI this costs nothing — the runner is ubuntu-x86_64,
    # so PORTABLE_TARGET is the NATIVE target there and the `default` cells
    # above already compile every test, example and bench against it.
    "portable check|cargo check -p proxima-tensor --target ${PORTABLE_TARGET}"

    # tiers. `alloc` alone is the no_std floor this crate claims.
    #
    # WHAT THESE TWO CELLS DO NOT COVER: `pub mod cpu` is `#[cfg(feature =
    # "std")]` (`src/lib.rs:204`), so an `alloc`-only build compiles ZERO lines
    # of `cpu.rs` -- the largest file in the crate and the one carrying every
    # kernel. Three separate agents in one session read these cells green and
    # reported them as coverage for a `cpu.rs` change; they prove the crate's
    # alloc tier still builds, and nothing whatsoever about the kernels. The
    # cell that actually compiles `cpu.rs` against a non-host arch is
    # `portable check` above (default features, so `std`, so `cpu`).
    "alloc tier check|cargo check -p proxima-tensor --no-default-features --features alloc"
    "alloc tier clippy|cargo clippy -p proxima-tensor --no-default-features --features alloc -- -D warnings"
    "std without config|cargo clippy -p proxima-tensor --all-targets --no-default-features --features std -- -D warnings"

    # each feature ALONE — unification must not be what makes it compile
    "config alone clippy|cargo clippy -p proxima-tensor --all-targets --no-default-features --features config -- -D warnings"
    "test-support alone clippy|cargo clippy -p proxima-tensor --no-default-features --features std,test-support -- -D warnings"

    # instrument carries four tests the default cell does not run, plus the
    # telemetry span wiring. The two cells therefore report DIFFERENT counts —
    # quoting one as the other's baseline is a real mistake that has been made:
    # a task brief cited the instrument number as the default expectation, and
    # the agent had to measure to catch it. Whatever the absolute numbers drift
    # to, `instrument tests` is always `default tests` plus exactly those four.
    "instrument clippy|cargo clippy -p proxima-tensor --all-targets --features instrument -- -D warnings"
    "instrument tests|cargo nextest run -p proxima-tensor --features instrument --no-fail-fast"

    # the default clippy cell above never compiles the non-default feature
    # surface (`instrument`, `tensor-bgpool`, `test-support`, `q4k-int8-dot`,
    # `config`) unified together, which is how lint errors that only surface
    # under unification would sit unseen. an explicit feature list, NOT
    # --all-features: --all-features pulls in `ggml-bench`, whose build.rs
    # wants a statically linked ggml checkout on disk (see the module header
    # above) -- that feature belongs with the bench harnesses, not this
    # correctness gate, so it is named here and left out on purpose.
    "all-non-bench-features clippy|cargo clippy -p proxima-tensor --all-targets --no-default-features --features alloc,std,config,instrument,tensor-bgpool,test-support,q4k-int8-dot -- -D warnings"

    # the default rustdoc cell above never compiles q4k-int8-dot's items, which
    # is how ~11 doc errors (private-item intra-doc links, one arch-invisible
    # link) sat in cpu.rs unseen until a dedicated `cargo doc` run found them.
    "q4k-int8-dot rustdoc|cargo doc -p proxima-tensor --no-deps --features q4k-int8-dot"

    # the backend nothing else in the tree compiles. clippy --all-targets first
    # so a break shows as a compile error rather than a missing test.
    "tensor-bgpool clippy|cargo clippy -p proxima-tensor --all-targets --features tensor-bgpool -- -D warnings"
    "tensor-bgpool tests|cargo nextest run -p proxima-tensor --features tensor-bgpool --no-fail-fast"

    # examples are compiled by --all-targets and never RUN by it. spec_block
    # asserts every softmax row sums to 1.0, so running it proves the TOML
    # spec path actually computes attention rather than merely parsing;
    # algebra_reach proves the non-gemm algebra instantiations still evaluate.
    #
    # only attention_block is driven here: spec_block hardcodes model=8 and one
    # buffer order ([x, scale, wq, wk, wv]), and gqa_attention.toml's five
    # inputs are [x, wq, wk, wv, group_ones] — a different arity story. The gqa
    # specs are evaluated by their own two cases in spec.rs, which the default
    # tests cell already runs.
    "spec_block example RUNS|cargo run -p proxima-tensor --example spec_block -- proxima-tensor/specs/attention_block.toml 4"
    "algebra_reach example RUNS|cargo run -p proxima-tensor --example algebra_reach -- 64 1 1"
)

passed=0
failed=0
declare -a failures

for cell in "${cells[@]}"; do
    label="${cell%%|*}"
    command="${cell#*|}"

    printf '\n== %s ==\n%s\n' "$label" "$command"

    if eval "$command"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

# doctests get their own cell because the count has to be asserted. nextest
# does not run doctests at all, so `cargo test --doc` is the only thing that
# reaches them, and it reports success on an empty match.
printf '\n== doctests (count asserted nonzero) ==\n'
doc_log="$(mktemp)"
if cargo test --doc -p proxima-tensor > "$doc_log" 2>&1; then
    doc_count="$(awk '/^test result:/ { total += $4 } END { print total + 0 }' "$doc_log")"
    printf 'doctests passed: %s\n' "$doc_count"
    if [ "$doc_count" -lt 1 ]; then
        printf 'RED: cargo test --doc ran %s doctests. an empty match exits 0 and reads as green.\n' "$doc_count"
        failed=$((failed + 1))
        failures+=("doctests ran zero")
    else
        passed=$((passed + 1))
    fi
else
    cat "$doc_log"
    failed=$((failed + 1))
    failures+=("doctests")
fi
rm -f "$doc_log"

printf '\n== proxima-tensor-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-tensor-gate: all green.\n'
