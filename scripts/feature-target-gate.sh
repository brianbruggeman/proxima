#!/usr/bin/env bash
# feature-target-gate.sh — compile every TEST TARGET of the crates a branch
# touches, once per feature that gates test-only code.
#
# why this exists: `cargo check --all-targets --features std` prints "Finished"
# for every target whose gating feature is unset. A `#[cfg(feature = "...")]`
# test module that does not compile is therefore invisible to it — and it stays
# invisible to every crate that compiles this one as a *dependency*, because a
# dependency is never built with its tests. Two classes have reached a branch
# through that hole on 2026-11-08:
#
#   - a builder call site left on the old arity in a `metal-attn-split-rows`
#     test module (E0061, invisible to `--features std`);
#   - the same call site passing a bare integer where the parameter had become
#     `SharedExpertWidth` (E0308, same hole).
#
# The feature sets below are the ones in tree that gate test-only code. A new
# gating feature adds a row here; a row that selects nothing is reported as
# SELECTED-NOTHING rather than passing, because a gate that runs zero checks is
# indistinguishable from a passing one.
#
# usage: scripts/feature-target-gate.sh [base]        (default: origin/main)
#        scripts/feature-target-gate.sh --all          every crate below
#        scripts/feature-target-gate.sh --list [base]  print the selected crates and rows, run nothing
#        a changed crate also selects every gated crate that depends on it (normal or dev
#        dependency, transitively), read from `cargo metadata --no-deps`
#        inside a .wt/ worktree every cargo call goes through `git wt cargo`, which pins
#        the target to <worktree>/target whatever CARGO_TARGET_DIR the env exports;
#        elsewhere (main checkout, pre-push hook) bare cargo honours CARGO_TARGET_DIR
set -euo pipefail

# the caller's tree, not the script's: this repo is worked in worktrees, and a
# gate that always measured the primary checkout would grade the wrong commit
if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_root"

# one source of truth for the target: `git wt cargo` (tools/main-only-worktrees/policy.py)
# sets CARGO_TARGET_DIR to <worktree>/target itself, so the gate delegates instead of copying the path
case "$repo_root" in
    */.wt/*)
        cargo_cmd=(git wt cargo)
        target_dir="$repo_root/target"
        ;;
    *)
        cargo_cmd=(cargo)
        target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
        ;;
esac
printf 'feature-target-gate: target=%s\n' "$target_dir"

all=0
list=0
base="origin/main"
for arg in "$@"; do
    case "$arg" in
        --all) all=1 ;;
        --list) list=1 ;;
        *) base="$arg" ;;
    esac
done

# crate:feature-set pairs. An empty feature set means "default features".
# crate:@nostd:feature-set is a tier row: `cargo check --lib --no-default-features`
# plus those features (empty = bare). --lib because tests and examples need std;
# the tier claim is that the library itself builds without it, and a default-
# features --all-targets build can never notice that break.
# crate:@libtest: is `cargo test --lib --no-run` with default features: the lib's own
# test target, without the examples and integration tests that need std.
# crate:@libtestrun:feature-set is `cargo val --lib` plus those features: it RUNS the lib's
# tests, because a std-gated test tree that compiles can still be red and nothing else runs it.
# crate:@nodefault:feature-set is `cargo check --all-targets --no-default-features`
# plus those features: it compiles TEST targets for a feature set that excludes the
# crate's default features (omega defaults to metal; std,cuda never builds with it).
matrix=(
    "proxima-tensor:std"
    "proxima-model-interop:@libtest:"
    "proxima-model-interop:@libtestrun:std"
    "proxima-model-interop:std"
    "proxima-model-interop:std,metal,instrument"
    "proxima-model-interop:std,cuda"
    "proxima-model-interop:std,proxima-autograd"
    "proxima-omega:"
    "proxima-omega:@nodefault:std,cuda"
    "proxima-omega:cuda,wgpu-backend"
    "proxima-omega:wgpu-backend"
    "proxima-omega:cuda-driver"
    "proxima-omega:instrument,alloc-count,metal-attn-variants"
    "proxima-omega:onnx-door,mil-writer,coreml-lane"
    "proxima-omega:instrument,top-fraction-fusion"
    "proxima-omega:train,cpu"
    "proxima-omega:instrument,metal-reduction-literal"
    "proxima-omega:metal-q4k-mask-fma"
    "proxima-omega:metal-attn-variants"
    "proxima-omega:metal-moe-mul-mat-id"
    "proxima-omega:gated-delta-net-fusion"
    "proxima-omega:metal-gathered-packed-row,metal-q4_0-native,metal-horizontal-merge"
    "proxima-tensor:std,instrument"
    "proxima-tensor:std,gated-delta-net-fusion"
    "proxima-tensor:std,reduce-epilogue-fusion,identity-copy-alias,moe-stacked-combine,epilogue-profile-probe,kv-capacity-bucket"
    "proxima-model-interop:std,slow-tests"
    "proxima-model-interop:std,host-checkpoints"
    "proxima-model-interop:std,metal,instrument,metal-attn-variants,metal-reduction-literal"
    "proxima-model-interop:std,metal,moe-expert-prefetch"
    "proxima-numeric:std"
    "proxima-numeric:config"
    "proxima-gguf:"
    "proxima-tokenizer:"
    "proxima-tokenizer:hf"
    "proxima-autograd:std"
    "proxima-autograd:std,safetensors"
    "proxima-autograd:train-step-profile"
    "proxima-tensor:@nostd:alloc"
    "proxima-numeric:@nostd:alloc"
    "proxima-numeric:@nostd:"
    "proxima-gguf:@nostd:alloc"
    "proxima-tokenizer:@nostd:"
    "proxima-autograd:@nostd:alloc"
    "proxima-omega:@nostd:alloc"
    "proxima-omega:@nostd:"
    "proxima-model-interop:@nostd:"
)

# the package name of a top-level directory: the `name` in its manifest's
# [package] section, or the directory name when it has no manifest; a crate's
# directory need not match its name (omega/ holds proxima-omega).
package_name() {
    local dir="$1" name=""
    if [ -f "$dir/Cargo.toml" ]; then
        name="$(awk '
            /^\[package\]/ { in_package = 1; next }
            /^\[/ { in_package = 0 }
            in_package && /^name[ \t]*=/ {
                sub(/^name[ \t]*=[ \t]*"/, "")
                sub(/".*$/, "")
                print
                exit
            }' "$dir/Cargo.toml")"
    fi
    printf '%s\n' "${name:-$dir}"
}

# a crate the matrix names but no top-level directory's package_name yields is never selected by
# changed_crates, so its row would silently never run: fail naming it
unmapped_matrix_crates() {
    local crate dir found
    while IFS= read -r crate; do
        found=0
        for dir in */; do
            if [ "$(package_name "${dir%/}")" = "$crate" ]; then
                found=1
                break
            fi
        done
        if [ "$found" = 0 ]; then
            printf '%s\n' "$crate"
        fi
    done < <(printf '%s\n' "${matrix[@]%%:*}" | sort -u)
}

unmapped="$(unmapped_matrix_crates)"
if [ -n "$unmapped" ]; then
    printf 'feature-target-gate: matrix crate maps to no top-level directory: %s\n' "$unmapped" >&2
    exit 3
fi

gated_crates() {
    printf '%s\n' "${matrix[@]%%:*}" | sort -u
}

# "<crate> <dependency>" per line for every workspace package, normal and dev dependencies alike
# (a dev-dependency edge is what puts a crate's test targets in a dependency's blast radius).
# a failed read or an empty edge list is fatal: empty edges would silently skip every dependent
dependency_edges() {
    local metadata edges
    if ! command -v jq >/dev/null 2>&1; then
        echo 'feature-target-gate: needs jq to read cargo metadata' >&2
        exit 3
    fi
    if ! metadata="$(cargo metadata --format-version 1 --no-deps)"; then
        echo 'feature-target-gate: cargo metadata failed (stderr above); cannot derive dependents' >&2
        exit 3
    fi
    if ! edges="$(printf '%s' "$metadata" | jq -r '.packages[] | .name as $crate | .dependencies[] | "\($crate) \(.name)"')"; then
        echo 'feature-target-gate: jq failed reading cargo metadata; cannot derive dependents' >&2
        exit 3
    fi
    if [ -z "$edges" ]; then
        echo 'feature-target-gate: cargo metadata yielded no dependency edges; cannot derive dependents' >&2
        exit 3
    fi
    printf '%s\n' "$edges"
}

changed_files() {
    git diff --name-only "$base"...HEAD
    git diff --name-only HEAD
    git ls-files --others --exclude-standard
}

# an unresolvable base makes the committed diff fail, and a failed diff reads as "nothing changed"
if [ "$all" = 0 ] && ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
    printf 'feature-target-gate: base %s does not resolve; cannot tell what changed (pass a base or --all)\n' "$base" >&2
    exit 3
fi

directly_changed_crates() {
    local dir
    changed_files \
        | cut -d/ -f1 \
        | sort -u \
        | while IFS= read -r dir; do package_name "$dir"; done \
        | grep -Fx -f <(gated_crates) || true
}

# "<crate>\t<reason>" per selected crate: `changed`, or `depends on X` for the first selected X
# it depends on; repeats until no gated crate is newly pulled in (transitive closure)
select_crates() {
    local selected="" crate dep reason grew=1 edges gated
    if [ "$all" = 1 ]; then
        gated_crates | while IFS= read -r crate; do printf '%s\tall\n' "$crate"; done
        return
    fi
    edges="$(dependency_edges)" || return 3
    gated="$(gated_crates)"
    for crate in $(directly_changed_crates); do
        selected="$selected $crate"
        printf '%s\tchanged\n' "$crate"
    done
    while [ "$grew" = 1 ]; do
        grew=0
        for crate in $gated; do
            case " $selected " in *" $crate "*) continue ;; esac
            dep="$(printf '%s\n' "$edges" | awk -v crate="$crate" -v sel="$selected" '
                BEGIN { n = split(sel, chosen, " "); for (i = 1; i <= n; i++) want[chosen[i]] = 1 }
                $1 == crate && ($2 in want) { print $2; exit }')"
            if [ -n "$dep" ]; then
                selected="$selected $crate"
                grew=1
                printf '%s\tdepends on %s\n' "$crate" "$dep"
            fi
        done
    done
}

# a gate measures the working tree, not the commit: say which one, because a
# dirty tree's failures are the caller's uncommitted work and nothing else
dirty="clean"
if [ -n "$(git status --porcelain)" ]; then
    dirty="dirty (uncommitted changes are part of what this measures)"
fi
printf 'feature-target-gate: %s %s\n' "$(git rev-parse --short HEAD)" "$dirty"

# the script is not a crate, so its edit selects nothing by directory; it changes every row, so select every crate
if [ "$all" = 0 ] && changed_files | grep -Fx 'scripts/feature-target-gate.sh' >/dev/null; then
    echo 'feature-target-gate: gate script changed; selecting every crate'
    all=1
fi

selection="$(select_crates)" || exit 3
crates="$(printf '%s\n' "$selection" | cut -f1 | tr '\n' ' ')"
if [ -z "${crates// /}" ]; then
    echo "feature-target-gate: no test-bearing crate changed against $base"
    exit 0
fi
printf '%s\n' "$selection" | while IFS=$'\t' read -r crate reason; do
    printf 'feature-target-gate: selected %s (%s)\n' "$crate" "$reason"
done

if [ "$list" = 1 ]; then
    rows=0
    for row in "${matrix[@]}"; do
        case " $crates " in
            *" ${row%%:*} "*) printf 'feature-target-gate: row %s\n' "$row"; rows=$((rows + 1)) ;;
        esac
    done
    printf 'feature-target-gate: --list %s crates, %s rows, nothing run\n' "$(printf '%s\n' "$crates" | wc -w | tr -d ' ')" "$rows"
    exit 0
fi

failed=0
ran=0
while IFS=: read -r crate features tier_features; do
    case " $crates " in
        *" $crate "*) ;;
        *) continue ;;
    esac
    verb=(check)
    format_args=(--message-format short)
    if [ "$features" = "@nostd" ]; then
        if [ -n "$tier_features" ]; then
            label="$crate --lib --no-default-features --features $tier_features"
            args=(-p "$crate" --lib --no-default-features --features "$tier_features")
        else
            label="$crate --lib --no-default-features"
            args=(-p "$crate" --lib --no-default-features)
        fi
    elif [ "$features" = "@libtest" ]; then
        label="$crate lib test (default features)"
        verb=(test --no-run)
        args=(-p "$crate" --lib)
    elif [ "$features" = "@libtestrun" ]; then
        label="$crate lib test --features $tier_features"
        verb=(val)
        format_args=()
        args=(-p "$crate" --lib --features "$tier_features")
    elif [ "$features" = "@nodefault" ]; then
        label="$crate --all-targets --no-default-features --features $tier_features"
        args=(-p "$crate" --all-targets --no-default-features --features "$tier_features")
    elif [ -n "$features" ]; then
        label="$crate --features $features"
        args=(-p "$crate" --all-targets --features "$features")
    else
        label="$crate (default features)"
        args=(-p "$crate" --all-targets)
    fi
    printf '== %s ... ' "$label"
    if output="$("${cargo_cmd[@]}" "${verb[@]}" "${args[@]}" ${format_args[@]+"${format_args[@]}"} 2>&1)"; then
        errors="$(printf '%s\n' "$output" | grep -cE '^error' || true)"
        if [ "$errors" != 0 ]; then
            printf 'FAIL (%s errors)\n' "$errors"
            printf '%s\n' "$output" | grep -E '^error|^  -->' | head -20
            failed=1
        elif [ "$features" = "@libtestrun" ]; then
            run_count="$(printf '%s\n' "$output" | grep -E '^SUMMARY ' | grep -oE 'passed=[0-9]+' | grep -oE '[0-9]+' | tail -1 || true)"
            if [ -z "$run_count" ] || [ "$run_count" = 0 ]; then
                printf 'FAIL (no SUMMARY line or 0 tests passed)\n'
                failed=1
            else
                printf 'ok (%s tests run)\n' "$run_count"
            fi
        else
            printf 'ok\n'
        fi
    else
        printf 'FAIL\n'
        printf '%s\n' "$output" | grep -E '^error|^ *FAIL' | head -20
        failed=1
    fi
    ran=$((ran + 1))
done < <(printf '%s\n' "${matrix[@]}")

# pinned-fixture tests compare emitted output against omega/tests/fixtures; they must run, not just compile, or a fixture can drift on main
# a row that runs zero tests fails: zero reads like a pass
# set format: features:test names (space separated)
omega_fixture_sets=(
    "metal:packed_row_codecs_msl_identity packed_row_blocked_s1_byte_identity q4k_rendered_matvec_parity q56k_rendered_matvec_parity q6k_multi_row_rendered_parity block_codec_render_parity level_codec_tiled_gemm_parity bf16_projection_ouro_q_proj"
    "metal,metal-grouped-gemm:expert_grouped_gemm_stacked_parity"
)
case " $crates " in
    *" proxima-omega "*)
        if [ "$(uname -s)" != Darwin ]; then
            echo '== proxima-omega fixture tests ... skipped (macos-only metal tests)'
        else
            for fixture_set in "${omega_fixture_sets[@]}"; do
                fixture_features="${fixture_set%%:*}"
                fixture_tests="${fixture_set#*:}"
                test_args=()
                for fixture_test in $fixture_tests; do
                    test_args+=(--test "$fixture_test")
                done
                printf '== proxima-omega fixture tests --features %s ... ' "$fixture_features"
                if output="$("${cargo_cmd[@]}" nextest run -p proxima-omega --features "$fixture_features" --no-fail-fast "${test_args[@]}" 2>&1)"; then
                    run_count="$(printf '%s\n' "$output" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1 || true)"
                    if [ -z "$run_count" ] || [ "$run_count" = 0 ]; then
                        printf 'FAIL (0 tests run)\n'
                        failed=1
                    else
                        printf 'ok (%s tests run)\n' "$run_count"
                    fi
                else
                    printf 'FAIL\n'
                    printf '%s\n' "$output" | grep -E '^ *(FAIL|SIGABRT|SIGSEGV)|^error' | head -20
                    failed=1
                fi
                ran=$((ran + 1))
            done
        fi
        ;;
esac

if [ "$ran" = 0 ]; then
    echo "feature-target-gate: SELECTED-NOTHING — the crate list and the matrix disagree" >&2
    exit 2
fi

if [ "$failed" != 0 ]; then
    echo "feature-target-gate: FAIL ($ran checks run)" >&2
    exit 1
fi
echo "feature-target-gate: ok ($ran checks run)"
