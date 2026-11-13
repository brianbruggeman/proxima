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
#        CARGO_TARGET_DIR is honoured; CI=true skips --all-targets colouring
set -euo pipefail

# the caller's tree, not the script's: this repo is worked in worktrees, and a
# gate that always measured the primary checkout would grade the wrong commit
if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$repo_root"

all=0
base="origin/main"
case "${1:-}" in
    --all) all=1 ;;
    "") ;;
    *) base="$1" ;;
esac

# crate:feature-set pairs. An empty feature set means "default features".
# crate:@nostd:feature-set is a tier row: `cargo check --lib --no-default-features`
# plus those features (empty = bare). --lib because tests and examples need std;
# the tier claim is that the library itself builds without it, and a default-
# features --all-targets build can never notice that break.
matrix=(
    "proxima-tensor:std"
    "proxima-model-interop:std"
    "proxima-model-interop:std,metal,instrument"
    "proxima-model-interop:std,cuda"
    "proxima-omega:"
    "proxima-numeric:std"
    "proxima-numeric:config"
    "proxima-gguf:"
    "proxima-tensor:@nostd:alloc"
    "proxima-numeric:@nostd:alloc"
    "proxima-numeric:@nostd:"
    "proxima-gguf:@nostd:alloc"
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

changed_crates() {
    if [ "$all" = 1 ]; then
        printf '%s\n' proxima-tensor proxima-model-interop proxima-omega proxima-numeric proxima-gguf
        return
    fi
    local dir
    git diff --name-only "$base"...HEAD 2>/dev/null \
        | cut -d/ -f1 \
        | sort -u \
        | while IFS= read -r dir; do package_name "$dir"; done \
        | grep -Ex 'proxima-tensor|proxima-model-interop|proxima-omega|proxima-numeric|proxima-gguf' || true
}

# a gate measures the working tree, not the commit: say which one, because a
# dirty tree's failures are the caller's uncommitted work and nothing else
dirty="clean"
if [ -n "$(git status --porcelain)" ]; then
    dirty="dirty (uncommitted changes are part of what this measures)"
fi
printf 'feature-target-gate: %s %s\n' "$(git rev-parse --short HEAD)" "$dirty"

crates="$(changed_crates | tr '\n' ' ')"
if [ -z "$crates" ]; then
    echo "feature-target-gate: no test-bearing crate changed against $base"
    exit 0
fi

failed=0
ran=0
while IFS=: read -r crate features tier_features; do
    case " $crates " in
        *" $crate "*) ;;
        *) continue ;;
    esac
    if [ "$features" = "@nostd" ]; then
        if [ -n "$tier_features" ]; then
            label="$crate --lib --no-default-features --features $tier_features"
            args=(-p "$crate" --lib --no-default-features --features "$tier_features")
        else
            label="$crate --lib --no-default-features"
            args=(-p "$crate" --lib --no-default-features)
        fi
    elif [ -n "$features" ]; then
        label="$crate --features $features"
        args=(-p "$crate" --all-targets --features "$features")
    else
        label="$crate (default features)"
        args=(-p "$crate" --all-targets)
    fi
    printf '== %s ... ' "$label"
    if output="$(cargo check "${args[@]}" --message-format short 2>&1)"; then
        errors="$(printf '%s\n' "$output" | grep -cE '^error' || true)"
        if [ "$errors" != 0 ]; then
            printf 'FAIL (%s errors)\n' "$errors"
            printf '%s\n' "$output" | grep -E '^error|^  -->' | head -20
            failed=1
        else
            printf 'ok\n'
        fi
    else
        printf 'FAIL\n'
        printf '%s\n' "$output" | grep -E '^error' | head -20
        failed=1
    fi
    ran=$((ran + 1))
done < <(printf '%s\n' "${matrix[@]}")

if [ "$ran" = 0 ]; then
    echo "feature-target-gate: SELECTED-NOTHING — the crate list and the matrix disagree" >&2
    exit 2
fi

if [ "$failed" != 0 ]; then
    echo "feature-target-gate: FAIL ($ran checks run)" >&2
    exit 1
fi
echo "feature-target-gate: ok ($ran checks run)"
